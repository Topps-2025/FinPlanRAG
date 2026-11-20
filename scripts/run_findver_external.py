"""Tier-1 external validity: FinDVer claim verification.

FinDVer (EMNLP 2024) provides claims over long SEC 10-K/10-Q filings with
expert entailment labels and gold evidence paragraph indices.  This runner
compares fixed evidence budgets with a state-conditioned acquisition policy
(the FinPlan decision interface) using the same reader and provider.

Usage:
    PYTHONPATH="src:scripts" python scripts/run_findver_external.py \
      --data vendor/official_baselines/findver/data/testmini.json \
      --reports-dir vendor/official_baselines/findver/financial_reports \
      --provider deepseek --model 'deepseek-flash[1m]' \
      --base-url https://api.deepseek.com/anthropic \
      --api-key-env ANTHROPIC_AUTH_TOKEN \
      --out paper/results/external_tier1/findver/testmini/deepseek
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from external_common import (  # noqa: E402
    call_json,
    format_evidence,
    make_client,
    obligation_terms,
    parse_date_from_report_name,
    parse_json_object,
    run_parallel,
    uncovered_terms,
    write_json,
    write_jsonl,
)
from finplan_rag.models import Chunk
from finplan_rag.retrieval import BM25Index, tokens
from official_prompts import findver_system_prompt  # noqa: E402

VERDICTS = ("entailed", "refuted", "insufficient")
READER_SYSTEM = """You are a financial claim verifier. You receive a statement and \
excerpts from a financial document. Decide whether the statement is entailed \
(supported by the excerpts) or refuted (contradicted by the excerpts). If the \
excerpts do not contain enough evidence to decide, return "insufficient" and \
set sufficient to false.

Return only a JSON object with these keys:
- verdict: "entailed", "refuted", or "insufficient"
- confidence: a number between 0 and 1
- sufficient: true if the excerpts are enough to decide, otherwise false
- missing: a short description of the missing evidence (empty if sufficient)
- evidence_ids: the paragraph ids you used
- rationale: at most 40 words, no markdown"""


def load_examples(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list in {path}")
    return data


@functools.lru_cache(maxsize=512)
def load_report(path_str: str) -> dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def paragraph_chunks(report_path: Path, report_name: str) -> tuple[dict[str, Any], list[Chunk]]:
    report = load_report(str(report_path))
    stamp = parse_date_from_report_name(report_name)
    chunks: list[Chunk] = []
    for index, paragraph in enumerate(report.get("context", [])):
        text = str(paragraph.get("context", ""))[:6000]
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{report_name}::p{index}",
                document_id=report_name,
                entity=report_name.split("-", 1)[0],
                available_at=stamp,
                text=text,
                terms=tokens(text),
                metadata={"paragraph": index},
            )
        )
    return report, chunks


def truth_of(example: dict[str, Any]) -> str:
    value = example.get("entailment_label")
    if isinstance(value, bool):
        return "entailed" if value else "refuted"
    text = str(value).strip().lower()
    return "entailed" if text in {"true", "entailed", "yes", "1"} else "refuted"


def normalize_verdict(value: Any) -> str | None:
    if isinstance(value, bool):
        return "entailed" if value else "refuted"
    text = str(value or "").strip().lower()
    if text in {"entailed", "entail", "true", "supported", "yes"}:
        return "entailed"
    if text in {"refuted", "refute", "false", "contradicted", "no"}:
        return "refuted"
    if text in {"insufficient", "unknown", "abstain", "nei", "not enough evidence"}:
        return "insufficient"
    return None


def parse_reader(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    if not parsed:
        return None
    verdict = normalize_verdict(
        parsed.get("verdict", parsed.get("label", parsed.get("answer")))
    )
    if verdict is None:
        return None
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(1.0, max(0.0, confidence))
    sufficient = parsed.get("sufficient")
    if isinstance(sufficient, str):
        sufficient = sufficient.strip().lower() in {"true", "yes", "1"}
    if sufficient is None:
        sufficient = verdict != "insufficient"
    evidence_ids = parsed.get("evidence_ids") or []
    if not isinstance(evidence_ids, list):
        evidence_ids = []
    return {
        "verdict": verdict,
        "confidence": confidence,
        "sufficient": bool(sufficient),
        "missing": str(parsed.get("missing", ""))[:400],
        "evidence_ids": [str(item) for item in evidence_ids][:20],
        "rationale": str(parsed.get("rationale", ""))[:400],
    }


def parse_official(text: str) -> dict[str, Any] | None:
    parsed = parse_json_object(text)
    if parsed:
        for key in ("conclusion", "label", "verdict", "answer", "entailment_label"):
            verdict = normalize_verdict(parsed.get(key))
            if verdict:
                return {
                    "verdict": verdict,
                    "confidence": 0.5,
                    "sufficient": True,
                    "missing": "",
                    "evidence_ids": [],
                    "rationale": str(parsed.get("reasoning", ""))[:400],
                }
    matches = re.findall(r"\b(entailed|refuted)\b", text or "", flags=re.I)
    if not matches:
        return None
    verdict = matches[-1].lower()
    return {
        "verdict": verdict,
        "confidence": 0.5,
        "sufficient": True,
        "missing": "",
        "evidence_ids": [],
        "rationale": "official FinDVer CoT output",
    }


def reader_prompt(statement: str, chunks: Sequence[Chunk], round_index: int = 0) -> str:
    intro = (
        "Use only the excerpts below. Do not use outside knowledge or facts "
        "that are not in the excerpts.\n"
    )
    if round_index:
        intro += (
            f"This is acquisition round {round_index + 1}; the excerpts include "
            "additional paragraphs retrieved after you reported missing evidence.\n"
        )
    return (
        intro
        + "Statement to verify:\n"
        + statement.strip()
        + "\n\nFinancial document excerpts:\n"
        + format_evidence(chunks)
        + "\n\nReturn the JSON object now."
    )


def official_prompt(statement: str, chunks: Sequence[Chunk]) -> tuple[str, str]:
    system = findver_system_prompt()
    user = (
        "Financial Document:\n"
        + format_evidence(chunks)
        + "\n\nStatement to verify: "
        + statement.strip()
        + "\n\nLet's think step by step to verify the given statement."
    )
    return system, user


def evaluate_one(
    example: dict[str, Any],
    *,
    reports_dir: Path,
    client,
    conditions: Sequence[str],
    ks: Sequence[int],
    planner_budget: int,
    full_context_chars: int,
    final_reader: str = "generic",
) -> list[dict[str, Any]]:
    report_name = str(example["report"])
    report_path = reports_dir / report_name
    _, chunks = paragraph_chunks(report_path, report_name)
    index = BM25Index(chunks)
    by_paragraph = {int(chunk.metadata["paragraph"]): chunk for chunk in chunks}
    gold_ids = [int(item) for item in example.get("relevant_context", [])]
    gold_chunks = [by_paragraph[i] for i in gold_ids if i in by_paragraph]
    statement = str(example["statement"])
    truth = truth_of(example)
    rows: list[dict[str, Any]] = []

    def add_row(condition: str, result: dict[str, Any], acquired: Sequence[Chunk], extra=None):
        acquisition_usage = result.get("usage", {})
        if final_reader == "official" and condition not in {"official_cot_gold", "official_cot_k10"}:
            system, user = official_prompt(statement, acquired)
            final = call_json(client, user, system=system, max_tokens=900)
            final_usage = final.get("usage", {})
            merged_usage = {
                key: int(acquisition_usage.get(key, 0) or 0) + int(final_usage.get(key, 0) or 0)
                for key in ("input_tokens", "output_tokens")
            }
            result = {
                **final,
                "parsed": parse_official(final.get("full_text", "")),
                "usage": merged_usage,
                "latency_seconds": float(result.get("latency_seconds", 0.0) or 0.0)
                + float(final.get("latency_seconds", 0.0) or 0.0),
                "acquisition_usage": acquisition_usage,
                "final_usage": final_usage,
            }
        parsed = result.get("parsed")
        evidence_ids = [int(chunk.metadata["paragraph"]) for chunk in acquired]
        gold_set = set(gold_ids)
        recall = len(gold_set.intersection(evidence_ids)) / max(1, len(gold_set))
        row = {
            "example_id": example.get("example_id"),
            "subset": example.get("subset"),
            "report": report_name,
            "truth": truth,
            "condition": condition,
            "parsed": parsed,
            "error": result.get("error"),
            "usage": result.get("usage", {}),
            "latency_seconds": result.get("latency_seconds", 0.0),
            "raw_text": result.get("raw_text", "")[:1200],
            "acquired_paragraphs": evidence_ids,
            "gold_paragraphs": gold_ids,
            "evidence_recall": recall,
            "num_paragraphs": len(acquired),
        }
        if extra:
            row.update(extra)
        rows.append(row)

    def empty_result() -> dict[str, Any]:
        return {
            "parsed": None,
            "raw_text": "",
            "full_text": "",
            "usage": {},
            "latency_seconds": 0.0,
            "error": None,
        }

    # Oracle gold evidence (upper bound, not a method).
    if "oracle_gold" in conditions and gold_chunks:
        if final_reader == "official":
            result = empty_result()
        else:
            result = call_json(client, reader_prompt(statement, gold_chunks), system=READER_SYSTEM)
        add_row("oracle_gold", {**result, "parsed": parse_reader(result.get("parsed"))}, gold_chunks)

    # Fixed BM25 budgets.
    for k in ks:
        condition = f"fixed_k_{k}"
        if condition not in conditions:
            continue
        acquired = [hit.chunk for hit in index.search(statement, limit=k)]
        if final_reader == "official":
            result = empty_result()
        else:
            result = call_json(client, reader_prompt(statement, acquired), system=READER_SYSTEM)
        add_row(condition, {**result, "parsed": parse_reader(result.get("parsed"))}, acquired)

    # Truncated long-context baseline.
    if "full_context" in conditions:
        acquired: list[Chunk] = []
        used = 0
        for chunk in chunks:
            if used + len(chunk.text) > full_context_chars and acquired:
                break
            acquired.append(chunk)
            used += len(chunk.text)
        result = call_json(client, reader_prompt(statement, acquired), system=READER_SYSTEM, max_tokens=700)
        add_row("full_context", {**result, "parsed": parse_reader(result.get("parsed"))}, acquired)

    # Official FinDVer CoT prompt with gold and fixed-k evidence.
    if "official_cot_gold" in conditions and gold_chunks:
        system, user = official_prompt(statement, gold_chunks)
        result = call_json(client, user, system=system, max_tokens=900)
        add_row(
            "official_cot_gold",
            {**result, "parsed": parse_official(result.get("full_text", ""))},
            gold_chunks,
        )
    if "official_cot_k10" in conditions:
        acquired = [hit.chunk for hit in index.search(statement, limit=10)]
        system, user = official_prompt(statement, acquired)
        result = call_json(client, user, system=system, max_tokens=900)
        add_row(
            "official_cot_k10",
            {**result, "parsed": parse_official(result.get("full_text", ""))},
            acquired,
        )

    # State-conditioned acquisition: start small, expand only when the reader
    # reports that the evidence is insufficient.
    if "planner" in conditions:
        acquired: list[Chunk] = []
        acquired_ids: set[str] = set()
        last_result: dict[str, Any] | None = None
        terms = obligation_terms(statement)
        rounds = max(1, planner_budget // 2)
        for round_index in range(rounds):
            target = min(planner_budget, 2 * (round_index + 1))
            needed = target - len(acquired)
            if needed <= 0:
                break
            missing_terms = uncovered_terms(terms, acquired)
            query = statement
            previous = (last_result or {}).get("parsed") or {}
            if previous.get("missing"):
                query += " " + str(previous["missing"])
            if missing_terms:
                query += " " + " ".join(missing_terms)
            candidates = [
                hit
                for hit in index.search(query, limit=target + 24, exclude=acquired_ids)
                if hit.chunk.chunk_id not in acquired_ids
            ]
            if missing_terms:
                # Evidence obligations: prefer paragraphs that actually carry
                # the unclosed claim components, then fall back to BM25 order.
                candidates.sort(
                    key=lambda hit: (
                        -sum(1 for term in missing_terms if term in hit.chunk.text.lower().replace(",", "")),
                        -hit.score,
                    )
                )
            new_chunks = [hit.chunk for hit in candidates[:needed]]
            if not new_chunks:
                break
            acquired.extend(new_chunks)
            acquired_ids.update(chunk.chunk_id for chunk in new_chunks)
            last_result = call_json(
                client,
                reader_prompt(statement, acquired, round_index=round_index),
                system=READER_SYSTEM,
            )
            last_result["parsed"] = parse_reader(last_result.get("parsed"))
            still_missing = uncovered_terms(terms, acquired)
            if last_result["parsed"] and last_result["parsed"]["sufficient"] and not still_missing:
                break
            if (
                last_result["parsed"]
                and last_result["parsed"]["verdict"] != "insufficient"
                and last_result["parsed"]["confidence"] >= 0.80
                and not still_missing
            ):
                break
        if last_result is None:
            last_result = {"parsed": None, "error": "no_evidence", "usage": {}, "latency_seconds": 0.0}
        add_row(
            "planner",
            last_result,
            acquired,
            extra={
                "planner_budget": planner_budget,
                "obligation_terms": terms,
                "uncovered_after": uncovered_terms(terms, acquired),
            },
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--provider", required=True, choices=["deepseek", "glm", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--extra-body", default=None, help="JSON object for OpenAI-compatible servers")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--conditions",
        default="oracle_gold,fixed_k,planner,official_cot_gold,official_cot_k10",
    )
    parser.add_argument("--ks", default="1,2,4,8,16")
    parser.add_argument("--planner-budget", type=int, default=8)
    parser.add_argument("--full-context-chars", type=int, default=30000)
    parser.add_argument("--final-reader", choices=["generic", "official"], default="generic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = {item.strip() for item in args.conditions.split(",") if item.strip()}
    ks = [int(item) for item in args.ks.split(",") if item.strip()]
    if "fixed_k" in conditions:
        conditions.remove("fixed_k")
        conditions.update(f"fixed_k_{k}" for k in ks)
    extra_body = json.loads(args.extra_body) if args.extra_body else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        extra_body=extra_body,
        cache_dir=out_dir / "llm_cache",
    )
    examples = load_examples(Path(args.data))
    if args.limit:
        examples = examples[: args.limit]
    reports_dir = Path(args.reports_dir)

    def worker(example):
        return evaluate_one(
            example,
            reports_dir=reports_dir,
            client=client,
            conditions=conditions,
            ks=ks,
            planner_budget=args.planner_budget,
            full_context_chars=args.full_context_chars,
            final_reader=args.final_reader,
        )

    started = time.time()
    all_rows: list[dict[str, Any]] = []
    for rows in run_parallel(examples, worker, workers=args.workers):
        all_rows.extend(rows)
    all_rows.sort(key=lambda row: (str(row["example_id"]), row["condition"]))
    write_jsonl(out_dir / "per_example.jsonl", all_rows)
    write_json(
        out_dir / "run_config.json",
        {
            "data": str(args.data),
            "reports_dir": str(args.reports_dir),
            "examples": len(examples),
            "conditions": sorted(conditions),
            "ks": ks,
            "planner_budget": args.planner_budget,
            "full_context_chars": args.full_context_chars,
            "final_reader": args.final_reader,
            "provider": args.provider,
            "model": client.model,
            "base_url": client.base_url,
            "workers": args.workers,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"wrote {len(all_rows)} rows to {out_dir / 'per_example.jsonl'}")


if __name__ == "__main__":
    main()
