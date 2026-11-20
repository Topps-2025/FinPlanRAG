"""Tier-1 external validity: DocFinQA long-document financial QA.

DocFinQA (ACL 2024) augments 7,437 FinQA questions with the full 10-K context.
This runner compares fixed BM25 evidence budgets with a state-conditioned
acquisition policy (the FinPlan decision interface) using the same reader and
provider.  Gold answers come from the DocFinQA release; FinQA's released
program constants are used only as a silver evidence proxy, never as an input
to the reader.

Usage:
    PYTHONPATH="src:scripts" python scripts/run_docfinqa_external.py \
      --data /root/autodl-tmp/datasets/docfinqa/test.json \
      --finqa /root/autodl-tmp/datasets/finqa/test.json \
      --provider deepseek --model 'deepseek-flash[1m]' \
      --base-url https://api.deepseek.com/anthropic \
      --api-key-env ANTHROPIC_AUTH_TOKEN \
      --out paper/results/external_tier1/docfinqa/test300/deepseek \
      --limit 300 --sample-seed 20261109
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from external_common import (  # noqa: E402
    call_json,
    format_evidence,
    make_client,
    obligation_terms,
    run_parallel,
    uncovered_terms,
    word_chunks,
    write_json,
    write_jsonl,
)
from finplan_rag.models import Chunk
from finplan_rag.retrieval import BM25Index, tokens

READER_SYSTEM = """You are a financial analyst answering a question about a long \
SEC filing. Use only the excerpts below. The question may require arithmetic; \
do the arithmetic carefully. If the excerpts do not contain enough information \
to answer, return "insufficient" and set sufficient to false.

Return only a JSON object with these keys:
- answer: the final answer as a short string (number or phrase)
- confidence: a number between 0 and 1
- sufficient: true if the excerpts are enough to answer, otherwise false
- missing: a short description of the missing information (empty if sufficient)
- evidence_ids: the paragraph ids you used
- rationale: at most 40 words, no markdown"""

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def stable_id(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def normalize_question(text: str) -> str:
    text = str(text or "").lower().replace(" ", " ")
    text = re.sub(r"[^a-z0-9$.%]+", " ", text)
    return " ".join(text.split())


def load_finqa_map(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, Any]] = {}
    for record in data:
        qa = record.get("qa") or {}
        question = normalize_question(qa.get("question", ""))
        if question:
            mapping[question] = record
    return mapping


def prepare_context(text: str) -> str:
    # DocFinQA contains both real tabs and escaped ``\\t`` sequences.
    text = str(text or "").replace("\\t", " ").replace("\t", " ").replace("#", "\n")
    text = re.sub(r"[ ]{2,}", " ", text)
    return text


def _is_table_line(line: str) -> bool:
    if line.count("|") >= 2:
        return True
    numbers = re.findall(r"\$?\(?-?\d[\d,]*(?:\.\d+)?\)?%?", line)
    return len(numbers) >= 2 and len(line.split()) <= 22


def make_doc_chunks(context: str, document_id: str) -> list[Chunk]:
    from datetime import datetime, timezone

    stamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    lines = [line.strip() for line in prepare_context(context).splitlines() if line.strip()]
    chunks: list[Chunk] = []
    index = 0
    cursor = 0
    while cursor < len(lines):
        if _is_table_line(lines[cursor]):
            block: list[str] = []
            while (
                cursor < len(lines)
                and _is_table_line(lines[cursor])
                and sum(len(row) for row in block) < 4000
            ):
                block.append(lines[cursor])
                cursor += 1
            text = "\n".join(block)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}::c{index}",
                    document_id=document_id,
                    entity="",
                    available_at=stamp,
                    text=text,
                    terms=tokens(text),
                    metadata={"paragraph": index, "kind": "table"},
                )
            )
            index += 1
        else:
            block = []
            while (
                cursor < len(lines)
                and not _is_table_line(lines[cursor])
                and sum(len(part) for part in block) < 1200
            ):
                block.append(lines[cursor])
                cursor += 1
            for piece in word_chunks(" ".join(block), window=160, stride=120):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}::c{index}",
                        document_id=document_id,
                        entity="",
                        available_at=stamp,
                        text=piece,
                        terms=tokens(piece),
                        metadata={"paragraph": index, "kind": "text"},
                    )
                )
                index += 1
    return chunks


def gold_numbers(record: dict[str, Any] | None) -> list[str]:
    if not record:
        return []
    qa = record.get("qa") or {}
    values: set[str] = set()
    for step in qa.get("steps", []) or []:
        for key in ("arg1", "arg2", "res"):
            text = str(step.get(key, "")).strip().replace(",", "")
            if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
                continue
            try:
                value = float(text)
            except ValueError:
                continue
            if abs(value) < 2 or 1900 <= abs(value) <= 2100:
                continue
            values.add(text)
    answer = str(qa.get("answer", "")).strip().replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", answer):
        try:
            value = float(answer)
            if abs(value) >= 2 and not (1900 <= abs(value) <= 2100):
                values.add(answer)
        except ValueError:
            pass
    return sorted(values)


def number_in_text(number: str, text: str) -> bool:
    text = text.replace(",", "")
    pattern = r"(?<![0-9.])" + re.escape(number) + r"(?![0-9])"
    if re.search(pattern, text):
        return True
    # OCR sometimes splits thousands ("5 829"); accept the digits with spaces.
    spaced = r"\s*".join(re.escape(ch) for ch in number)
    return bool(re.search(r"(?<![0-9.])" + spaced + r"(?![0-9])", text))


def normalize_answer(text: str) -> str:
    text = str(text or "").lower().replace(" ", " ")
    text = text.replace(",", "").replace("$", " ").replace("%", " ")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9.\- ]+", " ", text)
    return " ".join(text.split())


def numeric_value(text: str) -> float | None:
    cleaned = str(text or "").replace(",", "").replace("$", "").replace("%", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def exact_match(prediction: str, gold: str) -> bool:
    pred_norm, gold_norm = normalize_answer(prediction), normalize_answer(gold)
    if pred_norm == gold_norm:
        return pred_norm != ""
    pred_num, gold_num = numeric_value(prediction), numeric_value(gold)
    if pred_num is not None and gold_num is not None:
        return math.isclose(pred_num, gold_num, rel_tol=0.01, abs_tol=0.01)
    return False


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = 0
    remaining = list(gold_tokens)
    for token in pred_tokens:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if not common:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_reader(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    if not parsed:
        return None
    answer = str(parsed.get("answer", parsed.get("label", ""))).strip()
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(1.0, max(0.0, confidence))
    sufficient = parsed.get("sufficient")
    if isinstance(sufficient, str):
        sufficient = sufficient.strip().lower() in {"true", "yes", "1"}
    if sufficient is None:
        sufficient = answer.lower() not in {"", "insufficient", "unknown", "n/a"}
    evidence_ids = parsed.get("evidence_ids") or []
    if not isinstance(evidence_ids, list):
        evidence_ids = []
    return {
        "answer": answer,
        "confidence": confidence,
        "sufficient": bool(sufficient),
        "missing": str(parsed.get("missing", ""))[:400],
        "evidence_ids": [str(item) for item in evidence_ids][:20],
        "rationale": str(parsed.get("rationale", ""))[:400],
    }


def reader_prompt(question: str, chunks: Sequence[Chunk], round_index: int = 0) -> str:
    intro = "Use only the excerpts below. Do not use outside knowledge.\n"
    if round_index:
        intro += (
            f"This is acquisition round {round_index + 1}; the excerpts include "
            "additional paragraphs retrieved after you reported missing information.\n"
        )
    return (
        intro
        + "Question: "
        + question.strip()
        + "\n\nDocument excerpts:\n"
        + format_evidence(chunks)
        + "\n\nReturn the JSON object now."
    )


def evaluate_one(
    example: dict[str, Any],
    *,
    finqa_map: dict[str, dict[str, Any]],
    client,
    conditions: Sequence[str],
    ks: Sequence[int],
    planner_budget: int,
    full_context_chars: int,
) -> list[dict[str, Any]]:
    question = str(example.get("Question", ""))
    gold = str(example.get("Answer", ""))
    finqa = finqa_map.get(normalize_question(question))
    nums = gold_numbers(finqa)
    chunks = make_doc_chunks(str(example.get("Context", "")), document_id=f"doc{stable_id(question)}")
    index = BM25Index(chunks)
    rows: list[dict[str, Any]] = []

    def add_row(condition: str, result: dict[str, Any], acquired: Sequence[Chunk], extra=None):
        parsed = result.get("parsed")
        answer = parsed["answer"] if parsed else ""
        correct = bool(parsed) and exact_match(answer, gold)
        f1 = token_f1(answer, gold) if parsed else 0.0
        text = " ".join(chunk.text for chunk in acquired)
        found = [number for number in nums if number_in_text(number, text)]
        recall = len(found) / len(nums) if nums else None
        row = {
            "example_id": f"docfinqa-{stable_id(question)}",
            "question": question,
            "gold_answer": gold,
            "finqa_joined": finqa is not None,
            "condition": condition,
            "parsed": parsed,
            "correct": correct,
            "f1": f1,
            "error": result.get("error"),
            "usage": result.get("usage", {}),
            "latency_seconds": result.get("latency_seconds", 0.0),
            "raw_text": result.get("raw_text", "")[:1200],
            "acquired_paragraphs": [int(chunk.metadata["paragraph"]) for chunk in acquired],
            "num_paragraphs": len(acquired),
            "silver_gold_numbers": nums,
            "silver_recall": recall,
        }
        if extra:
            row.update(extra)
        rows.append(row)

    for k in ks:
        condition = f"fixed_k_{k}"
        if condition not in conditions:
            continue
        acquired = [hit.chunk for hit in index.search(question, limit=k)]
        result = call_json(client, reader_prompt(question, acquired), system=READER_SYSTEM)
        add_row(condition, {**result, "parsed": parse_reader(result.get("parsed"))}, acquired)

    if "full_context" in conditions:
        acquired: list[Chunk] = []
        used = 0
        for chunk in chunks:
            if used + len(chunk.text) > full_context_chars and acquired:
                break
            acquired.append(chunk)
            used += len(chunk.text)
        result = call_json(client, reader_prompt(question, acquired), system=READER_SYSTEM)
        add_row("full_context", {**result, "parsed": parse_reader(result.get("parsed"))}, acquired)

    if "planner" in conditions:
        acquired: list[Chunk] = []
        acquired_ids: set[str] = set()
        last_result: dict[str, Any] | None = None
        terms = obligation_terms(question)
        rounds = max(1, planner_budget // 2)
        for round_index in range(rounds):
            target = min(planner_budget, 2 * (round_index + 1))
            needed = target - len(acquired)
            if needed <= 0:
                break
            missing_terms = uncovered_terms(terms, acquired)
            query = question
            previous = (last_result or {}).get("parsed") or {}
            if previous.get("missing"):
                query += " " + str(previous["missing"])
            if missing_terms:
                query += " " + " ".join(missing_terms)
            candidates = [
                hit
                for hit in index.search(query, limit=target + 32, exclude=acquired_ids)
                if hit.chunk.chunk_id not in acquired_ids
            ]
            if missing_terms:
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
                reader_prompt(question, acquired, round_index=round_index),
                system=READER_SYSTEM,
            )
            last_result["parsed"] = parse_reader(last_result.get("parsed"))
            still_missing = uncovered_terms(terms, acquired)
            if last_result["parsed"] and last_result["parsed"]["sufficient"] and not still_missing:
                break
            if (
                last_result["parsed"]
                and last_result["parsed"]["answer"].lower() not in {"", "insufficient", "unknown"}
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


def sample_examples(examples: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(examples):
        return examples
    rng = random.Random(seed)
    return sorted(rng.sample(examples, limit), key=lambda item: str(item.get("Question", "")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--finqa", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--provider", required=True, choices=["deepseek", "glm", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--extra-body", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--sample-seed", type=int, default=20261109)
    parser.add_argument(
        "--conditions",
        default="fixed_k,planner",
    )
    parser.add_argument("--ks", default="1,2,4,8,16")
    parser.add_argument("--planner-budget", type=int, default=8)
    parser.add_argument("--full-context-chars", type=int, default=30000)
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
    examples = sample_examples(
        json.loads(Path(args.data).read_text(encoding="utf-8")),
        args.limit,
        args.sample_seed,
    )
    finqa_map = load_finqa_map(Path(args.finqa))

    def worker(example):
        return evaluate_one(
            example,
            finqa_map=finqa_map,
            client=client,
            conditions=conditions,
            ks=ks,
            planner_budget=args.planner_budget,
            full_context_chars=args.full_context_chars,
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
            "finqa": str(args.finqa),
            "examples": len(examples),
            "sample_seed": args.sample_seed,
            "conditions": sorted(conditions),
            "ks": ks,
            "planner_budget": args.planner_budget,
            "full_context_chars": args.full_context_chars,
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
