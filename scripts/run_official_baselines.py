"""Run source-grounded official-code baselines on the claim-outcome benchmark.

Each method loads prompt templates and decision helpers from the vendored
official repository at a pinned commit (see ``official_prompts.PROVENANCE``),
then calls the same configured LLM backbone through the same point-in-time
evidence interface.  The shared task adapter maps the method's native output
space to below/within/above.

This is a source-grounded adaptation, not an unmodified run of the original
systems: the original datasets, retrievers, and some learned checkpoints are
not available or not compatible with this task.  The provenance artifact
records every reused and substituted component.

Usage:
    PYTHONPATH="src:scripts" python scripts/run_official_baselines.py \
      --data paper/data/dataset_v4.json --verified-only --workers 6 \
      --provider deepseek --model 'deepseek-flash[1m]' \
      --base-url https://api.deepseek.com/anthropic \
      --out paper/results/official/deepseek
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from deepseek_client import LLMClient, LLMError
from official_prompts import (
    PROVENANCE,
    adaptiverag_definitions,
    crag_format_prompt,
    finfact_prompts,
    findver_system_prompt,
    prompt_manifest,
    selfrag_prompts,
)
from run_company_disjoint_benchmark import FOLDS, TEST_YEARS, VERIFIED_LABELS, fold_of
from run_named_baselines import base_prompt, normalize, parse_json, retrieved_block
from run_rag_baseline import retrieve
from run_real_benchmark import LABELS, company_key, load_records, parse_date

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("findver", "finfact", "selfrag", "crag", "flare", "adaptive")

ADAPTER = """Benchmark adaptation: the realized outcome is one of below, within, or above the guided range. Use only evidence available at the cutoff. Retrieved earlier claims from other companies are calibration evidence, not direct evidence about this claim. If the method's native output is binary or open/closed, map its judgment to this three-way outcome. Return only JSON with keys label (below/within/above), confidence (0 to 1), credible_set (the smallest subset of below/within/above supported by the evidence), and rationale (at most 30 words)."""


def make_text_call(
    client: LLMClient, system: str | None, prompt: str, max_tokens: int = 200
) -> dict[str, Any]:
    """Call a method's auxiliary classifier/critic without task-JSON repair."""
    started = time.time()
    try:
        response = client.complete(
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=0.0,
            thinking=False,
            use_cache=True,
        )
        return {
            "text": client.text_of(response),
            "usage": client.usage_of(response),
            "latency_seconds": time.time() - started,
            "error": None,
        }
    except LLMError as error:
        return {
            "text": "",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "latency_seconds": time.time() - started,
            "error": f"api_error:{error}",
        }


def make_call(client: LLMClient, system: str | None, prompt: str, max_tokens: int = 900) -> dict[str, Any]:
    started = time.time()
    try:
        response = client.complete(
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=0.0,
            thinking=False,
            use_cache=True,
        )
        text = client.text_of(response)
        parsed = normalize(parse_json(text))
        usage = client.usage_of(response)
        if parsed is None:
            repair = client.complete(
                prompt
                + "\nYour previous response was not valid JSON. Return only the requested JSON object with label below/within/above, confidence, credible_set, and rationale.",
                system=system,
                max_tokens=max_tokens,
                temperature=0.0,
                thinking=False,
                use_cache=True,
            )
            text = client.text_of(repair)
            parsed = normalize(parse_json(text))
            repair_usage = client.usage_of(repair)
            usage = {
                "input_tokens": usage["input_tokens"] + repair_usage["input_tokens"],
                "output_tokens": usage["output_tokens"] + repair_usage["output_tokens"],
            }
        return {
            "parsed": parsed,
            "raw_text": text[:1600],
            "usage": usage,
            "latency_seconds": time.time() - started,
            "error": None if parsed else "parse_failure",
        }
    except LLMError as error:
        return {
            "parsed": None,
            "raw_text": "",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "latency_seconds": time.time() - started,
            "error": f"api_error:{error}",
        }


def structured_evidence(record, by_company) -> str:
    return base_prompt(record, by_company).strip()


def run_findver(client, record, by_company, retrieved, max_tokens):
    system = findver_system_prompt() + "\n\n" + ADAPTER
    prompt = (
        "Financial Document:\n"
        + structured_evidence(record, by_company)
        + "\n\n"
        + retrieved_block(retrieved, by_company)
        + "\n\nStatement to verify: The realized revenue for the target period will be below, within, or above the guided range.\n"
    )
    return make_call(client, system, prompt, max_tokens)


def run_finfact(client, record, by_company, retrieved, max_tokens):
    system = finfact_prompts()["open_book_system"] + "\n\n" + ADAPTER
    prompt = (
        "Claim: The realized revenue for the target period will be below, within, or above the guided range.\n"
        "Evidence:\n"
        + structured_evidence(record, by_company)
        + "\n\n"
        + retrieved_block(retrieved, by_company)
    )
    return make_call(client, system, prompt, max_tokens)


def run_selfrag(client, record, by_company, retrieved, max_tokens):
    prompts = selfrag_prompts()
    instruction = (
        ADAPTER
        + "\n\nGiven the point-in-time evidence, produce the best three-way outcome. "
        "First answer without retrieved claims; if external evidence is needed, retrieve and revise."
    )
    initial_prompt = prompts["prompt_dict"]["prompt_no_input"].format(
        instruction=instruction + "\n\n" + structured_evidence(record, by_company)
    )
    initial = make_call(client, None, initial_prompt, max_tokens)
    parsed = initial.get("parsed")
    if parsed is not None and float(parsed.get("confidence", 0.0)) >= 0.70:
        return initial
    critic_prompt = prompts["need_retrieval"]["context"].format(
        instruction=instruction + "\n\n" + structured_evidence(record, by_company)
    )
    critic = make_text_call(client, None, critic_prompt, 160)
    critic_text = (critic.get("text") or "").lower()
    need_retrieval = "[yes]" in critic_text or "need retrieval?: yes" in critic_text or parsed is None
    if not need_retrieval:
        return initial
    revised_prompt = prompts["prompt_dict"]["prompt_no_input_retrieval"].format(
        paragraph=retrieved_block(retrieved, by_company),
        instruction=instruction + "\n\n" + structured_evidence(record, by_company),
    )
    revised = make_call(client, None, revised_prompt, max_tokens)
    return revised if revised.get("parsed") is not None else initial


def run_crag(client, record, by_company, retrieved, max_tokens):
    prompt_builder = crag_format_prompt()
    question = (
        "Decide whether the realized revenue for the target period will be below, within, or above the guided range.\n"
        "Point-in-time evidence:\n"
        + structured_evidence(record, by_company)
    )
    paragraphs = retrieved_block(retrieved, by_company)
    eval_prompt = (
        "You are the retrieval evaluator in CRAG. Decide whether the retrieved documents are correct, "
        "incorrect, or ambiguous for the question. Correct means they contain sufficient relevant evidence; "
        "incorrect means they are irrelevant or misleading; ambiguous means they are partially useful. "
        'Return only JSON: {"quality":"correct|incorrect|ambiguous","rationale":"at most 20 words"}.\n\n'
        "Question:\n" + question + "\n\nRetrieved documents:\n" + paragraphs
    )
    evaluation = make_text_call(client, None, eval_prompt, 180)
    quality = "ambiguous"
    raw_eval = parse_json(evaluation.get("text") or "") or {}
    if isinstance(raw_eval, dict):
        quality = str(raw_eval.get("quality", "ambiguous")).lower()
    if quality not in ("correct", "incorrect", "ambiguous"):
        quality = "ambiguous"
    chosen = paragraphs if quality == "correct" else ("" if quality == "incorrect" else paragraphs)
    answer_prompt = prompt_builder(0, "custom", question, chosen, "llama")
    answer_prompt += "\n\n" + ADAPTER
    result = make_call(client, None, answer_prompt, max_tokens)
    result["route"] = quality
    return result


def run_flare(client, record, by_company, retrieved, max_tokens):
    # The official FLARE loop predicts a look-ahead sentence and triggers
    # retrieval on low-confidence tokens.  API backbones do not expose token
    # log-probabilities, so the trigger is the model's stated confidence.
    base_prompt_text = (
        "Answer from the point-in-time evidence. If the evidence is insufficient, state a calibrated confidence below 0.70.\n\n"
        + structured_evidence(record, by_company)
        + "\n\n"
        + ADAPTER
    )
    initial = make_call(client, None, base_prompt_text, max_tokens)
    parsed = initial.get("parsed")
    if parsed is not None and float(parsed.get("confidence", 0.0)) >= 0.70:
        initial["route"] = "no_retrieval"
        return initial
    revised_prompt = (
        "FLARE active retrieval has been triggered because the initial confidence was below 0.70. "
        "Use the retrieved earlier claims to revise or confirm the answer.\n\n"
        + structured_evidence(record, by_company)
        + "\n\n"
        + retrieved_block(retrieved, by_company)
        + "\n\n"
        + ADAPTER
    )
    revised = make_call(client, None, revised_prompt, max_tokens)
    revised["route"] = "retrieve"
    return revised if revised.get("parsed") is not None else initial


def run_adaptive(client, record, by_company, retrieved, max_tokens):
    definitions = adaptiverag_definitions()
    classify_prompt = (
        "You are Adaptive-RAG's query-complexity classifier. Classify the evidence need for this claim "
        "as A, B, or C.\n"
        f"A: {definitions['A']}\nB: {definitions['B']}\nC: {definitions['C']}\n"
        'Return only JSON: {"complexity":"A|B|C"}.\n\n'
        + structured_evidence(record, by_company)
    )
    classification = make_text_call(client, None, classify_prompt, 120)
    raw = parse_json(classification.get("text") or "") or {}
    route = str(raw.get("complexity", "C")).upper()
    if route not in ("A", "B", "C"):
        route = "C"
    if route == "A":
        prompt = structured_evidence(record, by_company) + "\n\n" + ADAPTER
    elif route == "B":
        prompt = (
            structured_evidence(record, by_company)
            + "\n\n"
            + retrieved_block(retrieved, by_company)
            + "\n\n"
            + ADAPTER
        )
    else:
        prompt = (
            structured_evidence(record, by_company)
            + "\n\n"
            + retrieved_block(retrieved, by_company)
            + "\n\nVerify each retrieved claim's relevance and reliability before answering.\n\n"
            + ADAPTER
        )
    result = make_call(client, None, prompt, max_tokens)
    result["route"] = route
    return result


METHODS = {
    "findver": run_findver,
    "finfact": run_finfact,
    "selfrag": run_selfrag,
    "crag": run_crag,
    "flare": run_flare,
    "adaptive": run_adaptive,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "paper/data/dataset_v4.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--extra-body", default=None)
    args = parser.parse_args()

    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")

    rows = load_records(args.data)
    by_company: dict[str, list] = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    selected = [
        record
        for _, _, record in rows
        if record.get("split") == "test"
        and int(record.get("cutoff_year") or 0) in TEST_YEARS
        and (not args.verified_only or record.get("verification") in VERIFIED_LABELS)
    ]
    selected.sort(key=lambda record: str(record["cutoff"]))
    if args.limit:
        selected = selected[: args.limit]
    print(f"claims: {len(selected)}; variants: {variants}; workers: {args.workers}")

    client = LLMClient(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        reasoning_effort=args.reasoning_effort,
        extra_body=json.loads(args.extra_body) if args.extra_body else None,
    )

    retrieved_by_id: dict[str, list] = {}
    for year in TEST_YEARS:
        boundary = parse_date(f"{year}-01-01")
        year_test = [record for record in selected if int(record.get("cutoff_year") or 0) == year]
        for fold in range(FOLDS):
            train_rows = [
                row
                for row in rows
                if fold_of(company_key(row[2]), 1) != fold and row[1] < boundary
            ]
            fold_test = [record for record in year_test if fold_of(company_key(record), 1) == fold]
            if not train_rows or not fold_test:
                continue
            for record in fold_test:
                retrieved_by_id[record["claim_id"]] = retrieve(
                    train_rows, record, by_company, args.k, mode="multi"
                )

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        started = time.time()
        method = METHODS[variant]
        rows_out: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    method,
                    client,
                    record,
                    by_company,
                    retrieved_by_id[record["claim_id"]],
                    args.max_tokens,
                ): record
                for record in selected
            }
            for future in as_completed(futures):
                record = futures[future]
                result = future.result()
                ids = [
                    str(item["claim_id"])
                    for (_, _, item), _ in retrieved_by_id[record["claim_id"]]
                ]
                rows_out.append(
                    {
                        "claim_id": record["claim_id"],
                        "variant": variant,
                        "truth": str(record["outcome"]["bucket"]),
                        "parsed": result.get("parsed"),
                        "error": result.get("error"),
                        "raw_text": result.get("raw_text", ""),
                        "usage": result.get("usage", {}),
                        "latency_seconds": result.get("latency_seconds", 0.0),
                        "route": result.get("route"),
                        "retrieved_ids": ids,
                    }
                )
        rows_out.sort(key=lambda row: row["claim_id"])
        all_rows[variant] = rows_out
        valid = sum(1 for row in rows_out if row["parsed"] is not None)
        print(f"{variant}: {valid}/{len(rows_out)} valid in {time.time() - started:.1f}s")

    with (args.out / "per_claim.jsonl").open("w", encoding="utf-8") as handle:
        for variant, variant_rows in all_rows.items():
            for row in variant_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema": "finplan-official-source-baselines-v1",
        "data": str(args.data),
        "n_claims": len(selected),
        "verified_only": bool(args.verified_only),
        "provider": client.provider,
        "backbone": client.model,
        "base_url": client.base_url,
        "variant_sources": {
            name: PROVENANCE["adaptiverag" if name == "adaptive" else name]
            for name in variants
        },
        "adaptation": ADAPTER,
        "variants": variants,
        "latency_seconds": {
            name: sum(row["latency_seconds"] for row in rows_out)
            for name, rows_out in all_rows.items()
        },
        "usage": {
            name: {
                "input_tokens": sum(row["usage"].get("input_tokens", 0) for row in rows_out),
                "output_tokens": sum(row["usage"].get("output_tokens", 0) for row in rows_out),
            }
            for name, rows_out in all_rows.items()
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "provenance.json").write_text(
        json.dumps(prompt_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {sum(len(v) for v in all_rows.values())} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
