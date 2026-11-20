"""Run named, strong baselines adapted to the claim-outcome benchmark.

The baselines share the same configured LLM backbone (DeepSeek, GLM 5.3
flash, or an OpenAI-compatible endpoint) and point-in-time evidence, but use
the decision/verification/retrieval strategy of a named method:

* FinDVer-style claim verification;
* Fin-Fact-style evidence-grounded fact checking;
* Self-RAG-style self-reflective retrieval;
* FLARE-style confidence-triggered active retrieval;
* Adaptive-RAG-style complexity routing;
* CRAG-style retrieval evaluation and correction.

These are adaptations, not official code releases. The script records the
adaptation name and never claims an exact reproduction of the original system.

Usage:
    PYTHONPATH="src;scripts" python scripts/run_named_baselines.py \
      --verified-only --variants findver,finfact,selfrag,crag,flare,adaptive \
      --out paper/results/named_baselines
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

from deepseek_client import DeepSeekError, LLMClient
from run_company_disjoint_benchmark import (
    FOLDS,
    TEST_YEARS,
    VERIFIED_LABELS,
    fold_of,
)
from run_deepseek_baselines import SYSTEM, history_for
from run_rag_baseline import DEFAULT_DATA, retrieve, serialize
from run_real_benchmark import LABELS, company_key, load_records, parse_date

REPO_ROOT = Path(__file__).resolve().parents[1]
SINGLE_CALL_VARIANTS = ("findver", "finfact", "selfrag", "crag")
ALL_VARIANTS = (*SINGLE_CALL_VARIANTS, "flare", "adaptive")


def retrieved_block(retrieved, by_company) -> str:
    lines = [
        "Retrieved earlier claims from other companies (their realized outcomes "
        "are public before the cutoff):"
    ]
    for index, ((_, _, item), score) in enumerate(retrieved, start=1):
        actual = str(item["outcome"]["bucket"])
        lines.append(
            f"{index}. similarity={score:.3f}; actual outcome={actual}; "
            f"{serialize(item, by_company)}"
        )
    return "\n".join(lines)


def base_prompt(record, by_company) -> str:
    low = float(record["guidance_low_musd"])
    high = float(record["guidance_high_musd"])
    mid = float(record["guidance_mid_musd"])
    current = float(record["current_q_musd"])
    width = (high - low) / mid if mid else 0.0
    target_year = record.get("target_year")
    target_quarter = record.get("target_quarter")
    period = (
        f"{target_year} Q{int(target_quarter)}"
        if target_quarter
        else "the period named in the claim"
    )
    claim = str(record.get("claim_text") or "").replace("\n", " ")[:1400]
    return (
        f"Cutoff date: {record.get('cutoff')}\n"
        f"Company: {record.get('company') or 'Unknown'} "
        f"(ticker: {record.get('ticker') or 'N/A'})\n"
        f"Sector: {record.get('sector') or 'other'}\n"
        f"Target period: {period}\n"
        f"Claim: {claim}\n"
        f"Guided range: ${low:.1f}M to ${high:.1f}M "
        f"(midpoint ${mid:.1f}M; width {width:.3f} of midpoint).\n"
        f"Current-quarter value available at cutoff: ${current:.1f}M. This is a "
        f"preliminary or contemporaneous estimate, not the final reported actual.\n"
        f"{history_for(record, by_company)}\n"
        "The final reported actual is not constrained to the guided range; "
        "companies can miss below or beat above.\n"
    )


def output_instruction() -> str:
    return (
        "Return only JSON with keys label (below/within/above), confidence "
        "(0 to 1), credible_set (the smallest subset of "
        "below/within/above supported by the evidence), and rationale "
        "(at most 30 words). Example: "
        '{"label":"within","confidence":0.72,'
        '"credible_set":["within","above"],"rationale":"..."}'
    )


def single_call_prompt(record, by_company, retrieved, variant: str) -> str:
    context = base_prompt(record, by_company)
    evidence = retrieved_block(retrieved, by_company)
    if variant == "findver":
        instruction = (
            "You are FinDVer, a financial claim-verification system. Verify the "
            "claim's realized outcome against the point-in-time evidence and the "
            "retrieved earlier claims. Treat retrieved outcomes as calibration "
            "evidence, not as direct evidence about this claim. If the evidence "
            "does not support one label, return the smallest credible set rather "
            "than guessing.\n"
        )
    elif variant == "finfact":
        instruction = (
            "You are Fin-Fact, an evidence-grounded financial fact checker. "
            "Assess the claim against the structured evidence and retrieved "
            "claims. First identify the direction of each relevant piece of "
            "evidence, then decide the realized outcome. Do not use future "
            "filings or outcomes. If the evidence is insufficient for a point "
            "verdict, return the smallest credible set.\n"
        )
    elif variant == "selfrag":
        instruction = (
            "You are Self-RAG, a self-reflective retrieval-augmented reader. "
            "For each retrieved claim, judge whether it is relevant and "
            "reliable; ignore irrelevant claims and claims whose outcome is not "
            "comparable. Then answer from the reliable evidence and the current "
            "claim. Return the smallest credible set when the evidence is not "
            "sufficient for one label.\n"
        )
    elif variant == "crag":
        instruction = (
            "You are CRAG, a corrective retrieval-augmented reader. Evaluate "
            "whether the retrieved claims are relevant and reliable. If they "
            "are, use them; if they are irrelevant or contradictory, correct "
            "the retrieval by ignoring them and relying on the structured "
            "point-in-time evidence. Then decide the realized outcome. Return "
            "the smallest credible set when needed.\n"
        )
    else:
        raise ValueError(variant)
    return context + "\n" + evidence + "\n\n" + instruction + output_instruction()


def flare_prompt(record, by_company, retrieved, initial: dict | None = None) -> str:
    context = base_prompt(record, by_company)
    if initial is None:
        instruction = (
            "You are FLARE, a forward-looking active-retrieval reader. First "
            "answer from the claim evidence alone, without retrieved claims. "
            "State a calibrated confidence. If confidence is below 0.70, the "
            "system will trigger retrieval and ask you to revise.\n"
        )
        return context + "\n" + instruction + output_instruction()
    label = initial.get("label", "unknown")
    confidence = float(initial.get("confidence", 0.0))
    instruction = (
        "You are FLARE, a forward-looking active-retrieval reader. Your "
        f"initial no-retrieval answer was {label} with confidence "
        f"{confidence:.2f}; active retrieval has now been triggered. Use the "
        "retrieved earlier claims to revise or confirm the answer. Do not use "
        "future outcomes.\n"
    )
    return (
        context
        + "\n"
        + retrieved_block(retrieved, by_company)
        + "\n\n"
        + instruction
        + output_instruction()
    )


def adaptive_classifier_prompt(record, by_company) -> str:
    return (
        base_prompt(record, by_company)
        + "\nYou are Adaptive-RAG's query-complexity classifier. Classify the "
        "evidence need as simple, moderate, or complex. simple means the "
        "structured evidence is likely sufficient; moderate means one retrieval "
        "pass is useful; complex means retrieval plus explicit verification is "
        "needed. Return only JSON: {\"complexity\":\"simple|moderate|complex\"}.\n"
    )


def adaptive_answer_prompt(record, by_company, retrieved, complexity: str) -> str:
    context = base_prompt(record, by_company)
    if complexity == "simple":
        instruction = (
            "You are Adaptive-RAG. The router selected the simple path: answer "
            "from the structured point-in-time evidence without retrieved "
            "claims.\n"
        )
        return context + "\n" + instruction + output_instruction()
    if complexity == "moderate":
        evidence = retrieved_block(retrieved[:3], by_company)
        instruction = (
            "You are Adaptive-RAG. The router selected the moderate path: use "
            "one retrieval pass with the three most similar earlier claims, "
            "then answer.\n"
        )
    else:
        evidence = retrieved_block(retrieved, by_company)
        instruction = (
            "You are Adaptive-RAG. The router selected the complex path: use "
            "retrieval and explicit verification against the retrieved claims "
            "before answering.\n"
        )
    return context + "\n" + evidence + "\n\n" + instruction + output_instruction()


def parse_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def normalize(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    label = str(value.get("label", "")).strip().lower()
    if label not in LABELS:
        return None
    try:
        confidence = float(value.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(1.0, max(0.0, confidence))
    credible = value.get("credible_set")
    if not isinstance(credible, list):
        credible = [label]
    credible = [str(item).lower() for item in credible if str(item).lower() in LABELS]
    if label not in credible:
        credible = [label, *credible]
    return {
        "label": label,
        "confidence": confidence,
        "credible_set": credible,
        "rationale": str(value.get("rationale", ""))[:400],
    }


def call(client, prompt: str, max_tokens: int) -> dict[str, Any]:
    started = time.time()
    try:
        response = client.complete(
            prompt,
            system=SYSTEM,
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
                + "\nYour previous response was not valid JSON. Return only the "
                "requested JSON object with label below/within/above and a "
                "credible_set list.",
                system=SYSTEM,
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
            "raw_text": text[:1200],
            "usage": usage,
            "latency_seconds": time.time() - started,
            "error": None if parsed else "parse_failure",
        }
    except DeepSeekError as error:
        return {
            "parsed": None,
            "raw_text": "",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "latency_seconds": time.time() - started,
            "error": f"api_error:{error}",
        }


def call_complexity(client, prompt: str, max_tokens: int) -> dict[str, Any]:
    started = time.time()
    try:
        response = client.complete(
            prompt,
            system=SYSTEM,
            max_tokens=max_tokens,
            temperature=0.0,
            thinking=False,
            use_cache=True,
        )
        text = client.text_of(response)
        value = parse_json(text) or {}
        complexity = str(value.get("complexity", "complex")).lower()
        if complexity not in ("simple", "moderate", "complex"):
            complexity = "complex"
        return {
            "parsed": {"complexity": complexity},
            "raw_text": text[:1200],
            "usage": client.usage_of(response),
            "latency_seconds": time.time() - started,
            "error": None,
        }
    except DeepSeekError as error:
        return {
            "parsed": {"complexity": "complex"},
            "raw_text": "",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "latency_seconds": time.time() - started,
            "error": f"api_error:{error}",
        }


def make_row(record, variant, result, *, route=None, retrieved_ids=None) -> dict[str, Any]:
    return {
        "claim_id": record["claim_id"],
        "variant": variant,
        "truth": str(record["outcome"]["bucket"]),
        "parsed": result["parsed"],
        "error": result["error"],
        "raw_text": result["raw_text"],
        "usage": result["usage"],
        "latency_seconds": result["latency_seconds"],
        "route": route,
        "retrieved_ids": retrieved_ids or [],
    }


def run_single_call_variant(
    client, variant, test, retrieved_by_id, by_company, workers, max_tokens
):
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                call,
                client,
                single_call_prompt(record, by_company, retrieved_by_id[record["claim_id"]], variant),
                max_tokens,
            ): record
            for record in test
        }
        for future in as_completed(futures):
            record = futures[future]
            result = future.result()
            ids = [
                str(item["claim_id"])
                for (_, _, item), _ in retrieved_by_id[record["claim_id"]]
            ]
            rows.append(make_row(record, variant, result, retrieved_ids=ids))
    return sorted(rows, key=lambda row: row["claim_id"])


def run_flare(
    client, test, retrieved_by_id, by_company, workers, max_tokens, threshold=0.70
):
    initial: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                call,
                client,
                flare_prompt(record, by_company, []),
                max_tokens,
            ): record
            for record in test
        }
        for future in as_completed(futures):
            record = futures[future]
            initial[record["claim_id"]] = future.result()
    revisions = {
        record["claim_id"]: record
        for record in test
        if initial[record["claim_id"]]["parsed"] is None
        or initial[record["claim_id"]]["parsed"]["confidence"] < threshold
    }
    revised: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                call,
                client,
                flare_prompt(
                    record,
                    by_company,
                    retrieved_by_id[record["claim_id"]],
                    initial[record["claim_id"]]["parsed"]
                    or {"label": "unknown", "confidence": 0.0},
                ),
                max_tokens,
            ): record
            for record in revisions.values()
        }
        for future in as_completed(futures):
            record = futures[future]
            revised[record["claim_id"]] = future.result()
    rows = []
    for record in test:
        claim_id = record["claim_id"]
        used_retrieval = claim_id in revisions
        result = revised[claim_id] if used_retrieval else initial[claim_id]
        ids = (
            [
                str(item["claim_id"])
                for (_, _, item), _ in retrieved_by_id[claim_id]
            ]
            if used_retrieval
            else []
        )
        rows.append(
            make_row(
                record,
                "flare",
                result,
                route="retrieve" if used_retrieval else "no_retrieval",
                retrieved_ids=ids,
            )
        )
    return sorted(rows, key=lambda row: row["claim_id"])


def run_adaptive(
    client, test, retrieved_by_id, by_company, workers, max_tokens
):
    routes: dict[str, str] = {}
    classification: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                call_complexity,
                client,
                adaptive_classifier_prompt(record, by_company),
                max_tokens,
            ): record
            for record in test
        }
        for future in as_completed(futures):
            record = futures[future]
            result = future.result()
            classification[record["claim_id"]] = result
            parsed = result["parsed"] or {}
            complexity = str(parsed.get("complexity", "complex")).lower()
            if complexity not in ("simple", "moderate", "complex"):
                complexity = "complex"
            routes[record["claim_id"]] = complexity
    answers: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                call,
                client,
                adaptive_answer_prompt(
                    record,
                    by_company,
                    retrieved_by_id[record["claim_id"]],
                    routes[record["claim_id"]],
                ),
                max_tokens,
            ): record
            for record in test
        }
        for future in as_completed(futures):
            record = futures[future]
            answers[record["claim_id"]] = future.result()
    rows = []
    for record in test:
        claim_id = record["claim_id"]
        route = routes[claim_id]
        ids = (
            [
                str(item["claim_id"])
                for (_, _, item), _ in retrieved_by_id[claim_id]
            ]
            if route in ("moderate", "complex")
            else []
        )
        rows.append(
            make_row(
                record,
                "adaptive",
                answers[claim_id],
                route=route,
                retrieved_ids=ids,
            )
        )
    return sorted(rows, key=lambda row: row["claim_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(ALL_VARIANTS))
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--extra-body", default=None, help="JSON merged into each request body")
    args = parser.parse_args()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(ALL_VARIANTS))
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
        and (
            not args.verified_only
            or record.get("verification") in VERIFIED_LABELS
        )
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
        year_test = [
            record
            for record in selected
            if int(record.get("cutoff_year") or 0) == year
        ]
        for fold in range(FOLDS):
            train_rows = [
                row
                for row in rows
                if fold_of(company_key(row[2]), 1) != fold
                and row[1] < boundary
            ]
            fold_test = [
                record
                for record in year_test
                if fold_of(company_key(record), 1) == fold
            ]
            if not train_rows or not fold_test:
                continue
            for record in fold_test:
                retrieved_by_id[record["claim_id"]] = retrieve(
                    train_rows, record, by_company, args.k, mode="multi"
                )

    all_rows: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        started = time.time()
        if variant in SINGLE_CALL_VARIANTS:
            rows_out = run_single_call_variant(
                client,
                variant,
                selected,
                retrieved_by_id,
                by_company,
                args.workers,
                args.max_tokens,
            )
        elif variant == "flare":
            rows_out = run_flare(
                client,
                selected,
                retrieved_by_id,
                by_company,
                args.workers,
                args.max_tokens,
            )
        else:
            rows_out = run_adaptive(
                client,
                selected,
                retrieved_by_id,
                by_company,
                args.workers,
                args.max_tokens,
            )
        all_rows[variant] = rows_out
        valid = sum(1 for row in rows_out if row["parsed"] is not None)
        print(f"{variant}: {valid}/{len(rows_out)} valid in {time.time() - started:.1f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "per_claim.jsonl").open("w", encoding="utf-8") as handle:
        for variant, rows_out in all_rows.items():
            for row in rows_out:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema": "finplan-named-baselines-v1",
        "data": str(args.data),
        "n_claims": len(selected),
        "verified_only": bool(args.verified_only),
        "backbone": client.model,
        "provider": client.provider,
        "base_url": client.base_url,
        "adaptations": {
            "findver": "FinDVer-style claim verification",
            "finfact": "Fin-Fact-style evidence-grounded fact checking",
            "selfrag": "Self-RAG-style self-reflective retrieval",
            "flare": "FLARE-style confidence-triggered active retrieval",
            "adaptive": "Adaptive-RAG-style complexity routing",
            "crag": "CRAG-style corrective retrieval",
        },
        "variants": variants,
        "latency_seconds": {
            name: sum(row["latency_seconds"] for row in rows_out)
            for name, rows_out in all_rows.items()
        },
        "usage": {
            name: {
                "input_tokens": sum(row["usage"]["input_tokens"] for row in rows_out),
                "output_tokens": sum(row["usage"]["output_tokens"] for row in rows_out),
            }
            for name, rows_out in all_rows.items()
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {sum(len(v) for v in all_rows.values())} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
