"""Run DeepSeek point-answer, verification, selective, and planner baselines.

Usage examples:
    PYTHONPATH=src:scripts python scripts/run_deepseek_baselines.py --limit 20
    PYTHONPATH=src:scripts python scripts/run_deepseek_baselines.py --variants point,masked,evidence,selective
    PYTHONPATH=src:scripts python scripts/run_deepseek_baselines.py --self-consistency 5 --planner

The script caches every API response by prompt hash. It never prints the key.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from run_real_benchmark import (
    LABELS,
    balanced_accuracy,
    company_key,
    load_records,
    macro_f1,
    parse_date,
)
from run_real_decision_eval import planner_choice
from run_company_disjoint_benchmark import run_fold_seed
from finplan_rag.decision.loss import LossConfig, action_loss
from deepseek_client import DeepSeekClient, DeepSeekError

REPO_ROOT = Path(__file__).resolve().parents[1]
# Released benchmark copy. Override with --data if needed.
DEFAULT_DATA = REPO_ROOT / "paper" / "data" / "dataset_v4.json"
OUT = REPO_ROOT / "paper" / "results" / "deepseek_baselines"

SYSTEM = (
    "You are a careful financial analyst. Use only information available at "
    "the cutoff date. Do not use later filings or future outcomes. Return only "
    "the requested JSON object, with no markdown and no extra text."
)


def load_model_prior(rows, by_company, verified_only: bool) -> dict[str, dict[str, float]]:
    result = run_fold_seed(
        rows,
        by_company,
        1,
        calibrate=True,
        verified_only=verified_only,
    )
    prior = {}
    for key, probabilities in zip(result["keys"], result["probabilities"]):
        prior[str(key)] = {
            label: float(probability)
            for label, probability in zip(LABELS, probabilities)
        }
    return prior


def history_for(record: dict[str, Any], by_company: dict[str, list]) -> str:
    key = company_key(record)
    cutoff = parse_date(record.get("cutoff"))
    past = [
        (c, o, r)
        for c, o, r in by_company.get(key, [])
        if cutoff is not None and o < cutoff
    ]
    counts = Counter(str(r["outcome"]["bucket"]) for _, _, r in past)
    n = sum(counts.values())
    if n == 0:
        return "No prior outcomes are public before the cutoff."
    parts = ", ".join(f"{label}={counts[label]}" for label in LABELS)
    surprises = []
    for _, outcome_date, prior in sorted(past, key=lambda item: item[1]):
        prior_mid = float(prior.get("guidance_mid_musd") or 0.0)
        actual = float(prior.get("outcome", {}).get("actual_musd") or 0.0)
        if prior_mid:
            surprises.append((outcome_date, actual / prior_mid - 1.0))
    surprise_text = ""
    if surprises:
        mean_surprise = sum(value for _, value in surprises) / len(surprises)
        last_surprise = surprises[-1][1]
        surprise_text = (
            f" Historical mean surprise vs guidance midpoint: {mean_surprise:+.3%}; "
            f"most recent surprise: {last_surprise:+.3%}."
        )
    previous = sorted(
        [(c, r) for c, _, r in past if c < cutoff],
        key=lambda item: item[0],
    )
    previous_text = ""
    if previous:
        prior = previous[-1][1]
        previous_text = (
            f" Previous guidance midpoint was ${float(prior['guidance_mid_musd']):.1f}M; "
            f"guidance width was {float(prior['guidance_width_pct']):.3f}."
        )
    return (
        f"Prior outcomes public before cutoff: n={n}; {parts}.{previous_text}"
        f"{surprise_text}"
    )


FEW_SHOT_IDS = (
    ("guide_search_inod_20160511_100665", "above"),
    ("guide_search_cik1540400_20141008_366680", "within"),
    ("guide_search_cik846876_20120117_000002", "below"),
)


def few_shot_block(by_company: dict[str, list]) -> str:
    by_id = {}
    for values in by_company.values():
        for _, _, record in values:
            by_id[str(record.get("claim_id"))] = record
    lines = [
        "Examples from earlier filings (not this claim). The current-quarter "
        "value is often at the range floor, yet the actual outcome can be below, "
        "within, or above; history and the guidance width matter."
    ]
    for index, (claim_id, actual) in enumerate(FEW_SHOT_IDS, start=1):
        record = by_id.get(claim_id)
        if record is None:
            continue
        low = float(record["guidance_low_musd"])
        high = float(record["guidance_high_musd"])
        mid = float(record["guidance_mid_musd"])
        current = float(record["current_q_musd"])
        claim = str(record.get("claim_text") or "").replace("\n", " ")[:260]
        lines.append(
            f"Example {index} (actual outcome: {actual})\n"
            f"Claim: {claim}\n"
            f"Guided range: ${low:.1f}M to ${high:.1f}M (midpoint ${mid:.1f}M). "
            f"Current-quarter value at cutoff: ${current:.1f}M.\n"
            f"{history_for(record, by_company)}"
        )
    return "\n\n".join(lines) + "\n\n"


def prompt_for(
    record: dict[str, Any],
    by_company: dict[str, list],
    variant: str,
    prompt_version: str = "v1",
    model_prior: dict[str, float] | None = None,
) -> str:
    low = float(record["guidance_low_musd"])
    high = float(record["guidance_high_musd"])
    mid = float(record["guidance_mid_musd"])
    current = float(record["current_q_musd"])
    width = (high - low) / mid if mid else 0.0
    company = str(record.get("company") or "Unknown")
    ticker = str(record.get("ticker") or "N/A")
    if variant in ("masked", "masked_verification"):
        company = "Company A"
        ticker = "N/A"
    target_year = record.get("target_year")
    target_quarter = record.get("target_quarter")
    period = (
        f"{target_year} Q{int(target_quarter)}"
        if target_quarter
        else "the period named in the claim"
    )
    claim = str(record.get("claim_text") or "").replace("\n", " ")[:1400]
    history = history_for(record, by_company)
    base = (
        f"Cutoff date: {record.get('cutoff')}\n"
        f"Company: {company} (ticker: {ticker})\n"
        f"Sector: {record.get('sector') or 'other'}\n"
        f"Target period: {period}\n"
        f"Claim: {claim}\n"
        f"Guided range: ${low:.1f}M to ${high:.1f}M (midpoint ${mid:.1f}M; "
        f"width {width:.3f} of midpoint).\n"
        f"Current-quarter value available at cutoff: ${current:.1f}M. This is a "
        f"preliminary or contemporaneous estimate, not the final reported actual.\n"
        f"{history}\n"
    )
    if prompt_version in ("v2", "v3"):
        base += (
            "The final reported actual is not constrained to the guided range; "
            "companies can miss below or beat above. Do not default to within "
            "merely because the current-quarter value lies inside the range. "
            "Use the company history, including its mean surprise, and the "
            "guidance width to form an explicit revenue estimate before "
            "classifying.\n"
        )
    if prompt_version == "v3":
        base = few_shot_block(by_company) + base
    if model_prior:
        base += (
            "A calibrated statistical model, trained only on other companies and "
            "earlier outcomes, gives the following probabilities for this claim: "
            + ", ".join(f"{label}={model_prior.get(label, 0.0):.3f}" for label in LABELS)
            + ". Use this as a prior. Override it only if the evidence clearly "
            "contradicts the model.\n"
        )
    if variant == "evidence":
        evidence_lines = []
        for item in record.get("evidence", []):
            if item.get("kind") == "guidance":
                evidence_lines.append(
                    f"- guidance: ${float(item['low_musd']):.1f}M to "
                    f"${float(item['high_musd']):.1f}M, dated {item.get('date')}"
                )
            elif item.get("kind") == "current_q":
                evidence_lines.append(
                    f"- current quarter: ${float(item['value_musd']):.1f}M, "
                    f"reliability={item.get('reliability')}, dated {item.get('date')}"
                )
        base += "Evidence available at cutoff:\n" + "\n".join(evidence_lines) + "\n"
    estimate_prefix = (
        "First estimate the final reported revenue in $M from the current-quarter "
        "estimate and the company history, then compare that estimate with the "
        "guided range. "
        if prompt_version in ("v2", "v3")
        else ""
    )
    estimate_field = "\"estimate_musd\":0.0," if prompt_version in ("v2", "v3") else ""
    if variant == "selective":
        instruction = (
            estimate_prefix
            + "Predict whether actual revenue for the target period will be below, "
            "within, or above the guided range. If the evidence is insufficient "
            "to choose one label, return label \"abstain\" instead of guessing.\n"
            "Return only: {" + estimate_field + "\"label\":\"below|within|above|abstain\","
            "\"confidence\":0.0,\"reason\":\"at most 25 words\"}"
        )
    elif variant in ("verification", "masked_verification"):
        instruction = (
            estimate_prefix
            + "Decide whether the claim is likely to be below, within, or above "
            "the guided range. Also state whether the available evidence is "
            "sufficient for a point answer and give the smallest credible set.\n"
            "Return only: {" + estimate_field + "\"label\":\"below|within|above\","
            "\"confidence\":0.0,\"sufficient\":true,"
            "\"credible_set\":[\"below|within|above\"],"
            "\"reason\":\"at most 25 words\"}"
        )
    else:
        instruction = (
            estimate_prefix
            + "Predict whether actual revenue for the target period will be below, "
            "within, or above the guided range.\n"
            "Return only: {" + estimate_field + "\"label\":\"below|within|above\","
            "\"confidence\":0.0,\"reason\":\"at most 25 words\"}"
        )
    return base + instruction


def parse_output(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value


def normalize_output(value: dict[str, Any]) -> dict[str, Any] | None:
    label = str(value.get("label", "")).strip().lower()
    if label not in (*LABELS, "abstain"):
        return None
    try:
        confidence = float(value.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(1.0, max(0.0, confidence))
    result: dict[str, Any] = {
        "label": label,
        "confidence": confidence,
        "reason": str(value.get("reason", ""))[:300],
    }
    if "estimate_musd" in value:
        try:
            result["estimate_musd"] = float(value.get("estimate_musd"))
        except (TypeError, ValueError):
            pass
    if "sufficient" in value:
        result["sufficient"] = bool(value.get("sufficient"))
    if "credible_set" in value:
        credible = value.get("credible_set")
        if isinstance(credible, list):
            result["credible_set"] = [str(x).lower() for x in credible if str(x).lower() in LABELS]
        else:
            result["credible_set"] = []
    return result


def call_one(
    client: DeepSeekClient,
    record: dict[str, Any],
    by_company: dict[str, list],
    variant: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    seed: int | None = None,
    prompt_version: str = "v1",
    thinking_budget: int = 0,
    model_prior: dict[str, float] | None = None,
) -> dict[str, Any]:
    prompt = prompt_for(
        record,
        by_company,
        variant,
        prompt_version=prompt_version,
        model_prior=model_prior,
    )
    started = time.time()
    try:
        response = client.complete(
            prompt,
            system=SYSTEM,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            thinking=thinking_budget > 0,
            thinking_budget=thinking_budget,
        )
        text = client.text_of(response)
        parsed = normalize_output(parse_output(text) or {})
        usage = client.usage_of(response)
        error = None
        if parsed is None:
            error = "parse_failure"
            repair_prompt = (
                prompt
                + "\nYour previous response was not valid JSON. Return only the "
                "requested JSON object, with label in below/within/above"
                + ("/abstain" if variant == "selective" else "")
                + " and confidence between 0 and 1."
            )
            repair = client.complete(
                repair_prompt,
                system=SYSTEM,
                max_tokens=max_tokens,
                temperature=0.0,
                seed=seed,
            )
            text = client.text_of(repair)
            parsed = normalize_output(parse_output(text) or {})
            error = None if parsed is not None else "parse_failure"
            usage = {
                "input_tokens": usage["input_tokens"] + client.usage_of(repair)["input_tokens"],
                "output_tokens": usage["output_tokens"] + client.usage_of(repair)["output_tokens"],
            }
        return {
            "claim_id": record["claim_id"],
            "variant": variant,
            "truth": str(record["outcome"]["bucket"]),
            "parsed": parsed,
            "error": error,
            "raw_text": text[:1000],
            "usage": usage,
            "latency_seconds": time.time() - started,
        }
    except DeepSeekError as error:
        return {
            "claim_id": record["claim_id"],
            "variant": variant,
            "truth": str(record["outcome"]["bucket"]),
            "parsed": None,
            "error": f"api_error:{error}",
            "raw_text": "",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "latency_seconds": time.time() - started,
        }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truths = [row["truth"] for row in rows]
    predictions = [row["parsed"]["label"] for row in rows if row["parsed"]]
    answered = [(row["truth"], row["parsed"]["label"]) for row in rows if row["parsed"] and row["parsed"]["label"] != "abstain"]
    n = len(rows)
    valid = len(predictions)
    correct = sum(1 for truth, pred in answered if truth == pred)
    return {
        "n": n,
        "valid": valid,
        "invalid_rate": (n - valid) / n if n else 0.0,
        "coverage": len(answered) / n if n else 0.0,
        "accuracy_answered": correct / len(answered) if answered else None,
        "accuracy_all": correct / n if n else 0.0,
        "balanced_accuracy_answered": (
            balanced_accuracy([t for t, _ in answered], [p for _, p in answered])
            if answered
            else None
        ),
        "macro_f1_answered": (
            macro_f1([t for t, _ in answered], [p for _, p in answered])
            if answered
            else None
        ),
        "abstain_rate": sum(1 for row in rows if row["parsed"] and row["parsed"]["label"] == "abstain") / n if n else 0.0,
    }


def point_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [(row["truth"], row["parsed"]["label"]) for row in rows if row["parsed"]]
    if not pairs:
        return {"n": len(rows), "valid": 0, "accuracy": None, "balanced_accuracy": None, "macro_f1": None}
    truths = [t for t, _ in pairs]
    preds = [p for _, p in pairs]
    return {
        "n": len(rows),
        "valid": len(pairs),
        "invalid_rate": (len(rows) - len(pairs)) / len(rows),
        "accuracy": float(np.mean([t == p for t, p in pairs])),
        "balanced_accuracy": balanced_accuracy(truths, preds),
        "macro_f1": macro_f1(truths, preds),
    }


def bootstrap_metric(rows: list[dict[str, Any]], metric: str, *, n_resamples: int = 5000, seed: int = 20261102) -> dict[str, Any]:
    pairs = [(row["truth"], row["parsed"]["label"]) for row in rows if row["parsed"] and row["parsed"]["label"] != "abstain"]
    if not pairs:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    truths = [t for t, _ in pairs]
    preds = [p for _, p in pairs]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(pairs), len(pairs))
        y = [truths[i] for i in index]
        p = [preds[i] for i in index]
        if metric == "accuracy":
            values.append(float(np.mean([a == b for a, b in zip(y, p)])))
        elif metric == "balanced_accuracy":
            values.append(balanced_accuracy(y, p))
        elif metric == "macro_f1":
            values.append(macro_f1(y, p))
        else:
            raise ValueError(metric)
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "n": len(pairs),
    }


def self_consistency(
    client: DeepSeekClient,
    records: list[dict[str, Any]],
    by_company: dict[str, list],
    *,
    samples: int,
    workers: int,
    prompt_version: str = "v1",
    thinking_budget: int = 0,
) -> list[dict[str, Any]]:
    tasks = []
    for record in records:
        for sample in range(samples):
            tasks.append((record, sample))
    outputs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                call_one,
                client,
                record,
                by_company,
                "point",
                temperature=0.7,
                seed=sample + 1,
                prompt_version=prompt_version,
                thinking_budget=thinking_budget,
            ): (record["claim_id"], sample)
            for record, sample in tasks
        }
        for future in as_completed(futures):
            claim_id, sample = futures[future]
            row = future.result()
            outputs[claim_id].append(row)
    results = []
    for record in records:
        rows = outputs.get(record["claim_id"], [])
        labels = [row["parsed"]["label"] for row in rows if row["parsed"] and row["parsed"]["label"] in LABELS]
        counts = Counter(labels)
        if counts:
            label = counts.most_common(1)[0][0]
            confidence = counts[label] / max(1, sum(counts.values()))
        else:
            label = None
            confidence = 0.0
        results.append(
            {
                "claim_id": record["claim_id"],
                "truth": str(record["outcome"]["bucket"]),
                "parsed": (
                    {"label": label, "confidence": confidence, "reason": "self-consistency"}
                    if label
                    else None
                ),
                "votes": dict(counts),
                "samples": len(rows),
                "usage": {
                    "input_tokens": sum(row.get("usage", {}).get("input_tokens", 0) for row in rows),
                    "output_tokens": sum(row.get("usage", {}).get("output_tokens", 0) for row in rows),
                },
                "latency_seconds": sum(row.get("latency_seconds", 0.0) for row in rows),
            }
        )
    return results


def planner_from_votes(rows: list[dict[str, Any]], config: LossConfig) -> dict[str, Any]:
    losses = []
    actions = Counter()
    for row in rows:
        votes = row.get("votes", {})
        total = sum(votes.get(label, 0) for label in LABELS)
        if total == 0:
            continue
        belief = {label: votes.get(label, 0) / total for label in LABELS}
        probability = np.asarray([belief[label] for label in LABELS], dtype=float)
        action, prediction, credible, _ = planner_choice(probability, config)
        actions[action] += 1
        losses.append(action_loss(action, row["truth"], config, prediction=prediction, credible_set=credible))
    return {
        "n": len(losses),
        "mean_decision_loss": float(np.mean(losses)) if losses else None,
        "answer_rate": actions["answer"] / len(losses) if losses else None,
        "adjudicate_rate": actions["adjudicate"] / len(losses) if losses else None,
        "abstain_rate": actions["abstain"] / len(losses) if losses else None,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.4f}"


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# DeepSeek baselines",
        "",
        f"- data: `{summary['data']}`",
        f"- claims: {summary['n_claims']}",
        f"- verified only: {summary['verified_only']}",
        f"- variants: {', '.join(summary['variants'])}",
        "",
        "## Point-answer and selective results",
        "",
        "| variant | valid | accuracy | balanced acc | macro F1 | coverage | abstain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in summary["results"].items():
        lines.append(
            f"| {name} | {result.get('valid', 0)} | "
            f"{fmt(result.get('accuracy', result.get('accuracy_all')))} | "
            f"{fmt(result.get('balanced_accuracy', result.get('balanced_accuracy_answered')))} | "
            f"{fmt(result.get('macro_f1', result.get('macro_f1_answered')))} | "
            f"{fmt(result.get('coverage', 1.0))} | {fmt(result.get('abstain_rate', 0.0))} |"
        )
    lines += [
        "",
        "## Bootstrap intervals",
        "",
        "| variant | metric | mean | 95% CI |",
        "|---|---|---:|---|",
    ]
    for name, intervals in summary.get("bootstrap", {}).items():
        for metric, value in intervals.items():
            if value.get("mean") is None:
                continue
            lines.append(
                f"| {name} | {metric} | {value['mean']:.4f} | "
                f"[{value['ci_low']:.4f}, {value['ci_high']:.4f}] |"
            )
    lines += [
        "",
        "## Usage",
        "",
        f"- input tokens: {summary['usage']['input_tokens']}",
        f"- output tokens: {summary['usage']['output_tokens']}",
        f"- total latency (seconds): {summary['usage']['latency_seconds']:.1f}",
        "",
        "## Bad cases",
        "",
        "See `bad_cases.md` for the first errors by variant.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=0, help="pilot limit; 0 means all")
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--split", choices=["test", "dev", "validation", "dev_validation", "all"], default="test")
    parser.add_argument(
        "--variants",
        default="point,masked,evidence,selective,verification",
        help="comma-separated: point,masked,masked_verification,evidence,selective,verification",
    )
    parser.add_argument("--self-consistency", type=int, default=0)
    parser.add_argument("--planner", action="store_true", help="evaluate a planner on self-consistency votes")
    parser.add_argument("--prompt-version", choices=["v1", "v2", "v3"], default="v3")
    parser.add_argument("--thinking-budget", type=int, default=0, help="0 disables reasoning; >0 enables it with this budget")
    parser.add_argument("--model-prior", action="store_true", help="include the calibrated logistic model probabilities in the prompt")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4000)
    args = parser.parse_args()

    if args.self_consistency:
        variants = ["self_consistency"]
    else:
        variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    client = DeepSeekClient()
    rows = load_records(args.data)
    by_company: dict[str, list] = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    if args.split == "all":
        split_ok = lambda record: True
    elif args.split == "dev_validation":
        split_ok = lambda record: record.get("split") in ("dev", "validation")
    else:
        split_ok = lambda record: record.get("split") == args.split
    test = [
        record
        for _, _, record in rows
        if split_ok(record)
        and (
            not args.verified_only
            or record.get("verification") in {"xbrl_search", "xbrl_verified", "xbrl_override"}
        )
    ]
    test = sorted(test, key=lambda record: str(record["cutoff"]))
    if args.limit:
        test = test[: args.limit]
    prior = load_model_prior(rows, by_company, args.verified_only) if args.model_prior else None
    if prior:
        print(f"loaded logistic prior for {len(prior)} claims")
    print(f"claims: {len(test)}; variants: {variants}; workers: {args.workers}")

    results: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0}

    if args.self_consistency:
        rows_out = self_consistency(
            client,
            test,
            by_company,
            samples=args.self_consistency,
            workers=args.workers,
            prompt_version=args.prompt_version,
            thinking_budget=args.thinking_budget,
        )
        all_rows["self_consistency"] = rows_out
        for row in rows_out:
            usage["input_tokens"] += row.get("usage", {}).get("input_tokens", 0)
            usage["output_tokens"] += row.get("usage", {}).get("output_tokens", 0)
            usage["latency_seconds"] += row.get("latency_seconds", 0.0)
        results["self_consistency"] = point_metrics(rows_out)
        bootstrap["self_consistency"] = {
            metric: bootstrap_metric(rows_out, metric)
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        summary_extra = {}
        if args.planner:
            summary_extra["planner"] = planner_from_votes(rows_out, LossConfig())
        results["self_consistency"].update(summary_extra)
    else:
        tasks = [(record, variant) for record in test for variant in variants]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    call_one,
                    client,
                    record,
                    by_company,
                    variant,
                    max_tokens=args.max_tokens,
                    prompt_version=args.prompt_version,
                    thinking_budget=args.thinking_budget,
                    model_prior=prior.get(record["claim_id"]) if prior else None,
                ): (record, variant)
                for record, variant in tasks
            }
            for future in as_completed(futures):
                record, variant = futures[future]
                row = future.result()
                all_rows.setdefault(variant, []).append(row)
                usage["input_tokens"] += row["usage"]["input_tokens"]
                usage["output_tokens"] += row["usage"]["output_tokens"]
                usage["latency_seconds"] += row["latency_seconds"]
        for variant in variants:
            rows_out = sorted(all_rows.get(variant, []), key=lambda row: row["claim_id"])
            if variant == "selective":
                results[variant] = metrics(rows_out)
            else:
                results[variant] = point_metrics(rows_out)
            bootstrap[variant] = {
                metric: bootstrap_metric(rows_out, metric)
                for metric in ("accuracy", "balanced_accuracy", "macro_f1")
            }

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "finplan-deepseek-baselines-v1",
        "data": str(args.data),
        "n_claims": len(test),
        "verified_only": bool(args.verified_only),
        "variants": variants,
        "results": results,
        "bootstrap": bootstrap,
        "usage": usage,
        "model": client.model,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out / "per_claim.jsonl").open("w", encoding="utf-8") as handle:
        for variant, rows_out in all_rows.items():
            for row in rows_out:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    bad_lines = ["# Bad cases", ""]
    for variant, rows_out in all_rows.items():
        errors = [row for row in rows_out if row.get("parsed") is None or row["parsed"]["label"] != row["truth"]]
        bad_lines.append(f"## {variant}: {len(errors)} errors / {len(rows_out)}")
        bad_lines.append("")
        for row in errors[:25]:
            parsed = row.get("parsed") or {}
            bad_lines.append(
                f"- `{row['claim_id']}` truth={row['truth']} pred={parsed.get('label')} "
                f"conf={parsed.get('confidence')} error={row.get('error')} "
                f"reason={parsed.get('reason', '')[:120]}"
            )
        bad_lines.append("")
    (args.out / "bad_cases.md").write_text("\n".join(bad_lines), encoding="utf-8")
    report = render(summary)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
