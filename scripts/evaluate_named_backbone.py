"""Evaluate one backbone's named-baseline outputs against the FinPlan reference.

The runner (``run_named_baselines.py``) stores parsed labels, confidences and
credible sets.  This script turns those into per-claim probability rows, computes
the full metric suite for every variant, and runs the paired comparison against
the frozen FinPlan logistic reference.

Usage:
    PYTHONPATH="src:scripts" python scripts/evaluate_named_backbone.py \
      --input paper/results/qwen3/named/per_claim.jsonl \
      --reference paper/results/benchmark_per_claim_verified_calibrated/per_claim.jsonl \
      --out paper/results/qwen3/named
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from compute_metrics_suite import LABELS, compute
from convert_llm_predictions import belief
from run_company_disjoint_benchmark import VERIFIED_LABELS
from run_real_benchmark import DEFAULT_DATA, load_records

REPO_ROOT = Path(__file__).resolve().parents[1]

DISPLAY = {
    "findver": "FinDVer-style",
    "finfact": "Fin-Fact-style",
    "selfrag": "Self-RAG-style",
    "crag": "CRAG-style",
    "flare": "FLARE-style",
    "adaptive": "Adaptive-RAG-style",
}


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--variants", default=",".join(DISPLAY))
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args()

    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(DISPLAY))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")

    records = {record["claim_id"]: record for _, _, record in load_records(args.data)}
    rows = load_rows(args.input)
    args.out.mkdir(parents=True, exist_ok=True)

    metrics_summary: dict[str, dict] = {}
    failures: dict[str, int] = {}
    prob_paths: dict[str, Path] = {}
    for variant in variants:
        variant_rows = [row for row in rows if row.get("variant") == variant]
        if not variant_rows:
            continue
        parsed_rows = []
        failure_count = 0
        for row in variant_rows:
            record = records.get(row["claim_id"])
            if record is None or (
                row.get("truth") and str(row["truth"]) not in LABELS
            ):
                continue
            parsed = row.get("parsed") or {}
            if parsed.get("label") not in LABELS:
                failure_count += 1
                continue
            parsed_rows.append(
                {
                    "seed": 1,
                    "claim_id": row["claim_id"],
                    "truth": row["truth"],
                    "probabilities": belief(parsed),
                    "year": int(record.get("cutoff_year") or 0),
                    "sector": record.get("sector"),
                    "variant": variant,
                }
            )
        failures[display_name(variant)] = failure_count
        if not parsed_rows:
            continue
        prob_path = args.out / f"prob_{variant}.jsonl"
        with prob_path.open("w", encoding="utf-8") as handle:
            for row in parsed_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        prob_paths[display_name(variant)] = prob_path

        truths = np.asarray([LABELS.index(row["truth"]) for row in parsed_rows], dtype=int)
        probabilities = np.asarray([row["probabilities"] for row in parsed_rows], dtype=float)
        metrics = compute(
            probabilities, truths, bootstrap=not args.no_bootstrap
        )
        with (args.out / f"metrics_{variant}.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        metrics_summary[display_name(variant)] = {
            key: metrics[key]
            for key in (
                "n",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "mcc",
                "auroc_macro",
                "auprc_macro",
                "ece",
                "brier",
                "nll",
                "selective_accuracy_50",
                "aurc",
            )
        }

    with (args.out / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_summary, handle, ensure_ascii=False, indent=2)
    (args.out / "parse_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    comparison_path = None
    if args.reference and args.reference.exists() and prob_paths:
        comparison_path = args.out / "model_comparison_logistic.json"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "compare_models.py"),
            "--inputs",
            f"FinPlan_Logistic={args.reference}",
            *[f"{name}={path}" for name, path in prob_paths.items()],
            "--out",
            str(comparison_path),
        ]
        subprocess.run(command, check=True)

    summary = {
        "input": str(args.input),
        "n_input_rows": len(rows),
        "variants": variants,
        "parse_failures": failures,
        "metrics": metrics_summary,
        "comparison": str(comparison_path) if comparison_path else None,
    }
    (args.out / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:3000])
    return 0


def display_name(variant: str) -> str:
    return DISPLAY.get(variant, variant)


if __name__ == "__main__":
    raise SystemExit(main())
