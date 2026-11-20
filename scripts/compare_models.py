"""Compare per-claim model outputs with paired bootstrap and McNemar tests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from compute_metrics_suite import LABELS, compute

REPO_ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    by_claim = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_claim[row["claim_id"]].append(row)
    ids = sorted(by_claim)
    truths = np.asarray([LABELS.index(by_claim[i][0]["truth"]) for i in ids], dtype=int)
    probabilities = np.asarray(
        [np.mean([row["probabilities"] for row in by_claim[i]], axis=0) for i in ids],
        dtype=float,
    )
    return ids, truths, probabilities


def bootstrap_difference(a, b, n_boot=2000, seed=20261103):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(n_boot)]
    )
    return {
        "mean": float(np.mean(diff)),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def bootstrap_metric_difference(prob_a, prob_b, truths, metric, n_boot=500, seed=20261103):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        index = rng.integers(0, len(truths), size=len(truths))
        if len(np.unique(truths[index])) < 2:
            continue
        try:
            values.append(
                compute(prob_a[index], truths[index], bootstrap=False)[metric]
                - compute(prob_b[index], truths[index], bootstrap=False)[metric]
            )
        except ValueError:
            continue
    return {
        "mean": float(np.mean(values)) if values else float("nan"),
        "ci95": [
            float(np.quantile(values, 0.025)) if values else float("nan"),
            float(np.quantile(values, 0.975)) if values else float("nan"),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="name=path")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = {}
    for item in args.inputs:
        name, raw_path = item.split("=", 1)
        data[name] = load(Path(raw_path))
    # Align on the intersection of claim ids. A parser failure can drop a claim
    # from one baseline; paired tests must then use the common claims.
    common = sorted(set.intersection(*(set(item[0]) for item in data.values())))
    if not common:
        raise ValueError("no common claims across inputs")
    aligned = {}
    for name, (ids, truths, probabilities) in data.items():
        index = {claim_id: i for i, claim_id in enumerate(ids)}
        keep = [index[claim_id] for claim_id in common]
        aligned[name] = (
            common,
            truths[keep],
            probabilities[keep],
        )
    data = aligned
    reference = next(iter(data))
    ref_ids, ref_truths, ref_prob = data[reference]
    result = {"reference": reference, "n_common": len(common), "models": {}, "comparisons": {}}
    for name, (ids, truths, probabilities) in data.items():
        if ids != ref_ids:
            raise ValueError(f"claim order mismatch for {name}")
        metrics = compute(probabilities, truths, bootstrap=False)
        result["models"][name] = {
            key: metrics[key]
            for key in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "mcc",
                "kappa",
                "auroc_macro",
                "auprc_macro",
                "ece",
                "brier",
                "nll",
                "selective_accuracy_50",
                "aurc",
            )
        }
        if name == reference:
            continue
        pred_ref = ref_prob.argmax(axis=1)
        pred = probabilities.argmax(axis=1)
        correct_ref = pred_ref == ref_truths
        correct = pred == truths
        b = int(np.sum(correct & ~correct_ref))
        c = int(np.sum(~correct & correct_ref))
        p_value = (
            float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
        )
        result["comparisons"][f"{name}_vs_{reference}"] = {
            "accuracy_difference": float(correct.mean() - correct_ref.mean()),
            "mcnemar_b": b,
            "mcnemar_c": c,
            "mcnemar_p": p_value,
            "accuracy_bootstrap": bootstrap_difference(correct, correct_ref),
            "auroc_bootstrap": bootstrap_metric_difference(
                probabilities, ref_prob, truths, "auroc_macro"
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["models"], indent=2))
    print(json.dumps(result["comparisons"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
