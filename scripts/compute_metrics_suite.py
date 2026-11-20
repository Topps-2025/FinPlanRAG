"""Full metric suite for per-claim probability artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

LABELS = ("below", "within", "above")


def ece(probabilities, truths, bins=10):
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == truths).astype(float)
    indices = np.clip((confidences * bins).astype(int), 0, bins - 1)
    value = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(value)


def classwise_ece(probabilities, truths, bins=10):
    values = []
    for index in range(len(LABELS)):
        target = (truths == index).astype(float)
        confidence = probabilities[:, index]
        bucket = np.clip((confidence * bins).astype(int), 0, bins - 1)
        value = 0.0
        for b in range(bins):
            mask = bucket == b
            if mask.any():
                value += mask.mean() * abs(target[mask].mean() - confidence[mask].mean())
        values.append(value)
    return float(np.mean(values))


def calibration_slope(probabilities, truths):
    confidences = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == truths).astype(int)
    if len(np.unique(correct)) < 2:
        return {"slope": float("nan"), "intercept": float("nan")}
    clipped = np.clip(confidences, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(logit, correct)
    return {"slope": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}


def selective(probabilities, truths):
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    order = np.argsort(-confidences, kind="stable")
    correct = (predictions == truths)[order]
    n = len(correct)
    risks = []
    for coverage in np.linspace(0.1, 1.0, 10):
        k = max(1, int(math.ceil(coverage * n)))
        risks.append(1.0 - float(correct[:k].mean()))
    k50 = max(1, int(math.ceil(0.5 * n)))
    k80 = max(1, int(math.ceil(0.8 * n)))
    k90 = max(1, int(math.ceil(0.9 * n)))
    return {
        "selective_accuracy_50": float(correct[:k50].mean()),
        "risk_at_80": float(1.0 - correct[:k80].mean()),
        "risk_at_90": float(1.0 - correct[:k90].mean()),
        "aurc": float(np.mean(risks)),
    }


def bootstrap(values, n_boot=5000, seed=20261103):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    )
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def metric_values(probabilities, truths):
    probabilities = np.asarray(probabilities, dtype=float)
    truths = np.asarray(truths, dtype=int)
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(len(LABELS))[truths]
    return {
        "accuracy": accuracy_score(truths, predictions),
        "balanced_accuracy": balanced_accuracy_score(truths, predictions),
        "macro_f1": f1_score(truths, predictions, average="macro", zero_division=0),
        "auroc_macro": roc_auc_score(truths, probabilities, multi_class="ovr", average="macro"),
        "auprc_macro": average_precision_score(one_hot, probabilities, average="macro"),
        "ece": ece(probabilities, truths),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "nll": float(-np.mean(np.log(np.clip(probabilities[np.arange(len(truths)), truths], 1e-12, 1.0)))),
        **selective(probabilities, truths),
    }


def bootstrap_metric(probabilities, truths, key, n_boot=2000, seed=20261103):
    rng = np.random.default_rng(seed)
    values = []
    n = len(truths)
    for _ in range(n_boot):
        index = rng.integers(0, n, size=n)
        if len(np.unique(truths[index])) < 2:
            continue
        try:
            values.append(metric_values(probabilities[index], truths[index])[key])
        except ValueError:
            continue
    if not values:
        return [float("nan"), float("nan")]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def compute(probabilities, truths, bootstrap=True):
    probabilities = np.asarray(probabilities, dtype=float)
    truths = np.asarray(truths, dtype=int)
    predictions = probabilities.argmax(axis=1)
    per_class = precision_recall_fscore_support(
        truths, predictions, labels=range(len(LABELS)), zero_division=0
    )
    one_hot = np.eye(len(LABELS))[truths]
    metrics = {
        "n": int(len(truths)),
        "accuracy": float(accuracy_score(truths, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truths, predictions)),
        "macro_f1": float(f1_score(truths, predictions, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(truths, predictions)),
        "kappa": float(cohen_kappa_score(truths, predictions)),
        "per_class": {
            label: {
                "precision": float(per_class[0][i]),
                "recall": float(per_class[1][i]),
                "f1": float(per_class[2][i]),
                "support": int(per_class[3][i]),
            }
            for i, label in enumerate(LABELS)
        },
        "auroc_macro": float(
            roc_auc_score(truths, probabilities, multi_class="ovr", average="macro")
        ),
        "auroc_weighted": float(
            roc_auc_score(truths, probabilities, multi_class="ovr", average="weighted")
        ),
        "auprc_macro": float(
            average_precision_score(one_hot, probabilities, average="macro")
        ),
        "auprc_weighted": float(
            average_precision_score(one_hot, probabilities, average="weighted")
        ),
        "ece": ece(probabilities, truths),
        "classwise_ece": classwise_ece(probabilities, truths),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "nll": float(
            -np.mean(
                np.log(
                    np.clip(
                        probabilities[np.arange(len(truths)), truths], 1e-12, 1.0
                    )
                )
            )
        ),
        **calibration_slope(probabilities, truths),
        **selective(probabilities, truths),
    }
    metrics["per_class_auroc"] = {
        label: float(
            roc_auc_score(
                (truths == index).astype(int), probabilities[:, index]
            )
        )
        for index, label in enumerate(LABELS)
    }
    metrics["per_class_auprc"] = {
        label: float(
            average_precision_score(
                (truths == index).astype(int), probabilities[:, index]
            )
        )
        for index, label in enumerate(LABELS)
    }
    metrics["bootstrap"] = {} if not bootstrap else {
        key: bootstrap_metric(probabilities, truths, key)
        for key in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "auroc_macro",
            "auprc_macro",
            "ece",
            "brier",
            "nll",
            "selective_accuracy_50",
            "aurc",
        )
    }
    metrics["risk_coverage"] = {
        f"{coverage:.2f}": float(
            1.0
            - np.mean(
                (
                    probabilities.argmax(axis=1)
                    == truths
                )[np.argsort(-probabilities.max(axis=1), kind="stable")][
                    : max(1, int(math.ceil(coverage * len(truths))))
                ]
            )
        )
        for coverage in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-claim", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.per_claim.read_text(encoding="utf-8").splitlines()
    ]
    by_claim = defaultdict(list)
    for row in rows:
        by_claim[row["claim_id"]].append(row)
    claims = sorted(by_claim)
    truths = np.asarray(
        [LABELS.index(by_claim[claim][0]["truth"]) for claim in claims], dtype=int
    )
    probabilities = np.asarray(
        [
            np.mean([row["probabilities"] for row in by_claim[claim]], axis=0)
            for claim in claims
        ],
        dtype=float,
    )
    metrics = compute(probabilities, truths, bootstrap=not args.no_bootstrap)
    metrics["claim_ids"] = claims
    metrics["predicted_labels"] = [LABELS[i] for i in probabilities.argmax(axis=1)]
    slices = {}
    for field in ("year", "sector", "truth"):
        values = {}
        for value in sorted({str(by_claim[claim][0].get(field)) for claim in claims}):
            index = np.asarray(
                [
                    i
                    for i, claim in enumerate(claims)
                    if str(by_claim[claim][0].get(field)) == value
                ],
                dtype=int,
            )
            if len(index) < 5:
                continue
            values[value] = {
                "n": int(len(index)),
                "accuracy": float(
                    accuracy_score(truths[index], probabilities[index].argmax(axis=1))
                ),
                "balanced_accuracy": (
                    float(
                        balanced_accuracy_score(
                            truths[index], probabilities[index].argmax(axis=1)
                        )
                    )
                    if len(np.unique(truths[index])) > 1
                    else None
                ),
                "macro_f1": float(
                    f1_score(
                        truths[index],
                        probabilities[index].argmax(axis=1),
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        slices[field] = values
    metrics["slices"] = slices
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in metrics.items()
                if k not in ("claim_ids", "predicted_labels", "slices")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
