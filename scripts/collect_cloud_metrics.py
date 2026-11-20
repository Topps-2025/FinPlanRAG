"""Collect comparable metrics for every cloud method and backbone.

Reads the released per-claim probability files and writes one compact JSON
table used by the paper.  No model call is made.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from summarize_cloud_results import LOSS, LABELS, DISPLAY, actual_loss  # noqa: E402

ROOT = REPO_ROOT / "paper" / "results" / "cloud_2026-11-20"
METHODS = ["findver", "finfact", "selfrag", "crag", "flare", "adaptive"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ece(probabilities: np.ndarray, truths: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == truths).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def brier(probabilities: np.ndarray, truths: np.ndarray) -> float:
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(truths)), truths] = 1.0
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def aurc(probabilities: np.ndarray, truths: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    errors = (prediction != truths).astype(float)
    order = np.argsort(-confidence)
    cumulative = np.cumsum(errors[order])
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    risk = cumulative / np.arange(1, len(errors) + 1)
    return float(np.trapezoid(risk, coverage))


def selective_accuracy(probabilities: np.ndarray, truths: np.ndarray, fraction: float = 0.5) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    k = max(1, int(round(fraction * len(truths))))
    order = np.argsort(-confidence)[:k]
    return float((prediction[order] == truths[order]).mean())


def metric_block(probabilities: list[list[float]], truths: list[str]) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    y = np.asarray([label_to_index[value] for value in truths], dtype=int)
    prediction = probs.argmax(axis=1)
    losses = []
    actions = []
    set_sizes = []
    for index, truth in enumerate(truths):
        loss, readout = actual_loss(list(probs[index]), truth)
        losses.append(loss)
        actions.append(readout["action"])
        set_sizes.append(len(readout["set"]))
    return {
        "n": len(truths),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "kappa": float(cohen_kappa_score(y, prediction)),
        "nll": float(log_loss(y, probs, labels=list(range(len(LABELS))))),
        "brier": brier(probs, y),
        "ece": ece(probs, y),
        "auroc": float(roc_auc_score(y, probs, multi_class="ovr", average="macro")),
        "auprc": float(average_precision_score(y, probs, average="macro")),
        "aurc": aurc(probs, y),
        "selective_accuracy_50": selective_accuracy(probs, y),
        "decision_loss": float(np.mean(losses)),
        "coverage": float(np.mean([action != "abstain" for action in actions])),
        "mean_set_size": float(np.mean(set_sizes)),
        "abstain_rate": float(np.mean([action == "abstain" for action in actions])),
    }


def main() -> None:
    output: dict[str, Any] = {"schema": "finplan-cloud-metrics-v1", "backbones": {}}
    for backbone in ("deepseek", "glm", "qwen3"):
        block: dict[str, Any] = {}
        fusion_rows = load_jsonl(ROOT / backbone / "evolution" / "temporal_test_probs_new_cloud.jsonl")
        truths = [LABELS[row["truth"]] if isinstance(row["truth"], int) else str(row["truth"]) for row in fusion_rows]
        block["fusion"] = metric_block([row["probabilities"] for row in fusion_rows], truths)
        block["llm_only"] = metric_block([row["llm_probabilities"] for row in fusion_rows], truths)
        block["rf"] = metric_block([row["rf_probabilities"] for row in fusion_rows], truths)
        for method in METHODS:
            path = ROOT / backbone / "named" / f"prob_{method}.jsonl"
            rows = load_jsonl(path)
            block[DISPLAY[method]] = metric_block([row["probabilities"] for row in rows], [row["truth"] for row in rows])
        output["backbones"][backbone] = block
    target = ROOT / "metrics_table.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)
    for backbone, block in output["backbones"].items():
        print(backbone)
        for method, values in block.items():
            print(f"  {method:22s} acc={values['accuracy']:.3f} macro={values['macro_f1']:.3f} "
                  f"loss={values['decision_loss']:.3f} auroc={values['auroc']:.3f} ece={values['ece']:.3f}")


if __name__ == "__main__":
    main()
