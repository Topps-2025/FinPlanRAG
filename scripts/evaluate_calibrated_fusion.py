"""Evaluate a historically calibrated anchor posterior fused with RF.

The calibration bundle is trained on validation beliefs only by
``train_anchor_calibration.py``.  This script applies the bundle to the test
beliefs, chooses the anchor/RF fusion weight on validation, and reports paired
bootstrap differences against RF and against the uncalibrated anchor fusion.

Usage:
    python scripts/evaluate_calibrated_fusion.py \
      --beliefs paper/results/cloud/deepseek/evolution/temporal_beliefs_new_cloud.jsonl \
      --bundle paper/results/cloud/deepseek/calibration/bundle.json \
      --rf-test paper/results/rf_calibrated_repro/per_claim.jsonl \
      --out paper/results/cloud/deepseek/calibrated_fusion.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from anchor_engine import answer_loss, full_action_readout, set_loss  # noqa: E402

LABELS = ("below", "within", "above")
LOSS = {"over": 12.0, "under": 8.0, "wrong": 10.0, "miss": 8.0, "width": 2.0, "abstain": 5.0}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values] if total > 0 else [1.0 / len(values)] * len(values)


def feature_vector(row: dict[str, Any]) -> list[float]:
    belief = normalize([float(value) for value in row.get("belief") or [0.0] * 3])
    scores = row.get("bucket_scores") or {}
    score = normalize([float(scores.get(label, 0.0)) for label in LABELS])
    entropy = -sum(value * math.log(value) for value in belief if value > 0)
    ordered = sorted(belief, reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else 0.0
    n_anchors = min(12.0, float(row.get("n_anchors") or 0.0)) / 12.0
    n_hard = min(12.0, float(row.get("n_hard") or 0.0)) / 12.0
    return [
        belief[0], belief[1], belief[2],
        score[0], score[1], score[2],
        entropy, margin, n_anchors, n_hard,
        float(row.get("mean_reliability") or 0.5),
        float(row.get("max_reliability") or 0.5),
    ]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def calibrated(rows: list[dict[str, Any]], bundle: dict[str, Any]) -> np.ndarray:
    x = np.asarray([feature_vector(row) for row in rows], dtype=float)
    mean = np.asarray(bundle["mean"], dtype=float)
    std = np.asarray(bundle["std"], dtype=float)
    coef = np.asarray(bundle["coef"], dtype=float)
    xb = np.concatenate([(x - mean) / std, np.ones((len(x), 1))], axis=1)
    return softmax((xb @ coef) / max(float(bundle.get("temperature") or 1.0), 1e-6))


def actual_loss(probabilities: list[float], truth: str) -> float:
    readout = full_action_readout(probabilities, LABELS, LOSS)
    if readout["action"] == "answer":
        return answer_loss(readout["point"], truth, LABELS, LOSS)
    if readout["action"] == "adjudicate":
        return set_loss(readout["set"], truth, LOSS)
    return float(LOSS["abstain"])


def evaluate(rows: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, Any]:
    truths = [row["truth"] for row in rows]
    pred = [LABELS[int(np.argmax(probabilities[i]))] for i in range(len(rows))]
    accuracy = sum(a == b for a, b in zip(pred, truths)) / len(rows)
    recalls = []
    for label in LABELS:
        mask = [i for i, truth in enumerate(truths) if truth == label]
        if mask:
            recalls.append(sum(pred[i] == label for i in mask) / len(mask))
    macro = 0.0
    for label in LABELS:
        tp = sum(pred[i] == label and truths[i] == label for i in range(len(rows)))
        fp = sum(pred[i] == label and truths[i] != label for i in range(len(rows)))
        fn = sum(pred[i] != label and truths[i] == label for i in range(len(rows)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        macro += 2 * precision * recall / max(precision + recall, 1e-9)
    losses = [actual_loss(probabilities[i].tolist(), truths[i]) for i in range(len(rows))]
    return {
        "n": len(rows),
        "accuracy": accuracy,
        "balanced_accuracy": sum(recalls) / max(len(recalls), 1),
        "macro_f1": macro / len(LABELS),
        "decision_loss": float(np.mean(losses)),
        "per_claim_loss": losses,
    }


def boot(a: list[float], b: list[float], n: int = 10000, seed: int = 20261120) -> dict[str, float]:
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)])
    return {
        "mean": float(diff.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beliefs", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--rf-test", default="paper/results/rf_calibrated_repro/per_claim.jsonl")
    parser.add_argument("--rf-validation", default="paper/results/evolution_v1/rf_validation.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.beliefs))
    validation = [row for row in rows if row.get("split") == "validation" and row.get("truth") in LABELS]
    test = [row for row in rows if row.get("split") == "test" and row.get("truth") in LABELS]
    bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))

    def load_rf(path: Path) -> dict[str, np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for row in load_jsonl(path):
            grouped[str(row["claim_id"])].append(np.asarray(row["probabilities"], dtype=float))
        return {key: np.mean(value, axis=0) for key, value in grouped.items()}

    rf = load_rf(Path(args.rf_test))
    rf_val = load_rf(Path(args.rf_validation))

    validation = [row for row in validation if row["claim_id"] in rf_val]
    test = [row for row in test if row["claim_id"] in rf]
    val_cal = calibrated(validation, bundle)
    val_anchor = np.asarray([normalize([float(v) for v in row["belief"]]) for row in validation])
    val_rf = np.asarray([rf_val[row["claim_id"]] for row in validation])
    grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    best = None
    for source_name, source in (("anchor", val_anchor), ("calibrated", val_cal)):
        for weight in grid:
            fused = weight * source + (1.0 - weight) * val_rf
            score = evaluate(validation, fused)
            if best is None or score["decision_loss"] < best["decision_loss"]:
                best = {"source": source_name, "weight": weight, "decision_loss": score["decision_loss"]}
    test_cal = calibrated(test, bundle)
    test_anchor = np.asarray([normalize([float(v) for v in row["belief"]]) for row in test])
    test_rf = np.asarray([rf[row["claim_id"]] for row in test])
    source = test_cal if best["source"] == "calibrated" else test_anchor
    fused = best["weight"] * source + (1.0 - best["weight"]) * test_rf
    result = {
        "validation_selection": best,
        "test_calibrated_anchor": evaluate(test, test_cal),
        "test_anchor": evaluate(test, test_anchor),
        "test_rf": evaluate(test, test_rf),
        "test_fused": evaluate(test, fused),
        "fusion_weight": best["weight"],
        "fusion_source": best["source"],
        "n_validation": len(validation),
        "n_test": len(test),
    }
    result["paired_fused_vs_rf"] = boot(result["test_fused"]["per_claim_loss"], result["test_rf"]["per_claim_loss"])
    result["paired_fused_vs_anchor"] = boot(result["test_fused"]["per_claim_loss"], result["test_anchor"]["per_claim_loss"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "test_fused"}, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
