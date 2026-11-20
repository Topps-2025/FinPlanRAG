"""Train a portable calibration bundle for the AnchorEvolve CLI.

The CLI needs a historical state, not a hand-set prior.  This script learns
four objects from revealed validation claims and writes them to one JSON
bundle:

* a label prior from revealed outcomes;
* Beta reliability per anchor type (the "core evolution" state);
* a small softmax calibrator over anchor-posterior features;
* a temperature and a fusion weight selected on a held-out calibration split.

The test rows are never used for fitting or selection.  The bundle stores
feature means/stds and coefficients, so the CLI needs only NumPy-free Python
to apply it.

Usage:
    python scripts/train_anchor_calibration.py \
        --beliefs paper/results/evolution_v1/temporal_beliefs_new.jsonl \
        --anchors paper/results/evolution_v1/anchors_validation.jsonl \
        --out paper/results/anchor_cli/calibration/bundle.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from anchor_engine import (  # noqa: E402
    answer_loss,
    full_action_readout,
    set_loss,
)

LABELS = ["below", "within", "above"]
FEATURE_NAMES = [
    "belief_below",
    "belief_within",
    "belief_above",
    "score_below",
    "score_within",
    "score_above",
    "entropy",
    "margin",
    "n_anchors_scaled",
    "n_hard_scaled",
    "mean_reliability",
    "max_reliability",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 1e-12:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


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


def train_softmax(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 0.02,
    lr: float = 0.15,
    iterations: int = 1200,
) -> np.ndarray:
    n, d = x.shape
    xb = np.concatenate([x, np.ones((n, 1))], axis=1)
    counts = np.bincount(y, minlength=len(LABELS)).astype(float)
    weights = np.where(counts > 0, n / (len(LABELS) * np.maximum(counts, 1.0)), 0.0)
    sample_weight = weights[y]
    y_onehot = np.eye(len(LABELS))[y]
    w = np.zeros((d + 1, len(LABELS)))
    for _ in range(iterations):
        probs = softmax(xb @ w)
        grad = xb.T @ ((probs - y_onehot) * sample_weight[:, None])
        grad /= max(sample_weight.sum(), 1e-9)
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return w


def predict_logits(x: np.ndarray, w: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    xb = np.concatenate([(x - mean) / std, np.ones((len(x), 1))], axis=1)
    return xb @ w


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best_t, best_nll = 1.0, float("inf")
    for temperature in np.arange(0.5, 3.01, 0.05):
        probs = softmax(logits / temperature)
        nll = -np.mean(np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12)))
        if nll < best_nll:
            best_t, best_nll = float(temperature), float(nll)
    return best_t, best_nll


def decision_loss(belief: list[float], truth: str, loss: dict[str, float]) -> float:
    readout = full_action_readout(belief, LABELS, loss)
    action = readout["action"]
    if action == "answer":
        return answer_loss(readout["point"], truth, LABELS, loss)
    if action == "adjudicate":
        return set_loss(readout["set"], truth, loss)
    return float(loss.get("abstain") or 5.0)


def evaluate(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    loss: dict[str, float],
) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    y = np.array([LABELS.index(row["truth"]) for row in rows])
    pred = probabilities.argmax(axis=1)
    accuracy = float(np.mean(pred == y))
    recalls = []
    for index in range(len(LABELS)):
        mask = y == index
        if mask.any():
            recalls.append(float(np.mean(pred[mask] == index)))
    balanced = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = 0.0
    for index in range(len(LABELS)):
        tp = np.sum((pred == index) & (y == index))
        fp = np.sum((pred == index) & (y != index))
        fn = np.sum((pred != index) & (y == index))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        macro_f1 += 2 * precision * recall / max(precision + recall, 1e-9)
    macro_f1 /= len(LABELS)
    nll = float(-np.mean(np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-12))))
    losses = [
        decision_loss(probabilities[i].tolist(), row["truth"], loss)
        for i, row in enumerate(rows)
    ]
    return {
        "n": len(rows),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": float(macro_f1),
        "nll": nll,
        "decision_loss": float(np.mean(losses)),
    }


def type_reliability(
    anchor_rows: list[dict[str, Any]],
    mapping: dict[str, str],
    *,
    min_observations: int = 5,
) -> dict[str, dict[str, float | int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "n": 0})
    for row in anchor_rows:
        if row.get("split") != "validation":
            continue
        truth = str(row.get("truth") or "")
        if truth not in LABELS:
            continue
        for anchor in row.get("anchors") or []:
            implied = str(anchor.get("implied_bucket") or "")
            if implied not in LABELS:
                continue
            raw_type = str(anchor.get("type") or "other")
            canonical = mapping.get(raw_type, "other")
            counts[canonical]["n"] += 1
            counts[canonical]["correct"] += int(implied == truth)
    result: dict[str, dict[str, float | int]] = {}
    for anchor_type, value in counts.items():
        n = int(value["n"])
        correct = int(value["correct"])
        if n < min_observations:
            reliability = 0.5
        else:
            reliability = (correct + 1.0) / (n + 2.0)
        result[anchor_type] = {
            "n": n,
            "correct": correct,
            "reliability": round(float(reliability), 6),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beliefs", default="paper/results/evolution_v1/temporal_beliefs_new.jsonl")
    parser.add_argument("--anchors", default="paper/results/evolution_v1/anchors_validation.jsonl")
    parser.add_argument("--config", default="configs/anchor_cli.yaml")
    parser.add_argument("--out", default="paper/results/anchor_cli/calibration/bundle.json")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8")) or {}
    loss = config.get("loss") or {}
    cal_cfg = config.get("calibration") or {}
    mapping = cal_cfg.get("type_mapping") or {}

    rows = load_jsonl(REPO_ROOT / args.beliefs)
    validation = [row for row in rows if row.get("split") == "validation" and row.get("truth") in LABELS]
    test = [row for row in rows if row.get("split") == "test" and row.get("truth") in LABELS]
    if len(validation) < 20:
        raise SystemExit(f"not enough validation rows: {len(validation)}")

    train_rows, calibration_rows = [], []
    for row in validation:
        digest = hashlib.sha1(
            f"{args.seed}:{row.get('claim_id')}".encode("utf-8")
        ).hexdigest()
        bucket = int(digest[:8], 16) % 1000
        if bucket < int(args.calibration_fraction * 1000):
            calibration_rows.append(row)
        else:
            train_rows.append(row)
    if not calibration_rows:
        calibration_rows = train_rows[-max(10, len(train_rows) // 5):]

    x_train = np.array([feature_vector(row) for row in train_rows], dtype=float)
    y_train = np.array([LABELS.index(row["truth"]) for row in train_rows], dtype=int)
    x_cal = np.array([feature_vector(row) for row in calibration_rows], dtype=float)
    y_cal = np.array([LABELS.index(row["truth"]) for row in calibration_rows], dtype=int)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-9] = 1.0
    x_train_std = (x_train - mean) / std
    w = train_softmax(x_train_std, y_train)

    logits_cal = predict_logits(x_cal, w, mean, std)
    temperature, _ = fit_temperature(logits_cal, y_cal)
    probs_cal = softmax(logits_cal / temperature)
    anchor_cal = np.array(
        [normalize([float(value) for value in row.get("belief") or [0.0] * 3]) for row in calibration_rows]
    )

    best_w, best_loss = 0.0, float("inf")
    for fusion_weight in np.arange(0.0, 1.001, 0.05):
        fused = fusion_weight * anchor_cal + (1.0 - fusion_weight) * probs_cal
        value = float(
            np.mean([
                decision_loss(fused[i].tolist(), row["truth"], loss)
                for i, row in enumerate(calibration_rows)
            ])
        )
        if value < best_loss:
            best_w, best_loss = float(fusion_weight), value
    anchor_cal_loss = float(
        np.mean([
            decision_loss(anchor_cal[i].tolist(), row["truth"], loss)
            for i, row in enumerate(calibration_rows)
        ])
    )
    calibrator_enabled = best_loss <= anchor_cal_loss + 1e-9

    metrics: dict[str, Any] = {
        "train": evaluate(train_rows, softmax(predict_logits(x_train, w, mean, std) / temperature), loss),
        "calibration_anchor": evaluate(calibration_rows, anchor_cal, loss),
        "calibration_calibrated": evaluate(calibration_rows, probs_cal, loss),
        "calibration_fused": evaluate(
            calibration_rows,
            best_w * anchor_cal + (1.0 - best_w) * probs_cal,
            loss,
        ),
    }
    if test:
        x_test = np.array([feature_vector(row) for row in test], dtype=float)
        probs_test = softmax(predict_logits(x_test, w, mean, std) / temperature)
        anchor_test = np.array(
            [normalize([float(value) for value in row.get("belief") or [0.0] * 3]) for row in test]
        )
        metrics["test_anchor"] = evaluate(test, anchor_test, loss)
        metrics["test_calibrated"] = evaluate(test, probs_test, loss)
        metrics["test_fused"] = evaluate(test, best_w * anchor_test + (1.0 - best_w) * probs_test, loss)

    prior_counts = Counter(str(row.get("truth")) for row in validation)
    total = sum(prior_counts.values())
    prior = {
        label: round((prior_counts.get(label, 0) + 1.0) / (total + len(LABELS)), 6)
        for label in LABELS
    }
    reliabilities = type_reliability(
        load_jsonl(REPO_ROOT / args.anchors),
        mapping,
        min_observations=int(cal_cfg.get("min_type_observations") or 5),
    )
    bundle = {
        "version": 1,
        "trained_on": str(REPO_ROOT / args.beliefs),
        "n_train": len(train_rows),
        "n_calibration": len(calibration_rows),
        "n_test": len(test),
        "feature_names": FEATURE_NAMES,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "coef": [[float(value) for value in row] for row in w],
        "temperature": float(temperature),
        "fusion_weight": float(best_w),
        "calibrator_enabled": bool(calibrator_enabled),
        "prior": prior,
        "type_reliability": reliabilities,
        "type_mapping": mapping,
        "applies_to_metric_kinds": list(
            cal_cfg.get("applies_to_metric_kinds") or ["level", "growth"]
        ),
        "metrics": metrics,
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "calibrator_enabled": calibrator_enabled,
                      "fusion_weight": best_w, "temperature": temperature,
                      "metrics": metrics}, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
