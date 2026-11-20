"""Standard metrics for the v2 decision pilot.

These metrics do not use the asymmetric decision-loss weights.  They are the
kind of metrics a reviewer can audit directly: top-1 accuracy, macro F1,
coverage, set size, ECE, Brier score, log loss, selective accuracy/AURC, and
accuracy under an evidence/delay cost budget.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from .schema import DecisionCase, DecisionOutput

LABELS = ("below", "within", "above")


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else float("nan")


def top1(belief: Mapping[str, float]) -> str:
    return max(LABELS, key=lambda label: (belief.get(label, 0.0), -LABELS.index(label)))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return mean(scores)


def coverage(outputs: Sequence[DecisionOutput], cases: Sequence[DecisionCase]) -> float:
    values = []
    for output, case in zip(outputs, cases):
        if output.action == "answer":
            values.append(1.0 if output.prediction == case.gold_hypothesis else 0.0)
        elif output.action == "adjudicate":
            values.append(1.0 if case.gold_hypothesis in output.credible_set else 0.0)
        else:
            values.append(0.0)
    return mean(values)


def mean_set_size(outputs: Sequence[DecisionOutput]) -> float:
    sizes = []
    for output in outputs:
        if output.action == "answer":
            sizes.append(1)
        elif output.action == "adjudicate":
            sizes.append(len(output.credible_set))
        else:
            sizes.append(0)
    return mean(sizes)


def ece(outputs: Sequence[DecisionOutput], cases: Sequence[DecisionCase], bins: int = 10) -> float:
    if not outputs:
        return float("nan")
    confidences = []
    correct = []
    for output, case in zip(outputs, cases):
        prediction = top1(output.belief)
        confidences.append(float(output.belief.get(prediction, 0.0)))
        correct.append(1.0 if prediction == case.gold_hypothesis else 0.0)
    probs = np.asarray(confidences)
    labels = np.asarray(correct)
    indices = np.clip((probs * bins).astype(int), 0, bins - 1)
    value = 0.0
    for index in range(bins):
        mask = indices == index
        if not mask.any():
            continue
        value += mask.sum() / probs.size * abs(labels[mask].mean() - probs[mask].mean())
    return float(value)


def brier_score(outputs: Sequence[DecisionOutput], cases: Sequence[DecisionCase]) -> float:
    values = []
    for output, case in zip(outputs, cases):
        values.append(sum(
            (float(output.belief.get(label, 0.0)) - (1.0 if label == case.gold_hypothesis else 0.0)) ** 2
            for label in LABELS
        ))
    return mean(values)


def log_loss(outputs: Sequence[DecisionOutput], cases: Sequence[DecisionCase], floor: float = 1e-12) -> float:
    values = []
    for output, case in zip(outputs, cases):
        probability = max(floor, float(output.belief.get(case.gold_hypothesis, 0.0)))
        values.append(-math.log(probability))
    return mean(values)


def selective_metrics(
    outputs: Sequence[DecisionOutput],
    cases: Sequence[DecisionCase],
    coverage_point: float = 0.5,
) -> dict[str, float]:
    if not outputs:
        return {"selective_accuracy": float("nan"), "aurc": float("nan")}
    confidences = []
    correct = []
    for output, case in zip(outputs, cases):
        prediction = top1(output.belief)
        confidences.append(float(output.belief.get(prediction, 0.0)))
        correct.append(1.0 if prediction == case.gold_hypothesis else 0.0)
    probs = np.asarray(confidences)
    correct_array = np.asarray(correct)
    order = np.argsort(-probs, kind="stable")
    sorted_correct = correct_array[order]
    n = sorted_correct.size
    risks = []
    for cov in np.linspace(0.1, 1.0, 10):
        k = max(1, int(math.ceil(cov * n)))
        risks.append(1.0 - float(sorted_correct[:k].mean()))
    k = max(1, int(math.ceil(coverage_point * n)))
    return {
        "selective_accuracy": float(sorted_correct[:k].mean()),
        "aurc": float(np.mean(risks)),
    }


def budgeted_accuracy(
    outputs: Sequence[DecisionOutput],
    cases: Sequence[DecisionCase],
    budget: float,
    fallback: str = "above",
) -> float:
    """Top-1 accuracy when a policy may spend at most ``budget``.

    A policy whose realized cost exceeds the budget cannot use its output; it
    falls back to a zero-cost majority answer.  This is a standard
    accuracy-under-cost-budget metric, not a hand-set loss.
    """

    values = []
    for output, case in zip(outputs, cases):
        if output.total_cost <= budget + 1e-12:
            prediction = top1(output.belief)
        else:
            prediction = fallback
        values.append(1.0 if prediction == case.gold_hypothesis else 0.0)
    return mean(values)


def evaluate(
    outputs: Sequence[DecisionOutput],
    cases: Sequence[DecisionCase],
    budgets: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
) -> dict[str, object]:
    truth = [case.gold_hypothesis for case in cases]
    predictions = [top1(output.belief) for output in outputs]
    return {
        "n": len(cases),
        "top1_accuracy": mean(1.0 if p == t else 0.0 for p, t in zip(predictions, truth)),
        "macro_f1": macro_f1(truth, predictions),
        "coverage": coverage(outputs, cases),
        "mean_set_size": mean_set_size(outputs),
        "ece": ece(outputs, cases),
        "brier": brier_score(outputs, cases),
        "log_loss": log_loss(outputs, cases),
        **selective_metrics(outputs, cases),
        "answer_rate": mean(1.0 if o.action == "answer" else 0.0 for o in outputs),
        "adjudicate_rate": mean(1.0 if o.action == "adjudicate" else 0.0 for o in outputs),
        "abstain_rate": mean(1.0 if o.action == "abstain" else 0.0 for o in outputs),
        "wait_rate": mean(1.0 if o.decision_days > 0.0 else 0.0 for o in outputs),
        "mean_total_cost": mean(o.total_cost for o in outputs),
        "mean_decision_days": mean(o.decision_days for o in outputs),
        "leakage_rate": mean(1.0 if o.leaked else 0.0 for o in outputs),
        "budgeted_accuracy": {
            f"{budget:.2f}": budgeted_accuracy(outputs, cases, budget) for budget in budgets
        },
    }


def aggregate_seeds(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    if array.size == 1:
        return {
            "mean": float(array[0]),
            "std": 0.0,
            "ci_low": float(array[0]),
            "ci_high": float(array[0]),
            "n": 1,
        }
    half = 1.96 * float(array.std(ddof=1)) / math.sqrt(array.size)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "ci_low": float(array.mean() - half),
        "ci_high": float(array.mean() + half),
        "n": int(array.size),
    }
