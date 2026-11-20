"""Decision loss and aggregation metrics for the v2 pilot.

The loss is intentionally asymmetric.  In a financial decision an overclaim is
usually costlier than an underclaim, and an uninformative adjudication is not
free.  The parameters are exposed so the pilot can report a sensitivity grid
instead of a single number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .schema import DecisionCase, DecisionOutput

LABELS = ("below", "within", "above")


@dataclass(frozen=True)
class LossConfig:
    wrong: float = 10.0
    over: float = 12.0
    under: float = 8.0
    miss: float = 8.0
    width: float = 2.0
    abstain: float = 5.0
    alpha: float = 0.1

    def as_dict(self) -> dict[str, float]:
        return {
            "wrong": self.wrong,
            "over": self.over,
            "under": self.under,
            "miss": self.miss,
            "width": self.width,
            "abstain": self.abstain,
            "alpha": self.alpha,
        }


def answer_loss(prediction: str, truth: str, config: LossConfig) -> float:
    if prediction == truth:
        return 0.0
    if prediction == "above" and truth == "below":
        return config.over
    if prediction == "below" and truth == "above":
        return config.under
    return config.wrong


def adjudicate_loss(credible_set: Sequence[str], truth: str, config: LossConfig) -> float:
    base = 0.0 if truth in credible_set else config.miss
    return base + config.width * max(0, len(credible_set) - 1)


def action_loss(
    action: str,
    truth: str,
    config: LossConfig,
    *,
    prediction: str | None = None,
    credible_set: Sequence[str] = (),
) -> float:
    if action == "answer":
        if prediction is None:
            raise ValueError("answer action requires a prediction")
        return answer_loss(prediction, truth, config)
    if action == "adjudicate":
        return adjudicate_loss(credible_set, truth, config)
    if action == "abstain":
        return config.abstain
    raise ValueError(f"unknown action: {action}")


def expected_action_loss(
    action: str,
    belief: Mapping[str, float],
    config: LossConfig,
    *,
    prediction: str | None = None,
    credible_set: Sequence[str] = (),
) -> float:
    return sum(
        belief.get(label, 0.0) * action_loss(
            action,
            label,
            config,
            prediction=prediction,
            credible_set=credible_set,
        )
        for label in LABELS
    )


def credible_set(belief: Mapping[str, float], alpha: float) -> tuple[str, ...]:
    """Smallest label set whose posterior mass reaches 1 - alpha."""

    ordered = sorted(LABELS, key=lambda label: (-belief.get(label, 0.0), LABELS.index(label)))
    if not ordered:
        return ()
    target = max(0.0, 1.0 - alpha)
    chosen: list[str] = []
    mass = 0.0
    for label in ordered:
        chosen.append(label)
        mass += belief.get(label, 0.0)
        if mass >= target - 1e-12:
            break
    return tuple(chosen)


def aggregate(
    outputs: Sequence[DecisionOutput],
    cases: Sequence[DecisionCase],
    config: LossConfig,
) -> dict[str, float | int | None]:
    if not outputs:
        return {"n": 0}
    case_by_id = {case.case_id: case for case in cases}
    by_action = {"answer": 0, "adjudicate": 0, "abstain": 0}
    correct_answer = 0
    covered = 0
    leaked = 0
    for output in outputs:
        case = case_by_id[output.case_id]
        by_action[output.action] = by_action.get(output.action, 0) + 1
        if output.action == "answer" and output.prediction == case.gold_hypothesis:
            correct_answer += 1
        if output.action == "answer":
            covered += int(output.prediction == case.gold_hypothesis)
        elif output.action == "adjudicate":
            covered += int(case.gold_hypothesis in output.credible_set)
        leaked += int(output.leaked)
    n = len(outputs)
    answer_outputs = [o for o in outputs if o.action == "answer"]
    adjudicate_outputs = [o for o in outputs if o.action == "adjudicate"]
    return {
        "n": n,
        "decision_loss": sum(o.decision_loss for o in outputs) / n,
        "total_loss": sum(o.decision_loss + o.total_cost for o in outputs) / n,
        "evidence_cost": sum(o.evidence_cost for o in outputs) / n,
        "delay_cost": sum(o.delay_cost for o in outputs) / n,
        "total_cost": sum(o.total_cost for o in outputs) / n,
        "coverage": covered / n,
        "answer_rate": by_action["answer"] / n,
        "adjudicate_rate": by_action["adjudicate"] / n,
        "abstain_rate": by_action["abstain"] / n,
        "answer_accuracy": (correct_answer / len(answer_outputs)) if answer_outputs else None,
        "adjudication_coverage": (
            sum(case_by_id[o.case_id].gold_hypothesis in o.credible_set for o in adjudicate_outputs)
            / len(adjudicate_outputs)
            if adjudicate_outputs
            else None
        ),
        "wait_rate": sum(1 for o in outputs if o.decision_days > 0.0) / n,
        "preliminary_rate": sum(
            1 for o in outputs if any("::preliminary" in item for item in o.used_evidence)
        ) / n,
        "mean_decision_days": sum(o.decision_days for o in outputs) / n,
        "leakage_rate": leaked / n,
        "ece": expected_calibration_error(outputs, case_by_id),
    }


def expected_calibration_error(
    outputs: Sequence[DecisionOutput],
    case_by_id: Mapping[str, DecisionCase],
    bins: int = 10,
) -> float | None:
    """ECE over point answers; adjudications and abstentions are excluded."""

    answer_outputs = [o for o in outputs if o.action == "answer" and o.prediction]
    if not answer_outputs:
        return None
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for output in answer_outputs:
        confidence = float(output.belief.get(output.prediction or "", 0.0))
        confidence = min(1.0, max(0.0, confidence))
        index = min(bins - 1, int(confidence * bins))
        correct = int(output.prediction == case_by_id[output.case_id].gold_hypothesis)
        buckets[index].append((confidence, correct))
    total = len(answer_outputs)
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_confidence = sum(c for c, _ in bucket) / len(bucket)
        mean_accuracy = sum(y for _, y in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(mean_confidence - mean_accuracy)
    return ece
