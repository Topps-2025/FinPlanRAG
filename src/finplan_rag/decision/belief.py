"""Belief updates and one-step decision values for the v2 pilot.

The pilot is deliberately a small, auditable decision problem.  The belief
model is not presented as a strong financial forecaster; its role is to make
the decision interface executable so the planner can be compared with fixed
policies under the same evidence, costs, and loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from .loss import LABELS, LossConfig, expected_action_loss
from .schema import DecisionCase

POSITIONS = LABELS
SUBSETS: tuple[tuple[str, ...], ...] = tuple(
    subset for size in range(1, len(LABELS) + 1) for subset in combinations(LABELS, size)
)
ACTION_PRIORITY = {"answer": 0, "adjudicate": 1, "abstain": 2}


@dataclass(frozen=True)
class StopAction:
    action: str
    expected_loss: float
    prediction: str | None = None
    credible_set: tuple[str, ...] = ()


def normalize(distribution: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, distribution.get(label, 0.0)) for label in LABELS)
    if total <= 0:
        return {label: 1.0 / len(LABELS) for label in LABELS}
    return {label: max(0.0, distribution.get(label, 0.0)) / total for label in LABELS}


def prior_from_cases(cases: Iterable[DecisionCase], smoothing: float = 1.0) -> dict[str, float]:
    counts = {label: smoothing for label in LABELS}
    n = 0
    for case in cases:
        if case.gold_hypothesis in counts:
            counts[case.gold_hypothesis] += 1.0
            n += 1
    return normalize(counts)


def posterior(prior: Mapping[str, float], likelihood: Mapping[str, float]) -> dict[str, float]:
    return normalize({label: prior.get(label, 0.0) * likelihood.get(label, 0.0) for label in LABELS})


def evidence_by_kind(case: DecisionCase, kind: str) -> list:
    return [item for item in case.evidence if item.kind == kind]


def current_q_position(case: DecisionCase) -> str | None:
    """Position of the current-quarter anchor relative to the guided range."""

    guidance = next((item for item in case.evidence if item.kind == "guidance"), None)
    current = next((item for item in case.evidence if item.kind == "current_q"), None)
    if guidance is None or current is None:
        return None
    low = guidance.payload.get("low")
    high = guidance.payload.get("high")
    value = current.payload.get("value")
    if low is None or high is None or value is None:
        return None
    if value < low:
        return "below"
    if value > high:
        return "above"
    return "within"


def position_distribution_from_cases(cases: Iterable[DecisionCase]) -> dict[str, float]:
    counts = {label: 1.0 for label in POSITIONS}
    for case in cases:
        position = current_q_position(case)
        if position in counts:
            counts[position] += 1.0
    return normalize(counts)


def current_q_likelihood_from_cases(
    cases: Sequence[DecisionCase],
    prior: Mapping[str, float],
    smoothing: float = 1.0,
    blend: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Empirical P(outcome | current-quarter position), smoothed and blended.

    The signal is weak in the v4 guidance data.  Blending with the prior keeps
    the likelihood from overfitting small cells while preserving the observed
    directional information.
    """

    counts = {position: {label: smoothing for label in LABELS} for position in POSITIONS}
    for case in cases:
        position = current_q_position(case)
        if position in counts and case.gold_hypothesis in LABELS:
            counts[position][case.gold_hypothesis] += 1.0
    table: dict[str, dict[str, float]] = {}
    for position in POSITIONS:
        empirical = normalize(counts[position])
        table[position] = normalize({
            label: blend * empirical[label] + (1.0 - blend) * prior.get(label, 0.0)
            for label in LABELS
        })
    return table


def uniform_likelihood() -> dict[str, float]:
    return {label: 1.0 / len(LABELS) for label in LABELS}


def confusion_likelihood(observed: str, precision: float) -> dict[str, float]:
    """P(observed | truth) for a noisy future signal."""

    if observed not in LABELS:
        return uniform_likelihood()
    precision = min(1.0, max(0.0, precision))
    other = (1.0 - precision) / (len(LABELS) - 1) if len(LABELS) > 1 else 0.0
    return {truth: precision if truth == observed else other for truth in LABELS}


def best_stop_action(belief: Mapping[str, float], config: LossConfig) -> StopAction:
    candidates: list[StopAction] = []
    for label in LABELS:
        loss = expected_action_loss("answer", belief, config, prediction=label)
        candidates.append(StopAction("answer", loss, prediction=label))
    for subset in SUBSETS:
        mass = sum(belief.get(label, 0.0) for label in subset)
        if mass < 1.0 - config.alpha - 1e-12:
            continue
        loss = expected_action_loss("adjudicate", belief, config, credible_set=subset)
        candidates.append(StopAction("adjudicate", loss, credible_set=subset))
    candidates.append(StopAction("abstain", config.abstain))
    candidates.sort(
        key=lambda item: (
            round(item.expected_loss, 12),
            ACTION_PRIORITY[item.action],
            len(item.credible_set),
            item.prediction or "",
        )
    )
    return candidates[0]


def wait_expected_loss(
    belief: Mapping[str, float],
    config: LossConfig,
    precision: float,
) -> float:
    """Expected decision loss after observing a noisy future signal.

    The planner observes a bucket-level signal whose confusion matrix is
    ``precision`` on the diagonal and uniform elsewhere.  The expected loss is
    the average of the best stop loss over the possible observations.
    """

    total = 0.0
    for observed in LABELS:
        likelihood = confusion_likelihood(observed, precision)
        joint = {truth: belief.get(truth, 0.0) * likelihood[truth] for truth in LABELS}
        probability = sum(joint.values())
        if probability <= 0:
            continue
        conditional = normalize(joint)
        total += probability * best_stop_action(conditional, config).expected_loss
    return total


def plan_expected_loss(
    belief: Mapping[str, float],
    config: LossConfig,
    *,
    actual_cost: float,
    delay_cost: float,
    wait_precision: float,
    allow_wait: bool,
) -> tuple[float, str]:
    """Best of stopping now and waiting for the future signal."""

    stop = best_stop_action(belief, config)
    if not allow_wait:
        return stop.expected_loss, "stop"
    wait = actual_cost + delay_cost + wait_expected_loss(belief, config, wait_precision)
    if wait < stop.expected_loss - 1e-12:
        return wait, "wait"
    return stop.expected_loss, "stop"
