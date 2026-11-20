"""Fixed policies and a one-step decision planner for the v2 pilot.

The planner is intentionally simple and auditable.  It chooses among three
evidence plans (none, guidance + current-quarter anchor), then decides whether
to stop or wait for the later outcome filing.  The decision is based on the
same loss the evaluator uses, not on the gold outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Mapping, Sequence

from .belief import (
    StopAction,
    best_stop_action,
    confusion_likelihood,
    current_q_position,
    normalize,
    posterior,
    uniform_likelihood,
    wait_expected_loss,
)
from .loss import LABELS, LossConfig, action_loss, credible_set
from .pilot_data import observed_actual_bucket
from .schema import DecisionCase, DecisionOutput, EvidenceItem


@dataclass(frozen=True)
class DecisionConfig:
    loss: LossConfig = field(default_factory=LossConfig)
    gamma: float = 0.01
    actual_cost: float = 2.0
    guidance_cost: float = 0.5
    current_q_cost: float = 0.5
    expected_delay_days: float = 100.0
    wait_precision: float = 1.0
    allow_wait: bool = True
    stop_margin: float = 1.0
    preliminary_margin: float = 0.0
    signal_strength: float = 0.0


@dataclass(frozen=True)
class PolicyContext:
    prior: Mapping[str, float]
    current_q_likelihood: Mapping[str, Mapping[str, float]]
    position_distribution: Mapping[str, float]
    config: DecisionConfig


def _evidence(case: DecisionCase, kind: str) -> EvidenceItem | None:
    return next((item for item in case.evidence if item.kind == kind), None)


def _current_q_belief(case: DecisionCase, context: PolicyContext) -> dict[str, float]:
    position = current_q_position(case)
    current = _evidence(case, "current_q")
    if position is None or current is None:
        return dict(context.prior)
    likelihood = context.current_q_likelihood.get(position)
    if likelihood is None:
        likelihood = uniform_likelihood()
    reliability = min(1.0, max(0.0, float(current.reliability)))
    strength = max(0.0, float(context.config.signal_strength))
    temperature = max(0.2, 1.0 + strength * (reliability - 0.5) * 2.0)
    sharpened = normalize({
        label: max(1e-12, likelihood.get(label, 0.0)) ** temperature for label in LABELS
    })
    uniform = uniform_likelihood()
    tempered = normalize({
        label: reliability * sharpened.get(label, 0.0) + (1.0 - reliability) * uniform[label]
        for label in LABELS
    })
    return posterior(context.prior, tempered)


def _belief_after_wait(
    belief: Mapping[str, float],
    case: DecisionCase,
    context: PolicyContext,
) -> tuple[dict[str, float], str]:
    observed = observed_actual_bucket(case, context.config.wait_precision)
    likelihood = confusion_likelihood(observed, context.config.wait_precision)
    return posterior(belief, likelihood), observed


def _preliminary_evidence(case: DecisionCase) -> EvidenceItem | None:
    return _evidence(case, "preliminary")


def _preliminary_belief(
    belief: Mapping[str, float],
    case: DecisionCase,
) -> tuple[dict[str, float], str, float] | None:
    preliminary = _preliminary_evidence(case)
    if preliminary is None:
        return None
    observed = str(preliminary.payload.get("observed_bucket", ""))
    precision = float(preliminary.payload.get("precision", 0.7))
    likelihood = confusion_likelihood(observed, precision)
    return posterior(belief, likelihood), observed, precision


def _preliminary_expected_loss(
    belief: Mapping[str, float],
    case: DecisionCase,
    context: PolicyContext,
    expected_precision: float = 0.7,
) -> float | None:
    preliminary = _preliminary_evidence(case)
    if preliminary is None:
        return None
    cost = float(preliminary.cost)
    total = 0.0
    for observed in LABELS:
        likelihood = confusion_likelihood(observed, expected_precision)
        joint = {truth: belief.get(truth, 0.0) * likelihood[truth] for truth in LABELS}
        probability = sum(joint.values())
        if probability <= 0:
            continue
        conditional = normalize(joint)
        stop = best_stop_action(conditional, context.config.loss)
        wait = _expected_wait_cost(case, context) + wait_expected_loss(
            conditional, context.config.loss, context.config.wait_precision
        )
        total += probability * min(
            stop.expected_loss + context.config.preliminary_margin,
            wait,
        )
    return cost + total


def _true_delay_days(case: DecisionCase) -> float:
    return float(case.metadata.get("delay_days", context_default_delay()))


def context_default_delay() -> float:
    return 100.0


def _expected_wait_cost(case: DecisionCase, context: PolicyContext) -> float:
    config = context.config
    expected_delay = float(case.metadata.get("expected_delay_days", config.expected_delay_days))
    return config.actual_cost + config.gamma * expected_delay


def _should_wait(
    belief: Mapping[str, float],
    case: DecisionCase,
    context: PolicyContext,
    *,
    allow_wait: bool,
    margin: float | None = None,
) -> tuple[bool, float, StopAction]:
    config = context.config
    stop = best_stop_action(belief, config.loss)
    if not allow_wait:
        return False, stop.expected_loss, stop
    wait_loss = _expected_wait_cost(case, context) + wait_expected_loss(
        belief, config.loss, config.wait_precision
    )
    # A conservative margin prevents the planner from stopping on a weak
    # signal whose apparent advantage is smaller than the model's calibration
    # error.  The margin is a design choice, not a learned parameter.
    effective_margin = config.stop_margin if margin is None else margin
    if wait_loss < stop.expected_loss + effective_margin - 1e-12:
        return True, wait_loss, stop
    return False, stop.expected_loss, stop


def _make_output(
    case: DecisionCase,
    policy: str,
    belief: Mapping[str, float],
    stop: StopAction,
    loss: LossConfig,
    *,
    used: Sequence[EvidenceItem],
    evidence_cost: float,
    delay_cost: float,
    decision_days: float,
    note: str = "",
) -> DecisionOutput:
    truth = case.gold_hypothesis
    if stop.action == "answer":
        decision = action_loss("answer", truth, loss, prediction=stop.prediction)
        prediction = stop.prediction
        credible: tuple[str, ...] = ()
    elif stop.action == "adjudicate":
        decision = action_loss("adjudicate", truth, loss, credible_set=stop.credible_set)
        prediction = None
        credible = stop.credible_set
    else:
        decision = action_loss("abstain", truth, loss)
        prediction = None
        credible = ()
    return DecisionOutput(
        case_id=case.case_id,
        policy=policy,
        action=stop.action,
        prediction=prediction,
        credible_set=credible,
        belief=dict(belief),
        decision_loss=decision,
        evidence_cost=evidence_cost,
        delay_cost=delay_cost,
        total_cost=evidence_cost + delay_cost,
        decision_days=decision_days,
        used_evidence=tuple(item.evidence_id for item in used),
        leaked=_leaked(case, used, decision_days),
        note=note,
    )


def _leaked(case: DecisionCase, used: Sequence[EvidenceItem], decision_days: float) -> bool:
    decision_time = case.cutoff + timedelta(days=decision_days)
    return any(item.available_at > decision_time for item in used)


def _stop_output_config(
    case: DecisionCase,
    policy: str,
    belief: Mapping[str, float],
    context: PolicyContext,
    *,
    used: Sequence[EvidenceItem],
    evidence_cost: float,
    decision_days: float = 0.0,
    note: str = "",
) -> DecisionOutput:
    stop = best_stop_action(belief, context.config.loss)
    return _make_output(
        case,
        policy,
        belief,
        stop,
        context.config.loss,
        used=used,
        evidence_cost=evidence_cost,
        delay_cost=context.config.gamma * decision_days,
        decision_days=decision_days,
        note=note,
    )


def _wait_output(
    case: DecisionCase,
    policy: str,
    belief: Mapping[str, float],
    context: PolicyContext,
    *,
    used: Sequence[EvidenceItem],
    evidence_cost: float,
    note: str = "",
) -> DecisionOutput:
    config = context.config
    delay_days = _true_delay_days(case)
    after_wait, observed = _belief_after_wait(belief, case, context)
    stop = best_stop_action(after_wait, config.loss)
    output = _make_output(
        case,
        policy,
        after_wait,
        stop,
        config.loss,
        used=used,
        evidence_cost=evidence_cost,
        delay_cost=config.gamma * delay_days,
        decision_days=delay_days,
        note=note,
    )
    return DecisionOutput(**{**output.__dict__, "note": f"{note} observed={observed}"})


class Policy:
    name = "policy"

    def run(self, case: DecisionCase, context: PolicyContext) -> DecisionOutput:
        raise NotImplementedError

    def _plan_value(
        self,
        case: DecisionCase,
        context: PolicyContext,
        belief: Mapping[str, float],
        evidence_cost: float,
        *,
        allow_wait: bool,
    ) -> float:
        config = context.config
        stop = best_stop_action(belief, config.loss)
        value = stop.expected_loss + evidence_cost + config.stop_margin
        if allow_wait:
            expected_delay = float(case.metadata.get("expected_delay_days", config.expected_delay_days))
            wait = (
                evidence_cost
                + config.actual_cost
                + config.gamma * expected_delay
                + wait_expected_loss(belief, config.loss, config.wait_precision)
            )
            value = min(value, wait)
        return value

    def _choose_plan(
        self,
        case: DecisionCase,
        context: PolicyContext,
        *,
        allow_wait: bool,
        allow_evidence: bool = True,
    ) -> tuple[str, dict[str, float], list[EvidenceItem], float]:
        prior = dict(context.prior)
        options: list[tuple[str, dict[str, float], list[EvidenceItem], float]] = [
            ("none", prior, [], 0.0)
        ]
        guidance = _evidence(case, "guidance")
        current = _evidence(case, "current_q")
        if allow_evidence and guidance is not None and current is not None:
            belief = _current_q_belief(case, context)
            cost = context.config.guidance_cost + context.config.current_q_cost
            options.append(("current", belief, [guidance, current], cost))
        scored = [
            (name, belief, used, cost, self._plan_value(case, context, belief, cost, allow_wait=allow_wait))
            for name, belief, used, cost in options
        ]
        scored.sort(key=lambda item: (round(item[4], 12), 0 if item[0] == "none" else 1))
        name, belief, used, cost, _ = scored[0]
        return name, belief, used, cost

    def _execute(
        self,
        case: DecisionCase,
        context: PolicyContext,
        *,
        policy: str,
        belief: Mapping[str, float],
        used: Sequence[EvidenceItem],
        evidence_cost: float,
        allow_wait: bool,
    ) -> DecisionOutput:
        should_wait, _, _ = _should_wait(belief, case, context, allow_wait=allow_wait)
        if should_wait:
            return _wait_output(
                case,
                policy,
                belief,
                context,
                used=list(used) + [item for item in case.evidence if item.kind == "actual"],
                evidence_cost=evidence_cost + context.config.actual_cost,
            )
        return _stop_output_config(
            case,
            policy,
            belief,
            context,
            used=used,
            evidence_cost=evidence_cost,
        )


class AlwaysAnswer(Policy):
    name = "always_answer"

    def run(self, case: DecisionCase, context: PolicyContext) -> DecisionOutput:
        belief = dict(context.prior)
        prediction = max(LABELS, key=lambda label: (belief.get(label, 0.0), -LABELS.index(label)))
        stop = StopAction("answer", 0.0, prediction=prediction)
        return _make_output(
            case, self.name, belief, stop, context.config.loss,
            used=[], evidence_cost=0.0, delay_cost=0.0, decision_days=0.0,
        )


class AlwaysAdjudicate(Policy):
    name = "always_adjudicate"

    def run(self, case: DecisionCase, context: PolicyContext) -> DecisionOutput:
        belief = dict(context.prior)
        chosen = credible_set(belief, context.config.loss.alpha)
        stop = StopAction("adjudicate", 0.0, credible_set=chosen)
        return _make_output(
            case, self.name, belief, stop, context.config.loss,
            used=[], evidence_cost=0.0, delay_cost=0.0, decision_days=0.0,
        )


class FixedTopK(Policy):
    name = "fixed_topk"

    def run(self, case: DecisionCase, context: PolicyContext) -> DecisionOutput:
        belief = _current_q_belief(case, context)
        used = [item for item in case.evidence if item.kind in {"guidance", "current_q"}]
        cost = context.config.guidance_cost + context.config.current_q_cost
        prediction = max(LABELS, key=lambda label: belief.get(label, 0.0))
        stop = StopAction("answer", 0.0, prediction=prediction)
        return _make_output(
            case, self.name, belief, stop, context.config.loss,
            used=used, evidence_cost=cost, delay_cost=0.0, decision_days=0.0,
        )


class FixedBudgetAdjudicate(Policy):
    name = "fixed_budget_adjudicate"

    def run(self, case: DecisionCase, context: PolicyContext) -> DecisionOutput:
        belief = _current_q_belief(case, context)
        used = [item for item in case.evidence if item.kind in {"guidance", "current_q"}]
        cost = context.config.guidance_cost + context.config.current_q_cost
        chosen = credible_set(belief, context.config.loss.alpha)
        stop = StopAction("adjudicate", 0.0, credible_set=chosen)
        return _make_output(
            case, self.name, belief, stop, context.config.loss,
            used=used, evidence_cost=cost, delay_cost=0.0, decision_days=0.0,
        )


class PreciseFirst(Policy):
    name = "precise_first"

    def run(self, case: DecisionCase, context: PolicyContext) -> DecisionOutput:
        belief = _current_q_belief(case, context)
        used = [item for item in case.evidence if item.kind in {"guidance", "current_q"}]
        cost = context.config.guidance_cost + context.config.current_q_cost
        top = max(LABELS, key=lambda label: belief.get(label, 0.0))
        if belief.get(top, 0.0) >= 0.60:
            stop = StopAction("answer", 0.0, prediction=top)
        else:
            stop = StopAction("adjudicate", 0.0, credible_set=credible_set(belief, context.config.loss.alpha))
        return _make_output(
            case, self.name, belief, stop, context.config.loss,
            used=used, evidence_cost=cost, delay_cost=0.0, decision_days=0.0,
        )


class PrehocRouter(Policy):
    name = "prehoc_router"

    def run(self, case: DecisionCase, context: PolicyContext) -> DecisionOutput:
        belief = _current_q_belief(case, context)
        used = [item for item in case.evidence if item.kind in {"guidance", "current_q"}]
        cost = context.config.guidance_cost + context.config.current_q_cost
        width = case.metadata.get("guidance_width_pct")
        top = max(LABELS, key=lambda label: belief.get(label, 0.0))
        narrow = width is not None and float(width) <= 10.0
        if narrow and belief.get(top, 0.0) >= 0.50:
            stop = StopAction("answer", 0.0, prediction=top)
        else:
            stop = StopAction("adjudicate", 0.0, credible_set=credible_set(belief, context.config.loss.alpha))
        return _make_output(
            case, self.name, belief, stop, context.config.loss,
            used=used, evidence_cost=cost, delay_cost=0.0, decision_days=0.0,
        )


class AlwaysWait(Policy):
    name = "always_wait"

    def run(self, case: DecisionCase, context: PolicyContext) -> DecisionOutput:
        belief = dict(context.prior)
        actual = [item for item in case.evidence if item.kind == "actual"]
        return _wait_output(case, self.name, belief, context, used=actual, evidence_cost=context.config.actual_cost)


class FixedRuleWait(Policy):
    name = "fixed_rule_wait"

    def run(self, case: DecisionCase, context: PolicyContext) -> DecisionOutput:
        belief = _current_q_belief(case, context)
        used = [item for item in case.evidence if item.kind in {"guidance", "current_q"}]
        cost = context.config.guidance_cost + context.config.current_q_cost
        top = max(LABELS, key=lambda label: belief.get(label, 0.0))
        if belief.get(top, 0.0) < 0.60:
            actual = [item for item in case.evidence if item.kind == "actual"]
            return _wait_output(
                case,
                self.name,
                belief,
                context,
                used=used + actual,
                evidence_cost=cost + context.config.actual_cost,
            )
        stop = StopAction("answer", 0.0, prediction=top)
        return _make_output(
            case, self.name, belief, stop, context.config.loss,
            used=used, evidence_cost=cost, delay_cost=0.0, decision_days=0.0,
        )


class Planner(Policy):
    name = "planner"

    def __init__(self, *, allow_wait: bool = True, allow_evidence: bool = True) -> None:
        self.allow_wait = allow_wait
        self.allow_evidence = allow_evidence

    def run(self, case: DecisionCase, context: PolicyContext) -> DecisionOutput:
        name, belief, used, cost = self._choose_plan(
            case, context, allow_wait=self.allow_wait, allow_evidence=self.allow_evidence
        )
        return self._execute(
            case,
            context,
            policy=self.name,
            belief=belief,
            used=used,
            evidence_cost=cost,
            allow_wait=self.allow_wait,
        )


class PlannerNoWait(Planner):
    name = "planner_no_wait"

    def __init__(self) -> None:
        super().__init__(allow_wait=False, allow_evidence=True)


class PlannerNoEvidence(Planner):
    name = "planner_no_evidence"

    def __init__(self) -> None:
        super().__init__(allow_wait=True, allow_evidence=False)


class PlannerPreliminary(Planner):
    """Planner that can acquire a controlled preliminary signal before waiting.

    The preliminary signal is available before the final outcome filing.  The
    policy decides whether to acquire it, observes the signal, and then decides
    whether to stop or wait for the final filing.  A pre-hoc router cannot
    observe the signal content before choosing its mode; this is the controlled
    test for P3 (mid-retrieval mode switching).
    """

    name = "planner_preliminary"

    def __init__(self) -> None:
        super().__init__(allow_wait=True, allow_evidence=True)

    def run(self, case: DecisionCase, context: PolicyContext) -> DecisionOutput:
        preliminary = _preliminary_evidence(case)
        if preliminary is None:
            return super().run(case, context)

        options: list[tuple[str, dict[str, float], list[EvidenceItem], float, float]] = [
            (
                "none",
                dict(context.prior),
                [],
                0.0,
                self._plan_value(case, context, dict(context.prior), 0.0, allow_wait=True),
            )
        ]
        guidance = _evidence(case, "guidance")
        current = _evidence(case, "current_q")
        if guidance is not None and current is not None:
            belief = _current_q_belief(case, context)
            cost = context.config.guidance_cost + context.config.current_q_cost
            options.append((
                "current",
                belief,
                [guidance, current],
                cost,
                self._plan_value(case, context, belief, cost, allow_wait=True),
            ))
        preliminary_value = _preliminary_expected_loss(context.prior, case, context)
        if preliminary_value is not None:
            options.append((
                "preliminary",
                dict(context.prior),
                [preliminary],
                float(preliminary.cost),
                preliminary_value,
            ))
        options.sort(key=lambda item: (round(item[4], 12), 0 if item[0] == "none" else 1))
        name, belief, used, cost, _ = options[0]

        if name == "preliminary":
            updated = _preliminary_belief(context.prior, case)
            if updated is None:  # pragma: no cover - guarded by preliminary
                return super().run(case, context)
            posterior_belief, observed, precision = updated
            decision_days = max(
                0.0,
                (preliminary.available_at - case.cutoff).total_seconds() / 86400.0,
            )
            should_wait, _, _ = _should_wait(
                posterior_belief,
                case,
                context,
                allow_wait=True,
                margin=context.config.preliminary_margin,
            )
            if should_wait:
                return _wait_output(
                    case,
                    self.name,
                    posterior_belief,
                    context,
                    used=[preliminary] + [item for item in case.evidence if item.kind == "actual"],
                    evidence_cost=float(preliminary.cost) + context.config.actual_cost,
                    note=f"observed={observed},precision={precision}",
                )
            return _stop_output_config(
                case,
                self.name,
                posterior_belief,
                context,
                used=[preliminary],
                evidence_cost=float(preliminary.cost),
                decision_days=decision_days,
                note=f"observed={observed},precision={precision}",
            )

        return self._execute(
            case,
            context,
            policy=self.name,
            belief=belief,
            used=used,
            evidence_cost=cost,
            allow_wait=True,
        )


class OracleRouter(Policy):
    name = "oracle_router"

    def run(self, case: DecisionCase, context: PolicyContext) -> DecisionOutput:
        belief = _current_q_belief(case, context)
        used = [item for item in case.evidence if item.kind in {"guidance", "current_q"}]
        cost = context.config.guidance_cost + context.config.current_q_cost
        return _stop_output_config(case, self.name, belief, context, used=used, evidence_cost=cost)


class Oracle(Policy):
    name = "oracle"

    def run(self, case: DecisionCase, context: PolicyContext) -> DecisionOutput:
        stop = StopAction("answer", 0.0, prediction=case.gold_hypothesis)
        return _make_output(
            case, self.name, dict(context.prior), stop, context.config.loss,
            used=[], evidence_cost=0.0, delay_cost=0.0, decision_days=0.0,
        )


def default_policies() -> list[Policy]:
    return [
        AlwaysAnswer(),
        AlwaysAdjudicate(),
        FixedTopK(),
        FixedBudgetAdjudicate(),
        PreciseFirst(),
        PrehocRouter(),
        AlwaysWait(),
        FixedRuleWait(),
        PlannerNoWait(),
        Planner(),
        PlannerNoEvidence(),
        PlannerPreliminary(),
        OracleRouter(),
        Oracle(),
    ]
