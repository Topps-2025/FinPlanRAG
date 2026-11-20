from datetime import datetime, timezone

from finplan_rag.decision.belief import best_stop_action
from finplan_rag.decision.loss import (
    LossConfig,
    adjudicate_loss,
    answer_loss,
    credible_set,
    expected_action_loss,
)
from finplan_rag.decision.metrics import budgeted_accuracy, evaluate
from finplan_rag.decision.pilot_data import preliminary_signal, v4_case
from finplan_rag.decision.policies import (
    DecisionConfig,
    Planner,
    PolicyContext,
)
from finplan_rag.decision.schema import DecisionCase, DecisionOutput, EvidenceItem, Hypothesis


def _case(gold: str = "above", *, delay_days: int = 30) -> DecisionCase:
    cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
    actual_at = datetime(2024, 1, 1 + delay_days, tzinfo=timezone.utc)
    evidence = (
        EvidenceItem(
            "guidance", cutoff, 1.0, "guidance", "cik", "8-K", 1.0,
            {"low": 100.0, "high": 110.0, "mid": 105.0, "width_pct": 9.5},
        ),
        EvidenceItem(
            "current", cutoff, 1.0, "current_q", "cik", "8-K", 0.9,
            {"value": 105.0, "reliability": "high"},
        ),
        EvidenceItem(
            "actual", actual_at, 2.0, "actual", "cik", "8-K", 1.0,
            {"bucket": gold, "actual_musd": 120.0, "noise_draw": 0.0},
        ),
    )
    return DecisionCase(
        case_id="case-1",
        task_type="mode_ambiguous",
        query="Will the company report above its guided range?",
        hypotheses=(Hypothesis("below", "below"), Hypothesis("within", "within"), Hypothesis("above", "above")),
        evidence=evidence,
        cutoff=cutoff,
        gold_hypothesis=gold,
        split="test",
        metadata={"delay_days": float(delay_days), "expected_delay_days": float(delay_days)},
    )


def _context(gamma: float, *, allow_wait: bool = True) -> PolicyContext:
    return PolicyContext(
        prior={"below": 0.2, "within": 0.3, "above": 0.5},
        current_q_likelihood={
            "below": {"below": 0.45, "within": 0.35, "above": 0.20},
            "within": {"below": 0.25, "within": 0.45, "above": 0.30},
            "above": {"below": 0.20, "within": 0.30, "above": 0.50},
        },
        position_distribution={"below": 0.2, "within": 0.6, "above": 0.2},
        config=DecisionConfig(
            loss=LossConfig(),
            gamma=gamma,
            actual_cost=1.0,
            expected_delay_days=30.0,
            wait_precision=1.0,
            allow_wait=allow_wait,
        ),
    )


def test_credible_set_and_loss_direction():
    assert credible_set({"below": 0.1, "within": 0.3, "above": 0.6}, 0.1) == ("above", "within")
    config = LossConfig()
    assert answer_loss("above", "below", config) == config.over
    assert answer_loss("below", "above", config) == config.under
    assert answer_loss("within", "above", config) == config.wrong
    assert adjudicate_loss(("above", "within"), "below", config) == config.miss + config.width


def test_best_stop_action_uses_asymmetric_loss():
    belief = {"below": 0.2, "within": 0.3, "above": 0.5}
    action = best_stop_action(belief, LossConfig())
    assert action.expected_loss <= expected_action_loss("answer", belief, LossConfig(), prediction="above")
    assert action.action in {"answer", "adjudicate", "abstain"}


def test_planner_waits_when_delay_is_free_and_does_not_leak():
    case = _case("above", delay_days=30)
    output = Planner().run(case, _context(gamma=0.0))
    assert output.action == "answer"
    assert output.prediction == "above"
    assert output.decision_loss == 0.0
    assert output.decision_days == 30.0
    assert output.leaked is False


def test_planner_stops_when_delay_is_expensive_and_does_not_leak():
    case = _case("above", delay_days=30)
    output = Planner().run(case, _context(gamma=1.0))
    assert output.decision_days == 0.0
    assert output.leaked is False
    assert output.action in {"answer", "adjudicate", "abstain"}


def test_preliminary_signal_is_seeded_and_deterministic():
    first = preliminary_signal("case-1", "above", seed=7)
    second = preliminary_signal("case-1", "above", seed=7)
    assert first == second
    assert first[1] in {0.5, 0.9}
    assert first[0] in {"below", "within", "above"}


def test_budgeted_accuracy_respects_cost_budget():
    case = _case("above")
    output = DecisionOutput(
        case_id="case-1",
        policy="test",
        action="answer",
        prediction="above",
        credible_set=(),
        belief={"below": 0.0, "within": 0.0, "above": 1.0},
        decision_loss=0.0,
        evidence_cost=0.5,
        delay_cost=0.0,
        total_cost=0.5,
        decision_days=0.0,
        used_evidence=(),
        leaked=False,
    )
    assert budgeted_accuracy([output], [case], 1.0, fallback="below") == 1.0
    assert budgeted_accuracy([output], [case], 0.0, fallback="below") == 0.0
    metrics = evaluate([output], [case], budgets=(0.0, 1.0))
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["budgeted_accuracy"]["1.00"] == 1.0


def test_v4_case_schema_and_expected_delay():
    record = {
        "claim_id": "synthetic-1",
        "company": "Example Corp",
        "ticker": "EX",
        "cutoff": "2024-01-15",
        "target_year": 2024,
        "target_quarter": 1,
        "guidance_low_musd": 100.0,
        "guidance_high_musd": 110.0,
        "guidance_mid_musd": 105.0,
        "guidance_width_pct": 9.5,
        "current_q_musd": 104.0,
        "current_reliability": "high",
        "outcome": {
            "bucket": "above",
            "actual_musd": 120.0,
            "date": "2024-04-20",
            "actual_source": "xbrl_search",
        },
        "split": "test",
    }
    case = v4_case(record)
    assert case is not None
    assert case.gold_hypothesis == "above"
    assert case.metadata["delay_days"] == 96.0
    assert case.metadata["expected_delay_days"] == 121.0
    assert {item.kind for item in case.evidence} == {"guidance", "current_q", "actual"}
    assert case.evidence[-1].available_at > case.cutoff
