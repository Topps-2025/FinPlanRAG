"""Tests for the nested conformal decision layer."""

from __future__ import annotations

import numpy as np

from finplan_rag.decision.conformal import ConformalConfig, NestedConformalDecisionLayer
from finplan_rag.decision.loss import LossConfig


def test_lac_and_aps_sets() -> None:
    probabilities = np.asarray([[0.7, 0.2, 0.1]])
    lac = NestedConformalDecisionLayer.lac_sets(probabilities, 0.85)
    aps = NestedConformalDecisionLayer.aps_sets(probabilities, 0.8)
    assert lac[0] == ("below", "within")
    assert aps[0] == ("below", "within")


def test_threshold_respects_coverage() -> None:
    layer = NestedConformalDecisionLayer()
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.4, 0.4, 0.2],
        ]
    )
    truths = ["below", "within", "above", "below"]
    threshold, loss, coverage = layer.select_threshold(
        probabilities, truths, LossConfig(), "lac", 0.1
    )
    assert 0.0 <= threshold <= 1.0
    assert coverage >= 0.9 - 1e-12
    assert loss >= 0.0


def test_fit_selects_and_predicts() -> None:
    rng = np.random.default_rng(7)
    probabilities = rng.dirichlet([1.0, 1.0, 1.0], size=60)
    truths = [["below", "within", "above"][int(np.argmax(row))] for row in probabilities]
    layer = NestedConformalDecisionLayer(
        ConformalConfig(methods=("lac", "aps"), alphas=(0.1, 0.2), n_thresholds=21)
    )
    layer.fit(
        probabilities[:30],
        truths[:30],
        probabilities[30:45],
        truths[30:45],
        LossConfig(),
    )
    layer.recalibrate(probabilities, truths, LossConfig())
    decisions = layer.predict(probabilities[45:], LossConfig())
    assert layer.method in ("lac", "aps")
    assert layer.alpha in (0.1, 0.2)
    assert len(decisions) == 15
    assert all(decision.credible_set for decision in decisions)
    assert all(decision.action in ("answer", "adjudicate", "abstain") for decision in decisions)


def test_action_prefers_confident_answer() -> None:
    action, prediction = NestedConformalDecisionLayer.choose_action(
        [0.95, 0.03, 0.02], ("below",), LossConfig()
    )
    assert prediction == "below"
    assert action in ("answer", "adjudicate", "abstain")


def test_threshold_objective_defaults_to_realized() -> None:
    config = ConformalConfig()
    assert config.threshold_objectives == ("realized",)
    assert config.threshold_objective == "realized"


def test_expected_objective_remains_available() -> None:
    rng = np.random.default_rng(11)
    probabilities = rng.dirichlet([1.0, 1.0, 1.0], size=40)
    truths = [["below", "within", "above"][int(np.argmax(row))] for row in probabilities]
    layer = NestedConformalDecisionLayer(
        ConformalConfig(
            n_thresholds=11,
            threshold_objectives=("expected", "realized"),
        )
    )
    layer.fit(
        probabilities[:20],
        truths[:20],
        probabilities[20:30],
        truths[20:30],
        LossConfig(),
    )
    assert layer.threshold_objective in ("expected", "realized")
