"""Tests for the loss-aware policy portfolio."""

from __future__ import annotations

from finplan_rag.decision.portfolio import PortfolioConfig, select_policy


def test_default_fixed_when_no_candidate_clears_margin() -> None:
    selected = select_policy(
        {"fixed": 3.00, "abstain": 2.90, "planner": 2.95},
        PortfolioConfig(switching_margin=0.15),
    )
    assert selected == "fixed"


def test_abstain_selected_when_clearly_better() -> None:
    selected = select_policy(
        {"fixed": 3.00, "abstain": 2.00, "planner": 2.50},
        PortfolioConfig(switching_margin=0.15),
    )
    assert selected == "abstain"


def test_planner_selected_when_clearly_better() -> None:
    selected = select_policy(
        {"fixed": 5.00, "abstain": 4.90, "planner": 4.30},
        PortfolioConfig(switching_margin=0.15),
    )
    assert selected == "planner"


def test_margin_is_strict() -> None:
    selected = select_policy(
        {"fixed": 3.00, "abstain": 2.85, "planner": 3.00},
        PortfolioConfig(switching_margin=0.15),
    )
    assert selected == "fixed"
