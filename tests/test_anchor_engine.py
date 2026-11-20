"""Unit tests for the AnchorEvolve CLI engine.

These tests exercise the deterministic parts of the pipeline: claim contracts,
anchor qualification, date-label projection and the full-action readout.  They
do not call the search providers or the LLM.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from anchor_engine import (  # noqa: E402
    CaseSpec,
    ClaimContract,
    _date_label_from_text,
    build_query_plan,
    estimate_date,
    full_action_readout,
    load_case,
    qualify_anchors,
)
from forecast_bridge import build_forecast  # noqa: E402

LOSS = {
    "wrong": 10.0,
    "over": 12.0,
    "under": 8.0,
    "miss": 8.0,
    "width": 2.0,
    "abstain": 5.0,
    "alpha": 0.1,
}
CONFIG = {
    "anchors": {
        "hard_reliability": 0.8,
        "soft_likelihood": 0.7,
        "llm_score_weight": 0.5,
        "undated_reliability_cap": 0.55,
        "unverified_quote_factor": 0.7,
        "max_anchors": 12,
    }
}


def revenue_claim() -> ClaimContract:
    return ClaimContract(
        claim_id="revenue",
        text="2026年营业收入同比增速",
        metric="营业收入同比增速",
        claim_type="metric",
        labels=["低于140%", "140%至170%", "高于170%"],
        target_period="2026年年度报告",
    )


def date_claim() -> ClaimContract:
    return ClaimContract(
        claim_id="date",
        text="2026年年度报告披露日期",
        metric="年度报告披露日期",
        claim_type="date",
        labels=["不晚于2027-03-31", "2027-04-01至2027-04-15", "晚于2027-04-15"],
        target_period="2026年年度报告",
        date_windows=[
            {"label": "不晚于2027-03-31", "start": "2027-01-01", "end": "2027-03-31"},
            {"label": "2027-04-01至2027-04-15", "start": "2027-04-01", "end": "2027-04-15"},
            {"label": "晚于2027-04-15", "start": "2027-04-16", "end": "2027-04-30"},
        ],
    )


def evidence() -> list[dict]:
    return [
        {
            "source_id": "W1",
            "title": "三季报",
            "url": "https://example.com/1",
            "date": "2026-10-29",
            "text": "2026年前三季度营业收入695.47亿元，同比增长178.13%。",
            "snippet": "2026年前三季度营业收入695.47亿元，同比增长178.13%。",
        },
        {
            "source_id": "W2",
            "title": "旧年报",
            "url": "https://example.com/2",
            "date": "2026-03-30",
            "text": "2025年营业收入382.40亿元，同比增长60.25%。",
            "snippet": "2025年营业收入382.40亿元，同比增长60.25%。",
        },
    ]


def anchor(**overrides) -> dict:
    value = {
        "type": "company_history",
        "relation": "trend",
        "value": "2026年前三季度营业收入695.47亿元，同比+178.13%",
        "scope": "company",
        "source_id": "W1",
        "quote": "2026年前三季度营业收入695.47亿元，同比增长178.13%",
        "date": "2026-10-29",
        "reliability": 0.9,
        "hard": False,
        "implied_label": "高于170%",
        "rationale": "cumulative growth already above the top threshold",
    }
    value.update(overrides)
    return value


def test_load_case_has_numeric_thresholds() -> None:
    spec = load_case(REPO_ROOT / "configs" / "anchor_cases" / "zhongji_2026.yaml")
    assert spec.company == "中际旭创"
    assert len(spec.claims) == 4
    assert spec.claims[0].label_defs["高于170%"]


def test_soft_anchor_accepted() -> None:
    parsed = {"anchors": [anchor()]}
    rows = qualify_anchors(parsed, revenue_claim(), evidence(), date(2026, 11, 19), CONFIG)
    assert len(rows) == 1
    assert rows[0]["status"] == "accepted"
    assert rows[0]["qualification"] == "soft"
    assert rows[0]["quote_verified"] is True


def test_hard_anchor_requires_verified_quote_and_date() -> None:
    parsed = {"anchors": [anchor(hard=True)]}
    rows = qualify_anchors(parsed, revenue_claim(), evidence(), date(2026, 11, 19), CONFIG)
    assert rows[0]["qualification"] == "hard"

    parsed = {"anchors": [anchor(hard=True, reliability=0.79)]}
    rows = qualify_anchors(parsed, revenue_claim(), evidence(), date(2026, 11, 19), CONFIG)
    assert rows[0]["qualification"] == "soft"


def test_stale_anchor_rejected() -> None:
    parsed = {"anchors": [anchor(date="2026-12-01")]}
    rows = qualify_anchors(parsed, revenue_claim(), evidence(), date(2026, 11, 19), CONFIG)
    assert rows[0]["status"] == "rejected"
    assert rows[0]["qualification"] == "stale"


def test_unknown_source_rejected() -> None:
    parsed = {"anchors": [anchor(source_id="W9")]}
    rows = qualify_anchors(parsed, revenue_claim(), evidence(), date(2026, 11, 19), CONFIG)
    assert rows[0]["status"] == "rejected"
    assert rows[0]["qualification"] == "adversarial"


def test_prior_year_reference_anchor_rejected_for_growth_claim() -> None:
    parsed = {
        "anchors": [
            anchor(
                value="2025年营业收入382.40亿元",
                source_id="W2",
                quote="2025年营业收入382.40亿元，同比增长60.25%",
                date="2026-03-30",
                implied_label="低于140%",
            )
        ]
    }
    rows = qualify_anchors(parsed, revenue_claim(), evidence(), date(2026, 11, 19), CONFIG)
    assert rows[0]["status"] == "rejected"
    assert rows[0]["qualification"] == "vacuous"


def test_unverified_quote_is_downgraded() -> None:
    parsed = {"anchors": [anchor(quote="这句话不在原文里")]}
    rows = qualify_anchors(parsed, revenue_claim(), evidence(), date(2026, 11, 19), CONFIG)
    assert rows[0]["quote_verified"] is False
    assert rows[0]["reliability_effective"] < 0.9
    assert rows[0]["qualification"] == "soft"


def test_date_label_projection() -> None:
    claim = date_claim()
    assert _date_label_from_text("2025年年度报告于2026-03-31披露", claim) == "不晚于2027-03-31"
    assert _date_label_from_text("2024年年度报告于2025-04-21披露", claim) == "晚于2027-04-15"
    assert _date_label_from_text("2024年年度报告于2025-04-10披露", claim) == "2027-04-01至2027-04-15"
    assert _date_label_from_text("2025-03-30 和 2025-04-21", claim) is None


def test_regulation_anchor_is_vacuous_for_date_claim() -> None:
    item = evidence()[0]
    item["text"] += " 创业板年报法定截止日为次年4月30日。"
    parsed = {
        "anchors": [
            anchor(
                type="regulation",
                value="创业板年报法定截止日为次年4月30日",
                quote="创业板年报法定截止日为次年4月30日",
                implied_label="晚于2027-04-15",
            )
        ]
    }
    rows = qualify_anchors(parsed, date_claim(), [item], date(2026, 11, 19), CONFIG)
    assert rows[0]["status"] == "rejected"
    assert rows[0]["qualification"] == "vacuous"


def test_readout_confident_answer() -> None:
    result = full_action_readout([0.9, 0.05, 0.05], revenue_claim().labels, LOSS)
    assert result["action"] == "answer"
    assert result["point"] == "低于140%"
    assert result["set"] == ["低于140%"]


def test_readout_flat_belief_adjudicates() -> None:
    result = full_action_readout([0.34, 0.33, 0.33], revenue_claim().labels, LOSS)
    assert result["action"] == "adjudicate"
    assert len(result["set"]) >= 2


def test_readout_never_uses_singleton_adjudication() -> None:
    for belief in ([0.7, 0.2, 0.1], [0.45, 0.4, 0.15], [0.34, 0.33, 0.33]):
        result = full_action_readout(belief, revenue_claim().labels, LOSS)
        if result["action"] == "adjudicate":
            assert len(result["set"]) >= 2


def test_date_estimate_point_and_interval() -> None:
    claim = date_claim()
    readout = {"set": ["不晚于2027-03-31"]}
    parsed = {
        "date_estimate": {
            "point": "2027-03-30",
            "low": "2027-03-20",
            "high": "2027-04-10",
            "basis": "history",
        }
    }
    result = estimate_date(claim, [0.7, 0.2, 0.1], readout, parsed)
    assert result["point"] == "2027-03-30"
    assert result["low"] == "2027-03-20"
    assert result["high"] == "2027-04-10"
    assert result["consistent_with_window"] is True


def test_date_estimate_falls_back_to_window_midpoint() -> None:
    claim = date_claim()
    readout = {"set": ["不晚于2027-03-31"]}
    parsed = {"date_estimate": {"point": "2027-05-10", "low": "2027-05-01", "high": "2027-05-20"}}
    result = estimate_date(claim, [0.7, 0.2, 0.1], readout, parsed)
    assert result["consistent_with_window"] is False
    assert result["window_start"] <= result["point"] <= result["window_end"]


def query_spec(company: str = "新易盛", ticker: str = "300502.SZ") -> CaseSpec:
    return CaseSpec(
        case_id="query-test",
        question="",
        company=company,
        ticker=ticker,
        target_period="2026年年度报告",
        claims=[],
        cutoff="2026-11-19",
    )


def test_query_plan_reserves_required_facets() -> None:
    claim = ClaimContract(
        claim_id="revenue",
        text="2026年营业收入高于2025年",
        metric="营业收入",
        claim_type="metric",
        labels=["高", "中", "低"],
        target_period="2026年年度报告",
        queries=[
            "新易盛 2025年年度报告 营业收入",
            "新易盛 2026年 营业收入 指引",
            "新易盛 2026年 业绩预告 营业收入",
        ],
    )
    plan = build_query_plan(
        claim,
        query_spec(),
        max_queries=8,
        max_parser_queries=2,
        max_per_facet=2,
    )
    facets = [entry["facet"] for entry in plan]
    assert "latest_period" in facets
    assert "prior_period" in facets
    assert sum(1 for entry in plan if entry["facet"] == "parser") <= 2
    assert any(
        entry["required"] and entry["facet"] == "latest_period" for entry in plan
    )


def test_query_plan_date_claim_uses_date_facet() -> None:
    plan = build_query_plan(
        date_claim(),
        query_spec(),
        max_queries=8,
        max_parser_queries=2,
        max_per_facet=2,
    )
    facets = [entry["facet"] for entry in plan]
    assert "date_disclosure" in facets
    assert "latest_period" not in facets


def test_query_plan_other_metric_keeps_required_facets() -> None:
    claim = ClaimContract(
        claim_id="margin",
        text="2026年毛利率高于2025年",
        metric="毛利率",
        claim_type="metric",
        labels=["高", "中", "低"],
        target_period="2026年年度报告",
        queries=["新易盛 2025年 毛利率", "新易盛 2026年 毛利率 预测"],
    )
    plan = build_query_plan(
        claim,
        query_spec(),
        max_queries=8,
        max_parser_queries=2,
        max_per_facet=2,
    )
    facets = [entry["facet"] for entry in plan]
    assert "latest_period" in facets
    assert "prior_period" in facets
    assert any("毛利率" in entry["query"] for entry in plan)


BRIDGE_CONFIG = {
    "bridge": {
        "default_interval_pct": 0.10,
        "reliability_base": 0.55,
        "reliability_step": 0.08,
        "max_reliability": 0.90,
    }
}


def numeric_anchor(role: str, value: float, unit: str = "亿元", **overrides) -> dict:
    row = {
        "status": "accepted",
        "qualification": "soft",
        "model_role": role,
        "numeric_value": value,
        "numeric_unit": unit,
        "numeric_period": "2026Q1-Q3",
        "numeric_verified": True,
        "model_role_valid": True,
        "reliability_effective": 0.9,
        "source_id": "W1",
        "quote_verified": True,
    }
    row.update(overrides)
    return row


def growth_claim() -> ClaimContract:
    return ClaimContract(
        claim_id="growth",
        text="2026年营业收入同比增速",
        metric="营业收入同比增速",
        claim_type="metric",
        labels=["低于140%", "140%至170%", "高于170%"],
        target_period="2026年年度报告",
        unit="%",
        metric_kind="growth",
        model_template="growth_bridge",
        thresholds=[
            {"label": "低于140%", "low": None, "high": 140},
            {"label": "140%至170%", "low": 140, "high": 170},
            {"label": "高于170%", "low": 170, "high": None},
        ],
        reference_value=248.42,
        reference_unit="亿元",
    )


def test_growth_bridge_uses_ytd_and_q4() -> None:
    claim = growth_claim()
    anchors = [
        numeric_anchor("ytd_actual", 318.5),
        numeric_anchor("prior_actual", 248.42, "亿元", numeric_period="2025FY"),
        numeric_anchor("q4_forecast", 147.0),
    ]
    result = build_forecast(claim, anchors, BRIDGE_CONFIG)
    assert result is not None
    assert result.template == "growth_bridge"
    assert result.point is not None
    assert abs(result.point - 87.38) < 0.1
    assert result.label == "低于140%"
    assert "318.50" in result.formula


def test_capacity_bridge_adds_expansion_and_retirement() -> None:
    claim = ClaimContract(
        claim_id="capacity",
        text="2026年产能",
        metric="产能",
        claim_type="metric",
        labels=["低于", "接近", "高于"],
        target_period="2026年年度报告",
        unit="万只",
        metric_kind="capacity",
        model_template="capacity_bridge",
        thresholds=[
            {"label": "低于", "low": None, "high": 110},
            {"label": "接近", "low": 110, "high": 120},
            {"label": "高于", "low": 120, "high": None},
        ],
    )
    anchors = [
        numeric_anchor("current_capacity", 100, "万只"),
        numeric_anchor("expansion", 20, "万只"),
        numeric_anchor("retirement", 5, "万只"),
    ]
    result = build_forecast(claim, anchors, BRIDGE_CONFIG)
    assert result is not None
    assert result.point == 115
    assert result.label == "接近"


def test_bridge_skips_unverified_numeric() -> None:
    claim = growth_claim()
    anchors = [
        numeric_anchor("ytd_actual", 318.5),
        numeric_anchor("prior_actual", 248.42, numeric_verified=False),
        numeric_anchor("q4_forecast", 147.0),
    ]
    result = build_forecast(claim, anchors, BRIDGE_CONFIG)
    assert result is not None
    assert result.inputs.get("prior_actual") == 248.42
    assert result.input_sources.get("prior_actual") == "claim.reference"


def test_query_plan_includes_parser_facets() -> None:
    claim = ClaimContract(
        claim_id="capacity",
        text="2026年产能",
        metric="产能",
        claim_type="metric",
        labels=["高", "中", "低"],
        target_period="2026年年度报告",
        queries=["新易盛 2026年 产能"],
        query_facets=[
            {
                "name": "capacity_orders",
                "required": True,
                "queries": ["新易盛 2026年 订单", "新易盛 2026年 产能 扩张"],
            }
        ],
    )
    plan = build_query_plan(
        claim,
        query_spec(),
        max_queries=8,
        max_parser_queries=2,
        max_per_facet=2,
    )
    facets = [entry["facet"] for entry in plan]
    assert "claim_capacity_orders" in facets
    assert any(
        entry["facet"] == "claim_capacity_orders" and entry["required"]
        for entry in plan
    )
