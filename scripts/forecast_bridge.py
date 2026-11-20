"""Deterministic numeric bridges over qualified AnchorEvolve anchors.

The LLM extracts numbers and assigns each number a model role (for example
``ytd_actual`` or ``q4_forecast``).  This module performs the arithmetic in
plain Python and records the formula.  It is deliberately a small library of
templates, not arbitrary code generation:

* ``level_bridge``: YTD actual + remaining-period forecast.
* ``growth_bridge``: level bridge divided by the prior-year actual.
* ``margin_bridge``: current margin plus price/mix and cost deltas.
* ``capacity_bridge``: current capacity + expansion - retirement.
* ``production_bridge``: capacity times utilization times yield.

If the required inputs are missing, the bridge returns no point estimate and
lists the missing roles.  The caller can then fall back to the LLM's auxiliary
numeric estimate or simply omit the numeric output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


ROLE_ALIASES = {
    "revenue_ytd": "ytd_actual",
    "ytd_revenue": "ytd_actual",
    "year_to_date": "ytd_actual",
    "prior_year": "prior_actual",
    "prior_revenue": "prior_actual",
    "fy2025_actual": "prior_actual",
    "q4": "q4_forecast",
    "fourth_quarter": "q4_forecast",
    "q4_revenue": "q4_forecast",
    "capacity_current": "current_capacity",
    "capacity_expansion": "expansion",
    "new_capacity": "expansion",
    "capex_capacity": "expansion",
    "capacity_retirement": "retirement",
    "capacity_utilization": "utilization",
    "utilization_rate": "utilization",
    "gross_margin": "current_margin",
    "current_gross_margin": "current_margin",
    "price_mix": "price_mix_delta",
    "product_mix": "price_mix_delta",
    "cost": "cost_delta",
    "cost_change": "cost_delta",
}


@dataclass
class BridgeResult:
    template: str
    point: float | None
    low: float | None
    high: float | None
    unit: str
    formula: str
    inputs: dict[str, float] = field(default_factory=dict)
    input_sources: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    method: str = ""
    interval_method: str = ""
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "unit": self.unit,
            "formula": self.formula,
            "inputs": dict(self.inputs),
            "input_sources": dict(self.input_sources),
            "missing": list(self.missing),
            "method": self.method,
            "interval_method": self.interval_method,
            "label": self.label,
        }


def _canonical_role(anchor: Mapping[str, Any]) -> str:
    role = str(anchor.get("model_role") or anchor.get("role") or "").strip().lower()
    return ROLE_ALIASES.get(role, role)


def _numeric_anchors(anchors: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        anchor
        for anchor in anchors
        if anchor.get("status") == "accepted"
        and anchor.get("numeric_value") is not None
        and anchor.get("numeric_verified", True) is not False
        and anchor.get("model_role_valid", True) is not False
    ]


def _quarters_from_period(period: str) -> int | None:
    text = str(period or "").lower()
    if "q1-q3" in text or "前三季度" in text or "1-3q" in text:
        return 3
    if "h1" in text or "半年度" in text or "上半年" in text:
        return 2
    if "q1" in text or "一季度" in text:
        return 1
    return None


def _pick(
    anchors: Sequence[Mapping[str, Any]],
    role: str,
) -> Mapping[str, Any] | None:
    candidates = [anchor for anchor in anchors if _canonical_role(anchor) == role]
    if not candidates:
        return None
    candidates.sort(
        key=lambda anchor: (
            -float(anchor.get("reliability_effective") or 0.0),
            0 if anchor.get("quote_verified") else 1,
        )
    )
    return candidates[0]


def _value(anchor: Mapping[str, Any] | None) -> float | None:
    if anchor is None:
        return None
    try:
        return float(anchor.get("numeric_value"))
    except (TypeError, ValueError):
        return None


def _source(anchor: Mapping[str, Any] | None) -> str:
    if anchor is None:
        return ""
    return str(anchor.get("source_id") or "")


def _fraction(value: float | None, anchor: Mapping[str, Any] | None) -> float | None:
    if value is None:
        return None
    unit = str((anchor or {}).get("numeric_unit") or "").strip()
    if "%" in unit:
        return value / 100.0
    if value > 1.5:
        return value / 100.0
    return value


def _interval(
    point: float | None,
    low: float | None,
    high: float | None,
    *,
    default_pct: float,
) -> tuple[float | None, float | None, str]:
    if point is None:
        return low, high, ""
    if low is not None and high is not None:
        return low, high, "inputs"
    span = abs(point) * default_pct
    return point - span, point + span, "heuristic"


def _level_bridge(
    anchors: Sequence[Mapping[str, Any]],
    claim: Any,
    config: Mapping[str, Any],
) -> BridgeResult:
    ytd = _pick(anchors, "ytd_actual")
    remaining = _pick(anchors, "q4_forecast") or _pick(anchors, "remaining_forecast")
    q3 = _pick(anchors, "q3_revenue")
    seasonal = _pick(anchors, "seasonal_factor")
    consensus = _pick(anchors, "consensus")
    target = _pick(anchors, "fy_target") or _pick(anchors, "target")
    prior = _pick(anchors, "prior_actual")
    inputs: dict[str, float] = {}
    sources: dict[str, str] = {}

    def finish(point: float, formula: str, method: str, unit: str) -> BridgeResult:
        low = _value(_pick(anchors, "remaining_low"))
        high = _value(_pick(anchors, "remaining_high"))
        low = point - (float(_value(remaining)) - low) if low is not None and _value(remaining) is not None else None
        high = point + (high - float(_value(remaining))) if high is not None and _value(remaining) is not None else None
        low, high, interval_method = _interval(
            point, low, high,
            default_pct=float((config.get("bridge") or {}).get("default_interval_pct") or 0.1),
        )
        return BridgeResult(
            template="level_bridge",
            point=point,
            low=low,
            high=high,
            unit=unit,
            formula=formula,
            inputs=inputs,
            input_sources=sources,
            method=method,
            interval_method=interval_method,
        )

    if _value(ytd) is not None and _value(remaining) is not None:
        point = float(_value(ytd)) + float(_value(remaining))
        inputs["ytd_actual"] = float(_value(ytd))
        inputs["q4_forecast"] = float(_value(remaining))
        sources["ytd_actual"] = _source(ytd)
        sources["q4_forecast"] = _source(remaining)
        return finish(
            point,
            f"{_value(ytd):.2f} + {_value(remaining):.2f} = {point:.2f}",
            "ytd_plus_q4",
            str((remaining or {}).get("numeric_unit") or (ytd or {}).get("numeric_unit") or claim.unit or ""),
        )

    if _value(ytd) is not None and _value(q3) is not None and _value(seasonal) is not None:
        q4 = float(_value(q3)) * float(_value(seasonal))
        point = float(_value(ytd)) + q4
        inputs["ytd_actual"] = float(_value(ytd))
        inputs["q3_revenue"] = float(_value(q3))
        inputs["seasonal_factor"] = float(_value(seasonal))
        sources["ytd_actual"] = _source(ytd)
        sources["q3_revenue"] = _source(q3)
        sources["seasonal_factor"] = _source(seasonal)
        return finish(
            point,
            f"{_value(ytd):.2f} + {_value(q3):.2f} × {_value(seasonal):.3f} = {point:.2f}",
            "ytd_plus_seasonal_q4",
            str((ytd or {}).get("numeric_unit") or (q3 or {}).get("numeric_unit") or claim.unit or ""),
        )

    quarters = _quarters_from_period(str((ytd or {}).get("numeric_period") or ""))
    if _value(ytd) is not None and quarters:
        point = float(_value(ytd)) / quarters * 4.0
        inputs["ytd_actual"] = float(_value(ytd))
        inputs["ytd_quarters"] = float(quarters)
        sources["ytd_actual"] = _source(ytd)
        return finish(
            point,
            f"{_value(ytd):.2f} / {quarters} × 4 = {point:.2f}",
            "run_rate",
            str((ytd or {}).get("numeric_unit") or claim.unit or ""),
        )

    if _value(prior) is not None:
        # A prior-year actual alone is not a forecast; report the missing slot.
        return BridgeResult(
            template="level_bridge",
            point=None,
            low=None,
            high=None,
            unit=str(claim.unit or ""),
            formula="",
            inputs={"prior_actual": float(_value(prior))},
            input_sources={"prior_actual": _source(prior)},
            missing=["ytd_actual", "q4_forecast"],
            method="missing_inputs",
        )
    return BridgeResult(
        template="level_bridge",
        point=None,
        low=None,
        high=None,
        unit=str(claim.unit or ""),
        formula="",
        missing=["ytd_actual", "q4_forecast"],
        method="missing_inputs",
    )


def _growth_bridge(
    anchors: Sequence[Mapping[str, Any]],
    claim: Any,
    config: Mapping[str, Any],
) -> BridgeResult:
    level = _level_bridge(anchors, claim, config)
    prior = _pick(anchors, "prior_actual")
    prior_value = _value(prior)
    if prior_value is None and claim.reference_value is not None:
        prior_value = float(claim.reference_value)
    if level.point is None or prior_value in (None, 0):
        return BridgeResult(
            template="growth_bridge",
            point=None,
            low=None,
            high=None,
            unit="%",
            formula=level.formula,
            inputs=dict(level.inputs),
            input_sources=dict(level.input_sources),
            missing=list(dict.fromkeys(level.missing + ["prior_actual"])),
            method="missing_inputs",
        )
    point = (level.point / float(prior_value) - 1.0) * 100.0
    low = (
        (level.low / float(prior_value) - 1.0) * 100.0
        if level.low is not None
        else None
    )
    high = (
        (level.high / float(prior_value) - 1.0) * 100.0
        if level.high is not None
        else None
    )
    low, high, interval_method = _interval(
        point,
        low,
        high,
        default_pct=float((config.get("bridge") or {}).get("default_interval_pct") or 0.1),
    )
    if level.interval_method != "inputs":
        interval_method = "heuristic"
    inputs = dict(level.inputs)
    inputs["prior_actual"] = float(prior_value)
    sources = dict(level.input_sources)
    sources["prior_actual"] = _source(prior) or "claim.reference"
    formula = f"({level.formula}) / {prior_value:.2f} - 1 = {point:.2f}%"
    return BridgeResult(
        template="growth_bridge",
        point=point,
        low=low,
        high=high,
        unit="%",
        formula=formula,
        inputs=inputs,
        input_sources=sources,
        method="level_growth",
        interval_method=interval_method,
    )


def _margin_bridge(
    anchors: Sequence[Mapping[str, Any]],
    claim: Any,
    config: Mapping[str, Any],
) -> BridgeResult:
    current = _pick(anchors, "current_margin")
    price_mix = _pick(anchors, "price_mix_delta")
    cost = _pick(anchors, "cost_delta")
    current_value = _value(current)
    if current_value is None:
        return BridgeResult(
            template="margin_bridge",
            point=None,
            low=None,
            high=None,
            unit="%",
            formula="",
            missing=["current_margin"],
            method="missing_inputs",
        )
    point = current_value
    inputs = {"current_margin": current_value}
    sources = {"current_margin": _source(current)}
    formula = f"{current_value:.2f}"
    changed = False
    for name, anchor in (("price_mix_delta", price_mix), ("cost_delta", cost)):
        value = _value(anchor)
        if value is not None:
            point += value
            inputs[name] = value
            sources[name] = _source(anchor)
            formula += f" + {value:.2f}"
            changed = True
    low, high, interval_method = _interval(
        point,
        None,
        None,
        default_pct=float((config.get("bridge") or {}).get("default_interval_pct") or 0.1),
    )
    return BridgeResult(
        template="margin_bridge",
        point=point,
        low=low,
        high=high,
        unit="%",
        formula=(
            f"{formula} = {point:.2f}%" if changed
            else f"current margin = {point:.2f}%"
        ),
        inputs=inputs,
        input_sources=sources,
        method="margin_additive" if changed else "current_level",
        interval_method=interval_method,
    )


def _capacity_bridge(
    anchors: Sequence[Mapping[str, Any]],
    claim: Any,
    config: Mapping[str, Any],
) -> BridgeResult:
    current = _pick(anchors, "current_capacity")
    expansion = _pick(anchors, "expansion")
    retirement = _pick(anchors, "retirement")
    current_value = _value(current)
    if current_value is None:
        return BridgeResult(
            template="capacity_bridge",
            point=None,
            low=None,
            high=None,
            unit=str(claim.unit or ""),
            formula="",
            missing=["current_capacity"],
            method="missing_inputs",
        )
    point = current_value
    inputs = {"current_capacity": current_value}
    sources = {"current_capacity": _source(current)}
    formula = f"{current_value:.2f}"
    changed = False
    for name, anchor in (("expansion", expansion), ("retirement", retirement)):
        value = _value(anchor)
        if value is not None:
            point += value if name == "expansion" else -value
            inputs[name] = value
            sources[name] = _source(anchor)
            formula += f" + {value:.2f}" if name == "expansion" else f" - {value:.2f}"
            changed = True
    low, high, interval_method = _interval(
        point,
        None,
        None,
        default_pct=float((config.get("bridge") or {}).get("default_interval_pct") or 0.1),
    )
    return BridgeResult(
        template="capacity_bridge",
        point=point,
        low=low,
        high=high,
        unit=str((current or {}).get("numeric_unit") or claim.unit or ""),
        formula=(
            f"{formula} = {point:.2f}" if changed
            else f"current capacity = {point:.2f}"
        ),
        inputs=inputs,
        input_sources=sources,
        method="capacity_addition" if changed else "current_level",
        interval_method=interval_method,
    )


def _production_bridge(
    anchors: Sequence[Mapping[str, Any]],
    claim: Any,
    config: Mapping[str, Any],
) -> BridgeResult:
    capacity = _pick(anchors, "capacity") or _pick(anchors, "current_capacity")
    utilization = _pick(anchors, "utilization")
    yield_anchor = _pick(anchors, "yield")
    capacity_value = _value(capacity)
    if capacity_value is None:
        return BridgeResult(
            template="production_bridge",
            point=None,
            low=None,
            high=None,
            unit=str(claim.unit or ""),
            formula="",
            missing=["capacity"],
            method="missing_inputs",
        )
    point = capacity_value
    inputs = {"capacity": capacity_value}
    sources = {"capacity": _source(capacity)}
    formula = f"{capacity_value:.2f}"
    changed = False
    util_value = _fraction(_value(utilization), utilization)
    if util_value is not None:
        point *= util_value
        inputs["utilization"] = util_value
        sources["utilization"] = _source(utilization)
        formula += f" × {util_value:.3f}"
        changed = True
    yield_value = _fraction(_value(yield_anchor), yield_anchor)
    if yield_value is not None:
        point *= yield_value
        inputs["yield"] = yield_value
        sources["yield"] = _source(yield_anchor)
        formula += f" × {yield_value:.3f}"
        changed = True
    low, high, interval_method = _interval(
        point,
        None,
        None,
        default_pct=float((config.get("bridge") or {}).get("default_interval_pct") or 0.1),
    )
    return BridgeResult(
        template="production_bridge",
        point=point,
        low=low,
        high=high,
        unit=str((capacity or {}).get("numeric_unit") or claim.unit or ""),
        formula=(
            f"{formula} = {point:.2f}" if changed
            else f"current capacity = {point:.2f}"
        ),
        inputs=inputs,
        input_sources=sources,
        method=(
            "capacity_times_utilization_times_yield"
            if changed else "current_level"
        ),
        interval_method=interval_method,
    )


def label_for_value(value: float | None, thresholds: Sequence[Mapping[str, Any]]) -> str | None:
    if value is None:
        return None
    for row in thresholds:
        low = row.get("low")
        high = row.get("high")
        if (low is None or value >= float(low)) and (high is None or value < float(high)):
            return str(row.get("label"))
    return None


def build_forecast(
    claim: Any,
    anchors: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> BridgeResult | None:
    template = str(getattr(claim, "model_template", "") or "")
    numeric = _numeric_anchors(anchors)
    if template == "level_bridge":
        result = _level_bridge(numeric, claim, config)
    elif template == "growth_bridge":
        result = _growth_bridge(numeric, claim, config)
    elif template == "margin_bridge":
        result = _margin_bridge(numeric, claim, config)
    elif template == "capacity_bridge":
        result = _capacity_bridge(numeric, claim, config)
    elif template == "production_bridge":
        result = _production_bridge(numeric, claim, config)
    else:
        return None
    if result is not None:
        result.label = label_for_value(result.point, claim.thresholds)
    return result


def bridge_anchor(
    result: BridgeResult | None,
    claim: Any,
    cutoff: str,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    if result is None or result.point is None or result.label is None:
        return None
    bridge_cfg = config.get("bridge") or {}
    base = float(bridge_cfg.get("reliability_base") or 0.55)
    step = float(bridge_cfg.get("reliability_step") or 0.08)
    cap = float(bridge_cfg.get("max_reliability") or 0.90)
    reliability = min(cap, base + step * len(result.inputs))
    method_caps = {
        "run_rate": 0.60,
        "current_level": 0.55,
        "target_floor": 0.50,
        "consensus": 0.55,
    }
    reliability = min(reliability, method_caps.get(result.method, cap))
    return {
        "type": "model_bridge",
        "relation": "level",
        "value": result.formula,
        "scope": "company",
        "source_id": "MODEL",
        "quote": result.formula,
        "date": cutoff,
        "reliability": reliability,
        "hard": False,
        "implied_label": result.label,
        "implied_label_matched": result.label,
        "rationale": result.formula,
        "qualification": "model",
        "status": "accepted",
        "quote_verified": True,
        "synthetic": True,
        "reliability_effective": reliability,
        "model_role": "model_bridge",
        "numeric_value": result.point,
        "numeric_unit": result.unit,
        "numeric_period": getattr(claim, "target_period", ""),
        "reasons": [],
    }
