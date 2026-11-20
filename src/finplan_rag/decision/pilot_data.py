"""Build decision-level pilot cases from the v4 revenue-guidance dataset.

The source data is the 1,005-claim point-in-time revenue-guidance benchmark
created for AnchorEvolve.  This module does not modify the source data.  It
maps each claim to:

* hypotheses: below / within / above the guided range;
* evidence: the guidance range, the current-quarter anchor, and the later
  outcome filing (available only after its filing date);
* a gold outcome bucket;
* a delay (days from cutoff to the outcome filing) used to price waiting.

The mapping is a decision-level abstraction.  It is not an end-to-end reader
benchmark, and the outcome filing is a future evidence item, not a label the
policy may use without paying the delay cost.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .schema import DecisionCase, EvidenceItem, Hypothesis

LABELS = ("below", "within", "above")
HYPOTHESES = tuple(Hypothesis(label, label) for label in LABELS)
DEFAULT_EXPECTED_DELAY_DAYS = 100.0


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value)[:19])
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def reliability_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    return {"low": 0.3, "medium": 0.6, "high": 0.9}.get(str(value).lower(), 0.5)


def quarter_end_date(year: Any, quarter: Any) -> datetime | None:
    """End date of the target quarter, used for a point-in-time delay estimate."""

    try:
        year_int = int(year)
        quarter_int = int(quarter)
    except (TypeError, ValueError):
        return None
    if quarter_int not in (1, 2, 3, 4):
        return None
    month_day = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[quarter_int]
    return datetime(year_int, month_day[0], month_day[1], tzinfo=timezone.utc)


def expected_delay_days(record: dict[str, Any], cutoff: datetime) -> float:
    """Estimate the filing delay from the target quarter, not the true outcome.

    The target quarter and its end are known at the cutoff.  Adding a fixed
    reporting lag gives a point-in-time estimate of when the outcome filing is
    likely to arrive.  The evaluator still prices waiting with the realized
    delay, so the planner does not see the true outcome date.
    """

    end = quarter_end_date(record.get("target_year"), record.get("target_quarter"))
    if end is None:
        return DEFAULT_EXPECTED_DELAY_DAYS
    estimated = end + timedelta(days=45)
    value = (estimated - cutoff).total_seconds() / 86400.0
    # The v4 target-year field is noisy for some claims; use the target-quarter
    # estimate only when it is in the plausible filing window.
    if not 45.0 <= value <= 200.0:
        return DEFAULT_EXPECTED_DELAY_DAYS
    return value


def _noise_draw(case_id: str) -> float:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") / float(2**64)


def preliminary_signal(case_id: str, gold: str, seed: int = 20261031) -> tuple[str, float]:
    """Deterministic controlled preliminary signal for the mode-switch diagnostic.

    Half of the cases receive a high-precision preliminary signal and half a
    low-precision one.  The assignment is independent of the query features a
    pre-hoc router can see, so the router cannot tell which cases are
    informative before acquiring the signal.  The seed changes the assignment.
    """

    digest = hashlib.sha256(f"preliminary:{seed}:{case_id}".encode("utf-8")).hexdigest()
    value = int(digest, 16)
    precision = 0.9 if value % 2 == 0 else 0.5
    draw = ((value >> 8) % 1000) / 1000.0
    if draw < precision:
        observed = gold
    else:
        others = [label for label in LABELS if label != gold]
        observed = others[(value >> 16) % len(others)]
    return observed, precision


def v4_case(
    record: dict[str, Any],
    *,
    actual_cost: float = 2.0,
    include_preliminary: bool = False,
    seed: int = 20261031,
) -> DecisionCase | None:
    cutoff = parse_date(record.get("cutoff"))
    outcome = record.get("outcome") or {}
    outcome_date = parse_date(outcome.get("date"))
    bucket = outcome.get("bucket")
    if cutoff is None or outcome_date is None or bucket not in LABELS:
        return None
    low = record.get("guidance_low_musd")
    high = record.get("guidance_high_musd")
    mid = record.get("guidance_mid_musd")
    current = record.get("current_q_musd")
    if low is None or high is None:
        return None
    width_pct = record.get("guidance_width_pct")
    delay_days = max(0.0, (outcome_date - cutoff).total_seconds() / 86400.0)
    evidence = [
        EvidenceItem(
            evidence_id=f"{record['claim_id']}::guidance",
            available_at=cutoff,
            cost=0.5,
            kind="guidance",
            origin=str(record.get("cik") or record.get("ticker") or "company"),
            source_type="8-K",
            reliability=1.0,
            payload={
                "low": float(low),
                "high": float(high),
                "mid": float(mid) if mid is not None else (float(low) + float(high)) / 2.0,
                "width_pct": float(width_pct) if width_pct is not None else None,
            },
        ),
    ]
    if current is not None:
        evidence.append(
            EvidenceItem(
                evidence_id=f"{record['claim_id']}::current_q",
                available_at=cutoff,
                cost=0.5,
                kind="current_q",
                origin=str(record.get("cik") or record.get("ticker") or "company"),
                source_type="8-K",
                reliability=reliability_value(record.get("current_reliability")),
                payload={"value": float(current), "reliability": record.get("current_reliability")},
            )
        )
    evidence.append(
        EvidenceItem(
            evidence_id=f"{record['claim_id']}::actual",
            available_at=outcome_date,
            cost=float(actual_cost),
            kind="actual",
            origin=str(record.get("cik") or record.get("ticker") or "company"),
            source_type="8-K",
            reliability=1.0,
            payload={
                "bucket": bucket,
                "actual_musd": outcome.get("actual_musd"),
                "source": outcome.get("actual_source"),
                "noise_draw": _noise_draw(str(record.get("claim_id", ""))),
            },
        )
    )
    preliminary_observed = None
    preliminary_precision = None
    if include_preliminary:
        preliminary_observed, preliminary_precision = preliminary_signal(
            str(record["claim_id"]), bucket, seed=seed
        )
        evidence.append(
            EvidenceItem(
                evidence_id=f"{record['claim_id']}::preliminary",
                available_at=cutoff + timedelta(days=30),
                cost=0.5,
                kind="preliminary",
                origin=str(record.get("cik") or record.get("ticker") or "company"),
                source_type="8-K",
                reliability=preliminary_precision,
                payload={
                    "observed_bucket": preliminary_observed,
                    "precision": preliminary_precision,
                },
            )
        )
    return DecisionCase(
        case_id=str(record["claim_id"]),
        task_type="mode_ambiguous",
        query=f"Will {record.get('company', 'the company')} report revenue {bucket} its guided range for "
        f"Q{record.get('target_quarter')} {record.get('target_year')}?",
        hypotheses=HYPOTHESES,
        evidence=tuple(evidence),
        cutoff=cutoff,
        gold_hypothesis=bucket,
        split=str(record.get("split", "unknown")),
        metadata={
            "company": record.get("company"),
            "company_key": str(record.get("cik") or record.get("ticker") or record.get("company") or ""),
            "ticker": record.get("ticker"),
            "sector": record.get("sector"),
            "industry": record.get("industry"),
            "target_year": record.get("target_year"),
            "target_quarter": record.get("target_quarter"),
            "guidance_width_pct": width_pct,
            "current_reliability": record.get("current_reliability"),
            "anchor_availability": record.get("anchor_availability"),
            "delay_days": delay_days,
            "expected_delay_days": expected_delay_days(record, cutoff),
            "outcome_date": outcome_date.isoformat(),
            "actual_source": outcome.get("actual_source"),
            "preliminary_observed": preliminary_observed,
            "preliminary_precision": preliminary_precision,
        },
    )


def load_v4_cases(
    path: str | Path,
    *,
    split: str | None = None,
    max_cases: int | None = None,
    actual_cost: float = 2.0,
    include_preliminary: bool = False,
    seed: int = 20261031,
) -> list[DecisionCase]:
    with Path(path).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("v4 dataset must be a JSON list")
    cases: list[DecisionCase] = []
    for record in records:
        if split is not None and record.get("split") != split:
            continue
        case = v4_case(
            record,
            actual_cost=actual_cost,
            include_preliminary=include_preliminary,
            seed=seed,
        )
        if case is not None:
            cases.append(case)
        if max_cases is not None and len(cases) >= max_cases:
            break
    return cases


def split_cases(cases: Iterable[DecisionCase], split: str) -> list[DecisionCase]:
    return [case for case in cases if case.split == split]


def observed_actual_bucket(case: DecisionCase, precision: float) -> str:
    """Deterministic noisy observation of the future filing bucket."""

    actual = next((item for item in case.evidence if item.kind == "actual"), None)
    if actual is None:
        return case.gold_hypothesis
    precision = min(1.0, max(0.0, precision))
    draw = float(actual.payload.get("noise_draw", 0.0))
    if draw < precision:
        return str(actual.payload.get("bucket", case.gold_hypothesis))
    others = [label for label in LABELS if label != case.gold_hypothesis]
    return others[int(draw * 1_000_000) % len(others)]
