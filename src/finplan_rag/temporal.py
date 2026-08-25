"""Financial-period parsing and filing-obligation rules.

The parser is deliberately deterministic and conservative.  It maps language
about fiscal periods to the filings that physically report those periods; it
does not inspect a gold answer or a hidden document list.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

from .models import DocumentObligation

ORDINAL_QUARTERS = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}
COUNT_WORDS = {"two": 2, "three": 3, "four": 4}


def period_requests(question: str) -> set[tuple[int, str]]:
    """Return requested ``(fiscal_year, period)`` pairs from an English query.

    ``Q4`` is normalized to ``FY`` because annual SEC filings are 10-K/FY.
    Half-year and cumulative-quarter phrases map to the filing that reports
    the cumulative period rather than to every earlier standalone quarter.
    """

    requests: set[tuple[int, str]] = set()
    flags = re.IGNORECASE
    for first, second, year in re.findall(
        r"\bQ([1-4])\s+and\s+Q([1-4])\s+of\s+(?:FY\s*|fiscal\s+year\s+)?(20\d{2})\b",
        question, flags,
    ):
        requests.update({(int(year), f"Q{first}"), (int(year), f"Q{second}")})
    for quarter, year in re.findall(
        r"\bQ([1-4])\s+(?:of\s+)?(?:FY\s*|fiscal\s+year\s+)?(20\d{2})\b",
        question, flags,
    ):
        requests.add((int(year), f"Q{quarter}"))
    for ordinal, year in re.findall(
        r"\b(first|second|third|fourth)\s+quarter\s+(?:of\s+)?(?:FY\s*|fiscal\s+year\s+)?(20\d{2})\b",
        question, flags,
    ):
        requests.add((int(year), ORDINAL_QUARTERS[ordinal.lower()]))
    for count, year in re.findall(
        r"\bfirst\s+(two|three|four|\d+)\s+quarters\s+of\s+(?:FY\s*|fiscal\s+year\s+)?(20\d{2})\b",
        question, flags,
    ):
        n = int(count) if count.isdigit() else COUNT_WORDS[count.lower()]
        requests.add((int(year), f"Q{min(n, 4)}"))
    for half, year in re.findall(
        r"\b(first|second)\s+half\s+of\s+(?:FY\s*|fiscal\s+year\s+)?(20\d{2})\b",
        question, flags,
    ):
        if half.lower() == "first":
            requests.add((int(year), "Q2"))
        else:
            requests.update({(int(year), "Q3"), (int(year), "Q4")})
    for match in re.finditer(r"\b(?:fiscal\s+year|FY)\s*(20\d{2})\b", question, flags):
        prefix = question[max(0, match.start() - 32) : match.start()]
        # In "the first three quarters of FY 2024", FY labels the
        # cumulative Q3 request; it is not an additional annual obligation.
        if re.search(r"(?:quarters?|half)\s+of\s*$", prefix, flags):
            continue
        requests.add((int(match.group(1)), "Q4"))
    if requests and re.search(r"\bsame\s+period\s+(?:in\s+the\s+)?(?:previous|prior|last)\s+year\b", question, flags):
        requests.update((year - 1, period) for year, period in tuple(requests))
    return requests


def obligation_signature(year: int, period: str) -> tuple[int, str, str]:
    """Map a period to the SEC filing type and fiscal-period label."""

    normalized = period.upper()
    return (year, "10-K", "FY") if normalized in {"Q4", "FY", "ANNUAL"} else (year, "10-Q", normalized)


def build_obligations(
    question: str,
    entities: Sequence[str],
    filings: Iterable[Mapping[str, object]],
    cutoff: object | None = None,
) -> tuple[DocumentObligation, ...]:
    """Resolve parsed periods against visible filing metadata.

    ``filings`` must contain ``entity``, ``fiscal_year``, ``filing_type``,
    ``fiscal_period`` and optionally ``available_at``/``document_id``.
    """

    visible = []
    for filing in filings:
        if cutoff is not None and filing.get("available_at") is not None and str(filing["available_at"]) > str(cutoff):
            continue
        visible.append(filing)
    requested = period_requests(question)
    out: dict[tuple[str, int, str, str], DocumentObligation] = {}
    for entity in entities:
        candidates = [f for f in visible if str(f.get("entity", "")).lower() == entity.lower()]
        if requested:
            signatures = {obligation_signature(year, period) for year, period in requested}
            selected = [f for f in candidates if (int(f["fiscal_year"]), str(f["filing_type"]), str(f["fiscal_period"])) in signatures]
        else:
            selected = [f for f in candidates if str(f["filing_type"]) == "10-K" and str(f["fiscal_period"]) == "FY"]
        for filing in selected:
            item = DocumentObligation(entity, int(filing["fiscal_year"]), str(filing["filing_type"]), str(filing["fiscal_period"]), str(filing.get("document_id", "")))
            out[item.key] = item
    return tuple(out[key] for key in sorted(out))
