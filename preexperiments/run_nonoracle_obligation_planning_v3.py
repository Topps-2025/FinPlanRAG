"""FinPlan v5: cutoff-aware entity and temporal obligation planning.

This revision was designed after inspecting validation3 errors.  Results on
validation3 are developmental and must not be described as confirmatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

try:
    from .storage_paths import DATA_ROOT, RESULTS_ROOT
except ImportError:
    from storage_paths import DATA_ROOT, RESULTS_ROOT

from run_lofin_multidoc_pilot import BM25, make_chunks, run_method, score
from run_nonoracle_obligation_planning import aggregate, obligations, prf
from run_nonoracle_obligation_planning_v2 import GENERIC_NAME_TOKENS, stem_token


YEAR_RE = re.compile(r"\b(20\d{2})\b")
CONCATENATED_RANGE_RE = re.compile(r"(?<!\d)(20\d{2})(20\d{2})(?!\d)")
WINDOW_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5}
RANGE_CUES_RE = re.compile(r"\b(from|between|over|trend|change|through|to)\b", re.I)
EVENT_TRANSITION_RE = re.compile(
    r"\b(prior|before|after|post|pre|divestiture|sale|acquisition|merger|restructur|segment structure)\b",
    re.I,
)


def canonical_text(value: str) -> str:
    value = value.lower().replace("&", " and ").replace("’", "'")
    value = value.replace("'", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def canonical_tokens(value: str) -> Set[str]:
    return {
        stem_token(token)
        for token in canonical_text(value).split()
        if stem_token(token) and stem_token(token) not in GENERIC_NAME_TOKENS
    }


def explicit_years(question: str) -> Tuple[List[int], bool]:
    repaired_ranges = [
        (int(match.group(1)), int(match.group(2)))
        for match in CONCATENATED_RANGE_RE.finditer(question)
    ]
    cleaned = CONCATENATED_RANGE_RE.sub(" ", question)
    years = {int(year) for year in YEAR_RE.findall(cleaned)}
    for lower, upper in repaired_ranges:
        years.update((lower, upper))
    return sorted(years), bool(repaired_ranges)


def relative_window_size(question: str) -> int | None:
    match = re.search(
        r"\b(?:past|last|previous|preceding)\s+(\d+|two|three|four|five)\s+(?:fiscal\s+)?years?\b",
        canonical_text(question),
    )
    if not match:
        return None
    raw = match.group(1)
    return int(raw) if raw.isdigit() else WINDOW_WORDS[raw]


def visible_years_by_ticker(
    documents: Sequence[Mapping[str, object]], cutoff: str
) -> Mapping[str, Set[int]]:
    visible: Dict[str, Set[int]] = {}
    for document in documents:
        available_at = str(document.get("available_at") or document.get("filing_date") or "")
        if cutoff and available_at and available_at[:10] > cutoff[:10]:
            continue
        visible.setdefault(str(document["ticker"]), set()).add(int(document["year"]))
    return visible


def asof_companies(
    companies: Sequence[Mapping[str, object]],
    documents: Sequence[Mapping[str, object]],
    cutoff: str,
) -> List[Dict[str, object]]:
    visible = visible_years_by_ticker(documents, cutoff)
    result: List[Dict[str, object]] = []
    for company in companies:
        ticker = str(company["ticker"])
        years = sorted(set(int(year) for year in company["available_years"]) & visible.get(ticker, set()))
        if years:
            copied = dict(company)
            copied["available_years"] = years
            result.append(copied)
    return result


def match_companies(question: str, companies: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    normalized = canonical_text(question)
    question_tokens = canonical_tokens(question)
    upper_tokens = set(re.findall(r"\b[A-Z][A-Z0-9.\-]{1,5}\b", question))
    company_tokens: Dict[str, List[Set[str]]] = {
        str(company["ticker"]): [canonical_tokens(str(alias)) for alias in company["aliases"]]
        for company in companies
    }
    token_df = Counter()
    for aliases in company_tokens.values():
        token_df.update(set().union(*aliases) if aliases else set())

    matched: List[Mapping[str, object]] = []
    for company in companies:
        ticker = str(company["ticker"])
        exact_alias = any(
            len(canonical_text(str(alias))) >= 3
            and re.search(
                rf"(?<![a-z0-9]){re.escape(canonical_text(str(alias)))}(?![a-z0-9])",
                normalized,
            )
            for alias in company["aliases"]
        )
        ticker_hit = ticker in upper_tokens
        coverage_hit = False
        for alias_tokens in company_tokens[ticker]:
            overlap = alias_tokens & question_tokens
            full_coverage = bool(alias_tokens) and alias_tokens <= question_tokens
            distinctive = overlap and any(token_df[token] == 1 and len(token) >= 4 for token in overlap)
            if full_coverage or distinctive:
                coverage_hit = True
                break
        if exact_alias or ticker_hit or coverage_hit:
            matched.append(company)
    return matched


def select_years(question: str, available: Sequence[int]) -> Tuple[List[int], str]:
    if not available:
        return [], "no-visible-filing"
    normalized = canonical_text(question)
    years, repaired_range = explicit_years(question)
    # Bind the year adjacent to "10-K".  A loose forward window incorrectly
    # attaches a forecast endpoint in phrases such as "from 2024 to 2028 as
    # stated in the 2023 10-K".
    filing_mentions = [
        int(year)
        for year in re.findall(r"\b(20\d{2})\s+(?:fiscal\s+year\s+)?10\s*-?\s*k\b", question, flags=re.I)
    ]
    if filing_mentions:
        return [year for year in filing_mentions if year in available], "explicit-10k"

    window = relative_window_size(question)
    if window:
        anchor = max(years) if years else max(available)
        eligible = [year for year in available if year <= anchor]
        return eligible[-window:], "relative-window"

    if len(years) >= 2 and (repaired_range or RANGE_CUES_RE.search(normalized)):
        lower, upper = min(years), max(years)
        return [year for year in available if lower <= year <= upper], "explicit-range"

    selected = [year for year in years if year in available]
    if selected:
        return selected, "explicit-year"

    if years and EVENT_TRANSITION_RE.search(normalized):
        # The named event year may precede the corresponding annual filing.
        # Use the latest two filings visible at the question's cognition cutoff
        # to compare the pre/post disclosure state without future leakage.
        return list(available[-2:]), "event-transition-visible-pair"

    return [max(available)], "latest-visible-fallback"


def cutoff_aware_rule(
    question: str,
    cutoff: str,
    companies: Sequence[Mapping[str, object]],
    documents: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], Mapping[str, str]]:
    visible_companies = asof_companies(companies, documents, cutoff)
    matched = match_companies(question, visible_companies)
    plans: List[Dict[str, object]] = []
    reasons: Dict[str, str] = {}
    for company in matched:
        ticker = str(company["ticker"])
        years, reason = select_years(question, sorted(int(year) for year in company["available_years"]))
        if years:
            plans.append({"ticker": ticker, "years": sorted(set(years))})
            reasons[ticker] = reason
    return sorted(plans, key=lambda leg: str(leg["ticker"])), reasons


def run(data_path: Path, registry_path: Path, out_path: Path, budget: int) -> Mapping[str, object]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    documents = data["documents"]
    chunks = make_chunks(documents)
    index = BM25(chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rows: List[Dict[str, object]] = []
    predictions: List[Dict[str, object]] = []
    for case in data["cases"]:
        cutoff = str(case.get("cutoff", ""))
        legs, reasons = cutoff_aware_rule(str(case["question"]), cutoff, registry["companies"], documents)
        gold_docs = set(str(doc) for doc in case["gold_doc_ids"])
        gold_entities = {doc.split("_")[0] for doc in gold_docs}
        predicted_docs = obligations(legs)
        entity_metrics = prf({str(leg["ticker"]) for leg in legs}, gold_entities)
        obligation_metrics = prf(predicted_docs, gold_docs)
        planned_case = dict(case)
        planned_case["candidate_legs"] = legs
        cascade = run_method(planned_case, index, budget, "finplan_v3_path_bound")
        cascade_metrics = score(case, cascade, chunks_by_id)
        visible = visible_years_by_ticker(documents, cutoff)
        future_obligations = sum(
            int(year) not in visible.get(str(leg["ticker"]), set())
            for leg in legs
            for year in leg["years"]
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "template": case["template"],
                "entity_recall": entity_metrics["recall"],
                "entity_precision": entity_metrics["precision"],
                "entity_exact": entity_metrics["exact"],
                "obligation_recall": obligation_metrics["recall"],
                "obligation_precision": obligation_metrics["precision"],
                "obligation_exact": obligation_metrics["exact"],
                "obligation_covers_gold": obligation_metrics["covers_gold"],
                "cascade_leg_recall": cascade_metrics["leg_recall"],
                "cascade_closure": cascade_metrics["closure"],
                "cascade_wrong_doc_rate": cascade_metrics["wrong_doc_rate"],
                "future_obligations": future_obligations,
                "queries": cascade_metrics["queries"],
            }
        )
        predictions.append(
            {"case_id": case["case_id"], "predicted_legs": legs, "temporal_reasons": reasons, "retrieval": cascade}
        )
    fields = tuple(key for key in rows[0] if key not in {"case_id", "template"})
    result = {
        "schema": "finplan-nonoracle-obligation-cascade.v3",
        "status": "post-validation3-error-analysis-development-not-confirmatory",
        "protocol": {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases": len(data["cases"]),
            "planner": "cutoff-visible registry + canonical entity coverage + explicit temporal obligations",
            "gold_fields_hidden_from_planner": ["candidate_legs", "gold_doc_ids", "gold_pages", "reference_answer"],
            "budget": budget,
        },
        "interpretation_boundary": [
            "This revision was designed from validation3 errors; validation3 is now development data.",
            "A fresh frozen set is required for confirmatory comparison.",
            "The experiment evaluates document obligations and closure, not final answer correctness.",
        ],
        "summary": aggregate(rows, fields),
        "rows": rows,
        "predictions": predictions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_ROOT / "lofin_multidoc_validation3_v1.json")
    parser.add_argument("--registry", default=DATA_ROOT / "sec_company_registry_validation3_v1.json")
    parser.add_argument("--out", default=RESULTS_ROOT / "nonoracle_obligation_v5_validation3_development.json")
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    result = run(Path(args.data), Path(args.registry), Path(args.out), args.budget)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
