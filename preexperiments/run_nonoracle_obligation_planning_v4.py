"""FinPlan v6 development: financial alias equivalence and temporal boundaries.

The revision is based on frozen validation4 failures.  Results on validation4
are developmental; a new mixed-filing set is required for confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

try:
    from .storage_paths import DATA_ROOT, RESULTS_ROOT
except ImportError:
    from storage_paths import DATA_ROOT, RESULTS_ROOT

from run_lofin_multidoc_pilot import BM25, make_chunks, run_method, score
from run_nonoracle_obligation_planning import aggregate, obligations, prf
from run_nonoracle_obligation_planning_v3 import (
    WINDOW_WORDS,
    asof_companies,
    canonical_text,
    explicit_years,
    match_companies as match_companies_v5,
    select_years as select_years_v5,
    visible_years_by_ticker,
)


def parse_number(raw: str) -> int:
    return int(raw) if raw.isdigit() else WINDOW_WORDS[raw]


def temporal_boundary_request(question: str) -> Tuple[str, int] | None:
    normalized = canonical_text(question)
    recent = re.search(
        r"\brecent\s+(\d+|two|three|four|five|six)[ -]year\s+period\b",
        normalized,
    )
    if recent:
        return "recent-period", parse_number(recent.group(1))
    ago = re.search(r"\b(\d+|two|three|four|five|six)\s+years?\s+ago\b", normalized)
    if ago:
        return "years-ago", parse_number(ago.group(1))
    return None


def nearest_year(available: Sequence[int], target: int) -> int:
    return min(available, key=lambda year: (abs(year - target), year))


def select_years(question: str, available: Sequence[int]) -> Tuple[List[int], str]:
    boundary = temporal_boundary_request(question)
    if not boundary:
        return select_years_v5(question, available)
    if not available:
        return [], "no-visible-filing"
    kind, span = boundary
    years, _ = explicit_years(question)
    anchor = max(years) if years else max(available)
    target = anchor - (span - 1 if kind == "recent-period" else span)
    selected = sorted({nearest_year(available, target), nearest_year(available, anchor)})
    return selected, f"temporal-boundaries:{kind}"


def match_companies(question: str, companies: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    matched = {str(company["ticker"]): company for company in match_companies_v5(question, companies)}
    squashed_question = canonical_text(question).replace(" ", "")
    for company in companies:
        ticker = str(company["ticker"])
        if ticker in matched:
            continue
        for alias in company["aliases"]:
            squashed_alias = canonical_text(str(alias)).replace(" ", "")
            if len(squashed_alias) >= 6 and squashed_alias in squashed_question:
                matched[ticker] = company
                break
    return [matched[ticker] for ticker in sorted(matched)]


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
        "schema": "finplan-nonoracle-obligation-cascade.v4",
        "status": "post-validation4-error-analysis-development-not-confirmatory",
        "protocol": {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases": len(data["cases"]),
            "planner": "v5 + squashed financial aliases + relative temporal boundary obligations",
            "gold_fields_hidden_from_planner": ["candidate_legs", "gold_doc_ids", "gold_pages", "reference_answer"],
            "budget": budget,
        },
        "interpretation_boundary": [
            "This revision was designed from frozen validation4 errors; validation4 is development data for v6.",
            "A fresh mixed-filing frozen set is required for confirmation.",
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
    parser.add_argument("--data", default=DATA_ROOT / "lofin_multidoc_validation4_v1.json")
    parser.add_argument("--registry", default=DATA_ROOT / "sec_company_registry_validation4_v1.json")
    parser.add_argument("--out", default=RESULTS_ROOT / "nonoracle_obligation_v6_validation4_development.json")
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    result = run(Path(args.data), Path(args.registry), Path(args.out), args.budget)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
