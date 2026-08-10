"""FinPlan v9 (runner file v7): v8 + fiscal-year-N parsing + first-n fix.

Pre-declared in the validation8 manifest (lofin_validation8_manifest.json)
before any method ran, as the two resolution changes carried over from the
validation7 negative results:
  (a) "fiscal year N" / "FY N" -> annual obligation (N, 10-K, FY).  The
      validation7 fy_q1n stratum was 0/3 because the phrase never entered
      the obligation key; this pattern closes that gap.
  (b) "first n quarters of Y" -> only the cumulative filing (Y, Qn).  The
      YTD figure for the first n quarters is physically reported in the Qn
      10-Q only (nine months in Q3, six months in Q2); requesting Q1..Qn
      over-retrieved Q1 on validation7's h1_9m stratum (Q1 10-Q carries the
      standalone Q1 figure, not any cumulative one).

The runner also implements the pre-declared P4 honesty comparator
(metadata_v9_fill): the exact same v9 obligations with the metadata-
decomposition retrieval policy (fill_missing loop, no cascade, same budget).
If it matches v9's closure, the paper's strategy claim must shrink to the
representation layer.
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

from run_lofin_multidoc_pilot import split_words
from run_nonoracle_obligation_planning import aggregate, prf
from run_nonoracle_obligation_planning_v4 import match_companies, select_years
from run_nonoracle_obligation_planning_v6 import (
    make_period_chunks,
    obligation_signature,
    precise_path,
    visible_filings,
)
from run_sec_real_pilot import BM25, Chunk, parse_time, tokens

METHODS = ("finplan_v9_cascade", "metadata_v9_fill")

ORDINAL_QUARTERS = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}
COUNT_WORDS = {"two": 2, "three": 3, "four": 4}


def period_requests(question: str) -> Set[Tuple[int, str]]:
    requests: Set[Tuple[int, str]] = set()
    for first, second, year in re.findall(
        r"\bQ([1-4])\s+and\s+Q([1-4])\s+of\s+(?:FY\s*|fiscal\s+year\s+)?(20\d{2})\b",
        question,
        flags=re.I,
    ):
        requests.update({(int(year), f"Q{first}"), (int(year), f"Q{second}")})
    for quarter, year in re.findall(
        r"\bQ([1-4])\s+(?:of\s+)?(?:FY\s*|fiscal\s+year\s+)?(20\d{2})\b",
        question,
        flags=re.I,
    ):
        requests.add((int(year), f"Q{quarter}"))
    for ordinal, year in re.findall(
        r"\b(first|second|third|fourth)\s+quarter\s+(?:of\s+)?(?:FY\s*|fiscal\s+year\s+)?(20\d{2})\b",
        question,
        flags=re.I,
    ):
        requests.add((int(year), ORDINAL_QUARTERS[ordinal.lower()]))
    for count, year in re.findall(
        r"\bfirst\s+(two|three|four|\d+)\s+quarters\s+of\s+(?:FY\s*|fiscal\s+year\s+)?(20\d{2})\b",
        question,
        flags=re.I,
    ):
        # (b) first-n fix: the YTD figure for "the first n quarters" is
        # reported only in the Qn 10-Q (nine months -> Q3, six months -> Q2).
        # Earlier quarters carry standalone (non-cumulative) figures and are
        # not obligations for this phrase.
        n = int(count) if count.isdigit() else COUNT_WORDS[count.lower()]
        requests.add((int(year), f"Q{min(n, 4)}"))
    for half, year in re.findall(
        r"\b(first|second)\s+half\s+of\s+(?:FY\s*|fiscal\s+year\s+)?(20\d{2})\b",
        question,
        flags=re.I,
    ):
        if half.lower() == "first":
            # six months ended Jun 30 are reported in the Q2 10-Q only
            requests.add((int(year), "Q2"))
        else:
            # H2 = FY - 9M: nine months in the Q3 10-Q, annual in the FY 10-K
            requests.add((int(year), "Q3"))
            requests.add((int(year), "Q4"))
    # (a) fiscal-year-N parsing: "fiscal year N" / "FY N" -> the annual
    # figure is reported in the FY 10-K.  The Q4 signature collapses to the
    # same (N, 10-K, FY) obligation, so overlaps with explicit Q4 requests
    # deduplicate.
    for year in re.findall(r"\bfiscal\s+year\s+(20\d{2})\b", question, flags=re.I):
        requests.add((int(year), "Q4"))
    for year in re.findall(r"\bFY\s*(20\d{2})\b", question, flags=re.I):
        requests.add((int(year), "Q4"))
    if requests and re.search(r"\bsame\s+period\s+(?:in\s+the\s+)?(?:previous|prior|last)\s+year\b", question, re.I):
        requests.update((year - 1, period) for year, period in list(requests))
    return requests


def filing_obligation_rule(
    question: str,
    cutoff: str,
    companies: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], Mapping[str, str]]:
    matched = match_companies(question, companies)
    requested_periods = period_requests(question)
    obligations: List[Dict[str, object]] = []
    reasons: Dict[str, str] = {}
    for company in matched:
        ticker = str(company["ticker"])
        filings = visible_filings(company, cutoff)
        by_signature = {
            (int(item["fiscal_year"]), str(item["filing_type"]), str(item["fiscal_period"])): item
            for item in filings
        }
        selected: List[Mapping[str, object]] = []
        if requested_periods:
            for year, period in sorted(requested_periods):
                filing = by_signature.get(obligation_signature(year, period))
                if filing is not None:
                    selected.append(filing)
            reasons[ticker] = "explicit-fiscal-periods"
        else:
            annual = sorted(
                int(item["fiscal_year"])
                for item in filings
                if item["filing_type"] == "10-K" and item["fiscal_period"] == "FY"
            )
            for year in select_years(question, annual)[0]:
                filing = by_signature.get((year, "10-K", "FY"))
                if filing is not None:
                    selected.append(filing)
            reasons[ticker] = "annual-fallback"
        for filing in selected:
            obligations.append(
                {
                    "ticker": ticker,
                    "fiscal_year": int(filing["fiscal_year"]),
                    "filing_type": str(filing["filing_type"]),
                    "fiscal_period": str(filing["fiscal_period"]),
                    "doc_id": str(filing["doc_id"]),
                }
            )
    unique = {str(item["doc_id"]): item for item in obligations}
    return [unique[key] for key in sorted(unique)], reasons


def retrieve_obligations(
    question: str,
    cutoff: str,
    obligations: Sequence[Mapping[str, object]],
    index: BM25,
    budget: int,
    method: str,
) -> Mapping[str, object]:
    used: List[str] = []
    used_docs: List[str] = []
    actions: List[str] = []

    def obligation_query(obligation: Mapping[str, object]) -> str:
        return (
            f"{question} {obligation['ticker']} {obligation['fiscal_year']} "
            f"{obligation['fiscal_period']} {obligation['filing_type']}"
        )

    if method == "finplan_v9_cascade":
        for obligation in obligations:
            if len(used_docs) >= budget:
                break
            candidates = index.search(
                obligation_query(obligation),
                parse_time(cutoff),
                used,
                False,
                limit=1,
                required_terms=[precise_path(obligation)],
            )
            actions.append(
                f"obligation:{obligation['ticker']}:{obligation['fiscal_year']}:"
                f"{obligation['fiscal_period']}:{obligation['filing_type']}"
            )
            if candidates:
                used.append(candidates[0].chunk_id)
                used_docs.append(candidates[0].doc_id)
    elif method == "metadata_v9_fill":
        # P4 comparator: same obligations, metadata-decomposition retrieval
        # policy (fill_missing only, no initial query, no cascade actions).
        for obligation in obligations:
            if len(used_docs) >= budget:
                break
            if str(obligation["doc_id"]) in used_docs:
                continue
            candidates = index.search(
                obligation_query(obligation),
                parse_time(cutoff),
                used,
                False,
                limit=1,
                required_terms=[precise_path(obligation)],
            )
            actions.append(f"missing:{obligation['doc_id']}")
            if candidates:
                used.append(candidates[0].chunk_id)
                used_docs.append(candidates[0].doc_id)
    else:
        raise ValueError(method)
    return {"used": used, "used_docs": used_docs, "actions": actions}


def run(data_path: Path, registry_path: Path, out_path: Path, budget: int) -> Mapping[str, object]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    chunks = make_period_chunks(data["documents"])
    index = BM25(chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rows: List[Dict[str, object]] = []
    predictions: List[Dict[str, object]] = []
    for case in data["cases"]:
        cutoff = str(case["cutoff"])
        planned, reasons = filing_obligation_rule(str(case["question"]), cutoff, registry["companies"])
        predicted_docs = {str(item["doc_id"]) for item in planned}
        gold_docs = {str(doc) for doc in case["gold_doc_ids"]}
        predicted_entities = {str(item["ticker"]) for item in planned}
        gold_entities = {doc.split("_")[0] for doc in gold_docs}
        entity_metrics = prf(predicted_entities, gold_entities)
        obligation_metrics = prf(predicted_docs, gold_docs)
        for method in METHODS:
            retrieval = retrieve_obligations(str(case["question"]), cutoff, planned, index, budget, method)
            used_docs = set(retrieval["used_docs"])
            overlap = len(used_docs & gold_docs)
            future = float(
                any(chunks_by_id[chunk_id].available_at > parse_time(cutoff) for chunk_id in retrieval["used"])
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "template": case["template"],
                    "method": method,
                    "entity_recall": entity_metrics["recall"],
                    "entity_precision": entity_metrics["precision"],
                    "entity_exact": entity_metrics["exact"],
                    "obligation_recall": obligation_metrics["recall"],
                    "obligation_precision": obligation_metrics["precision"],
                    "obligation_exact": obligation_metrics["exact"],
                    "obligation_covers_gold": obligation_metrics["covers_gold"],
                    "cascade_leg_recall": overlap / len(gold_docs),
                    "cascade_closure": float(gold_docs <= used_docs),
                    "cascade_wrong_doc_rate": len(used_docs - gold_docs) / max(1, len(used_docs)),
                    "future_leak": future,
                    "queries": float(len(retrieval["actions"])),
                }
            )
        predictions.append(
            {
                "case_id": case["case_id"],
                "predicted_obligations": planned,
                "reasons": reasons,
                "retrieval_by_method": {
                    method: retrieve_obligations(str(case["question"]), cutoff, planned, index, budget, method)
                    for method in METHODS
                },
            }
        )
    fields = tuple(key for key in rows[0] if key not in {"case_id", "template", "method"})
    result = {
        "schema": "finplan-nonoracle-filing-obligation-v9.v1",
        "status": "post-validation8-frozen-not-confirmatory",
        "protocol": {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases": len(data["cases"]),
            "planner": "v9 = v8 + fiscal-year-N parsing + first-n-quarters cumulative-only; aliases default",
            "gold_fields_hidden_from_planner": ["candidate_legs", "candidate_obligations", "gold_doc_ids", "gold_pages", "reference_answer"],
            "budget": budget,
        },
        "interpretation_boundary": [
            "v9 changes pre-declared in the validation8 manifest before any method ran; validation7's fy_q1n 0/3 and h1_9m over-retrieval are the documented failure modes being confirmed or refuted here.",
            "metadata_v9_fill is the pre-declared P4 honesty comparator: identical obligations, metadata-decomposition retrieval policy (fill_missing, no cascade), same budget.",
            "The experiment evaluates filing closure, not page evidence or final answers.",
        ],
        "summary": aggregate(rows, fields),
        "by_method": {method: aggregate([row for row in rows if row["method"] == method], fields) for method in METHODS},
        "rows": rows,
        "predictions": predictions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_ROOT / "lofin_validation8_v1.json")
    parser.add_argument("--registry", default=DATA_ROOT / "sec_filing_registry_validation8_v2.json")
    parser.add_argument("--out", default=RESULTS_ROOT / "nonoracle_obligation_v9_validation8_frozen.json")
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    result = run(Path(args.data), Path(args.registry), Path(args.out), args.budget)
    print(json.dumps(result["by_method"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
