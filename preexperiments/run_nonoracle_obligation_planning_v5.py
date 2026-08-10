"""FinPlan v7 development: filing-type and fiscal-period obligation graph."""

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
from run_sec_real_pilot import BM25, Chunk, parse_time, tokens


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
        n = int(count) if count.isdigit() else COUNT_WORDS[count.lower()]
        requests.update((int(year), f"Q{quarter}") for quarter in range(1, min(n, 4) + 1))
    if requests and re.search(r"\bsame\s+period\s+(?:in\s+the\s+)?(?:previous|prior|last)\s+year\b", question, re.I):
        requests.update((year - 1, period) for year, period in list(requests))
    return requests


def obligation_signature(year: int, period: str) -> Tuple[int, str, str]:
    if period == "Q4":
        return year, "10-K", "FY"
    return year, "10-Q", period


def visible_filings(company: Mapping[str, object], cutoff: str) -> List[Mapping[str, object]]:
    return [
        filing
        for filing in company.get("available_filings", [])
        if not cutoff or str(filing["available_at"])[:10] <= cutoff[:10]
    ]


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


def precise_path(obligation: Mapping[str, object]) -> str:
    form = str(obligation["filing_type"]).replace("-", "").lower()
    return (
        f"path{str(obligation['ticker']).lower()}"
        f"{int(obligation['fiscal_year'])}{str(obligation['fiscal_period']).lower()}{form}"
    )


def make_period_chunks(documents: Sequence[Mapping[str, object]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in documents:
        obligation = {
            "ticker": document["ticker"],
            "fiscal_year": document["year"],
            "filing_type": document.get("form", "10-K"),
            "fiscal_period": document.get("fiscal_period", "FY"),
        }
        generic_path = f"path{str(document['ticker']).lower()}{int(document['year'])}"
        for index, text in enumerate(split_words(str(document["text"]))):
            chunks.append(
                Chunk(
                    chunk_id=f"{document['doc_id']}::c{index}",
                    doc_id=str(document["doc_id"]),
                    lineage_id=str(document["doc_id"]),
                    role=str(document.get("form", "10-K")),
                    available_at=parse_time(str(document["available_at"])),
                    text=text,
                    terms=tuple(tokens(text)) + (generic_path, precise_path(obligation)),
                )
            )
    return chunks


def retrieve_obligations(
    question: str,
    cutoff: str,
    obligations: Sequence[Mapping[str, object]],
    index: BM25,
    budget: int,
) -> Mapping[str, object]:
    used: List[str] = []
    used_docs: List[str] = []
    actions: List[str] = []
    for obligation in obligations:
        if len(used_docs) >= budget:
            break
        query = (
            f"{question} {obligation['ticker']} {obligation['fiscal_year']} "
            f"{obligation['fiscal_period']} {obligation['filing_type']}"
        )
        candidates = index.search(
            query,
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
        retrieval = retrieve_obligations(str(case["question"]), cutoff, planned, index, budget)
        used_docs = set(retrieval["used_docs"])
        overlap = len(used_docs & gold_docs)
        future = float(
            any(chunks_by_id[chunk_id].available_at > parse_time(cutoff) for chunk_id in retrieval["used"])
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
                "cascade_leg_recall": overlap / len(gold_docs),
                "cascade_closure": float(gold_docs <= used_docs),
                "cascade_wrong_doc_rate": len(used_docs - gold_docs) / max(1, len(used_docs)),
                "future_leak": future,
                "queries": float(len(retrieval["actions"])),
            }
        )
        predictions.append(
            {"case_id": case["case_id"], "predicted_obligations": planned, "reasons": reasons, "retrieval": retrieval}
        )
    fields = tuple(key for key in rows[0] if key not in {"case_id", "template"})
    result = {
        "schema": "finplan-nonoracle-filing-obligation-cascade.v5",
        "status": "post-validation5-period-aware-development-not-confirmatory",
        "protocol": {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases": len(data["cases"]),
            "planner": "entity + filing type + fiscal year + fiscal period obligations",
            "gold_fields_hidden_from_planner": ["candidate_legs", "candidate_obligations", "gold_doc_ids", "gold_pages", "reference_answer"],
            "budget": budget,
        },
        "interpretation_boundary": [
            "v7 was designed after validation5 exposed entity-year collision; validation5 is development data for v7.",
            "A new frozen mixed-filing set is required for confirmation.",
            "The experiment evaluates filing closure, not page evidence or final answers.",
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
    parser.add_argument("--data", default=DATA_ROOT / "lofin_mixed_filing_validation5_v1.json")
    parser.add_argument("--registry", default=DATA_ROOT / "sec_filing_registry_validation5_v2.json")
    parser.add_argument("--out", default=RESULTS_ROOT / "nonoracle_obligation_v7_validation5_development.json")
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    result = run(Path(args.data), Path(args.registry), Path(args.out), args.budget)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
