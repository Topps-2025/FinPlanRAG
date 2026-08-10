"""FinPlan v4 candidate-coverage planner after v1 non-oracle failure analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set

try:
    from .storage_paths import DATA_ROOT, RESULTS_ROOT
except ImportError:
    from storage_paths import DATA_ROOT, RESULTS_ROOT

from run_lofin_multidoc_pilot import BM25, make_chunks, run_method, score
from run_nonoracle_obligation_planning import YEAR_RE, aggregate, normalize, obligations, prf


GENERIC_NAME_TOKENS = {
    "and", "co", "company", "corp", "corporation", "group", "holding", "holdings",
    "inc", "incorporated", "industries", "international", "ltd", "md", "nv", "plc",
    "trust", "worldwide", "global", "communications", "health", "energy",
    "under", "united", "state", "states",
}


def stem_token(token: str) -> str:
    token = token.strip("'")
    if token.endswith("'s"):
        token = token[:-2]
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        token = token[:-1]
    return token


def content_tokens(value: str) -> Set[str]:
    return {
        stem_token(token)
        for token in normalize(value).split()
        if stem_token(token) and stem_token(token) not in GENERIC_NAME_TOKENS
    }


def candidate_coverage_rule(question: str, companies: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    normalized = normalize(question)
    normalized = re.sub(r"\bjp\s+morgan\b", "jpmorgan", normalized)
    question_tokens = content_tokens(normalized)
    upper_tokens = set(re.findall(r"\b[A-Z][A-Z0-9.\-]{1,5}\b", question))

    company_tokens: Dict[str, List[Set[str]]] = {
        str(company["ticker"]): [content_tokens(str(alias)) for alias in company["aliases"]]
        for company in companies
    }
    token_df = Counter()
    for token_sets in company_tokens.values():
        token_df.update(set().union(*token_sets) if token_sets else set())

    matched: List[Mapping[str, object]] = []
    for company in companies:
        ticker = str(company["ticker"])
        exact_alias = any(
            len(str(alias)) >= 3
            and re.search(rf"(?<![a-z0-9]){re.escape(str(alias))}(?![a-z0-9])", normalized)
            for alias in company["aliases"]
        )
        ticker_hit = ticker in upper_tokens
        rare_overlap = False
        for alias_tokens in company_tokens[ticker]:
            overlap = alias_tokens & question_tokens
            full_name_coverage = bool(alias_tokens) and alias_tokens <= question_tokens
            distinctive_overlap = overlap and any(token_df[token] == 1 and len(token) >= 4 for token in overlap)
            if full_name_coverage or distinctive_overlap:
                rare_overlap = True
                break
        if exact_alias or ticker_hit or rare_overlap:
            matched.append(company)

    years = sorted({int(year) for year in YEAR_RE.findall(question)})
    filing_mentions = [
        int(year)
        for year in re.findall(r"\b(20\d{2})\b[^.]{0,30}\b10\s*-?\s*k\b", question, flags=re.I)
    ]
    plans: List[Dict[str, object]] = []
    for company in matched:
        available = sorted(int(year) for year in company["available_years"])
        if filing_mentions:
            selected = [year for year in filing_mentions if year in available]
        elif len(years) >= 2 and re.search(r"\b(from|between|over|trend|change)\b", normalized):
            selected = [year for year in available if min(years) <= year <= max(years)]
        else:
            selected = [year for year in years if year in available]
        if not selected and available:
            selected = [max(available)]
        plans.append({"ticker": str(company["ticker"]), "years": sorted(set(selected))})
    return sorted(plans, key=lambda leg: str(leg["ticker"]))


def run(data_path: Path, registry_path: Path, out_path: Path, budget: int) -> Mapping[str, object]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    companies = registry["companies"]
    chunks = make_chunks(data["documents"])
    index = BM25(chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rows: List[Dict[str, object]] = []
    predictions: List[Dict[str, object]] = []
    for case in data["cases"]:
        legs = candidate_coverage_rule(str(case["question"]), companies)
        gold_docs = set(str(doc) for doc in case["gold_doc_ids"])
        gold_entities = {doc.split("_")[0] for doc in gold_docs}
        predicted_docs = obligations(legs)
        predicted_entities = {str(leg["ticker"]) for leg in legs}
        entity_metrics = prf(predicted_entities, gold_entities)
        obligation_metrics = prf(predicted_docs, gold_docs)
        planned_case = dict(case)
        planned_case["candidate_legs"] = legs
        cascade = run_method(planned_case, index, budget, "finplan_v3_path_bound")
        cascade_metrics = score(case, cascade, chunks_by_id)
        row = {
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
            "queries": cascade_metrics["queries"],
        }
        rows.append(row)
        predictions.append({"case_id": case["case_id"], "predicted_legs": legs, "retrieval": cascade})
    fields = tuple(key for key in rows[0] if key not in {"case_id", "template"})
    result = {
        "schema": "finplan-nonoracle-obligation-cascade.v2",
        "status": "development-split-architecture-revision-not-confirmatory",
        "protocol": {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases": len(data["cases"]),
            "planner": "rare-token candidate coverage plus explicit filing-year obligations",
            "gold_fields_hidden_from_planner": ["candidate_legs", "gold_doc_ids", "gold_pages", "reference_answer"],
            "budget": budget,
        },
        "summary": aggregate(rows, fields),
        "rows": rows,
        "predictions": predictions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_ROOT / "lofin_multidoc_validation2_v1.json")
    parser.add_argument("--registry", default=DATA_ROOT / "sec_company_registry_validation2_v1.json")
    parser.add_argument("--out", default=RESULTS_ROOT / "nonoracle_obligation_planning_v2_development.json")
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    result = run(Path(args.data), Path(args.registry), Path(args.out), args.budget)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
