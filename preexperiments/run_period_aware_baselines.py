"""Transparent same-metadata retrieval-policy baselines for filing obligations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

try:
    from .storage_paths import DATA_ROOT, RESULTS_ROOT
except ImportError:
    from storage_paths import DATA_ROOT, RESULTS_ROOT

from run_nonoracle_obligation_planning import aggregate
from run_nonoracle_obligation_planning_v5 import (
    filing_obligation_rule,
    make_period_chunks,
    precise_path,
)
from run_sec_real_pilot import BM25, parse_time


METHODS = (
    "single_shot",
    "period_metadata_decomposition",
    "generic_adaptive_period",
    "hirec_period",
)


def retrieve_policy(
    question: str,
    cutoff: str,
    obligations: Sequence[Mapping[str, object]],
    index: BM25,
    budget: int,
    method: str,
) -> Mapping[str, object]:
    used_chunks: List[str] = []
    used_docs: List[str] = []
    actions: List[str] = []

    def initial(k: int) -> None:
        actions.append("initial-query")
        candidates = index.search(
            question,
            parse_time(cutoff),
            used_chunks,
            False,
            limit=min(k, budget),
        )
        for chunk in candidates:
            if chunk.doc_id in used_docs:
                continue
            used_chunks.append(chunk.chunk_id)
            used_docs.append(chunk.doc_id)
            if len(used_docs) >= budget:
                break

    def fill_missing() -> None:
        for obligation in obligations:
            if len(used_docs) >= budget:
                break
            if str(obligation["doc_id"]) in used_docs:
                continue
            query = (
                f"{question} {obligation['ticker']} {obligation['fiscal_year']} "
                f"{obligation['fiscal_period']} {obligation['filing_type']}"
            )
            candidates = index.search(
                query,
                parse_time(cutoff),
                used_chunks,
                False,
                limit=1,
                required_terms=[precise_path(obligation)],
            )
            actions.append(f"missing:{obligation['doc_id']}")
            if candidates:
                used_chunks.append(candidates[0].chunk_id)
                used_docs.append(candidates[0].doc_id)

    if method == "single_shot":
        initial(budget)
    elif method == "period_metadata_decomposition":
        fill_missing()
    elif method == "generic_adaptive_period":
        initial(1)
        fill_missing()
    elif method == "hirec_period":
        initial(2)
        fill_missing()
    else:
        raise ValueError(method)
    return {"used": used_chunks, "used_docs": used_docs, "actions": actions}


def run(data_path: Path, registry_path: Path, out_path: Path, budget: int) -> Mapping[str, object]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    chunks = make_period_chunks(data["documents"])
    index = BM25(chunks)
    rows: List[Dict[str, object]] = []
    trajectories: List[Dict[str, object]] = []
    for case in data["cases"]:
        obligations, reasons = filing_obligation_rule(
            str(case["question"]), str(case["cutoff"]), registry["companies"]
        )
        gold = {str(doc) for doc in case["gold_doc_ids"]}
        for method in METHODS:
            prediction = retrieve_policy(
                str(case["question"]), str(case["cutoff"]), obligations, index, budget, method
            )
            used = set(str(doc) for doc in prediction["used_docs"])
            rows.append(
                {
                    "case_id": case["case_id"],
                    "stratum": case.get("stratum", "unstratified"),
                    "method": method,
                    "leg_recall": len(gold & used) / len(gold),
                    "closure": float(gold <= used),
                    "wrong_doc_rate": len(used - gold) / max(1, len(used)),
                    "queries": float(len(prediction["actions"])),
                    "documents": float(len(used)),
                }
            )
            trajectories.append(
                {
                    "case_id": case["case_id"],
                    "method": method,
                    "planned_obligations": obligations,
                    "entity_candidates": sorted(reasons),
                    **prediction,
                }
            )
    summary = {
        method: aggregate(
            [row for row in rows if row["method"] == method],
            ("leg_recall", "closure", "wrong_doc_rate", "queries", "documents"),
        )
        for method in METHODS
    }
    strata = {
        stratum: {
            method: aggregate(
                [row for row in rows if row["method"] == method and row["stratum"] == stratum],
                ("leg_recall", "closure", "wrong_doc_rate", "queries", "documents"),
            )
            for method in METHODS
        }
        for stratum in sorted({str(row["stratum"]) for row in rows})
    }
    result = {
        "schema": "finplan-period-aware-baseline-comparison.v1",
        "status": "transparent-same-metadata-baselines-not-original-hirec",
        "protocol": {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "planner_source": "shared frozen v7 filing_obligation_rule",
            "budget": budget,
        },
        "interpretation_boundary": [
            "All methods share the same non-oracle period-aware obligation planner.",
            "hirec_period is a transparent initial-k plus complementary-query approximation, not original HiREC.",
            "Single-filing and multi-filing strata are reported separately.",
        ],
        "summary": summary,
        "strata": strata,
        "rows": rows,
        "trajectories": trajectories,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_ROOT / "lofin_validation6_v1.json")
    parser.add_argument("--registry", default=DATA_ROOT / "sec_filing_registry_validation6_v2.json")
    parser.add_argument("--out", default=RESULTS_ROOT / "period_aware_baselines_validation6.json")
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    result = run(Path(args.data), Path(args.registry), Path(args.out), args.budget)
    print(json.dumps({"summary": result["summary"], "strata": result["strata"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
