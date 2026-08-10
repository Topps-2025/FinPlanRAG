"""Run document-level retrieval planning on real LOFin-linked SEC 10-Ks.

The benchmark's reference answer is deliberately not used to manufacture a
reader score. The primary metrics are evidence-leg coverage and closure. An
oracle reader can only be counted correct when all annotated documents are
retrieved. This isolates planning while keeping the experiment honest about
the absence of page-aligned spans in the SEC HTML replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

try:
    from .storage_paths import DATA_ROOT, RESULTS_ROOT
except ImportError:
    from storage_paths import DATA_ROOT, RESULTS_ROOT

from run_sec_real_pilot import BM25, Chunk, parse_time, tokens


METHODS = (
    "single_shot",
    "fixed_decomposition",
    "metadata_decomposition",
    "generic_adaptive",
    "generic_path_bound",
    "hirec_style",
    "hirec_path_bound",
    "finplan_v1",
    "finplan_v3_no_path",
    "finplan_v3_path_bound",
    "finplan_no_time",
)


def split_words(text: str, size: int = 180, stride: int = 135) -> List[str]:
    words = text.split()
    return [" ".join(words[start : start + size]) for start in range(0, len(words), stride) if words[start : start + size]]


def make_chunks(documents: Sequence[Mapping[str, object]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in documents:
        # Structured filing identity used only by the path-bound ablation.
        # It is derived from public document metadata, not from answer text.
        path_key = f"path{str(doc['ticker']).lower()}{int(doc['year'])}"
        for i, text in enumerate(split_words(str(doc["text"]))):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['doc_id']}::c{i}",
                    doc_id=str(doc["doc_id"]),
                    lineage_id=str(doc["doc_id"]),
                    role="10-K",
                    available_at=parse_time(str(doc["available_at"])),
                    text=text,
                    terms=tuple(tokens(text)) + (path_key,),
                )
            )
    return chunks


def query_for(case: Mapping[str, object], leg: Mapping[str, object] | None = None, template: str = "") -> str:
    base = str(case["question"])
    if leg is None:
        return base
    years = " ".join(str(year) for year in leg["years"])
    # The question already carries the task-specific accounting/comparison
    # concept.  A shared generic suffix was tested on the exploratory split
    # and harmed retrieval, so the cross-track interface shares state fields
    # without forcing one query vocabulary on every financial template.
    return f"{base} {leg['ticker']} {years}"


def obligation_legs(legs: Sequence[Mapping[str, object]]) -> List[Tuple[Mapping[str, object], int]]:
    """Expand entity legs into independently auditable entity-year obligations."""
    return [(leg, int(year)) for leg in legs for year in leg["years"]]


def obligation_doc_id(leg: Mapping[str, object], year: int) -> str:
    return f"{leg['ticker']}_{year}_10K"


def obligation_covered(leg: Mapping[str, object], year: int, observed_docs: Sequence[str]) -> bool:
    return obligation_doc_id(leg, year) in set(observed_docs)


def leg_doc_ids(case: Mapping[str, object]) -> Dict[str, str]:
    return {
        f"{leg['ticker']}::{year}": f"{leg['ticker']}_{year}_10K"
        for leg in case["candidate_legs"]
        for year in leg["years"]
    }


def run_method(case: Mapping[str, object], index: BM25, budget: int, method: str) -> Dict[str, object]:
    cutoff = parse_time(str(case["cutoff"]))
    allow_future = method == "finplan_no_time"
    used: List[str] = []
    actions: List[str] = []
    legs = list(case["candidate_legs"])
    obligations = obligation_legs(legs)
    observed_docs: List[str] = []
    v3_family = method in {"finplan_v3_no_path", "finplan_v3_path_bound", "finplan_no_time"}
    path_bound = method in {
        "metadata_decomposition",
        "generic_path_bound",
        "hirec_path_bound",
        "finplan_v3_path_bound",
        "finplan_no_time",
    }

    def retrieve(query: str, label: str, required: Sequence[str] = (), k: int = 1) -> None:
        actions.append(label)
        remaining = max(0, budget - len(set(observed_docs)))
        requested = min(k, remaining)
        if not requested:
            return
        # BM25 excludes used passages, not whole filings. Request enough
        # candidates to skip filings already observed in prior actions.
        candidates = index.search(
            query,
            cutoff,
            used,
            allow_future,
            limit=requested + len(set(observed_docs)),
            required_terms=required,
        )
        added = 0
        for chunk in candidates:
            if chunk.doc_id in observed_docs:
                continue
            used.append(chunk.chunk_id)
            observed_docs.append(chunk.doc_id)
            added += 1
            if added >= requested:
                break

    if method == "single_shot":
        retrieve(query_for(case), "full_query", k=budget)
    elif method in {"fixed_decomposition", "metadata_decomposition"}:
        for leg, year in obligations:
            if len(actions) >= budget:
                break
            one_year_leg = {**leg, "years": [year]}
            required = [f"path{str(leg['ticker']).lower()}{year}"] if path_bound else ()
            retrieve(query_for(case, one_year_leg), f"leg:{leg['ticker']}:{year}", required=required)
    elif method in {"generic_adaptive", "generic_path_bound"}:
        retrieve(query_for(case), "full_query")
        for leg, year in obligations:
            if len(actions) >= budget:
                break
            if not obligation_covered(leg, year, observed_docs):
                one_year_leg = {**leg, "years": [year]}
                required = [f"path{str(leg['ticker']).lower()}{year}"] if path_bound else ()
                retrieve(query_for(case, one_year_leg), f"missing:{leg['ticker']}:{year}", required=required)
    elif method in {"hirec_style", "hirec_path_bound"}:
        retrieve(query_for(case), "initial_evidence", k=2)
        for leg, year in obligations:
            if len(actions) >= budget:
                break
            if not obligation_covered(leg, year, observed_docs):
                one_year_leg = {**leg, "years": [year]}
                required = [f"path{str(leg['ticker']).lower()}{year}"] if path_bound else ()
                retrieve(query_for(case, one_year_leg), f"complementary:{leg['ticker']}:{year}", required=required)
    else:
        order: Sequence[object] = legs
        if v3_family:
            # The state tracks independent entity/year obligations before any
            # downstream comparison or accounting derivation.
            order = sorted(obligations, key=lambda item: (str(item[0]["ticker"]), item[1]))
        for item in order:
            if len(actions) >= budget:
                break
            if isinstance(item, tuple):
                leg, year = item
                planned_leg = {**leg, "years": [year]}
            else:
                leg = item
                year = int(leg["years"][0])
                planned_leg = leg
            required = [f"path{str(leg['ticker']).lower()}{year}"] if path_bound else ()
            retrieve(
                query_for(case, planned_leg, method),
                f"obligation:{leg['ticker']}:{year}",
                required=required,
            )
        # Cross-document definition/unit alignment is intentionally excluded:
        # LOFin supplies filing-level gold documents but no separately
        # annotated alignment document, so scoring that action as right or
        # wrong would be unsupported.

    return {"used": used, "used_docs": observed_docs, "actions": actions}


def score(case: Mapping[str, object], prediction: Mapping[str, object], chunks: Mapping[str, Chunk]) -> Dict[str, object]:
    gold = set(case["gold_doc_ids"])
    used_docs = set(prediction["used_docs"])
    covered = len(gold.intersection(used_docs)) / len(gold)
    closure = float(gold.issubset(used_docs))
    wrong = len(used_docs - gold) / max(1, len(used_docs))
    cutoff = parse_time(str(case["cutoff"]))
    future = float(any(chunks[chunk_id].available_at > cutoff for chunk_id in prediction["used"]))
    return {
        "leg_recall": covered,
        "closure": closure,
        "oracle_reader_correct": closure,
        "wrong_doc_rate": wrong,
        "future_leak": future,
        "queries": float(len(prediction["actions"])),
        "documents": float(len(used_docs)),
    }


def aggregate(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    names = ("leg_recall", "closure", "oracle_reader_correct", "wrong_doc_rate", "future_leak", "queries", "documents")
    out = {name: sum(float(row[name]) for row in rows) / max(1, len(rows)) for name in names}
    out.update(
        {
            "n": float(len(rows)),
            "closure_successes": sum(float(row["closure"]) for row in rows),
            "future_leak_cases": sum(float(row["future_leak"]) for row in rows),
        }
    )
    return out


def run(data_path: Path, budget: int, selected_split: str = "all") -> Mapping[str, object]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    chunks = make_chunks(data["documents"])
    index = BM25(chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    cases = [case for case in data["cases"] if selected_split == "all" or case["split"] == selected_split]
    rows: List[Dict[str, object]] = []
    trajectories: List[Dict[str, object]] = []
    for case in cases:
        for method in METHODS:
            prediction = run_method(case, index, budget, method)
            metrics = score(case, prediction, chunks_by_id)
            rows.append({"case_id": case["case_id"], "split": case["split"], "template": case["template"], "method": method, **metrics})
            trajectories.append({"case_id": case["case_id"], "method": method, **prediction})
    summary = {
        split: {method: aggregate([row for row in rows if row["split"] == split and row["method"] == method]) for method in METHODS}
        for split in sorted({str(case["split"]) for case in cases})
    }
    return {
        "schema": "finplan-lofin-multidoc-pilot-results.v1",
        "status": "document-closure-not-full-answer-generation",
        "protocol": {"data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "selected_split": selected_split, "cases": len(cases), "documents": len(data["documents"]), "budget": budget, "chunk_window": 180, "chunk_stride": 135},
        "summary": summary,
        "rows": rows,
        "trajectories": trajectories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_ROOT / "lofin_multidoc_pilot_v1.json")
    parser.add_argument("--out", default=RESULTS_ROOT / "lofin_multidoc_pilot_v1.json")
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--split", choices=("all", "exploratory", "holdout"), default="all")
    args = parser.parse_args()
    result = run(Path(args.data), args.budget, args.split)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
