"""A4 (–Available time) ablation on the frozen validation7 set.

Pre-declared 2026-08-11, before any run (see
lofin_validation7_ablation_time_freeze.json for the frozen hashes and the
pre-declared predictions P1–P4).

What is ablated: the `available_at <= cutoff` constraint, at every point it
appears in each method's pipeline.
  - planning stage: `visible_filings(company, cutoff)` is called with
    cutoff=None (all filings visible);
  - retrieval stage: `index.search(..., allow_future=False, ...)` becomes
    allow_future=True.

What is NOT changed: corpus, index construction, templates, budget,
obligation/period parsing rules, BM25 scoring.  The `time` variant replays
the exact frozen pipeline and must reproduce the frozen results byte-for-byte
(determinism check, reported in the output).

Methods (identical to the frozen validation7 runs):
  - single_shot / period_metadata_decomposition / generic_adaptive_period /
    hirec_period: shared frozen v7 planner + v5 index (see
    run_period_aware_baselines.py);
  - finplan_v8_cascade: v8 planner (half-year aliases) + v6 index + path-bound
    obligation retrieval (see run_nonoracle_obligation_planning_v6.py).

Rows gain three future-leak fields that the frozen baseline rows lacked:
  - future_leak: 1 if any used chunk's available_at is strictly after cutoff;
  - future_leak_docs: number of used docs that are future;
  - future_leak_rate: future_leak_docs / len(used_docs).

Pre-declared predictions (recorded in the freeze file before running):
  P1  finplan_v8_cascade: no_time ≡ time on closure and future_leak (0/14 for
      both variants) — precise_path binding makes the retrieval time gate
      redundant, and no question references a period that exists only in
      future filings, so planning-gate removal is a no-op.
  P2  single_shot: no_time shows >= 1 future_leak case — the Q1-2025 10-Qs
      carry Q1-2024 comparatives and FY-2024 data that match the question
      terms; closure may change in either direction (measured, not predicted).
  P3  period_metadata_decomposition: no_time ≡ time — all of its retrieval is
      required-terms (path-bound) fill_missing, so no future doc can be
      returned.
  P4  generic/hirec: future leak possible only through the unbounded initial
      query; fill_missing stays path-bound (measured).
"""

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
    filing_obligation_rule as v5_filing_obligation_rule,
    make_period_chunks as v5_make_period_chunks,
    precise_path,
)
from run_nonoracle_obligation_planning_v6 import (
    filing_obligation_rule as v8_filing_obligation_rule,
    make_period_chunks as v8_make_period_chunks,
)
from run_sec_real_pilot import BM25, parse_time


METHODS = (
    "single_shot",
    "period_metadata_decomposition",
    "generic_adaptive_period",
    "hirec_period",
    "finplan_v8_cascade",
)
VARIANTS = ("time", "no_time")


def retrieve_policy(
    question: str,
    cutoff: str,
    obligations: Sequence[Mapping[str, object]],
    index: BM25,
    budget: int,
    method: str,
    no_time: bool,
) -> Mapping[str, object]:
    """Byte-identical to run_period_aware_baselines.retrieve_policy except
    that the search-time gate is allow_future=no_time."""
    used_chunks: List[str] = []
    used_docs: List[str] = []
    actions: List[str] = []

    def initial(k: int) -> None:
        actions.append("initial-query")
        candidates = index.search(
            question,
            parse_time(cutoff),
            used_chunks,
            no_time,
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
                no_time,
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


def retrieve_obligations_cascade(
    question: str,
    cutoff: str,
    obligations: Sequence[Mapping[str, object]],
    index: BM25,
    budget: int,
    no_time: bool,
) -> Mapping[str, object]:
    """Byte-identical to run_nonoracle_obligation_planning_v6.retrieve_obligations
    except that the search-time gate is allow_future=no_time."""
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
            no_time,
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


def run_case(
    case: Mapping[str, object],
    index_v5: BM25,
    index_v8: BM25,
    registry: Mapping[str, object],
    budget: int,
    no_time: bool,
) -> List[Dict[str, object]]:
    question = str(case["question"])
    cutoff = str(case["cutoff"])
    gold = {str(doc) for doc in case["gold_doc_ids"]}
    planning_cutoff = None if no_time else cutoff
    rows: List[Dict[str, object]] = []
    for method in METHODS:
        if method == "finplan_v8_cascade":
            planned, _ = v8_filing_obligation_rule(question, planning_cutoff, registry["companies"])
            index = index_v8
            prediction = retrieve_obligations_cascade(
                question, cutoff, planned, index, budget, no_time
            )
            planner = "v8"
        else:
            planned, _ = v5_filing_obligation_rule(question, planning_cutoff, registry["companies"])
            index = index_v5
            prediction = retrieve_policy(question, cutoff, planned, index, budget, method, no_time)
            planner = "v7-shared"
        used_docs = list(prediction["used_docs"])
        used_set = set(used_docs)
        cutoff_dt = parse_time(cutoff)
        chunks_by_id = {c.chunk_id: c for c in (index_v5.chunks + index_v8.chunks)}
        future_used = [
            cid for cid in prediction["used"] if chunks_by_id[cid].available_at > cutoff_dt
        ]
        future_doc_ids = sorted({cid.split("::")[0] for cid in future_used})
        rows.append(
            {
                "case_id": case["case_id"],
                "stratum": case.get("stratum", "unstratified"),
                "method": method,
                "variant": "no_time" if no_time else "time",
                "planner": planner,
                "leg_recall": len(gold & used_set) / len(gold),
                "closure": float(gold <= used_set),
                "wrong_doc_rate": len(used_set - gold) / max(1, len(used_set)),
                "future_leak": float(len(future_used) > 0),
                "future_leak_docs": float(len(future_doc_ids)),
                "future_leak_rate": len(future_doc_ids) / max(1, len(used_docs)),
                "queries": float(len(prediction["actions"])),
                "documents": float(len(used_docs)),
                "planned_obligations": [str(o["doc_id"]) for o in planned],
            }
        )
    return rows


def run(
    data_path: Path,
    registry_path: Path,
    out_path: Path,
    budget: int,
    frozen_baselines: Path,
    frozen_v8: Path,
    only_cases: Sequence[str] = (),
) -> Mapping[str, object]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if only_cases:
        wanted = set(only_cases)
        data = {
            **data,
            "cases": [case for case in data["cases"] if str(case["case_id"]) in wanted],
        }
    chunks_v5 = v5_make_period_chunks(data["documents"])
    chunks_v8 = v8_make_period_chunks(data["documents"])
    index_v5 = BM25(chunks_v5)
    index_v8 = BM25(chunks_v8)
    rows: List[Dict[str, object]] = []
    for case in data["cases"]:
        for no_time in (False, True):
            rows.extend(run_case(case, index_v5, index_v8, registry, budget, no_time))

    # Determinism check: the `time` variant must reproduce the frozen runs.
    frozen_b = json.loads(frozen_baselines.read_text(encoding="utf-8"))
    frozen_v = json.loads(frozen_v8.read_text(encoding="utf-8"))
    discrepancies: List[str] = []
    for row in rows:
        if row["variant"] != "time":
            continue
        if row["method"] == "finplan_v8_cascade":
            fr = next(r for r in frozen_v["rows"] if r["case_id"] == row["case_id"])
            expected = {
                "closure": fr["cascade_closure"],
                "wrong_doc_rate": fr["cascade_wrong_doc_rate"],
                "queries": fr["queries"],
            }
            for field, exp in expected.items():
                if abs(float(row[field]) - float(exp)) > 1e-9:
                    discrepancies.append(
                        f"{row['case_id']}/{row['method']}/{field}: {row[field]} != {exp}"
                    )
        else:
            fr = next(
                r
                for r in frozen_b["rows"]
                if r["case_id"] == row["case_id"] and r["method"] == row["method"]
            )
            for field in ("closure", "wrong_doc_rate", "queries", "documents"):
                if abs(float(row[field]) - float(fr[field])) > 1e-9:
                    discrepancies.append(
                        f"{row['case_id']}/{row['method']}/{field}: {row[field]} != {fr[field]}"
                    )

    fields = ("leg_recall", "closure", "wrong_doc_rate", "future_leak", "future_leak_rate", "queries", "documents")
    summary = {
        method: {
            variant: aggregate(
                [row for row in rows if row["method"] == method and row["variant"] == variant],
                fields,
            )
            for variant in VARIANTS
        }
        for method in METHODS
    }
    result = {
        "schema": "finplan-validation7-ablation-available-time.v1",
        "status": "pre-declared-ablation-not-confirmatory",
        "protocol": {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "budget": budget,
            "ablation": "A4 (-Available time): available_at <= cutoff removed at planning "
            "(visible_filings cutoff=None) and retrieval (allow_future=True); nothing else changes",
        },
        "interpretation_boundary": [
            "future_leak is strict: available_at > cutoff, so fy_q1n gold filings dated exactly at cutoff are not leaks.",
            "The `time` variant is a determinism replay of the frozen runs, not new evidence.",
            "The ablation removes a safety constraint, so worse no_time results are expected for some methods; "
            "the question is which methods lose the protection and how much.",
        ],
        "determinism_vs_frozen": {"discrepancy_count": len(discrepancies), "discrepancies": discrepancies[:20]},
        "summary": summary,
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_ROOT / "lofin_validation7_v1.json")
    parser.add_argument("--registry", default=DATA_ROOT / "sec_filing_registry_validation7_v2.json")
    parser.add_argument("--out", default=RESULTS_ROOT / "validation7_ablation_time_v1.json")
    parser.add_argument(
        "--frozen-baselines",
        default=RESULTS_ROOT / "period_aware_baselines_validation7.json",
    )
    parser.add_argument("--frozen-v8", default=RESULTS_ROOT / "nonoracle_obligation_v8_validation7_frozen.json")
    parser.add_argument("--cases", default="")
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    result = run(
        Path(args.data),
        Path(args.registry),
        Path(args.out),
        args.budget,
        Path(args.frozen_baselines),
        Path(args.frozen_v8),
        only_cases=[c.strip() for c in args.cases.split(",") if c.strip()],
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
