"""LOFin full-set DenseRAG + Hybrid(RRF) retrieval baselines (Part B).

answer_accuracy_dense_hybrid_protocol_v1.json, Part B.  The frozen BM25
benchmark (lofin_full_benchmark_frozen.final.json) is the comparison floor:
before any dense/hybrid row is produced, each group is re-run with the
SHARED BM25 index and the emitted rows must equal the frozen rows bit-for-bit
(parity gate, hard stop on mismatch).  Only after a group passes parity are
DenseIndex / HybridIndex rows produced for it.

Policies are imported verbatim from the frozen runners:
- 4 transparent baselines: v5 filing_obligation_rule + retrieve_policy
  (run_period_aware_baselines, the same objects the frozen run used);
- finplan_v9_cascade + metadata_v9_fill: v7 filing_obligation_rule +
  retrieve_obligations (run_nonoracle_obligation_planning_v7).
Chunks are built with the same make_period_chunks v7 itself imports (v6).

v7/v8/v4 are covered by the frozen BM25 evidence (amendment_5 scoping note).

Resumable per group: completed groups are recognized by their result files in
results/tmp_lofin_full_dh/ and are not recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\preexperiments")
from storage_paths import RESULTS_ROOT, MODELS_ROOT

from run_lofin_full_benchmark import GROUPS, load_group
from run_period_aware_baselines import retrieve_policy
from run_nonoracle_obligation_planning_v5 import (
    filing_obligation_rule as v5_rule,
)
from run_nonoracle_obligation_planning_v6 import make_period_chunks
from run_nonoracle_obligation_planning_v7 import (
    filing_obligation_rule as v7_rule,
    retrieve_obligations,
)
from run_sec_real_pilot import BM25, parse_time
from run_lofin_multidoc_dense import DenseIndex
from run_lofin_multidoc_hybrid import HybridIndex

PROTOCOL = Path(r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\docs\04-数据与实验\answer_accuracy_dense_hybrid_protocol_v1.json")
FROZEN = RESULTS_ROOT / "lofin_full_benchmark_frozen.final.json"
MODEL_EN = MODELS_ROOT / "bge-small-en-v1.5"
TMP = RESULTS_ROOT / "tmp_lofin_full_dh"
BUDGET = 4

BASELINE_METHODS = ("single_shot", "period_metadata_decomposition",
                    "generic_adaptive_period", "hirec_period")
V9_METHODS = ("finplan_v9_cascade", "metadata_v9_fill")
ALL_METHODS = BASELINE_METHODS + V9_METHODS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prf(predicted: set, gold: set) -> tuple[float, float, float]:
    inter = len(predicted & gold)
    p = inter / len(predicted) if predicted else 0.0
    r = inter / len(gold) if gold else 0.0
    return r, p, float(predicted == gold)


def compute_rows(data: dict, registry: dict, index, chunks_by_id: dict,
                 budget: int) -> list[dict]:
    """Same row schema as the frozen combined file, per method family.
    v9-family rows carry template (no stratum); baseline rows carry stratum
    (no template) - exactly as in lofin_full_benchmark_frozen.final.json."""
    rows: list[dict] = []
    for case in data["cases"]:
        q, cutoff = str(case["question"]), str(case["cutoff"])
        gold = {str(d) for d in case["gold_doc_ids"]}
        gold_entities = {d.split("_")[0] for d in gold}
        # v9 family (v7 planner + v7 retrieve_obligations)
        planned, _ = v7_rule(q, cutoff, registry["companies"])
        pred_docs = {str(o["doc_id"]) for o in planned}
        pred_entities = {str(o["ticker"]) for o in planned}
        for method in V9_METHODS:
            pred = retrieve_obligations(q, cutoff, planned, index, budget, method)
            used = {str(d) for d in pred["used_docs"]}
            future = float(any(
                chunks_by_id[cid].available_at > parse_time(cutoff)
                for cid in pred["used"]))
            e_r, e_p, e_x = _prf(pred_entities, gold_entities)
            o_r, o_p, o_x = _prf(pred_docs, gold)
            rows.append({
                "case_id": case["case_id"],
                "template": case.get("template", ""),
                "method": method,
                "entity_recall": e_r,
                "entity_precision": e_p,
                "entity_exact": e_x,
                "obligation_recall": o_r,
                "obligation_precision": o_p,
                "obligation_exact": o_x,
                "obligation_covers_gold": float(gold <= pred_docs),
                "cascade_leg_recall": len(gold & used) / len(gold),
                "cascade_closure": float(gold <= used),
                "cascade_wrong_doc_rate": len(used - gold) / max(1, len(used)),
                "future_leak": future,
                "queries": float(len(pred["actions"])),
            })
        # transparent baselines (v5 planner + shared retrieve_policy)
        obligations, _ = v5_rule(q, cutoff, registry["companies"])
        for method in BASELINE_METHODS:
            pred = retrieve_policy(q, cutoff, obligations, index, budget, method)
            used = {str(d) for d in pred["used_docs"]}
            rows.append({
                "case_id": case["case_id"],
                "stratum": case.get("stratum", "unstratified"),
                "method": method,
                "leg_recall": len(gold & used) / len(gold),
                "closure": float(gold <= used),
                "wrong_doc_rate": len(used - gold) / max(1, len(used)),
                "queries": float(len(pred["actions"])),
                "documents": float(len(used)),
            })
    return rows


def run_group(gid: str, group: dict, frozen: dict, index_kind: str,
              budget: int) -> tuple[dict | None, dict]:
    data, registry, _ = load_group(gid, group)
    chunks = make_period_chunks(data["documents"])
    chunks_by_id = {c.chunk_id: c for c in chunks}

    parity = {}
    if index_kind in ("parity", "dense", "hybrid"):
        index = BM25(chunks)
        rows = compute_rows(data, registry, index, chunks_by_id, budget)
        mism = []
        for row in rows:
            want = frozen[(row["case_id"], row["method"])]
            for k, v in want.items():
                if k in row and row[k] != v:
                    mism.append((row["case_id"], row["method"], k, row[k], v))
        if mism:
            raise AssertionError(f"{gid}: parity mismatch: {mism[:5]}")
        parity = {"bm25_parity_ok": True, "rows": len(rows)}

    if index_kind == "dense":
        index = DenseIndex(chunks, MODEL_EN, samples_per_doc=12)
        rows = compute_rows(data, registry, index, chunks_by_id, budget)
    elif index_kind == "hybrid":
        index = HybridIndex(chunks, MODEL_EN, samples_per_doc=12, rrf_k=60)
        rows = compute_rows(data, registry, index, chunks_by_id, budget)
    else:
        rows = None
    return rows, parity


def main(index_kind: str, start: int, end: int) -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    frozen_map = {(r["case_id"], r["method"]): r for r in frozen["rows"]}
    groups = json.loads(GROUPS.read_text(encoding="utf-8"))["groups"]
    gids = list(groups.keys())[start:end]
    TMP.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    done = skipped = 0
    for gid in gids:
        out = TMP / f"{gid}_{index_kind}_result.json"
        if out.exists():
            all_rows.extend(json.loads(out.read_text(encoding="utf-8"))["rows"])
            skipped += 1
            continue
        rows, parity = run_group(gid, groups[gid], frozen_map, index_kind, BUDGET)
        payload = {"group": gid, "parity": parity, "rows": rows or []}
        out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if rows:
            all_rows.extend(rows)
        done += 1
        print(f"[{index_kind}] {gid} done={done} skipped={skipped} "
              f"rows={len(rows or [])}", flush=True)

    result = {
        "schema": f"finplan-lofin-full-{index_kind}-result.v1",
        "status": "frozen-benchmark-extension",
        "protocol_sha256": sha256(PROTOCOL),
        "protocol_file": str(PROTOCOL),
        "budget": BUDGET,
        "n_cases": len({r["case_id"] for r in all_rows}),
        "n_rows": len(all_rows),
        "methods": list(ALL_METHODS),
        "model": str(MODEL_EN),
        "rows": all_rows,
    }
    out_file = RESULTS_ROOT / f"lofin_full_{index_kind}_v1.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"saved {out_file} ({len(all_rows)} rows)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=("parity", "dense", "hybrid"),
                        required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2 ** 31)
    args = parser.parse_args()
    main(args.index, args.start, args.end)
