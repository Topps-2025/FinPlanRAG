"""FinGLM 复赛A full-set DenseRAG + Hybrid(RRF) retrieval baselines (Part B).

answer_accuracy_dense_hybrid_protocol_v1.json, Part B.  Three gates precede
any dense/hybrid row:

1. Parity gate (full set): the CNBM25 index is built with
   run_finglm_internal.build_index (byte-identical to the frozen runs) and all
   7 policies are re-run on it; every emitted row must equal the frozen rows
   field-for-field (finglm_full_baselines.json for the 4 external baselines,
   finglm_full_internal_v1.json for finplan_v4/v9_cascade/metadata_v9_fill).
2. Interface invariants on DenseIndex (the only new surface): required_terms
   returns exactly the marker document; cutoff excludes later docs; full
   double-run determinism on a 30-case slice across all methods.
3. The same DenseIndex class passes the LOFin full-set BM25 parity gate
   (cross-coverage, run_lofin_full_dense_hybrid.py).

Hybrid = RRF(k=60) over the SHARED CNBM25 (whole-doc) + DenseIndex (12
uniform char windows per doc, bge-small-zh-v1.5), doc-level dedup, sorted
(score desc, doc_id).

Row schema is identical to the frozen files per method family:
- external: case_id/stratum/method/leg_recall/closure/wrong_doc_rate/queries/documents
- internal: + obligations
Both use stratum = case.answer_type exactly as the frozen runners do.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\preexperiments")
from storage_paths import RESULTS_ROOT, MODELS_ROOT

from run_sec_real_pilot import Chunk, parse_time
from run_finglm_baselines import (
    CNBM25, chinese_obligation_rule, precise_path, retrieve_policy,
    cn_tokens,
)
from run_finglm_internal import (
    build_index, chinese_internal_obligation_rule, retrieve,
)
from run_lofin_multidoc_dense import DenseIndex

PROTOCOL = Path(r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\docs\04-数据与实验\answer_accuracy_dense_hybrid_protocol_v1.json")
CORPUS_DIR = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\data\finglm_full_corpus_v1")
REGISTRY_PATH = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\data\finglm_full_registry_v1.json")
CASES_PATH = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\data\finglm_full_cases_v1.json")
FROZEN_EXT = RESULTS_ROOT / "finglm_full_baselines.json"
FROZEN_INT = RESULTS_ROOT / "finglm_full_internal_v1.json"
MODEL_ZH = MODELS_ROOT / "bge-small-zh-v1.5"
BUDGET = 4
N_WINDOWS = 12
RRF_K = 60

EXTERNAL_METHODS = ("single_shot", "period_metadata_decomposition",
                    "generic_adaptive_period", "hirec_period")
INTERNAL_METHODS = ("finplan_v4", "finplan_v9_cascade", "metadata_v9_fill")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_window_chunks(corpus_dir: Path, registry: dict,
                       code_files: Sequence[Path] | None = None) -> list[Chunk]:
    """12 uniform char windows per document; terms = path markers only, so
    required_terms filtering is doc-deterministic (every window of the marker
    doc carries both markers - same doc-level semantics as CNBM25 postings)."""
    companies = registry["companies"]
    chunks: list[Chunk] = []
    files = sorted(Path(corpus_dir).glob("*.json")) if code_files is None else code_files
    for code_file in files:
        code = code_file.stem
        for doc in json.loads(code_file.read_text(encoding="utf-8"))["documents"]:
            obligation = {"ticker": doc["ticker"], "fiscal_year": doc["year"],
                          "filing_type": doc.get("form", "10-K"),
                          "fiscal_period": doc.get("fiscal_period", "FY")}
            generic_path = f"path{str(doc['ticker']).lower()}{int(doc['year'])}"
            precise = precise_path(obligation)
            text = doc["text"]
            n = N_WINDOWS
            positions = [int(round(i * len(text) / n)) for i in range(n + 1)]
            for i in range(n):
                start, end = positions[i], positions[i + 1]
                window = text[start:end] if i < n - 1 else text[start:]
                chunks.append(Chunk(
                    chunk_id=f"{doc['doc_id']}::d{i}",
                    doc_id=doc["doc_id"],
                    lineage_id=doc["doc_id"],
                    role=str(doc.get("form", "10-K")),
                    available_at=parse_time(doc["available_at"]),
                    text=window,
                    terms=(generic_path, precise),
                ))
    return chunks


class HybridRRF:
    """RRF(k=60) fusion over the SHARED CNBM25 + DenseIndex, doc-deduped,
    sorted (score desc, doc_id).  Interface-identical to the frozen indexes:
    search(query, cutoff, used, allow_future, limit, required_terms)."""

    def __init__(self, lexical: CNBM25, dense: DenseIndex, k: int = RRF_K) -> None:
        self.lexical = lexical
        self.dense = dense
        self.k = k

    def search(self, query: str, cutoff, used: list[str], allow_future: bool,
               limit: int = 1, required_terms: tuple = ()) -> list[Chunk]:
        lex = self.lexical.search(query, cutoff, used, allow_future,
                                  limit=len(self.lexical.chunks),
                                  required_terms=required_terms)
        dns = self.dense.search(query, cutoff, used, allow_future,
                                limit=len(self.dense.representatives),
                                required_terms=required_terms)
        scores: dict[str, float] = {}
        winner: dict[str, Chunk] = {}
        for rank, chunk in enumerate(lex):
            score = scores.get(chunk.doc_id, 0.0) + 1.0 / (self.k + rank + 1)
            scores[chunk.doc_id] = score
            winner.setdefault(chunk.doc_id, chunk)
        for rank, chunk in enumerate(dns):
            score = scores.get(chunk.doc_id, 0.0) + 1.0 / (self.k + rank + 1)
            scores[chunk.doc_id] = score
            winner.setdefault(chunk.doc_id, chunk)
        ordered = sorted(scores, key=lambda d: (-scores[d], d))
        return [winner[d] for d in ordered[:limit]]


def compute_rows(cases: list[dict], companies: dict, index, budget: int,
                 internal: bool) -> list[dict]:
    rows: list[dict] = []
    for case in cases:
        q, cutoff = str(case["question"]), str(case["cutoff"])
        gold = {str(d) for d in case["gold_doc_ids"]}
        stratum = case.get("answer_type", "unstratified")
        if internal:
            obligations, _ = chinese_internal_obligation_rule(q, companies)
            for method in INTERNAL_METHODS:
                pred = retrieve(q, cutoff, obligations, index, budget, method)
                used = {str(d) for d in pred["used_docs"]}
                rows.append({
                    "case_id": case["case_id"], "stratum": stratum,
                    "method": method,
                    "leg_recall": len(gold & used) / max(1, len(gold)),
                    "closure": float(gold <= used),
                    "wrong_doc_rate": len(used - gold) / max(1, len(used)),
                    "queries": float(len(pred["actions"])),
                    "documents": float(len(used)),
                    "obligations": float(len(obligations)),
                })
        else:
            obligations, _ = chinese_obligation_rule(q, companies)
            for method in EXTERNAL_METHODS:
                pred = retrieve_policy(q, cutoff, obligations, index, budget, method)
                used = {str(d) for d in pred["used_docs"]}
                rows.append({
                    "case_id": case["case_id"], "stratum": stratum,
                    "method": method,
                    "leg_recall": len(gold & used) / max(1, len(gold)),
                    "closure": float(gold <= used),
                    "wrong_doc_rate": len(used - gold) / max(1, len(used)),
                    "queries": float(len(pred["actions"])),
                    "documents": float(len(used)),
                })
    return rows


def parity_check(cases: list[dict], companies: dict, index) -> dict:
    """Re-run all 7 policies on CNBM25 and compare field-for-field against the
    frozen files.  Returns the mismatch list (empty = pass)."""
    frozen_ext = {(r["case_id"], r["method"]): r for r in
                  json.loads(FROZEN_EXT.read_text(encoding="utf-8"))["rows"]}
    frozen_int = {(r["case_id"], r["method"]): r for r in
                  json.loads(FROZEN_INT.read_text(encoding="utf-8"))["rows"]}
    mism: list[tuple] = []
    for row in compute_rows(cases, companies, index, BUDGET, internal=False):
        want = frozen_ext[(row["case_id"], row["method"])]
        for k, v in want.items():
            if row[k] != v:
                mism.append((row["case_id"], row["method"], k, row[k], v))
    for row in compute_rows(cases, companies, index, BUDGET, internal=True):
        want = frozen_int[(row["case_id"], row["method"])]
        for k, v in want.items():
            if row[k] != v:
                mism.append((row["case_id"], row["method"], k, row[k], v))
    return {"n_compared": len(frozen_ext) + len(frozen_int),
            "mismatches": mism[:20], "pass": not mism}


def interface_invariants(cases: list[dict], companies: dict,
                         dense: DenseIndex,
                         code_files: Sequence[Path] | None = None) -> dict:
    """The only genuinely new surface vs the frozen runs.  Three checks on
    DenseIndex: required_terms marker determinism, cutoff filtering, and
    double-run determinism over a 30-case slice."""
    results: dict = {}
    # 1. required_terms returns exactly the marker document
    corpus_doc_ids = {c.doc_id for c in dense.representatives}
    marker_checks = 0
    marker_ok = 0
    for case in cases[:20]:
        obligations, _ = chinese_internal_obligation_rule(
            str(case["question"]), companies)
        for ob in obligations[:2]:
            required = [precise_path(ob)]
            hits = dense.search(f"path {ob['ticker']} {ob['fiscal_year']}",
                                parse_time(str(case["cutoff"])), [], False,
                                limit=1, required_terms=tuple(required))
            marker_checks += 1
            if hits and hits[0].doc_id == ob["doc_id"]:
                marker_ok += 1
            elif not hits and ob["doc_id"] not in corpus_doc_ids:
                marker_ok += 1  # obligation doc not in corpus (excluded set)
            if marker_checks >= 20:
                break
        if marker_checks >= 20:
            break
    results["required_terms_marker"] = f"{marker_ok}/{marker_checks}"
    results["marker_ok"] = marker_checks > 0 and marker_ok == marker_checks

    # 2. cutoff excludes later docs: search just before a doc's available_at
    cutoff_checks = cutoff_ok = 0
    files = sorted(Path(CORPUS_DIR).glob("*.json")) if code_files is None else code_files
    for code_file in files[:10]:
        for doc in json.loads(code_file.read_text(encoding="utf-8"))["documents"][:2]:
            obligation = {"ticker": doc["ticker"], "fiscal_year": doc["year"],
                          "filing_type": doc.get("form", "10-K"),
                          "fiscal_period": doc.get("fiscal_period", "FY")}
            before = parse_time(doc["available_at"]) - timedelta(days=1)
            hits = dense.search(f"{doc['ticker']} {doc['year']} 报告",
                                before, [], False, limit=1,
                                required_terms=(precise_path(obligation),))
            cutoff_checks += 1
            # markers are unique per (code,year,period,form): with the target
            # doc's own marker, no other doc may satisfy required_terms, so a
            # correct cutoff filter must return nothing (or never the target)
            if not hits or hits[0].doc_id != doc["doc_id"]:
                cutoff_ok += 1
    results["cutoff_filter"] = f"{cutoff_ok}/{cutoff_checks}"
    results["cutoff_ok"] = cutoff_checks > 0 and cutoff_ok == cutoff_checks

    # 3. double-run determinism on a 30-case slice, all 7 methods
    slice_cases = cases[:30]
    first = compute_rows(slice_cases, companies, dense, BUDGET, internal=False)
    first += compute_rows(slice_cases, companies, dense, BUDGET, internal=True)
    second = compute_rows(slice_cases, companies, dense, BUDGET, internal=False)
    second += compute_rows(slice_cases, companies, dense, BUDGET, internal=True)
    results["double_run_deterministic"] = bool(first == second)
    return results


def main(only: str = "all", smoke_codes: int = 0) -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    companies = registry["companies"]
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]

    if smoke_codes:
        # subset build over the first N code files that actually appear as
        # gold companies.  NOTE: subset BM25 rankings/IDF differ from the
        # full-corpus frozen run, so smoke does NOT compare against frozen
        # rows - it exercises the whole chain structurally (index build,
        # policies, rows, dense embeddings, interface invariants).
        gold_codes = {str(g).split("_")[0] for c in cases
                      for g in c["gold_doc_ids"]}
        smoke_files = [p for p in sorted(Path(CORPUS_DIR).glob("*.json"))
                       if p.stem in gold_codes][:smoke_codes]
        smoke_codes_set = {p.stem for p in smoke_files}
        smoke_cases = [c for c in cases
                       if all(str(g).split("_")[0] in smoke_codes_set
                              for g in c["gold_doc_ids"])]
        print(f"SMOKE: {len(smoke_codes_set)} codes, {len(smoke_cases)} cases",
              flush=True)
        cases = smoke_cases
    else:
        smoke_files = None

    lexical = build_index(CORPUS_DIR, registry, code_files=smoke_files)
    print(f"CNBM25: {len(lexical.chunks)} docs, {len(lexical.postings)} terms, "
          f"avgdl={lexical.avgdl:.0f}", flush=True)

    if smoke_codes:
        rows_ext = compute_rows(cases, companies, lexical, BUDGET, internal=False)
        rows_int = compute_rows(cases, companies, lexical, BUDGET, internal=True)
        assert len(rows_ext) == len(cases) * 4 and len(rows_int) == len(cases) * 3, \
            (len(rows_ext), len(rows_int), len(cases))
        dense_chunks = make_window_chunks(CORPUS_DIR, registry, code_files=smoke_files)
        dense = DenseIndex(dense_chunks, MODEL_ZH, samples_per_doc=N_WINDOWS)
        inv = interface_invariants(cases, companies, dense, code_files=smoke_files)
        print(f"SMOKE rows={len(rows_ext) + len(rows_int)} "
              f"dense_reps={len(dense.representatives)} invariants={inv}", flush=True)
        assert inv["marker_ok"], inv
        assert inv["cutoff_ok"], inv
        assert inv["double_run_deterministic"], inv
        print("SMOKE PASS", flush=True)
        return

    gate = parity_check(cases, companies, lexical)
    print(f"parity: pass={gate['pass']} compared={gate['n_compared']} "
          f"mism={gate['mismatches'][:3]}", flush=True)
    if not gate["pass"]:
        raise AssertionError(f"parity gate failed: {gate['mismatches'][:5]}")

    dense_chunks = make_window_chunks(CORPUS_DIR, registry)
    print(f"dense windows: {len(dense_chunks)} chunks", flush=True)
    dense = DenseIndex(dense_chunks, MODEL_ZH, samples_per_doc=N_WINDOWS)
    print(f"dense embeddings: {len(dense.representatives)} reps", flush=True)

    inv = interface_invariants(cases, companies, dense)
    print(f"invariants: {inv}", flush=True)
    assert inv["marker_ok"], inv
    assert inv["cutoff_ok"], inv
    assert inv["double_run_deterministic"], inv

    out = {}
    for name, index in (("dense", dense),
                        ("hybrid", HybridRRF(lexical, dense))):
        rows = (compute_rows(cases, companies, index, BUDGET, internal=False)
                + compute_rows(cases, companies, index, BUDGET, internal=True))
        summary = {}
        for method in EXTERNAL_METHODS + INTERNAL_METHODS:
            rs = [r for r in rows if r["method"] == method]
            summary[method] = {k: (sum(r[k] for r in rs) / len(rs) if rs else 0.0)
                               for k in ("leg_recall", "closure", "wrong_doc_rate",
                                         "queries", "documents")}
        result = {
            "schema": f"finplan-finglm-full-{name}-result.v1",
            "status": "frozen-benchmark-extension",
            "protocol_sha256": sha256(PROTOCOL),
            "budget": BUDGET,
            "n_cases": len({r["case_id"] for r in rows}),
            "n_rows": len(rows),
            "methods": list(EXTERNAL_METHODS + INTERNAL_METHODS),
            "model": str(MODEL_ZH),
            "parity_gate": gate,
            "interface_invariants": inv,
            "summary": summary,
            "rows": rows,
        }
        out_file = RESULTS_ROOT / f"finglm_full_{name}_v1.json"
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"saved {out_file} ({len(rows)} rows)", flush=True)
        out[name] = str(out_file)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("all", "parity"), default="all")
    ap.add_argument("--smoke-codes", type=int, default=0,
                    help="subset index build over first N code files (smoke)")
    args = ap.parse_args()
    main(args.only, args.smoke_codes)
