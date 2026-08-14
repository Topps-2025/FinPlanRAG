"""Audit for the answer-layer + dense/hybrid extension
(answer_accuracy_dense_hybrid_protocol_v1.json, amendment_1).

L1 (answer support): per-method support rates (full + by stratum/layer),
oracle ceiling (verbatim), random-doc noise floor, closure-conditioned
support, core-length stratification (<=3 / 4-5 / 6+ chars), exact McNemar
pairs on the support binary (method vs oracle, method vs method).

Part B (dense/hybrid): once lofin_full_{dense,hybrid}_v1.json and
finglm_full_{dense,hybrid}_v1.json exist, closure / wrong_doc_rate vs the
frozen BM25 rows with exact McNemar (closure) + paired permutation
(wrong_doc_rate, seed 20260812, 20000) exactly as the existing audit flow.

The audit reports numbers; it does not decide significance for L1
(amendment_1: L1 method ranking is not a retrieval-quality claim on LOFin).
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\preexperiments")
from storage_paths import RESULTS_ROOT

SEED = 20260812
N_PERM = 20000

# --- exact McNemar / paired permutation (same flow as the frozen audits) ---

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    lo = sum(math.comb(n, k) * 0.5 ** n for k in range(0, b))
    hi = sum(math.comb(n, k) * 0.5 ** n for k in range(b, n + 1))
    return min(1.0, 2.0 * min(lo, hi))


def paired_permutation(diffs: list[float], seed: int, n_perm: int) -> float:
    rng = random.Random(seed)
    observed = abs(sum(diffs))
    count = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def support_by_method(rows: list[dict]) -> dict[str, float]:
    agg = defaultdict(list)
    for r in rows:
        agg[r["method"]].append(r["support"])
    return {m: sum(v) / len(v) for m, v in agg.items()}


def support_by_stratum(rows: list[dict], method: str) -> dict[str, float]:
    agg = defaultdict(list)
    for r in rows:
        if r["method"] == method:
            agg[r["stratum"]].append(r["support"])
    return {s: sum(v) / len(v) for s, v in agg.items()}


def audit_l1(lofin_l1: dict, finglm_l1: dict) -> dict:
    out: dict = {"lofin": {}, "finglm": {}}

    # --- LOFin ---
    lr = lofin_l1["rows"]
    lo_support = support_by_method(lr)
    out["lofin"]["support_by_method"] = {k: round(v, 4)
                                         for k, v in sorted(lo_support.items())}
    out["lofin"]["oracle"] = round(lo_support.get("oracle", 0.0), 4)
    for m in ("single_shot", "finplan_v9_cascade"):
        out["lofin"][f"support_by_stratum_{m}"] = {
            s: round(v, 4) for s, v in sorted(support_by_stratum(lr, m).items())}

    # closure-conditioned support: rows where the method's used_docs covered
    # the gold (from the frozen closure metric, joined per case/method)
    frozen = json.loads((RESULTS_ROOT / "lofin_full_benchmark_frozen.final.json")
                        .read_text(encoding="utf-8"))
    closure_map = {(r["case_id"], r["method"]): r["cascade_closure"] if "cascade_closure" in r
                   else r["closure"] for r in frozen["rows"]}
    cond: dict[str, list[int]] = defaultdict(list)
    for r in lr:
        if r["method"] == "oracle":
            continue
        if closure_map.get((r["case_id"], r["method"]), 0) == 1.0:
            cond[r["method"]].append(r["support"])
    out["lofin"]["closure_conditioned_support"] = {
        m: round(sum(v) / len(v), 4) for m, v in sorted(cond.items()) if v}

    # exact McNemar on support binary: method vs oracle, per method
    by_case = defaultdict(dict)
    for r in lr:
        by_case[r["case_id"]][r["method"]] = r["support"]
    pairs: dict[str, dict] = {}
    for m in [x for x in lo_support if x != "oracle"]:
        b = sum(1 for c in by_case.values() if c.get(m) == 1 and c.get("oracle") == 0)
        c = sum(1 for c in by_case.values() if c.get(m) == 0 and c.get("oracle") == 1)
        pairs[m] = {"b": b, "c": c, "p": round(mcnemar_exact(b, c), 4)}
    out["lofin"]["mcnemar_support_vs_oracle"] = pairs

    # --- FinGLM ---
    fr = finglm_l1["rows"]
    fi_support = support_by_method(fr)
    out["finglm"]["support_by_method"] = {k: round(v, 4)
                                          for k, v in sorted(fi_support.items())}
    out["finglm"]["oracle"] = round(fi_support.get("oracle", 0.0), 4)
    out["finglm"]["support_by_layer_oracle"] = {
        s: round(v, 4) for s, v in sorted(support_by_stratum(fr, "oracle").items())}
    out["finglm"]["internal_reconstruction"] = finglm_l1["internal_reconstruction"]
    return out


def audit_part_b() -> dict:
    out: dict = {}
    for dataset in ("lofin", "finglm"):
        out[dataset] = {}
        for kind in ("dense", "hybrid"):
            path = RESULTS_ROOT / f"{dataset}_full_{kind}_v1.json"
            if not path.exists():
                out[dataset][kind] = "pending"
                continue
            rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
            frozen_name = ("lofin_full_benchmark_frozen.final.json" if dataset == "lofin"
                           else "finglm_full_baselines.json")
            frozen = json.loads((RESULTS_ROOT / frozen_name).read_text(encoding="utf-8"))["rows"]
            out[dataset][kind] = summarize_retrieval_rows(rows, frozen)
    return out


def summarize_retrieval_rows(rows: list[dict], frozen_rows: list[dict]) -> dict:
    agg = defaultdict(list)
    for r in rows:
        closure = r.get("cascade_closure", r.get("closure"))
        wdr = r.get("cascade_wrong_doc_rate", r.get("wrong_doc_rate"))
        agg[r["method"]].append((closure, wdr))
    frozen_map = {(r["case_id"], r["method"]): r for r in frozen_rows}
    res: dict = {}
    for m, v in sorted(agg.items()):
        closure = sum(x[0] for x in v) / len(v)
        wdr = sum(x[1] for x in v) / len(v)
        b = c = 0
        diffs = []
        for r in rows:
            if r["method"] != m:
                continue
            f = frozen_map.get((r["case_id"], r["method"]))
            if f is None:
                continue
            f_closure = f.get("cascade_closure", f.get("closure"))
            f_wdr = f.get("cascade_wrong_doc_rate", f.get("wrong_doc_rate"))
            cur = r.get("cascade_closure", r.get("closure"))
            if cur == 1.0 and f_closure == 0.0:
                b += 1
            elif cur == 0.0 and f_closure == 1.0:
                c += 1
            diffs.append(r.get("cascade_wrong_doc_rate", r.get("wrong_doc_rate")) - f_wdr)
        res[m] = {
            "closure": round(closure, 4),
            "wdr": round(wdr, 4),
            "frozen_closure": round(sum(
                f.get("cascade_closure", f.get("closure")) for f in frozen_rows
                if f["method"] == m) / max(1, sum(1 for f in frozen_rows if f["method"] == m)), 4),
            "closure_mcnemar_p": round(mcnemar_exact(b, c), 4),
            "wdr_perm_p": round(paired_permutation(diffs, SEED, N_PERM), 4),
            "n": len(v),
        }
    return res


def main() -> None:
    lofin_l1 = json.loads((RESULTS_ROOT / "lofin_full_answers_l1_v1.json")
                          .read_text(encoding="utf-8"))
    finglm_l1 = json.loads((RESULTS_ROOT / "finglm_full_answers_l1_v1.json")
                           .read_text(encoding="utf-8"))
    report = {"schema": "finplan-answer-audit-v1",
              "amendment_1": "L1 = verbatim numeric-core coverage; oracle = verbatim ceiling (derived-number answers absent); LOFin method ranking not a retrieval claim",
              "l1": audit_l1(lofin_l1, finglm_l1),
              "part_b": audit_part_b()}
    out_file = RESULTS_ROOT / "answer_accuracy_audit_v1.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
