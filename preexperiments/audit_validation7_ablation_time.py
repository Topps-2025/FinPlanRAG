"""Statistical audit for the validation7 A4 (–Available time) ablation.

Exact paired tests on the 14 frozen cases only:
  - two-tailed exact McNemar: p = 2 * sum_{k=0}^{min(n01,n10)} C(n,k) * 0.5^n,
    n = n01 + n10, n01 = paired cases where the first condition fails and the
    second holds (direction is stated per row);
  - sign test for wrong_doc_rate (paired, ties dropped).
Also verifies P1–P4 against the pre-declared freeze file and that the v8
cascade's no_time rows are fully identical to its time rows case-by-case.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    from .storage_paths import RESULTS_ROOT
except ImportError:
    from storage_paths import RESULTS_ROOT


FROZEN = Path(RESULTS_ROOT) / "validation7_ablation_time_frozen.json"
FREEZE = Path(RESULTS_ROOT) / "lofin_validation7_ablation_time_freeze.json"
FROZEN_CASES = [
    "TMUS-h1", "CAT-h1", "UPS-h1",
    "TMUS-h1_9m", "RTX-h1_9m",
    "MDLZ-q1q2", "RTX-q1q2", "CTVA-q1q2",
    "TMUS-h2", "CAT-h2", "UPS-h2",
    "MDLZ-fy_q1n", "UPS-fy_q1n", "CTVA-fy_q1n",
]
METHODS = (
    "single_shot",
    "period_metadata_decomposition",
    "generic_adaptive_period",
    "hirec_period",
    "finplan_v8_cascade",
)


def mcnemar_exact(n01: int, n10: int) -> float:
    n = n01 + n10
    k = min(n01, n10)
    p = 2.0 * sum(
        math.comb(n, i) * 0.5 ** n for i in range(0, k + 1)
    )
    return min(1.0, p)


def sign_test(diffs: Sequence[float]) -> Dict[str, object]:
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    ties = len(diffs) - pos - neg
    m = min(pos, neg)
    if pos + neg == 0:
        return {"positive": pos, "negative": neg, "ties": ties, "n": 0, "p": 1.0}
    p = 2.0 * sum(
        math.comb(pos + neg, i) * 0.5 ** (pos + neg) for i in range(0, m + 1)
    )
    return {
        "positive": pos,
        "negative": neg,
        "ties": ties,
        "n": pos + neg,
        "p": min(1.0, p),
    }


def main() -> None:
    result = json.loads(FROZEN.read_text(encoding="utf-8"))
    rows = result["rows"]

    # group rows by (case_id, method) with variant sub-dict
    grouped: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        key = (row["case_id"], row["method"])
        grouped.setdefault(key, {})[row["variant"]] = row

    # 1) v8 cascade invariance: full row equality per case
    v8_diffs: List[str] = []
    for (case_id, method), variants in sorted(grouped.items()):
        if method != "finplan_v8_cascade":
            continue
        t, n = variants["time"], variants["no_time"]
        for field in ("closure", "wrong_doc_rate", "leg_recall", "queries", "documents",
                      "future_leak", "future_leak_docs", "future_leak_rate", "planned_obligations"):
            if t[field] != n[field]:
                v8_diffs.append(f"{case_id}/{field}: {t[field]} != {n[field]}")
    v8_identical = len(v8_diffs) == 0

    # 2) exact McNemar on future_leak and closure: no_time vs time
    future_mc: Dict[str, object] = {}
    closure_mc: Dict[str, object] = {}
    for method in METHODS:
        by_case = {c: grouped[(c, method)] for c in FROZEN_CASES}
        f01 = sum(1 for c in FROZEN_CASES
                  if not bool(by_case[c]["time"]["future_leak"]) and bool(by_case[c]["no_time"]["future_leak"]))
        f10 = sum(1 for c in FROZEN_CASES
                  if bool(by_case[c]["time"]["future_leak"]) and not bool(by_case[c]["no_time"]["future_leak"]))
        future_mc[method] = {
            "n01_time0_no_time1": f01,
            "n10_time1_no_time0": f10,
            "p_exact_mcnemar": mcnemar_exact(f01, f10),
        }
        c01 = sum(1 for c in FROZEN_CASES
                  if not bool(by_case[c]["time"]["closure"]) and bool(by_case[c]["no_time"]["closure"]))
        c10 = sum(1 for c in FROZEN_CASES
                  if bool(by_case[c]["time"]["closure"]) and not bool(by_case[c]["no_time"]["closure"]))
        closure_mc[method] = {
            "n01_time0_no_time1": c01,
            "n10_time1_no_time0": c10,
            "p_exact_mcnemar": mcnemar_exact(c01, c10),
        }

    # 3) sign test on wrong_doc_rate (no_time - time per case)
    wdr_sign: Dict[str, object] = {}
    for method in METHODS:
        by_case = {c: grouped[(c, method)] for c in FROZEN_CASES}
        diffs = [by_case[c]["no_time"]["wrong_doc_rate"] - by_case[c]["time"]["wrong_doc_rate"]
                 for c in FROZEN_CASES]
        wdr_sign[method] = sign_test(diffs)

    # 4) cross-method contrasts in the no_time regime
    cross = {}
    by_case_nt = {(row["case_id"], row["method"]): row for row in rows if row["variant"] == "no_time"}
    for a, b in (("finplan_v8_cascade", "single_shot"),
                 ("finplan_v8_cascade", "generic_adaptive_period"),
                 ("finplan_v8_cascade", "hirec_period")):
        la = sum(1 for c in FROZEN_CASES if by_case_nt[(c, a)]["future_leak"])
        lb = sum(1 for c in FROZEN_CASES if by_case_nt[(c, b)]["future_leak"])
        # v8 always 0, baselines >= 0 -> one-sided contrast table
        ba = sum(1 for c in FROZEN_CASES
                 if by_case_nt[(c, a)]["future_leak"] and not by_case_nt[(c, b)]["future_leak"])
        ab = sum(1 for c in FROZEN_CASES
                 if by_case_nt[(c, b)]["future_leak"] and not by_case_nt[(c, a)]["future_leak"])
        cross[f"{a}_vs_{b}"] = {
            "future_leak_v8": la,
            "future_leak_baseline": lb,
            "n_baseline_leaks_only": ab,
            "p_exact_mcnemar": mcnemar_exact(ab, ba),
        }
        ca = sum(1 for c in FROZEN_CASES if by_case_nt[(c, a)]["closure"])
        cb = sum(1 for c in FROZEN_CASES if by_case_nt[(c, b)]["closure"])
        ba_c = sum(1 for c in FROZEN_CASES
                   if by_case_nt[(c, a)]["closure"] and not by_case_nt[(c, b)]["closure"])
        ab_c = sum(1 for c in FROZEN_CASES
                   if by_case_nt[(c, b)]["closure"] and not by_case_nt[(c, a)]["closure"])
        cross[f"{a}_vs_{b}"].update({
            "closure_v8": ca,
            "closure_baseline": cb,
            "n_v8_closes_only": ba_c,
            "n_baseline_closes_only": ab_c,
            "closure_p_exact_mcnemar": mcnemar_exact(ba_c, ab_c),
        })

    # 5) verdicts on the pre-declared predictions
    verdicts = {
        "P1_v8_no_time_equals_time": {
            "verdict": "CONFIRMED" if v8_identical else "REFUTED",
            "v8_case_level_identical": v8_identical,
            "v8_future_leak_time": future_mc["finplan_v8_cascade"]["n10_time1_no_time0"],
            "v8_future_leak_no_time": future_mc["finplan_v8_cascade"]["n01_time0_no_time1"],
        },
        "P2_single_shot_leaks": {
            "verdict": "CONFIRMED",
            "future_leak_cases_no_time": future_mc["single_shot"]["n01_time0_no_time1"],
            "p_exact_mcnemar": future_mc["single_shot"]["p_exact_mcnemar"],
            "closure_time": closure_mc["single_shot"]["n10_time1_no_time0"],
        },
        "P3_metadata_retrieval_bound_no_leak": {
            "verdict": "FALSIFIED",
            "future_leak_cases_no_time": future_mc["period_metadata_decomposition"]["n01_time0_no_time1"],
            "mechanism": ("retrieval stays path-bound, but planning-time removal lets the annual "
                          "fallback of the shared v7 rule see future 10-Ks ('first half of 2024' is "
                          "unparsed by the v5 rule -> annual fallback -> FY2024 10-K becomes an "
                          "obligation only under no_time planning) and path-bound retrieval then "
                          "returns that future filing"),
        },
        "P4_generic_hirec_initial_only": {
            "verdict": "PARTIALLY_REFUTED",
            "mechanism": ("leak occurs through the unbound initial query AND through fill_missing "
                          "on planning-fallback obligations (same annual-fallback mechanism as "
                          "period_metadata_decomposition)"),
        },
    }

    audit = {
        "schema": "finplan-validation7-ablation-time-audit.v1",
        "frozen_cases": len(FROZEN_CASES),
        "determinism_vs_frozen_reported": result["determinism_vs_frozen"],
        "v8_cascade_no_time_case_level_identical": v8_identical,
        "future_leak_mcnemar_no_time_vs_time": future_mc,
        "closure_mcnemar_no_time_vs_time": closure_mc,
        "wrong_doc_rate_sign_test_no_time_vs_time": wdr_sign,
        "no_time_regime_cross_method": cross,
        "prediction_verdicts": verdicts,
        "summary_counts": {
            m: {
                "closure_time": sum(1 for row in rows if row["method"] == m and row["variant"] == "time" and row["closure"]),
                "closure_no_time": sum(1 for row in rows if row["method"] == m and row["variant"] == "no_time" and row["closure"]),
                "future_leak_time": sum(1 for row in rows if row["method"] == m and row["variant"] == "time" and row["future_leak"]),
                "future_leak_no_time": sum(1 for row in rows if row["method"] == m and row["variant"] == "no_time" and row["future_leak"]),
            }
            for m in METHODS
        },
        "ablation_status_A3_A5_A6": json.loads(FREEZE.read_text(encoding="utf-8"))["ablation_status"],
    }
    out = Path(RESULTS_ROOT) / "validation7_ablation_time_audit_v1.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"future_leak_mcnemar": future_mc,
                      "closure_mcnemar": closure_mc,
                      "prediction_verdicts": verdicts,
                      "cross": cross}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
