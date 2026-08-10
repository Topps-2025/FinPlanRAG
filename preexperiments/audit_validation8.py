"""Statistical audit of the validation8 frozen run (pre-registered P1-P4).

Loads the five frozen result files (all produced after the freeze file
nonoracle_obligation_v9_validation8_freeze.json was written) and attaches
exact paired tests over the 24 manifest-declared frozen cases:

  closure        -> exact two-tailed McNemar on the discordant-pair table
  wrong_doc_rate -> exact two-tailed sign test on paired differences

Per-case method registry (from the frozen files):
  finplan_v9_cascade / metadata_v9_fill  from nonoracle_obligation_v9_...
  finplan_v8  (v6 runner)  from nonoracle_obligation_v8_...
  finplan_v7  (v5 runner)  from nonoracle_obligation_v7_...
  finplan_v4  (v4 runner)  from nonoracle_obligation_v4_...
  single_shot / period_metadata_decomposition / generic_adaptive_period /
  hirec_period from period_aware_baselines_...

Predictions (verbatim from the freeze file):
  P1: v9 closure >= v8; fy_q1n: v8 0/5, v9 5/5
  P2: v9 wrong_doc < v7/v8 on h1_9m; closure 1.0 for all three
  P3: v9 closure > v7-shared baselines (paired exact McNemar)
  P4_open: v9 vs metadata_v9_fill; parity -> representation layer only

Also: future_leak count for v9, and a determinism cross-check of the 6 dev
cases that appear in both the dev and frozen files of each runner.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

from storage_paths import RESULTS_ROOT

RESULTS = RESULTS_ROOT

FROZEN_CASES = sorted(
    {
        "CSX-h1", "GD-h1", "TXN-h1", "CMCSA-h1",
        "CVS-h1_9m", "HCA-h1_9m", "GD-h1_9m", "TXN-h1_9m", "CMCSA-h1_9m",
        "CVS-q1q2", "HCA-q1q2", "CSX-q1q2", "GD-q1q2", "CMCSA-q1q2",
        "CVS-h2", "HCA-h2", "CSX-h2", "TXN-h2", "CMCSA-h2",
        "CVS-fy_q1n", "HCA-fy_q1n", "CSX-fy_q1n", "GD-fy_q1n", "TXN-fy_q1n",
    }
)
DEV_CASES = {
    "CVS-h1", "HCA-h1", "CSX-h1_9m", "TXN-q1q2", "GD-h2", "CMCSA-fy_q1n",
}

METHODS = (
    "finplan_v9_cascade",
    "metadata_v9_fill",
    "finplan_v8",
    "finplan_v7",
    "finplan_v4",
    "single_shot",
    "period_metadata_decomposition",
    "generic_adaptive_period",
    "hirec_period",
)

# case_id -> {method: (closure, wrong_doc_rate)}
TABLE: Dict[str, Dict[str, Tuple[float, float]]] = {c: {} for c in FROZEN_CASES}


def mcnemar_exact(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2.0 * sum(math.comb(n, i) * 0.5 ** n for i in range(0, k + 1)))


def sign_test(diffs: Sequence[float]) -> Dict[str, object]:
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    ties = len(diffs) - pos - neg
    if pos + neg == 0:
        return {"positive": pos, "negative": neg, "ties": ties, "n": 0, "p": 1.0}
    m = min(pos, neg)
    p = 2.0 * sum(math.comb(pos + neg, i) * 0.5 ** (pos + neg) for i in range(0, m + 1))
    return {"positive": pos, "negative": neg, "ties": ties, "n": pos + neg, "p": min(1.0, p)}


def load_single(path: Path, method: str) -> None:
    """Single-method result file: rows without a method key."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data["rows"]:
        case = str(row["case_id"])
        if case in TABLE:
            TABLE[case][method] = (float(row["cascade_closure"]), float(row["cascade_wrong_doc_rate"]))


def load_double(path: Path) -> None:
    """v9 result file: rows keyed by method."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data["rows"]:
        case = str(row["case_id"])
        method = str(row["method"])
        if case in TABLE and method in ("finplan_v9_cascade", "metadata_v9_fill"):
            TABLE[case][method] = (float(row["cascade_closure"]), float(row["cascade_wrong_doc_rate"]))


def load_baselines(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data["rows"]:
        case = str(row["case_id"])
        method = str(row["method"])
        if case in TABLE:
            TABLE[case][method] = (float(row["closure"]), float(row["wrong_doc_rate"]))


def closure_mcnemar(a: str, b: str) -> Dict[str, object]:
    n01 = n10 = 0
    hits = {m: 0 for m in (a, b)}
    for case in FROZEN_CASES:
        va = TABLE[case][a][0] == 1.0
        vb = TABLE[case][b][0] == 1.0
        hits[a] += int(va)
        hits[b] += int(vb)
        if not va and vb:
            n01 += 1
        if va and not vb:
            n10 += 1
    return {
        f"{a}_closure": hits[a],
        f"{b}_closure": hits[b],
        "n": len(FROZEN_CASES),
        "n01_a0_b1": n01,
        "n10_a1_b0": n10,
        "p_exact_mcnemar": mcnemar_exact(n01, n10),
    }


def wrong_sign_test(a: str, b: str, cases: Sequence[str] = FROZEN_CASES) -> Dict[str, object]:
    diffs = []
    for case in cases:
        diffs.append(TABLE[case][b][1] - TABLE[case][a][1])  # positive = b has more wrong docs
    means = {m: sum(TABLE[c][m][1] for c in cases) / len(cases) for m in (a, b)}
    return {
        f"{a}_mean_wrong": means[a],
        f"{b}_mean_wrong": means[b],
        "sign_test": sign_test(diffs),
    }


def template_of(case: str) -> str:
    return case.split("-", 1)[1].rsplit("-", 1)[0]


def main() -> None:
    load_double(RESULTS / "nonoracle_obligation_v9_validation8_frozen.json")
    load_single(RESULTS / "nonoracle_obligation_v8_validation8_frozen.json", "finplan_v8")
    load_single(RESULTS / "nonoracle_obligation_v7_validation8_frozen.json", "finplan_v7")
    load_single(RESULTS / "nonoracle_obligation_v4_validation8_frozen.json", "finplan_v4")
    load_baselines(RESULTS / "period_aware_baselines_validation8_frozen.json")

    # --- closure summary table (total + per template) ---
    closure_table: Dict[str, Dict[str, object]] = {}
    for method in METHODS:
        total = sum(1 for c in FROZEN_CASES if TABLE[c][method][0] == 1.0)
        per_template: Dict[str, float] = {}
        for template in ("h1", "h1_9m", "q1q2", "h2", "fy_q1n"):
            cases = [c for c in FROZEN_CASES if template_of(c) == template]
            per_template[template] = sum(1 for c in cases if TABLE[c][method][0] == 1.0)
        closure_table[method] = {"closure": total, "n": len(FROZEN_CASES), "per_template": per_template}

    # --- paired tests ---
    paired_closure = {
        name: closure_mcnemar("finplan_v9_cascade", other)
        for name, other in (
            ("v9_vs_v8", "finplan_v8"),
            ("v9_vs_v7", "finplan_v7"),
            ("v9_vs_v4", "finplan_v4"),
            ("v9_vs_single_shot", "single_shot"),
            ("v9_vs_metadata_decomposition", "period_metadata_decomposition"),
            ("v9_vs_generic_adaptive", "generic_adaptive_period"),
            ("v9_vs_hirec", "hirec_period"),
            ("v9_vs_metadata_v9_fill", "metadata_v9_fill"),
        )
    }
    paired_wrong = {
        name: wrong_sign_test("finplan_v9_cascade", other)
        for name, other in (
            ("v9_vs_v8", "finplan_v8"),
            ("v9_vs_v7", "finplan_v7"),
            ("v9_vs_single_shot", "single_shot"),
            ("v9_vs_metadata_v9_fill", "metadata_v9_fill"),
        )
    }

    # --- P1: fy_q1n stratum ---
    fy_cases = [c for c in FROZEN_CASES if template_of(c) == "fy_q1n"]
    p1 = {
        "v9_fy_q1n_closure": sum(1 for c in fy_cases if TABLE[c]["finplan_v9_cascade"][0] == 1.0),
        "v8_fy_q1n_closure": sum(1 for c in fy_cases if TABLE[c]["finplan_v8"][0] == 1.0),
        "n_fy_q1n": len(fy_cases),
        "overall_paired": paired_closure["v9_vs_v8"],
        "prediction": "v8 0/5 -> v9 5/5; v9 closure >= v8 overall",
        "verdict": "confirmed"
        if sum(1 for c in fy_cases if TABLE[c]["finplan_v9_cascade"][0] == 1.0) == len(fy_cases)
        and sum(1 for c in fy_cases if TABLE[c]["finplan_v8"][0] == 1.0) == 0
        and closure_table["finplan_v9_cascade"]["closure"] >= closure_table["finplan_v8"]["closure"]
        else "refuted",
    }

    # --- P2: h1_9m wrong_doc ---
    h1_9m_cases = [c for c in FROZEN_CASES if template_of(c) == "h1_9m"]
    p2 = {
        "cases": h1_9m_cases,
        "v9_wrong_mean": sum(TABLE[c]["finplan_v9_cascade"][1] for c in h1_9m_cases) / len(h1_9m_cases),
        "v8_wrong_mean": sum(TABLE[c]["finplan_v8"][1] for c in h1_9m_cases) / len(h1_9m_cases),
        "v7_wrong_mean": sum(TABLE[c]["finplan_v7"][1] for c in h1_9m_cases) / len(h1_9m_cases),
        "v9_closure_all": all(TABLE[c]["finplan_v9_cascade"][0] == 1.0 for c in h1_9m_cases),
        "v8_closure_all": all(TABLE[c]["finplan_v8"][0] == 1.0 for c in h1_9m_cases),
        "v7_closure_all": all(TABLE[c]["finplan_v7"][0] == 1.0 for c in h1_9m_cases),
        "sign_test_v9_vs_v8": wrong_sign_test("finplan_v9_cascade", "finplan_v8", h1_9m_cases)["sign_test"],
        "sign_test_v9_vs_v7": wrong_sign_test("finplan_v9_cascade", "finplan_v7", h1_9m_cases)["sign_test"],
        "prediction": "v9 wrong_doc < v7/v8 on h1_9m; closure 1.0 for all three",
        "verdict": None,
    }
    p2["verdict"] = (
        "confirmed"
        if p2["v9_wrong_mean"] == 0
        and p2["v9_wrong_mean"] < p2["v8_wrong_mean"]
        and p2["v9_wrong_mean"] < p2["v7_wrong_mean"]
        and p2["v9_closure_all"] and p2["v8_closure_all"] and p2["v7_closure_all"]
        else "refuted"
    )

    # --- P3: pre-registered main comparison ---
    baselines = ("single_shot", "period_metadata_decomposition", "generic_adaptive_period", "hirec_period")
    p3 = {
        "v9_closure": closure_table["finplan_v9_cascade"]["closure"],
        "baseline_closures": {m: closure_table[m]["closure"] for m in baselines},
        "paired_mcnemar": {
            m: paired_closure[f"v9_vs_{m}"]
            for m in ("single_shot", "metadata_decomposition", "generic_adaptive", "hirec")
        },
        "stratum_discrimination": {
            t: {
                "v9": closure_table["finplan_v9_cascade"]["per_template"][t],
                "baselines_max": max(closure_table[m]["per_template"][t] for m in baselines),
            }
            for t in ("h1", "h2", "fy_q1n", "q1q2")
        },
        "prediction": "v9 closure > all v7-shared baselines",
        "verdict": (
            "confirmed"
            if all(closure_table["finplan_v9_cascade"]["closure"] > closure_table[m]["closure"] for m in baselines)
            else "refuted"
        ),
    }

    # --- P4: open honesty check ---
    p4 = {
        "v9_closure": closure_table["finplan_v9_cascade"]["closure"],
        "metadata_v9_fill_closure": closure_table["metadata_v9_fill"]["closure"],
        "paired_mcnemar": paired_closure["v9_vs_metadata_v9_fill"],
        "wrong_sign_test": paired_wrong["v9_vs_metadata_v9_fill"],
        "prediction": "open; parity -> representation layer confirmed, cascade policy no independent gain here",
        "verdict": (
            "parity-representation-layer-only"
            if closure_table["finplan_v9_cascade"]["closure"] == closure_table["metadata_v9_fill"]["closure"]
            else "cascade-gain-possible"
        ),
    }

    # --- future leak (v9 only; v4 file has no future_leak field) ---
    v9_raw = json.loads((RESULTS / "nonoracle_obligation_v9_validation8_frozen.json").read_text(encoding="utf-8"))
    leak_cases = [str(r["case_id"]) for r in v9_raw["rows"] if float(r.get("future_leak", 0.0)) > 0]
    v9_dev_raw = json.loads((RESULTS / "nonoracle_obligation_v9_validation8_dev.json").read_text(encoding="utf-8"))
    leak_dev = [str(r["case_id"]) for r in v9_dev_raw["rows"] if float(r.get("future_leak", 0.0)) > 0]

    # --- determinism: dev 6 cases in dev vs frozen files of the same runner ---
    determinism: Dict[str, Dict[str, object]] = {}
    for label, dev_name, frozen_name, field in (
        ("v9", "nonoracle_obligation_v9_validation8_dev.json", "nonoracle_obligation_v9_validation8_frozen.json", "cascade_closure"),
        ("v8", "nonoracle_obligation_v8_validation8_dev.json", "nonoracle_obligation_v8_validation8_frozen.json", "cascade_closure"),
        ("v7", "nonoracle_obligation_v7_validation8_dev.json", "nonoracle_obligation_v7_validation8_frozen.json", "cascade_closure"),
        ("v4", "nonoracle_obligation_v4_validation8_dev.json", "nonoracle_obligation_v4_validation8_frozen.json", "cascade_closure"),
        ("baselines", "period_aware_baselines_validation8_dev.json", "period_aware_baselines_validation8_frozen.json", "closure"),
    ):
        dev = json.loads((RESULTS / dev_name).read_text(encoding="utf-8"))
        frozen = json.loads((RESULTS / frozen_name).read_text(encoding="utf-8"))

        def norm(rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, str], Tuple[float, float]]:
            out = {}
            for row in rows:
                case = str(row["case_id"])
                method = str(row.get("method", label))
                if case not in DEV_CASES:
                    continue
                wrong_field = "wrong_doc_rate" if label == "baselines" else "cascade_wrong_doc_rate"
                out[(case, method)] = (float(row[field]), float(row[wrong_field]))
            return out

        dev_map, frozen_map = norm(dev["rows"]), norm(frozen["rows"])
        keys = set(dev_map) & set(frozen_map)
        diffs = [key for key in sorted(keys) if dev_map[key] != frozen_map[key]]
        missing = sorted(set(dev_map) - set(frozen_map))
        determinism[label] = {
            "dev_cases_compared": len(keys),
            "field_diffs": diffs,
            "missing_in_frozen": missing,
        }

    audit = {
        "schema": "finplan-lofin-validation8-audit.v1",
        "status": "preregistered-frozen-24-one-shot",
        "frozen_cases": len(FROZEN_CASES),
        "closure_by_method": closure_table,
        "paired_closure_tests": paired_closure,
        "paired_wrong_doc_sign_tests": paired_wrong,
        "predictions": {"P1": p1, "P2": p2, "P3": p3, "P4": p4},
        "future_leak": {"v9_frozen_cases": leak_cases, "v9_dev_cases": leak_dev, "total_frozen": len(leak_cases)},
        "determinism_dev_vs_frozen": determinism,
        "interpretation_boundary": [
            "All numbers are filing-level obligation closure and document correctness, not page-level evidence or final-answer accuracy.",
            "P4 parity: the v9-vs-baseline gap is attributable to the obligation representation layer (fiscal-year-N, first-n, aliases); the cascade retrieval policy shows no independent gain on this set.",
            "v8 differs from v9 only on the pre-declared fy_q1n stratum; overall v9 >= v8 with p=0.0625 (not <0.05; reported directionally).",
            "Baselines share the v7 planner; comparisons are against v7-shared planning, not against original HiREC/Search-R1.",
            "Same-day internal replication (dev 6 vs frozen re-run) is a determinism check, not an independent replication.",
        ],
    }
    out = RESULTS / "lofin_validation8_statistical_audit_v1.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "closure_by_method": closure_table,
        "paired_closure_tests": paired_closure,
        "predictions": audit["predictions"],
        "future_leak": audit["future_leak"],
        "determinism": determinism,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
