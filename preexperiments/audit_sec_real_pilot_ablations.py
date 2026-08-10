"""Post-hoc paired statistical audit of the A3/A4 ablations already measured
in the SEC real-data pilot (run_sec_real_pilot.py).

This is NOT a pre-registered analysis: the pilot was run before this audit
(exploratory 17 lineages, holdout 12 lineages).  The audit only attaches exact
paired tests to ablations that were already designed into the pilot runner
(finplan_no_limit = A3 -Limiting evidence, finplan_no_time = A4 -Available
time, both against finplan_v3_path_bound).  Verdict strength is capped
accordingly: descriptive support, not confirmatory.

Metrics audited per case (paired, exact two-tailed McNemar / sign test):
  - closure (binary)
  - future_leak (binary, A4)
  - overclaim (binary)
  - limit_action_recall (binary: 1 if the method took a limiting-evidence
    action when one existed; use binaryized 0/1)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

try:
    from .storage_paths import RESULTS_ROOT
except ImportError:
    from storage_paths import RESULTS_ROOT


def mcnemar_exact(n01: int, n10: int) -> float:
    n = n01 + n10
    k = min(n01, n10)
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) * 0.5 ** n for i in range(0, k + 1)))


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
    return {"positive": pos, "negative": neg, "ties": ties, "n": pos + neg, "p": min(1.0, p)}


def audit_file(path: Path, name: str) -> Dict[str, object]:
    r = json.loads(path.read_text(encoding="utf-8"))
    rows = r["rows"]
    cases = sorted({row["case_id"] for row in rows})
    methods = {"finplan_v3_path_bound": "reference", "finplan_no_limit": "A3", "finplan_no_time": "A4"}
    by_case: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], {})[row["method"]] = row
    out: Dict[str, object] = {
        "file": str(path),
        "cases": len(cases),
        "name": name,
        "case_structure": "case_id = lineage :: slice (pre_terminal/post_terminal), each scored as one case",
        "n_slices": {s: sum(1 for row in rows if row["slice"] == s) for s in sorted({row["slice"] for row in rows})},
    }

    for label, ablation, fields in (
        ("A3_no_limit", "finplan_no_limit", ("closure", "limit_action_recall", "limiting_recall")),
        ("A4_no_time", "finplan_no_time", ("closure", "future_leak", "overclaim")),
    ):
        ref_method = "finplan_v3_path_bound"
        block: Dict[str, object] = {}
        for field in fields:
            ref_vals = [by_case[c][ref_method][field] for c in cases]
            abl_vals = [by_case[c][ablation][field] for c in cases]
            if field in ("closure", "future_leak", "overclaim"):
                # binary: n01 = ref fails (0) & ablation holds (1); n10 = ref holds & ablation fails
                n01 = sum(1 for a, b in zip(ref_vals, abl_vals) if not bool(a) and bool(b))
                n10 = sum(1 for a, b in zip(ref_vals, abl_vals) if bool(a) and not bool(b))
                block[field] = {
                    "reference_hits": sum(1 for v in ref_vals if bool(v)),
                    "ablation_hits": sum(1 for v in abl_vals if bool(v)),
                    "n01_ref0_abl1": n01,
                    "n10_ref1_abl0": n10,
                    "p_exact_mcnemar": mcnemar_exact(n01, n10),
                }
            else:
                # recall-style continuous-ish: sign test on ablation - reference
                diffs = [b - a for a, b in zip(ref_vals, abl_vals) if a is not None and b is not None]
                block[field] = {
                    "reference_mean": sum(v for v in ref_vals if v is not None) / max(1, sum(1 for v in ref_vals if v is not None)),
                    "ablation_mean": sum(v for v in abl_vals if v is not None) / max(1, sum(1 for v in abl_vals if v is not None)),
                    "sign_test": sign_test(diffs),
                }
        out[label] = block

    # A5-adjacent evidence: wrong_lineage (retrieval crossed into a different
    # lineage's filings) for path-bound vs unbound methods.  This is the only
    # lineage measurement on real data; it is descriptive, not a full -Lineage
    # ablation (no method in the pilot removes the P dimension while keeping
    # everything else identical).
    lineage_adjacent: Dict[str, object] = {}
    for method in ("finplan_v3_path_bound", "generic_path_bound", "hirec_path_bound",
                   "hirec_style", "single_shot"):
        vals = [by_case[c][method]["wrong_lineage"] for c in cases]
        lineage_adjacent[method] = {
            "wrong_lineage_cases": sum(1 for v in vals if v and v > 0.5),
            "wrong_lineage_mean": sum(v for v in vals if v is not None) / max(1, sum(1 for v in vals if v is not None)),
        }
    out["lineage_adjacent_wrong_lineage"] = lineage_adjacent
    return out


def main() -> None:
    exploratory = audit_file(Path(RESULTS_ROOT) / "sec_real_pilot_v1.json", "exploratory-17-lineages")
    holdout = audit_file(Path(RESULTS_ROOT) / "sec_real_pilot_holdout_v1.json", "holdout-12-lineages")
    audit = {
        "schema": "finplan-sec-real-pilot-ablation-audit.v1",
        "status": "post-hoc-descriptive-not-preregistered",
        "interpretation_boundary": [
            "The pilot predates this audit; the ablations were designed into the runner at pilot time, "
            "but the paired tests were not pre-declared. Descriptive support only, not confirmatory.",
            "limit_action_recall/limiting_recall sign tests use the raw per-case values (0/0.5/1); "
            "binary fields use exact two-tailed McNemar with direction n01/n10 stated.",
        ],
        "exploratory": exploratory,
        "holdout": holdout,
    }
    out = Path(RESULTS_ROOT) / "sec_real_pilot_ablation_audit_v1.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
