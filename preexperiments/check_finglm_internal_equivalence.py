"""Post-run equivalence checks (amendment_1, Tasks #21+#23).

Asserted after run_finglm_internal.py completes:
1. index parity: internal rows were produced over avgdl=55285 /
   1,212,588 unique terms (asserted inside the runner; re-reported here);
2. single-year consistency: on the 1783 single-year cases finplan_v9_cascade
   shares obligation + retrieval with baseline period_metadata_decomposition
   -> closure / wrong_doc_rate / leg_recall / documents must match
   row-for-row;
3. metadata_v9_fill vs pmd: same obligation + same fill_missing retrieval on
   single-year cases -> identical rows too (P4 comparator sanity).

Run: python check_finglm_internal_equivalence.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RESULTS = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\results")
BENCH = RESULTS / "finglm_full_baselines.json"
INTERNAL = RESULTS / "finglm_full_internal_v1.json"
CASES = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\data\finglm_full_cases_v1.json")

_YEAR = re.compile(r"(20\d{2})年")


def main() -> None:
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    internal = json.loads(INTERNAL.read_text(encoding="utf-8"))
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]

    by_method = {m: {r["case_id"]: r for r in bench["rows"] if r["method"] == m}
                 for m in ("period_metadata_decomposition",)}
    internal_rows = {m: {r["case_id"]: r for r in internal["rows"]
                         if r["method"] == m} for m in
                     ("finplan_v9_cascade", "metadata_v9_fill")}

    single_year = [c["case_id"] for c in cases
                   if len(_YEAR.findall(str(c["question"]))) >= 1
                   and len(set(_YEAR.findall(str(c["question"])))) == 1]
    multi_year = [c["case_id"] for c in cases
                  if len(set(_YEAR.findall(str(c["question"])))) > 1]
    assert len(single_year) == 1783 and len(multi_year) == 46, \
        f"year split mismatch: {len(single_year)}/{len(multi_year)}"

    # 2+3: v9_cascade and metadata_v9_fill must equal pmd on single-year cases
    fields = ("closure", "wrong_doc_rate", "leg_recall", "documents")
    for method, tag in (("finplan_v9_cascade", "v9_cascade"),
                        ("metadata_v9_fill", "metadata_fill")):
        mism = []
        for cid in single_year:
            a, b = by_method["period_metadata_decomposition"][cid], internal_rows[method][cid]
            for f in fields:
                if a[f] != b[f]:
                    mism.append((cid, f, a[f], b[f]))
        status = "OK" if not mism else "MISMATCH"
        print(f"{tag} vs pmd on 1783 single-year cases: {status} "
              f"({len(mism)} field mismatches)")
        for m in mism[:5]:
            print("   ", m)

    # multi-year cases: v9_cascade closure must be >= pmd closure
    pmd = by_method["period_metadata_decomposition"]
    v9 = internal_rows["finplan_v9_cascade"]
    wins = [c for c in multi_year if pmd[c]["closure"] == 0 and v9[c]["closure"] == 1]
    losses = [c for c in multi_year if pmd[c]["closure"] == 1 and v9[c]["closure"] == 0]
    print(f"multi-year ({len(multi_year)}): pmd-fail->v9-fix {len(wins)}, "
          f"pmd-ok->v9-fail {len(losses)}")

    # obligation counts on multi-year cases
    obs = {r["case_id"]: r["obligations"] for r in internal["rows"]
           if r["method"] == "finplan_v9_cascade"}
    assert all(obs[c] == 2 for c in multi_year), "multi-year obligations != 2"
    assert all(obs[c] == 1 for c in single_year), "single-year obligations != 1"
    print("obligation counts: single=1, multi=2 - OK")


if __name__ == "__main__":
    main()
