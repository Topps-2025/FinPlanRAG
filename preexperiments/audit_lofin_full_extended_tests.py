"""Extended paired tests on the frozen LOFin full run (Task #18, A/B paths).

The frozen audit (lofin_full_benchmark_audit_v1.json) only ran paired exact
McNemar on closure (full stratum).  The claims that are actually candidate
for the paper need two more families of tests:

  A. wrong_doc_rate (continuous, per-case): paired permutation test
     (H0: mean pairwise difference = 0, two-sided), v9_cascade vs each
     baseline.  Random sign-flipping with a fixed seed -> reproducible.
  B. closure McNemar restricted to the multi_doc stratum (gold_doc_ids > 1),
     v9_cascade vs each baseline (and internal diagnostics).

Everything reads only the frozen result JSON and the frozen cases file;
no re-derivation of metrics.  Output:
  results/lofin_full_benchmark_extended_tests_v1.json

Usage: python audit_lofin_full_extended_tests.py [frozen.json]
Pure stdlib (no numpy/scipy): permutation loop is plain Python.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

DATA = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\data")
RESULTS = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\results")
BENCH = RESULTS / "lofin_full_benchmark_frozen.json"
CASES = DATA / "lofin_full_cases_v1.json"

METHODS = ["finplan_v9_cascade", "metadata_v9_fill", "finplan_v8", "finplan_v7",
           "finplan_v4", "single_shot", "period_metadata_decomposition",
           "generic_adaptive_period", "hirec_period"]

# same field mapping as the frozen audit
CLOSURE_FIELD = {m: "closure" for m in METHODS[5:]}
CLOSURE_FIELD.update({m: "cascade_closure" for m in METHODS[:5]})
WRONG_DOC_FIELD = {m: "wrong_doc_rate" for m in METHODS[5:]}
WRONG_DOC_FIELD.update({m: "cascade_wrong_doc_rate" for m in METHODS[:5]})

PERMUTATIONS = 20_000
SEED = 20260812


def permutation_p(diffs: list[float], n_perm: int, seed: int) -> float:
    """Two-sided paired permutation test on the mean of pairwise diffs.

    d_i = a_i - b_i; statistic = |mean(d)|.  Null: sign flips are arbitrary.
    Reproducible via a seeded Random instance.
    """
    n = len(diffs)
    obs = abs(sum(diffs) / n)
    rng = random.Random(seed)
    sign_choices = (1.0, -1.0)
    count = 0
    for _ in range(n_perm):
        s = 0.0
        for d in diffs:
            s += d * sign_choices[rng.getrandbits(1)]
        if abs(s / n) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)  # +1: observed permutation included


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    n01 = n10 = 0
    for x, y in zip(a, b):
        if x and not y:
            n10 += 1
        elif y and not x:
            n01 += 1
    total = n01 + n10
    if total == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        p = 2.0 * sum(math.comb(total, j) * (0.5 ** total) for j in range(k + 1))
        p = min(1.0, p)
    return {"a_wins": n10, "b_wins": n01, "discordant": total, "exact_p": p}


def main() -> None:
    bench_path = Path(sys.argv[1]) if len(sys.argv) > 1 else BENCH
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    cases = {c["case_id"]: c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}

    rows = bench["rows"]
    per_method: dict[str, dict[str, dict]] = {}
    for r in rows:
        per_method.setdefault(r["method"], {})[r["case_id"]] = r
    common = {cid for cid in per_method[METHODS[0]] if cid in cases}

    def multi_doc(cid: str) -> bool:
        return len(cases[cid].get("gold_doc_ids", [])) > 1

    # --- A: wrong_doc_rate paired permutation, v9 vs each other method ---
    a = METHODS[0]
    wd_tests = []
    for b in METHODS[1:]:
        cids = [c for c in common if c in per_method[b]]
        diffs = [float(per_method[a][c][WRONG_DOC_FIELD[a]])
                 - float(per_method[b][c][WRONG_DOC_FIELD[b]]) for c in cids]
        mean_diff = sum(diffs) / len(diffs)
        p = permutation_p(diffs, PERMUTATIONS, SEED)
        wd_tests.append({"a": a, "b": b, "n": len(cids),
                         "mean_diff": round(mean_diff, 4), "p": p})
        print(f"wrong_doc perm: {a} vs {b:<28} n={len(cids):>4} "
              f"mean_diff={mean_diff:+.4f} p={p:.4f}")

    # --- B: multi_doc stratum closure McNemar, v9 vs each other method ---
    md_cids = [c for c in common if multi_doc(c)]
    print(f"\nmulti_doc stratum n={len(md_cids)}")
    md_tests = []
    for b in METHODS[1:]:
        mc = mcnemar([per_method[a][c][CLOSURE_FIELD[a]] >= 1.0 for c in md_cids],
                     [per_method[b][c][CLOSURE_FIELD[b]] >= 1.0 for c in md_cids])
        md_tests.append({"a": a, "b": b, "n": len(md_cids), **mc})
        print(f"multi_doc McNemar: {a} vs {b:<28} wins {mc['a_wins']}/{mc['b_wins']} "
              f"disc {mc['discordant']} p={mc['exact_p']:.4f}")

    out = RESULTS / "lofin_full_benchmark_extended_tests_v1.json"
    out.write_text(json.dumps({
        "schema": "finplan-lofin-full-extended-tests.v1",
        "seed": SEED,
        "permutations": PERMUTATIONS,
        "wrong_doc_rate_permutation_vs_v9": wd_tests,
        "multi_doc_closure_mcnemar_vs_v9": md_tests,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
