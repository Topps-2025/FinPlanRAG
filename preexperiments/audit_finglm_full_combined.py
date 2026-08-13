"""FinGLM combined audit (Tasks #21+#23): external baselines + internal
methods, pairwise significance tests per protocol (amendment_1).

Input: results/finglm_full_baselines.json (4 external methods) +
results/finglm_full_internal_v1.json (3 internal methods).
Combined methods: single_shot, period_metadata_decomposition,
generic_adaptive_period, hirec_period, finplan_v4, finplan_v9_cascade,
metadata_v9_fill  (v7/v8 are represented by finplan_v9_cascade per
amendment_1's equivalence note - no separate rows).

Tests, identical procedures to audit_finglm_full_baselines.py:
- closure: exact McNemar on the paired {0,1} closure indicators;
- wrong_doc_rate: two-sided paired permutation (sign-flip, seed 20260812,
  20,000 permutations, +1 pseudo-count);
- applied on the overall layer and each of the 5 answer_type strata.

No multiple-comparison adjustment; all p-values reported in full.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import sys
from pathlib import Path

RESULTS = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\results")
BENCH = RESULTS / "finglm_full_baselines.json"
INTERNAL = RESULTS / "finglm_full_internal_v1.json"
OUT = RESULTS / "finglm_full_combined_audit_v1.json"

METHODS = ["single_shot", "period_metadata_decomposition",
           "generic_adaptive_period", "hirec_period",
           "finplan_v4", "finplan_v9_cascade", "metadata_v9_fill"]
PERMUTATIONS = 20_000
SEED = 20260812


def permutation_p(diffs: list[float], n_perm: int, seed: int) -> float:
    n = len(diffs)
    obs = abs(sum(diffs) / n)
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        s = 0.0
        for d in diffs:
            s += d * (1.0 if rng.getrandbits(1) else -1.0)
        if abs(s / n) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


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


def layer_tests(rows: list[dict], layer: str) -> dict:
    by = {m: [r for r in rows if r["method"] == m] for m in METHODS}
    out = {"layer": layer, "n_cases": len(by[METHODS[0]]), "pairs": {}}
    for a, b in itertools.combinations(METHODS, 2):
        ra, rb = by[a], by[b]
        assert [r["case_id"] for r in ra] == [r["case_id"] for r in rb], \
            "paired rows must share case order"
        closure_m = mcnemar([r["closure"] for r in ra],
                            [r["closure"] for r in rb])
        wd_p = permutation_p([x["wrong_doc_rate"] - y["wrong_doc_rate"]
                              for x, y in zip(ra, rb)],
                             PERMUTATIONS, SEED)
        out["pairs"][f"{a}|{b}"] = {
            "closure": closure_m,
            "wrong_doc_rate_mean_diff": round(
                sum(x["wrong_doc_rate"] - y["wrong_doc_rate"]
                    for x, y in zip(ra, rb)) / max(1, len(ra)), 4),
            "wrong_doc_rate_p": wd_p,
        }
    return out


def main() -> None:
    bench_path = Path(sys.argv[1]) if len(sys.argv) > 1 else BENCH
    internal_path = Path(sys.argv[2]) if len(sys.argv) > 2 else INTERNAL
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT

    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    internal = json.loads(internal_path.read_text(encoding="utf-8"))
    rows = bench["rows"] + internal["rows"]
    assert len(rows) == 1829 * 7, f"expected 12803 rows, got {len(rows)}"

    layers = ["overall"] + sorted({r["stratum"] for r in rows})
    tests = [layer_tests(rows, "overall")] + [
        layer_tests([r for r in rows if r["stratum"] == s], s) for s in layers[1:]]

    summary = {**bench.get("summary", {}), **internal.get("summary", {})}
    out = {
        "schema": "finplan-finglm-full-combined-audit.v1",
        "protocol": "docs/04-数据与实验/finglm_protocol_v1.json (amendment_1)",
        "benchmark": str(bench_path),
        "internal": str(internal_path),
        "n_cases": 1829,
        "methods": METHODS,
        "permutations": PERMUTATIONS,
        "seed": SEED,
        "layers": tests,
        "summary_overall": summary,
        "note": "file-level closure (single gold doc per case); wrong_doc_rate "
                "paired permutation; no multiple-comparison adjustment; "
                "v7/v8 represented by finplan_v9_cascade (amendment_1).",
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"audit: {len(layers)} layers, {sum(len(t['pairs']) for t in tests)} "
          f"pairwise tests -> {out_path}", flush=True)
    for t in tests:
        sig = [(k, v["closure"].get("exact_p"), v["wrong_doc_rate_p"])
               for k, v in t["pairs"].items()
               if v["closure"].get("exact_p", 1) < 0.05
               or v["wrong_doc_rate_p"] < 0.05]
        print(f"  {t['layer']}: n={t['n_cases']} significant pairs: {sig}")


if __name__ == "__main__":
    main()
