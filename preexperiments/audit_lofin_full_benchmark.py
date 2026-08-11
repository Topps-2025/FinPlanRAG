"""Audit the frozen LOFin full benchmark result (Task #17).

Metrics per protocol lofin_full_benchmark_protocol_v1.json:
  closure (cascade_closure: gold subseteq used), wrong_doc_rate, future_leak
  (resolved from retrieval chunks vs case cutoff), obligation_exact;
  strata: full / 3 answer-type subsets / single-vs-multi-doc / company-naming
  surface (amendment_3: finqa/secqa views are not read);
  statistics: paired exact McNemar on all 36 method pairs + sign test;
  predeclared predictions P1-P6 checks (honest reporting only).

Inputs: results/lofin_full_benchmark_frozen.json (driver output; pass another
        path as argv[1] to audit a different result, e.g. the dev smoke file),
        data/lofin_full_cases_v1.json (stratum / gold / cutoffs),
        data/lofin_full_corpus_v1/*.json (doc_id -> available_at).
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping

DATA = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\data")
RESULTS = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\results")
BENCH = RESULTS / "lofin_full_benchmark_frozen.json"
CASES = DATA / "lofin_full_cases_v1.json"
CORPUS_DIR = DATA / "lofin_full_corpus_v1"

METHODS = ["finplan_v9_cascade", "metadata_v9_fill", "finplan_v8", "finplan_v7",
           "finplan_v4", "single_shot", "period_metadata_decomposition",
           "generic_adaptive_period", "hirec_period"]

# row field names differ across runners: planner family (v4-v9) vs baselines
CLOSURE_FIELD = {m: "closure" for m in METHODS[5:]}          # baselines
CLOSURE_FIELD.update({m: "cascade_closure" for m in METHODS[:5]})
WRONG_DOC_FIELD = {m: "wrong_doc_rate" for m in METHODS[5:]}  # baselines
WRONG_DOC_FIELD.update({m: "cascade_wrong_doc_rate" for m in METHODS[:5]})
OBLIG_FIELD = {m: "obligation_exact" for m in METHODS[:5]}


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
        import math
        # exact two-sided binomial: 2 * P(X <= min(n01, n10)), capped at 1
        k = min(n01, n10)
        p = 2.0 * sum(math.comb(total, j) * (0.5 ** total) for j in range(k + 1))
        p = min(1.0, p)
    return {"a_wins": n10, "b_wins": n01, "discordant": total, "exact_p": p}


def sign_test(a: list[float], b: list[float]) -> dict:
    n_gt = n_lt = 0
    for x, y in zip(a, b):
        if x > y:
            n_gt += 1
        elif x < y:
            n_lt += 1
    import math
    n = n_gt + n_lt
    if n == 0:
        return {"n_gt": 0, "n_lt": 0, "exact_p": 1.0}
    k = min(n_gt, n_lt)
    p = 2.0 * sum(math.comb(n, j) * (0.5 ** n) for j in range(k + 1))
    return {"n_gt": n_gt, "n_lt": n_lt, "exact_p": min(1.0, p)}


def main() -> None:
    bench_path = Path(sys.argv[1]) if len(sys.argv) > 1 else BENCH
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    cases = {c["case_id"]: c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
    # doc_id -> available_at across the whole corpus
    avail: dict[str, str] = {}
    for p in CORPUS_DIR.glob("*.json"):
        for d in json.loads(p.read_text(encoding="utf-8"))["documents"]:
            avail[d["doc_id"]] = d["available_at"]

    rows = bench["rows"]
    print(f"rows: {len(rows)}  methods present: {sorted({r['method'] for r in rows})}")
    per_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_method[r["method"]].append(r)

    def subset_of(r: dict) -> bool:
        return cases.get(r["case_id"], {}).get("stratum", "?")

    # P5 surface = company ALIASES from the registry (official name + former
    # names), per protocol wording ("题面含公司名(别名 ≥6 字符匹配)").  Raw
    # evidence tickers are mostly 2-5 chars (that rule leaves the stratum
    # empty); full alias strings never appear verbatim in questions, so match
    # on the alias's meaningful-word bigrams ("JPMorgan Chase" from
    # "JPMorgan Chase & Co").  Words carrying no company identity are noise.
    _NOISE = {"&", "and", "co", "co.", "inc", "inc.", "incorporated", "corp",
              "corporation", "company", "ltd", "plc", "llc", "trust", "the",
              "of", "for", "in", "on", "holdings", "holding", "group", "group.",
              "common", "stock", "shares", "capital", "bancorporation"}
    registry = json.loads((DATA / "lofin_full_registry_v1.json").read_text(encoding="utf-8"))["companies"]
    alias_bigrams_of_ticker: dict[str, list[tuple[str, str]]] = {}
    for t, e in registry.items():
        bigrams = set()
        for a in e.get("aliases", []):
            if len(a) < 6:
                continue
            words = [w for w in a.lower().split() if w not in _NOISE]
            bigrams.update((words[i], words[i + 1]) for i in range(len(words) - 1))
        alias_bigrams_of_ticker[t] = sorted(bigrams)

    def named_surface(r: dict) -> bool:
        """question text contains a company-name alias bigram (P5)."""
        c = cases.get(r["case_id"])
        if not c:
            return False
        q = str(c["question"]).lower()
        for doc in c["gold_doc_ids"]:
            ticker = doc.split("_")[0].upper()
            if any(f"{a} {b}" in q for a, b in alias_bigrams_of_ticker.get(ticker, [])):
                return True
        return False

    def multi_doc(r: dict) -> bool:
        return len(cases.get(r["case_id"], {}).get("gold_doc_ids", [])) > 1

    # P2: fiscal-year phrasing stratum.  Regexes copied verbatim from the
    # frozen inventory (inventory_lofin_full.py PATTERNS, same lineage as the
    # protocol's question_phrasing_coverage counts: fiscal_year_N 109 + FY_N
    # 105, overlapping, non-exclusive).
    _FY_PATTERNS = (re.compile(r"fiscal year\s+(?:of\s+)?(20\d\d)", re.I),
                    re.compile(r"\bfy\s+((?:20)?\d\d)\b", re.I))

    def fiscal_year_phrasing(r: dict) -> bool:
        q = str(cases.get(r["case_id"], {}).get("question", ""))
        return any(p.search(q) for p in _FY_PATTERNS)

    # amendment_3: question set reads the three by_answer_type files only
    # (finqa/secqa were redundant data-source views); strata follow suit.
    strata = {
        "full": lambda r: True,
        "textual": lambda r: subset_of(r) == "textual",
        "numeric_table": lambda r: subset_of(r) == "numeric_table",
        "numeric_text": lambda r: subset_of(r) == "numeric_text",
        "single_doc": lambda r: not multi_doc(r),
        "multi_doc": multi_doc,
        "names_company": named_surface,
        "no_company_name": lambda r: not named_surface(r),
        "fiscal_year_phrasing": fiscal_year_phrasing,
    }

    # per-case future_leak: v7 rows carry it; v4-v8 from retrieval chunks;
    # baselines from the per-runner trajectories files (used_docs).
    # driver writes predictions keyed by RUNNER name; map method -> runner
    RUNNER_OF = {"finplan_v8": "v8", "finplan_v7": "v7", "finplan_v4": "v4"}
    leak_cache: dict[tuple[str, str], int] = {}

    def case_leak(case_id: str, method: str, row: dict) -> int | None:
        key = (case_id, method)
        if key in leak_cache:
            return leak_cache[key]
        cutoff = str(cases[case_id]["cutoff"])
        leak = 0
        if method in ("finplan_v9_cascade", "metadata_v9_fill"):
            leak = int(row.get("future_leak", 0))
        elif method in ("finplan_v8", "finplan_v7", "finplan_v4"):
            pred = next((p for p in bench.get("predictions", [])
                         if p.get("case_id") == case_id and p.get("method") == RUNNER_OF[method]
                         and p.get("retrieval")), None)
            if pred is not None:
                for chunk_id in pred["retrieval"].get("used", []):
                    doc_id = chunk_id.split("::")[0]
                    if doc_id in avail and avail[doc_id] > cutoff:
                        leak = 1
                        break
        leak_cache[key] = leak
        return leak

    report: dict = {}
    for sname, fn in strata.items():
        report[sname] = {}
        for method in METHODS:
            rs = [r for r in per_method.get(method, []) if fn(r)]
            if not rs:
                report[sname][method] = None
                continue
            cf = CLOSURE_FIELD[method]
            closure = sum(float(r[cf]) for r in rs) / len(rs)
            wf = WRONG_DOC_FIELD[method]
            wrong = sum(float(r[wf]) for r in rs) / len(rs)
            of = OBLIG_FIELD.get(method)
            oblig = sum(float(r[of]) for r in rs) / len(rs) if of else None
            n_leak = sum(1 for r in rs if case_leak(r["case_id"], method, r))
            report[sname][method] = {
                "n": len(rs),
                "closure": round(closure, 4),
                "wrong_doc_rate": round(wrong, 4),
                "obligation_exact": round(oblig, 4) if oblig is not None else None,
                "future_leak": n_leak,
            }
    print("\n=== per-stratum per-method (closure | wrong_doc | obligation_exact | future_leak) ===")
    for sname, m in report.items():
        line = f"{sname:>16}: "
        for method in METHODS:
            v = m.get(method)
            if v:
                oblig = f"{v['obligation_exact']:.2f}" if v["obligation_exact"] is not None else "-"
                line += f"{method[:12]}={v['closure']:.3f}/{v['wrong_doc_rate']:.2f}/{oblig}/{v['future_leak']} "
        print(line)

    # paired McNemar on closure, all 36 pairs (full stratum)
    print("\n=== paired exact McNemar on closure (36 pairs) ===")
    full = {m: {r["case_id"]: float(r[CLOSURE_FIELD[m]]) for r in per_method.get(m, [])} for m in METHODS}
    pairs = []
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1:]:
            common = [cid for cid in full[a] if cid in full[b]]
            mc = mcnemar([full[a][c] >= 1.0 for c in common], [full[b][c] >= 1.0 for c in common])
            pairs.append((a, b, mc["a_wins"], mc["b_wins"], mc["discordant"], mc["exact_p"]))
            print(f"  {a:>28} vs {b:<28} wins {mc['a_wins']}/{mc['b_wins']} disc {mc['discordant']} p={mc['exact_p']:.4f}")
    # sign test on closure values (soft comparisons)
    print("\n=== sign test on closure values ===")
    for a, b in [(m, n) for i, m in enumerate(METHODS) for n in METHODS[i + 1:]]:
        common = [cid for cid in full[a] if cid in full[b]]
        st = sign_test([full[a][c] for c in common], [full[b][c] for c in common])
        if st["n_gt"] or st["n_lt"]:
            print(f"  {a:>28} vs {b:<28} gt {st['n_gt']} lt {st['n_lt']} p={st['exact_p']:.4f}")

    # P-checks (lines persisted in the JSON so the docs renderer embeds them
    # verbatim; no re-derivation in a second script)
    print("\n=== P-checks ===")
    fullm = report["full"]
    p_lines = []
    p_lines.append(f"P1 (low closure on open questions): closure range "
                   f"{min(fullm[m]['closure'] for m in METHODS):.3f}..{max(fullm[m]['closure'] for m in METHODS):.3f}")
    fy_cases = [cid for cid in full["finplan_v9_cascade"] if fiscal_year_phrasing({"case_id": cid})]
    fy_common = [c for c in fy_cases if c in full["finplan_v8"]]
    mc_fy = mcnemar([full["finplan_v9_cascade"][c] >= 1.0 for c in fy_common],
                    [full["finplan_v8"][c] >= 1.0 for c in fy_common])
    p_lines.append(f"P2 (v9 vs v8 on fiscal-year phrasing, n={len(fy_common)}): "
                   f"v9 wins {mc_fy['a_wins']}, v8 wins {mc_fy['b_wins']}, "
                   f"discordant {mc_fy['discordant']}, exact_p={mc_fy['exact_p']:.4f}")
    leaks = {m: fullm[m]["future_leak"] for m in METHODS}
    p_lines.append(f"P4 (future_leak=0): {leaks}")
    names = report["names_company"]["finplan_v9_cascade"]["closure"] if report["names_company"].get("finplan_v9_cascade") else None
    nonames = report["no_company_name"]["finplan_v9_cascade"]["closure"] if report["no_company_name"].get("finplan_v9_cascade") else None
    p_lines.append(f"P5 (naming surface, v9): named={names} unnamed={nonames}")
    v9 = fullm["finplan_v9_cascade"]["closure"] if fullm.get("finplan_v9_cascade") else None
    meta = fullm["metadata_v9_fill"]["closure"] if fullm.get("metadata_v9_fill") else None
    p_lines.append(f"P6_open (v9 vs metadata_v9_fill): v9={v9} meta_fill={meta}")
    for line in p_lines:
        print(line)

    out = RESULTS / "lofin_full_benchmark_audit_v1.json"
    out.write_text(json.dumps({"schema": "finplan-lofin-full-audit.v1", "per_stratum": report,
                               "mcnemar_pairs": [{"a": a, "b": b, "a_wins": aw, "b_wins": bw,
                                                  "discordant": d, "exact_p": p} for a, b, aw, bw, d, p in pairs],
                               "p2_fiscal_year_phrasing": {"a": "finplan_v9_cascade", "b": "finplan_v8",
                                                           "n_cases": len(fy_common), **mc_fy},
                               "p_checks": p_lines},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
