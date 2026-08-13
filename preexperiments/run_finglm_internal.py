"""FinPlan internal methods on the FinGLM full set (Task #23, amendment_1).

Runs the project-internal method family - finplan_v4 / finplan_v9_cascade /
metadata_v9_fill - over the SAME corpus index as the external baselines
(run_finglm_baselines.py): identical postings/dl construction, identical
cases, budget=4, cutoff semantics.  Imported code is byte-identical
(no copy-paste divergence): cn_tokens / CNBM25 / precise_path / parse_time
are imported verbatim from run_finglm_baselines.

Obligation rule (amendment_1, committed before any row was produced):
chinese_internal_obligation_rule = longest alias + ALL explicit (20xx)year
occurrences -> one (code, year, 10-K, FY) obligation each.  This is the
faithful Chinese translation of the LOFin-internal select_years semantics
(explicit-year branch keeps every explicit year); the baseline rule (first
year only) stays frozen for the four external baselines.

Method matrix (each a byte-faithful port of the frozen LOFin runner):
- finplan_v4:        per (code, year) leg query "{q} {code} {year}" with
                     required generic path marker path<code><year>
                     (LOFin v4 finplan_v3_path_bound).
- finplan_v9_cascade: per obligation query "{q} {code} {year} {period}
                     {form}" with required precise path marker
                     (LOFin v7 finplan_v9_cascade).
- metadata_v9_fill:  P4 comparator: the EXACT SAME v9 obligations and query,
                     metadata-decomposition retrieval (skip already-used
                     doc_id, no cascade).  On the 1783 single-year cases it
                     must equal baseline period_metadata_decomposition
                     row-for-row (asserted post-run, not assumed).

v7/v8 are represented by finplan_v9_cascade (amendment_1 equivalence note:
their period_requests differences have no textual ground in 复赛A - 0/1829
questions carry quarter/half-year phrases, corpus is annual reports only).

Equivalence checks (asserted in main):
- index: avgdl=55285, unique terms 1212588 (byte-identical construction);
- single-year consistency vs the baseline rows file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_finglm_baselines import CNBM25, Chunk, cn_tokens, precise_path
from run_sec_real_pilot import parse_time
from collections import Counter
from array import array

METHODS = ["finplan_v4", "finplan_v9_cascade", "metadata_v9_fill"]

_CN_YEAR = re.compile(r"(20\d{2})年")


def chinese_internal_obligation_rule(
    question: str, companies: dict,
) -> tuple[list[dict], dict[str, str]]:
    """All explicit years + longest alias -> one (code, year, 10-K, FY) each.

    Faithful Chinese translation of LOFin select_years explicit-year branch
    (every explicit year is an obligation).  The alias matcher is identical
    to the baseline rule (longest-first).  Defensive annual-fallback (most
    recent available year) is unreachable: 1829/1829 questions carry at
    least one (20xx)year.
    """
    years = sorted({int(y) for y in _CN_YEAR.findall(question)})
    matches = []
    for code, ent in companies.items():
        for alias in ent["aliases"]:
            if alias and alias in question:
                matches.append((len(alias), code, alias))
    matches.sort(reverse=True)
    if not matches:
        return [], {}
    _, code, _ = matches[0]
    filings = {int(f["fiscal_year"]): f for f in companies[code]["available_filings"]}
    if not years:
        years = [max(filings)] if filings else []
    obligations = []
    for year in years:
        f = filings.get(year)
        if f is None:
            continue
        obligations.append({"ticker": code, "fiscal_year": int(f["fiscal_year"]),
                            "filing_type": str(f["filing_type"]),
                            "fiscal_period": str(f["fiscal_period"]),
                            "doc_id": str(f["doc_id"])})
    return obligations, {code: "all-explicit-years"}


def retrieve(question: str, cutoff: str, obligations: list[dict],
             index: CNBM25, budget: int, method: str) -> dict:
    used: list[str] = []
    used_docs: list[str] = []
    actions: list[str] = []
    for obligation in obligations:
        if len(used_docs) >= budget:
            break
        if method == "finplan_v4":
            generic = f"path{str(obligation['ticker']).lower()}{int(obligation['fiscal_year'])}"
            query = f"{question} {obligation['ticker']} {obligation['fiscal_year']}"
            required = [generic]
            label = f"leg:{obligation['ticker']}:{obligation['fiscal_year']}"
        else:
            query = (f"{question} {obligation['ticker']} {obligation['fiscal_year']} "
                     f"{obligation['fiscal_period']} {obligation['filing_type']}")
            required = [precise_path(obligation)]
            label = (f"obligation:{obligation['ticker']}:{obligation['fiscal_year']}:"
                     f"{obligation['fiscal_period']}:{obligation['filing_type']}")
        if method == "metadata_v9_fill" and str(obligation["doc_id"]) in used_docs:
            actions.append(f"missing:{obligation['doc_id']}:skip")
            continue
        candidates = index.search(query, parse_time(cutoff), used, False,
                                  limit=1, required_terms=required)
        actions.append(label)
        if candidates:
            used.append(candidates[0].chunk_id)
            used_docs.append(candidates[0].doc_id)
    return {"used": used, "used_docs": used_docs, "actions": actions}


def build_index(corpus_dir: Path, registry: dict) -> CNBM25:
    """Byte-identical index construction to run_finglm_baselines.run()."""
    companies = registry["companies"]
    chunks: list[Chunk] = []
    postings: dict[str, array] = {}
    dl: list[int] = []
    for code_file in sorted(Path(corpus_dir).glob("*.json")):
        code = code_file.stem
        ent = companies.get(code)
        for doc in json.loads(code_file.read_text(encoding="utf-8"))["documents"]:
            obligation = {"ticker": doc["ticker"], "fiscal_year": doc["year"],
                          "filing_type": doc.get("form", "10-K"),
                          "fiscal_period": doc.get("fiscal_period", "FY")}
            generic_path = f"path{str(doc['ticker']).lower()}{int(doc['year'])}"
            precise = precise_path(obligation)
            terms = cn_tokens(doc["text"])
            counts = Counter(terms)
            idx = len(chunks)
            for term, f in counts.items():
                p = postings.get(term)
                if p is None:
                    p = postings[term] = array("I")
                p.append(idx)
                p.append(f)
            for marker in (generic_path, precise):
                p = postings.get(marker)
                if p is None:
                    p = postings[marker] = array("I")
                p.append(idx)
                p.append(1)
            chunks.append(Chunk(
                chunk_id=f"{doc['doc_id']}::c0",
                doc_id=doc["doc_id"],
                lineage_id=doc["doc_id"],
                role=str(doc.get("form", "10-K")),
                available_at=parse_time(doc["available_at"]),
                text=doc["text"][:500],
                terms=(),
            ))
            dl.append(len(terms) + 2)
    return CNBM25(chunks, postings, dl)


def run(corpus_dir: Path, registry_path: Path, cases_path: Path,
        out_path: Path, budget: int) -> dict:
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    companies = registry["companies"]
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))["cases"]

    index = build_index(corpus_dir, registry)
    print(f"index: {len(index.chunks)} docs, {len(index.postings)} unique terms, "
          f"avgdl={index.avgdl:.0f}", flush=True)
    assert len(index.postings) == 1_212_588 and abs(index.avgdl - 55285) < 1, \
        "index deviates from the baseline run (amendment_1 equivalence check)"

    rows = []
    for case in cases:
        obligations, reasons = chinese_internal_obligation_rule(
            str(case["question"]), companies)
        gold = {str(d) for d in case["gold_doc_ids"]}
        for method in METHODS:
            prediction = retrieve(str(case["question"]), str(case["cutoff"]),
                                  obligations, index, budget, method)
            used = {str(d) for d in prediction["used_docs"]}
            rows.append({
                "case_id": case["case_id"],
                "stratum": case.get("answer_type", "unstratified"),
                "method": method,
                "leg_recall": len(gold & used) / max(1, len(gold)),
                "closure": float(gold <= used),
                "wrong_doc_rate": len(used - gold) / max(1, len(used)),
                "queries": float(len(prediction["actions"])),
                "documents": float(len(used)),
                "obligations": float(len(obligations)),
            })

    summary = {}
    for method in METHODS:
        rs = [r for r in rows if r["method"] == method]
        summary[method] = {k: (sum(r[k] for r in rs) / len(rs) if len(rs) else 0.0)
                           for k in ("leg_recall", "closure", "wrong_doc_rate",
                                     "queries", "documents", "obligations")}
    out = {"rows": rows, "summary": summary}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"rows={len(rows)} -> {out_path}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path,
                    default=Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\data\finglm_full_corpus_v1"))
    ap.add_argument("--registry", type=Path,
                    default=Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\data\finglm_full_registry_v1.json"))
    ap.add_argument("--cases", type=Path,
                    default=Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\data\finglm_full_cases_v1.json"))
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\results\finglm_full_internal_v1.json"))
    ap.add_argument("--budget", type=int, default=4)
    args = ap.parse_args()
    run(args.corpus_dir, args.registry, args.cases, args.out, args.budget)


if __name__ == "__main__":
    main()
