"""FinGLM baselines runner (Task #21): single_shot / period_metadata_decomposition
/ generic_adaptive_period / hirec_period on the FinGLM full corpus.

Chinese-adapted but the *same* retrieve policies as the LOFin baselines
(run_period_aware_baselines.retrieve_policy is imported verbatim; the policy
is language-agnostic - it only reads obligation fields and the BM25 index
interface).  Differences vs LOFin, all documented in the protocol:
- tokenizer: jieba (len>=2 CJK words or digit strings), no English regex;
- index: per-document single chunk (annual report = one unit), BM25 formula
  identical (k1=2.2, b=0.75, same idf/log smoothing);
- obligation rule: Chinese question -> (year, company alias) -> the single
  (code, year) annual report; filing_type is the annual-report analogue 10-K,
  fiscal_period FY;
- corpus is read streaming per-code (11,588 docs ~1.7GB text must never be
  fully resident); chunk.text is dropped after tokenization.

Memory note (2026-08-12): the original scan implementation kept every
document's full token list resident in the chunks list (~916M token
instances, ~30GB virtual) and died with MemoryError on the 13.9GB machine
after ~80min of tokenization.  The index is now a streaming inverted index:
each document's term list is counted and dropped immediately; the index
keeps only packed (chunk_idx, tf) postings (~8 bytes per unique term-doc
pair; ~90M pairs ~0.7GB).  Ranking is byte-identical to the scan version -
same BM25 formula (k1=2.2, b=0.75, same idf/log smoothing), same
(-score, available_at, chunk_id) tie-break, same dedup-by-doc_id; float
accumulation stays term-major in query order so every score sum is
bit-for-bit the same.

run(corpus_dir, registry_path, cases_path, out_path, budget) mirrors the
LOFin runner interface shape; outputs rows + trajectories + summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from array import array
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import jieba

sys.path.insert(0, r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\preexperiments")
from run_period_aware_baselines import retrieve_policy
from run_sec_real_pilot import parse_time

METHODS = ["single_shot", "period_metadata_decomposition",
           "generic_adaptive_period", "hirec_period"]

_CN_WORD = re.compile(r"^[\u4e00-\u9fff]{2,}$")
_CN_DIGIT = re.compile(r"^\d[\d,.]*$")


def cn_tokens(text: str) -> tuple[str, ...]:
    """jieba segmentation, keeping CJK words (>=2 chars) and digit strings.

    Single-char function words (的/了/是) and bare punctuation carry no
    retrieval signal for annual-report QA and would dominate the counters.
    """
    out = []
    for w in jieba.lcut(text):
        w = w.strip()
        if not w:
            continue
        if _CN_DIGIT.match(w) or _CN_WORD.match(w):
            out.append(w)
    return tuple(out)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    lineage_id: str
    role: str
    available_at: datetime
    text: str  # diagnostic only; not the full report
    terms: tuple[str, ...]  # dropped after indexing (always ()); dl lives in CNBM25


class CNBM25:
    """Same BM25 scoring as run_sec_real_pilot.BM25, built streaming.

    Holds a compact inverted index: term -> array('I') of interleaved
    (chunk_idx, tf) pairs, chunk_idx ascending.  document lengths (dl) are
    kept per chunk.  search() replicates the original O(N) scan exactly:
    same idf formula, same term-major accumulation order (bit-identical
    scores), same filters (used / allow_future / required_terms), same
    (-score, available_at, chunk_id) sort and dedup-by-doc_id.
    """

    def __init__(self, chunks: Sequence[Chunk], postings: dict[str, array],
                 dl: Sequence[int]) -> None:
        self.chunks = list(chunks)
        self.postings = postings
        self.dl = list(dl)
        self.avgdl = sum(self.dl) / max(1, len(self.chunks))

    def search(self, query: str, cutoff: datetime, used: Sequence[str],
               allow_future: bool, limit: int = 1,
               required_terms: Sequence[str] = ()) -> list[Chunk]:
        qterms = cn_tokens(query)
        used_set = set(used)
        n = len(self.chunks)
        # candidates: chunks holding every required term (path markers)
        if required_terms:
            present = []
            for t in required_terms:
                p = self.postings.get(t)
                if p is None:
                    return []
                present.append(p)
            base = min(present, key=len)
            candidates = {base[j] for j in range(0, len(base), 2)}
            for p in present:
                if p is base:
                    continue
                keep = {p[j] for j in range(0, len(p), 2)}
                candidates.intersection_update(keep)
                if not candidates:
                    return []
        else:
            candidates = None
        scores: dict[int, float] = {}
        for t in qterms:
            p = self.postings.get(t)
            if p is None:
                continue
            idf = math.log(1.0 + (n - (len(p) // 2) + 0.5) / ((len(p) // 2) + 0.5))
            for j in range(0, len(p), 2):
                i, f = p[j], p[j + 1]
                if candidates is not None and i not in candidates:
                    continue
                ch = self.chunks[i]
                if ch.chunk_id in used_set:
                    continue
                if not allow_future and ch.available_at > cutoff:
                    continue
                dl = self.dl[i]
                scores[i] = scores.get(i, 0.0) + idf * (f * 2.2) / (
                    f + 1.2 * (0.25 + 0.75 * dl / max(1.0, self.avgdl)))
        scored = [(scores[i], self.chunks[i]) for i in scores]
        scored.sort(key=lambda item: (-item[0], item[1].available_at, item[1].chunk_id))
        out: list[Chunk] = []
        docs: set[str] = set()
        for _, chunk in scored:
            if chunk.doc_id in docs:
                continue
            out.append(chunk)
            docs.add(chunk.doc_id)
            if len(out) >= limit:
                break
        return out


def precise_path(obligation: dict) -> str:
    form = str(obligation["filing_type"]).replace("-", "").lower()
    return (f"path{str(obligation['ticker']).lower()}"
            f"{int(obligation['fiscal_year'])}{str(obligation['fiscal_period']).lower()}{form}")


def chinese_obligation_rule(
    question: str, companies: dict,
) -> tuple[list[dict], dict[str, str]]:
    """Chinese question -> obligation(s).

    FinGLM 复赛A questions name exactly one company and one report year
    ("根据2020年XX公司的年报..."), so the obligation is deterministic:
    longest alias (full name, short name) present in the question, plus the
    first 4-digit year.  Returns [obligation], {code: reason}.
    """
    m = re.search(r"(20\d{2})年", question)
    year = int(m.group(1)) if m else None
    # longest-first alias match
    matches = []
    for code, ent in companies.items():
        for alias in ent["aliases"]:
            if alias and alias in question:
                matches.append((len(alias), code, alias))
    matches.sort(reverse=True)
    if not matches:
        return [], {}
    _, code, _ = matches[0]
    reasons = {code: "explicit-year-alias"}
    filings = {f["fiscal_year"]: f for f in companies[code]["available_filings"]}
    if year is None or year not in filings:
        return [], reasons
    f = filings[year]
    obligation = {"ticker": code, "fiscal_year": int(f["fiscal_year"]),
                  "filing_type": str(f["filing_type"]),
                  "fiscal_period": str(f["fiscal_period"]),
                  "doc_id": str(f["doc_id"])}
    return [obligation], reasons


def run(corpus_dir: Path, registry_path: Path, cases_path: Path,
        out_path: Path, budget: int) -> dict:
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    companies = registry["companies"]
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))["cases"]

    # streaming index build: one chunk per document; term lists are counted
    # and dropped immediately, only (chunk_idx, tf) postings stay resident
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
            # identical path markers as LOFin's make_period_chunks: the
            # fill_missing policy matches required_terms against them
            generic_path = f"path{str(doc['ticker']).lower()}{int(doc['year'])}"
            terms = cn_tokens(doc["text"])
            counts = Counter(terms)
            idx = len(chunks)
            for term, f in counts.items():
                p = postings.get(term)
                if p is None:
                    p = postings[term] = array("I")
                p.append(idx)
                p.append(f)
            chunks.append(Chunk(
                chunk_id=f"{doc['doc_id']}::c0",
                doc_id=doc["doc_id"],
                lineage_id=doc["doc_id"],
                role=str(doc.get("form", "10-K")),
                available_at=parse_time(doc["available_at"]),
                text=doc["text"][:500],
                terms=(),
            ))
            dl.append(len(terms))
    index = CNBM25(chunks, postings, dl)
    print(f"index: {len(chunks)} docs, {len(index.postings)} unique terms, "
          f"avgdl={index.avgdl:.0f}", flush=True)

    rows, trajectories = [], []
    for case in cases:
        obligations, reasons = chinese_obligation_rule(
            str(case["question"]), companies)
        gold = {str(d) for d in case["gold_doc_ids"]}
        for method in METHODS:
            prediction = retrieve_policy(
                str(case["question"]), str(case["cutoff"]), obligations,
                index, budget, method)
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
            })
            trajectories.append({
                "case_id": case["case_id"],
                "method": method,
                "planned_obligations": obligations,
                "entity_candidates": sorted(reasons),
                **prediction,
            })

    summary = {}
    for method in METHODS:
        rs = [r for r in rows if r["method"] == method]
        summary[method] = {k: (sum(r[k] for r in rs) / len(rs) if len(rs) else 0.0)
                           for k in ("leg_recall", "closure", "wrong_doc_rate",
                                     "queries", "documents")}
    out = {"rows": rows, "trajectories": trajectories, "summary": summary}
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
                    default=Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\results\finglm_full_baselines.json"))
    ap.add_argument("--budget", type=int, default=4)
    args = ap.parse_args()
    run(args.corpus_dir, args.registry, args.cases, args.out, args.budget)


if __name__ == "__main__":
    main()
