"""Build the FinGLM full-run registry, cases and text corpus (Task #21 prep).

Data lineage (all public, external-data area, read-only upstream):
- 11,588 annual reports: ModelScope modelscope/chatglm_llm_fintech_raw_dataset
  `allpdf/` (LFS), names byte-identical to MetaGLM/FinGLM reports_list.csv.
- 2,000 QA pairs: FinGLM 复赛A questions/answers (JSONL, local snapshot).

FinGLM has NO original gold document sets (audit_china_public_financial_data
recorded gold_document_sets=False).  The per-question gold document set is
DERIVED from the answer prompt metadata (ent_name + year -> the single annual
report).  This is a self-built evaluation convention and is declared as such
in the protocol; it is not an original dataset annotation.

Exclusions (recorded with reasons, reported honestly):
- prompt empty / no ent_name: 88 generic accounting-concept questions
  (no company/year in the question -> no derivable gold document);
- ent_name not in the reports list (renamed/newly listed), or the
  (code, year) report missing from the list.

Corpus storage: data/finglm_full_corpus_v1/<code>.json, one document per
report year (doc_id = <code>_<year>_annual), resumable at (code, year)
granularity: years already written are skipped, partial per-code files are
completed (never skipped whole — incremental runs must not drop a code's
remaining years).  PDF text via pypdfium2 (page-level, joined by \n\n),
same lineage as the frozen China page pilot.

Usage:
  python build_finglm_full_corpus.py [--phase registry|text|all] [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments")
DATA = ROOT / "data"
CSV = DATA / "external" / "china" / "finglm_reports_list.csv"
QUESTIONS = DATA / "external" / "china" / "finglm_A_questions.json"
ANSWERS = DATA / "external" / "china" / "finglm_A_answers.json"
PDF_DIR = DATA / "external" / "china" / "documents" / "finglm_full"
OUT_DIR = DATA / "finglm_full_corpus_v1"
REGISTRY = DATA / "finglm_full_registry_v1.json"
CASES = DATA / "finglm_full_cases_v1.json"

NAME_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})__(?P<full>.+?)__(?P<code>\d{6})__"
    r"(?P<short>.+?)__(?P<year>\d{4})年__年度报告\.pdf$")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def parse_report_list() -> dict[str, dict]:
    """code -> {code, name(full), short, years: {year: {date, filename}}}"""
    out: dict[str, dict] = {}
    with open(CSV, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            m = NAME_PATTERN.match(row["name"])
            if not m:
                print("unparsed name:", row["name"])
                continue
            d = m.groupdict()
            code = d["code"]
            ent = out.setdefault(code, {"code": code, "name": d["full"],
                                        "short": d["short"], "years": {}})
            ent["years"][int(d["year"])] = {"date": d["date"], "filename": row["name"]}
    return out


def build_registry(companies: dict[str, dict]) -> None:
    reg = {"companies": {}}
    for code, ent in companies.items():
        filings = [{"doc_id": f"{code}_{year}_annual", "fiscal_year": year,
                    "filing_type": "10-K", "fiscal_period": "FY",
                    "available_at": info["date"]}
                   for year, info in sorted(ent["years"].items())]
        reg["companies"][code] = {
            "ticker": code,
            "name": ent["name"],
            "short_name": ent["short"],
            "aliases": [ent["name"], ent["short"], code],
            "available_filings": filings,
        }
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"registry: {len(reg['companies'])} companies -> {REGISTRY}")


def build_cases(companies: dict[str, dict]) -> None:
    questions = load_jsonl(QUESTIONS)
    answers = load_jsonl(ANSWERS)
    by_id = {a["id"]: a for a in answers}
    cases, excluded = [], []
    for q in questions:
        a = by_id.get(q["id"], {})
        pr = a.get("prompt") or {}
        ent, year_s = pr.get("ent_name"), pr.get("year")
        reason = None
        code = None
        if not ent or not year_s:
            reason = "no_prompt_generic_concept_question"
        else:
            year = int(year_s)
            for c, e in companies.items():
                if e["name"] == ent:
                    code = c
                    break
            if code is None:
                reason = "ent_name_not_in_reports_list"
            elif year not in companies[code]["years"]:
                reason = "report_year_missing"
        if reason:
            excluded.append({"id": q["id"], "question": q["question"],
                             "reason": reason,
                             "ent_name": ent, "year": year_s})
            continue
        year = int(year_s)
        date = companies[code]["years"][year]["date"]
        cases.append({
            "case_id": f"finglm_{q['id']}",
            "question": q["question"],
            "company_code": code,
            "company_name": companies[code]["name"],
            "fiscal_year": year,
            "gold_doc_ids": [f"{code}_{year}_annual"],
            "cutoff": date,
            "answer_type": a.get("type", "?"),
        })
    CASES.write_text(json.dumps({"cases": cases, "excluded": excluded},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    reasons = Counter(x["reason"] for x in excluded)
    print(f"cases: {len(cases)} resolved, {len(excluded)} excluded "
          f"(rate {len(cases)/(len(cases)+len(excluded)):.4f}); reasons: "
          f"{dict(reasons)} -> {CASES}")


def extract_text(pdf_path: Path) -> str:
    pdf = pdfium.PdfDocument(pdf_path)
    parts = []
    for i in range(len(pdf)):
        parts.append(pdf[i].get_textpage().get_text_range())
    return "\n\n".join(parts).replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_job(j) -> dict:
    """Module-level worker (picklable for ProcessPoolExecutor on Windows)."""
    code, ent, year, pdf, out = j
    try:
        t0 = time.time()
        text = extract_text(pdf)
        doc = {"doc_id": f"{code}_{year}_annual", "ticker": code,
               "year": year, "form": "10-K", "fiscal_period": "FY",
               "available_at": ent["years"][year]["date"],
               "bytes": pdf.stat().st_size,
               "chars": len(text),
               "text": text,
               "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
               "elapsed_s": round(time.time() - t0, 2)}
        return {"code": code, "year": year, "status": "ok", "doc": doc}
    except Exception as exc:
        return {"code": code, "year": year, "status": "error", "error": str(exc)}


def write_code_file(out: Path, docs: list[dict]) -> int:
    """Merge a code's freshly parsed docs into its per-code file.

    Merge semantics identical to the old batch-end write; only the flush
    timing differs (per-code, on completion, instead of batch-end).
    """
    existing = []
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))["documents"]
    combined = {d["doc_id"]: d for d in existing + docs}
    out.write_text(json.dumps({"documents": list(combined.values())},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    return len(combined)


def build_text(companies: dict[str, dict], workers: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    pending = 0
    for code, ent in companies.items():
        out = OUT_DIR / f"{code}.json"
        # resumable at (code, year) granularity, NOT per code: a per-code
        # file may be partial when an earlier run was incremental (its job
        # list was a snapshot of PDFs available at submission time; codes
        # with some years pending are flushed per-code on completion of the
        # submitted subset).  Skipping the whole code would permanently
        # drop those missing years (measured: 1,972 partial files / 3,355
        # missing reports).  Only years already written are skipped.
        written = set()
        if out.exists():
            written = {d["doc_id"] for d in
                       json.loads(out.read_text(encoding="utf-8"))["documents"]}
        for year, info in ent["years"].items():
            if f"{code}_{year}_annual" in written:
                continue  # already parsed (this run or an earlier one)
            pdf = PDF_DIR / info["filename"]
            if not pdf.exists():
                # incremental mode: PDF not yet downloaded; a later run
                # picks it up (its doc_id is still absent, so the
                # resumable skip logic re-submits it next round)
                pending += 1
                continue
            jobs.append((code, ent, year, pdf, out))

    print(f"text phase: {len(jobs)} reports to parse, workers={workers}, "
          f"{pending} pending download (picked up on a later run)", flush=True)
    per_code: dict[str, list[dict]] = {}
    remaining = Counter(j[0] for j in jobs)
    status = {"ok": 0, "error": 0}
    n_docs = 0
    t_start = time.time()
    # ProcessPoolExecutor: pdfium holds the GIL during page text extraction;
    # threads give ~1x scaling (measured), processes scale ~linearly (8 procs
    # = 20x single-process throughput on a mixed 6-PDF sample).
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(parse_job, j) for j in jobs]
        done = 0
        for fut in as_completed(futs):
            done += 1
            rec = fut.result()
            status[rec["status"]] += 1
            if rec["status"] == "ok":
                per_code.setdefault(rec["code"], []).append(rec["doc"])
            remaining[rec["code"]] -= 1
            if remaining[rec["code"]] == 0 and rec["code"] in per_code:
                # flush as soon as a code's last job completes: a
                # batch-end-only write loses ALL in-memory work if the
                # process is killed mid-batch (observed: the first batch-1
                # run died and ~1,000 parsed reports were lost)
                n_docs += write_code_file(OUT_DIR / f"{rec['code']}.json",
                                          per_code.pop(rec["code"]))
            if done % 200 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)} ok={status['ok']} "
                      f"err={status['error']} @ {time.time()-t_start:.0f}s", flush=True)
    # safety pass: flush any code left in memory (should not happen)
    for code, docs in per_code.items():
        n_docs += write_code_file(OUT_DIR / f"{code}.json", docs)
    print(f"text done: {n_docs} documents, {status['error']} errors -> {OUT_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["registry", "text", "all"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    companies = parse_report_list()
    print(f"reports list: {sum(len(e['years']) for e in companies.values())} "
          f"reports, {len(companies)} companies")
    if args.phase in ("registry", "all"):
        build_registry(companies)
        build_cases(companies)
    if args.phase in ("text", "all"):
        build_text(companies, args.workers)


if __name__ == "__main__":
    main()
