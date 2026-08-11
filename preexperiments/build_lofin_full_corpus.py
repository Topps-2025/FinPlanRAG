"""Build the full LOFin benchmark corpus on real SEC EDGAR filings (Task #15).

Pre-registered by preexperiments/lofin_full_benchmark_protocol_v1.json (repo,
committed 17ca0d2) BEFORE any download or method run.  This builder:

  1. Loads the 5 original LOFin question files + the verified ticker->CIK map.
  2. Fetches full submission histories (recent + archived tables) per CIK.
  3. Labels fiscal years: gold docs from the LOFin doc_name (authoritative);
     non-gold 10-K/10-Q by the data-driven reportDate/fiscal-calendar mapping
     (protocol amendment_2; companyfacts doesn't expose the fiscal focus).
  4. Builds the 3,008-case table (cutoff = max gold available_at, gold =
     evidence doc_names verbatim, candidate_legs per protocol rule).
  5. Groups questions by (match_companies(question) union {C_q}) and bounds
     each company's download by B_company = max cutoff over its groups.
  6. Downloads normalized filing text (download_text from the repo pilot
     builder, hard-coded SEC UA, resumable on-disk cache).
  7. Writes per-company corpus files, the company registry, the cases file,
     group definitions and a SHA-256 manifest.

External data only under D:/Engineering/FinPlanRAG/database.  Nothing here
modifies any runner or the protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\preexperiments")

import requests
from bs4 import BeautifulSoup

from build_lofin_multidoc_pilot import download_text, normalize
from build_sec_company_registry import normalize_name
from run_nonoracle_obligation_planning_v4 import match_companies

# Hard-coded SEC user agent with contact email (placeholders without a contact
# address are rejected with HTTP 403 by www.sec.gov).
SEC_UA = "FinPlan-RAG academic research research@example.com"

REPO = Path(r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG")
PROTOCOL = REPO / "preexperiments" / "lofin_full_benchmark_protocol_v1.json"
DATA = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\data")
LOFIN = DATA / "external" / "lofin"
SEC_CACHE = DATA / "lofin_full_sec"
TEXT_CACHE = DATA / "lofin_full_corpus_text"
CORPUS_DIR = DATA / "lofin_full_corpus_v1"
# Read only the by_answer_type partition of the LOFin test set (protocol
# amendment_3): textual + numeric_table + numeric_text are pairwise disjoint
# and their union equals all 1,595 unique qids; the by_data_source files
# (finqa/secqa) are fully redundant views (0 qids unique to them) and must
# not enter the question set -- reading them would double-count 1,436 qids.
SUBSETS = ["textual_test.jsonl", "numeric_table_test.jsonl", "numeric_text_test.jsonl"]
DELAY = 0.3
# Diagnostic: log per-file wall time in the download phase (temporary)
_DIAG_TIMING = os.environ.get("LOFIN_DIAG_TIMING", "") == "1"


def fetch_with_deadline(session: requests.Session, url: str, deadline: float = 90.0) -> bytes:
    """Streamed GET with a hard wall-clock deadline per attempt.

    SEC's archive server intermittently throttles (hangs connections, returns
    transient 404s) under bursts; requests' timeout=(10, 10) only bounds each
    socket op, so a slow trickle can stall a run indefinitely without a total
    cap.  The deadline bounds the whole response regardless of chunk pacing.
    """
    t0 = time.time()
    with session.get(url, timeout=(10, 10), stream=True) as resp:
        resp.raise_for_status()
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=1 << 16):
            total += len(chunk)
            chunks.append(chunk)
            if time.time() - t0 > deadline:
                raise TimeoutError(f"download exceeded {deadline:.0f}s wall-clock ({total} bytes)")
        return b"".join(chunks)


def download_text_robust(url: str, session: requests.Session) -> tuple[str, str]:
    """fetch_with_deadline + the protocol's text normalization (identical to
    download_text in build_lofin_multidoc_pilot.py: BeautifulSoup get_text).

    Returns (text, parser_name): parser_name records which parser produced
    the text (lxml preferred; html.parser fallback) for the corpus manifest.
    """
    content = fetch_with_deadline(session, url)
    # Decode ourselves (utf-8, latin-1 fallback) and pass str to lxml so it
    # never runs chardet's pure-Python encoding scan over the full document.
    # py-spy showed every worker parked in dammit.py for 5MB modern 10-Ks;
    # that alone costs ~15s/file and capped the pipeline at ~20 files/min.
    try:
        html = content.decode("utf-8")
    except UnicodeDecodeError:
        html = content.decode("latin-1")
    # Fast path: parse with libxml2 (C) and walk the tree ourselves, skipping
    # <script>/<style> subtrees.  Output is byte-identical to
    # BeautifulSoup(html, "lxml").get_text(" ", strip=True) — verified 0 diff
    # chars on a 64MB multi-exhibit submission — but ~10x faster (BS4's soup
    # conversion is Python-level per node and burns 40-140s on the giant
    # table-heavy 2005-2010 10-Qs/10-Ks).  The explicit stack replaces the
    # XPath ancestor:: filter, which measured 137s on that same file.
    try:
        return normalize(_extract_text_fast(html)), "lxml"
    except Exception as _fast_exc:
        # Last-resort fallback: BS4 html.parser (near-quadratic on the giant
        # table-heavy filings, but correct); its use is recorded per file.
        if _DIAG_TIMING:
            print(f"    [lxml-fallback {type(_fast_exc).__name__}: {str(_fast_exc)[:100]}]", flush=True)
        from bs4 import XMLParsedAsHTMLWarning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            return normalize(BeautifulSoup(html, "html.parser").get_text(" ", strip=True)), "html.parser"


def _extract_text_fast(html: str) -> str:
    """BeautifulSoup get_text(' ', strip=True)-equivalent text extraction.

    Builds the tree with libxml2's html parser (C speed), then walks it with
    an explicit stack, keeping element text and tails in document order and
    dropping <script>/<style> subtrees while preserving their tails (text
    following those tags belongs to the parent and must be kept).
    """
    import lxml.html
    # lxml refuses str input carrying an XML encoding declaration (older
    # XBRL/iXBRL filings open with <?xml ...?>); the declaration is a
    # processing instruction with no text, so dropping it cannot change the
    # extraction.  Without this those files fall to the slow html.parser
    # path (~4% of filings, 15-25s each).
    html = re.sub(r"^\s*<\?xml[^>]*\?>", "", html, count=1)
    tree = lxml.html.document_fromstring(html)
    parts: list[str] = []
    stack = [(tree, False)]  # (element, children_done)
    while stack:
        el, done = stack.pop()
        if el.tag in ("script", "style"):
            if el.tail:
                parts.append(el.tail)
            continue
        if not done:
            if el.text:
                parts.append(el.text)
            stack.append((el, True))
            for child in reversed(list(el)):
                stack.append((child, False))
        elif el.tail:
            parts.append(el.tail)
    return " ".join(p.strip() for p in parts if p.strip())


def find_primary_doc(session: requests.Session, cik: str, accn: str) -> str | None:
    """Directory-listing fallback for empty/stale primaryDocument names.

    Old EDGAR filings (1994-1998, agent-filed) have empty primaryDocument in
    the submissions API; the accn directory lists the real files.  Prefer the
    complete-submission {accn}.txt, else the first .htm.
    """
    accn_compact = accn.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_compact}/"
    content = fetch_with_deadline(session, url, deadline=45.0)
    resp_text = content.decode("utf-8", errors="replace")
    prefix = f"/Archives/edgar/data/{int(cik)}/{accn_compact}/"
    names = sorted({
        link.split("/")[-1]
        for link in re.findall(r'href="([^"]+)"', resp_text, re.I)
        if re.fullmatch(r"[^/]+\.(?:htm|html|txt)", link.split("/")[-1], re.I)
        and link.startswith(prefix)
    })
    if not names:
        return None
    # prefer the complete-submission {accn}.txt (dashed form), else first .htm
    for n in names:
        if n.replace("-", "").startswith(accn_compact) and n.endswith(".txt"):
            return n
    htm = [n for n in names if n.endswith((".htm", ".html"))]
    return htm[0] if htm else names[0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def doc_parse(doc: str) -> dict:
    parts = doc.split("_")
    tail = parts[1:]
    m10k = re.fullmatch(r"20\d{2}", tail[0]) if tail else None
    m10q = re.fullmatch(r"(20\d{2})Q([1-4])", tail[0]) if tail else None
    form = (tail[1] if m10k else tail[1] if m10q else tail[-1] if tail else "?")
    form = {"10K": "10-K", "10Q": "10-Q"}.get(form, form)
    if m10k:
        return {"year": int(m10k.group(0)), "form": form, "quarter": None}
    if m10q:
        return {"year": int(m10q.group(1)), "form": form, "quarter": f"Q{m10q.group(2)}"}
    return {"year": None, "form": form, "quarter": None}


def load_questions() -> list[dict]:
    """Main-set questions (1,572 unique qids, amendment_3) + excluded-23 counters.

    Reads the three by_answer_type files only (SUBSETS).  They are pairwise
    disjoint (verified), so each qid appears exactly once -- the answer-format
    variants seen across the 5-file view (e.g. AAPL/2006/page_100.pdf-1
    "$ 240.41" vs "240.41") cannot arise here; the qid-level assert below is
    the build-time guard on that invariant.
    """
    rows = []
    excluded = 0
    by_subset_excluded = {}
    for name in SUBSETS:
        subset = name.replace("_test.jsonl", "")
        for line in (LOFIN / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            evs = [{"doc_name": str(e["doc_name"]), "page_num": e.get("page_num")} for e in row.get("evidences", [])]
            if any(str(e["doc_name"]).endswith(("_8K", "_EARNINGS")) for e in evs):
                excluded += 1
                by_subset_excluded[subset] = by_subset_excluded.get(subset, 0) + 1
                continue
            rows.append({
                "qid": str(row["qid"]),
                "subset": subset,
                "question": str(row["question"]),
                "answer": row.get("answer"),
                "evidences": evs,
            })
    return rows, excluded, by_subset_excluded


def submissions_all(cik: str, session: requests.Session) -> tuple[dict, list[dict]]:
    """Recent payload + archived file-table payloads, disk-cached."""
    cik10 = f"{int(cik):010d}"
    main_path = SEC_CACHE / f"submissions_CIK{cik10}.json"
    if main_path.exists():
        payload = json.loads(main_path.read_text(encoding="utf-8"))
        table_paths = sorted(SEC_CACHE.glob(f"submissions_CIK{cik10}_files_*.json"))
        claimed = payload.get("filings", {}).get("files", [])
        if len(table_paths) >= len(claimed):
            tables = [payload["filings"]["recent"]]
            tables += [json.loads(p.read_text(encoding="utf-8")) for p in table_paths]
            return payload, tables
        # partial cache (interrupted earlier run: C/831001 had 3 of 39 pages,
        # silently starving its whole history out of the corpus): invalidate
        # and re-fetch rather than build on a truncated archive.
        for stale in [main_path, *table_paths]:
            stale.unlink()
        print(f"    [cache partial {len(table_paths)}/{len(claimed)} pages, re-fetching {cik10}]", flush=True)
    resp = session.get(f"https://data.sec.gov/submissions/CIK{cik10}.json", timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    main_path.write_text(json.dumps(payload), encoding="utf-8")
    tables = [payload["filings"]["recent"]]
    for i, item in enumerate(payload["filings"].get("files", [])):
        r = session.get(f"https://data.sec.gov/submissions/{item['name']}", timeout=60)
        r.raise_for_status()
        table = r.json()
        (SEC_CACHE / f"submissions_CIK{cik10}_files_{i}.json").write_text(json.dumps(table), encoding="utf-8")
        tables.append(table)
        time.sleep(DELAY)
    return payload, tables


def iter_filings(tables: list[dict]):
    for table in tables:
        forms = table.get("form", [])
        n = len(forms)
        for i in range(n):
            yield {
                "form": str(forms[i]),
                "accn": str(table["accessionNumber"][i]) if i < len(table.get("accessionNumber", [])) else "",
                "filing_date": str(table["filingDate"][i]) if i < len(table.get("filingDate", [])) else "",
                "report_date": str(table["reportDate"][i]) if i < len(table.get("reportDate", [])) else "",
                "primary_doc": str(table["primaryDocument"][i]) if i < len(table.get("primaryDocument", [])) else "",
            }


def label_non_gold(f: dict, fy_end_month: int) -> None:
    """Data-driven fiscal labeling (protocol amendment_2):
    10-K: fiscal_year = year(reportDate) - 1 iff reportDate month <= 2
      (a Jan/Feb-dated 10-K ends the fiscal year that began in the prior
      calendar year; a Dec-dated 10-K ends that calendar year).  The -1
      condition keys on the FILING's own reportDate month, NOT the company's
      fy_end_month: AAP/SNA (Jan-end) 10-Ks dated in December were being
      mislabeled one year back (AAP_2024_10K <-> reportDate 2024-12-28).
    10-Q: mapped through the company's fiscal calendar: fiscal_year =
      year(reportDate) + 1 iff reportDate month > fy_end_month; fiscal_period
      = Q{1 + months from fiscal-year start // 3}, fy_start = fy_end_month+1
      (mod 12).  Dec-end companies reduce to the calendar quarter (AMD);
      May-end Oracle's FY2023 Q1 is reportDate 2022-08-31 (gold 10-Q
      ORCL_2023Q1_10Q, protocol amendment_2 evidence)."""
    m = re.match(r"((?:19|20)\d\d)-(\d\d)", f["report_date"])
    if not m:
        return
    y, mo = int(m.group(1)), int(m.group(2))
    if f["form"] == "10-K":
        f["fiscal_year"] = y - 1 if mo <= 2 else y
        f["fiscal_period"] = "FY"
    else:
        fy_start = fy_end_month % 12 + 1
        months_from_start = (mo - fy_start) % 12
        f["fiscal_year"] = y if mo <= fy_end_month else y + 1
        f["fiscal_period"] = f"Q{months_from_start // 3 + 1}"


def main(limit_companies: int | None = None, ticker_filter: str | None = None) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    map_file = json.loads((DATA / "sec_ticker_cik_lofin_final_v1.json").read_text(encoding="utf-8"))
    ticker_map = {t: str(e["cik"]) for t, e in map_file["tickers"].items()}
    if ticker_filter:
        ticker_map = {t: c for t, c in ticker_map.items() if ticker_filter in t}

    rows, excluded, excluded_by_subset = load_questions()
    assert excluded == 23 and excluded_by_subset == {"numeric_table": 1, "textual": 22}, (excluded, excluded_by_subset)
    assert len(rows) == 1572, len(rows)
    assert len({r["qid"] for r in rows}) == len(rows), "duplicate qid across answer-type files (disjointness violated)"

    session = requests.Session()
    session.headers.update({"User-Agent": SEC_UA})
    SEC_CACHE.mkdir(parents=True, exist_ok=True)
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)

    tickers = sorted(ticker_map) if (limit_companies is None or ticker_filter) else sorted(ticker_map)[:limit_companies]
    # build every company that appears as evidence in ANY position: 23 tickers
    # (AEP, CVX, UNH, ...) appear only as second/third evidence in multi-company
    # questions; their gold docs must anchor to their own corpora.
    q_tickers = {ev["doc_name"].split("_")[0] for row in rows for ev in row["evidences"]}
    tickers = [t for t in tickers if t in q_tickers]
    if ticker_filter or limit_companies is not None:
        keep = set(tickers)
        rows = [r for r in rows if {ev["doc_name"].split("_")[0] for ev in r["evidences"]} & keep]

    # ---- Phase 1-2: submissions + fiscal labeling per company ----
    companies: dict[str, dict] = {}
    filings_by_company: dict[str, list[dict]] = {}
    for i, ticker in enumerate(tickers, 1):
        cik = ticker_map[ticker]
        payload, tables = submissions_all(cik, session)
        filings = [f for f in iter_filings(tables) if f["form"] in ("10-K", "10-Q") and f["accn"]]
        filings.sort(key=lambda f: (f["filing_date"], f["accn"]))
        # fiscal-year-end month-day from the latest 10-K reportDate
        k_dates = [f["report_date"] for f in filings if f["form"] == "10-K" and f["report_date"]]
        fy_end = k_dates[-1][:10] if k_dates else ""
        fy_end_month = int(fy_end[5:7]) if len(fy_end) >= 7 else 12
        entry = {
            "ticker": ticker,
            "cik": f"{int(cik):010d}",
            "official_name": str(payload.get("name", ticker)),
            "aliases": sorted({ticker.lower(), normalize_name(str(payload.get("name", ticker))),
                               *[normalize_name(str(n.get("name", ""))) for n in payload.get("formerNames", []) if n.get("name")]}),
            "fiscal_year_end": fy_end,
            "available_filings": [],
        }
        for alias in list(entry["aliases"]):
            stripped = re.sub(r"\b(inc|incorporated|corp|corporation|company|co|plc|ltd|llc|trust|holdings?)\b", " ", alias)
            stripped = re.sub(r"\s+", " ", stripped).strip()
            if stripped:
                entry["aliases"].append(stripped)
        entry["aliases"] = sorted(set(entry["aliases"]))
        for f in filings:
            label_non_gold(f, fy_end_month)  # provisional; gold docs re-labeled below
            entry["available_filings"].append({
                "doc_id": None,  # set in the gold-anchoring pass or from (fy, form, period)
                "accn": f["accn"],
                "fiscal_year": f.get("fiscal_year"),
                "filing_type": f["form"],
                "fiscal_period": f.get("fiscal_period"),
                "available_at": f"{f['filing_date']}T00:00:00.000Z",
                "report_date": f["report_date"],
                "primary_doc": f["primary_doc"],
                "text_path": None,
            })
        filings_by_company[ticker] = entry["available_filings"]
        companies[ticker] = entry
        print(f"[{i}/{len(tickers)}] {ticker} cik={entry['cik']} filings={len(entry['available_filings'])} "
              f"fy_end={fy_end}", flush=True)

    # ---- Phase 2: gold anchoring (LOFin doc_name labels are authoritative) ----
    # Each evidence doc anchors against ITS OWN ticker's filings (doc.split('_')[0]),
    # NOT the first evidence's ticker -- multi-company questions (e.g. openqa_293,
    # "Walmart and Target") carry evidence from several companies.
    for row in rows:
        for doc in sorted({str(e["doc_name"]) for e in row["evidences"]}):
            dt = doc.split("_")[0]
            if dt not in companies:
                continue
            spec = doc_parse(doc)
            pool = [f for f in companies[dt]["available_filings"]
                    if f["filing_type"] == spec["form"] and f.get("fiscal_year") == spec["year"]
                    and (spec["quarter"] is None or f["fiscal_period"] == spec["quarter"])]
            if not pool:
                pool = [f for f in companies[dt]["available_filings"]
                        if f["filing_type"] == spec["form"] and f.get("fiscal_year") == spec["year"]]
            if not pool:
                continue
            filing = pool[0]
            filing["doc_id"] = doc
            filing["fiscal_year"] = spec["year"]
            if spec["quarter"]:
                filing["fiscal_period"] = spec["quarter"]

    # ---- Phase 3: cases (gold doc anchoring + cutoffs) ----
    cases = []
    gold_anchor_failures = []
    for row in rows:
        ticker = row["evidences"][0]["doc_name"].split("_")[0]
        gold_docs = sorted({str(e["doc_name"]) for e in row["evidences"]})
        cutoffs = []
        for doc in gold_docs:
            spec = doc_parse(doc)
            # anchor to the gold doc's OWN ticker's filings (same rule as Phase 2)
            dt = doc.split("_")[0]
            if dt not in companies:
                gold_anchor_failures.append((row["qid"], doc))
                continue
            matches = [f for f in companies[dt]["available_filings"]
                       if f["filing_type"] == spec["form"]
                       and f["fiscal_year"] == spec["year"]
                       and (spec["quarter"] is None or f["fiscal_period"] == spec["quarter"])]
            if spec["quarter"] is None and spec["form"] == "10-Q":
                matches = [f for f in companies[dt]["available_filings"]
                           if f["filing_type"] == "10-Q" and f["fiscal_year"] == spec["year"]]
            if not matches:
                gold_anchor_failures.append((row["qid"], doc))
                continue
            filing = matches[0]
            filing["doc_id"] = doc  # gold doc_id = LOFin doc_name verbatim
            cutoffs.append(filing["available_at"])
        if not cutoffs:
            continue  # unanchorable gold -> excluded at build time, reported
        cutoff = max(cutoffs)
        m = re.match(r"([A-Z0-9.\-]+)/(20\d\d)/", row["qid"])
        if m:
            candidate_legs = [{"ticker": m.group(1), "years": [int(m.group(2))]}]
        else:
            years = sorted({int(y) for doc in gold_docs for y in re.findall(r"(20\d\d)", doc)})
            candidate_legs = [{"ticker": ticker, "years": years}]
        cases.append({
            "case_id": row["qid"],
            "split": "frozen",
            "stratum": row["subset"],
            "template": "lofin_open",
            "question": row["question"],
            "reference_answer": row["answer"],
            "candidate_legs": candidate_legs,
            "candidate_obligations": [],
            "gold_doc_ids": gold_docs,
            "gold_spans": [],
            "cutoff": cutoff,
        })

    # ---- Phase 4: company matching + groups ----
    # match_companies recompiles its alias patterns on cache eviction; the
    # 3008-question x 198-company sweep exceeds re's 512-pattern cache and
    # degrades to O(n^2) recompilation.  Bump the cache for this build pass.
    import re as _re
    _re._MAXCACHE = 200_000
    all_companies = list(companies.values())
    for case in cases:
        ticker = case["gold_doc_ids"][0].split("_")[0]
        matched = {str(c["ticker"]) for c in match_companies(case["question"], all_companies)}
        case["matched_companies"] = sorted(matched)
        case["group_key"] = sorted(matched | {ticker})

    groups: dict[str, dict] = {}
    for case in cases:
        key = tuple(case["group_key"])
        groups.setdefault(key, {"companies": key, "case_ids": [], "cutoffs": []})
        groups[key]["case_ids"].append(case["case_id"])
        groups[key]["cutoffs"].append(case["cutoff"])
    for g in groups.values():
        g["B"] = max(g["cutoffs"])

    # per-company download bound = max B over groups containing the company
    company_B: dict[str, str] = {}
    for g in groups.values():
        for ticker in g["companies"]:
            company_B[ticker] = max(company_B.get(ticker, ""), g["B"])

    # ---- Phase 5: text download (resumable) ----
    download_plan = []
    for ticker, filings in filings_by_company.items():
        bound = company_B.get(ticker)
        if bound is None:
            continue
        for f in filings:
            if f["available_at"] <= bound:
                download_plan.append((ticker, f))
    # dedupe by accn (a filing appears once per company)
    seen_accn = set()
    plan = []
    for ticker, f in download_plan:
        if f["accn"] in seen_accn:
            continue
        seen_accn.add(f["accn"])
        plan.append((ticker, f))
    print(f"download plan: {len(plan)} filings for {len(company_B)} companies", flush=True)

    def download_one(ticker: str, f: dict) -> tuple[str, dict, bool, Exception | None]:
        """Fetch one filing's text into the cache (called from worker threads)."""
        cache = TEXT_CACHE / ticker / f"{f['accn']}.txt"
        if cache.exists():
            return ticker, f, True, None
        cache.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        # data/{company_cik}/{accn}/{doc}: the accn prefix in the accession
        # number is the *agent* CIK for agent-filed submissions, which is
        # NOT mirrored under /Archives/edgar/data/{agent_cik}/ -- always
        # use the company's own CIK directory.
        cik_dir = companies[ticker]["cik"]
        accn_compact = f["accn"].replace("-", "")
        doc = f["primary_doc"]
        # Legacy complete-submission key: data/{cik}/{accn_dashed}.txt.  The
        # S3-backed compact accn dirs intermittently miss objects for old
        # filings (NoSuchKey 404s, e.g. ADI 1994) while the legacy key serves
        # the full submission text.  Used as fallback on 404; recorded in f.
        legacy_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik_dir)}/{f['accn']}.txt"
        last_exc: Exception | None = None
        for attempt in range(6):
            # FRESH session per attempt: EDGAR throttles long-lived keep-alive
            # connections with S3 NoSuchKey 404s while fresh connections serve
            # fine (observed: the build's reused sessions failed every request
            # while the same URLs succeeded on new connections).
            s = requests.Session()
            s.headers.update({"User-Agent": SEC_UA})
            try:
                if not doc:
                    t_fb = time.time()
                    doc = find_primary_doc(s, cik_dir, f["accn"])
                    if not doc:
                        raise FileNotFoundError("no document in directory listing")
                    if _DIAG_TIMING:
                        print(f"    [fallback {time.time() - t_fb:5.1f}s] {ticker} {f['accn']}", flush=True)
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik_dir)}/{accn_compact}/{doc}"
                try:
                    text, parser_used = download_text_robust(url, s)
                    f["source"] = "primary_document"
                except requests.exceptions.HTTPError as exc:
                    if exc.response is None or exc.response.status_code != 404:
                        raise
                    text, parser_used = download_text_robust(legacy_url, s)
                    f["source"] = "legacy_complete_submission"
                cache.write_text(text, encoding="utf-8")
                f["primary_doc"] = doc
                f["parser"] = parser_used
                if _DIAG_TIMING:
                    print(f"    [{time.time() - t0:6.1f}s {parser_used} {f.get('source')}] {ticker} {f['accn']} {doc} {len(text)//1024}KB", flush=True)
                return ticker, f, True, None
            except Exception as exc:
                last_exc = exc
                retriable = (isinstance(exc, requests.exceptions.HTTPError)
                             and exc.response is not None
                             and exc.response.status_code in (404, 403, 500, 502, 503))
                if retriable and not doc:
                    doc = ""  # stale primaryDocument -> force listing fallback
                # EDGAR's S3-backed archive intermittently 404s valid objects
                # (mirror inconsistency; S3 NoSuchKey XML bodies).  Back off
                # exponentially and rotate to a fresh connection next attempt
                # so the retry may land on a different origin.
                if retriable:
                    time.sleep(min(60, 2 ** (attempt + 1)) + random.uniform(0, 2))
                else:
                    time.sleep(2 * (attempt + 1))
        return ticker, f, False, last_exc

    failed: list[str] = []
    todo: list[tuple[str, dict]] = []
    for ticker, f in plan:
        cache = TEXT_CACHE / ticker / f"{f['accn']}.txt"
        if cache.exists():
            f["text_path"] = str(cache)
        else:
            todo.append((ticker, f))
    print(f"download plan: {len(plan)} filings ({len(todo)} to fetch)", flush=True)
    if todo:
        from concurrent.futures import ThreadPoolExecutor
        done = 0
        with ThreadPoolExecutor(max_workers=8) as ex:
            for ticker, f, ok, err in ex.map(lambda item: download_one(*item), todo):
                if ok:
                    f["text_path"] = str(TEXT_CACHE / ticker / f"{f['accn']}.txt")
                else:
                    failed.append(f"{ticker} {f['accn']} {type(err).__name__}: {err}")
                    if _DIAG_TIMING:
                        print(f"    [FAIL] {ticker} {f['accn']} {type(err).__name__}: {err}", flush=True)
                done += 1
                if done % 100 == 0 or done == len(todo):
                    print(f"  downloaded {done}/{len(todo)} ({len(failed)} failed)", flush=True)

    # ---- Phase 6: write outputs ----
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    corpus_manifest = {}
    for ticker, entry in companies.items():
        bound = company_B.get(ticker)
        docs = []
        for f in entry["available_filings"]:
            if f["text_path"] is None or (bound and f["available_at"] > bound):
                continue
            spec = doc_parse(f["doc_id"]) if f.get("doc_id") else None
            if f["doc_id"] is None:
                f["doc_id"] = (f"{ticker}_{f['fiscal_year']}_10K" if f["fiscal_period"] == "FY"
                               else f"{ticker}_{f['fiscal_year']}{f['fiscal_period']}_10Q")
            docs.append({
                "doc_id": f["doc_id"],
                "ticker": ticker,
                "year": f["fiscal_year"],
                "form": f["filing_type"],
                "fiscal_period": f["fiscal_period"],
                "cik": entry["cik"],
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(f['accn'][:10])}/{f['accn'].replace('-', '')}/{f['primary_doc']}",
                "accession": f["accn"],
                "filing_date": f["available_at"][:10],
                "report_date": f["report_date"],
                "available_at": f["available_at"],
                "parser": f.get("parser", "html.parser"),
                "source": f.get("source", "primary_document"),
                "text_sha256": sha256(Path(f["text_path"])),
                "text": Path(f["text_path"]).read_text(encoding="utf-8"),
            })
        docs.sort(key=lambda d: (d["available_at"], d["doc_id"]))
        entry["available_years"] = sorted({int(d["year"]) for d in docs})
        entry["available_filings"] = [
            {k: v for k, v in f.items() if k in ("doc_id", "accn", "fiscal_year", "filing_type", "fiscal_period", "available_at", "report_date")}
            for f in entry["available_filings"] if f.get("doc_id") and f["text_path"] is not None and (bound is None or f["available_at"] <= bound)
        ]
        out_file = CORPUS_DIR / f"{ticker}.json"
        out = {
            "schema": "finplan-lofin-full-corpus-company.v1",
            "ticker": ticker,
            "cik": entry["cik"],
            "official_name": entry["official_name"],
            "aliases": entry["aliases"],
            "available_years": entry["available_years"],
            "B": bound,
            "documents": docs,
        }
        out_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        corpus_manifest[ticker] = {"file": str(out_file), "sha256": sha256(out_file), "n_documents": len(docs)}
        # aliases must survive in the registry (match_companies reads them)

    registry_out = {
        "schema": "finplan-lofin-full-registry.v1",
        "status": "public-corpus-metadata",
        "protocol_sha256": sha256(PROTOCOL),
        "map_sha256": sha256(DATA / "sec_ticker_cik_lofin_final_v1.json"),
        "companies": companies,
    }
    (DATA / "lofin_full_registry_v1.json").write_text(json.dumps(registry_out, ensure_ascii=False, indent=2), encoding="utf-8")
    for case in cases:
        case.pop("matched_companies", None)
        case.pop("group_key", None)
    cases_out = {
        "schema": "finplan-lofin-full-cases.v1",
        "protocol_sha256": sha256(PROTOCOL),
        "n_cases": len(cases),
        "cases": cases,
    }
    (DATA / "lofin_full_cases_v1.json").write_text(json.dumps(cases_out, ensure_ascii=False, indent=2), encoding="utf-8")
    groups_out = {
        "schema": "finplan-lofin-full-groups.v1",
        "protocol_sha256": sha256(PROTOCOL),
        "n_groups": len(groups),
        "groups": {f"g{i}": {"companies": g["companies"], "case_ids": g["case_ids"], "B": g["B"]}
                   for i, g in enumerate(sorted(groups.values(), key=lambda g: g["companies"]))},
    }
    (DATA / "lofin_full_groups_v1.json").write_text(json.dumps(groups_out, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schema": "finplan-lofin-full-corpus-manifest.v1",
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "protocol_sha256": sha256(PROTOCOL),
        "generated_at": "2026-08-11",
        "n_questions": len(rows),
        "n_excluded_8k_earnings": excluded,
        "n_cases": len(cases),
        "n_groups": len(groups),
        "n_downloads": len(plan),
        "n_download_failures": len(failed),
        "download_failures": failed,
        "gold_anchor_failures": gold_anchor_failures,
        "registry": {"file": "lofin_full_registry_v1.json", "sha256": sha256(DATA / "lofin_full_registry_v1.json")},
        "cases_file": {"file": "lofin_full_cases_v1.json", "sha256": sha256(DATA / "lofin_full_cases_v1.json")},
        "groups_file": {"file": "lofin_full_groups_v1.json", "sha256": sha256(DATA / "lofin_full_groups_v1.json")},
        "corpus_files": corpus_manifest,
    }
    (DATA / "lofin_full_corpus_manifest_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("corpus_files", "download_failures")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(limit_companies=int(sys.argv[1]) if len(sys.argv) > 1 else None,
         ticker_filter=sys.argv[2] if len(sys.argv) > 2 else None)
