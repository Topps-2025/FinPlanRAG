"""Build a real multi-document LOFin pilot from official SEC 10-K filings.

The QA/evidence annotations come from LOFin. The builder resolves each
``TICKER_YEAR_10K`` evidence document to the corresponding SEC filing, records
its acceptance time, downloads the primary filing, and creates document-group
disjoint exploratory/holdout splits. Evidence page numbers are preserved, but
this pilot evaluates document-level retrieval closure because SEC HTML page
boundaries need not match the benchmark PDFs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

try:
    from .storage_paths import DATA_ROOT
except ImportError:
    from storage_paths import DATA_ROOT

import requests
from bs4 import BeautifulSoup


DOC_RE = re.compile(r"^(?P<ticker>[A-Z0-9.\-]+)_(?P<year>20\d{2})_10K$")
DEFAULT_USER_AGENT = "FinPlan-RAG academic research (contact not provided)"
HISTORICAL_CIK_OVERRIDES = {
    # Current SEC ticker mappings may omit or point past the historical filer
    # that issued the benchmark year's report. These CIKs are verified against
    # the corresponding SEC submissions records.
    "AEP": "4904",
    "BLK": "1364742",
    "FI": "798354",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def load_rows(root: Path) -> List[Mapping[str, object]]:
    rows: Dict[str, Mapping[str, object]] = {}
    for name in ("secqa_test.jsonl", "textual_test.jsonl"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.setdefault(str(row["qid"]), row)
    return sorted(rows.values(), key=lambda row: int(re.search(r"\d+", str(row["qid"])).group()))


def eligible(row: Mapping[str, object]) -> Tuple[str, ...] | None:
    docs = tuple(sorted({str(item["doc_name"]) for item in row.get("evidences", [])}))
    if not 2 <= len(docs) <= 3:
        return None
    parsed = [DOC_RE.fullmatch(doc) for doc in docs]
    if not all(parsed):
        return None
    if min(int(match.group("year")) for match in parsed if match) < 2021:
        return None
    return docs


def select_splits(rows: Sequence[Mapping[str, object]], exploratory: int, holdout: int) -> List[Dict[str, object]]:
    candidates = [(row, eligible(row)) for row in rows]
    candidates = [(row, docs) for row, docs in candidates if docs is not None]
    selected: List[Dict[str, object]] = []
    exploratory_docs: set[str] = set()
    for row, docs in candidates:
        if len([x for x in selected if x["split"] == "exploratory"]) >= exploratory:
            break
        selected.append({"split": "exploratory", "row": row, "docs": docs})
        exploratory_docs.update(docs)
    used_qids = {str(item["row"]["qid"]) for item in selected}
    for row, docs in candidates:
        if len([x for x in selected if x["split"] == "holdout"]) >= holdout:
            break
        if str(row["qid"]) in used_qids or exploratory_docs.intersection(docs):
            continue
        selected.append({"split": "holdout", "row": row, "docs": docs})
    return selected


def selections_from_manifest(
    rows: Sequence[Mapping[str, object]], manifest_path: Path
) -> List[Dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_qid = {str(row["qid"]): row for row in rows}
    selected: List[Dict[str, object]] = []
    seen_docs: set[str] = set()
    for item in manifest["cases"]:
        qid = str(item["qid"])
        row = by_qid[qid]
        docs = tuple(item["docs"])
        if docs != eligible(row):
            raise ValueError(f"manifest evidence mismatch for {qid}")
        if seen_docs.intersection(docs):
            raise ValueError(f"manifest document overlap for {qid}")
        seen_docs.update(docs)
        selected.append({"split": str(item["split"]), "row": row, "docs": docs})
    return selected


def submission_tables(cik: str, session: requests.Session) -> Sequence[Mapping[str, object]]:
    base = session.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", timeout=45)
    base.raise_for_status()
    payload = base.json()
    tables: List[Mapping[str, object]] = [payload["filings"]["recent"]]
    for item in payload["filings"].get("files", []):
        response = session.get(f"https://data.sec.gov/submissions/{item['name']}", timeout=45)
        response.raise_for_status()
        tables.append(response.json())
    return tables


def locate_10k(tables: Sequence[Mapping[str, object]], year: int) -> Dict[str, str]:
    candidates: List[Tuple[int, Dict[str, str]]] = []
    for table in tables:
        for i, form in enumerate(table["form"]):
            if form != "10-K":
                continue
            report_date = str(table["reportDate"][i])
            filing_date = str(table["filingDate"][i])
            report_year = int(report_date[:4]) if re.match(r"20\d{2}", report_date) else 0
            filing_year = int(filing_date[:4])
            score = 100 * (report_year == year) + 20 * (filing_year in {year, year + 1}) - abs((report_year or filing_year) - year)
            accepted_values = table.get("acceptanceDateTime", [""] * len(table["form"]))
            accepted = str(accepted_values[i]) or f"{filing_date}T23:59:59Z"
            candidates.append(
                (
                    score,
                    {
                        "accession": str(table["accessionNumber"][i]),
                        "document": str(table["primaryDocument"][i]),
                        "filing_date": filing_date,
                        "report_date": report_date,
                        "available_at": accepted,
                    },
                )
            )
    if not candidates:
        raise KeyError(f"10-K not found for {year}")
    score, record = max(candidates, key=lambda item: item[0])
    if score < 100:
        raise KeyError(f"exact report-year 10-K not found for {year}")
    return record


def archive_url(cik: str, accession: str, document: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{document}"


def download_text(url: str, session: requests.Session) -> str:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return normalize(BeautifulSoup(response.content, "html.parser").get_text(" ", strip=True))


def build(
    lofin_root: Path,
    output: Path,
    exploratory: int,
    holdout: int,
    delay: float,
    selection_manifest: Path | None = None,
    reuse_existing: bool = False,
) -> Mapping[str, object]:
    session = requests.Session()
    session.headers.update({"User-Agent": os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)})
    ticker_response = session.get("https://www.sec.gov/files/company_tickers.json", timeout=45)
    ticker_response.raise_for_status()
    ticker_map = {str(item["ticker"]).upper(): str(item["cik_str"]) for item in ticker_response.json().values()}
    ticker_map.update(HISTORICAL_CIK_OVERRIDES)

    rows = load_rows(lofin_root)
    selections = (
        selections_from_manifest(rows, selection_manifest)
        if selection_manifest is not None
        else select_splits(rows, exploratory, holdout)
    )
    doc_names = sorted({doc for item in selections for doc in item["docs"]})
    documents: Dict[str, Dict[str, object]] = {}
    reused_documents = 0
    if reuse_existing and output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        for doc in prior.get("documents", []):
            doc_id = str(doc["doc_id"])
            if doc_id not in doc_names:
                continue
            text = str(doc["text"])
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != doc["text_sha256"]:
                raise ValueError(f"cached text hash mismatch for {doc_id}")
            documents[doc_id] = dict(doc)
            reused_documents += 1
    tables: Dict[str, Sequence[Mapping[str, object]]] = {}
    failures: Dict[str, str] = {}

    for doc_name in doc_names:
        if doc_name in documents:
            continue
        match = DOC_RE.fullmatch(doc_name)
        assert match
        ticker, year = match.group("ticker"), int(match.group("year"))
        try:
            cik = ticker_map[ticker]
            if cik not in tables:
                tables[cik] = submission_tables(cik, session)
                time.sleep(delay)
            filing = locate_10k(tables[cik], year)
            url = archive_url(cik, filing["accession"], filing["document"])
            text = download_text(url, session)
            documents[doc_name] = {
                "doc_id": doc_name,
                "ticker": ticker,
                "year": year,
                "cik": f"{int(cik):010d}",
                "url": url,
                **filing,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            }
            time.sleep(delay)
        except Exception as exc:  # preserve failure evidence instead of silently dropping it
            failures[doc_name] = f"{type(exc).__name__}: {exc}"

    cases: List[Dict[str, object]] = []
    for item in selections:
        docs = tuple(item["docs"])
        if not all(doc in documents for doc in docs):
            continue
        row = item["row"]
        parsed = [DOC_RE.fullmatch(doc) for doc in docs]
        tickers = sorted({match.group("ticker") for match in parsed if match})
        years = sorted({int(match.group("year")) for match in parsed if match})
        template = "comparative_investigation" if len(tickers) > 1 else "accounting_derivation"
        evidence_pages = {str(ev["doc_name"]): int(ev["page_num"]) for ev in row.get("evidences", [])}
        cutoff = max(str(documents[doc]["available_at"]) for doc in docs)
        cases.append(
            {
                "case_id": str(row["qid"]),
                "split": item["split"],
                "template": template,
                "question": row["question"],
                "reference_answer": row["answer"],
                "candidate_legs": [{"ticker": ticker, "years": sorted({int(DOC_RE.fullmatch(doc).group('year')) for doc in docs if DOC_RE.fullmatch(doc).group('ticker') == ticker})} for ticker in tickers],
                "gold_doc_ids": list(docs),
                "gold_pages": evidence_pages,
                "cutoff": cutoff,
            }
        )

    result = {
        "schema": "finplan-lofin-multidoc-pilot.v1",
        "status": "real-sec-documents-document-level-gold",
        "protocol": {
            "lofin_root": str(lofin_root),
            "requested_exploratory": exploratory,
            "requested_holdout": holdout,
            "selection_manifest": str(selection_manifest) if selection_manifest else None,
            "selection_manifest_sha256": hashlib.sha256(selection_manifest.read_bytes()).hexdigest() if selection_manifest else None,
            "reused_documents_with_verified_hash": reused_documents,
            "cases": len(cases),
            "documents": len(documents),
            "failed_documents": failures,
            "split_rule": "holdout gold document ids are disjoint from exploratory gold document ids",
            "user_agent_contact_provided": bool(os.environ.get("SEC_USER_AGENT")),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "documents": list(documents.values()),
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lofin", default=DATA_ROOT / "external" / "lofin")
    parser.add_argument("--out", default=DATA_ROOT / "lofin_multidoc_pilot_v1.json")
    parser.add_argument("--exploratory", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--selection-manifest")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    result = build(
        Path(args.lofin),
        Path(args.out),
        args.exploratory,
        args.holdout,
        args.delay,
        Path(args.selection_manifest) if args.selection_manifest else None,
        args.reuse_existing,
    )
    print(json.dumps(result["protocol"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
