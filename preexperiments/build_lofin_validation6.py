"""Build the stratified, filing-independent LOFin validation6 set."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

try:
    from .storage_paths import DATA_ROOT
except ImportError:
    from storage_paths import DATA_ROOT

import requests

from build_lofin_multidoc_pilot import (
    HISTORICAL_CIK_OVERRIDES,
    archive_url,
    download_text,
    load_rows,
    locate_10k,
    submission_tables,
)
from build_lofin_mixed_filing_validation5 import (
    MIXED_FILING_CIK_OVERRIDES,
    NON_USER_PLACEHOLDER_AGENT,
    locate_10q,
    parse_doc,
)


def selected_rows(rows: Sequence[Mapping[str, object]], manifest_path: Path) -> List[Dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_qid = {str(row["qid"]): row for row in rows}
    selected: List[Dict[str, object]] = []
    seen_docs: set[str] = set()
    seen_tickers: set[str] = set()
    for item in manifest["cases"]:
        qid = str(item["qid"])
        row = by_qid[qid]
        docs = tuple(str(doc) for doc in item["docs"])
        evidence_docs = {str(ev["doc_name"]) for ev in row.get("evidences", [])}
        if set(docs) != evidence_docs:
            raise ValueError(f"manifest evidence mismatch for {qid}")
        parsed = [parse_doc(doc) for doc in docs]
        tickers = {str(part["ticker"]) for part in parsed}
        if seen_docs.intersection(docs) or seen_tickers.intersection(tickers):
            raise ValueError(f"within-split document/company overlap for {qid}")
        seen_docs.update(docs)
        seen_tickers.update(tickers)
        selected.append(
            {
                "split": str(item["split"]),
                "stratum": str(item["stratum"]),
                "row": row,
                "docs": docs,
            }
        )
    return selected


def build(
    lofin_root: Path,
    manifest_path: Path,
    output: Path,
    prior_paths: Sequence[Path],
    delay: float,
) -> Mapping[str, object]:
    rows = load_rows(lofin_root)
    selections = selected_rows(rows, manifest_path)
    doc_names = sorted({doc for item in selections for doc in item["docs"]})
    prior_docs: set[str] = set()
    for prior_path in prior_paths:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        prior_docs.update(str(doc["doc_id"]) for doc in prior["documents"])
    overlap = sorted(prior_docs.intersection(doc_names))
    if overlap:
        raise ValueError(f"validation6 overlaps prior documents: {overlap}")

    session = requests.Session()
    session.headers.update({"User-Agent": NON_USER_PLACEHOLDER_AGENT})
    ticker_response = session.get("https://www.sec.gov/files/company_tickers.json", timeout=45)
    ticker_response.raise_for_status()
    ticker_map = {str(item["ticker"]).upper(): str(item["cik_str"]) for item in ticker_response.json().values()}
    ticker_map.update(HISTORICAL_CIK_OVERRIDES)
    ticker_map.update(MIXED_FILING_CIK_OVERRIDES)
    tables: Dict[str, Sequence[Mapping[str, object]]] = {}
    documents: Dict[str, Dict[str, object]] = {}
    failures: Dict[str, str] = {}
    for doc_name in doc_names:
        parts = parse_doc(doc_name)
        ticker, year = str(parts["ticker"]), int(parts["year"])
        try:
            cik = ticker_map[ticker]
            if cik not in tables:
                tables[cik] = submission_tables(cik, session)
                time.sleep(delay)
            filing = (
                locate_10k(tables[cik], year)
                if parts["form"] == "10-K"
                else locate_10q(tables[cik], year, str(parts["quarter"]))
            )
            url = archive_url(cik, filing["accession"], filing["document"])
            text = download_text(url, session)
            documents[doc_name] = {
                "doc_id": doc_name,
                "ticker": ticker,
                "year": year,
                "form": parts["form"],
                "fiscal_period": parts["quarter"] or "FY",
                "cik": f"{int(cik):010d}",
                "url": url,
                **filing,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            }
            time.sleep(delay)
        except Exception as exc:
            failures[doc_name] = f"{type(exc).__name__}: {exc}"

    cases: List[Dict[str, object]] = []
    for item in selections:
        docs = tuple(item["docs"])
        if not all(doc in documents for doc in docs):
            continue
        row = item["row"]
        parsed = [parse_doc(doc) for doc in docs]
        tickers = sorted({str(part["ticker"]) for part in parsed})
        obligations = [
            {
                "ticker": part["ticker"],
                "fiscal_year": part["year"],
                "filing_type": part["form"],
                "fiscal_period": part["quarter"] or "FY",
                "doc_id": doc,
            }
            for doc, part in zip(docs, parsed)
        ]
        candidate_legs = [
            {
                "ticker": ticker,
                "years": sorted({int(part["year"]) for part in parsed if part["ticker"] == ticker}),
            }
            for ticker in tickers
        ]
        cases.append(
            {
                "case_id": str(row["qid"]),
                "split": item["split"],
                "stratum": item["stratum"],
                "template": "comparative_investigation" if len(tickers) > 1 else "accounting_derivation",
                "question": row["question"],
                "reference_answer": row["answer"],
                "candidate_legs": candidate_legs,
                "candidate_obligations": obligations,
                "gold_doc_ids": list(docs),
                "gold_pages": {str(ev["doc_name"]): int(ev["page_num"]) for ev in row.get("evidences", [])},
                "cutoff": max(str(documents[doc]["available_at"]) for doc in docs),
            }
        )
    result = {
        "schema": "finplan-lofin-validation6.v1",
        "status": "stratified-real-sec-frozen-validation",
        "protocol": {
            "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "prior_document_overlap": len(overlap),
            "requested_cases": len(selections),
            "completed_cases": len(cases),
            "documents": len(documents),
            "failed_documents": failures,
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "user_agent_policy": "hard-coded non-user placeholder; no local or git email read",
        },
        "documents": [documents[name] for name in sorted(documents)],
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lofin", default=DATA_ROOT / "external" / "lofin")
    parser.add_argument("--manifest", default="preexperiments/lofin_validation6_manifest.json")
    parser.add_argument("--out", default=DATA_ROOT / "lofin_validation6_v1.json")
    parser.add_argument(
        "--prior",
        nargs="+",
        default=[
            str(DATA_ROOT / "lofin_multidoc_pilot_v1.json"),
            str(DATA_ROOT / "lofin_multidoc_validation2_v1.json"),
            str(DATA_ROOT / "lofin_multidoc_validation3_v1.json"),
            str(DATA_ROOT / "lofin_multidoc_validation4_v1.json"),
            str(DATA_ROOT / "lofin_mixed_filing_validation5_v1.json"),
        ],
    )
    parser.add_argument("--delay", type=float, default=0.12)
    args = parser.parse_args()
    result = build(Path(args.lofin), Path(args.manifest), Path(args.out), [Path(path) for path in args.prior], args.delay)
    print(json.dumps(result["protocol"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
