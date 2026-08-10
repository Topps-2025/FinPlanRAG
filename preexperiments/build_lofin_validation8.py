"""Build validation8: self-constructed same-company multi-filing SEC set.

Larger confirmatory batch for the v9 changes pre-declared in
lofin_validation8_manifest.json (fiscal-year-N parsing, first-n-quarters
fix, aliases as default).  Six fresh companies (CSX/CVS/GD/TXN/CMCSA/HCA,
all Dec-31 fiscal years, none of the 120 prior-batch tickers) x 5 filings
each = 30 previously unused SEC
documents; 30 cases (dev 6 / frozen 24).  Gold obligations are derived
from (a) which filing physically reports the requested period figure
(XBRL companyfacts, accession-matched to the downloaded filing) and
(b) a text-locality check that the value appears near a revenue label
in the corpus text.  The manifest is frozen before this builder runs;
no method results exist yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import date, timedelta
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
    locate_10k,
    submission_tables,
)
from build_lofin_mixed_filing_validation5 import (
    MIXED_FILING_CIK_OVERRIDES,
    NON_USER_PLACEHOLDER_AGENT,
    locate_10q,
    parse_doc,
)

REVENUE_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "NetRevenue",  # e.g. PepsiCo's "Net revenue" line
    "SalesRevenueNet",
)

REVENUE_LABELS = (
    "net revenues",
    "total revenues",
    "net sales",
    "revenues",
    "total net sales",
    "service revenues",
    "revenue",  # singular, e.g. UPS's "Revenue" / "Consolidated revenue"
)

# template -> [(fact_key, start, end, expected_form)]
FACT_REQUESTS = {
    "h1": [("six_months", "{y}-01-01", "{y}-06-30", "10-Q")],
    "h1_9m": [
        ("six_months", "{y}-01-01", "{y}-06-30", "10-Q"),
        ("nine_months", "{y}-01-01", "{y}-09-30", "10-Q"),
    ],
    "q1q2": [
        ("q1", "{y}-01-01", "{y}-03-31", "10-Q"),
        ("q2", "{y}-04-01", "{y}-06-30", "10-Q"),
    ],
    "h2": [
        ("nine_months", "{y}-01-01", "{y}-09-30", "10-Q"),
        ("annual", "{y}-01-01", "{y}-12-31", "10-K"),
    ],
    "fy_q1n": [
        ("annual", "{y}-01-01", "{y}-12-31", "10-K"),
        ("q1_next", "{yn}-01-01", "{yn}-03-31", "10-Q"),
    ],
}


def squashed(value: str) -> str:
    return re.sub(r"\s+", "", value)


def locate_value(value: int, text: str) -> Dict[str, object] | None:
    normalized = squashed(text)
    forms = [f"{value:,}", f"{value}"]
    if str(value).endswith("000000"):
        forms.append(f"{value // 1_000_000:,}")
    candidates = sorted({form for form in forms if form in normalized}, key=len, reverse=True)
    for form in candidates:
        # check every occurrence, not just the first: e.g. CAT 10-K mentions
        # "64,809" in prose long before the income statement row that sits next
        # to "Total sales and revenues".  No digit boundaries: in columnar
        # income statements adjacent columns read "...19,196 39,366 38,828..."
        # and after whitespace-squashing the match's neighbours are digits.
        # Longer forms (full and unit-less) are tried first and the revenue
        # label proximity check rejects most false positives.
        for match in re.finditer(re.escape(form), normalized):
            start = match.start()
            # lowercased: income-statement labels may be title-cased
            # ("Revenue $ 3,701 ..."), which would otherwise fail the
            # case-sensitive label check even though the value sits next
            # to the label (CSX Q2 2024 10-Q).
            window = normalized[max(0, start - 250): start + 60].lower()
            for label in REVENUE_LABELS:
                if label.replace(" ", "") in window:
                    return {"value_form": form, "offset": start, "context": text[max(0, start - 150): start + 60]}
    return None


def fetch_facts(cik: str, session: requests.Session) -> Mapping[str, object]:
    response = session.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json",
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def match_range(
    keyed: Mapping[tuple[str, str], Mapping[str, object]],
    start: str,
    end: str,
    tol_days: int = 2,
) -> Mapping[str, object] | None:
    """Find a fact keyed (start, end) in the company's facts, tolerating
    small end-date drift.  Some registrants run 13-week fiscal quarters
    ending on a Saturday (e.g. General Dynamics: Q3 2024 ends 2024-09-29,
    Q1 2025 ends 2025-03-30) while the templates use calendar quarter-end
    dates; the filing and the value are still the gold ones (verified by
    accession and text locality below)."""
    if (start, end) in keyed:
        return (start, end)
    end_dt = date.fromisoformat(end)
    for delta in range(1, tol_days + 1):
        for candidate in (end_dt - timedelta(days=delta), end_dt + timedelta(days=delta)):
            key = (start, candidate.isoformat())
            if key in keyed:
                return key
    return None


def pick_concept(facts: Mapping[str, object], required: Sequence[tuple[str, str]]) -> str | None:
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in REVENUE_CONCEPTS:
        if concept not in us_gaap:
            continue
        rows = us_gaap[concept].get("units", {}).get("USD", [])
        keyed = {
            (str(item.get("start")), str(item.get("end"))): item
            for item in rows
            if item.get("form") in ("10-Q", "10-K") and item.get("start")
        }
        if all(match_range(keyed, start, end) is not None for start, end in required):
            return concept
    return None


def build(
    manifest_path: Path,
    output: Path,
    prior_paths: Sequence[Path],
    delay: float,
) -> Mapping[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    cases_in_manifest = manifest["cases"]
    templates = manifest["template_bank"]

    tickers = sorted({str(item["company"]) for item in cases_in_manifest})
    doc_names = sorted({str(doc) for item in cases_in_manifest for doc in item["docs"]})
    # corpus: each company's Q1/Q2/Q3 10-Q + FY 10-K + Q1(next) 10-Q
    corpus_docs: set[str] = set()
    for ticker in tickers:
        corpus_docs.update(
            [
                f"{ticker}_2024Q1_10Q",
                f"{ticker}_2024Q2_10Q",
                f"{ticker}_2024Q3_10Q",
                f"{ticker}_2024_10K",
                f"{ticker}_2025Q1_10Q",
            ]
        )
    corpus_docs = sorted(corpus_docs)
    assert set(doc_names) <= set(corpus_docs), "manifest docs must be a subset of the corpus"

    prior_docs: set[str] = set()
    prior_tickers: set[str] = set()
    for prior_path in prior_paths:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        prior_docs.update(str(doc["doc_id"]) for doc in prior["documents"])
        prior_tickers.update(str(doc["ticker"]) for doc in prior["documents"])
    overlap = sorted(prior_docs.intersection(corpus_docs))
    ticker_overlap = sorted(prior_tickers.intersection(tickers))
    if overlap:
        raise ValueError(f"validation8 overlaps prior documents: {overlap}")
    if ticker_overlap:
        raise ValueError(f"validation8 overlaps prior companies: {ticker_overlap}")

    session = requests.Session()
    session.headers.update({"User-Agent": NON_USER_PLACEHOLDER_AGENT})
    ticker_response = session.get("https://www.sec.gov/files/company_tickers.json", timeout=45)
    ticker_response.raise_for_status()
    ticker_map = {str(item["ticker"]).upper(): str(item["cik_str"]) for item in ticker_response.json().values()}
    ticker_map.update(HISTORICAL_CIK_OVERRIDES)
    ticker_map.update(MIXED_FILING_CIK_OVERRIDES)

    tables: Dict[str, Sequence[Mapping[str, object]]] = {}
    filings: Dict[str, Dict[str, object]] = {}
    failures: Dict[str, str] = {}
    for doc_name in corpus_docs:
        parts = parse_doc(doc_name)
        ticker, year = str(parts["ticker"]), int(parts["year"])
        quarter = str(parts["quarter"]) if parts["quarter"] else None
        try:
            cik = ticker_map[ticker]
            if cik not in tables:
                tables[cik] = submission_tables(cik, session)
                time.sleep(delay)
            filing = locate_10k(tables[cik], year) if parts["form"] == "10-K" else locate_10q(tables[cik], year, quarter)
            url = archive_url(cik, filing["accession"], filing["document"])
            text = download_text(url, session)
            filings[doc_name] = {
                "doc_id": doc_name,
                "ticker": ticker,
                "year": year,
                "form": parts["form"],
                "fiscal_period": quarter or "FY",
                "cik": f"{int(cik):010d}",
                "url": url,
                **filing,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            }
            time.sleep(delay)
        except Exception as exc:
            failures[doc_name] = f"{type(exc).__name__}: {exc}"
    missing = sorted(set(corpus_docs) - set(filings))
    if missing:
        raise RuntimeError(f"corpus download failures: {missing} -> {failures}")

    # XBRL facts: pick one revenue concept per company and verify each required
    # fact (a) exists with the exact period range, (b) comes from the accession
    # of the gold filing, (c) is locatable in the corpus text near a revenue label.
    facts_cache: Dict[str, Mapping[str, object]] = {}
    facts_log: List[Dict[str, object]] = []
    for ticker in tickers:
        cik = ticker_map[ticker]
        required = []
        for item in manifest["cases"]:
            if str(item["company"]) != ticker:
                continue
            for fact_key, start, end, form in FACT_REQUESTS[str(item["template"])]:
                required.append((fact_key, start.format(y=item["year"], yn=item["year"] + 1), end.format(y=item["year"], yn=item["year"] + 1), form))
        req_ranges = sorted({(start, end) for _, start, end, _ in required})
        facts = facts_cache.setdefault(ticker, fetch_facts(cik, session))
        concept = pick_concept(facts, req_ranges)
        if concept is None:
            raise RuntimeError(f"{ticker}: no revenue concept covers {req_ranges}")
        all_rows: Dict[tuple[str, str], List[Mapping[str, object]]] = {}
        for item in facts["facts"]["us-gaap"][concept]["units"]["USD"]:
            if item.get("start") and item.get("form") in ("10-Q", "10-K"):
                all_rows.setdefault((str(item["start"]), str(item["end"])), []).append(item)
        for fact_key, start, end, form in required:
            # The reporting filing is determined by the END of the period range:
            # YTD facts (Jan 1 start) live in the 10-Q filed at the quarter end
            # (six-months -> Q2 10-Q, nine-months -> Q3 10-Q), not in Q1.
            if form == "10-Q":
                doc_id = f"{ticker}_{end[:4]}Q{(int(end[5:7]) + 2) // 3}_10Q"
            else:
                doc_id = f"{ticker}_{end[:4]}_10K"
            gold_accn = str(filings[doc_id]["accession"]).replace("-", "")
            matched_key = match_range(all_rows, start, end)
            matches = [
                item
                for item in (all_rows.get(matched_key, []) if matched_key is not None else [])
                if str(item.get("accn", "")).replace("-", "") == gold_accn
            ]
            if not matches:
                raise RuntimeError(
                    f"{ticker} {fact_key}: no fact {start}~{end} from gold filing {doc_id} "
                    f"(accn {gold_accn}; available accns: "
                    f"{[str(i.get('accn')) for i in all_rows.get((start, end), [])][:5]})"
                )
            row = matches[0]
            span = locate_value(int(row["val"]), filings[doc_id]["text"])
            if span is None:
                raise RuntimeError(f"{ticker} {fact_key}: value {row['val']} not locatable near a revenue label in {doc_id}")
            facts_log.append(
                {
                    "ticker": ticker,
                    "concept": concept,
                    "fact_key": fact_key,
                    "start": start,
                    "end": end,
                    "value": row["val"],
                    "doc_id": doc_id,
                    "accn": row.get("accn"),
                    "span": span,
                }
            )
        time.sleep(0.2)

    cases: List[Dict[str, object]] = []
    for item in cases_in_manifest:
        ticker, template = str(item["company"]), str(item["template"])
        year = int(item["year"])
        gold_docs = list(item["docs"])
        facts = {str(fact["fact_key"]): fact for fact in facts_log if fact["ticker"] == ticker and fact["doc_id"] in gold_docs}
        values = {key: int(fact["value"]) for key, fact in facts.items()}
        if template == "h1":
            reference_answer = {"six_months": values["six_months"]}
        elif template == "h1_9m":
            reference_answer = {"six_months": values["six_months"], "nine_months": values["nine_months"]}
        elif template == "q1q2":
            reference_answer = {"q1": values["q1"], "q2": values["q2"]}
        elif template == "h2":
            reference_answer = {
                "nine_months": values["nine_months"],
                "annual": values["annual"],
                "derived_h2": values["annual"] - values["nine_months"],
            }
        elif template == "fy_q1n":
            reference_answer = {"annual": values["annual"], "q1_next": values["q1_next"]}
        else:
            raise ValueError(template)
        question = templates[template]["question"].replace("{Name}", str(item["name"])).replace("{Y}", str(year)).replace("{Y+1}", str(year + 1))
        obligations = [
            {
                "ticker": parse_doc(doc)["ticker"],
                "fiscal_year": parse_doc(doc)["year"],
                "filing_type": parse_doc(doc)["form"],
                "fiscal_period": parse_doc(doc)["quarter"] or "FY",
                "doc_id": doc,
            }
            for doc in gold_docs
        ]
        gold_spans: Dict[str, List[Dict[str, object]]] = {}
        for fact in facts_log:
            if fact["ticker"] == ticker and fact["doc_id"] in gold_docs:
                gold_spans.setdefault(str(fact["doc_id"]), []).append(
                    {
                        "fact_key": fact["fact_key"],
                        "value": fact["value"],
                        "value_form": fact["span"]["value_form"],
                        "offset": fact["span"]["offset"],
                        "context": fact["span"]["context"],
                    }
                )
        cases.append(
            {
                "case_id": str(item["case_id"]),
                "split": str(item["split"]),
                "stratum": str(templates[template]["stratum"]),
                "template": template,
                "question": question,
                "reference_answer": reference_answer,
                "candidate_legs": [{"ticker": ticker, "years": [year]}],
                "candidate_obligations": obligations,
                "gold_doc_ids": gold_docs,
                "gold_spans": gold_spans,
                "cutoff": max(str(filings[doc]["available_at"]) for doc in gold_docs),
            }
        )

    result = {
        "schema": "finplan-lofin-validation8.v1",
        "status": "self-constructed-real-sec-frozen-validation",
        "protocol": {
            "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": manifest_sha256,
            "prior_document_overlap": len(overlap),
            "prior_company_overlap": len(ticker_overlap),
            "requested_cases": len(cases_in_manifest),
            "completed_cases": len(cases),
            "corpus_documents": len(filings),
            "failed_documents": failures,
            "fact_verifications": len(facts_log),
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "user_agent_policy": "hard-coded non-user placeholder; no local or git email read",
        },
        "documents": [filings[name] for name in sorted(filings)],
        "cases": cases,
        "fact_verification": facts_log,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="preexperiments/lofin_validation8_manifest.json")
    parser.add_argument("--out", default=DATA_ROOT / "lofin_validation8_v1.json")
    parser.add_argument(
        "--prior",
        nargs="+",
        default=[
            str(DATA_ROOT / "lofin_multidoc_pilot_v1.json"),
            str(DATA_ROOT / "lofin_multidoc_validation2_v1.json"),
            str(DATA_ROOT / "lofin_multidoc_validation3_v1.json"),
            str(DATA_ROOT / "lofin_multidoc_validation4_v1.json"),
            str(DATA_ROOT / "lofin_mixed_filing_validation5_v1.json"),
            str(DATA_ROOT / "lofin_validation6_v1.json"),
            str(DATA_ROOT / "lofin_validation7_v1.json"),
        ],
    )
    parser.add_argument("--delay", type=float, default=0.12)
    args = parser.parse_args()
    result = build(Path(args.manifest), Path(args.out), [Path(path) for path in args.prior], args.delay)
    print(json.dumps(result["protocol"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
