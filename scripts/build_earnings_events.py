"""Build real earnings-event timelines from SEC submissions.

For each claim, find the earliest 8-K (or 8-K/A) with Item 2.02 (Results of
Operations and Financial Condition) filed strictly after the claim cutoff and
no later than the dataset's 10-Q outcome date. That 8-K is the company's
earnings release: it publicly reveals the quarter's revenue and is the real
terminal evidence for the decision problem. No human annotation is used.

The script caches every SEC submissions JSON under ``paper/data/sec_cache`` and
writes ``paper/data/earnings_events.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import requests

from run_real_benchmark import DEFAULT_DATA, load_records, parse_date

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "paper" / "data" / "sec_cache"
OUT = REPO_ROOT / "paper" / "data" / "earnings_events.json"
SEC_BASE = "https://data.sec.gov/submissions"
HEADERS = {
    "User-Agent": "FinPlanResearch/1.0 (research@example.com)",
    "Accept-Encoding": "gzip, deflate",
}


def cik_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text.isdigit():
        return None
    return f"{int(text):010d}"


def fetch_json(url: str, path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    session = requests.Session()
    session.trust_env = False
    for attempt in range(5):
        try:
            response = session.get(url, headers=HEADERS, timeout=45)
            if response.status_code == 200:
                payload = response.json()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")
                return payload
            if response.status_code in (403, 429, 500, 502, 503):
                time.sleep(2.0 ** attempt)
                continue
            raise RuntimeError(f"HTTP {response.status_code}: {url}")
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2.0 ** attempt)
    raise RuntimeError(url)


def submissions_for(cik: str) -> list[dict[str, Any]]:
    base = CACHE / f"CIK{cik}.json"
    payload = fetch_json(f"{SEC_BASE}/CIK{cik}.json", base)
    rows = []
    recent = payload.get("filings", {}).get("recent", {})
    rows.extend(iter_rows(recent))
    for item in payload.get("filings", {}).get("files", []) or []:
        name = item.get("name")
        if not name:
            continue
        child = CACHE / name
        child_payload = fetch_json(f"{SEC_BASE}/{name}", child)
        rows.extend(iter_rows(child_payload))
    # Deduplicate by accession number.
    seen = set()
    unique = []
    for row in rows:
        key = row.get("accessionNumber")
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    unique.sort(key=lambda row: row.get("filingDate") or "")
    return unique


def iter_rows(payload: dict[str, Any]):
    keys = (
        "accessionNumber",
        "filingDate",
        "reportDate",
        "form",
        "items",
        "primaryDocument",
    )
    if not all(key in payload for key in keys):
        return
    n = len(payload["form"])
    for index in range(n):
        yield {key: payload[key][index] for key in keys}


def is_item_202(row: dict[str, Any]) -> bool:
    items = str(row.get("items") or "")
    parts = {part.strip() for part in items.replace(";", ",").split(",") if part.strip()}
    return "2.02" in parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sleep", type=float, default=0.12)
    args = parser.parse_args()
    rows = load_records(args.data)
    by_cik = defaultdict(list)
    for cutoff, outcome, record in rows:
        cik = cik_string(record.get("cik"))
        if cik:
            by_cik[cik].append((cutoff, outcome, record))
    print(f"companies with CIK: {len(by_cik)}")
    events = {}
    coverage = Counter()
    earlier = []
    for index, (cik, items) in enumerate(sorted(by_cik.items()), start=1):
        try:
            filings = submissions_for(cik)
        except Exception as error:  # noqa: BLE001
            print(f"{cik}: submissions failed: {error}")
            continue
        for cutoff, outcome, record in items:
            if cutoff is None or outcome is None:
                coverage["missing_dates"] += 1
                continue
            candidates = []
            for row in filings:
                filed = parse_date(row.get("filingDate"))
                if filed is None:
                    continue
                if not (cutoff < filed <= outcome):
                    continue
                if not str(row.get("form", "")).startswith("8-K"):
                    continue
                if not is_item_202(row):
                    continue
                accession = str(row.get("accessionNumber") or "").replace("-", "")
                url = (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{accession}/{row.get('primaryDocument')}"
                )
                candidates.append(
                    {
                        "date": filed.isoformat(),
                        "accession": row.get("accessionNumber"),
                        "url": url,
                        "form": row.get("form"),
                    }
                )
            if candidates:
                candidates.sort(key=lambda item: item["date"])
                event = candidates[0]
                events[record["claim_id"]] = {
                    **event,
                    "source": "8-K Item 2.02",
                    "days_before_10q": (outcome - date.fromisoformat(event["date"])).days,
                    "n_candidates": len(candidates),
                }
                coverage["earnings_8k"] += 1
                earlier.append((outcome - date.fromisoformat(event["date"])).days)
            else:
                events[record["claim_id"]] = {
                    "date": outcome.isoformat(),
                    "source": "10-Q fallback",
                    "days_before_10q": 0,
                    "n_candidates": 0,
                }
                coverage["fallback_10q"] += 1
        if index % 25 == 0:
            print(f"{index}/{len(by_cik)} companies, claims={len(events)}")
        time.sleep(args.sleep)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(events, indent=2), encoding="utf-8")
    summary = {
        "n_claims": len(events),
        "coverage": dict(coverage),
        "mean_days_before_10q": sum(earlier) / len(earlier) if earlier else None,
        "median_days_before_10q": sorted(earlier)[len(earlier) // 2] if earlier else None,
        "max_days_before_10q": max(earlier) if earlier else None,
        "min_days_before_10q": min(earlier) if earlier else None,
    }
    (args.out.parent / "earnings_events_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
