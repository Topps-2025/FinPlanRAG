"""Build a public SEC company/year registry for non-oracle obligation planning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

try:
    from .storage_paths import DATA_ROOT
except ImportError:
    from storage_paths import DATA_ROOT
from typing import Dict, Mapping

import requests


DEFAULT_USER_AGENT = "FinPlan-RAG academic research (contact not provided)"


def normalize_name(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def build(data_path: Path) -> Mapping[str, object]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    session = requests.Session()
    session.headers.update({"User-Agent": os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)})
    by_ticker: Dict[str, Dict[str, object]] = {}
    submissions_cache: Dict[str, Mapping[str, object]] = {}
    for doc in data["documents"]:
        ticker = str(doc["ticker"])
        cik = str(doc["cik"])
        if cik not in submissions_cache:
            response = session.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", timeout=45)
            response.raise_for_status()
            submissions_cache[cik] = response.json()
        submission = submissions_cache[cik]
        official_name = str(submission.get("name", ticker))
        former_names = [str(item.get("name")) for item in submission.get("formerNames", []) if item.get("name")]
        entry = by_ticker.setdefault(
            ticker,
            {
                "ticker": ticker,
                "cik": f"{int(cik):010d}",
                "official_name": official_name,
                "aliases": [],
                "available_years": [],
            },
        )
        aliases = {ticker.lower(), normalize_name(official_name), *[normalize_name(name) for name in former_names]}
        # Removing legal suffixes creates transparent aliases such as
        # "digital realty" from "Digital Realty Trust, Inc.".
        for alias in list(aliases):
            stripped = re.sub(r"\b(inc|incorporated|corp|corporation|company|co|plc|ltd|llc|trust|holdings?)\b", " ", alias)
            stripped = re.sub(r"\s+", " ", stripped).strip()
            if stripped:
                aliases.add(stripped)
        entry["aliases"] = sorted(set(entry["aliases"]).union(aliases))
        entry["available_years"] = sorted(set(entry["available_years"]).union({int(doc["year"])}))
    return {
        "schema": "finplan-sec-company-registry.v1",
        "status": "public-corpus-metadata-not-gold-obligations",
        "protocol": {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "companies": len(by_ticker),
            "user_agent_contact_provided": bool(os.environ.get("SEC_USER_AGENT")),
        },
        "companies": [by_ticker[ticker] for ticker in sorted(by_ticker)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_ROOT / "lofin_multidoc_validation2_v1.json")
    parser.add_argument("--out", default=DATA_ROOT / "sec_company_registry_validation2_v1.json")
    args = parser.parse_args()
    result = build(Path(args.data))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["protocol"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
