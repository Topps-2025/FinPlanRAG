"""Enrich the public company registry with filing type and fiscal period."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


def build(data_path: Path, company_registry_path: Path) -> Mapping[str, object]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    base = json.loads(company_registry_path.read_text(encoding="utf-8"))
    filings_by_ticker: dict[str, list[dict[str, object]]] = {}
    for document in data["documents"]:
        form = str(document.get("form", "10-K"))
        period = str(document.get("fiscal_period", "FY"))
        filings_by_ticker.setdefault(str(document["ticker"]), []).append(
            {
                "doc_id": str(document["doc_id"]),
                "fiscal_year": int(document["year"]),
                "filing_type": form,
                "fiscal_period": period,
                "available_at": str(document["available_at"]),
            }
        )
    companies = []
    for company in base["companies"]:
        copied = dict(company)
        copied["available_filings"] = sorted(
            filings_by_ticker.get(str(company["ticker"]), []),
            key=lambda item: (
                int(item["fiscal_year"]),
                str(item["fiscal_period"]),
                str(item["filing_type"]),
            ),
        )
        companies.append(copied)
    return {
        "schema": "finplan-sec-filing-registry.v2",
        "status": "public-corpus-filing-metadata-not-gold-obligations",
        "protocol": {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "company_registry_sha256": hashlib.sha256(company_registry_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "companies": len(companies),
            "filings": sum(len(company["available_filings"]) for company in companies),
        },
        "companies": companies,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--company-registry", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = build(Path(args.data), Path(args.company_registry))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["protocol"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
