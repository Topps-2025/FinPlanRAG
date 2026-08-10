"""Build a small point-in-time acquisition/control pilot from public SEC filings.

The manifest fixes transaction lineages and official filings. The builder only
downloads the named filings, verifies target/lifecycle cues, records the SEC
acceptance timestamp as ``available_at``, and creates pre/post event replay
cases. It does not infer gold state from a retrieval model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from .storage_paths import DATA_ROOT
except ImportError:
    from storage_paths import DATA_ROOT
from typing import Dict, Iterable, List, Mapping, Sequence

import requests
from bs4 import BeautifulSoup


DEFAULT_USER_AGENT = "FinPlan-RAG academic research (contact not provided)"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def sec_url(cik: str, accession: str, document: str) -> str:
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession.replace('-', '')}/{document}"
    )


def submission_index(cik: str, session: requests.Session) -> Sequence[Mapping[str, object]]:
    response = session.get(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout=45)
    response.raise_for_status()
    base = response.json()
    tables: List[Mapping[str, object]] = [base["filings"]["recent"]]
    for item in base["filings"].get("files", []):
        history = session.get(f"https://data.sec.gov/submissions/{item['name']}", timeout=45)
        history.raise_for_status()
        tables.append(history.json())
    return tables


def accession_metadata(tables: Sequence[Mapping[str, object]], accession: str) -> Dict[str, str]:
    for recent in tables:
        for i, value in enumerate(recent["accessionNumber"]):
            if value == accession:
                accepted = recent.get("acceptanceDateTime", [""] * len(recent["accessionNumber"]))[i]
                filing_date = recent["filingDate"][i]
                if accepted:
                    available_at = accepted
                else:
                    available_at = f"{filing_date}T23:59:59Z"
                return {
                    "filing_date": filing_date,
                    "available_at": available_at,
                    "form": recent["form"][i],
                    "items": recent.get("items", [""] * len(recent["accessionNumber"]))[i],
                }
    raise KeyError(f"accession not found in recent submissions: {accession}")


def download_filing(
    cik: str,
    accession: str,
    document: str,
    session: requests.Session,
) -> str:
    response = session.get(sec_url(cik, accession, document), timeout=45)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    return normalize_text(soup.get_text(" ", strip=True))


def cue_flags(text: str, target: str) -> Dict[str, bool]:
    low = text.lower()
    target_low = target.lower()
    return {
        "target": target_low in low,
        "agreement": bool(
            re.search(
                r"(?:entered into|announc\w*|execut\w*)[^.]{0,260}"
                r"(?:agreement|merger|acqui\w*)|(?:will|to) acquire",
                low,
            )
        ),
        "completed": bool(
            re.search(
                r"(?:completed|completion|closed)[^.]{0,220}"
                r"(?:acquisition|transaction|merger)|"
                r"(?:acquisition|transaction|merger)[^.]{0,160}(?:completed|closed)",
                low,
            )
        ),
        "terminated": bool(
            re.search(
                r"(?:terminated|termination|terminate|abandoned|withdrawn)[^.]{0,240}"
                r"(?:agreement|transaction|merger|acquisition)|"
                r"(?:agreement|transaction|merger|acquisition)[^.]{0,180}"
                r"(?:terminated|termination|abandoned|withdrawn)",
                low,
            )
        ),
        "conditional": bool(re.search(r"subject to[^.]{0,180}(?:condition|approval|closing)", low)),
    }


def parse_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build(manifest_path: Path, output_path: Path, delay: float) -> Mapping[str, object]:
    manifest = load_json(manifest_path)
    session = requests.Session()
    session.headers.update({"User-Agent": os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)})

    documents: List[Dict[str, object]] = []
    cases: List[Dict[str, object]] = []
    indexes: Dict[str, Sequence[Mapping[str, object]]] = {}

    for lineage in manifest["lineages"]:  # type: ignore[index]
        cik = lineage["cik"]
        if cik not in indexes:
            indexes[cik] = submission_index(cik, session)
            time.sleep(delay)

        event_docs: Dict[str, Dict[str, object]] = {}
        for role in ("agreement", "terminal"):
            spec = lineage.get(role)
            if spec is None:
                continue
            metadata = accession_metadata(indexes[cik], spec["accession"])
            text = download_filing(cik, spec["accession"], spec["document"], session)
            flags = cue_flags(text, lineage["target"])
            record: Dict[str, object] = {
                "doc_id": f"{lineage['lineage_id']}::{role}",
                "lineage_id": lineage["lineage_id"],
                "company": lineage["company"],
                "target": lineage["target"],
                "role": role,
                "terminal_event": spec.get("event"),
                "cik": cik,
                "accession": spec["accession"],
                "document": spec["document"],
                "url": sec_url(cik, spec["accession"], spec["document"]),
                **metadata,
                "cue_flags": flags,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            }
            documents.append(record)
            event_docs[role] = record
            time.sleep(delay)

        agreement = event_docs["agreement"]
        agreement_time = parse_timestamp(str(agreement["available_at"]))
        if not agreement["cue_flags"]["target"]:  # type: ignore[index]
            raise ValueError(f"agreement target cue missing: {lineage['lineage_id']}")

        terminal = event_docs.get("terminal")
        if terminal is not None:
            terminal_time = parse_timestamp(str(terminal["available_at"]))
            # Use a natural historical replay point one day before the
            # terminal filing rather than an artificial one-second boundary.
            pre_cutoff = terminal_time - timedelta(days=1)
        else:
            pre_cutoff = agreement_time + timedelta(days=30)

        cases.append(
            {
                "case_id": f"{lineage['lineage_id']}::pre_terminal",
                "lineage_id": lineage["lineage_id"],
                "company": lineage["company"],
                "target": lineage["target"],
                "cutoff": pre_cutoff.isoformat(),
                "truth_state": "conditional",
                "gold_doc_ids": [agreement["doc_id"]],
                "limiting_doc_ids": [],
                "slice": "pre_terminal",
            }
        )

        if terminal is not None:
            terminal_event = lineage["terminal"]["event"]
            cases.append(
                {
                    "case_id": f"{lineage['lineage_id']}::post_terminal",
                    "lineage_id": lineage["lineage_id"],
                    "company": lineage["company"],
                    "target": lineage["target"],
                    "cutoff": parse_timestamp(str(terminal["available_at"])).isoformat(),
                    "truth_state": "none" if terminal_event == "terminated" else "indirect",
                    "gold_doc_ids": [terminal["doc_id"]],
                    "limiting_doc_ids": [terminal["doc_id"]] if terminal_event == "terminated" else [],
                    "slice": "post_terminal",
                }
            )

    protocol = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": "SEC EDGAR official filings",
        "available_at": "SEC submissions acceptanceDateTime; filing-date fallback explicitly marked",
        "lineages": len(manifest["lineages"]),  # type: ignore[index]
        "documents": len(documents),
        "cases": len(cases),
        "user_agent_contact_provided": bool(os.environ.get("SEC_USER_AGENT")),
    }
    output: Mapping[str, object] = {
        "schema": "finplan-sec-real-pilot.v1",
        "status": "small-real-data-pilot-not-benchmark",
        "protocol": protocol,
        "documents": documents,
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="preexperiments/sec_pilot_manifest.json")
    parser.add_argument("--out", default=DATA_ROOT / "sec_real_pilot_v1.json")
    parser.add_argument("--delay", type=float, default=0.12)
    args = parser.parse_args()
    result = build(Path(args.manifest), Path(args.out), args.delay)
    print(json.dumps(result["protocol"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
