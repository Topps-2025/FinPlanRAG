"""Fetch point-in-time 8-K exhibits for the revenue-guidance benchmark.

Only ``claim_source.url`` is fetched: that is the document available at the
claim cutoff.  The later earnings release in ``earnings_events.json`` is an
outcome-reveal document and must never be used as an input feature.

The downloaded files are untrusted data.  This script writes them to a
dedicated directory and never executes anything from that directory.

Usage:
    python experiments/evolution/fetch_filings.py --config experiments/evolution/config.yaml --split test --sample 30
    python experiments/evolution/fetch_filings.py --config experiments/evolution/config.yaml --all
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evolution.filing_text import html_to_text  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_records(records: list[dict], split: str | None, sample: int, seed: int) -> list[dict]:
    if split:
        records = [row for row in records if row.get("split") == split]
    records = [row for row in records if (row.get("claim_source") or {}).get("url")]
    if sample and sample < len(records):
        rng = random.Random(seed)
        records = sorted(rng.sample(records, sample), key=lambda row: row["claim_id"])
    return records


def fetch_one(
    session: requests.Session,
    record: dict,
    cache_dir: Path,
    *,
    user_agent: str,
    timeout: int,
    delay: float,
) -> dict[str, Any]:
    claim_id = str(record["claim_id"])
    url = str(record["claim_source"]["url"])
    out_path = cache_dir / f"{claim_id}.txt"
    if out_path.exists() and out_path.stat().st_size > 200:
        return {"claim_id": claim_id, "status": "cached", "chars": out_path.stat().st_size}

    time.sleep(delay)
    try:
        response = session.get(
            url,
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
        )
        if response.status_code >= 400:
            return {"claim_id": claim_id, "status": f"http_{response.status_code}", "url": url}
        content_type = (response.headers.get("content-type") or "").lower()
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            try:
                import pymupdf  # type: ignore
            except Exception as error:  # noqa: BLE001
                return {"claim_id": claim_id, "status": f"pdf_unavailable:{error}", "url": url}
            document = pymupdf.open(stream=response.content, filetype="pdf")
            text = "\n".join(page.get_text() for page in document)
        else:
            text = html_to_text(response.text)
        if len(text) < 200:
            return {"claim_id": claim_id, "status": "too_short", "url": url, "chars": len(text)}
        header = (
            f"# claim_id={claim_id}\n"
            f"# url={url}\n"
            f"# date={record.get('claim_source', {}).get('date')}\n"
        )
        out_path.write_text(header + text, encoding="utf-8")
        return {"claim_id": claim_id, "status": "ok", "chars": len(text), "url": url}
    except Exception as error:  # noqa: BLE001
        return {"claim_id": claim_id, "status": f"error:{type(error).__name__}:{error}", "url": url}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/evolution/config.yaml")
    parser.add_argument("--split", default=None, choices=["dev", "validation", "test"])
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20261118)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    config = load_config(REPO_ROOT / args.config)
    data = json.loads((REPO_ROOT / config["data"]).read_text(encoding="utf-8"))
    split = None if args.all else (args.split or "test")
    sample = 0 if args.all else (args.sample or 30)
    records = select_records(data, split, sample, args.sample_seed)

    cache_dir = REPO_ROOT / config["filing_cache"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    filing_cfg = config["filing"]
    session = requests.Session()
    session.trust_env = False

    results = []
    for index, record in enumerate(records, 1):
        result = fetch_one(
            session,
            record,
            cache_dir,
            user_agent=filing_cfg["user_agent"],
            timeout=int(filing_cfg["timeout_seconds"]),
            delay=float(filing_cfg["request_delay_seconds"]),
        )
        results.append(result)
        if index % 10 == 0 or index == len(records):
            ok = sum(1 for item in results if item["status"] in {"ok", "cached"})
            print(f"{index}/{len(records)} fetched/cached={ok}", flush=True)

    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"config": args.config, "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    ok = sum(1 for item in results if item["status"] in {"ok", "cached"})
    print(f"done: {ok}/{len(results)}; manifest={manifest_path}")


if __name__ == "__main__":
    main()
