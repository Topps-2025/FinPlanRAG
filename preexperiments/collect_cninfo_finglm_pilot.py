"""Collect the frozen FinGLM cross-year pilot reports from CNINFO.

This collector uses only the public announcement query and static PDF URLs,
with a project-scoped User-Agent.  It never reads a user's local or git email.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from .storage_paths import DATA_ROOT
except ImportError:  # direct script execution
    from storage_paths import DATA_ROOT


ROOT = Path(__file__).resolve().parent
CHINA = DATA_ROOT / "external" / "china"
MANIFEST = ROOT / "china_public_data_pilot_manifest_v1.json"
OUT_DIR = CHINA / "documents"
OUTPUT = CHINA / "china_document_acquisition_v2.json"
USER_AGENT = "FinPlan-RAG academic research research@example.com"
CNINFO_QUERY = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STOCKS = "https://www.cninfo.com.cn/new/data/szse_stock.json"
CNINFO_STATIC = "https://static.cninfo.com.cn/"
CNINFO_QUERY_CODE_ALIASES = {"900943": "600272"}


def request_json(url: str, *, data: dict | None = None) -> dict:
    body = urlencode(data).encode("utf-8") if data is not None else None
    request = Request(
        url,
        data=body,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": "https://www.cninfo.com.cn/",
            "Origin": "https://www.cninfo.com.cn",
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
        method="POST" if data is not None else "GET",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, path: Path) -> tuple[int, str, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Referer": "https://www.cninfo.com.cn/"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    path.write_bytes(payload)
    return len(payload), hashlib.sha256(payload).hexdigest(), payload[:5].decode("ascii", errors="replace")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    stocks = request_json(CNINFO_STOCKS).get("stockList", [])
    org_ids = {str(item.get("code")): str(item.get("orgId")) for item in stocks}
    rows: list[dict] = []
    failures: list[dict] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for case in cases:
        code = str(case["stock_code"])
        query_code = CNINFO_QUERY_CODE_ALIASES.get(code, code)
        org_id = org_ids.get(query_code)
        if not org_id:
            failures.append({"qid": case["qid"], "code": code, "reason": "orgId-not-found"})
            continue
        query = {
            "pageNum": "1",
            "pageSize": "100",
            "column": "sse" if query_code.startswith(("6", "9")) else "szse",
            "tabName": "fulltext",
            "plate": "sh" if query_code.startswith(("6", "9")) else "sz",
            "stock": f"{query_code},{org_id}",
            "searchkey": "",
            "secid": "",
            "category": "category_ndbg_szsh",
            "trade": "",
            "seDate": "2019-01-01~2023-12-31",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        try:
            payload = request_json(CNINFO_QUERY, data=query)
        except Exception as exc:  # pragma: no cover - depends on external state
            failures.append({"qid": case["qid"], "code": code, "reason": f"query:{exc}"})
            continue
        announcements = payload.get("announcements") or []
        for year in case["years"]:
            candidates = [
                item for item in announcements
                if re.search(rf"{year}年", str(item.get("announcementTitle", "")))
                and "摘要" not in str(item.get("announcementTitle", ""))
                and ("年度报告" in str(item.get("announcementTitle", "")) or "年报全文" in str(item.get("announcementTitle", "")))
            ]
            if not candidates:
                failures.append({"qid": case["qid"], "code": code, "year": year, "reason": "annual-report-not-found"})
                continue
            item = sorted(candidates, key=lambda row: int(row.get("announcementTime", 0)))[-1]
            relative = str(item.get("adjunctUrl", "")).lstrip("/")
            source_url = CNINFO_STATIC + relative
            target = OUT_DIR / f"{code}_{year}_annual_report.pdf"
            try:
                size, digest, magic = download(source_url, target)
                rows.append({
                    "qid": case["qid"],
                    "security_code": code,
                    "source_query_code": query_code,
                    "company": case["company"],
                    "report_year": year,
                    "document_type": "年度报告",
                    "announcement_id": str(item.get("announcementId", "")),
                    "announcement_title": item.get("announcementTitle"),
                    "announcement_date": time.strftime("%Y-%m-%d", time.gmtime(int(item.get("announcementTime", 0)) / 1000)),
                    "source_url": source_url,
                    "local_file": f"documents/{target.name}",
                    "bytes": size,
                    "sha256": digest,
                    "magic": magic,
                    "download_status": "success" if magic == "%PDF-" else "unexpected-file",
                })
            except Exception as exc:  # pragma: no cover - depends on external state
                failures.append({"qid": case["qid"], "code": code, "year": year, "source_url": source_url, "reason": f"download:{exc}"})
            time.sleep(0.4)

    result = {
        "schema": "finplan-china-document-acquisition.v2",
        "status": "official-cninfo-collection-log",
        "source_api": CNINFO_QUERY,
        "user_agent_policy": "hard-coded project placeholder; no local or git email read",
        "requested_cases": len(cases),
        "requested_documents": sum(len(case["years"]) for case in cases),
        "downloaded_documents": len(rows),
        "failures": failures,
        "documents": rows,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"downloaded_documents": len(rows), "failures": len(failures)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
