"""Collect official CNINFO semiannual/Q3 reports for the frozen China mechanism pilot."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from collect_cninfo_finglm_pilot import (
    CNINFO_QUERY,
    CNINFO_QUERY_CODE_ALIASES,
    CNINFO_STATIC,
    CNINFO_STOCKS,
    USER_AGENT,
    download,
    request_json,
)
try:
    from .storage_paths import DATA_ROOT
except ImportError:  # direct script execution
    from storage_paths import DATA_ROOT

ROOT = Path(__file__).resolve().parent
CHINA = DATA_ROOT / "external" / "china"
MANIFEST = ROOT / "china_mixed_period_pilot_manifest_v1.json"
ACQUISITION = CHINA / "china_mixed_period_acquisition_v1.json"
OUT_DIR = CHINA / "documents" / "mixed_period"

TYPE_CONFIG = {
    "H1": ("category_bndbg_szsh", "半年度报告", re.compile(r"(?:20\d{2}年)?半年度报告$")),
    "Q3": ("category_sjdbg_szsh", "第三季度报告", re.compile(r"(?:20\d{2}年)?(?:第三季度报告|三季度报告)(?:正文)?$")),
}


def choose_announcement(items: list[dict], year: int, label: str) -> dict | None:
    _, _, pattern = TYPE_CONFIG[label]
    candidates = []
    for item in items:
        title = str(item.get("announcementTitle", ""))
        if "摘要" in title or "更正" in title or "修订" in title:
            continue
        if not re.search(rf"{year}年", title) and not re.search(rf"(?<!\d){year}(?!\d)", title):
            continue
        if pattern.search(title) or (label == "Q3" and ("第三季度报告" in title or "三季度报告" in title)):
            candidates.append(item)
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: int(row.get("announcementTime", 0)))[-1]


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stocks = request_json(CNINFO_STOCKS).get("stockList", [])
    org_ids = {str(item.get("code")): str(item.get("orgId")) for item in stocks}
    rows: list[dict] = []
    failures: list[dict] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for case in manifest["cases"]:
        code = str(case["stock_code"])
        query_code = CNINFO_QUERY_CODE_ALIASES.get(code, code)
        org_id = org_ids.get(query_code)
        if not org_id:
            failures.append({"qid": case["qid"], "reason": "orgId-not-found", "code": code})
            continue
        for period in sorted(TYPE_CONFIG):
            category, document_type, _ = TYPE_CONFIG[period]
            query = {
                "pageNum": "1", "pageSize": "100", "column": "sse" if query_code.startswith(("6", "9")) else "szse",
                "tabName": "fulltext", "plate": "sh" if query_code.startswith(("6", "9")) else "sz",
                "stock": f"{query_code},{org_id}", "searchkey": "", "secid": "", "category": category,
                "trade": "", "seDate": f"{case['year']}-01-01~{case['year'] + 1}-12-31", "sortName": "", "sortType": "", "isHLtitle": "true"
            }
            try:
                payload = request_json(CNINFO_QUERY, data=query)
                item = choose_announcement(payload.get("announcements") or [], int(case["year"]), period)
            except Exception as exc:  # pragma: no cover - external service
                failures.append({"qid": case["qid"], "period": period, "reason": f"query:{exc}"})
                continue
            if not item:
                failures.append({"qid": case["qid"], "period": period, "reason": "report-not-found"})
                continue
            relative = str(item.get("adjunctUrl", "")).lstrip("/")
            source_url = CNINFO_STATIC + relative
            target = OUT_DIR / f"{code}_{case['year']}_{period}.pdf"
            try:
                size, digest, magic = download(source_url, target)
                rows.append({
                    "qid": case["qid"], "security_code": code, "source_query_code": query_code,
                    "company": case["company"], "fiscal_year": int(case["year"]), "fiscal_period": period,
                    "document_type": document_type, "announcement_id": str(item.get("announcementId", "")),
                    "announcement_title": item.get("announcementTitle"), "announcement_date": time.strftime("%Y-%m-%d", time.gmtime(int(item.get("announcementTime", 0)) / 1000)),
                    "source_url": source_url, "local_file": f"documents/mixed_period/{target.name}",
                    "bytes": size, "sha256": digest, "magic": magic,
                    "download_status": "success" if magic == "%PDF-" else "unexpected-file",
                })
            except Exception as exc:  # pragma: no cover - external service
                failures.append({"qid": case["qid"], "period": period, "source_url": source_url, "reason": f"download:{exc}"})
            time.sleep(0.35)

    result = {
        "schema": "finplan-china-mixed-period-acquisition.v1", "status": "official-cninfo-collection-log",
        "source_api": CNINFO_QUERY, "user_agent": USER_AGENT, "requested_cases": len(manifest["cases"]),
        "requested_documents": len(manifest["cases"]) * len(TYPE_CONFIG),
        "downloaded_documents": len(rows), "failures": failures, "documents": rows,
    }
    ACQUISITION.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"downloaded_documents": len(rows), "failures": len(failures)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
