"""Official exchange announcement evidence for AnchorEvolve.

The Shenzhen Stock Exchange disclosure list hosted by cninfo is a public,
point-in-time record.  For a disclosure-date claim it is stronger evidence
than a media article because it records the actual announcement date and the
filing URL.  The module is company-agnostic: it resolves a ticker or company
name against the public stock list and queries the announcement endpoint.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from anchor_web import fetch_page

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "paper" / "results" / "anchor_cli" / "cache"
STOCK_LIST_URL = "https://www.cninfo.com.cn/new/data/szse_stock.json"
ANNOUNCEMENT_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
STATIC_BASE = "https://static.cninfo.com.cn/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _session() -> requests.Session:
    http = requests.Session()
    http.trust_env = False
    http.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Referer": "https://www.cninfo.com.cn/",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
    )
    return http


def load_szse_stocks(*, max_age_days: int = 7) -> list[dict[str, Any]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "cninfo_szse_stock.json"
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < max_age_days * 86400:
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass
    http = _session()
    response = http.get(STOCK_LIST_URL, timeout=40)
    response.raise_for_status()
    payload = response.json()
    stocks = list(payload.get("stockList") or [])
    cache_path.write_text(json.dumps(stocks, ensure_ascii=False), encoding="utf-8")
    return stocks


def lookup_stock(ticker: str = "", company: str = "") -> dict[str, Any] | None:
    try:
        stocks = load_szse_stocks()
    except Exception:
        return None
    code_match = re.search(r"(\d{6})", ticker or "")
    code = code_match.group(1) if code_match else ""
    if code:
        for row in stocks:
            if str(row.get("code")) == code:
                return row
    needle = (company or "").strip()
    if needle:
        for row in stocks:
            name = str(row.get("zwjc") or "")
            if name == needle or needle in name or name in needle:
                return row
    return None


def query_announcements(
    stock: dict[str, Any],
    *,
    category: str = "",
    searchkey: str = "",
    start: date,
    end: date,
    page_size: int = 30,
) -> list[dict[str, Any]]:
    http = _session()
    rows: list[dict[str, Any]] = []
    page = 1
    while page <= 3:
        data = {
            "pageNum": page,
            "pageSize": page_size,
            "column": "szse",
            "tabName": "fulltext",
            "plate": "",
            "stock": f"{stock.get('code')},{stock.get('orgId')}",
            "searchkey": searchkey,
            "secDate": "",
            "tdate": "",
            "type": "",
            "category": category,
            "trade": "",
            "seDate": f"{start.isoformat()}~{end.isoformat()}",
        }
        response = http.post(ANNOUNCEMENT_URL, data=data, timeout=40)
        response.raise_for_status()
        payload = response.json()
        batch = list(payload.get("announcements") or [])
        for item in batch:
            stamp = item.get("announcementTime")
            try:
                announced = datetime.fromtimestamp(int(stamp) / 1000).date()
            except (TypeError, ValueError, OSError):
                announced = None
            adjunct = str(item.get("adjunctUrl") or "")
            rows.append(
                {
                    "title": str(item.get("announcementTitle") or "").strip(),
                    "date": announced.isoformat() if announced else None,
                    "url": STATIC_BASE + adjunct if adjunct else "",
                    "adjunct_url": adjunct,
                    "searchkey": searchkey,
                    "category": category,
                }
            )
        if len(rows) >= int(payload.get("totalRecordNum") or 0) or not batch:
            break
        page += 1
    return rows


def annual_report_evidence(
    company: str,
    ticker: str,
    cutoff: date,
    *,
    lookback_days: int = 365 * 5,
) -> dict[str, Any] | None:
    stock = lookup_stock(ticker=ticker, company=company)
    if stock is None:
        return None
    start = cutoff - timedelta(days=lookback_days)
    try:
        rows = query_announcements(
            stock,
            category="category_ndbg_szsh",
            start=start,
            end=cutoff,
        )
    except Exception:
        return None
    annual = []
    for row in rows:
        title = row["title"]
        if "年度报告" not in title:
            continue
        if any(token in title for token in ("摘要", "半年度", "季度", "英文")):
            continue
        if not row.get("date"):
            continue
        annual.append(row)
    annual.sort(key=lambda row: row["date"], reverse=True)
    if not annual:
        return None
    lines = [
        f"巨潮资讯网年度报告披露记录（检索截止 {cutoff.isoformat()}）：",
    ]
    for row in annual:
        lines.append(f"- {row['date']}：{row['title']}；{row['url']}")
    text = "\n".join(lines)
    latest = annual[0]
    return {
        "title": f"巨潮资讯网：{stock.get('zwjc') or company}年度报告披露记录",
        "url": latest["url"] or "https://www.cninfo.com.cn/",
        "snippet": text[:600],
        "text": text,
        "date": latest["date"],
        "date_known": True,
        "retrieved_at": date.today().isoformat(),
        "query": f"cninfo:{stock.get('code')}:category_ndbg_szsh",
        "provider": "cninfo",
        "official": True,
        "source_type": "exchange_announcement",
        "query_index": -1,
        "facet": "date_disclosure",
    }


def metric_announcement_evidence(
    company: str,
    ticker: str,
    cutoff: date,
    *,
    year: int | None = None,
    max_items: int = 4,
    lookback_days: int = 365 * 2,
) -> list[dict[str, Any]]:
    """Official reports and targets for a numeric metric claim.

    The function gathers, in priority order: the latest quarterly report, the
    latest semiannual report, the annual report (prior-year actual), the
    target-year incentive-plan draft, and the latest earnings preview.  All
    documents are public cninfo records; the PDF text is parsed as text and
    never executed.
    """

    stock = lookup_stock(ticker=ticker, company=company)
    if stock is None:
        return []
    start = cutoff - timedelta(days=lookback_days)
    candidates: list[dict[str, Any]] = []

    def add(
        rows: list[dict[str, Any]],
        *,
        facet: str,
        priority: int,
        require_year: bool = False,
    ) -> None:
        filtered = [
            row
            for row in rows
            if row.get("date")
            and (not require_year or not year or str(year) in row["title"])
        ]
        filtered.sort(key=lambda row: row.get("date") or "", reverse=True)
        if filtered:
            candidates.append(
                {
                    "row": filtered[0],
                    "facet": facet,
                    "priority": priority,
                }
            )

    for category, label, facet, priority in (
        ("category_sjdbg_szsh", "季度报告", "latest_period", 0),
        ("category_bndbg_szsh", "半年度报告", "latest_period", 1),
        ("category_ndbg_szsh", "年度报告", "prior_period", 3),
    ):
        try:
            rows = query_announcements(
                stock,
                category=category,
                start=start,
                end=cutoff,
            )
        except Exception:
            rows = []
        rows = [
            row
            for row in rows
            if label in row["title"] and "摘要" not in row["title"]
        ]
        add(rows, facet=facet, priority=priority, require_year=category != "category_ndbg_szsh")

    try:
        incentive_rows = query_announcements(
            stock,
            searchkey="限制性股票",
            start=start,
            end=cutoff,
        )
    except Exception:
        incentive_rows = []
    drafts = [
        row
        for row in incentive_rows
        if "激励计划" in row["title"] and "草案" in row["title"]
    ]
    if not drafts:
        drafts = [row for row in incentive_rows if "激励计划" in row["title"]]
    if year and drafts:
        year_drafts = [row for row in drafts if str(year) in row["title"]]
        if year_drafts:
            drafts = year_drafts
    drafts.sort(key=lambda row: row.get("date") or "", reverse=True)
    drafts.sort(key=lambda row: 0 if "摘要" in row["title"] else 1)
    if drafts:
        candidates.append(
            {"row": drafts[0], "facet": "forward_disclosure", "priority": 2}
        )

    try:
        forecast_rows = query_announcements(
            stock,
            searchkey="业绩预告",
            start=start,
            end=cutoff,
        )
    except Exception:
        forecast_rows = []
    forecasts = [row for row in forecast_rows if "业绩预告" in row["title"]]
    if year:
        year_forecasts = [row for row in forecasts if str(year) in row["title"]]
        if year_forecasts:
            forecasts = year_forecasts
    forecasts.sort(key=lambda row: row.get("date") or "", reverse=True)
    if forecasts:
        candidates.append(
            {"row": forecasts[0], "facet": "forward_disclosure", "priority": 4}
        )

    seen_urls: set[str] = set()
    ordered = []
    for candidate in sorted(candidates, key=lambda item: item["priority"]):
        url = str(candidate["row"].get("url") or "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        ordered.append(candidate)

    items: list[dict[str, Any]] = []
    for candidate in ordered[: max(1, max_items)]:
        row = candidate["row"]
        page = fetch_page(row["url"])
        text = str((page or {}).get("text") or "")
        if len(text) < 200:
            text = f"{row['title']}\n" + str((page or {}).get("snippet") or "")
        if not text.strip():
            continue
        items.append(
            {
                "title": row["title"],
                "url": row["url"],
                "snippet": text[:600],
                "text": text,
                "date": row.get("date"),
                "date_known": bool(row.get("date")),
                "retrieved_at": date.today().isoformat(),
                "query": f"cninfo:{stock.get('code')}:{row.get('searchkey') or row.get('category') or 'announcement'}",
                "provider": "cninfo",
                "official": True,
                "source_type": "exchange_announcement",
                "query_index": -1,
                "facet": candidate["facet"],
            }
        )
    return items
