"""Build the historical A-share annual-report backtest.

For each company and fiscal year Y, the builder asks a forward-looking
question at January 31 of Y+1.  At that cutoff the Q3 report is public and the
annual report is not.  Financial statement values come from akshare's public
statement table; announcement dates and URLs come from cninfo.  No outcome or
label is invented.

Output: ``paper/data/ashare_backtest/claims.json`` plus ``manifest.json``.
The file is a benchmark, not a model result.

Usage:
    python scripts/build_ashare_backtest.py \
      --config configs/ashare_backtest.yaml
"""

from __future__ import annotations

import argparse
import functools
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from anchor_filings import lookup_stock, query_announcements  # noqa: E402

ANNUAL_CATEGORY = "category_ndbg_szsh"
Q3_CATEGORY = "category_sjdbg_szsh"


def _value(frame: pd.DataFrame, indicator: str, period: str) -> float | None:
    rows = frame[frame["指标"] == indicator]
    if rows.empty:
        return None
    raw = rows.iloc[0].get(period)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if pd.isna(value):
        return None
    return value


@functools.lru_cache(maxsize=None)
def _announcement(ticker: str, category: str, start: date, end: date) -> dict[str, Any] | None:
    stock = lookup_stock(ticker)
    if not stock:
        return None
    try:
        rows = query_announcements(stock, category=category, start=start, end=end)
    except Exception:
        return None
    for row in rows or []:
        title = str(row.get("title") or "")
        if not title:
            continue
        if category == ANNUAL_CATEGORY:
            if "年度报告" not in title or any(word in title for word in ("摘要", "英文", "已取消")):
                continue
        if category == Q3_CATEGORY:
            if "第三季度报告" not in title or any(word in title for word in ("摘要", "英文", "已取消")):
                continue
        return row
    return None


def _num(value: float | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _band_labels(prior: float, band: float, unit: str) -> tuple[list[str], list[dict], dict]:
    low, high = prior - band, prior + band
    labels = [
        f"低于{low:.4g}{unit}",
        f"{low:.4g}{unit}至{high:.4g}{unit}",
        f"高于{high:.4g}{unit}",
    ]
    thresholds = [
        {"label": labels[0], "low": None, "high": low},
        {"label": labels[1], "low": low, "high": high},
        {"label": labels[2], "low": high, "high": None},
    ]
    defs = {
        labels[0]: f"实际值低于{low:.4g}{unit}",
        labels[1]: f"实际值在{low:.4g}{unit}至{high:.4g}{unit}之间（含下限，不含上限）",
        labels[2]: f"实际值不低于{high:.4g}{unit}",
    }
    return labels, thresholds, defs


def _outcome_label(value: float, thresholds: list[dict]) -> str:
    for row in thresholds:
        low, high = row.get("low"), row.get("high")
        if (low is None or value >= float(low)) and (high is None or value < float(high)):
            return str(row["label"])
    raise ValueError(f"value {value} does not fit thresholds")


def _text_evidence(
    evidence_id: str,
    title: str,
    date_value: str,
    url: str,
    text: str,
    facet: str,
    metric: str,
    value: float | None,
    unit: str,
    period: str,
) -> dict[str, Any]:
    return {
        "id": evidence_id,
        "title": title,
        "date": date_value,
        "url": url,
        "text": text,
        "facet": facet,
        "metric": metric,
        "value": value,
        "unit": unit,
        "period": period,
    }


def build_claims_for_company(
    company: dict[str, Any],
    years: list[int],
    bands: dict[str, float],
) -> tuple[list[dict[str, Any]], list[str]]:
    ticker = str(company["ticker"])
    code = re.search(r"(\d{6})", ticker)
    if not code:
        return [], [f"{ticker}: no six-digit code"]
    symbol = code.group(1)
    try:
        frame = ak.stock_financial_abstract(symbol=symbol)
    except Exception as error:
        return [], [f"{ticker}: financial abstract failed: {error}"]

    claims: list[dict[str, Any]] = []
    missing: list[str] = []
    for year in years:
        prev = year - 1
        prev2 = year - 2
        cutoff = date(year + 1, 1, 31)
        revenue = {
            "prior2": _value(frame, "营业总收入", f"{prev2}1231"),
            "prior": _value(frame, "营业总收入", f"{prev}1231"),
            "ytd": _value(frame, "营业总收入", f"{year}0930"),
            "prior_ytd": _value(frame, "营业总收入", f"{prev}0930"),
            "actual": _value(frame, "营业总收入", f"{year}1231"),
        }
        profit = {
            "prior2": _value(frame, "归母净利润", f"{prev2}1231"),
            "prior": _value(frame, "归母净利润", f"{prev}1231"),
            "ytd": _value(frame, "归母净利润", f"{year}0930"),
            "prior_ytd": _value(frame, "归母净利润", f"{prev}0930"),
            "actual": _value(frame, "归母净利润", f"{year}1231"),
        }
        margin = {
            "prior": _value(frame, "毛利率", f"{prev}1231"),
            "ytd": _value(frame, "毛利率", f"{year}0930"),
            "prior_ytd": _value(frame, "毛利率", f"{prev}0930"),
            "actual": _value(frame, "毛利率", f"{year}1231"),
        }

        annual_url = ""
        q3_url = ""
        annual = _announcement(ticker, ANNUAL_CATEGORY, date(year + 1, 1, 1), date(year + 1, 6, 30))
        q3 = _announcement(ticker, Q3_CATEGORY, date(year, 10, 1), date(year, 11, 30))
        annual_prev = _announcement(ticker, ANNUAL_CATEGORY, date(prev + 1, 1, 1), date(prev + 1, 6, 30))
        annual_prev2 = _announcement(ticker, ANNUAL_CATEGORY, date(prev2 + 1, 1, 1), date(prev2 + 1, 6, 30))
        q3_prev = _announcement(ticker, Q3_CATEGORY, date(prev, 10, 1), date(prev, 11, 30))
        if annual:
            annual_url = str(annual.get("url") or "")
        if q3:
            q3_url = str(q3.get("url") or "")
        annual_prev_url = str(annual_prev.get("url") or "") if annual_prev else ""
        annual_prev2_url = str(annual_prev2.get("url") or "") if annual_prev2 else ""
        q3_prev_url = str(q3_prev.get("url") or "") if q3_prev else ""

        # Revenue growth.
        if all(revenue[key] is not None and revenue[key] > 0 for key in ("prior2", "prior", "ytd", "actual")):
            prior_growth = (float(revenue["prior"]) / float(revenue["prior2"]) - 1.0) * 100.0
            actual_growth = (float(revenue["actual"]) / float(revenue["prior"]) - 1.0) * 100.0
            labels, thresholds, defs = _band_labels(prior_growth, float(bands["revenue_growth_pp"]), "%")
            evidence = [
                _text_evidence(
                    "E1", f"{prev2}年年度报告", f"{prev2}-12-31", annual_prev2_url,
                    f"{prev2}年年度报告：营业总收入 {_num(revenue['prior2'] / 1e8)} 亿元。",
                    "prior_period", "营业收入", revenue["prior2"] / 1e8, "亿元", f"{prev2}FY",
                ),
                _text_evidence(
                    "E2", f"{prev}年年度报告", f"{prev}-12-31", annual_prev_url,
                    f"{prev}年年度报告：营业总收入 {_num(revenue['prior'] / 1e8)} 亿元。",
                    "prior_period", "营业收入", revenue["prior"] / 1e8, "亿元", f"{prev}FY",
                ),
                _text_evidence(
                    "E3", f"{year}年三季度报告", f"{year}-09-30", q3_url,
                    f"{year}年三季度报告：前三季度营业总收入 {_num(revenue['ytd'] / 1e8)} 亿元。",
                    "latest_period", "营业收入", revenue["ytd"] / 1e8, "亿元", f"{year}Q1-Q3",
                ),
                _text_evidence(
                    "E4", f"{prev}年三季度报告", f"{prev}-09-30", q3_prev_url,
                    f"{prev}年三季度报告：前三季度营业总收入 {_num(revenue['prior_ytd'] / 1e8)} 亿元。",
                    "prior_period", "营业收入", revenue["prior_ytd"] / 1e8, "亿元", f"{prev}Q1-Q3",
                ),
            ]
            claims.append({
                "claim_id": f"ashare_{symbol}_{year}_revenue_growth",
                "company": company["name"],
                "ticker": ticker,
                "year": year,
                "cutoff": cutoff.isoformat(),
                "target_period": f"{year}年年度报告",
                "metric": "营业收入同比增速",
                "metric_kind": "growth",
                "model_template": "growth_bridge",
                "labels": labels,
                "thresholds": thresholds,
                "label_defs": defs,
                "reference_value": float(revenue["prior"]) / 1e8,
                "reference_unit": "亿元",
                "unit": "%",
                "prior_growth": prior_growth,
                "evidence": evidence,
                "outcome": {
                    "label": _outcome_label(actual_growth, thresholds),
                    "value": actual_growth,
                    "unit": "%",
                    "date": str(annual.get("date") if annual else ""),
                    "source": annual_url,
                },
            })
        else:
            missing.append(f"{ticker} {year}: revenue growth inputs")

        # Net profit growth.
        if all(profit[key] is not None and profit[key] > 0 for key in ("prior2", "prior", "ytd", "actual")):
            prior_growth = (float(profit["prior"]) / float(profit["prior2"]) - 1.0) * 100.0
            actual_growth = (float(profit["actual"]) / float(profit["prior"]) - 1.0) * 100.0
            labels, thresholds, defs = _band_labels(prior_growth, float(bands["net_profit_growth_pp"]), "%")
            claims.append({
                "claim_id": f"ashare_{symbol}_{year}_net_profit_growth",
                "company": company["name"],
                "ticker": ticker,
                "year": year,
                "cutoff": cutoff.isoformat(),
                "target_period": f"{year}年年度报告",
                "metric": "归母净利润同比增速",
                "metric_kind": "growth",
                "model_template": "growth_bridge",
                "labels": labels,
                "thresholds": thresholds,
                "label_defs": defs,
                "reference_value": float(profit["prior"]) / 1e8,
                "reference_unit": "亿元",
                "unit": "%",
                "prior_growth": prior_growth,
                "evidence": [
                    _text_evidence(
                        "E1", f"{prev2}年年度报告", f"{prev2}-12-31", annual_prev2_url,
                        f"{prev2}年年度报告：归母净利润 {_num(profit['prior2'] / 1e8)} 亿元。",
                        "prior_period", "归母净利润", profit["prior2"] / 1e8, "亿元", f"{prev2}FY",
                    ),
                    _text_evidence(
                        "E2", f"{prev}年年度报告", f"{prev}-12-31", annual_prev_url,
                        f"{prev}年年度报告：归母净利润 {_num(profit['prior'] / 1e8)} 亿元。",
                        "prior_period", "归母净利润", profit["prior"] / 1e8, "亿元", f"{prev}FY",
                    ),
                    _text_evidence(
                        "E3", f"{year}年三季度报告", f"{year}-09-30", q3_url,
                        f"{year}年三季度报告：前三季度归母净利润 {_num(profit['ytd'] / 1e8)} 亿元。",
                        "latest_period", "归母净利润", profit["ytd"] / 1e8, "亿元", f"{year}Q1-Q3",
                    ),
                    _text_evidence(
                        "E4", f"{prev}年三季度报告", f"{prev}-09-30", q3_prev_url,
                        f"{prev}年三季度报告：前三季度归母净利润 {_num(profit['prior_ytd'] / 1e8)} 亿元。",
                        "prior_period", "归母净利润", profit["prior_ytd"] / 1e8, "亿元", f"{prev}Q1-Q3",
                    ),
                ],
                "outcome": {
                    "label": _outcome_label(actual_growth, thresholds),
                    "value": actual_growth,
                    "unit": "%",
                    "date": str(annual.get("date") if annual else ""),
                    "source": annual_url,
                },
            })
        else:
            missing.append(f"{ticker} {year}: net profit growth inputs")

        # Gross margin.
        if all(margin[key] is not None for key in ("prior", "ytd", "actual")):
            labels, thresholds, defs = _band_labels(float(margin["prior"]), float(bands["gross_margin_pp"]), "pp")
            claims.append({
                "claim_id": f"ashare_{symbol}_{year}_gross_margin",
                "company": company["name"],
                "ticker": ticker,
                "year": year,
                "cutoff": cutoff.isoformat(),
                "target_period": f"{year}年年度报告",
                "metric": "综合毛利率",
                "metric_kind": "margin",
                "model_template": "margin_bridge",
                "labels": labels,
                "thresholds": thresholds,
                "label_defs": defs,
                "reference_value": float(margin["prior"]),
                "reference_unit": "%",
                "unit": "%",
                "evidence": [
                    _text_evidence(
                        "E1", f"{prev}年年度报告", f"{prev}-12-31", annual_prev_url,
                        f"{prev}年年度报告：综合毛利率 {_num(margin['prior'])}%。",
                        "prior_period", "毛利率", float(margin["prior"]), "%", f"{prev}FY",
                    ),
                    _text_evidence(
                        "E2", f"{year}年三季度报告", f"{year}-09-30", q3_url,
                        f"{year}年三季度报告：前三季度综合毛利率 {_num(margin['ytd'])}%。",
                        "latest_period", "毛利率", float(margin["ytd"]), "%", f"{year}Q1-Q3",
                    ),
                    _text_evidence(
                        "E3", f"{prev}年三季度报告", f"{prev}-09-30", q3_prev_url,
                        f"{prev}年三季度报告：前三季度综合毛利率 {_num(margin['prior_ytd'])}%。",
                        "prior_period", "毛利率", float(margin["prior_ytd"]) if margin["prior_ytd"] is not None else None,
                        "%", f"{prev}Q1-Q3",
                    ),
                ],
                "outcome": {
                    "label": _outcome_label(float(margin["actual"]), thresholds),
                    "value": float(margin["actual"]),
                    "unit": "%",
                    "date": str(annual.get("date") if annual else ""),
                    "source": annual_url,
                },
            })
        else:
            missing.append(f"{ticker} {year}: gross margin inputs")

        # Disclosure date.
        history = []
        for prior_year in range(year - 4, year):
            row = _announcement(
                ticker, ANNUAL_CATEGORY,
                date(prior_year + 1, 1, 1), date(prior_year + 1, 6, 30),
            )
            if row:
                history.append({"year": prior_year, "date": str(row.get("date")), "title": str(row.get("title")), "url": str(row.get("url") or "")})
        if annual and annual.get("date"):
            actual_date = str(annual["date"])
            year_next = year + 1
            labels = [
                f"不晚于{year_next}-03-31",
                f"{year_next}-04-01至{year_next}-04-15",
                f"晚于{year_next}-04-15",
            ]
            windows = [
                {"label": labels[0], "start": f"{year_next}-01-01", "end": f"{year_next}-03-31"},
                {"label": labels[1], "start": f"{year_next}-04-01", "end": f"{year_next}-04-15"},
                {"label": labels[2], "start": f"{year_next}-04-16", "end": f"{year_next}-04-30"},
            ]
            if actual_date <= f"{year_next}-03-31":
                outcome_label = labels[0]
            elif actual_date <= f"{year_next}-04-15":
                outcome_label = labels[1]
            else:
                outcome_label = labels[2]
            history_text = "；".join(f"{item['year']}年年报于{item['date']}披露" for item in history) or "无历史记录"
            claims.append({
                "claim_id": f"ashare_{symbol}_{year}_disclosure_date",
                "company": company["name"],
                "ticker": ticker,
                "year": year,
                "cutoff": cutoff.isoformat(),
                "target_period": f"{year}年年度报告",
                "metric": "年度报告披露日期",
                "metric_kind": "date",
                "model_template": "date_window",
                "labels": labels,
                "date_windows": windows,
                "label_defs": {
                    labels[0]: f"{year}年年度报告在{year_next}年3月31日或之前披露",
                    labels[1]: f"{year}年年度报告在{year_next}年4月1日至4月15日之间披露",
                    labels[2]: f"{year}年年度报告在{year_next}年4月15日之后披露，但不晚于法定截止日4月30日",
                },
                "reference": f"历史披露：{history_text}；法定截止日{year_next}-04-30",
                "evidence": [
                    _text_evidence(
                        "E1", "历史年度报告披露记录", f"{year}-12-31", annual_url,
                        f"{history_text}。", "date_disclosure", "年度报告披露日期", None, "date", "history",
                    )
                ],
                "outcome": {
                    "label": outcome_label,
                    "value": actual_date,
                    "unit": "date",
                    "date": actual_date,
                    "source": annual_url,
                },
            })
        else:
            missing.append(f"{ticker} {year}: annual report disclosure date")
    return claims, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ashare_backtest.yaml")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    config = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    out_dir = REPO_ROOT / (args.out or config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    claims: list[dict[str, Any]] = []
    missing: list[str] = []
    for company in config["companies"]:
        company_claims, company_missing = build_claims_for_company(
            company, [int(year) for year in config["years"]], config["bands"]
        )
        claims.extend(company_claims)
        missing.extend(company_missing)

    claims.sort(key=lambda row: (row["ticker"], row["year"], row["metric"]))
    payload = {
        "schema": "finplan-ashare-annual-backtest-v1",
        "config": str(REPO_ROOT / args.config),
        "cutoff_template": config["cutoff_template"],
        "years": config["years"],
        "bands": config["bands"],
        "companies": config["companies"],
        "n_claims": len(claims),
        "claims": claims,
    }
    (out_dir / "claims.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schema": payload["schema"],
        "n_claims": len(claims),
        "n_companies": len(config["companies"]),
        "years": config["years"],
        "by_metric": {
            metric: sum(1 for row in claims if row["metric_kind"] == metric)
            for metric in ("growth", "margin", "date")
        },
        "missing": missing,
        "output": str(out_dir / "claims.json"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
