"""AnchorEvolve engine for forward-looking financial claims.

The engine turns a question into an explicit claim contract, retrieves
point-in-time web evidence, extracts source-grounded anchors with an LLM,
qualifies the anchors, updates a belief, and selects an action in the
full-action loss-aware readout.  Every intermediate object is written to an
audit directory so that a reader can trace a conclusion back to a source.

The module has no company-specific branches.  Case definitions live in YAML
files under ``configs/anchor_cases``; thresholds, models and search settings
live in ``configs/anchor_cli.yaml``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import secrets
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from string import Template
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from anchor_filings import (  # noqa: E402
    annual_report_evidence,
    metric_announcement_evidence,
)
from anchor_web import find_date, gather  # noqa: E402
from deepseek_client import LLMClient, LLMError  # noqa: E402
from forecast_bridge import bridge_anchor, build_forecast  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "anchor_cli.yaml"
PROMPT_DIR = REPO_ROOT / "experiments" / "anchor_cli" / "prompts"
ANCHOR_PROMPT = PROMPT_DIR / "anchor_general.txt"
PARSE_PROMPT = PROMPT_DIR / "parse_question.txt"
DEFAULT_OUT = REPO_ROOT / "paper" / "results" / "anchor_cli"

ALLOWED_SCOPES = {"company", "sector", "market", "regulation"}
DATE_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")


# ---------------------------------------------------------------------------
# Configuration and environment
# ---------------------------------------------------------------------------


def load_env(path: Path | None = None) -> None:
    """Load ``.env`` without overriding variables already set by the shell."""

    env_path = path or (REPO_ROOT / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    payload.setdefault("llm", {})
    payload.setdefault("search", {})
    payload.setdefault("anchors", {})
    payload.setdefault("loss", {})
    return payload


def parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    match = DATE_RE.search(text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def make_client(config: Mapping[str, Any]) -> LLMClient:
    llm = config.get("llm", {})
    return LLMClient(
        provider=str(llm.get("provider") or "deepseek"),
        model=str(llm.get("model") or "") or None,
        base_url=str(llm.get("base_url") or "") or None,
        api_key_env=str(llm.get("api_key_env") or "ANTHROPIC_AUTH_TOKEN"),
        cache_dir=REPO_ROOT / str(llm.get("cache_dir") or "paper/results/anchor_cli/llm_cache"),
        timeout=int(llm.get("timeout") or 180),
        max_retries=int(llm.get("max_retries") or 4),
        reasoning_effort=llm.get("reasoning_effort"),
        extra_body=llm.get("extra_body"),
    )


# ---------------------------------------------------------------------------
# Claim contract
# ---------------------------------------------------------------------------


def infer_metric_kind(metric: str, claim_type: str) -> str:
    if claim_type == "date":
        return "date"
    text = metric or ""
    if "毛利率" in text:
        return "margin"
    if "产能" in text:
        return "capacity"
    if "产量" in text or "排产" in text or "出货" in text:
        return "production"
    if "指引" in text:
        return "disclosure"
    if "增速" in text or "同比" in text:
        return "growth"
    return "level"


DEFAULT_MODEL_TEMPLATES = {
    "level": "level_bridge",
    "growth": "growth_bridge",
    "margin": "margin_bridge",
    "capacity": "capacity_bridge",
    "production": "production_bridge",
    "date": "date_window",
    "disclosure": "disclosure_check",
    "other": "level_bridge",
}


@dataclass
class ClaimContract:
    claim_id: str
    text: str
    metric: str
    claim_type: str
    labels: list[str]
    label_defs: dict[str, str] = field(default_factory=dict)
    target_period: str = ""
    unit: str = ""
    reference: str = ""
    queries: list[str] = field(default_factory=list)
    date_windows: list[dict[str, str]] = field(default_factory=list)
    prior: dict[str, float] = field(default_factory=dict)
    metric_kind: str = ""
    model_template: str = ""
    anchor_types: list[str] = field(default_factory=list)
    query_facets: list[dict[str, Any]] = field(default_factory=list)
    thresholds: list[dict[str, Any]] = field(default_factory=list)
    reference_value: float | None = None
    reference_unit: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], index: int = 0) -> "ClaimContract":
        labels = [
            str(item).strip()
            for item in (payload.get("labels") or [])
            if str(item).strip()
        ]
        label_defs = {
            str(key).strip(): str(value).strip()
            for key, value in (payload.get("label_defs") or {}).items()
        }
        if not labels and label_defs:
            labels = list(label_defs)
        windows = []
        for row in payload.get("date_windows") or []:
            label = str(row.get("label") or "").strip()
            start = str(row.get("start") or "").strip()
            end = str(row.get("end") or "").strip()
            if label and start and end:
                windows.append({"label": label, "start": start, "end": end})
        facets = []
        for row in payload.get("query_facets") or []:
            name = str(row.get("name") or "").strip()
            values = row.get("queries") if row.get("queries") is not None else row.get("templates")
            queries = [str(item).strip() for item in (values or []) if str(item).strip()]
            if name and queries:
                facets.append(
                    {
                        "name": name,
                        "required": bool(row.get("required", False)),
                        "queries": queries,
                    }
                )
        thresholds = []
        for row in payload.get("thresholds") or []:
            label = str(row.get("label") or "").strip()
            if label not in labels:
                continue
            low = row.get("low")
            high = row.get("high")
            try:
                low = float(low) if low is not None and str(low).strip() else None
            except (TypeError, ValueError):
                low = None
            try:
                high = float(high) if high is not None and str(high).strip() else None
            except (TypeError, ValueError):
                high = None
            thresholds.append({"label": label, "low": low, "high": high})
        try:
            reference_value = (
                float(payload.get("reference_value"))
                if payload.get("reference_value") is not None
                and str(payload.get("reference_value")).strip()
                else None
            )
        except (TypeError, ValueError):
            reference_value = None
        claim_id = str(payload.get("claim_id") or "").strip() or f"claim_{index + 1}"
        claim_type = str(payload.get("claim_type") or "metric").strip().lower()
        metric = str(payload.get("metric") or "").strip()
        metric_kind = str(payload.get("metric_kind") or "").strip().lower()
        if not metric_kind:
            metric_kind = infer_metric_kind(metric, claim_type)
        model_template = str(payload.get("model_template") or "").strip().lower()
        if not model_template:
            model_template = DEFAULT_MODEL_TEMPLATES.get(metric_kind, "level_bridge")
        return cls(
            claim_id=claim_id,
            text=str(payload.get("text") or payload.get("claim") or "").strip(),
            metric=metric,
            claim_type=claim_type,
            labels=labels,
            label_defs=label_defs,
            target_period=str(payload.get("target_period") or "").strip(),
            unit=str(payload.get("unit") or "").strip(),
            reference=str(payload.get("reference") or "").strip(),
            queries=[str(item).strip() for item in (payload.get("queries") or []) if str(item).strip()],
            date_windows=windows,
            prior={
                str(key): float(value)
                for key, value in (payload.get("prior") or {}).items()
            },
            metric_kind=metric_kind,
            model_template=model_template,
            anchor_types=[
                str(item).strip()
                for item in (payload.get("anchor_types") or [])
                if str(item).strip()
            ],
            query_facets=facets,
            thresholds=thresholds,
            reference_value=reference_value,
            reference_unit=str(payload.get("reference_unit") or "").strip(),
        )

    def validate(self) -> None:
        if not self.text:
            raise ValueError(f"claim {self.claim_id} has no text")
        if not self.metric:
            raise ValueError(f"claim {self.claim_id} has no metric")
        if not 2 <= len(self.labels) <= 6:
            raise ValueError(
                f"claim {self.claim_id} must have between 2 and 6 labels, got {len(self.labels)}"
            )
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"claim {self.claim_id} has duplicate labels")
        if self.claim_type not in {"metric", "date"}:
            raise ValueError(f"claim {self.claim_id} has unknown claim_type")
        if self.model_template not in {
            "level_bridge",
            "growth_bridge",
            "margin_bridge",
            "capacity_bridge",
            "production_bridge",
            "date_window",
            "disclosure_check",
        }:
            raise ValueError(
                f"claim {self.claim_id} has unknown model_template: {self.model_template}"
            )
        for row in self.thresholds:
            if row.get("label") not in self.labels:
                raise ValueError(
                    f"claim {self.claim_id} has a threshold for an unknown label: {row}"
                )
        if self.claim_type == "date":
            if not self.date_windows:
                raise ValueError(f"date claim {self.claim_id} needs date_windows")
            window_labels = {row["label"] for row in self.date_windows}
            missing = [label for label in self.labels if label not in window_labels]
            if missing:
                raise ValueError(
                    f"date claim {self.claim_id} has labels without windows: {missing}"
                )
            for row in self.date_windows:
                start, end = parse_date(row["start"]), parse_date(row["end"])
                if start is None or end is None or start > end:
                    raise ValueError(
                        f"date claim {self.claim_id} has an invalid window: {row}"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "metric": self.metric,
            "claim_type": self.claim_type,
            "labels": list(self.labels),
            "label_defs": dict(self.label_defs),
            "target_period": self.target_period,
            "unit": self.unit,
            "reference": self.reference,
            "queries": list(self.queries),
            "date_windows": list(self.date_windows),
            "prior": dict(self.prior),
            "metric_kind": self.metric_kind,
            "model_template": self.model_template,
            "anchor_types": list(self.anchor_types),
            "query_facets": list(self.query_facets),
            "thresholds": list(self.thresholds),
            "reference_value": self.reference_value,
            "reference_unit": self.reference_unit,
        }


@dataclass
class CaseSpec:
    case_id: str
    question: str
    company: str
    ticker: str
    target_period: str
    claims: list[ClaimContract]
    cutoff: str = ""
    source: str = ""

    def validate(self) -> None:
        if not self.company:
            raise ValueError("case has no company")
        if not self.claims:
            raise ValueError("case has no claims")
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("case has duplicate claim_id values")
        for claim in self.claims:
            claim.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "company": self.company,
            "ticker": self.ticker,
            "target_period": self.target_period,
            "cutoff": self.cutoff,
            "source": self.source,
            "claims": [claim.to_dict() for claim in self.claims],
        }


def load_case(path: Path | str) -> CaseSpec:
    case_path = Path(path)
    with case_path.open(encoding="utf-8") as handle:
        if case_path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle) or {}
    claims = [
        ClaimContract.from_dict(row, index)
        for index, row in enumerate(payload.get("claims") or [])
    ]
    spec = CaseSpec(
        case_id=str(payload.get("case_id") or case_path.stem),
        question=str(payload.get("question") or ""),
        company=str(payload.get("company") or ""),
        ticker=str(payload.get("ticker") or ""),
        target_period=str(payload.get("target_period") or ""),
        claims=claims,
        cutoff=str(payload.get("cutoff") or ""),
        source=str(case_path),
    )
    spec.validate()
    return spec


def claim_from_payload(payload: Mapping[str, Any], index: int = 0) -> ClaimContract:
    claim = ClaimContract.from_dict(payload, index)
    claim.validate()
    return claim


# ---------------------------------------------------------------------------
# Question parsing
# ---------------------------------------------------------------------------


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    candidates: list[str] = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)
    brace = re.search(r"\{.*\}", text, flags=re.S)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def call_llm(
    client: LLMClient,
    prompt: str,
    config: Mapping[str, Any],
    *,
    max_tokens: int | None = None,
) -> tuple[str, dict[str, int], str]:
    llm = config.get("llm", {})
    response = client.complete(
        prompt,
        system="Return only the JSON object. Do not add prose.",
        max_tokens=int(max_tokens or llm.get("max_tokens") or 3000),
        temperature=float(llm.get("temperature") or 0.0),
        thinking=False,
        use_cache=True,
    )
    text = LLMClient.text_of(response)
    usage = LLMClient.usage_of(response)
    digest = str(response.get("request_digest") or "")
    return text, usage, digest


def parse_question(
    question: str,
    *,
    config: Mapping[str, Any],
    client: LLMClient,
    company: str = "",
    ticker: str = "",
    target_period: str = "",
    today: str | None = None,
) -> dict[str, Any]:
    prompt = Template(PARSE_PROMPT.read_text(encoding="utf-8")).safe_substitute(
        today=today or date.today().isoformat(),
        question=question,
        company=company or "(not supplied)",
        ticker=ticker or "(not supplied)",
        target_period=target_period or "(not supplied)",
    )
    raw, usage, digest = call_llm(client, prompt, config, max_tokens=2200)
    parsed = parse_json_object(raw)
    if parsed is None:
        raise ValueError(f"question parser returned invalid JSON: {raw[:300]}")
    parsed["_raw_text"] = raw
    parsed["_usage"] = usage
    parsed["_request_digest"] = digest
    parsed["_prompt"] = prompt
    return parsed


def case_from_parsed(
    parsed: Mapping[str, Any],
    *,
    question: str,
    company: str = "",
    ticker: str = "",
    target_period: str = "",
    cutoff: str = "",
) -> CaseSpec:
    claims_payload = list(parsed.get("claims") or [])
    if not claims_payload:
        raise ValueError("question parser returned no claims")
    claims = [
        claim_from_payload(row, index) for index, row in enumerate(claims_payload)
    ]
    return CaseSpec(
        case_id=str(parsed.get("case_id") or "ask"),
        question=question,
        company=str(company or parsed.get("company") or "").strip(),
        ticker=str(ticker or parsed.get("ticker") or "").strip(),
        target_period=str(target_period or parsed.get("target_period") or "").strip(),
        claims=claims,
        cutoff=cutoff,
        source="llm_parser",
    )


# ---------------------------------------------------------------------------
# Evidence retrieval
# ---------------------------------------------------------------------------


def _target_years(period: str) -> tuple[int | None, int | None]:
    match = re.search(r"(20\d{2})", period or "")
    if not match:
        return None, None
    year = int(match.group(1))
    return year, year - 1


MODEL_ROLES = {
    "level_bridge": [
        "ytd_actual", "q4_forecast", "q3_revenue", "seasonal_factor",
        "remaining_forecast", "remaining_low", "remaining_high",
        "consensus", "fy_target", "prior_actual",
    ],
    "growth_bridge": [
        "ytd_actual", "q4_forecast", "q3_revenue", "seasonal_factor",
        "remaining_forecast", "remaining_low", "remaining_high",
        "consensus", "fy_target", "prior_actual",
    ],
    "margin_bridge": [
        "current_margin", "prior_margin", "price_mix_delta", "cost_delta",
    ],
    "capacity_bridge": [
        "current_capacity", "expansion", "retirement", "utilization",
    ],
    "production_bridge": [
        "capacity", "utilization", "yield", "orders", "seasonality",
    ],
    "date_window": [],
    "disclosure_check": [],
}

DEFAULT_QUERY_FACETS: list[dict[str, Any]] = [
    {
        "name": "latest_period",
        "required": True,
        "applies_to": "metric",
        "templates": [
            "{company} {year}年三季报 {metric}",
            "{company} {period} {metric}",
        ],
    },
    {
        "name": "prior_period",
        "required": True,
        "applies_to": "metric",
        "templates": ["{company} {prior_year}年 {metric}"],
    },
    {
        "name": "forward_disclosure",
        "required": False,
        "applies_to": "metric",
        "templates": [
            "{company} {year}年 业绩 预告 {metric}",
            "{company} {year}年 股权激励 考核 {metric}",
            "{company} {year}年 限制性股票 考核 {metric}",
            "{company} {year}年 投资者关系 活动记录 {metric}",
        ],
    },
    {
        "name": "industry",
        "required": False,
        "applies_to": "metric",
        "templates": ["{company} 所属行业 {year}年 {metric} 趋势"],
    },
    {
        "name": "date_disclosure",
        "required": True,
        "applies_to": "date",
        "templates": [
            "{company} 年度报告 披露日期",
            "{company} 往年 年报 披露 时间",
            "{company} {next_year}年 年报 预约披露",
        ],
    },
]


def _query_metric(metric: str) -> str:
    """Shorten a claim metric for a search query.

    A disclosure claim may have a metric such as "2026年全年营业收入指引".
    Using the full string in every template produces queries like
    "2026年三季报 2026年全年营业收入指引".  Remove the year and period words,
    keep the substantive topic, and fall back to the claim's own words.
    """

    text = re.sub(r"20\d{2}\s*年?", "", metric or "")
    for token in ("全年", "年度", "是否", "发布过", "披露过"):
        text = text.replace(token, "")
    text = re.sub(r"\s+", " ", text).strip()
    return text or "营业收入"


def facet_family(name: str) -> str:
    """Map parser-proposed facet names to the canonical evidence families."""

    text = str(name or "").strip().lower()
    if any(
        token in text
        for token in (
            "latest", "current", "reported", "report_period", "最新",
            "当期", "报告期", "本报告期",
        )
    ):
        return "latest_period"
    if any(
        token in text
        for token in (
            "prior", "previous", "prior_year", "上一年", "去年", "基期",
        )
    ):
        return "prior_period"
    if any(
        token in text
        for token in (
            "guidance", "incentive", "forecast", "consensus", "announcement",
            "disclosure", "指引", "激励", "预告", "目标", "公告", "展望",
        )
    ):
        return "forward_disclosure"
    if any(
        token in text
        for token in ("date", "披露日期", "预约", "annual_report_date")
    ):
        return "date_disclosure"
    return text or "other"


def _format_query_templates(
    templates: Sequence[str],
    values: Mapping[str, Any],
    *,
    limit: int,
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for template in templates:
        try:
            query = str(template).format(**values)
        except (KeyError, ValueError):
            continue
        if "{" in query or "}" in query:
            continue
        query = re.sub(r"\s+", " ", query).strip()
        if not query or query in seen:
            continue
        seen.add(query)
        output.append(query)
        if len(output) >= limit:
            break
    return output


def build_query_plan(
    claim: ClaimContract,
    spec: CaseSpec,
    *,
    max_queries: int = 8,
    max_parser_queries: int = 2,
    max_per_facet: int = 2,
    facets: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a facet-based query portfolio.

    The portfolio reserves at least one query for every required facet, then
    spends the remaining budget on parser queries and optional facets.  This
    replaces the flat "auto queries plus parser queries, then truncate" rule,
    which could drop the only latest-period query when the parser supplied
    several generic queries.
    """

    company = spec.company
    period = claim.target_period or spec.target_period
    year, prior_year = _target_years(period)
    values = {
        "company": company,
        "year": year or "",
        "prior_year": prior_year or "",
        "next_year": (year + 1) if year else "",
        "metric": _query_metric(claim.metric),
        "period": period,
    }
    base_facets = list(facets or DEFAULT_QUERY_FACETS)
    parser_facets: list[dict[str, Any]] = []
    for row in claim.query_facets:
        name = str(row.get("name") or "").strip()
        queries = [
            str(item).strip()
            for item in (row.get("queries") or row.get("templates") or [])
            if str(item).strip()
        ]
        if not name or not queries:
            continue
        parser_facets.append(
            {
                "name": f"claim_{name}",
                "required": bool(row.get("required", False)),
                "applies_to": claim.claim_type,
                "templates": queries,
            }
        )
    # Default required facets first, then claim-specific facets, then default
    # optional facets.  This keeps the generic latest/prior/date guarantees
    # while letting the parser add domain channels such as capacity or orders.
    facet_defs = (
        [facet for facet in base_facets if facet.get("required")]
        + parser_facets
        + [facet for facet in base_facets if not facet.get("required")]
    )
    candidates: list[dict[str, Any]] = []
    for facet in facet_defs:
        applies_to = str(facet.get("applies_to") or "all")
        if applies_to not in {"all", claim.claim_type}:
            continue
        templates = [
            str(template)
            for template in (facet.get("templates") or [])
            if str(template).strip()
        ]
        for query in _format_query_templates(templates, values, limit=max_per_facet):
            candidates.append(
                {
                    "facet": str(facet.get("name") or "other"),
                    "query": query,
                    "required": bool(facet.get("required")),
                }
            )
    parser_candidates = [
        {"facet": "parser", "query": re.sub(r"\s+", " ", str(query)).strip(), "required": False}
        for query in list(claim.queries[: max(0, max_parser_queries)])
        if str(query).strip()
    ]

    selected: list[dict[str, Any]] = []
    used_queries: set[str] = set()
    used_per_facet: dict[str, int] = defaultdict(int)

    def add(entry: Mapping[str, Any]) -> bool:
        query = str(entry.get("query") or "").strip()
        facet = str(entry.get("facet") or "other")
        if not query or query in used_queries:
            return False
        if used_per_facet[facet] >= max_per_facet:
            return False
        if len(selected) >= max_queries:
            return False
        selected.append(
            {
                "facet": facet,
                "query": query,
                "required": bool(entry.get("required")),
            }
        )
        used_queries.add(query)
        used_per_facet[facet] += 1
        return True

    # One query for every required facet first.
    required_facets: list[str] = []
    for entry in candidates:
        if not entry["required"]:
            continue
        if entry["facet"] not in required_facets:
            required_facets.append(entry["facet"])
    for facet in required_facets:
        for entry in candidates:
            if entry["facet"] == facet:
                add(entry)
                break

    # Then the remaining required-facet queries.
    for entry in candidates:
        if entry["required"]:
            add(entry)

    # Parser queries are specific to the claim; place them before optional
    # exploratory facets.
    for entry in parser_candidates:
        add(entry)

    # Finally the optional facets.
    for entry in candidates:
        if not entry["required"]:
            add(entry)

    return selected[:max_queries]


def build_queries(
    claim: ClaimContract,
    spec: CaseSpec,
    *,
    max_queries: int = 8,
    max_parser_queries: int = 2,
    max_per_facet: int = 2,
    facets: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    return [
        str(entry["query"])
        for entry in build_query_plan(
            claim,
            spec,
            max_queries=max_queries,
            max_parser_queries=max_parser_queries,
            max_per_facet=max_per_facet,
            facets=facets,
        )
    ]


def compose_evidence_text(
    text: str,
    claim: ClaimContract,
    spec: CaseSpec,
    *,
    max_chars: int = 2200,
) -> str:
    """Select the sentences most likely to contain the claim's evidence.

    A long announcement often begins with legal boilerplate and puts the
    assessment target deep in the document.  Taking the first N characters
    would hide the number.  This routine keeps the head plus the highest-
    scoring sentences that mention the metric, the company, a financial term,
    or a number with a unit.
    """

    text = re.sub(r"\s+", " ", text or "").strip()
    if not text or len(text) <= max_chars:
        return text
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？；;.!?])\s*", text)
        if len(sentence.strip()) >= 8
    ]
    metric_terms = [
        term
        for term in re.findall(r"[一-鿿]{2,}", claim.metric or "")
        if len(term) >= 2
    ]
    finance_terms = [
        "营业收入", "营收", "净利润", "归母净利润", "毛利率", "指引",
        "考核", "激励", "目标", "同比", "环比", "增长", "不低于",
        "不高于", "亿元",
    ]
    keywords = list(dict.fromkeys(metric_terms + finance_terms + [spec.company]))
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        score = 0
        for keyword in keywords:
            if keyword and keyword in sentence:
                score += 2 if keyword in metric_terms or keyword in finance_terms else 1
        if re.search(r"\d+(?:\.\d+)?\s*(?:亿元|亿|%|元)", sentence):
            score += 2
        if score:
            scored.append((score, index, sentence))
    if not scored:
        return text[:max_chars]
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[tuple[int, str]] = []
    used = 0
    for _, index, sentence in scored:
        if used + len(sentence) > max_chars:
            continue
        chosen.append((index, sentence))
        used += len(sentence)
        if used >= max_chars * 0.8:
            break
    chosen.sort(key=lambda item: item[0])
    excerpt = " ".join(sentence for _, sentence in chosen)
    if len(excerpt) < max_chars // 3:
        excerpt = text[:max_chars]
    return excerpt[:max_chars]


def canonical_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except Exception:
        return url
    # Redirect endpoints carry the destination in the query string.  Dropping
    # the query would collapse every 360/Sogou/Baidu result to one URL and
    # silently delete all but the first evidence item.
    redirect_hosts = ("baidu.com", "so.com", "sogou.com")
    if any(host in parts.netloc for host in redirect_hosts) and "/link" in parts.path:
        return url
    # Baijiahao content pages carry the article id in the query string.
    if "baijiahao.baidu.com" in parts.netloc:
        return url
    return f"{parts.scheme}://{parts.netloc}{parts.path}".rstrip("/")


def collect_evidence(
    claim: ClaimContract,
    spec: CaseSpec,
    config: Mapping[str, Any],
    cutoff: date,
    *,
    query_plan: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    search_cfg = config.get("search", {})
    max_queries = int(search_cfg.get("max_queries") or 8)
    max_parser_queries = int(search_cfg.get("max_parser_queries") or 2)
    max_per_facet = int(search_cfg.get("max_per_facet") or 2)
    top_k = int(search_cfg.get("top_k") or 5)
    fetch_top = int(search_cfg.get("fetch_top") or 1)
    max_evidence = int(search_cfg.get("max_evidence") or 8)
    exclude_patterns = [
        str(pattern).lower()
        for pattern in (search_cfg.get("exclude_url_patterns") or [])
        if str(pattern).strip()
    ]
    plan = list(query_plan) if query_plan is not None else build_query_plan(
        claim,
        spec,
        max_queries=max_queries,
        max_parser_queries=max_parser_queries,
        max_per_facet=max_per_facet,
        facets=search_cfg.get("query_facets"),
    )
    queries = [str(entry.get("query") or "") for entry in plan]

    raw_items: list[dict[str, Any]] = []
    search_log: list[dict[str, Any]] = []
    cninfo_cfg = (config.get("sources") or {}).get("cninfo") or {}
    if claim.claim_type == "date" and cninfo_cfg.get("enabled", True):
        official = annual_report_evidence(spec.company, spec.ticker, cutoff)
        if official:
            raw_items.append(official)
            search_log.append(
                {
                    "query": official.get("query"),
                    "ok": True,
                    "n_results": 1,
                    "provider": "cninfo",
                    "seconds": 0.0,
                }
            )
    if claim.claim_type == "metric" and cninfo_cfg.get("enabled", True):
        target_year, _ = _target_years(claim.target_period or spec.target_period)
        official_items = metric_announcement_evidence(
            spec.company,
            spec.ticker,
            cutoff,
            year=target_year,
            max_items=int(cninfo_cfg.get("metric_max_items") or 2),
        )
        for official in official_items:
            raw_items.append(official)
            search_log.append(
                {
                    "query": official.get("query"),
                    "ok": True,
                    "n_results": 1,
                    "provider": "cninfo",
                    "seconds": 0.0,
                }
            )
    for query_index, entry in enumerate(plan):
        query = str(entry.get("query") or "")
        facet = str(entry.get("facet") or "other")
        started = time.time()
        try:
            items = gather(query, top_k=top_k, fetch_top=fetch_top)
            search_log.append(
                {
                    "query": query,
                    "facet": facet,
                    "ok": True,
                    "n_results": len(items),
                    "seconds": round(time.time() - started, 2),
                }
            )
        except Exception as error:  # noqa: BLE001
            search_log.append(
                {
                    "query": query,
                    "facet": facet,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "seconds": round(time.time() - started, 2),
                }
            )
            continue
        for item in items:
            row = dict(item)
            row["query_index"] = query_index
            row["facet"] = facet
            raw_items.append(row)

    excluded: list[dict[str, Any]] = []
    deduped: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for item in raw_items:
        url = str(item.get("url") or "")
        title = str(item.get("title") or "").strip()
        if any(pattern in url.lower() for pattern in exclude_patterns):
            excluded.append(
                {
                    "title": title,
                    "url": url,
                    "date": item.get("date"),
                    "reason": "non-documentary source",
                }
            )
            continue
        key_url = canonical_url(url)
        key_title = re.sub(r"\s+", "", title)
        if key_url in seen_urls or (key_title and key_title in seen_titles):
            continue
        seen_urls.add(key_url)
        if key_title:
            seen_titles.add(key_title)
        published = parse_date(item.get("date"))
        if published is not None and published > cutoff:
            excluded.append(
                {
                    "title": title,
                    "url": url,
                    "date": published.isoformat(),
                    "reason": "published after cutoff",
                }
            )
            continue
        text = str(item.get("text") or item.get("snippet") or "")
        snippet = str(item.get("snippet") or "")
        prompt_text = compose_evidence_text(
            text,
            claim,
            spec,
            max_chars=int(search_cfg.get("max_chars_per_item") or 2200),
        )
        score = 0.0
        if item.get("official"):
            score += 20.0
        if published is not None:
            score += 3.0
        if re.search(r"\d", snippet or text):
            score += 1.0
        if spec.company and (spec.company in title or spec.company in snippet or spec.company in text):
            score += 1.0
        if len(text) > 500:
            score += 0.5
        if claim.metric and claim.metric in (snippet + text):
            score += 0.5
        score -= 0.05 * int(item.get("query_index") or 0)
        deduped.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "text": text,
                "prompt_text": prompt_text,
                "date": published.isoformat() if published else None,
                "date_known": published is not None,
                "retrieved_at": date.today().isoformat(),
                "query": item.get("query") or "",
                "query_index": int(item.get("query_index") or 0),
                "facet": item.get("facet") or "other",
                "provider": item.get("provider"),
                "official": bool(item.get("official")),
                "source_type": item.get("source_type"),
                "_score": score,
            }
        )

    # Round-robin across queries so that one query with five near-duplicate
    # articles cannot crowd out a specific disclosure channel.  Official
    # evidence (query_index -1) is always placed first.
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in deduped:
        grouped[item["query_index"]].append(item)
    for group in grouped.values():
        group.sort(key=lambda row: (-row["_score"], row["title"]))
    selected: list[dict[str, Any]] = []
    selected.extend(grouped.pop(-1, []))
    while len(selected) < max_evidence:
        added = False
        for key in sorted(grouped):
            if not grouped[key]:
                continue
            selected.append(grouped[key].pop(0))
            added = True
            if len(selected) >= max_evidence:
                break
        if not added:
            break
    if len(selected) < max_evidence:
        remaining = [item for group in grouped.values() for item in group]
        remaining.sort(key=lambda row: (-row["_score"], row["title"]))
        selected.extend(remaining[: max_evidence - len(selected)])
    selected = selected[:max_evidence]
    for index, item in enumerate(selected, 1):
        item["source_id"] = f"W{index}"
        item.pop("_score", None)
        item["text"] = item["text"][:6000]
    return selected, excluded, search_log


# ---------------------------------------------------------------------------
# Anchor extraction
# ---------------------------------------------------------------------------


def label_block(claim: ClaimContract) -> str:
    lines = []
    for label in claim.labels:
        definition = claim.label_defs.get(label, "")
        lines.append(f"- {label}: {definition}" if definition else f"- {label}")
    return "\n".join(lines)


def evidence_block(evidence: Sequence[Mapping[str, Any]], *, per_item: int = 1800) -> str:
    if not evidence:
        return "no evidence was retrieved"
    lines = []
    for item in evidence:
        body = str(
            item.get("prompt_text") or item.get("text") or item.get("snippet") or ""
        ).strip()
        body = re.sub(r"\s+", " ", body)[:per_item]
        date_label = item.get("date") or "unknown"
        lines.append(
            f"[{item['source_id']}] {item.get('title') or ''} | 日期: {date_label} | "
            f"{item.get('url') or ''}\n{body}"
        )
    return "\n\n".join(lines)


def build_anchor_prompt(
    claim: ClaimContract,
    spec: CaseSpec,
    evidence: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    cutoff: str,
) -> str:
    anchor_cfg = config.get("anchors", {})
    values = {
        "claim_id": claim.claim_id,
        "claim": claim.text,
        "company": spec.company,
        "ticker": spec.ticker or "",
        "target_period": claim.target_period or spec.target_period,
        "metric": claim.metric,
        "metric_kind": claim.metric_kind or "level",
        "claim_type": claim.claim_type,
        "cutoff": cutoff,
        "reference": claim.reference or "not specified",
        "unit": claim.unit or "not specified",
        "model_template": claim.model_template or "level_bridge",
        "model_roles": ", ".join(
            MODEL_ROLES.get(claim.model_template or "level_bridge", [])
        ) or "none",
        "anchor_types": ", ".join(claim.anchor_types) or "not specified",
        "label_block": label_block(claim),
        "evidence": evidence_block(
            evidence,
            per_item=int(config.get("search", {}).get("max_chars_per_item") or 1800),
        ),
    }
    template = Template(ANCHOR_PROMPT.read_text(encoding="utf-8"))
    return template.safe_substitute(values)


def extract_anchor_payload(
    claim: ClaimContract,
    spec: CaseSpec,
    evidence: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    cutoff: str,
    client: LLMClient,
) -> dict[str, Any]:
    prompt = build_anchor_prompt(claim, spec, evidence, config, cutoff)
    raw, usage, digest = call_llm(client, prompt, config)
    parsed = parse_json_object(raw)
    return {
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "raw_text": raw,
        "parsed": parsed,
        "usage": usage,
        "request_digest": digest,
        "error": None if parsed is not None else "json_parse_failed",
    }


# ---------------------------------------------------------------------------
# Anchor qualification
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return re.sub(r"\s+", "", (text or "")).lower()


def _match_label(value: str, labels: Sequence[str]) -> str | None:
    value = str(value or "").strip()
    if not value:
        return None
    if value in labels:
        return value
    for label in labels:
        if len(label) >= 2 and label in value:
            return label
        if len(value) >= 2 and value in label:
            return label
    return None


def _date_label_from_text(text: str, claim: ClaimContract) -> str | None:
    """Map a mentioned calendar date to a target-year window.

    Disclosure-history anchors often mention the prior year's date.  The
    month and day are the transferable part of the pattern, so the date is
    projected onto the target year and matched against the claim windows.
    The function returns a label only when every mentioned date agrees; a
    mixture of dates is left to the model.
    """

    if claim.claim_type != "date" or not claim.date_windows:
        return None
    first_start = parse_date(claim.date_windows[0].get("start"))
    if first_start is None:
        return None
    target_year = first_start.year
    windows = []
    for row in claim.date_windows:
        start, end = parse_date(row.get("start")), parse_date(row.get("end"))
        if start and end:
            windows.append((row["label"], start, end))
    labels: set[str] = set()
    for match in re.finditer(r"(20\d{2})\s*[-年/]\s*(\d{1,2})\s*[-月/]\s*(\d{1,2})", text or ""):
        month, day = int(match.group(2)), int(match.group(3))
        try:
            projected = date(target_year, month, day)
        except ValueError:
            continue
        for label, start, end in windows:
            if start <= projected <= end:
                labels.add(label)
                break
    if len(labels) == 1:
        return labels.pop()
    return None


def qualify_anchors(
    parsed: Mapping[str, Any] | None,
    claim: ClaimContract,
    evidence: Sequence[Mapping[str, Any]],
    cutoff: date,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not parsed:
        return []
    anchor_cfg = config.get("anchors", {})
    max_anchors = int(anchor_cfg.get("max_anchors") or 12)
    hard_threshold = float(anchor_cfg.get("hard_reliability") or 0.80)
    undated_cap = float(anchor_cfg.get("undated_reliability_cap") or 0.55)
    quote_factor = float(anchor_cfg.get("unverified_quote_factor") or 0.70)
    by_id = {item["source_id"]: item for item in evidence}
    output: list[dict[str, Any]] = []

    for raw_anchor in (parsed.get("anchors") or [])[:max_anchors]:
        if not isinstance(raw_anchor, Mapping):
            continue
        anchor = dict(raw_anchor)
        source_id = str(anchor.get("source_id") or "").strip().upper()
        anchor["source_id"] = source_id
        implied_raw = str(anchor.get("implied_label") or "").strip()
        implied = _match_label(implied_raw, claim.labels)
        anchor["implied_label_matched"] = implied
        anchor["implied_label_raw"] = implied_raw
        reasons: list[str] = []
        status = "accepted"
        qualification = "soft"

        if implied is None:
            status = "rejected"
            qualification = "vacuous"
            reasons.append("no valid implied outcome label")
        source = by_id.get(source_id)
        if source is None:
            status = "rejected"
            qualification = "adversarial"
            reasons.append("source_id is not in the retrieved evidence set")
            source = {}

        anchor_date = parse_date(anchor.get("date"))
        evidence_date = parse_date(source.get("date"))
        if anchor_date is not None and anchor_date > cutoff:
            status = "rejected"
            qualification = "stale"
            reasons.append("anchor date is after the cutoff")
        if evidence_date is not None and evidence_date > cutoff:
            status = "rejected"
            qualification = "stale"
            reasons.append("source date is after the cutoff")

        quote = str(anchor.get("quote") or "").strip()
        source_text = "\n".join(
            str(source.get(key) or "")
            for key in ("text", "prompt_text", "snippet")
        )
        quote_verified = len(_normalise(quote)) >= 6 and _normalise(quote) in _normalise(source_text)
        anchor["quote"] = quote
        anchor["quote_verified"] = quote_verified
        if not quote_verified:
            reasons.append("quote was not found verbatim in the cited source")
        if not quote:
            reasons.append("quote is empty")

        numeric_value = None
        numeric_low = None
        numeric_high = None
        for key, target in (
            ("numeric_value", "numeric_value"),
            ("numeric_low", "numeric_low"),
            ("numeric_high", "numeric_high"),
        ):
            raw = anchor.get(key)
            if raw is None or str(raw).strip() == "":
                anchor[target] = None
                continue
            try:
                anchor[target] = float(raw)
            except (TypeError, ValueError):
                anchor[target] = None
        numeric_value = anchor.get("numeric_value")
        numeric_low = anchor.get("numeric_low")
        numeric_high = anchor.get("numeric_high")
        numeric_verified = False
        if numeric_value is not None:
            numeric_quote = _normalise(quote).replace(",", "").replace("，", "")
            quote_numbers: list[float] = []
            for token in re.findall(r"\d[\d,]*(?:\.\d+)?", quote or ""):
                try:
                    quote_numbers.append(float(token.replace(",", "")))
                except ValueError:
                    continue
            for candidate in (numeric_value, numeric_low, numeric_high):
                if candidate is None:
                    continue
                for precision in (2, 4, 6):
                    raw_candidate = (
                        f"{float(candidate):.{precision}f}".rstrip("0").rstrip(".")
                    )
                    if raw_candidate in numeric_quote:
                        numeric_verified = True
                        break
                if numeric_verified:
                    break
                # The LLM may report 695.47 亿元 while the quote prints
                # 69,547,057,722 元.  Check scaled magnitudes with tolerance.
                for scale in (1.0, 1e4, 1e8, 1e-4, 1e-8):
                    target = float(candidate) * scale
                    if any(
                        abs(target - number) / max(abs(number), 1.0) < 1e-3
                        for number in quote_numbers
                    ):
                        numeric_verified = True
                        break
                if numeric_verified:
                    break
        numeric_unit = str(anchor.get("numeric_unit") or "").strip()
        if numeric_verified and "亿" not in numeric_unit:
            factor = None
            if "万元" in numeric_unit and abs(float(numeric_value)) >= 1e4:
                factor = 1e4
            elif ("元" in numeric_unit or numeric_unit == "") and abs(float(numeric_value)) >= 1e8:
                factor = 1e8
            if factor is not None:
                for key in ("numeric_value", "numeric_low", "numeric_high"):
                    if anchor.get(key) is not None:
                        anchor[key] = round(float(anchor[key]) / factor, 6)
                anchor["numeric_unit"] = "亿元"
                anchor["numeric_unit_normalized_from"] = numeric_unit or "元"
                numeric_value = anchor.get("numeric_value")
                numeric_low = anchor.get("numeric_low")
                numeric_high = anchor.get("numeric_high")
        anchor["numeric_verified"] = numeric_verified
        model_role = str(anchor.get("model_role") or "").strip().lower()
        numeric_period = str(anchor.get("numeric_period") or "").strip()
        target_year, prior_year = _target_years(claim.target_period or "")
        role_valid = True
        if model_role and target_year:
            current_roles = {
                "ytd_actual", "current_margin", "current_capacity",
                "capacity", "utilization", "yield", "q3_revenue",
            }
            prior_roles = {"prior_actual", "prior_margin"}
            if model_role in current_roles:
                if str(prior_year) in numeric_period and str(target_year) not in numeric_period:
                    role_valid = False
            elif model_role in prior_roles:
                if str(target_year) in numeric_period and str(prior_year) not in numeric_period:
                    role_valid = False
            elif model_role in {"consensus", "fy_target", "target"}:
                if str(prior_year) in numeric_period and str(target_year) not in numeric_period:
                    role_valid = False
        anchor["model_role_valid"] = role_valid
        if not role_valid:
            reasons.append("model_role conflicts with numeric_period")

        anchor_type = str(anchor.get("type") or "other").strip().lower()
        anchor["type"] = anchor_type
        if claim.claim_type == "date" and anchor_type == "regulation":
            status = "rejected"
            qualification = "vacuous"
            reasons.append(
                "a generic regulatory deadline does not discriminate among the date windows"
            )
        target_year, prior_year = _target_years(claim.target_period or "")
        if (
            status == "accepted"
            and claim.claim_type == "metric"
            and target_year
            and prior_year
            and ("增速" in claim.metric or "同比" in claim.text)
        ):
            text_for_year = f"{anchor.get('value') or ''} {quote}"
            has_target = bool(re.search(rf"{target_year}\s*年", text_for_year))
            has_prior = bool(re.search(rf"{prior_year}\s*年", text_for_year))
            if has_prior and not has_target:
                status = "rejected"
                qualification = "vacuous"
                reasons.append(
                    "anchor describes the prior-year reference value, not the target-period outcome"
                )
        if (
            status == "accepted"
            and claim.claim_type == "metric"
            and ("指引" in claim.text or "guidance" in claim.text.lower())
        ):
            text_check = f"{anchor.get('value') or ''} {quote}"
            negative_intent = any(
                token in (implied or "") for token in ("未发布", "未明确", "未披露")
            )
            middle_target = any(
                token in (implied or "") for token in ("存在", "目标", "考核", "激励", "预告", "展望")
            )
            if negative_intent and not middle_target and not any(
                token in text_check
                for token in ("未发布", "未披露", "未明确", "没有发布", "无指引")
            ):
                status = "rejected"
                qualification = "vacuous"
                reasons.append(
                    "a negative disclosure label needs an explicit statement of "
                    "absence, not the absence of a guidance in a quarterly result"
                )
            result_tokens = (
                "第一季度", "第二季度", "第三季度", "前三季度", "半年度",
                "上半年", "一季度", "中报", "三季报",
            )
            outlook_tokens = (
                "指引", "展望", "目标", "考核", "不低于", "不高于",
                "预计", "预测", "订单", "需求",
            )
            if (
                status == "accepted"
                and any(token in text_check for token in result_tokens)
                and not any(token in text_check for token in outlook_tokens)
            ):
                status = "rejected"
                qualification = "vacuous"
                reasons.append(
                    "a periodic result is not evidence about whether a "
                    "forward-looking guidance was issued"
                )
            operational_tokens = ("产能", "产量", "利用率", "出货量", "排产")
            target_tokens = ("指引", "目标", "考核", "预告", "展望")
            if (
                status == "accepted"
                and any(token in text_check for token in operational_tokens)
                and not any(token in text_check for token in target_tokens)
            ):
                status = "rejected"
                qualification = "vacuous"
                reasons.append(
                    "a capacity or production number is not evidence about "
                    "whether a guidance or target was disclosed"
                )
            if (
                status == "accepted"
                and "收入" in claim.text
                and any(token in text_check for token in operational_tokens)
                and not any(
                    token in text_check
                    for token in ("营业收入", "营收", "收入")
                )
            ):
                status = "rejected"
                qualification = "vacuous"
                reasons.append(
                    "a capacity target is not a revenue target"
                )
        if status == "accepted" and claim.claim_type == "date":
            mapped = _date_label_from_text(
                f"{anchor.get('value') or ''} {quote}", claim
            )
            if mapped and mapped != implied:
                anchor["label_corrected_from"] = implied
                implied = mapped
                anchor["implied_label_matched"] = mapped
                reasons.append(
                    f"implied label corrected to {mapped} from the date mentioned in the anchor"
                )

        scope = str(anchor.get("scope") or "company").strip().lower()
        anchor["scope"] = scope
        if scope not in ALLOWED_SCOPES:
            status = "rejected"
            qualification = "adversarial"
            reasons.append("scope does not match an allowed claim scope")

        reliability = float(anchor.get("reliability", 0.5) or 0.5)
        reliability = min(1.0, max(0.0, reliability))
        if not quote_verified:
            reliability *= quote_factor
        if numeric_value is not None and not numeric_verified:
            reliability *= 0.7
            reasons.append("numeric_value was not found in the quote")
        if not anchor.get("model_role_valid", True):
            reliability *= 0.8
        undated = anchor_date is None and evidence_date is None
        if undated:
            reliability = min(reliability, undated_cap)
            reasons.append("publication date is unknown; reliability is capped")
        if anchor_date is not None and evidence_date is not None and anchor_date != evidence_date:
            reliability *= 0.9
            reasons.append("anchor date differs from the source date")
        temporal_inconsistency = False
        if claim.claim_type == "date" and evidence_date is not None and quote:
            quote_dates = [
                parsed
                for parsed in (parse_date(match.group(0)) for match in DATE_RE.finditer(quote))
                if parsed is not None
            ]
            later_dates = [
                parsed for parsed in quote_dates if evidence_date < parsed <= cutoff
            ]
            if later_dates:
                temporal_inconsistency = True
                reliability *= 0.5
                reasons.append(
                    "quote mentions a date after the source publication date; "
                    "point-in-time consistency is uncertain"
                )
        anchor["temporal_inconsistency"] = temporal_inconsistency
        anchor["reliability_effective"] = round(reliability, 4)

        is_hard = (
            bool(anchor.get("hard"))
            and reliability >= hard_threshold
            and quote_verified
            and not undated
            and not temporal_inconsistency
            and evidence_date is not None
        )
        if status == "accepted":
            qualification = "hard" if is_hard else "soft"
        anchor["hard_effective"] = is_hard
        anchor["qualification"] = qualification
        anchor["status"] = status
        anchor["reasons"] = reasons
        anchor["evidence_title"] = source.get("title")
        anchor["evidence_url"] = source.get("url")
        anchor["evidence_date"] = source.get("date")
        output.append(anchor)
    return output


# ---------------------------------------------------------------------------
# Belief update and full-action readout
# ---------------------------------------------------------------------------


def normalise_scores(
    scores: Mapping[str, Any] | None, labels: Sequence[str]
) -> list[float]:
    values = []
    for label in labels:
        try:
            value = float((scores or {}).get(label, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        values.append(max(0.0, value))
    total = sum(values)
    if total <= 1e-12:
        return [1.0 / len(labels)] * len(labels)
    return [value / total for value in values]


def prior_belief(claim: ClaimContract) -> list[float]:
    if claim.prior:
        return normalise_scores(claim.prior, claim.labels)
    return [1.0 / len(claim.labels)] * len(claim.labels)


def update_belief(
    claim: ClaimContract,
    anchors: Sequence[Mapping[str, Any]],
    parsed: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> tuple[list[float], list[float], list[dict[str, Any]], list[str]]:
    anchor_cfg = config.get("anchors", {})
    p = float(anchor_cfg.get("soft_likelihood") or 0.70)
    gamma = float(anchor_cfg.get("llm_score_weight") or 0.50)
    labels = claim.labels
    n = len(labels)
    belief = prior_belief(claim)
    trace: list[dict[str, Any]] = []

    for index, anchor in enumerate(anchors):
        if anchor.get("status") != "accepted":
            continue
        label = anchor.get("implied_label_matched")
        if label not in labels:
            continue
        idx = labels.index(label)
        before = list(belief)
        rho = float(anchor.get("reliability_effective") or 0.5)
        if anchor.get("qualification") == "hard":
            belief = [0.02 / n] * n
            belief[idx] = 0.98
        else:
            likelihood = [(1.0 - p) / (n - 1)] * n if n > 1 else [1.0]
            likelihood[idx] = p
            belief = [
                belief[i] * (max(likelihood[i], 1e-9) ** max(rho, 0.0))
                for i in range(n)
            ]
        total = sum(belief)
        belief = [value / total for value in belief] if total > 0 else prior_belief(claim)
        trace.append(
            {
                "anchor_index": index,
                "implied_label": label,
                "qualification": anchor.get("qualification"),
                "reliability_effective": rho,
                "before": {labels[i]: round(before[i], 6) for i in range(n)},
                "after": {labels[i]: round(belief[i], 6) for i in range(n)},
            }
        )

    belief_anchor = list(belief)
    scores = normalise_scores((parsed or {}).get("label_scores"), labels)
    final = [gamma * belief_anchor[i] + (1.0 - gamma) * scores[i] for i in range(n)]
    total = sum(final)
    final = [value / total for value in final] if total > 0 else prior_belief(claim)

    # A hard anchor is a constraint, not just a high probability.  Applying it
    # only before the LLM-score blend lets the blend dilute the constraint.
    # Project the blended posterior onto the hard-anchor feasible set here.
    hard_labels = sorted(
        {
            str(anchor.get("implied_label_matched"))
            for anchor in anchors
            if anchor.get("status") == "accepted"
            and anchor.get("qualification") == "hard"
            and anchor.get("implied_label_matched") in labels
        }
    )
    projection = []
    if hard_labels and len(hard_labels) < n:
        before_projection = list(final)
        allowed_total = sum(final[labels.index(label)] for label in hard_labels)
        if len(hard_labels) == 1:
            final = [0.02 / (n - 1) if label != hard_labels[0] else 0.98 for label in labels]
        else:
            final = [
                0.98 * final[labels.index(label)] / allowed_total
                if label in hard_labels and allowed_total > 0
                else 0.02 / (n - len(hard_labels))
                for label in labels
            ]
        total = sum(final)
        final = [value / total for value in final] if total > 0 else prior_belief(claim)
        projection = hard_labels
        trace.append(
            {
                "anchor_index": None,
                "implied_label": ", ".join(hard_labels),
                "qualification": "hard_projection",
                "reliability_effective": 1.0,
                "before": {labels[i]: round(before_projection[i], 6) for i in range(n)},
                "after": {labels[i]: round(final[i], 6) for i in range(n)},
            }
        )
    return belief_anchor, final, trace, projection


# ---------------------------------------------------------------------------
# Historical calibration
# ---------------------------------------------------------------------------

_CALIBRATION_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
NUMERIC_METRIC_KINDS = {"level", "growth", "margin", "capacity", "production"}
CALIBRATION_LABELS = ["below", "within", "above"]


def load_calibration(config: Mapping[str, Any]) -> dict[str, Any] | None:
    cal_cfg = config.get("calibration") or {}
    if not cal_cfg.get("enabled", False):
        return None
    path = REPO_ROOT / str(
        cal_cfg.get("bundle_path")
        or "paper/results/anchor_cli/calibration/bundle.json"
    )
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
        cached = _CALIBRATION_CACHE.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
        bundle = json.loads(path.read_text(encoding="utf-8"))
        bundle["_path"] = str(path)
        _CALIBRATION_CACHE[str(path)] = (mtime, bundle)
        return bundle
    except Exception:
        return None


def calibration_applies(claim: ClaimContract, bundle: Mapping[str, Any] | None) -> bool:
    if not bundle or len(claim.labels) != 3:
        return False
    kind = claim.metric_kind or "level"
    if kind not in NUMERIC_METRIC_KINDS:
        return False
    allowed = bundle.get("applies_to_metric_kinds") or ["level", "growth"]
    return kind in {str(item) for item in allowed}


def ordered_label_indices(claim: ClaimContract) -> list[int] | None:
    """Return the claim's label indices in low-to-high order."""

    if len(claim.labels) != 3:
        return None
    if claim.thresholds:
        def key(index: int) -> tuple[float, float, float]:
            label = claim.labels[index]
            row = next(
                (item for item in claim.thresholds if item.get("label") == label),
                {},
            )
            low = row.get("low")
            high = row.get("high")
            return (
                0.0 if low is not None else 1.0,
                float(low) if low is not None else -1e18,
                float(high) if high is not None else 1e18,
            )

        order = sorted(range(3), key=key)
        if len(set(order)) == 3:
            return order
    if "高于" in claim.labels[0] and ("低于" in claim.labels[-1] or "低" in claim.labels[-1]):
        return [2, 1, 0]
    return [0, 1, 2]


def calibration_prior(
    claim: ClaimContract, bundle: Mapping[str, Any]
) -> dict[str, float]:
    order = ordered_label_indices(claim)
    if order is None:
        return {}
    prior = bundle.get("prior") or {}
    values = [float(prior.get(label, 1.0 / 3.0)) for label in CALIBRATION_LABELS]
    return {
        claim.labels[index]: values[position]
        for position, index in enumerate(order)
    }


def apply_type_reliability(
    anchors: Sequence[dict[str, Any]],
    bundle: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    cal_cfg = config.get("calibration") or {}
    shrink = float(cal_cfg.get("type_shrinkage") or 0.5)
    mapping = bundle.get("type_mapping") or {}
    reliabilities = bundle.get("type_reliability") or {}
    for anchor in anchors:
        if anchor.get("status") != "accepted":
            continue
        if anchor.get("qualification") == "hard":
            continue
        raw_type = str(anchor.get("type") or "other").strip().lower()
        canonical = str(mapping.get(raw_type) or raw_type)
        entry = reliabilities.get(canonical)
        if not entry:
            continue
        old = float(anchor.get("reliability_effective") or 0.5)
        learned = float(entry.get("reliability") or 0.5)
        updated = (1.0 - shrink) * old + shrink * learned
        anchor["type_reliability"] = learned
        anchor["type_reliability_n"] = int(entry.get("n") or 0)
        anchor["calibration_shrink"] = shrink
        anchor["reliability_effective"] = round(min(1.0, max(0.0, updated)), 4)


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values) if values else 0.0
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    if total <= 1e-12:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exponentials]


def _calibration_features(
    claim: ClaimContract,
    anchors: Sequence[Mapping[str, Any]],
    belief_anchor: Sequence[float],
    parsed: Mapping[str, Any] | None,
) -> tuple[list[float], list[int]] | None:
    order = ordered_label_indices(claim)
    if order is None:
        return None
    belief = [float(value) for value in belief_anchor]
    scores = normalise_scores((parsed or {}).get("label_scores"), claim.labels)
    ordered_belief = [belief[index] for index in order]
    ordered_scores = [scores[index] for index in order]
    entropy = -sum(value * math.log(value) for value in ordered_belief if value > 0)
    ranked = sorted(ordered_belief, reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else 0.0
    accepted = [
        anchor
        for anchor in anchors
        if anchor.get("status") == "accepted"
        and anchor.get("type") != "model_bridge"
    ]
    reliabilities = [
        float(anchor.get("reliability_effective") or 0.5) for anchor in accepted
    ]
    features = [
        ordered_belief[0], ordered_belief[1], ordered_belief[2],
        ordered_scores[0], ordered_scores[1], ordered_scores[2],
        entropy, margin,
        min(12.0, len(accepted)) / 12.0,
        min(12.0, sum(1 for anchor in accepted if anchor.get("qualification") == "hard")) / 12.0,
        sum(reliabilities) / len(reliabilities) if reliabilities else 0.5,
        max(reliabilities) if reliabilities else 0.5,
    ]
    return features, order


def apply_calibration(
    claim: ClaimContract,
    anchors: Sequence[Mapping[str, Any]],
    belief_anchor: Sequence[float],
    parsed: Mapping[str, Any] | None,
    bundle: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not calibration_applies(claim, bundle):
        return None
    if not bundle.get("calibrator_enabled", True):
        return {
            "applied": False,
            "reason": "calibrator disabled by bundle",
            "bundle": bundle.get("_path"),
        }
    built = _calibration_features(claim, anchors, belief_anchor, parsed)
    if built is None:
        return None
    features, order = built
    mean = bundle.get("mean") or []
    std = bundle.get("std") or []
    coef = bundle.get("coef") or []
    if len(mean) != len(features) or len(std) != len(features) or not coef:
        return None
    normalized = [
        (features[index] - float(mean[index])) / max(float(std[index]), 1e-9)
        for index in range(len(features))
    ]
    logits = []
    for class_index in range(3):
        value = float(coef[-1][class_index])
        for feature_index, feature in enumerate(normalized):
            value += feature * float(coef[feature_index][class_index])
        logits.append(value)
    temperature = max(0.05, float(bundle.get("temperature") or 1.0))
    calibrated_ordered = _softmax([value / temperature for value in logits])
    anchor_ordered = [float(belief_anchor[index]) for index in order]
    fusion_weight = min(1.0, max(0.0, float(bundle.get("fusion_weight") or 0.0)))
    fused_ordered = [
        fusion_weight * anchor_ordered[index]
        + (1.0 - fusion_weight) * calibrated_ordered[index]
        for index in range(3)
    ]

    def unorder(values: Sequence[float]) -> list[float]:
        output = [0.0] * 3
        for position, index in enumerate(order):
            output[index] = float(values[position])
        return output

    calibrated = unorder(calibrated_ordered)
    fused = unorder(fused_ordered)
    return {
        "applied": True,
        "bundle": bundle.get("_path"),
        "order": order,
        "features": {
            name: round(value, 6)
            for name, value in zip(
                bundle.get("feature_names") or [],
                features,
            )
        },
        "anchor": {
            label: round(float(anchor_ordered[position]), 6)
            for position, label in enumerate(
                [claim.labels[index] for index in order]
            )
        },
        "calibrated": {
            label: round(calibrated[index], 6)
            for index, label in enumerate(claim.labels)
        },
        "fused": {
            label: round(fused[index], 6)
            for index, label in enumerate(claim.labels)
        },
        "temperature": temperature,
        "fusion_weight": fusion_weight,
        "type_shrinkage": float(
            (config.get("calibration") or {}).get("type_shrinkage") or 0.5
        ),
    }


def normalize_numeric_estimate(parsed: Mapping[str, Any] | None) -> None:
    """Keep the auxiliary estimate in the claim's natural unit.

    The LLM sometimes converts 500 亿元 to 50,000,000,000 元.  The estimate is
    auxiliary, but a 50-billion display is confusing and can be mistaken for a
    different unit.  Convert back to 亿元 when the magnitude makes the source
    unit unambiguous.
    """

    if not isinstance(parsed, dict):
        return
    estimate = parsed.get("numeric_estimate")
    if not isinstance(estimate, dict):
        return
    try:
        point = float(estimate.get("point"))
    except (TypeError, ValueError):
        return
    unit = str(estimate.get("unit") or "").strip()
    factor = None
    new_unit = None
    if "亿" not in unit and abs(point) >= 1e8:
        factor, new_unit = 1e8, "亿元"
    elif "万元" in unit and abs(point) >= 1e4:
        factor, new_unit = 1e4, "亿元"
    if factor is None:
        return
    estimate["point"] = point / factor
    for key in ("low", "high"):
        value = estimate.get(key)
        if value is None:
            continue
        try:
            estimate[key] = float(value) / factor
        except (TypeError, ValueError):
            estimate[key] = None
    estimate["unit"] = new_unit


def answer_loss(
    prediction: str,
    truth: str,
    labels: Sequence[str],
    loss: Mapping[str, Any],
) -> float:
    if prediction == truth:
        return 0.0
    if len(labels) == 3:
        if prediction == labels[-1] and truth == labels[0]:
            return float(loss.get("over") or 12.0)
        if prediction == labels[0] and truth == labels[-1]:
            return float(loss.get("under") or 8.0)
    return float(loss.get("wrong") or 10.0)


def set_loss(
    chosen: Sequence[str],
    truth: str,
    loss: Mapping[str, Any],
) -> float:
    base = 0.0 if truth in chosen else float(loss.get("miss") or 8.0)
    return base + float(loss.get("width") or 2.0) * max(0, len(chosen) - 1)


def credible_set(belief: Sequence[float], labels: Sequence[str], alpha: float) -> list[str]:
    order = sorted(range(len(labels)), key=lambda i: (-belief[i], i))
    target = max(0.0, 1.0 - alpha)
    chosen: list[str] = []
    mass = 0.0
    for index in order:
        chosen.append(labels[index])
        mass += belief[index]
        if mass >= target - 1e-12:
            break
    return chosen


def full_action_readout(
    belief: Sequence[float],
    labels: Sequence[str],
    loss: Mapping[str, Any],
) -> dict[str, Any]:
    n = len(labels)
    total = sum(belief)
    belief = [value / total for value in belief] if total > 0 else [1.0 / n] * n
    # A singleton adjudication set is semantically a point answer with a
    # different loss label, so the adjudication action is restricted to sets
    # of size two or more.  This keeps the three actions distinct.
    subsets = [
        tuple(labels[i] for i in combination)
        for size in range(2, n + 1)
        for combination in itertools.combinations(range(n), size)
    ]
    answer_losses = []
    for label in labels:
        value = sum(
            belief[j] * answer_loss(label, labels[j], labels, loss) for j in range(n)
        )
        answer_losses.append(value)
    subset_losses = []
    for subset in subsets:
        value = sum(
            belief[j] * set_loss(subset, labels[j], loss) for j in range(n)
        )
        subset_losses.append(value)
    abstain = float(loss.get("abstain") or 5.0)

    best_answer_index = min(range(n), key=lambda i: (answer_losses[i], i))
    best_subset_index = min(
        range(len(subsets)), key=lambda i: (subset_losses[i], len(subsets[i]), i)
    )
    candidates = [
        (answer_losses[best_answer_index], 0, 0, "answer"),
        (subset_losses[best_subset_index], 1, len(subsets[best_subset_index]), "adjudicate"),
        (abstain, 2, 0, "abstain"),
    ]
    chosen_loss, _, _, action = min(candidates, key=lambda row: (row[0], row[1], row[2]))
    point = labels[best_answer_index]
    action_set = list(subsets[best_subset_index]) if action == "adjudicate" else ([point] if action == "answer" else [])
    alpha = float(loss.get("alpha") or 0.1)
    cred = credible_set(belief, labels, alpha)
    action_losses = [
        {
            "action": "answer",
            "label": labels[i],
            "expected_loss": round(answer_losses[i], 6),
        }
        for i in range(n)
    ]
    action_losses.extend(
        {
            "action": "adjudicate",
            "set": list(subset),
            "expected_loss": round(subset_losses[i], 6),
        }
        for i, subset in enumerate(subsets)
    )
    action_losses.append({"action": "abstain", "expected_loss": round(abstain, 6)})
    action_losses.sort(key=lambda row: row["expected_loss"])
    return {
        "action": action,
        "point": point if action != "abstain" else None,
        "argmax_label": point,
        "set": action_set,
        "expected_loss": round(chosen_loss, 6),
        "credible_set": cred,
        "credible_set_mass": round(sum(belief[labels.index(label)] for label in cred), 6),
        "alpha": alpha,
        "action_losses": action_losses,
    }


# ---------------------------------------------------------------------------
# Date estimate
# ---------------------------------------------------------------------------


def _midpoint(start: date, end: date) -> date:
    return start + timedelta(days=(end - start).days // 2)


def estimate_date(
    claim: ClaimContract,
    belief: Sequence[float],
    readout: Mapping[str, Any],
    parsed: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if claim.claim_type != "date":
        return None
    windows = {}
    for row in claim.date_windows:
        start, end = parse_date(row["start"]), parse_date(row["end"])
        if start and end:
            windows[row["label"]] = (start, end)
    if not windows:
        return None
    best_label = claim.labels[max(range(len(belief)), key=lambda i: belief[i])]
    start, end = windows.get(best_label, (None, None))
    estimate = (parsed or {}).get("date_estimate") or {}
    model_point = parse_date(estimate.get("point"))
    model_low = parse_date(estimate.get("low"))
    model_high = parse_date(estimate.get("high"))
    point = model_point
    consistent = point is not None and start is not None and end is not None and start <= point <= end
    if point is None or not consistent:
        point = _midpoint(start, end) if start and end else None
    low = model_low if model_low and point and model_low <= point else start
    high = model_high if model_high and point and model_high >= point else end
    decision_windows = []
    for label in readout.get("set") or []:
        if label in windows:
            window_start, window_end = windows[label]
            decision_windows.append(
                {
                    "label": label,
                    "start": window_start.isoformat(),
                    "end": window_end.isoformat(),
                }
            )
    return {
        "point": point.isoformat() if point else None,
        "low": low.isoformat() if low else None,
        "high": high.isoformat() if high else None,
        "window_label": best_label,
        "window_start": start.isoformat() if start else None,
        "window_end": end.isoformat() if end else None,
        "model_point": model_point.isoformat() if model_point else None,
        "model_low": model_low.isoformat() if model_low else None,
        "model_high": model_high.isoformat() if model_high else None,
        "consistent_with_window": bool(consistent),
        "basis": str(estimate.get("basis") or ""),
        "decision_windows": decision_windows,
    }


# ---------------------------------------------------------------------------
# Claim execution
# ---------------------------------------------------------------------------


def run_claim(
    claim: ClaimContract,
    spec: CaseSpec,
    config: Mapping[str, Any],
    cutoff: date,
    out_dir: Path | None = None,
    *,
    evidence_override: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.time()
    claim.target_period = claim.target_period or spec.target_period
    client = make_client(config)
    search_cfg = config.get("search", {})
    query_plan = build_query_plan(
        claim,
        spec,
        max_queries=int(search_cfg.get("max_queries") or 8),
        max_parser_queries=int(search_cfg.get("max_parser_queries") or 2),
        max_per_facet=int(search_cfg.get("max_per_facet") or 2),
        facets=search_cfg.get("query_facets"),
    )
    queries = [str(entry.get("query") or "") for entry in query_plan]
    if evidence_override is not None:
        evidence = []
        excluded: list[dict[str, Any]] = []
        search_log: list[dict[str, Any]] = []
        for item in evidence_override:
            row = dict(item)
            raw_date = str(row.get("date") or "")
            try:
                available = date.fromisoformat(raw_date[:10])
            except ValueError:
                available = None
            if available is not None and available > cutoff:
                excluded.append({**row, "reject_reason": "source date is after the cutoff"})
                continue
            evidence.append(row)
    else:
        evidence, excluded, search_log = collect_evidence(
            claim, spec, config, cutoff, query_plan=query_plan
        )
    required_facets = {
        facet_family(str(entry.get("facet")))
        for entry in query_plan
        if entry.get("required")
    }
    present_facets = {
        facet_family(str(item.get("facet") or "other")) for item in evidence
    }
    coverage_gaps = sorted(required_facets - present_facets)
    payload = extract_anchor_payload(
        claim, spec, evidence, config, cutoff.isoformat(), client
    )
    if out_dir is not None:
        prompt_dir = out_dir / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / f"{claim.claim_id}.txt").write_text(payload["prompt"], encoding="utf-8")
    parsed = payload.get("parsed")
    normalize_numeric_estimate(parsed)
    anchors = qualify_anchors(parsed, claim, evidence, cutoff, config)
    bridge_enabled = bool((config.get("bridge") or {}).get("enabled", True))
    forecast = build_forecast(claim, anchors, config) if bridge_enabled else None
    model_anchor = (
        bridge_anchor(forecast, claim, cutoff.isoformat(), config)
        if bridge_enabled and forecast is not None
        else None
    )
    if model_anchor is not None:
        anchors = list(anchors) + [model_anchor]

    calibration = load_calibration(config)
    calibration_result: dict[str, Any] | None = None
    if calibration is not None and calibration_applies(claim, calibration):
        if not claim.prior:
            calibrated_prior = calibration_prior(claim, calibration)
            if calibrated_prior:
                claim.prior = calibrated_prior
        apply_type_reliability(anchors, calibration, config)

    belief_anchor, belief_final, trace, hard_projection = update_belief(
        claim, anchors, parsed, config
    )
    if calibration is not None and calibration_applies(claim, calibration):
        if hard_projection:
            calibration_result = {
                "applied": False,
                "reason": "hard anchor projection takes precedence",
                "bundle": calibration.get("_path"),
            }
        else:
            calibration_result = apply_calibration(
                claim, anchors, belief_anchor, parsed, calibration, config
            )
            if calibration_result and calibration_result.get("applied"):
                belief_final = [
                    float(calibration_result["fused"][label]) for label in claim.labels
                ]

    coverage_penalty = 0.0
    if coverage_gaps and not hard_projection:
        per_gap = float(search_cfg.get("coverage_gap_penalty") or 0.0)
        max_penalty = float(search_cfg.get("max_coverage_penalty") or 0.45)
        coverage_penalty = min(max_penalty, per_gap * len(coverage_gaps))
        if coverage_penalty > 0:
            prior_vector = prior_belief(claim)
            belief_final = [
                (1.0 - coverage_penalty) * belief_final[index]
                + coverage_penalty * prior_vector[index]
                for index in range(len(claim.labels))
            ]
    readout = full_action_readout(belief_final, claim.labels, config.get("loss", {}))
    date_estimate = estimate_date(claim, belief_final, readout, parsed)
    seconds = round(time.time() - started, 2)
    return {
        "contract": claim.to_dict(),
        "cutoff": cutoff.isoformat(),
        "queries": queries,
        "query_plan": query_plan,
        "coverage_gaps": coverage_gaps,
        "coverage_penalty": coverage_penalty,
        "search_log": search_log,
        "evidence": evidence,
        "excluded_evidence": excluded,
        "llm": {
            "prompt_sha256": payload.get("prompt_sha256"),
            "request_digest": payload.get("request_digest"),
            "raw_text": payload.get("raw_text"),
            "parsed": parsed,
            "usage": payload.get("usage"),
            "error": payload.get("error"),
        },
        "anchors": anchors,
        "belief": {
            "prior": {label: round(value, 6) for label, value in zip(claim.labels, prior_belief(claim))},
            "anchor_only": {label: round(value, 6) for label, value in zip(claim.labels, belief_anchor)},
            "final": {label: round(value, 6) for label, value in zip(claim.labels, belief_final)},
            "hard_projection": hard_projection,
            "trace": trace,
        },
        "readout": readout,
        "date_estimate": date_estimate,
        "forecast": forecast.to_dict() if forecast is not None else None,
        "calibration": calibration_result,
        "numeric_estimate": (parsed or {}).get("numeric_estimate"),
        "seconds": seconds,
        "error": None,
    }


def run_claim_safe(
    claim: ClaimContract,
    spec: CaseSpec,
    config: Mapping[str, Any],
    cutoff: date,
    out_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        return run_claim(claim, spec, config, cutoff, out_dir, **kwargs)
    except Exception as error:  # noqa: BLE001
        return {
            "contract": claim.to_dict(),
            "cutoff": cutoff.isoformat(),
            "queries": [],
            "query_plan": [],
            "coverage_gaps": [],
            "coverage_penalty": 0.0,
            "search_log": [],
            "evidence": [],
            "excluded_evidence": [],
            "llm": {"error": f"{type(error).__name__}: {error}"},
            "anchors": [],
            "belief": {
                "prior": {},
                "anchor_only": {},
                "final": {},
                "hard_projection": [],
                "trace": [],
            },
            "readout": {
                "action": "abstain",
                "point": None,
                "set": [],
                "expected_loss": None,
                "credible_set": [],
                "action_losses": [],
            },
            "date_estimate": None,
            "forecast": None,
            "calibration": None,
            "numeric_estimate": None,
            "seconds": 0.0,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def make_run_id(case_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{case_id}-{secrets.token_hex(2)}"


def run_case(
    spec: CaseSpec,
    *,
    config: Mapping[str, Any] | None = None,
    cutoff: str | None = None,
    out_root: Path | str | None = None,
    max_workers: int = 4,
) -> dict[str, Any]:
    load_env()
    config = dict(config or load_config())
    cutoff_date = parse_date(cutoff or spec.cutoff) or date.today()
    out_root = Path(out_root) if out_root else DEFAULT_OUT
    run_id = make_run_id(spec.case_id or "case")
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")

    results: list[dict[str, Any]] = []
    workers = max(1, min(max_workers, len(spec.claims)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_claim_safe, claim, spec, config, cutoff_date, out_dir): index
            for index, claim in enumerate(spec.claims)
        }
        ordered: dict[int, dict[str, Any]] = {}
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
        results = [ordered[index] for index in range(len(spec.claims))]

    warnings = []
    for claim in spec.claims:
        if claim.claim_type == "metric" and not any(
            re.search(r"\d", label) for label in claim.labels
        ):
            warnings.append(
                f"claim {claim.claim_id} uses qualitative labels; provide numeric "
                "thresholds in a claims file for a fully auditable outcome space"
            )

    summary_claims = []
    action_counts: dict[str, int] = {}
    for row in results:
        readout = row.get("readout") or {}
        action = str(readout.get("action") or "error")
        action_counts[action] = action_counts.get(action, 0) + 1
        summary_claims.append(
            {
                "claim_id": row["contract"]["claim_id"],
                "text": row["contract"]["text"],
                "action": action,
                "point": readout.get("point"),
                "set": readout.get("set"),
                "expected_loss": readout.get("expected_loss"),
                "date_estimate": row.get("date_estimate"),
                "error": row.get("error"),
            }
        )
    result = {
        "run_id": run_id,
        "created_at": created_at,
        "case": spec.to_dict(),
        "cutoff": cutoff_date.isoformat(),
        "model": {
            "provider": config.get("llm", {}).get("provider"),
            "model": config.get("llm", {}).get("model"),
            "base_url": config.get("llm", {}).get("base_url"),
        },
        "config": {
            "search": config.get("search", {}),
            "anchors": config.get("anchors", {}),
            "loss": config.get("loss", {}),
        },
        "claims": results,
        "warnings": warnings,
        "summary": {
            "n_claims": len(results),
            "action_counts": action_counts,
            "claims": summary_claims,
        },
        "audit_dir": str(out_dir),
    }
    (out_dir / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "audit.md").write_text(render_audit_markdown(result), encoding="utf-8")
    return result


def run_ask(
    question: str,
    *,
    config: Mapping[str, Any] | None = None,
    client: LLMClient | None = None,
    cutoff: str | None = None,
    company: str = "",
    ticker: str = "",
    target_period: str = "",
    claims_file: str | Path | None = None,
    case_id: str | None = None,
    out_root: Path | str | None = None,
    max_workers: int = 4,
) -> dict[str, Any]:
    load_env()
    config = dict(config or load_config())
    if case_id:
        case_path = REPO_ROOT / "configs" / "anchor_cases" / f"{case_id}.yaml"
        if not case_path.exists():
            raise FileNotFoundError(f"unknown case: {case_id} ({case_path})")
        spec = load_case(case_path)
    elif claims_file:
        spec = load_case(claims_file)
    else:
        client = client or make_client(config)
        parsed = parse_question(
            question,
            config=config,
            client=client,
            company=company,
            ticker=ticker,
            target_period=target_period,
            today=cutoff or date.today().isoformat(),
        )
        spec = case_from_parsed(
            parsed,
            question=question,
            company=company,
            ticker=ticker,
            target_period=target_period,
            cutoff=cutoff or "",
        )
    if question and not spec.question:
        spec.question = question
    return run_case(
        spec,
        config=config,
        cutoff=cutoff,
        out_root=out_root,
        max_workers=max_workers,
    )


# ---------------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------------


def _md(value: Any, limit: int = 400) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.replace("|", "\\|")
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def render_audit_markdown(result: Mapping[str, Any]) -> str:
    lines: list[str] = []
    case = result.get("case") or {}
    lines.append(f"# FinPlanRAG 审计报告：{case.get('company') or case.get('case_id')}")
    lines.append("")
    lines.append(f"- run_id：`{result.get('run_id')}`")
    lines.append(f"- 生成时间：{result.get('created_at')}")
    lines.append(f"- 截止时点：{result.get('cutoff')}")
    lines.append(f"- 问题：{case.get('question')}")
    lines.append(f"- 模型：{result.get('model', {}).get('model')}")
    for warning in result.get("warnings") or []:
        lines.append(f"- 提示：{warning}")
    lines.append("")
    lines.append("## 结论摘要")
    lines.append("")
    lines.append("| 断言 | 动作 | 点答案 | 裁决集 | 期望损失 | 预计披露日 |")
    lines.append("|---|---|---|---|---|---|")
    for row in (result.get("summary") or {}).get("claims", []):
        date_row = row.get("date_estimate") or {}
        date_text = ""
        if date_row:
            date_text = f"{date_row.get('point') or '—'}（{date_row.get('low') or '—'} 至 {date_row.get('high') or '—'}）"
        lines.append(
            f"| {_md(row.get('text'), 80)} | {row.get('action')} | {_md(row.get('point'), 60)} | "
            f"{_md(', '.join(row.get('set') or []), 100)} | {row.get('expected_loss')} | {date_text} |"
        )
    lines.append("")

    for row in result.get("claims") or []:
        contract = row.get("contract") or {}
        lines.append(f"## 断言：{contract.get('text')}")
        lines.append("")
        lines.append(f"- claim_id：`{contract.get('claim_id')}`")
        lines.append(f"- 指标：{contract.get('metric')}")
        lines.append(f"- 类型：{contract.get('claim_type')}")
        lines.append(f"- 参考口径：{contract.get('reference')}")
        lines.append("- 结果空间：")
        for label in contract.get("labels") or []:
            definition = (contract.get("label_defs") or {}).get(label, "")
            lines.append(f"  - {label}：{definition}")
        if row.get("error"):
            lines.append(f"- **执行错误**：{row.get('error')}")
        lines.append("")

        query_plan = row.get("query_plan") or []
        if query_plan:
            lines.append("### 检索计划")
            lines.append("")
            result_counts = {
                str(item.get("query") or ""): item.get("n_results")
                for item in (row.get("search_log") or [])
            }
            lines.append("| 查询面 | 必需 | 检索词 | 结果数 |")
            lines.append("|---|---|---|---|")
            for entry in query_plan:
                lines.append(
                    f"| {_md(entry.get('facet'), 40)} | "
                    f"{'是' if entry.get('required') else '否'} | "
                    f"{_md(entry.get('query'), 100)} | "
                    f"{result_counts.get(str(entry.get('query') or ''), '')} |"
                )
            lines.append("")
            gaps = row.get("coverage_gaps") or []
            if gaps:
                lines.append(f"- **覆盖缺口**：{', '.join(str(gap) for gap in gaps)}")
                lines.append(
                    f"- 覆盖惩罚：后验向先验收缩 {row.get('coverage_penalty') or 0.0}"
                )
                lines.append("")

        evidence = row.get("evidence") or []
        lines.append("### 时点证据")
        lines.append("")
        if evidence:
            lines.append("| 编号 | 日期 | 来源 | 标题 | 链接 | 检索词 |")
            lines.append("|---|---|---|---|---|---|")
            for item in evidence:
                lines.append(
                    f"| {item.get('source_id')} | {item.get('date') or 'unknown'} | "
                    f"{item.get('provider') or 'web'} | {_md(item.get('title'), 90)} | "
                    f"{_md(item.get('url'), 160)} | {_md(item.get('query'), 60)} |"
                )
        else:
            lines.append("未检索到时点证据。")
        lines.append("")

        anchors = row.get("anchors") or []
        lines.append("### 锚点与资格判定")
        lines.append("")
        if anchors:
            lines.append("| 类型 | 取值/引文 | 来源 | 日期 | 引文核验 | 可靠性 | 资格 | 隐含标签 | 理由 |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for anchor in anchors:
                value = anchor.get("value") or anchor.get("quote") or ""
                lines.append(
                    f"| {_md(anchor.get('type'), 40)} | {_md(value, 120)} | {anchor.get('source_id')} | "
                    f"{anchor.get('date') or anchor.get('evidence_date') or 'unknown'} | "
                    f"{'通过' if anchor.get('quote_verified') else '未通过'} | "
                    f"{anchor.get('reliability_effective')} | {anchor.get('qualification')} | "
                    f"{_md(anchor.get('implied_label_matched'), 40)} | {_md('; '.join(anchor.get('reasons') or []), 120)} |"
                )
        else:
            lines.append("未抽取到有效锚点。")
        lines.append("")

        belief = row.get("belief") or {}
        labels = contract.get("labels") or []
        lines.append("### 信念更新")
        lines.append("")
        lines.append("| 标签 | 先验 | 仅锚点 | 融合后 |")
        lines.append("|---|---|---|---|")
        for label in labels:
            lines.append(
                f"| {label} | {(belief.get('prior') or {}).get(label)} | "
                f"{(belief.get('anchor_only') or {}).get(label)} | "
                f"{(belief.get('final') or {}).get(label)} |"
            )
        lines.append("")

        readout = row.get("readout") or {}
        lines.append("### 全动作损失感知读出")
        lines.append("")
        lines.append(f"- 选择动作：**{readout.get('action')}**")
        lines.append(f"- 点答案：{readout.get('point')}")
        lines.append(f"- 裁决集：{readout.get('set')}")
        lines.append(f"- 1-α 可信集：{readout.get('credible_set')}（质量 {readout.get('credible_set_mass')}）")
        lines.append(f"- 期望损失：{readout.get('expected_loss')}")
        lines.append("")
        if readout.get("action_losses"):
            lines.append("| 动作 | 期望损失 |")
            lines.append("|---|---|")
            for action in readout["action_losses"][:8]:
                label = action.get("label") or action.get("set") or ""
                lines.append(f"| {action.get('action')} {_md(label, 120)} | {action.get('expected_loss')} |")
            lines.append("")

        date_estimate = row.get("date_estimate")
        if date_estimate:
            lines.append("### 披露日期估计")
            lines.append("")
            lines.append(
                f"- 点估计：{date_estimate.get('point')}；区间：{date_estimate.get('low')} 至 {date_estimate.get('high')}"
            )
            lines.append(
                f"- 最高信念窗口：{date_estimate.get('window_label')}（{date_estimate.get('window_start')} 至 {date_estimate.get('window_end')}）"
            )
            lines.append(f"- 模型点估计与窗口一致：{date_estimate.get('consistent_with_window')}")
            lines.append(f"- 依据：{date_estimate.get('basis')}")
            lines.append("")

        calibration = row.get("calibration")
        if isinstance(calibration, dict):
            lines.append("### 历史校准")
            lines.append("")
            if calibration.get("applied"):
                lines.append(f"- bundle：{calibration.get('bundle')}")
                lines.append(f"- temperature：{calibration.get('temperature')}")
                lines.append(f"- fusion weight：{calibration.get('fusion_weight')}")
                lines.append(f"- type shrinkage：{calibration.get('type_shrinkage')}")
                if calibration.get("anchor"):
                    lines.append(f"- 锚点后验：{calibration.get('anchor')}")
                if calibration.get("calibrated"):
                    lines.append(f"- 校准后验：{calibration.get('calibrated')}")
                if calibration.get("fused"):
                    lines.append(f"- 融合后验：{calibration.get('fused')}")
            else:
                lines.append(f"- 未应用：{calibration.get('reason') or 'not applicable'}")
                if calibration.get("bundle"):
                    lines.append(f"- bundle：{calibration.get('bundle')}")
            lines.append("")

        forecast = row.get("forecast")
        if isinstance(forecast, dict):
            lines.append("### 锚点桥接模型")
            lines.append("")
            if forecast.get("point") is not None:
                unit = forecast.get("unit") or ""
                lines.append(f"- 模板：{forecast.get('template')}")
                lines.append(f"- 公式：{forecast.get('formula')}")
                lines.append(
                    f"- 点估计：{forecast.get('point')} {unit}；"
                    f"区间：{forecast.get('low')} 至 {forecast.get('high')} {unit}"
                )
                lines.append(f"- 区间来源：{forecast.get('interval_method')}")
                if forecast.get("inputs"):
                    lines.append(f"- 输入：{forecast.get('inputs')}")
                if forecast.get("input_sources"):
                    lines.append(f"- 输入来源：{forecast.get('input_sources')}")
                lines.append(
                    "- 说明：该数值由锚点中的数值按公式计算，不是 LLM 直接给出的点估计；"
                    "输入和公式保留在审计文件中。"
                )
            else:
                lines.append(f"- 模板：{forecast.get('template')}")
                lines.append(f"- 缺失输入：{', '.join(forecast.get('missing') or [])}")
                lines.append("- 说明：桥接模型输入不完整，未生成数值点估计。")
            lines.append("")

        numeric = row.get("numeric_estimate")
        if isinstance(numeric, dict) and numeric.get("point") is not None:
            lines.append("### 辅助数值估计")
            lines.append("")
            unit = numeric.get("unit") or ""
            lines.append(
                f"- 点估计：{numeric.get('point')} {unit}；区间：{numeric.get('low')} 至 {numeric.get('high')} {unit}"
            )
            lines.append(f"- 依据：{numeric.get('basis')}")
            sources = numeric.get("source_ids") or []
            if sources:
                lines.append(f"- 来源：{', '.join(str(item) for item in sources)}")
            lines.append("- 说明：该数值估计为辅助信息，不参与标签后验或动作选择。")
            lines.append("")

        excluded = row.get("excluded_evidence") or []
        if excluded:
            lines.append("### 被剔除的证据")
            lines.append("")
            for item in excluded:
                lines.append(
                    f"- {item.get('date') or 'unknown'}｜{_md(item.get('title'), 100)}｜"
                    f"{item.get('reason')}｜{item.get('url')}"
                )
            lines.append("")

    lines.append("## 审计说明")
    lines.append("")
    lines.append(
        "每条被采纳的锚点都记录了来源编号、原文引文、发布时间、资格判定和可靠性；"
        "每条被拒绝的锚点都记录了拒绝理由。证据在进入锚点抽取前按截止时点过滤，"
        "晚于截止时点的来源被列入过期证据。引文核验采用规范化后的逐字匹配，"
        "未通过核验的锚点会被降权且不能成为硬锚点。"
    )
    lines.append("")
    lines.append("本报告是时点判断，不是对已实现结果的复核。")
    return "\n".join(lines) + "\n"


def load_audit(run_id: str, out_root: Path | str | None = None) -> dict[str, Any]:
    root = Path(out_root) if out_root else DEFAULT_OUT
    path = root / run_id / "audit.json"
    if not path.exists():
        raise FileNotFoundError(f"audit not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def audit_markdown(run_id: str, out_root: Path | str | None = None) -> str:
    root = Path(out_root) if out_root else DEFAULT_OUT
    path = root / run_id / "audit.md"
    if not path.exists():
        raise FileNotFoundError(f"audit not found: {path}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Public summary
# ---------------------------------------------------------------------------


def public_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    claims = []
    for row in result.get("claims") or []:
        readout = row.get("readout") or {}
        claims.append(
            {
                "claim_id": row["contract"]["claim_id"],
                "claim": row["contract"]["text"],
                "action": readout.get("action"),
                "point": readout.get("point"),
                "set": readout.get("set"),
                "expected_loss": readout.get("expected_loss"),
                "credible_set": readout.get("credible_set"),
                "belief": (row.get("belief") or {}).get("final"),
                "n_anchors": len(row.get("anchors") or []),
                "n_accepted": sum(
                    1 for anchor in (row.get("anchors") or []) if anchor.get("status") == "accepted"
                ),
                "date_estimate": row.get("date_estimate"),
                "forecast": row.get("forecast"),
                "calibration": row.get("calibration"),
                "numeric_estimate": row.get("numeric_estimate"),
                "coverage_gaps": row.get("coverage_gaps") or [],
                "coverage_penalty": row.get("coverage_penalty") or 0.0,
                "error": row.get("error"),
            }
        )
    return {
        "run_id": result.get("run_id"),
        "case_id": (result.get("case") or {}).get("case_id"),
        "company": (result.get("case") or {}).get("company"),
        "cutoff": result.get("cutoff"),
        "audit_dir": result.get("audit_dir"),
        "summary": result.get("summary"),
        "warnings": result.get("warnings") or [],
        "claims": claims,
    }
