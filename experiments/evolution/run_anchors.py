"""Run config-driven AnchorEvolve extraction on point-in-time 8-K excerpts.

The runner is deliberately free of result-specific constants: every prompt
field, threshold, model name, path and worker count comes from the config.  A
JSON parse failure is recorded as an error; there is no oracle fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from string import Template
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from deepseek_client import LLMClient  # noqa: E402
from experiments.evolution.filing_text import (  # noqa: E402
    format_excerpt,
    read_filing,
    select_paragraphs,
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = [fenced.group(1)] if fenced else []
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


def parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def company_key(record: dict) -> str:
    return str(record.get("cik") or record.get("ticker") or record.get("company") or "")


def masked_company(record: dict, enabled: bool) -> str:
    raw = str(record.get("company") or "")
    if not enabled:
        return raw
    digest = hashlib.sha1(company_key(record).encode("utf-8")).hexdigest()[:10]
    return f"company-{digest}"


def mask_text(text: str, record: dict, enabled: bool) -> str:
    if not enabled:
        return text
    raw = str(record.get("company") or "")
    base = raw.split("(")[0].strip()
    needles = {
        raw,
        base,
        str(record.get("ticker") or ""),
        str(record.get("cik") or ""),
    }
    if base:
        words = base.split()
        if len(words) >= 2:
            needles.add(" ".join(words[:2]))
    output = text
    for needle in sorted(needles, key=len, reverse=True):
        if len(needle) < 3:
            continue
        output = re.sub(re.escape(needle), "[COMPANY]", output, flags=re.I)
    return output


def history_string(record: dict, by_company: dict[str, list], cutoff: date) -> str:
    key = company_key(record)
    past = [
        str((row.get("outcome") or {}).get("bucket"))
        for _, outcome_date, row in by_company.get(key, [])
        if outcome_date < cutoff
    ]
    counts = {label: past.count(label) for label in ("below", "within", "above")}
    if not past:
        return "no revealed outcomes for this company before the cutoff"
    return (
        f"below={counts['below']}, within={counts['within']}, "
        f"above={counts['above']}, n={len(past)}"
    )


def peer_string(record: dict, records: list[tuple[date, date, dict]], cutoff: date) -> str:
    sector = str(record.get("sector") or "other")
    key = company_key(record)
    past = [
        str((row.get("outcome") or {}).get("bucket"))
        for _, outcome_date, row in records
        if outcome_date < cutoff
        and str(row.get("sector") or "other") == sector
        and company_key(row) != key
    ]
    counts = {label: past.count(label) for label in ("below", "within", "above")}
    if not past:
        return f"sector={sector}; no revealed peer outcomes before the cutoff"
    return (
        f"sector={sector}; below={counts['below']}, within={counts['within']}, "
        f"above={counts['above']}, n={len(past)}"
    )


def evidence_string(record: dict) -> str:
    lines = []
    for index, item in enumerate(record.get("evidence") or [], 1):
        parts = [f"E{index}"]
        for key in ("kind", "source_type", "origin", "date", "low_musd", "high_musd", "value_musd", "reliability", "informativeness"):
            if item.get(key) is not None:
                parts.append(f"{key}={item.get(key)}")
        lines.append("; ".join(parts))
    return "\n".join(lines) if lines else "no structured evidence items"


def build_prompt(record: dict, config: dict[str, Any], records: list[tuple[date, date, dict]], by_company: dict[str, list], template: Template) -> tuple[str, list[tuple[str, str]]]:
    cutoff = parse_date(record.get("cutoff"))
    if cutoff is None:
        raise ValueError(f"missing cutoff for {record.get('claim_id')}")
    filing_path = REPO_ROOT / config["filing_cache"] / f"{record['claim_id']}.txt"
    if not filing_path.exists():
        raise FileNotFoundError(f"missing filing cache: {filing_path}")
    text = read_filing(filing_path)
    filing_cfg = config["filing"]
    selected = select_paragraphs(
        text,
        max_chars=int(filing_cfg["max_chars"]),
        head_chars=int(filing_cfg["head_chars"]),
        top_k=int(filing_cfg["top_k_paragraphs"]),
        min_chars=int(filing_cfg["min_paragraph_chars"]),
    )
    excerpt = format_excerpt(selected)
    mask = bool(config["anchors"].get("mask_company", True))
    values = {
        "claim": mask_text(str(record.get("claim_text") or ""), record, mask),
        "company": masked_company(record, mask),
        "sector": str(record.get("sector") or "other"),
        "cutoff": str(record.get("cutoff")),
        "target_year": str(record.get("target_year") or ""),
        "target_quarter": str(record.get("target_quarter") or ""),
        "guidance_low": record.get("guidance_low_musd"),
        "guidance_mid": record.get("guidance_mid_musd"),
        "guidance_high": record.get("guidance_high_musd"),
        "guidance_width_pct": record.get("guidance_width_pct"),
        "current_q": record.get("current_q_musd"),
        "current_reliability": record.get("current_reliability"),
        "company_history": history_string(record, by_company, cutoff),
        "peer_prior": peer_string(record, records, cutoff),
        "evolved_type_reliability": "no past anchor outcomes yet",
        "evidence": evidence_string(record),
        "excerpt": excerpt,
        "types": ", ".join(config["anchors"]["types"]),
        "relations": ", ".join(config["anchors"]["relations"]),
        "max_anchors": config["anchors"]["max_anchors"],
    }
    prompt = template.safe_substitute(values)
    definitions = config["anchors"].get("type_definitions") or []
    if definitions:
        lines = ["", "Additional anchor types discovered from validation errors:"]
        for item in definitions:
            lines.append(
                f"- {item.get('name')}: relation={item.get('relation')}; "
                f"fields={', '.join(item.get('fields') or [])}; "
                f"{item.get('description') or ''}"
            )
        prompt += "\n".join(lines)
    addendum = str(config["anchors"].get("prompt_addendum") or "").strip()
    if addendum:
        prompt += "\n\nAdditional extraction rules:\n" + addendum
    return prompt, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/evolution/config.yaml")
    parser.add_argument("--split", default="test", choices=["dev", "validation", "test"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--claim-ids", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    config = load_config(REPO_ROOT / args.config)
    data = json.loads((REPO_ROOT / config["data"]).read_text(encoding="utf-8"))
    records = [
        (parse_date(row.get("cutoff")), parse_date((row.get("outcome") or {}).get("date")), row)
        for row in data
        if parse_date(row.get("cutoff")) and parse_date((row.get("outcome") or {}).get("date"))
    ]
    records.sort(key=lambda item: item[0])
    by_company: dict[str, list] = {}
    for item in records:
        by_company.setdefault(company_key(item[2]), []).append(item)

    selected = [row for row in data if row.get("split") == args.split]
    if args.claim_ids:
        wanted = {value.strip() for value in args.claim_ids.split(",") if value.strip()}
        selected = [row for row in selected if str(row["claim_id"]) in wanted]
    if args.limit:
        selected = selected[: args.limit]
    selected.sort(key=lambda row: str(row["claim_id"]))

    template = Template((REPO_ROOT / config["prompt_path"]).read_text(encoding="utf-8"))
    results_dir = REPO_ROOT / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else results_dir / f"anchors_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[str(row.get("claim_id"))] = row
    todo = [row for row in selected if str(row["claim_id"]) not in existing]
    print(f"existing={len(existing)} todo={len(todo)} out={out_path}", flush=True)

    local = threading.local()

    def client() -> LLMClient:
        if not hasattr(local, "client"):
            llm = config["llm"]
            local.client = LLMClient(
                provider=llm["provider"],
                model=llm["model"],
                base_url=llm["base_url"],
                api_key_env=llm["api_key_env"],
                cache_dir=REPO_ROOT / llm["cache_dir"],
                timeout=180,
                max_retries=5,
                extra_body=llm.get("extra_body"),
            )
        return local.client

    def work(record: dict) -> dict:
        try:
            prompt, selected_paragraphs = build_prompt(record, config, records, by_company, template)
            response = client().complete(
                prompt,
                system="Return only the JSON object. Do not add prose.",
                max_tokens=int(config["llm"]["max_tokens"]),
                temperature=float(config["llm"]["temperature"]),
                thinking=False,
                use_cache=True,
            )
            raw = client().text_of(response)
            parsed = parse_json(raw)
            usage = client().usage_of(response)
            return {
                "claim_id": str(record["claim_id"]),
                "split": record.get("split"),
                "truth": str((record.get("outcome") or {}).get("bucket")),
                "cutoff": record.get("cutoff"),
                "target_year": record.get("target_year"),
                "target_quarter": record.get("target_quarter"),
                "sector": record.get("sector"),
                "prompt": prompt,
                "excerpt_ids": [source_id for source_id, _ in selected_paragraphs],
                "raw_text": raw,
                "parsed": parsed,
                "bucket_scores": (parsed or {}).get("bucket_scores"),
                "anchors": (parsed or {}).get("anchors"),
                "confidence": (parsed or {}).get("confidence"),
                "missing": (parsed or {}).get("missing"),
                "usage": usage,
                "parse_ok": parsed is not None,
                "error": None if parsed is not None else "json_parse_failed",
            }
        except Exception as error:  # noqa: BLE001
            return {
                "claim_id": str(record.get("claim_id")),
                "split": record.get("split"),
                "truth": str((record.get("outcome") or {}).get("bucket")),
                "parse_ok": False,
                "error": f"{type(error).__name__}:{error}",
            }

    with out_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=int(config["llm"]["workers"])) as pool:
        futures = {pool.submit(work, row): row for row in todo}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(todo):
                print(f"{index}/{len(todo)}", flush=True)

    print("done", out_path)


if __name__ == "__main__":
    main()
