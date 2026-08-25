"""Deterministic answer-support checks for retrieved financial evidence."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Sequence

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
REFUSAL_RE = re.compile(r"insufficient|cannot determine|not enough evidence|refuse", re.IGNORECASE)


def normalize(text: str) -> str:
    return re.sub(r"[\s,]", "", unicodedata.normalize("NFKC", str(text)).lower())


def numeric_cores(text: str) -> list[str]:
    return list(dict.fromkeys(match.group(0).replace(",", "") for match in NUMBER_RE.finditer(str(text))))


def supports(context: str, answer_variants: Sequence[str]) -> dict[str, object]:
    normalized = normalize(context)
    statuses = []
    for variant in answer_variants:
        cores = numeric_cores(variant)
        missing = [core for core in cores if not re.search(rf"(?<![0-9]){re.escape(core)}(?![0-9])", normalized)]
        statuses.append({"supported": not missing if cores else normalize(variant) in normalized, "missing": missing})
    return {"supported": any(item["supported"] for item in statuses), "variants": statuses}


def parse_reader_output(output: str) -> dict[str, object]:
    """Parse the last valid JSON reader object, with an honest refusal fallback."""
    parsed = None
    for match in JSON_RE.finditer(output):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and str(value.get("support", "")).lower() in {"yes", "no", "true", "false"}:
            parsed = {"answer": str(value.get("answer", "")), "support": str(value["support"]).lower() in {"yes", "true"}, "malformed": False}
    if parsed is not None:
        return parsed
    return {"answer": output.strip()[:200], "support": not bool(REFUSAL_RE.search(output)), "malformed": True}
