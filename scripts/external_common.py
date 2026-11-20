"""Shared utilities for the Tier-1 external validity experiments.

These experiments evaluate the FinPlan decision interface on datasets that were
not constructed by the paper:

* FinDVer (EMNLP 2024): claim verification over long SEC filings.
* DocFinQA (ACL 2024): long-document financial QA over full 10-K filings.

The adapters deliberately use the same reader, the same paragraph budget, and
the same provider client for every condition.  The only difference between the
fixed-budget and planner conditions is the evidence-acquisition policy.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from deepseek_client import LLMError, LLMClient
from finplan_rag.models import Chunk
from finplan_rag.retrieval import BM25Index, tokens

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[a-z0-9]+")
OBLIGATION_RE = re.compile(
    r"\$?\d[\d,]*(?:\.\d+)?%?|"
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
    flags=re.I,
)


def obligation_terms(text: str) -> list[str]:
    """Deterministic claim obligations: numbers, amounts, years, and months.

    These are the claim components that must be covered by acquired evidence.
    The list is intentionally conservative: it is auditable and does not rely on
    a model's self-reported sufficiency.
    """

    terms: list[str] = []
    seen: set[str] = set()
    for match in OBLIGATION_RE.findall(str(text or "")):
        term = match.strip().lower()
        term = term.replace("$", "").replace(",", "").replace("%", "")
        if term in {"", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}:
            continue
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def covered_terms(terms: Sequence[str], text: str) -> set[str]:
    lowered = str(text or "").lower().replace(",", "")
    covered: set[str] = set()
    for term in terms:
        if term.isdigit() and len(term) == 4 and term.startswith(("19", "20")):
            if re.search(r"(?<!\d)" + re.escape(term) + r"(?!\d)", lowered):
                covered.add(term)
        elif re.search(r"(?<![0-9.])" + re.escape(term) + r"(?![0-9])", lowered):
            covered.add(term)
    return covered


def uncovered_terms(terms: Sequence[str], chunks: Sequence[Any]) -> list[str]:
    text = " ".join(str(getattr(chunk, "text", "")) for chunk in chunks)
    covered = covered_terms(terms, text)
    return [term for term in terms if term not in covered]


# ---------------------------------------------------------------------------
# LLM plumbing
# ---------------------------------------------------------------------------


def make_client(
    *,
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    extra_body: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
) -> LLMClient:
    kwargs: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "cache_dir": cache_dir or (REPO_ROOT / "paper" / "results" / "external_tier1" / "llm_cache"),
        "timeout": 180,
        "max_retries": 5,
        "reasoning_effort": "low",
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    return LLMClient(**kwargs)


def parse_json_object(text: str) -> dict[str, Any] | None:
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
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def call_json(
    client: LLMClient,
    prompt: str,
    *,
    system: str,
    max_tokens: int = 700,
    repair: bool = True,
) -> dict[str, Any]:
    """Call the reader and return parsed JSON plus usage/cost metadata."""

    started = time.time()
    usage = {"input_tokens": 0, "output_tokens": 0}
    try:
        record = client.complete(
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=0.0,
            thinking=False,
            use_cache=True,
        )
        text = client.text_of(record)
        used = client.usage_of(record)
        usage = {k: usage[k] + used.get(k, 0) for k in usage}
        parsed = parse_json_object(text)
        if parsed is None and repair:
            repair_record = client.complete(
                prompt
                + "\nYour previous response was not valid JSON. Return only the "
                "requested JSON object, with no markdown and no prose.",
                system=system,
                max_tokens=max_tokens,
                temperature=0.0,
                thinking=False,
                use_cache=True,
            )
            text = client.text_of(repair_record)
            used = client.usage_of(repair_record)
            usage = {k: usage[k] + used.get(k, 0) for k in usage}
            parsed = parse_json_object(text)
        return {
            "parsed": parsed,
            "raw_text": text[:2000],
            "full_text": text[:12000],
            "usage": usage,
            "latency_seconds": time.time() - started,
            "error": None if parsed is not None else "parse_failure",
        }
    except LLMError as error:  # provider/runtime failure, not a data result
        return {
            "parsed": None,
            "raw_text": "",
            "usage": usage,
            "latency_seconds": time.time() - started,
            "error": f"api_error:{error}",
        }


# ---------------------------------------------------------------------------
# Paragraph chunks and retrieval
# ---------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def word_chunks(text: str, window: int = 160, stride: int = 120) -> list[str]:
    words = normalize_whitespace(text).split()
    chunks: list[str] = []
    if not words:
        return chunks
    for start in range(0, len(words), stride):
        chunk = " ".join(words[start : start + window])
        if chunk:
            chunks.append(chunk)
        if start + window >= len(words):
            break
    return chunks


def make_chunks(
    texts: Sequence[str],
    *,
    document_id: str,
    entity: str = "",
    available_at: datetime | None = None,
    max_chars: int = 6000,
    window: int = 160,
    stride: int = 120,
) -> list[Chunk]:
    stamp = available_at or datetime(2024, 1, 1, tzinfo=timezone.utc)
    chunks: list[Chunk] = []
    for index, text in enumerate(texts):
        for sub_index, piece in enumerate(word_chunks(str(text)[:max_chars], window, stride)):
            chunk_id = f"{document_id}::p{index}::w{sub_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    entity=entity,
                    available_at=stamp,
                    text=piece,
                    terms=tokens(piece),
                    metadata={"paragraph": index},
                )
            )
    return chunks


def retrieve(index: BM25Index, query: str, k: int, exclude: Iterable[str] = ()) -> list[Chunk]:
    return [hit.chunk for hit in index.search(query, limit=k, exclude=exclude)]


def format_evidence(chunks: Sequence[Chunk], max_chars_per_chunk: int = 3500) -> str:
    lines: list[str] = []
    for chunk in chunks:
        paragraph = chunk.metadata.get("paragraph", chunk.chunk_id)
        text = chunk.text[:max_chars_per_chunk].replace("\n", " ")
        lines.append(f"[paragraph id = {paragraph}] {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    return safe_div(sum(1 for a, b in zip(y_true, y_pred) if a == b), len(y_true))


def macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        scores.append(safe_div(2 * precision * recall, precision + recall))
    return sum(scores) / max(1, len(scores))


def aurc(confidences: Sequence[float], correct: Sequence[bool], answered: Sequence[bool]) -> float:
    """Area under the risk-coverage curve (lower is better).

    Abstentions have zero confidence and are placed at the end.  At each
    coverage level the risk is the error rate among answered examples; if no
    example is answered the risk is 1.0.
    """

    n = len(confidences)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: (-float(confidences[i]), i))
    points: list[tuple[float, float]] = []
    answered_count = 0
    correct_count = 0
    for rank, index in enumerate(order, start=1):
        if answered[index]:
            answered_count += 1
            correct_count += int(correct[index])
        coverage = rank / n
        risk = 1.0 - safe_div(correct_count, answered_count, default=0.0) if answered_count else 1.0
        points.append((coverage, risk))
    # Trapezoid over coverage.  Include the origin (0 coverage, risk 1).
    area = 0.0
    previous_cov, previous_risk = 0.0, 1.0
    for coverage, risk in points:
        area += (coverage - previous_cov) * (risk + previous_risk) / 2.0
        previous_cov, previous_risk = coverage, risk
    return area


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = 5000,
    seed: int = 20261109,
    alpha: float = 0.05,
) -> tuple[float, float]:
    import random

    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * (len(means) - 1))]
    hi = means[int((1 - alpha / 2) * (len(means) - 1))]
    return (lo, hi)


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = 5000,
    seed: int = 20261109,
    alpha: float = 0.05,
) -> dict[str, float]:
    import random

    if len(a) != len(b) or not a:
        return {"mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_two_sided": float("nan")}
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    means: list[float] = []
    for _ in range(resamples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * (len(means) - 1))]
    hi = means[int((1 - alpha / 2) * (len(means) - 1))]
    nonpositive = sum(1 for value in means if value <= 0) / len(means)
    nonnegative = sum(1 for value in means if value >= 0) / len(means)
    p_two = 2 * min(nonpositive, nonnegative)
    return {
        "mean_diff": sum(diffs) / n,
        "lo": lo,
        "hi": hi,
        "p_two_sided": min(1.0, p_two),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_parallel(items: Sequence[Any], fn, *, workers: int) -> list[Any]:
    results: list[Any] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def parse_date_from_report_name(name: str) -> datetime:
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
    if not match:
        return datetime(2024, 1, 1, tzinfo=timezone.utc)
    year, month, day = (int(part) for part in match.groups())
    return datetime(year, month, day, tzinfo=timezone.utc)
