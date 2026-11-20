"""Extract readable text from a point-in-time SEC exhibit.

Downloaded filings are untrusted data.  This module only receives a path and
returns text; it never executes or imports anything from the filing directory.
"""

from __future__ import annotations

import re
from pathlib import Path

from lxml import html as lxml_html


BLOCK_TAGS = {
    "p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "section", "article",
}


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("​", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def html_to_text(raw: str) -> str:
    """Convert an SEC HTML exhibit to block-preserving text.

    The SEC Archives wrap exhibits in ``<DOCUMENT><TEXT>``; lxml handles that
    as ordinary HTML.  We keep block boundaries because a reader/LLM needs
    table rows and paragraph breaks.
    """
    if "<html" not in raw.lower() and "<document" not in raw.lower():
        return _clean(raw)
    try:
        root = lxml_html.fromstring(raw)
    except Exception:
        return _clean(re.sub(r"<[^>]+>", " ", raw))

    lines: list[str] = []
    stack = [root]
    while stack:
        node = stack.pop()
        tag = str(node.tag).lower() if isinstance(node.tag, str) else ""
        if tag in {"script", "style"}:
            continue
        if tag in BLOCK_TAGS:
            if node.text:
                lines.append(node.text)
            for child in node:
                stack.append(child)
            if node.tail:
                lines.append(node.tail)
        else:
            if node.text:
                lines.append(node.text)
            for child in reversed(list(node)):
                stack.append(child)
            if node.tail:
                lines.append(node.tail)
    return _clean("\n".join(lines))


def read_filing(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # Strip the manifest header written by fetch_filings.py.
    if raw.startswith("# claim_id="):
        raw = raw.split("\n", 1)[1] if "\n" in raw else ""
    return raw


def select_paragraphs(
    text: str,
    *,
    max_chars: int,
    head_chars: int,
    top_k: int,
    min_chars: int,
) -> list[tuple[str, str]]:
    """Select point-in-time paragraphs and attach stable excerpt ids.

    The first ``head_chars`` characters are always kept because the headline
    results and guidance usually appear there.  Additional paragraphs are
    ranked by revenue/guidance language.  The returned ids (``P1``...) are what
    the anchor prompt cites as ``source_id``.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{1,}", text) if len(p.strip()) >= min_chars]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chosen: set[int] = set()
    consumed = 0
    for index, paragraph in enumerate(paragraphs):
        if consumed >= head_chars:
            break
        chosen.add(index)
        consumed += len(paragraph)

    keywords = (
        "revenue", "sales", "guidance", "outlook", "expect", "forecast",
        "quarter", "segment", "demand", "backlog", "margin", "growth",
        "bookings", "order", "pipeline", "customer",
    )
    scored = []
    for index, paragraph in enumerate(paragraphs):
        lower = paragraph.lower()
        score = sum(lower.count(word) for word in keywords)
        if score:
            scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    for _, index in scored[:top_k]:
        chosen.add(index)

    if not chosen:
        chosen = set(range(min(top_k, len(paragraphs))))
    ordered = sorted(chosen)
    selected: list[tuple[str, str]] = []
    total = 0
    for index in ordered:
        paragraph = paragraphs[index]
        if total + len(paragraph) > max_chars and selected:
            break
        selected.append((f"P{len(selected) + 1}", paragraph))
        total += len(paragraph)
    return selected


def relevant_excerpt(text: str, *, max_chars: int, top_k: int, min_chars: int) -> str:
    """Backward-compatible text-only excerpt."""
    selected = select_paragraphs(
        text,
        max_chars=max_chars,
        head_chars=min(5000, max_chars),
        top_k=top_k,
        min_chars=min_chars,
    )
    return "\n\n".join(paragraph for _, paragraph in selected)


def format_excerpt(selected: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[{source_id}] {paragraph}" for source_id, paragraph in selected)
