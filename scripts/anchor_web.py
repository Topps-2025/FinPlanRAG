"""Web search and page extraction for the AnchorEvolve CLI.

The search layer uses Baidu's HTML endpoint because it returns usable Chinese
financial snippets in the local environment.  Pages are fetched as untrusted
text and never executed.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import date
from typing import Any
from urllib.parse import quote_plus, urljoin

import requests
from lxml import html as lxml_html

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
DATE_RE = re.compile(r"(20\d{2})\s*[年\-/]\s*(\d{1,2})\s*[月\-/]\s*(\d{1,2})")


def session() -> requests.Session:
    value = requests.Session()
    value.trust_env = False
    value.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"})
    return value


def clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace(" ", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def normalize_pdf_text(text: str) -> str:
    """Join numbers split across PDF line breaks.

    Financial tables extracted by a plain PDF reader often break
    ``31,849,895,741.23`` across lines.  Joining digit-only line breaks and
    the spaces around thousands separators makes the number usable by the
    anchor extractor while leaving ordinary prose alone.
    """

    text = re.sub(r"(?<=\d)[ \t]*\n[ \t]*(?=\d)", "", text)
    text = re.sub(r"(?<=\d),\s+(?=\d{3})", ",", text)
    text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)
    return text


def find_date(text: str) -> str | None:
    match = DATE_RE.search(text or "")
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


REDIRECT_MARKERS = ("baidu.com/link?", "baidu.php", "sogou.com/link?", "so.com/link?")
REDIRECT_HOSTS = ("baidu.com", "sogou.com", "so.com")


def resolve_link(http: requests.Session, url: str) -> str:
    if not any(marker in url for marker in REDIRECT_MARKERS):
        return url
    try:
        response = http.get(url, allow_redirects=True, timeout=12)
        if response.status_code < 400 and not any(host in response.url for host in REDIRECT_HOSTS):
            return response.url
    except Exception:
        pass
    return url


def _direct_url(block: Any, href: str) -> str:
    """Baidu exposes the destination in ``mu`` for many result blocks."""

    candidate = str(block.get("mu") or "").strip()
    if candidate and candidate.lower() != "null" and "nourl." not in candidate:
        if candidate.startswith("http"):
            return candidate
    return href


def search_baidu(query: str, *, top_k: int = 8) -> list[dict[str, Any]]:
    http = session()
    url = "https://www.baidu.com/s?wd=" + quote_plus(query)
    response = None
    for attempt in range(2):
        try:
            response = http.get(url, timeout=30)
            response.raise_for_status()
            break
        except Exception:
            response = None
            if attempt == 0:
                time.sleep(0.8)
    if response is None:
        raise RuntimeError(f"Baidu search failed for query: {query}")
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding or "utf-8"
    root = lxml_html.fromstring(response.text)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in root.xpath(
        '//div[contains(@class,"result") and contains(@class,"c-container")]'
    ):
        links = block.xpath(".//h3//a")
        if not links:
            continue
        title = clean(links[0].text_content())
        href = links[0].get("href") or ""
        if not title or not href or "baidu.php" in href:
            continue
        direct = _direct_url(block, href)
        snippet_nodes = block.xpath('.//div[contains(@class,"c-abstract")]')
        snippet = clean(" ".join(node.text_content() for node in snippet_nodes))
        if not snippet:
            snippet = clean(" ".join(block.itertext()))
            snippet = snippet.replace(title, "", 1).strip()
        key = (title, href)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "title": title,
                "url": href,
                "direct_url": direct,
                "snippet": snippet[:600],
                "date": find_date(snippet) or find_date(title),
                "query": query,
            }
        )
        if len(results) >= top_k:
            break
    for item in results:
        resolved = resolve_link(http, item["url"])
        if "baidu.com/link?" in resolved and item.get("direct_url"):
            resolved = item["direct_url"]
        item["url"] = resolved
        item.pop("direct_url", None)
    return results


def search_sogou(query: str, *, top_k: int = 8) -> list[dict[str, Any]]:
    """Sogou HTML search.  The page is GBK-encoded, so parse bytes directly."""

    http = session()
    url = "https://www.sogou.com/web?query=" + quote_plus(query)
    response = http.get(url, timeout=30)
    response.raise_for_status()
    root = lxml_html.fromstring(response.content)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in root.xpath('//div[contains(@class,"vrwrap")]'):
        links = block.xpath(".//h3//a")
        if not links:
            continue
        title = clean(links[0].text_content())
        href = links[0].get("href") or ""
        if href.startswith("/"):
            href = urljoin("https://www.sogou.com", href)
        if not title or not href:
            continue
        snippet_nodes = block.xpath(
            './/div[contains(@class,"fz-mid")] | '
            './/div[contains(@class,"text-layout")] | '
            './/div[contains(@class,"str-text-info")]'
        )
        snippet = (
            clean(" ".join(node.text_content() for node in snippet_nodes))
            if snippet_nodes
            else clean(" ".join(block.itertext()))
        )
        snippet = snippet.replace(title, "", 1).strip()
        key = (title, href)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "title": title,
                "url": href,
                "snippet": snippet[:600],
                "date": find_date(snippet) or find_date(title),
                "query": query,
            }
        )
        if len(results) >= top_k:
            break
    for item in results:
        item["url"] = resolve_link(http, item["url"])
    return results


def search_so(query: str, *, top_k: int = 8) -> list[dict[str, Any]]:
    """360 Search HTML endpoint."""

    http = session()
    url = "https://www.so.com/s?q=" + quote_plus(query)
    response = http.get(url, timeout=30)
    response.raise_for_status()
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding or "utf-8"
    root = lxml_html.fromstring(response.text)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in root.xpath('//li[contains(@class,"res-list")]'):
        links = block.xpath(".//h3//a")
        if not links:
            continue
        title = clean(links[0].text_content())
        href = links[0].get("href") or ""
        if not title or not href:
            continue
        snippet_nodes = block.xpath(
            './/div[contains(@class,"res-desc")] | '
            './/p[contains(@class,"res-desc")] | '
            './/div[contains(@class,"mh-3")]'
        )
        snippet = (
            clean(" ".join(node.text_content() for node in snippet_nodes))
            if snippet_nodes
            else clean(" ".join(block.itertext()))
        )
        snippet = snippet.replace(title, "", 1).strip()
        key = (title, href)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "title": title,
                "url": href,
                "snippet": snippet[:600],
                "date": find_date(snippet) or find_date(title),
                "query": query,
            }
        )
        if len(results) >= top_k:
            break
    for item in results:
        item["url"] = resolve_link(http, item["url"])
    return results


_PROVIDER_LOCK = threading.Lock()
_PROVIDER_INDEX = 0
_PROVIDERS = (
    ("sogou", search_sogou),
    ("so", search_so),
    ("baidu", search_baidu),
    ("sogou", search_sogou),
    ("so", search_so),
)


def search_web(query: str, *, top_k: int = 8) -> list[dict[str, Any]]:
    """Round-robin providers and fall back when a provider returns nothing.

    Baidu occasionally serves a security-verification page after a burst of
    requests.  Rotating across independent providers keeps a multi-claim run
    from silently returning an empty evidence set.
    """

    global _PROVIDER_INDEX
    with _PROVIDER_LOCK:
        start = _PROVIDER_INDEX
        _PROVIDER_INDEX += 1
    for offset in range(len(_PROVIDERS)):
        name, function = _PROVIDERS[(start + offset) % len(_PROVIDERS)]
        try:
            items = function(query, top_k=top_k)
        except Exception:
            continue
        if items:
            for item in items:
                item["provider"] = name
            return items
    return []


def fetch_page(url: str, *, max_chars: int = 20000) -> dict[str, Any] | None:
    http = session()
    try:
        response = http.get(url, timeout=30)
        if response.status_code >= 400:
            return None
        content_type = (response.headers.get("content-type") or "").lower()
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            try:
                import pymupdf
            except Exception:
                return None
            document = pymupdf.open(stream=response.content, filetype="pdf")
            text = "\n".join(page.get_text() for page in document)
            text = normalize_pdf_text(text)
            title = url.rsplit("/", 1)[-1]
        else:
            if not response.encoding or response.encoding.lower() == "iso-8859-1":
                response.encoding = response.apparent_encoding or "utf-8"
            root = lxml_html.fromstring(response.text)
            titles = root.xpath("//title/text()")
            title = clean(titles[0]) if titles else url
            for bad in root.xpath("//script|//style|//noscript"):
                bad.getparent().remove(bad)
            text = clean(" ".join(root.itertext()))
        return {
            "title": title,
            "url": response.url,
            "text": text[:max_chars],
            "date": find_date(text[:2000]),
        }
    except Exception:
        return None


def gather(query: str, *, top_k: int = 8, fetch_top: int = 2) -> list[dict[str, Any]]:
    """Search and optionally fetch the top pages as evidence items."""
    results = search_web(query, top_k=top_k)
    evidence: list[dict[str, Any]] = []
    for index, item in enumerate(results, 1):
        text = item["snippet"]
        final_url = item["url"]
        if index <= fetch_top:
            page = fetch_page(item["url"])
            if page and len(page.get("text", "")) > 200:
                text = page["text"]
                final_url = page["url"]
        evidence.append(
            {
                "source_id": f"W{index}",
                "title": item["title"],
                "url": final_url,
                "snippet": item["snippet"],
                "text": text,
                "date": item.get("date") or find_date(text),
                "retrieved_at": date.today().isoformat(),
                "query": query,
                "provider": item.get("provider"),
            }
        )
    return evidence


def format_evidence(items: list[dict[str, Any]], *, max_chars: int = 24000) -> str:
    lines: list[str] = []
    used = 0
    for item in items:
        header = f"[{item['source_id']}] {item['title']} | {item.get('date') or 'date unknown'} | {item['url']}"
        body = item.get("text") or item.get("snippet") or ""
        body = body[: max(500, max_chars // max(1, len(items)))]
        chunk = header + "\n" + body
        if used + len(chunk) > max_chars and lines:
            break
        lines.append(chunk)
        used += len(chunk)
    return "\n\n".join(lines)


if __name__ == "__main__":
    import json
    import sys

    q = " ".join(sys.argv[1:]) or "中际旭创 2025 年年度报告 营业收入 归母净利润"
    print(json.dumps(gather(q, top_k=6, fetch_top=1), ensure_ascii=False, indent=1)[:4000])
