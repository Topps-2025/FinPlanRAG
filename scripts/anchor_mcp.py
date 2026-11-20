"""MCP server exposing AnchorEvolve to Claude Code.

The server is intentionally thin: the tools call the same engine as the CLI,
so a Claude Code session and a terminal run produce identical audit artifacts.
Run it with ``python scripts/run_anchor_mcp.py`` or ``python scripts/anchor_cli.py serve``.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from anchor_engine import (  # noqa: E402
    audit_markdown,
    load_config,
    load_env,
    make_client,
    parse_question,
    public_summary,
    run_ask,
)
from anchor_web import gather  # noqa: E402

server = MCPServer(
    "finplan-anchor",
    instructions=(
        "AnchorEvolve evaluates forward-looking financial claims with point-in-time "
        "web evidence. Use anchor_ask for a full adjudication and anchor_audit to "
        "inspect the evidence-to-anchor-to-belief trail."
    ),
)


def _ok(payload: dict) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False)


def _strip_private(value: Any) -> Any:
    """Drop prompt/debug fields from MCP responses.

    The parser stores the raw completion, prompt and request digest under
    underscore-prefixed keys so the CLI audit can reproduce the call.  MCP
    clients do not need those fields, and returning them makes the tool output
    large enough to crowd out the useful contract.
    """
    if isinstance(value, dict):
        return {
            key: _strip_private(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_strip_private(item) for item in value]
    return value


def _error(error: Exception) -> str:
    return json.dumps(
        {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()[-2000:],
        },
        ensure_ascii=False,
    )


@server.tool(
    name="anchor_ask",
    description=(
        "Evaluate a forward-looking financial claim whose outcome is not yet disclosed. "
        "The engine retrieves dated web evidence, builds source-grounded anchors, "
        "qualifies them against a cutoff, updates a belief, predicts disclosure dates "
        "for date claims, and returns a point answer, an adjudication set, or abstention. "
        "Use this for questions such as a company's next annual-report metrics or the "
        "expected disclosure date. The full evidence trail is saved to the returned "
        "audit_dir. A case_id may name a curated case under configs/anchor_cases."
    ),
)
def anchor_ask(
    question: str = "",
    cutoff: str = "",
    company: str = "",
    ticker: str = "",
    target_period: str = "",
    case_id: str = "",
    claims_file: str = "",
) -> str:
    try:
        load_env()
        config = load_config()
        result = run_ask(
            question,
            config=config,
            cutoff=cutoff or None,
            company=company,
            ticker=ticker,
            target_period=target_period,
            case_id=case_id or None,
            claims_file=claims_file or None,
        )
        return _ok({"result": public_summary(result)})
    except Exception as error:  # noqa: BLE001
        return _error(error)


@server.tool(
    name="anchor_case",
    description=(
        "Run a curated AnchorEvolve case by id. Available cases are YAML files under "
        "configs/anchor_cases. The default example is zhongji_2026, which asks for the "
        "2026 annual-report revenue growth, net-profit growth, gross margin and expected "
        "disclosure date of 中际旭创."
    ),
)
def anchor_case(case_id: str = "zhongji_2026", cutoff: str = "") -> str:
    try:
        load_env()
        config = load_config()
        result = run_ask(
            "",
            config=config,
            cutoff=cutoff or None,
            case_id=case_id,
        )
        return _ok({"result": public_summary(result)})
    except Exception as error:  # noqa: BLE001
        return _error(error)


@server.tool(
    name="anchor_search",
    description=(
        "Search Chinese financial web sources and return dated evidence items with "
        "titles, snippets, URLs and publication dates. Use this for a quick evidence "
        "check; use anchor_ask for a full anchor-based adjudication."
    ),
)
def anchor_search(query: str, top_k: int = 8, fetch_top: int = 1) -> str:
    try:
        items = gather(query, top_k=max(1, min(int(top_k), 12)), fetch_top=max(0, min(int(fetch_top), 5)))
        compact = [
            {
                "source_id": f"W{index}",
                "title": item.get("title"),
                "url": item.get("url"),
                "date": item.get("date"),
                "snippet": item.get("snippet"),
            }
            for index, item in enumerate(items, 1)
        ]
        return _ok({"query": query, "evidence": compact})
    except Exception as error:  # noqa: BLE001
        return _error(error)


@server.tool(
    name="anchor_audit",
    description=(
        "Return the full Markdown audit report for a previous anchor_ask or anchor_case "
        "run. The report lists every evidence item, every accepted or rejected anchor, "
        "the belief trace, the full-action expected losses, and the date estimate."
    ),
)
def anchor_audit(run_id: str) -> str:
    try:
        return _ok({"run_id": run_id, "report": audit_markdown(run_id)})
    except Exception as error:  # noqa: BLE001
        return _error(error)


@server.tool(
    name="anchor_contract",
    description=(
        "Convert a natural-language forward-looking question into an explicit claim "
        "contract without running the full adjudication. Use this when the user wants "
        "to inspect or edit the outcome labels, thresholds and search queries first."
    ),
)
def anchor_contract(
    question: str,
    company: str = "",
    ticker: str = "",
    target_period: str = "",
    cutoff: str = "",
) -> str:
    try:
        load_env()
        config = load_config()
        client = make_client(config)
        parsed = parse_question(
            question,
            config=config,
            client=client,
            company=company,
            ticker=ticker,
            target_period=target_period,
            today=cutoff or None,
        )
        return _ok({"contract": _strip_private(parsed)})
    except Exception as error:  # noqa: BLE001
        return _error(error)


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
