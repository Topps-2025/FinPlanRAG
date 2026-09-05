"""Stable tool and stdio adapters for code agents."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

from .planner import ObligationPlanner, plan_obligations
from .retrieval import BM25Index, make_chunks

TOOL_NAME = "finplan_rag_plan"


def tool_schema() -> dict[str, object]:
    """Return an OpenAI-compatible function/tool schema."""
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Plan point-in-time financial evidence retrieval and expose auditable closure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "filings": {"type": "array", "items": {"type": "object"}},
                    "documents": {"type": "array", "items": {"type": "object"}},
                    "cutoff": {"type": "string", "description": "ISO-8601 timestamp"},
                    "budget": {"type": "integer", "minimum": 1, "default": 4},
                },
                "required": ["question", "entities", "filings", "documents", "cutoff"],
                "additionalProperties": False,
            },
        },
    }


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


class FinPlanRAGAdapter:
    """JSON-friendly facade used by tool-calling agents and local CLIs."""

    def __init__(self, default_budget: int = 4) -> None:
        if default_budget < 1:
            raise ValueError("default_budget must be positive")
        self.default_budget = default_budget

    def handle(self, arguments: Mapping[str, Any]) -> dict[str, object]:
        question = str(arguments.get("question", "")).strip()
        entities = [str(item) for item in arguments.get("entities", [])]
        filings = list(arguments.get("filings", []))
        documents = list(arguments.get("documents", []))
        if not question:
            raise ValueError("question is required")
        if not entities:
            raise ValueError("entities must contain at least one entity")
        cutoff = _timestamp(str(arguments.get("cutoff", "")))
        budget = int(arguments.get("budget", self.default_budget))
        if budget < 1:
            raise ValueError("budget must be positive")
        obligations = plan_obligations(question, filings, entities, cutoff)
        planner = ObligationPlanner(BM25Index(make_chunks(documents)), budget=budget)
        state = planner.run(question, obligations, cutoff)
        return {
            "tool": TOOL_NAME,
            "question": question,
            "cutoff": cutoff.isoformat().replace("+00:00", "Z"),
            "obligations": [
                {"entity": item.entity, "fiscal_year": item.fiscal_year,
                 "filing_type": item.filing_type, "fiscal_period": item.fiscal_period,
                 "document_id": item.document_id}
                for item in state.obligations
            ],
            "covered_document_ids": list(state.covered_document_ids),
            "queries": state.queries,
            "closed": state.closed,
            "future_document_ids": list(state.future_document_ids),
            "recommendation": "answer_with_reader" if state.closed else "abstain_or_retrieve_more",
        }


def run_stdio() -> None:
    """Process one JSON request per line and emit one JSON response per line."""
    adapter = FinPlanRAGAdapter()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            response = {"ok": True, "result": adapter.handle(json.loads(line))}
        except Exception as exc:
            response = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        print(json.dumps(response, ensure_ascii=True, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="FinPlan-RAG agent adapter")
    parser.add_argument("--schema", action="store_true", help="print the tool schema")
    args = parser.parse_args()
    if args.schema:
        print(json.dumps(tool_schema(), ensure_ascii=True, indent=2, sort_keys=True))
    else:
        run_stdio()


if __name__ == "__main__":
    main()
