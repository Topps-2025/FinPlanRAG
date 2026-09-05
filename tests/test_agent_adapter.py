import json
import os
import subprocess
import sys

from finplan_rag.agent_adapter import FinPlanRAGAdapter, TOOL_NAME, tool_schema


def _request():
    return {
        "question": "FY 2024 revenue", "entities": ["ACME"],
        "filings": [{"entity": "ACME", "fiscal_year": 2024, "filing_type": "10-K", "fiscal_period": "FY", "document_id": "acme-2024", "available_at": "2025-02-01T00:00:00Z"}],
        "documents": [{"document_id": "acme-2024", "entity": "ACME", "fiscal_year": 2024, "filing_type": "10-K", "fiscal_period": "FY", "available_at": "2025-02-01T00:00:00Z", "text": "ACME 2024 FY revenue was 100."}],
        "cutoff": "2025-03-01T00:00:00Z",
    }


def test_tool_schema_and_facade_are_stable():
    assert tool_schema()["function"]["name"] == TOOL_NAME
    result = FinPlanRAGAdapter().handle(_request())
    assert result["closed"] is True
    assert result["covered_document_ids"] == ["acme-2024"]
    assert result["evidence"][0]["text"] == "ACME 2024 FY revenue was 100."
    assert result["recommendation"] == "answer_with_reader"


def test_stdio_protocol_returns_json_line():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    completed = subprocess.run([sys.executable, "-m", "finplan_rag.agent_adapter"], input=json.dumps(_request()) + "\n", text=True, capture_output=True, check=True, env=env)
    response = json.loads(completed.stdout)
    assert response["ok"] is True
    assert response["result"]["tool"] == TOOL_NAME
