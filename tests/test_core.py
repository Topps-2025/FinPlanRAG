from datetime import datetime, timezone

from finplan_rag.answer import parse_reader_output, supports
from finplan_rag.planner import ObligationPlanner, plan_obligations
from finplan_rag.retrieval import BM25Index, make_chunks
from finplan_rag.temporal import obligation_signature, period_requests


def test_period_parser_maps_cumulative_and_annual_language():
    assert period_requests("first three quarters of FY 2024") == {(2024, "Q3")}
    assert period_requests("first half of fiscal year 2024") == {(2024, "Q2")}
    assert period_requests("second half of FY2024") == {(2024, "Q3"), (2024, "Q4")}
    assert obligation_signature(2024, "Q4") == (2024, "10-K", "FY")


def test_planner_is_point_in_time_and_covers_filing_obligations():
    filings = [
        {"entity": "ACME", "fiscal_year": 2024, "filing_type": "10-K", "fiscal_period": "FY", "document_id": "acme-2024", "available_at": "2025-02-01T00:00:00Z"},
    ]
    obligations = plan_obligations("FY 2024 revenue", filings, ["ACME"])
    documents = [
        {"document_id": "acme-2024", "entity": "ACME", "available_at": "2025-02-01T00:00:00Z", "text": "ACME 2024 FY revenue was 100."},
        {"document_id": "future", "entity": "ACME", "available_at": "2026-01-01T00:00:00Z", "text": "ACME 2024 FY revenue was 999."},
    ]
    planner = ObligationPlanner(BM25Index(make_chunks(documents)), budget=2)
    state = planner.run("FY 2024 revenue", obligations, datetime(2025, 3, 1, tzinfo=timezone.utc))
    assert state.closed
    assert state.covered_document_ids == ("acme-2024",)
    assert state.future_document_ids == ()
    assert state.retrieved_chunks[0].document_id == "acme-2024"


def test_answer_support_and_reader_fallback_are_deterministic():
    assert supports("Revenue increased to 1,234.", ["1,234"])["supported"]
    parsed = parse_reader_output("Insufficient evidence to answer.")
    assert parsed["support"] is False
    assert parsed["malformed"] is True
