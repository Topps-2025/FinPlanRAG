import hashlib
import json
from pathlib import Path

try:
    from .storage_paths import DATA_ROOT, RESULTS_ROOT
except ImportError:
    from storage_paths import DATA_ROOT, RESULTS_ROOT

from run_nonoracle_obligation_planning_v3 import (
    canonical_text,
    explicit_years,
    relative_window_size,
    select_years,
)
from run_nonoracle_obligation_planning_v4 import (
    nearest_year,
    select_years as select_years_v6,
    temporal_boundary_request,
)
from run_nonoracle_obligation_planning_v5 import obligation_signature, period_requests


ROOT = Path(__file__).resolve().parent


def load(relative: str):
    path = DATA_ROOT / relative.removeprefix("data/") if relative.startswith("data/") else RESULTS_ROOT / relative.removeprefix("results/") if relative.startswith("results/") else ROOT / relative
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(relative: str) -> str:
    path = DATA_ROOT / relative.removeprefix("data/") if relative.startswith("data/") else RESULTS_ROOT / relative.removeprefix("results/") if relative.startswith("results/") else ROOT / relative
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validation3_is_company_and_document_ood_and_frozen_for_v4():
    freeze = load("nonoracle_obligation_v4_validation3_freeze.json")
    result = load("results/nonoracle_obligation_v4_validation3.json")
    migration = load("storage_migration_audit_v1.json")["files"]["run_nonoracle_obligation_planning_v2.py"]
    assert freeze["protocol"]["cases"] == 22
    assert freeze["protocol"]["document_overlap_with_prior_batches"] == 0
    assert freeze["hashes"]["runner_sha256"] == migration["frozen_sha256"]
    assert sha256("run_nonoracle_obligation_planning_v2.py") == migration["post_migration_sha256"]
    assert result["protocol"]["source_sha256"] == freeze["hashes"]["runner_sha256"]


def test_v4_improves_three_closure_pairs_but_is_not_significant():
    audit = load("results/nonoracle_obligation_validation3_statistical_audit_v1.json")
    closure = audit["metrics"]["cascade_closure"]
    paired = closure["paired_exact_mcnemar"]
    assert closure["target_successes"] == 18
    assert closure["baseline_successes"] == 15
    assert paired["target_only_successes"] == 3
    assert paired["baseline_only_successes"] == 0
    assert paired["two_sided_exact_p"] == 0.25


def test_temporal_parser_repairs_concatenated_range_and_relative_window():
    years, repaired = explicit_years("over the past three fiscal years (20222024)")
    assert years == [2022, 2024]
    assert repaired
    assert relative_window_size("in the last 3 years") == 3
    assert relative_window_size("over the past three fiscal years") == 3
    selected, reason = select_years("over the past three fiscal years (20222024)", [2022, 2023, 2024])
    assert selected == [2022, 2023, 2024]
    assert reason == "relative-window"


def test_temporal_parser_respects_relative_anchor_and_event_visibility():
    selected, reason = select_years("As of 2023, dividends in the last 3 years", [2021, 2022, 2023, 2024])
    assert selected == [2021, 2022, 2023]
    assert reason == "relative-window"
    selected, reason = select_years("in 2023 prior to the sale and after the divestiture", [2021, 2022])
    assert selected == [2021, 2022]
    assert reason == "event-transition-visible-pair"


def test_filing_year_is_not_confused_with_forecast_range():
    question = "capital expenditure from 2024 to 2028 as stated in the 2023 10-K"
    selected, reason = select_years(question, [2023])
    assert selected == [2023]
    assert reason == "explicit-10k"


def test_entity_normalization_splits_curly_and_ascii_apostrophes():
    assert canonical_text("O'Reilly") == "o reilly"
    assert canonical_text("O’Reilly") == "o reilly"


def test_v6_parses_financial_temporal_boundaries_without_company_specific_rules():
    assert temporal_boundary_request("over the recent 5-year period") == ("recent-period", 5)
    assert temporal_boundary_request("compared to five years ago") == ("years-ago", 5)
    assert nearest_year([2020, 2024], 2019) == 2020
    selected, reason = select_years_v6("As of 2023 over the recent 5-year period", [2020, 2024])
    assert selected == [2020, 2024]
    assert reason == "temporal-boundaries:recent-period"


def test_v6_development_closes_validation4_but_remains_nonconfirmatory():
    result = load("results/nonoracle_obligation_v6_validation4_development.json")
    assert result["status"] == "post-validation4-error-analysis-development-not-confirmatory"
    assert result["summary"]["cascade_closure"] == 1.0
    assert result["summary"]["future_obligations"] == 0.0


def test_validation4_frozen_v5_hashes_and_negative_result_are_preserved():
    data = load("data/lofin_multidoc_validation4_v1.json")
    freeze = load("nonoracle_obligation_v5_validation4_freeze.json")
    result = load("results/nonoracle_obligation_v5_validation4_frozen.json")
    migration = load("storage_migration_audit_v1.json")["files"]["run_nonoracle_obligation_planning_v3.py"]
    assert len(data["cases"]) == freeze["protocol"]["cases"] == 6
    assert len(data["documents"]) == freeze["protocol"]["documents"] == 12
    assert data["protocol"]["failed_documents"] == {}
    assert freeze["hashes"]["runner_sha256"] == migration["frozen_sha256"]
    assert sha256("run_nonoracle_obligation_planning_v3.py") == migration["post_migration_sha256"]
    assert sha256("data/lofin_multidoc_validation4_v1.json") == freeze["hashes"]["data_sha256"]
    assert sha256("data/sec_company_registry_validation4_v1.json") == freeze["hashes"]["registry_sha256"]
    assert result["protocol"]["source_sha256"] == freeze["hashes"]["runner_sha256"]
    assert result["summary"]["cascade_closure"] == 0.5


def test_validation5_is_frozen_company_and_document_disjoint_mixed_filing_data():
    data = load("data/lofin_mixed_filing_validation5_v1.json")
    freeze = load("nonoracle_obligation_v6_validation5_freeze.json")
    migration = load("storage_migration_audit_v1.json")["files"]["run_nonoracle_obligation_planning_v4.py"]
    assert len(data["cases"]) == freeze["protocol"]["cases"] == 8
    assert len(data["documents"]) == freeze["protocol"]["documents"] == 21
    assert data["protocol"]["failed_documents"] == {}
    assert data["protocol"]["prior_document_overlap"] == 0
    assert freeze["hashes"]["runner_sha256"] == migration["frozen_sha256"]
    assert sha256("run_nonoracle_obligation_planning_v4.py") == migration["post_migration_sha256"]
    assert sha256("data/lofin_mixed_filing_validation5_v1.json") == freeze["hashes"]["data_sha256"]
    assert any(document["form"] == "10-Q" for document in data["documents"])
    ticker_sets = [{doc.split("_")[0] for doc in case["gold_doc_ids"]} for case in data["cases"]]
    for index, tickers in enumerate(ticker_sets):
        assert all(tickers.isdisjoint(other) for other in ticker_sets[index + 1 :])


def test_frozen_v6_exposes_entity_year_collision_on_mixed_filings():
    result = load("results/nonoracle_obligation_v6_validation5_frozen.json")
    assert result["summary"]["entity_exact"] == 1.0
    assert result["summary"]["obligation_recall"] == 0.0
    assert result["summary"]["cascade_closure"] == 0.5
    assert result["summary"]["future_obligations"] == 0.0


def test_mixed_filing_strong_baseline_beats_current_finplan_representation():
    methods = load("results/lofin_mixed_filing_validation5_methods_v1.json")["summary"]["validation5"]
    assert methods["finplan_v1"]["closure"] == 2 / 8
    assert methods["finplan_v3_path_bound"]["closure"] == 5 / 8
    assert methods["hirec_path_bound"]["closure"] == 7 / 8
    assert methods["hirec_path_bound"]["closure"] > methods["finplan_v3_path_bound"]["closure"]
    audit = load("results/lofin_mixed_filing_validation5_statistical_audit_v1.json")["validation5"]["methods"]
    assert audit["hirec_path_bound"]["paired_vs_finplan_v3_path_bound"]["two_sided_exact_p"] == 0.5


def test_period_parser_maps_quarters_and_q4_to_financial_filing_obligations():
    assert period_requests("as of the second quarter of 2024") == {(2024, "Q2")}
    assert period_requests("over the first three quarters of fiscal year 2023") == {
        (2023, "Q1"),
        (2023, "Q2"),
        (2023, "Q3"),
    }
    assert period_requests("Q3 and Q4 of FY2024, and Q1 of FY2025") == {
        (2024, "Q3"),
        (2024, "Q4"),
        (2025, "Q1"),
    }
    assert obligation_signature(2024, "Q4") == (2024, "10-K", "FY")


def test_v7_period_aware_replay_is_development_not_confirmation():
    result = load("results/nonoracle_obligation_v7_validation5_development.json")
    registry = load("data/sec_filing_registry_validation5_v2.json")
    assert registry["protocol"]["companies"] == 14
    assert registry["protocol"]["filings"] == 21
    assert result["status"] == "post-validation5-period-aware-development-not-confirmatory"
    assert result["summary"]["obligation_exact"] == 1.0
    assert result["summary"]["cascade_closure"] == 1.0
    assert result["summary"]["future_leak"] == 0.0


def test_v5_development_result_has_no_future_obligations_and_closes_all_validation3_cases():
    result = load("results/nonoracle_obligation_v5_validation3_development.json")
    assert result["status"] == "post-validation3-error-analysis-development-not-confirmatory"
    assert result["summary"]["future_obligations"] == 0.0
    assert result["summary"]["cascade_closure"] == 1.0
    assert result["summary"]["obligation_exact"] == 1.0
