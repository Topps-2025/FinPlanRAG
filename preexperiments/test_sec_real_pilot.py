import json
from pathlib import Path

try:
    from .storage_paths import DATA_ROOT, RESULTS_ROOT
except ImportError:
    from storage_paths import DATA_ROOT, RESULTS_ROOT


ROOT = Path(__file__).resolve().parent


def load(relative: str):
    path = DATA_ROOT / relative.removeprefix("data/") if relative.startswith("data/") else RESULTS_ROOT / relative.removeprefix("results/") if relative.startswith("results/") else ROOT / relative
    return json.loads(path.read_text(encoding="utf-8"))


def test_sec_data_has_real_lineages_and_point_in_time_slices():
    exploratory = load("data/sec_real_pilot_v1.json")
    holdout = load("data/sec_real_pilot_holdout_v1.json")
    assert exploratory["protocol"]["lineages"] == 9
    assert holdout["protocol"]["lineages"] == 6
    assert all(doc["url"].startswith("https://www.sec.gov/Archives/edgar/data/") for doc in exploratory["documents"])
    assert {case["truth_state"] for case in exploratory["cases"]} >= {"conditional", "indirect", "none"}


def test_gap_diagnostic_is_observable_but_not_claimed_as_sota():
    result = load("results/sec_real_pilot_v1.json")
    assert result["status"] == "small-real-data-pilot-not-sota"
    assert result["summary"]["finplan_no_time"]["future_leak"] > 0.0
    for method, metrics in result["summary"].items():
        if method != "finplan_no_time":
            assert metrics["future_leak"] == 0.0


def test_exploratory_failure_triggered_architecture_revision():
    result = load("results/sec_real_pilot_v1.json")["summary"]
    strongest_generic = max(result["generic_adaptive"]["accuracy"], result["hirec_style"]["accuracy"])
    assert result["finplan_v1"]["accuracy"] < strongest_generic
    assert result["finplan_v3_path_bound"]["accuracy"] > strongest_generic
    assert result["finplan_v3_path_bound"]["wrong_lineage"] < result["finplan_v1"]["wrong_lineage"]


def test_frozen_v3_improves_on_prospective_holdout_without_sota_claim():
    result = load("results/sec_real_pilot_holdout_v1.json")["summary"]
    strongest_generic = max(result["generic_adaptive"]["accuracy"], result["hirec_style"]["accuracy"])
    assert result["finplan_v3_path_bound"]["accuracy"] > strongest_generic
    assert result["finplan_v3_path_bound"]["wrong_lineage"] == 0.0


def test_external_audit_covers_multiple_financial_task_rails():
    audit = load("results/financial_gap_audit_v1.json")
    assert audit["lofin"]["questions"] == 3031
    assert audit["finsearchcomp"]["questions"] == 635
    assert audit["lofin"]["subsets"]["secqa_test.jsonl"]["multi_evidence_rate"] > 0.9
    assert audit["lofin"]["subsets"]["textual_test.jsonl"]["multi_document_rate"] > 0.6
    assert audit["finsearchcomp"]["tier_counts"]["T3"] > 100
    assert audit["template_count"] >= 4
