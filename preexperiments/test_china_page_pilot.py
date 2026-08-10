import hashlib
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


def sha256(relative: str) -> str:
    path = DATA_ROOT / relative.removeprefix("data/") if relative.startswith("data/") else RESULTS_ROOT / relative.removeprefix("results/") if relative.startswith("results/") else ROOT / relative
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_page_pilot_freeze_and_result_hashes_match():
    freeze = load("china_finglm_page_pilot_freeze_v1.json")
    migration = load("storage_migration_audit_v1.json")
    data = load("data/china_finglm_page_pilot_v1.json")
    result = load("results/china_finglm_page_pilot_v1.json")
    assert freeze["status"] == "method-freeze-before-run"
    assert freeze["protocol"]["method_results_seen_before_freeze"] is False
    builder = migration["files"]["build_china_finglm_page_pilot.py"]
    runner = migration["files"]["run_china_finglm_page_pilot.py"]
    assert freeze["hashes"]["builder_sha256"] == builder["frozen_sha256"]
    assert freeze["hashes"]["runner_sha256"] == runner["frozen_sha256"]
    assert sha256("build_china_finglm_page_pilot.py") == builder["post_migration_sha256"]
    assert sha256("run_china_finglm_page_pilot.py") == runner["post_migration_sha256"]
    assert freeze["hashes"]["data_sha256"] == sha256("data/china_finglm_page_pilot_v1.json")
    assert result["protocol"]["runner_sha256"] == freeze["hashes"]["runner_sha256"]
    assert result["protocol"]["data_sha256"] == freeze["hashes"]["data_sha256"]
    assert result["protocol"]["cases"] == freeze["protocol"]["cases"] == 9
    assert result["protocol"]["documents"] == freeze["protocol"]["documents"] == 18
    assert result["protocol"]["pages"] == freeze["protocol"]["pages"] == 3567
    assert migration["status"] == "post-freeze-path-only-refactor-not-a-result-rerun"


def test_china_page_pilot_shows_representation_need_but_no_unique_strategy_gain():
    result = load("results/china_finglm_page_pilot_v1.json")
    frozen = result["summary"]["frozen_pilot"]
    assert frozen["finplan_v1_entity_only"]["closure"] == 0.0
    assert frozen["finplan_entity_year"]["closure"] == 1.0
    assert frozen["metadata_decomposition"]["closure"] == 1.0
    assert frozen["generic_adaptive_period"]["closure"] == 1.0
    assert frozen["hirec_period"]["closure"] == 1.0
    assert frozen["single_shot"]["closure"] == 1.0
    assert frozen["single_shot"]["wrong_doc_rate"] > frozen["finplan_entity_year"]["wrong_doc_rate"]
    comparison = result["paired_closure_vs_finplan"]["frozen_pilot"]
    assert comparison["metadata_decomposition"]["two_sided_exact_p"] == 1.0
    assert comparison["generic_adaptive_period"]["two_sided_exact_p"] == 1.0
    assert result["status"] == "transparent-public-pilot-not-sota-not-original-hirec"


def test_china_page_pilot_separates_raw_and_normalized_public_labels():
    result = load("results/china_finglm_page_pilot_v1.json")
    data = load("data/china_finglm_page_pilot_v1.json")
    audit = load("data/china_label_visual_audit_v1.json")
    assert data["label_audit"]["normalization_conflicts"] == [357, 1493]
    assert {item["qid"] for item in audit["audits"]} == {357, 1493}
    assert audit["scope"].startswith("仅修正标签质量诊断")
    all_rows = result["summary"]["all_descriptive"]
    assert all_rows["metadata_decomposition"]["field_oracle_semantic_correct"] == 1.0
    assert all_rows["metadata_decomposition"]["field_oracle_raw_gold_correct"] < 1.0


def test_page_evidence_is_complete_for_every_case():
    data = load("data/china_finglm_page_pilot_v1.json")
    assert all(case["evidence_complete"] for case in data["cases"])
    assert all(
        item["primary_page"] in item["candidate_pages"]
        for case in data["cases"]
        for item in case["gold_evidence"]
    )
