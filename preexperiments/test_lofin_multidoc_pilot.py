import hashlib
import json
from functools import lru_cache
from pathlib import Path

try:
    from .storage_paths import DATA_ROOT, MODELS_ROOT, RESULTS_ROOT
except ImportError:
    from storage_paths import DATA_ROOT, MODELS_ROOT, RESULTS_ROOT


ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load(relative: str):
    path = DATA_ROOT / relative.removeprefix("data/") if relative.startswith("data/") else RESULTS_ROOT / relative.removeprefix("results/") if relative.startswith("results/") else ROOT / relative
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(relative: str) -> str:
    path = DATA_ROOT / relative.removeprefix("data/") if relative.startswith("data/") else RESULTS_ROOT / relative.removeprefix("results/") if relative.startswith("results/") else MODELS_ROOT / relative.removeprefix("models/") if relative.startswith("models/") else ROOT / relative
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_lofin_pilot_uses_real_sec_documents_and_disjoint_splits():
    data = load("data/lofin_multidoc_pilot_v1.json")
    assert len(data["cases"]) == 17
    assert len(data["documents"]) == 38
    assert all(doc["url"].startswith("https://www.sec.gov/Archives/edgar/data/") for doc in data["documents"])
    assert all(hashlib.sha256(doc["text"].encode("utf-8")).hexdigest() == doc["text_sha256"] for doc in data["documents"])

    exploratory = {
        doc_id
        for case in data["cases"]
        if case["split"] == "exploratory"
        for doc_id in case["gold_doc_ids"]
    }
    holdout = {
        doc_id
        for case in data["cases"]
        if case["split"] == "holdout"
        for doc_id in case["gold_doc_ids"]
    }
    assert exploratory.isdisjoint(holdout)


def test_gold_closure_is_entity_year_based_and_has_multiple_documents():
    data = load("data/lofin_multidoc_pilot_v1.json")
    available = {doc["doc_id"] for doc in data["documents"]}
    for case in data["cases"]:
        expected = {
            f"{leg['ticker']}_{year}_10K"
            for leg in case["candidate_legs"]
            for year in leg["years"]
        }
        assert expected == set(case["gold_doc_ids"])
        assert 2 <= len(expected) <= 3
        assert expected <= available


def test_holdout_was_run_with_the_frozen_runner():
    freeze = load("lofin_multidoc_freeze_v1.json")
    migration = load("storage_migration_audit_v1.json")["files"]["run_lofin_multidoc_pilot.py"]
    assert freeze["hashes"]["runner_sha256"] == migration["frozen_sha256"]
    assert sha256("run_lofin_multidoc_pilot.py") == migration["post_migration_sha256"]
    assert sha256("data/lofin_multidoc_pilot_v1.json") == freeze["hashes"]["data_sha256"]
    assert sha256("results/lofin_multidoc_pilot_exploratory_v3.json") == freeze["hashes"]["exploratory_result_sha256"]

    exploratory = load("results/lofin_multidoc_pilot_exploratory_v3.json")
    holdout = load("results/lofin_multidoc_pilot_holdout_v1.json")
    assert exploratory["protocol"]["source_sha256"] == freeze["hashes"]["runner_sha256"]
    assert holdout["protocol"]["source_sha256"] == freeze["hashes"]["runner_sha256"]
    assert exploratory["protocol"]["selected_split"] == "exploratory"
    assert holdout["protocol"]["selected_split"] == "holdout"


def test_unbound_methods_exhibit_a_replicated_document_closure_gap():
    exploratory = load("results/lofin_multidoc_pilot_exploratory_v3.json")["summary"]["exploratory"]
    holdout = load("results/lofin_multidoc_pilot_holdout_v1.json")["summary"]["holdout"]
    for split in (exploratory, holdout):
        assert split["fixed_decomposition"]["closure"] < 1.0
        assert split["generic_adaptive"]["closure"] < 1.0
        assert split["hirec_style"]["closure"] < 1.0
        assert split["finplan_v1"]["closure"] < 1.0


def test_path_binding_closes_documents_but_does_not_beat_metadata_ceiling():
    exploratory = load("results/lofin_multidoc_pilot_exploratory_v3.json")["summary"]["exploratory"]
    holdout = load("results/lofin_multidoc_pilot_holdout_v1.json")["summary"]["holdout"]
    for split in (exploratory, holdout):
        assert split["finplan_v3_path_bound"]["closure"] == 1.0
        assert split["finplan_v3_path_bound"]["wrong_doc_rate"] == 0.0
        assert split["finplan_v3_path_bound"]["closure"] > split["finplan_v1"]["closure"]
        assert split["finplan_v3_path_bound"]["closure"] == split["metadata_decomposition"]["closure"]
        assert split["finplan_v3_path_bound"]["wrong_doc_rate"] == split["metadata_decomposition"]["wrong_doc_rate"]


def test_time_ablation_is_non_identifying_on_this_track():
    exploratory = load("results/lofin_multidoc_pilot_exploratory_v3.json")["summary"]["exploratory"]
    holdout = load("results/lofin_multidoc_pilot_holdout_v1.json")["summary"]["holdout"]
    for split in (exploratory, holdout):
        assert split["finplan_v3_path_bound"]["future_leak"] == 0.0
        assert split["finplan_no_time"]["future_leak"] == 0.0
        assert split["finplan_no_time"]["closure"] == split["finplan_v3_path_bound"]["closure"]


def test_validation2_is_frozen_and_document_disjoint_from_v1():
    data = load("data/lofin_multidoc_validation2_v1.json")
    freeze = load("lofin_multidoc_validation2_freeze_v1.json")
    manifest = load("lofin_multidoc_validation2_manifest.json")
    old = load("data/lofin_multidoc_pilot_v1.json")
    assert len(data["cases"]) == freeze["protocol"]["cases"] == 30
    assert len(data["documents"]) == freeze["protocol"]["documents"] == 67
    assert data["protocol"]["failed_documents"] == {}
    assert data["protocol"]["selection_manifest_sha256"] == freeze["hashes"]["manifest_sha256"]
    assert sha256("data/lofin_multidoc_validation2_v1.json") == freeze["hashes"]["data_sha256"]
    migration = load("storage_migration_audit_v1.json")["files"]["run_lofin_multidoc_pilot.py"]
    assert freeze["hashes"]["runner_sha256"] == migration["frozen_sha256"]
    assert sha256("run_lofin_multidoc_pilot.py") == migration["post_migration_sha256"]
    assert len(manifest["cases"]) == 30

    old_docs = {doc["doc_id"] for doc in old["documents"]}
    new_docs = {doc["doc_id"] for doc in data["documents"]}
    assert old_docs.isdisjoint(new_docs)
    assert len(new_docs) == 67


def test_validation2_replicates_unbound_gap_and_path_bound_ceiling():
    result = load("results/lofin_multidoc_validation2_v1.json")["summary"]["validation2"]
    assert result["finplan_v1"]["closure"] == 20 / 30
    assert result["generic_adaptive"]["closure"] == 22 / 30
    assert result["hirec_style"]["closure"] == 27 / 30
    assert result["finplan_v3_no_path"]["closure"] == 22 / 30
    assert result["finplan_v3_path_bound"]["closure"] == 1.0
    assert result["metadata_decomposition"]["closure"] == 1.0
    assert result["generic_path_bound"]["closure"] == 1.0
    assert result["hirec_path_bound"]["closure"] == 1.0


def test_validation2_statistical_audit_keeps_strong_baseline_boundary():
    audit = load("results/lofin_multidoc_validation2_statistical_audit_v1.json")
    methods = audit["validation2"]["methods"]
    assert methods["finplan_v3_path_bound"]["wilson_95"][0] > 0.88
    assert methods["finplan_v1"]["paired_vs_finplan_v3_path_bound"]["two_sided_exact_p"] < 0.01
    assert methods["generic_adaptive"]["paired_vs_finplan_v3_path_bound"]["two_sided_exact_p"] < 0.01
    assert methods["hirec_style"]["paired_vs_finplan_v3_path_bound"]["two_sided_exact_p"] > 0.05
    assert methods["metadata_decomposition"]["paired_vs_finplan_v3_path_bound"]["two_sided_exact_p"] == 1.0


def test_validation2_runner_hash_is_recorded_in_result():
    result = load("results/lofin_multidoc_validation2_v1.json")
    freeze = load("lofin_multidoc_validation2_freeze_v1.json")
    assert result["protocol"]["source_sha256"] == freeze["hashes"]["runner_sha256"]
    assert result["protocol"]["data_sha256"] == freeze["hashes"]["data_sha256"]


def test_dense_and_hybrid_robustness_replicate_the_unbound_gap():
    dense = load("results/lofin_multidoc_validation2_dense_v1.json")["summary"]["validation2"]
    hybrid = load("results/lofin_multidoc_validation2_hybrid_v1.json")["summary"]["validation2"]
    for result in (dense, hybrid):
        assert result["finplan_v1"]["closure"] < 0.7
        assert result["generic_adaptive"]["closure"] < 0.75
        assert result["hirec_style"]["closure"] < 1.0
        assert result["finplan_v3_path_bound"]["closure"] == 1.0
        assert result["finplan_v3_path_bound"]["wrong_doc_rate"] == 0.0
        assert result["metadata_decomposition"]["closure"] == 1.0
        assert result["generic_path_bound"]["closure"] == 1.0


def test_dense_and_hybrid_results_match_frozen_runners_and_model():
    dense_freeze = load("lofin_multidoc_dense_freeze_v1.json")
    hybrid_freeze = load("lofin_multidoc_hybrid_freeze_v1.json")
    dense = load("results/lofin_multidoc_validation2_dense_v1.json")
    hybrid = load("results/lofin_multidoc_validation2_hybrid_v1.json")
    migration = load("storage_migration_audit_v1.json")["files"]
    dense_runner = migration["run_lofin_multidoc_dense.py"]
    hybrid_runner = migration["run_lofin_multidoc_hybrid.py"]
    assert dense_freeze["hashes"]["runner_sha256"] == dense_runner["frozen_sha256"]
    assert hybrid_freeze["hashes"]["runner_sha256"] == hybrid_runner["frozen_sha256"]
    assert sha256("run_lofin_multidoc_dense.py") == dense_runner["post_migration_sha256"]
    assert sha256("run_lofin_multidoc_hybrid.py") == hybrid_runner["post_migration_sha256"]
    assert sha256("models/bge-small-en-v1.5/model.safetensors") == dense_freeze["hashes"]["model_safetensors_sha256"]
    assert dense["protocol"]["source_sha256"] == dense_freeze["hashes"]["runner_sha256"]
    assert hybrid["protocol"]["source_sha256"] == hybrid_freeze["hashes"]["runner_sha256"]
    assert dense["protocol"]["data_sha256"] == dense_freeze["hashes"]["data_sha256"]
    assert hybrid["protocol"]["data_sha256"] == hybrid_freeze["hashes"]["data_sha256"]


def test_dense_hybrid_statistics_reject_overclaim_against_bound_baselines():
    for relative, label in (
        ("results/lofin_multidoc_validation2_dense_statistical_audit_v1.json", "dense_validation2"),
        ("results/lofin_multidoc_validation2_hybrid_statistical_audit_v1.json", "hybrid_validation2"),
    ):
        methods = load(relative)[label]["methods"]
        assert methods["generic_adaptive"]["paired_vs_finplan_v3_path_bound"]["two_sided_exact_p"] < 0.01
        assert methods["metadata_decomposition"]["paired_vs_finplan_v3_path_bound"]["two_sided_exact_p"] == 1.0
        assert methods["generic_path_bound"]["paired_vs_finplan_v3_path_bound"]["two_sided_exact_p"] == 1.0
        assert methods["hirec_path_bound"]["paired_vs_finplan_v3_path_bound"]["two_sided_exact_p"] == 1.0


def test_result_status_forbids_end_to_end_or_sota_interpretation():
    for relative in (
        "results/lofin_multidoc_pilot_exploratory_v3.json",
        "results/lofin_multidoc_pilot_holdout_v1.json",
    ):
        result = load(relative)
        assert result["status"] == "document-closure-not-full-answer-generation"


def test_statistical_audit_reports_small_holdout_uncertainty():
    audit = load("results/lofin_multidoc_statistical_audit_v1.json")
    assert audit["status"] == "post-hoc-small-sample-uncertainty-not-sota"
    holdout = audit["holdout"]["methods"]
    assert holdout["finplan_v3_path_bound"]["successes"] == 8
    assert holdout["finplan_v3_path_bound"]["wilson_95"][0] < 0.7
    assert holdout["finplan_v1"]["paired_vs_finplan_v3_path_bound"]["two_sided_exact_p"] > 0.05
    assert holdout["metadata_decomposition"]["paired_vs_finplan_v3_path_bound"]["two_sided_exact_p"] == 1.0
