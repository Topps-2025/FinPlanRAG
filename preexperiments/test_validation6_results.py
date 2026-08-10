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
    path = DATA_ROOT / relative.removeprefix("data/") if relative.startswith("data/") else ROOT / relative
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validation6_is_new_file_disjoint_frozen_period_aware_data():
    data = load("data/lofin_validation6_v1.json")
    freeze = load("nonoracle_obligation_v7_validation6_freeze.json")
    migration = load("storage_migration_audit_v1.json")
    assert len(data["cases"]) == freeze["protocol"]["cases"] == 8
    assert len(data["documents"]) == freeze["protocol"]["documents"] == 14
    assert freeze["protocol"]["prior_document_overlap"] == 0
    assert freeze["protocol"]["within_batch_company_overlap"] == 0
    assert freeze["protocol"]["method_results_seen_before_freeze"] is False
    runner = migration["files"]["run_nonoracle_obligation_planning_v5.py"]
    assert freeze["hashes"]["runner_sha256"] == runner["frozen_sha256"]
    assert sha256("run_nonoracle_obligation_planning_v5.py") == runner["post_migration_sha256"]
    assert sha256("data/lofin_validation6_v1.json") == freeze["hashes"]["data_sha256"]
    assert sha256("lofin_validation6_manifest.json") == freeze["hashes"]["manifest_sha256"]


def test_frozen_v7_validation6_does_not_outperform_same_metadata_baselines():
    finplan = load("results/nonoracle_obligation_v7_validation6_frozen.json")
    baselines = load("results/period_aware_baselines_validation6.json")
    assert finplan["summary"]["cascade_closure"] == 6 / 8
    assert finplan["summary"]["cascade_wrong_doc_rate"] == 0.0
    assert finplan["summary"]["future_leak"] == 0.0
    summary = baselines["summary"]
    assert summary["single_shot"]["closure"] == 8 / 8
    assert summary["single_shot"]["wrong_doc_rate"] == 0.5625
    assert summary["period_metadata_decomposition"]["closure"] == 6 / 8
    assert summary["generic_adaptive_period"]["closure"] == 7 / 8
    assert summary["hirec_period"]["closure"] == 7 / 8
    assert finplan["summary"]["cascade_closure"] < summary["generic_adaptive_period"]["closure"]
