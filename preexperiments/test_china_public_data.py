import hashlib
import csv
import json
import re
from pathlib import Path

try:
    from .storage_paths import DATA_ROOT, RESULTS_ROOT
except ImportError:
    from storage_paths import DATA_ROOT, RESULTS_ROOT


ROOT = Path(__file__).resolve().parent
CHINA = DATA_ROOT / "external" / "china"


def load(relative: str):
    path = RESULTS_ROOT / relative.removeprefix("results/") if relative.startswith("results/") else ROOT / relative
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_china_public_source_files_are_pinned_and_readable():
    audit = load("results/china_public_data_audit_v1.json")
    finglm, finglm2 = audit["sources"]
    for source in (finglm, finglm2):
        for filename, expected_hash in source["local_files"].items():
            path = CHINA / filename
            assert path.exists()
            assert sha256(path) == expected_hash
            assert "\ufffd" not in path.read_text(encoding="utf-8")


def test_finglm_report_pool_supports_real_cross_year_selection():
    audit = load("results/china_public_data_audit_v1.json")["sources"][0]
    assert audit["qa"]["questions"] == 2000
    assert audit["qa"]["answers"] == 2000
    assert audit["qa"]["ids_aligned"]
    assert audit["reports"]["rows"] == 11588
    assert audit["reports"]["parse_failures"] == []
    assert audit["reports"]["unique_security_codes"] == 4366
    assert audit["reports"]["security_codes_with_2019_2020_2021"] == 3330
    assert audit["qa"]["explicit_xiangbi_questions"] == 9


def test_finglm2_is_not_misrepresented_as_complete_rag_benchmark():
    audit = load("results/china_public_data_audit_v1.json")["sources"][1]
    assert audit["questions"]["groups"] == 101
    assert audit["questions"]["subquestions"] == 311
    assert audit["questions"]["multi_turn"]
    assert not audit["data_availability"]["competition_database"]
    assert not audit["questions"]["gold_answers_in_repository"]
    assert audit["data_availability"]["classification"] == "questions-and-schema-only"


def test_china_pilot_selection_is_company_disjoint_and_frozen_before_methods():
    manifest = load("china_public_data_pilot_manifest_v1.json")
    assert not manifest["method_results_seen_before_freeze"]
    assert len(manifest["development_qids"]) == 4
    assert len(manifest["frozen_pilot_qids"]) == 5
    assert len(manifest["cases"]) == 9
    codes = [case["stock_code"] for case in manifest["cases"]]
    assert len(codes) == len(set(codes))
    assert all(len(case["years"]) == 2 for case in manifest["cases"])
    audit_ids = load("results/china_public_data_audit_v1.json")["sources"][0]["qa"][
        "explicit_xiangbi_ids"
    ]
    assert [case["qid"] for case in manifest["cases"]] == audit_ids
    with (CHINA / "finglm_reports_list.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        reports = list(csv.DictReader(handle))
    for case in manifest["cases"]:
        available_years = {
            int(match.group(1))
            for row in reports
            if case["stock_code"] in row["name"]
            for match in [re.search(r"__(20\d{2})年__年度报告\.pdf$", row["name"])]
            if match
        }
        assert set(case["years"]).issubset(available_years)
    assert manifest["constructed_extension"]["status"] == (
        "construction-rule-frozen-cases-not-yet-collected"
    )


def test_first_cninfo_development_case_has_two_verified_official_pdfs():
    acquisition = json.loads(
        (CHINA / "china_document_acquisition_v1.json").read_text(encoding="utf-8")
    )
    assert acquisition["status"] == "partial-development-documents-verified"
    assert acquisition["case_check"]["qid"] == 941
    assert acquisition["case_check"]["gold_document_closure"]
    assert acquisition["case_check"]["matches_public_finglm_gold"]
    assert acquisition["case_check"]["same"] is False
    assert {doc["report_year"] for doc in acquisition["documents"]} == {2019, 2021}
    for document in acquisition["documents"]:
        path = CHINA / document["local_file"]
        assert path.exists()
        assert path.read_bytes().startswith(b"%PDF-")
        assert path.stat().st_size == document["bytes"]
        assert sha256(path) == document["sha256"]
        assert document["visual_verification"] == "passed"


def test_all_frozen_china_pilot_documents_are_downloaded_from_cninfo():
    acquisition = json.loads(
        (CHINA / "china_document_acquisition_v2.json").read_text(encoding="utf-8")
    )
    assert acquisition["requested_cases"] == 9
    assert acquisition["requested_documents"] == 18
    assert acquisition["downloaded_documents"] == 18
    assert acquisition["failures"] == []
    assert {document["qid"] for document in acquisition["documents"]} == {
        357,
        941,
        1178,
        1221,
        1493,
        1626,
        1863,
        1958,
        1976,
    }
    for document in acquisition["documents"]:
        path = CHINA / document["local_file"]
        assert path.exists()
        assert path.stat().st_size == document["bytes"]
        assert path.read_bytes().startswith(b"%PDF-")
        assert sha256(path) == document["sha256"]
        assert document["download_status"] == "success"
    alias_rows = [
        item for item in acquisition["documents"] if item["security_code"] == "900943"
    ]
    assert len(alias_rows) == 2
    assert {item["source_query_code"] for item in alias_rows} == {"600272"}
