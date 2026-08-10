"""Build a page-level Chinese FinGLM pilot from verified CNINFO PDFs."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pypdfium2 as pdfium

try:
    from .storage_paths import DATA_ROOT
except ImportError:  # direct script execution
    from storage_paths import DATA_ROOT


ROOT = Path(__file__).resolve().parent
CHINA = DATA_ROOT / "external" / "china"
MANIFEST = ROOT / "china_public_data_pilot_manifest_v1.json"
ACQUISITION = CHINA / "china_document_acquisition_v2.json"
ANSWERS = CHINA / "finglm_A_answers.json"
OUTPUT = DATA_ROOT / "china_finglm_page_pilot_v1.json"

NAME_VARIANTS = str.maketrans({"贇": "赟"})
VALUE_KEY = re.compile(r"^(20\d{2})年法定代表人$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).translate(NAME_VARIANTS).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_pages(path: Path) -> list[dict]:
    pdf = pdfium.PdfDocument(path)
    pages = []
    for index in range(len(pdf)):
        text = pdf[index].get_textpage().get_text_range()
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        pages.append(
            {
                "page_number": index + 1,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return pages


def expected_values(answer: dict) -> dict[int, str]:
    values = {}
    for key, value in answer.get("prompt", {}).items():
        match = VALUE_KEY.match(str(key))
        if match:
            values[int(match.group(1))] = str(value)
    return values


def evidence_pages(pages: list[dict], expected: str) -> list[int]:
    canonical_expected = canonical_name(expected)
    candidates = []
    for page in pages:
        text = str(page["text"])
        if "法定代表人" not in text:
            continue
        if expected in text or canonical_expected in canonical_name(text):
            candidates.append(int(page["page_number"]))
    return candidates


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    acquisition = json.loads(ACQUISITION.read_text(encoding="utf-8"))
    answer_map = {item["id"]: item for item in load_jsonl(ANSWERS)}
    case_map = {item["qid"]: item for item in manifest["cases"]}

    documents = []
    document_map: dict[tuple[int, int], dict] = {}
    for source in sorted(
        acquisition["documents"], key=lambda item: (item["qid"], item["report_year"])
    ):
        path = CHINA / source["local_file"]
        if sha256(path) != source["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        pages = extract_pages(path)
        doc_id = f"{source['security_code']}_{source['report_year']}_AR"
        document = {
            "doc_id": doc_id,
            "qid": source["qid"],
            "security_code": source["security_code"],
            "source_query_code": source["source_query_code"],
            "company": source["company"],
            "report_year": source["report_year"],
            "document_type": source["document_type"],
            "announcement_date": source["announcement_date"],
            "source_url": source["source_url"],
            "pdf_sha256": source["sha256"],
            "page_count": len(pages),
            "pages": pages,
        }
        documents.append(document)
        document_map[(int(source["qid"]), int(source["report_year"]))] = document

    cases = []
    for qid in sorted(case_map):
        manifest_case = case_map[qid]
        answer = answer_map[qid]
        values = expected_values(answer)
        if set(values) != set(manifest_case["years"]):
            raise ValueError(f"Gold years do not match manifest for qid {qid}")
        evidence = []
        for year in manifest_case["years"]:
            document = document_map[(qid, year)]
            candidates = evidence_pages(document["pages"], values[year])
            preferred = [
                page["page_number"]
                for page in document["pages"]
                if page["page_number"] in candidates
                and "公司的法定代表人" in page["text"]
            ]
            evidence.append(
                {
                    "year": year,
                    "doc_id": document["doc_id"],
                    "expected_value": values[year],
                    "candidate_pages": candidates,
                    "primary_page": (preferred or candidates or [None])[0],
                }
            )

        ordered_values = [values[year] for year in manifest_case["years"]]
        normalized_values = [canonical_name(value) for value in ordered_values]
        normalized_same = len(set(normalized_values)) == 1
        raw_same = answer.get("prompt", {}).get("prom_answer") == "相同"
        cases.append(
            {
                "qid": qid,
                "split": manifest_case["split"],
                "question": manifest_case["question"],
                "company": manifest_case["company"],
                "security_code": manifest_case["stock_code"],
                "aliases": sorted(
                    {
                        manifest_case["company"],
                        str(answer.get("prompt", {}).get("ent_short_name", "")),
                        manifest_case["stock_code"],
                    }
                    - {""},
                    key=len,
                    reverse=True,
                ),
                "years": manifest_case["years"],
                "gold_doc_ids": [document_map[(qid, year)]["doc_id"] for year in manifest_case["years"]],
                "gold_evidence": evidence,
                "evidence_complete": all(item["primary_page"] is not None for item in evidence),
                "raw_finglm_values": ordered_values,
                "normalized_values": normalized_values,
                "raw_finglm_same": raw_same,
                "normalized_semantic_same": normalized_same,
                "label_audit_status": (
                    "normalization-conflict" if raw_same != normalized_same else "consistent"
                ),
            }
        )

    result = {
        "schema": "finplan-china-finglm-page-pilot.v1",
        "status": "page-corpus-built-before-method-run",
        "protocol": {
            "manifest_sha256": sha256(MANIFEST),
            "acquisition_sha256": sha256(ACQUISITION),
            "builder_sha256": sha256(Path(__file__)),
            "cases": len(cases),
            "documents": len(documents),
            "pages": sum(document["page_count"] for document in documents),
            "gold_values_used_only_for_evidence_scoring": True,
            "method_results_seen_before_build": False,
        },
        "label_audit": {
            "normalization_conflicts": [
                case["qid"] for case in cases if case["label_audit_status"] == "normalization-conflict"
            ],
            "rule": "NFKC + casefold + punctuation/space removal + 贇→赟",
            "reporting": "raw FinGLM and normalized semantic labels are reported separately",
        },
        "documents": documents,
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "documents": len(documents),
                "pages": result["protocol"]["pages"],
                "evidence_complete": sum(case["evidence_complete"] for case in cases),
                "normalization_conflicts": result["label_audit"]["normalization_conflicts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
