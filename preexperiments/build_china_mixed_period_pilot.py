"""Build the page-level China mixed-period pilot (H1/Q3/FY period collision).

Data sources (all already downloaded and SHA-verified):
- ``china_mixed_period_acquisition_v1.json``: 10 CNINFO H1/Q3 PDFs
- ``china_document_acquisition_v2.json``: 18 verified annual report PDFs
- ``china_mixed_period_pilot_manifest_v1.json``: frozen 5-case protocol

Each frozen case asks whether the legal representative at two specified
reporting periods (e.g. Q3 vs FY, or H1 vs FY) of the *same* company and
fiscal year is the same.  The same-company same-year third period document
is deliberately present in the corpus as a same-company distractor, so a
``security_code x fiscal_year`` obligation key is NOT sufficient.
"""

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
MANIFEST = ROOT / "china_mixed_period_pilot_manifest_v1.json"
MIXED_ACQUISITION = CHINA / "china_mixed_period_acquisition_v1.json"
ANNUAL_ACQUISITION = CHINA / "china_document_acquisition_v2.json"
ANSWERS = CHINA / "finglm_A_answers.json"
OUTPUT = DATA_ROOT / "china_mixed_period_page_pilot_v1.json"

# Legal-representative extraction.  PDF text may contain spaces inside
# names, so whitespace is removed before matching.  Names may be Chinese
# (2-4 chars) or Latin (e.g. "WangXinglong" in CNINFO Q3 statements).
# A non-greedy match is anchored by the terminator that follows the name:
# the job title "主管", "日期", "注册", "成立", section numbers "二、", or
# punctuation.  Common non-name words ("签名", "成立", "注册", ...) are
# blacklisted and the most frequent candidate wins the vote.
LEGAL_REPRESENTATIVE_PATTERN = re.compile(
    r"法定代表人\s*[：:]?\s*为?\s*"
    r"(?P<name>[一-鿿A-Za-z]{2,12}?)"
    r"(?=(?:主管|日期|注册|成立|签名|的?《|[一二三四五六七八九十]、|[，。；、,;。]))"
)
NAME_BLACKLIST = {"签名", "成立", "注册", "负责人", "姓名", "职务", "说明", "董事长", "总经理"}
VALUE_KEY = re.compile(r"^(20\d{2})年法定代表人$")

NAME_VARIANTS = str.maketrans({"贇": "赟"})
PERIOD_LABELS = {"H1": "半年度报告", "Q3": "第三季度报告", "FY": "年度报告"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).translate(NAME_VARIANTS).casefold()
    return re.sub(r"[^0-9a-z一-鿿]+", "", value)


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


def extract_legal_representative(pages: list[dict]) -> dict:
    """Extract the legal representative field value from a report.

    Counts normalized matches of ``法定代表人：<name>`` across all pages;
    blacklisted non-name words are dropped; the most frequent candidate
    wins.  Candidate evidence pages are pages containing both the keyword
    and the winning value.  If no value can be found at all the result is
    marked ``unverified`` so the builder does not silently fabricate a gold
    label (a Q3 report may legitimately omit the field).
    """
    counts: dict[str, int] = {}
    matches_per_page: dict[int, list[str]] = {}
    for page in pages:
        text = re.sub(r"\s+", "", str(page["text"]))
        matches = [
            canonical_name(value)
            for value in LEGAL_REPRESENTATIVE_PATTERN.findall(text)
            if value not in NAME_BLACKLIST
        ]
        if matches:
            matches_per_page[int(page["page_number"])] = matches
            for value in matches:
                counts[value] = counts.get(value, 0) + 1
    if not counts:
        return {"status": "unverified", "value": None, "candidate_pages": []}
    winner = max(counts, key=lambda key: (counts[key], key))
    candidates = [
        page_number
        for page_number, matches in matches_per_page.items()
        if winner in matches
    ]
    # Prefer the formal "company info" page: pages whose text mentions both
    # "公司" (in a section context) and the legal-representative keyword.
    preferred = [
        page_number
        for page_number in candidates
        if "法定代表人" in re.sub(r"\s+", "", str(pages[page_number - 1]["text"]))
        and "公司" in re.sub(r"\s+", "", str(pages[page_number - 1]["text"]))
    ]
    return {
        "status": "verified",
        "value": winner,
        "candidate_pages": sorted(candidates),
        "primary_page": (preferred or sorted(candidates))[0],
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mixed_acquisition = json.loads(MIXED_ACQUISITION.read_text(encoding="utf-8"))
    annual_acquisition = json.loads(ANNUAL_ACQUISITION.read_text(encoding="utf-8"))
    answer_map = {item["id"]: item for item in load_jsonl(ANSWERS)}

    case_map = {item["qid"]: item for item in manifest["cases"]}

    # --- 1. Collect the 10 H1/Q3 documents -------------------------------
    period_docs: list[dict] = []
    for source in sorted(
        mixed_acquisition["documents"],
        key=lambda item: (item["qid"], item["fiscal_period"]),
    ):
        path = CHINA / source["local_file"]
        if sha256(path) != source["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        pages = extract_pages(path)
        doc_id = f"{source['security_code']}_{source['fiscal_year']}_{source['fiscal_period']}"
        period_docs.append(
            {
                "doc_id": doc_id,
                "qid": source["qid"],
                "security_code": source["security_code"],
                "company": source["company"],
                "fiscal_year": int(source["fiscal_year"]),
                "fiscal_period": source["fiscal_period"],
                "document_type": source["document_type"],
                "announcement_date": source["announcement_date"],
                "source_url": source["source_url"],
                "pdf_sha256": source["sha256"],
                "page_count": len(pages),
                "pages": pages,
            }
        )

    # --- 2. Collect the 5 target annual reports --------------------------
    target_years = {
        (case["stock_code"], case["year"]) for case in case_map.values()
    }
    annual_docs: list[dict] = []
    for source in annual_acquisition["documents"]:
        key = (source["security_code"], int(source["report_year"]))
        if key not in target_years:
            continue
        path = CHINA / source["local_file"]
        if sha256(path) != source["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        pages = extract_pages(path)
        doc_id = f"{source['security_code']}_{source['report_year']}_FY"
        annual_docs.append(
            {
                "doc_id": doc_id,
                "qid": source["qid"],
                "security_code": source["security_code"],
                "company": source["company"],
                "fiscal_year": int(source["report_year"]),
                "fiscal_period": "FY",
                "document_type": source["document_type"],
                "announcement_date": source["announcement_date"],
                "source_url": source["source_url"],
                "pdf_sha256": source["sha256"],
                "page_count": len(pages),
                "pages": pages,
            }
        )

    documents = period_docs + annual_docs
    doc_by_key: dict[tuple[str, int, str], dict] = {
        (doc["security_code"], doc["fiscal_year"], doc["fiscal_period"]): doc
        for doc in documents
    }
    # Every company/year must have all three period documents available.
    missing = []
    for case in case_map.values():
        for period in ("H1", "Q3", "FY"):
            if (case["stock_code"], case["year"], period) not in doc_by_key:
                missing.append((case["qid"], case["stock_code"], case["year"], period))
    if missing:
        raise ValueError(f"Missing period documents: {missing}")

    # --- 3. Build per-case gold evidence ---------------------------------
    cases = []
    extraction_audit = {}
    for qid in sorted(case_map):
        manifest_case = case_map[qid]
        finglm_qid = int(qid.removeprefix("cnmix-"))
        answer = answer_map[finglm_qid]
        fininglm_years = {
            int(match.group(1)): str(value)
            for key, value in answer.get("prompt", {}).items()
            if (match := VALUE_KEY.match(str(key)))
        }
        gold_evidence = []
        extraction_audit[qid] = {}
        for period in manifest_case["periods"]:
            document = doc_by_key[
                (manifest_case["stock_code"], manifest_case["year"], period)
            ]
            extracted = extract_legal_representative(document["pages"])
            evidence = {
                "period": period,
                "doc_id": document["doc_id"],
                "candidate_pages": extracted["candidate_pages"],
                "primary_page": extracted.get("primary_page"),
                "document_value": extracted["value"],
                "extraction_status": extracted["status"],
            }
            if period == "FY" and fininglm_years:
                gold_value = canonical_name(fininglm_years[manifest_case["year"]])
                evidence["finglm_gold_value"] = gold_value
                evidence["value_consistent_with_finglm"] = (
                    extracted["status"] == "verified"
                    and extracted["value"] == gold_value
                )
            gold_evidence.append(evidence)
            extraction_audit[qid][period] = {
                "document_value": extracted["value"],
                "candidate_pages": extracted["candidate_pages"],
                "primary_page": extracted.get("primary_page"),
            }

        # Semantic gold: are the legal representatives the same across the
        # two target periods?  Both values must be verified (not fabricated).
        verified = [
            item["document_value"]
            for item in gold_evidence
            if item["extraction_status"] == "verified"
        ]
        gold_values = [item["document_value"] for item in gold_evidence]
        gold_available = len(verified) == len(gold_evidence)
        gold_available = gold_available and all(verified)
        normalized = [canonical_name(value) for value in verified]
        semantic_same = bool(gold_available and len(set(normalized)) == 1)
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
                "fiscal_year": manifest_case["year"],
                "periods": manifest_case["periods"],
                "gold_doc_ids": [
                    doc_by_key[
                        (manifest_case["stock_code"], manifest_case["year"], period)
                    ]["doc_id"]
                    for period in manifest_case["periods"]
                ],
                "gold_evidence": gold_evidence,
                "evidence_complete": all(
                    item["extraction_status"] == "verified"
                    and item["primary_page"] is not None
                    for item in gold_evidence
                ),
                "normalized_semantic_same": semantic_same,
                "semantic_gold_available": gold_available,
            }
        )

    result = {
        "schema": "finplan-china-mixed-period-page-pilot.v1",
        "status": "page-corpus-built-before-method-run",
        "protocol": {
            "manifest_sha256": sha256(MANIFEST),
            "mixed_acquisition_sha256": sha256(MIXED_ACQUISITION),
            "annual_acquisition_sha256": sha256(ANNUAL_ACQUISITION),
            "builder_sha256": sha256(Path(__file__)),
            "cases": len(cases),
            "documents": len(documents),
            "pages": sum(document["page_count"] for document in documents),
            "period_distractor_rule": (
                "each case corpus contains the same-company same-year H1, Q3 and FY "
                "reports; only the two manifest-specified periods are gold obligations"
            ),
            "gold_fields_hidden_from_methods": [
                "gold_doc_ids",
                "gold_evidence",
                "normalized_semantic_same",
                "semantic_gold_available",
            ],
        },
        "extraction_audit": extraction_audit,
        "documents": documents,
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "documents": len(documents),
                "pages": result["protocol"]["pages"],
                "evidence_complete": sum(case["evidence_complete"] for case in cases),
                "semantic_gold_available": sum(
                    case["semantic_gold_available"] for case in cases
                ),
                "extraction_audit": extraction_audit,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
