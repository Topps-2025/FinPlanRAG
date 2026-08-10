"""Audit the locally pinned public Chinese financial QA metadata.

The script deliberately separates repository availability from benchmark
completeness.  In particular, FinGLM2 publishes questions and a database
schema, but not the competition database or gold answers needed for a fully
reproducible RAG benchmark.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .storage_paths import DATA_ROOT, RESULTS_ROOT
except ImportError:  # direct script execution
    from storage_paths import DATA_ROOT, RESULTS_ROOT


ROOT = Path(__file__).resolve().parent
DATA = DATA_ROOT / "external" / "china"
OUTPUT = RESULTS_ROOT / "china_public_data_audit_v1.json"

FINGLM_COMMIT = "ff48a9dc5e6c7f9d9a9c6120024dccefd8cbaa80"
FINGLM2_COMMIT = "39cf08c54128e641eef3284f2b68d6d4c0fadb5d"

REPORT_PATTERN = re.compile(
    r"^(?P<disclosure_date>\d{4}-\d{2}-\d{2})__"
    r"(?P<company>.+?)__(?P<code>\d{6})__(?P<short_name>.+?)__"
    r"(?P<report_year>\d{4})年__年度报告\.pdf$"
)
COMPARISON_PATTERN = re.compile(
    r"相比|与\s*20\d{2}年|增长率|较\s*20\d{2}年|是否相同|发生不同"
)
CONTEXT_REFERENCE_PATTERN = re.compile(
    r"^(该公司|该股|该行业|这个概念|以上|上述|他|其)|分别是|上一年度"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_finglm() -> dict:
    question_path = DATA / "finglm_A_questions.json"
    answer_path = DATA / "finglm_A_answers.json"
    report_path = DATA / "finglm_reports_list.csv"

    questions = load_jsonl(question_path)
    answers = load_jsonl(answer_path)
    with report_path.open(encoding="utf-8-sig", newline="") as handle:
        report_rows = list(csv.DictReader(handle))

    parsed_reports = []
    parse_failures = []
    company_years: dict[str, set[int]] = defaultdict(set)
    for row in report_rows:
        match = REPORT_PATTERN.match(row["name"])
        if not match:
            parse_failures.append(row["name"])
            continue
        item = match.groupdict()
        item["report_year"] = int(item["report_year"])
        parsed_reports.append(item)
        company_years[item["code"]].add(item["report_year"])

    type_counts = Counter(answer.get("type", "missing") for answer in answers)
    year_counts = Counter(
        answer.get("prompt", {}).get("year")
        for answer in answers
        if answer.get("prompt", {}).get("year")
    )
    cross_period_candidates = [
        answer
        for answer in answers
        if COMPARISON_PATTERN.search(answer.get("question", ""))
    ]
    explicit_xiangbi = [
        answer for answer in answers if "相比" in answer.get("question", "")
    ]

    return {
        "name": "MetaGLM/FinGLM",
        "source_commit": FINGLM_COMMIT,
        "license_status": {
            "spdx": "NOASSERTION",
            "github_api_detected_license": None,
            "readme_terms": "相关资源仅供研究、交流使用；商业使用风险自负",
            "decision": "仅按学术研究数据使用，不写成 Apache/MIT 许可",
        },
        "local_files": {
            question_path.name: sha256(question_path),
            answer_path.name: sha256(answer_path),
            report_path.name: sha256(report_path),
        },
        "qa": {
            "round": "复赛A",
            "questions": len(questions),
            "answers": len(answers),
            "ids_aligned": [item["id"] for item in questions]
            == [item["id"] for item in answers],
            "type_counts": dict(sorted(type_counts.items())),
            "prompt_year_counts": dict(sorted(year_counts.items())),
            "surface_cross_period_candidates": len(cross_period_candidates),
            "explicit_xiangbi_questions": len(explicit_xiangbi),
            "explicit_xiangbi_ids": [item["id"] for item in explicit_xiangbi],
            "gold_answer_variants": True,
            "gold_evidence_spans": False,
            "gold_document_sets": False,
        },
        "reports": {
            "rows": len(report_rows),
            "parsed_rows": len(parsed_reports),
            "parse_failures": parse_failures,
            "unique_security_codes": len(company_years),
            "report_year_counts": dict(
                sorted(Counter(item["report_year"] for item in parsed_reports).items())
            ),
            "security_codes_with_at_least_two_years": sum(
                len(years) >= 2 for years in company_years.values()
            ),
            "security_codes_with_2019_2020_2021": sum(
                {2019, 2020, 2021}.issubset(years)
                for years in company_years.values()
            ),
            "document_types": ["年度报告"],
            "quarterly_or_event_announcements": False,
            "availability_caveat": (
                "文件名含公告日期，但没有交易所级精确发布时间；"
                "正式点时实验须回到官方公告页核验 available_at"
            ),
        },
        "fit": {
            "immediate": [
                "中文年报 reader 与检索",
                "跨年财务指标和法定代表人比较",
                "公司—报告年度义务规划",
            ],
            "not_sufficient_for": [
                "年报/半年报/季报混合期间规划",
                "交易所问询函—回复、并购进展和限制证据",
                "严格截止时点回放",
            ],
        },
    }


def audit_finglm2() -> dict:
    question_path = DATA / "finglm2_questions.json"
    schema_path = DATA / "finglm2_schema_api.txt"
    groups = load_json(question_path)
    subquestions = [question for group in groups for question in group["team"]]
    contextual_groups = []
    for group in groups:
        later_questions = [item["question"] for item in group["team"][1:]]
        if any(CONTEXT_REFERENCE_PATTERN.search(text) for text in later_questions):
            contextual_groups.append(group["tid"])

    return {
        "name": "MetaGLM/FinGLM2",
        "source_commit": FINGLM2_COMMIT,
        "license_status": {
            "spdx": "Apache-2.0",
            "scope_caveat": "仓库说明：无特殊说明或额外协议的内容使用 Apache-2.0",
        },
        "local_files": {
            question_path.name: sha256(question_path),
            schema_path.name: sha256(schema_path),
        },
        "questions": {
            "groups": len(groups),
            "subquestions": len(subquestions),
            "group_length_counts": dict(
                sorted(Counter(len(group["team"]) for group in groups).items())
            ),
            "context_reference_groups": len(contextual_groups),
            "context_reference_group_ids": contextual_groups,
            "multi_turn": True,
            "gold_answers_in_repository": False,
        },
        "data_availability": {
            "question_file": True,
            "database_schema": True,
            "competition_database": False,
            "document_corpus": False,
            "evidence_spans": False,
            "available_at": False,
            "classification": "questions-and-schema-only",
        },
        "fit": {
            "immediate": [
                "中文链式问题规划诊断",
                "指代消解和跨子问题状态保持",
                "SQL/工具调用规划模板设计",
            ],
            "not_sufficient_for": [
                "可直接评分的完整 RAG benchmark",
                "文档检索闭包",
                "点时证据和拒答评价",
            ],
        },
    }


def main() -> None:
    acquisition_path = DATA / "china_document_acquisition_v2.json"
    if not acquisition_path.exists():
        acquisition_path = DATA / "china_document_acquisition_v1.json"
    acquisition = load_json(acquisition_path) if acquisition_path.exists() else None
    audit = {
        "schema": "finplan-china-public-data-audit.v1",
        "status": "source-grounded-audit-not-benchmark-result",
        "audited_at": "2026-08-10",
        "sources": [audit_finglm(), audit_finglm2()],
        "access_log": {
            "github_api": "success",
            "modelscope_git_index": {
                "status": "success",
                "commit": "0d1da4119ba6f84124c5d71d1275c9502dbdb71c",
                "contains": "11,588 PDF LFS paths",
            },
            "modelscope_search_api": "HTTP 500 in current environment",
            "google_drive": "unreachable in current environment",
            "huggingface_narrow_search": "no verified CFinQA/FinGLM mirror found",
            "cninfo_official_api": {
                "status": "success" if acquisition else "not-yet-run",
                "verified_documents": len(acquisition["documents"]) if acquisition else 0,
                "verified_development_qids": (
                    sorted({item["qid"] for item in acquisition["documents"]})
                    if acquisition
                    else []
                ),
            },
        },
        "decision": {
            "public_benchmark_track": (
                "以 FinGLM 年报和人工问答建立中文 reader、跨年义务规划 pilot"
            ),
            "constructed_track": (
                "从巨潮资讯、上交所、深交所公开披露补建半年报/季报、"
                "问询函—回复、并购进展、处罚整改和截止时点样本"
            ),
            "separation_rule": (
                "公开原题与自行构建题分表报告；单年报 QA 不冒充多文档或点时闭包"
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
