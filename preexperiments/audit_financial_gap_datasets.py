"""Audit public financial benchmarks for multi-evidence and planning demand.

The audit reports observable dataset properties only. It does not infer that a
specific method fails merely because questions require multiple evidence
items. SEC replay results supply the direct method-failure diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Sequence

try:
    from .storage_paths import DATA_ROOT, RESULTS_ROOT
except ImportError:
    from storage_paths import DATA_ROOT, RESULTS_ROOT


LOFIN_TEMPLATE = {
    "finqa_test.jsonl": "accounting_derivation",
    "numeric_table_test.jsonl": "accounting_derivation",
    "numeric_text_test.jsonl": "accounting_derivation",
    "secqa_test.jsonl": "point_in_time_lookup",
    "textual_test.jsonl": "comparative_investigation",
}
FINSEARCH_TEMPLATE = {
    "T1": "point_in_time_lookup",
    "T2": "point_in_time_lookup",
    "T3": "comparative_investigation",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> List[Mapping[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_lofin(root: Path) -> Mapping[str, object]:
    subsets: Dict[str, object] = {}
    template_counts: Counter[str] = Counter()
    total = 0
    all_evidence_counts: List[int] = []
    all_multi_doc: List[bool] = []
    all_multi_page: List[bool] = []
    for name, template in LOFIN_TEMPLATE.items():
        path = root / name
        rows = read_jsonl(path)
        evidence_counts = [len(row.get("evidences", [])) for row in rows]
        multi_doc: List[bool] = []
        multi_page: List[bool] = []
        for row in rows:
            evidences = row.get("evidences", [])
            docs = {str(item.get("doc_name", "")) for item in evidences}
            pages = {(str(item.get("doc_name", "")), str(item.get("page_num", ""))) for item in evidences}
            multi_doc.append(len(docs) > 1)
            multi_page.append(len(pages) > 1)
        subsets[name] = {
            "questions": len(rows),
            "template": template,
            "mean_evidence_items": mean(evidence_counts) if evidence_counts else 0.0,
            "multi_evidence_rate": mean([n > 1 for n in evidence_counts]) if evidence_counts else 0.0,
            "multi_page_rate": mean(multi_page) if multi_page else 0.0,
            "multi_document_rate": mean(multi_doc) if multi_doc else 0.0,
            "sha256": sha256(path),
        }
        total += len(rows)
        template_counts[template] += len(rows)
        all_evidence_counts.extend(evidence_counts)
        all_multi_doc.extend(multi_doc)
        all_multi_page.extend(multi_page)
    return {
        "questions": total,
        "subsets": subsets,
        "template_counts": dict(template_counts),
        "overall": {
            "mean_evidence_items": mean(all_evidence_counts),
            "multi_evidence_rate": mean([n > 1 for n in all_evidence_counts]),
            "multi_page_rate": mean(all_multi_page),
            "multi_document_rate": mean(all_multi_doc),
        },
    }


def audit_finsearch(path: Path) -> Mapping[str, object]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    tier_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    explicit_time = 0
    comparative = 0
    for row in rows:
        match = re.match(r"\((T\d)\)", str(row["prompt_id"]))
        tier = match.group(1) if match else "unknown"
        tier_counts[tier] += 1
        template_counts[FINSEARCH_TEMPLATE.get(tier, "unmapped")] += 1
        prompt = str(row["prompt"])
        if re.search(r"20\d{2}|截至|历史|过去|当日|最新|as of|histor", prompt, re.I):
            explicit_time += 1
        if re.search(r"比较|相比|分别|排名|差异|各自|versus|compare|difference", prompt, re.I):
            comparative += 1
    return {
        "questions": len(rows),
        "tier_counts": dict(tier_counts),
        "tier_meaning": {
            "T1": "Time-Sensitive Data Fetching",
            "T2": "Simple Historical Lookup",
            "T3": "Complex Historical Investigation",
        },
        "template_counts": dict(template_counts),
        "explicit_time_expression_rate": explicit_time / len(rows),
        "surface_comparison_cue_rate": comparative / len(rows),
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lofin", default=DATA_ROOT / "external" / "lofin")
    parser.add_argument("--finsearch", default=DATA_ROOT / "external" / "finsearchcomp" / "finsearchcomp_data.json")
    parser.add_argument("--templates", default="preexperiments/financial_task_templates.json")
    parser.add_argument("--sec-results", default=RESULTS_ROOT / "sec_real_pilot_v1.json")
    parser.add_argument("--out", default=RESULTS_ROOT / "financial_gap_audit_v1.json")
    args = parser.parse_args()

    template_path = Path(args.templates)
    registry = json.loads(template_path.read_text(encoding="utf-8"))
    sec = json.loads(Path(args.sec_results).read_text(encoding="utf-8"))
    result = {
        "schema": "finplan-financial-gap-audit.v1",
        "status": "observable-dataset-properties-and-small-sec-method-diagnostic",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "template_registry_sha256": sha256(template_path),
        "template_count": len(registry["templates"]),
        "lofin": audit_lofin(Path(args.lofin)),
        "finsearchcomp": audit_finsearch(Path(args.finsearch)),
        "sec_direct_diagnostic": {
            "cases": sec["protocol"]["cases"],
            "lineages": sec["protocol"]["lineages"],
            "single_shot_wrong_lineage": sec["summary"]["single_shot"]["wrong_lineage"],
            "no_time_future_leak": sec["summary"]["finplan_no_time"]["future_leak"],
            "no_time_overclaim": sec["summary"]["finplan_no_time"]["overclaim"],
            "finplan_v1_accuracy": sec["summary"]["finplan_v1"]["accuracy"],
            "strongest_generic_accuracy": max(sec["summary"]["generic_adaptive"]["accuracy"], sec["summary"]["hirec_style"]["accuracy"]),
            "finplan_v3_accuracy": sec["summary"]["finplan_v3_path_bound"]["accuracy"],
        },
        "claim_boundary": "LOFin and FinSearchComp establish task demand; only the SEC replay supplies a direct controlled method-failure diagnostic. None of these small diagnostics establishes SOTA.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
