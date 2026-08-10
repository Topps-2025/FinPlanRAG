"""Run transparent Chinese page/file planning baselines on the frozen pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    from .storage_paths import DATA_ROOT, RESULTS_ROOT
except ImportError:  # direct script execution
    from storage_paths import DATA_ROOT, RESULTS_ROOT


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = DATA_ROOT / "china_finglm_page_pilot_v1.json"
DEFAULT_OUT = RESULTS_ROOT / "china_finglm_page_pilot_v1.json"

METHODS = (
    "single_shot",
    "fixed_decomposition",
    "metadata_decomposition",
    "generic_adaptive_period",
    "hirec_period",
    "finplan_v1_entity_only",
    "finplan_entity_year_no_path",
    "finplan_entity_year",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", "", value)


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    result = re.findall(r"[a-z0-9]+", text)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
        result.extend(sequence)
        result.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return result


class BM25:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.term_counts = [Counter(chunk["terms"]) for chunk in chunks]
        self.lengths = [sum(counter.values()) for counter in self.term_counts]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency = Counter()
        for counter in self.term_counts:
            document_frequency.update(counter.keys())
        total = len(chunks)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        *,
        security_code: str | None = None,
        report_year: int | None = None,
        excluded_docs: set[str] | None = None,
        limit: int = 200,
    ) -> list[tuple[float, dict]]:
        query_terms = Counter(tokens(query))
        excluded_docs = excluded_docs or set()
        ranked = []
        k1 = 1.2
        b = 0.75
        for index, chunk in enumerate(self.chunks):
            if chunk["doc_id"] in excluded_docs:
                continue
            if security_code and chunk["security_code"] != security_code:
                continue
            if report_year is not None and chunk["report_year"] != report_year:
                continue
            score = 0.0
            counter = self.term_counts[index]
            length = self.lengths[index]
            for term, query_frequency in query_terms.items():
                frequency = counter.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + k1 * (1 - b + b * length / max(1, self.avg_length))
                score += self.idf.get(term, 0.0) * frequency * (k1 + 1) / denominator * query_frequency
            if score > 0:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1]["doc_id"], item[1]["page_number"]))
        return ranked[:limit]


def build_chunks(data: dict) -> list[dict]:
    chunks = []
    for document in data["documents"]:
        for page in document["pages"]:
            chunks.append(
                {
                    "chunk_id": f"{document['doc_id']}::p{page['page_number']}",
                    "doc_id": document["doc_id"],
                    "security_code": document["security_code"],
                    "report_year": int(document["report_year"]),
                    "page_number": int(page["page_number"]),
                    "terms": tokens(str(page["text"])),
                }
            )
    return chunks


def registry_plan(question: str, cases: Iterable[dict], documents: list[dict]) -> tuple[list[str], list[dict]]:
    normalized = canonical(question)
    matched_codes = []
    for case in cases:
        if any(canonical(alias) in normalized for alias in case["aliases"] if len(canonical(alias)) >= 3):
            matched_codes.append(case["security_code"])
    matched_codes = sorted(set(matched_codes))
    requested_years = sorted({int(year) for year in re.findall(r"20\d{2}", question)})
    available = {(document["security_code"], int(document["report_year"])) for document in documents}
    obligations = [
        {"security_code": code, "report_year": year, "doc_id": f"{code}_{year}_AR"}
        for code in matched_codes
        for year in requested_years
        if (code, year) in available
    ]
    return matched_codes, obligations


def run_method(case: dict, cases: list[dict], documents: list[dict], index: BM25, budget: int, method: str) -> dict:
    entity_candidates, obligations = registry_plan(case["question"], cases, documents)
    used: list[dict] = []
    actions: list[str] = []

    def retrieve(
        query: str,
        label: str,
        *,
        code: str | None = None,
        year: int | None = None,
        k_docs: int = 1,
    ) -> None:
        remaining = budget - len({item["doc_id"] for item in used})
        if remaining <= 0:
            return
        ranked = index.search(
            query,
            security_code=code,
            report_year=year,
            excluded_docs={item["doc_id"] for item in used},
        )
        selected_docs = set()
        for score, chunk in ranked:
            if chunk["doc_id"] in selected_docs:
                continue
            used.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "page_number": chunk["page_number"],
                    "score": score,
                }
            )
            selected_docs.add(chunk["doc_id"])
            if len(selected_docs) >= min(k_docs, remaining):
                break
        actions.append(label)

    question = case["question"]
    if method == "single_shot":
        retrieve(question, "initial-query", k_docs=budget)
    elif method == "fixed_decomposition":
        for year in sorted({int(year) for year in re.findall(r"20\d{2}", question)}):
            retrieve(f"{question} {year}年 年度报告 法定代表人", f"year:{year}")
    elif method == "metadata_decomposition":
        for obligation in obligations:
            retrieve(
                f"{question} {obligation['security_code']} {obligation['report_year']}年 法定代表人",
                f"obligation:{obligation['doc_id']}",
                code=obligation["security_code"],
                year=obligation["report_year"],
            )
    elif method == "generic_adaptive_period":
        retrieve(question, "initial-query")
        observed = {item["doc_id"] for item in used}
        for obligation in obligations:
            if obligation["doc_id"] not in observed:
                retrieve(
                    f"{question} 补充检索 {obligation['report_year']}年 年度报告",
                    f"missing:{obligation['doc_id']}",
                    code=obligation["security_code"],
                    year=obligation["report_year"],
                )
                observed = {item["doc_id"] for item in used}
    elif method == "hirec_period":
        retrieve(question, "initial-evidence", k_docs=2)
        observed = {item["doc_id"] for item in used}
        for obligation in obligations:
            if obligation["doc_id"] not in observed:
                retrieve(
                    f"{question} 还缺少什么证据 {obligation['report_year']}年 法定代表人",
                    f"complementary:{obligation['doc_id']}",
                    code=obligation["security_code"],
                    year=obligation["report_year"],
                )
                observed = {item["doc_id"] for item in used}
    elif method == "finplan_v1_entity_only":
        if entity_candidates:
            latest = max(case["years"])
            retrieve(
                f"{question} {entity_candidates[0]} {latest}年 法定代表人",
                f"entity-only:{entity_candidates[0]}",
                code=entity_candidates[0],
                year=latest,
            )
    elif method == "finplan_entity_year_no_path":
        for obligation in obligations:
            retrieve(
                f"{question} {obligation['security_code']} {obligation['report_year']}年 法定代表人",
                f"unbound:{obligation['doc_id']}",
            )
    elif method == "finplan_entity_year":
        for obligation in obligations:
            retrieve(
                f"{question} {obligation['security_code']} {obligation['report_year']}年 法定代表人",
                f"path:{obligation['doc_id']}",
                code=obligation["security_code"],
                year=obligation["report_year"],
            )
    else:  # pragma: no cover
        raise ValueError(method)

    return {
        "entity_candidates": entity_candidates,
        "planned_obligations": obligations if method != "finplan_v1_entity_only" else obligations[-1:],
        "used": used,
        "used_docs": [item["doc_id"] for item in used],
        "actions": actions,
    }


def score(case: dict, prediction: dict) -> dict:
    gold_docs = set(case["gold_doc_ids"])
    used_docs = set(prediction["used_docs"])
    used_page_by_doc = {item["doc_id"]: item["page_number"] for item in prediction["used"]}
    evidence_hits = []
    for item in case["gold_evidence"]:
        evidence_hits.append(
            used_page_by_doc.get(item["doc_id"]) in set(item["candidate_pages"])
        )
    planned = {item["doc_id"] for item in prediction["planned_obligations"]}
    gold_obligations = set(case["gold_doc_ids"])
    entity_exact = prediction["entity_candidates"] == [case["security_code"]]
    field_oracle_available = all(evidence_hits)
    predicted_same = case["normalized_semantic_same"] if field_oracle_available else None
    return {
        "entity_exact": float(entity_exact),
        "obligation_recall": len(planned & gold_obligations) / len(gold_obligations),
        "obligation_exact": float(planned == gold_obligations),
        "leg_recall": len(used_docs & gold_docs) / len(gold_docs),
        "closure": float(gold_docs.issubset(used_docs)),
        "wrong_doc_rate": len(used_docs - gold_docs) / max(1, len(used_docs)),
        "evidence_page_recall": sum(evidence_hits) / len(evidence_hits),
        "evidence_page_closure": float(all(evidence_hits)),
        "field_oracle_semantic_correct": float(
            field_oracle_available and predicted_same == case["normalized_semantic_same"]
        ),
        "field_oracle_raw_gold_correct": float(
            field_oracle_available and predicted_same == case["raw_finglm_same"]
        ),
        "queries": float(len(prediction["actions"])),
        "documents": float(len(used_docs)),
    }


def summarize(rows: list[dict]) -> dict:
    metrics = [
        "entity_exact",
        "obligation_recall",
        "obligation_exact",
        "leg_recall",
        "closure",
        "wrong_doc_rate",
        "evidence_page_recall",
        "evidence_page_closure",
        "field_oracle_semantic_correct",
        "field_oracle_raw_gold_correct",
        "queries",
        "documents",
    ]
    return {
        metric: sum(float(row[metric]) for row in rows) / max(1, len(rows))
        for metric in metrics
    }


def exact_mcnemar(target: list[int], baseline: list[int]) -> dict:
    target_only = sum(t == 1 and b == 0 for t, b in zip(target, baseline))
    baseline_only = sum(t == 0 and b == 1 for t, b in zip(target, baseline))
    discordant = target_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(target_only, baseline_only) + 1))
        p_value = min(1.0, 2 * tail / (2**discordant))
    return {
        "target_only_successes": target_only,
        "baseline_only_successes": baseline_only,
        "discordant_pairs": discordant,
        "two_sided_exact_p": p_value,
    }


def run(data_path: Path, out_path: Path, budget: int = 4) -> dict:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    chunks = build_chunks(data)
    index = BM25(chunks)
    rows = []
    predictions = []
    for case in data["cases"]:
        for method in METHODS:
            prediction = run_method(case, data["cases"], data["documents"], index, budget, method)
            metrics = score(case, prediction)
            rows.append(
                {
                    "qid": case["qid"],
                    "split": case["split"],
                    "method": method,
                    "label_audit_status": case["label_audit_status"],
                    **metrics,
                }
            )
            predictions.append({"qid": case["qid"], "method": method, **prediction})

    summary = {}
    comparisons = {}
    for split_name, split_rows in (
        ("all_descriptive", rows),
        ("development", [row for row in rows if row["split"] == "development"]),
        ("frozen_pilot", [row for row in rows if row["split"] == "frozen-pilot"]),
    ):
        summary[split_name] = {
            method: summarize([row for row in split_rows if row["method"] == method])
            for method in METHODS
        }
        qids = sorted({row["qid"] for row in split_rows})
        finplan = {
            row["qid"]: int(row["closure"])
            for row in split_rows
            if row["method"] == "finplan_entity_year"
        }
        comparisons[split_name] = {}
        for method in METHODS:
            baseline = {
                row["qid"]: int(row["closure"])
                for row in split_rows
                if row["method"] == method
            }
            comparisons[split_name][method] = exact_mcnemar(
                [finplan[qid] for qid in qids], [baseline[qid] for qid in qids]
            )

    result = {
        "schema": "finplan-china-finglm-page-pilot-results.v1",
        "status": "transparent-public-pilot-not-sota-not-original-hirec",
        "protocol": {
            "data_sha256": sha256(data_path),
            "runner_sha256": sha256(Path(__file__)),
            "cases": len(data["cases"]),
            "documents": len(data["documents"]),
            "pages": len(chunks),
            "budget": budget,
            "retriever": "Unicode NFKC Chinese character+bigram BM25 over PDF pages",
            "gold_fields_hidden_from_retrieval": [
                "gold_doc_ids",
                "gold_evidence",
                "raw_finglm_values",
                "normalized_values",
                "raw_finglm_same",
                "normalized_semantic_same",
            ],
        },
        "interpretation_boundary": [
            "The public pilot has nine near-identical legal-representative comparison templates.",
            "The corpus contains only the 18 required reports plus cross-company distractors; it lacks same-company extra-year and mixed-filing distractors.",
            "HiREC-style is a transparent approximation, not the original implementation.",
            "Field-oracle metrics diagnose page coverage and label quality; they are not a real LLM reader result.",
        ],
        "summary": summary,
        "paired_closure_vs_finplan": comparisons,
        "rows": rows,
        "predictions": predictions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    result = run(args.data, args.out, args.budget)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
