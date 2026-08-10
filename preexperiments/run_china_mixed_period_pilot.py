"""Run transparent Chinese mixed-period (H1/Q3/FY) page planning baselines.

The frozen 5-case pilot asks whether the legal representative at two
specified reporting periods of the same company and fiscal year is the same
(e.g. Q3 vs FY, or H1 vs FY).  Each case corpus deliberately contains the
same-company same-year third period report as a same-company distractor, so
a ``security_code x fiscal_year`` obligation key must collapse the two
target periods and is expected to fail on obligation recall.

Development split (2 cases) is diagnostic; frozen-pilot split (3 cases) is
run once after the code is frozen and is not used for further tuning.
"""

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
DEFAULT_DATA = DATA_ROOT / "china_mixed_period_page_pilot_v1.json"
DEFAULT_OUT = RESULTS_ROOT / "china_mixed_period_pilot_v1.json"

METHODS = (
    "single_shot",
    "fixed_decomposition",
    "metadata_decomposition",
    "generic_adaptive_period",
    "hirec_period",
    "finplan_v1_entity_only",
    "finplan_entity_year",
    "finplan_period_aware",
)

# Question text uses "半年度" / "第三季度" / "年度报告" (not always with a
# "报告" suffix), so the word table contains bare period words.  Longer
# words are matched first; overlapping matches are consumed once.
PERIOD_WORDS = {
    "半年度报告": "H1",
    "第三季度报告": "Q3",
    "季度报告": "Q3",
    "半年度": "H1",
    "第三季度": "Q3",
    "年度报告": "FY",
    "半年报": "H1",
    "季报": "Q3",
    "年报": "FY",
}
PERIOD_NAMES = {"H1": "半年度报告", "Q3": "第三季度报告", "FY": "年度报告"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", "", value)


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    result = re.findall(r"[a-z0-9]+", text)
    for sequence in re.findall(r"[一-鿿]+", text):
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
        fiscal_year: int | None = None,
        fiscal_period: str | None = None,
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
            if fiscal_year is not None and chunk["fiscal_year"] != fiscal_year:
                continue
            if fiscal_period and chunk["fiscal_period"] != fiscal_period:
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
                    "fiscal_year": int(document["fiscal_year"]),
                    "fiscal_period": document["fiscal_period"],
                    "page_number": int(page["page_number"]),
                    "terms": tokens(str(page["text"])),
                }
            )
    return chunks


def parse_periods(question: str) -> list[str]:
    """Parse target periods from the question text.

    Longer words match first ("半年度" wins over "年度"); overlapping
    matches are consumed once so "半年度报告" is not double counted.
    """
    matches = []
    for word, period in sorted(PERIOD_WORDS.items(), key=lambda item: -len(item[0])):
        for match in re.finditer(re.escape(word), question):
            matches.append((match.start(), match.end(), period))
    matches.sort()
    periods = []
    covered_until = -1
    for start, end, period in matches:
        if start < covered_until:
            continue
        periods.append(period)
        covered_until = end
    return sorted(set(periods))


def plan_obligations(
    case: dict,
    cases: list[dict],
    documents: list[dict],
    *,
    period_aware: bool,
) -> tuple[list[str], list[dict]]:
    """Plan obligations from the question text and the frozen registry.

    The registry contributes entity aliases and the fiscal year; the target
    periods are parsed from the question itself ("2021年半年度报告和2021年
    年度报告" -> H1 and FY), so gold documents are NOT read by the planner.
    """
    normalized = canonical(case["question"])
    matched_codes = []
    for candidate in cases:
        if any(canonical(alias) in normalized for alias in candidate["aliases"] if len(canonical(alias)) >= 3):
            matched_codes.append(candidate["security_code"])
    matched_codes = sorted(set(matched_codes))
    requested_years = sorted({int(year) for year in re.findall(r"20\d{2}", case["question"])})
    periods = parse_periods(case["question"])
    available = {
        (document["security_code"], int(document["fiscal_year"]), document["fiscal_period"])
        for document in documents
    }
    obligations = []
    for code in matched_codes:
        for year in requested_years:
            if period_aware:
                for period in periods:
                    if (code, year, period) in available:
                        obligations.append(
                            {
                                "security_code": code,
                                "fiscal_year": year,
                                "fiscal_period": period,
                                "doc_id": f"{code}_{year}_{period}",
                            }
                        )
            else:
                if any((code, year, p) in available for p in ("H1", "Q3", "FY")):
                    obligations.append(
                        {
                            "security_code": code,
                            "fiscal_year": year,
                            "fiscal_period": None,
                            "doc_id": f"{code}_{year}",
                        }
                    )
    return matched_codes, obligations


def run_method(case: dict, cases: list[dict], documents: list[dict], index: BM25, budget: int, method: str) -> dict:
    entity_candidates, obligations = plan_obligations(
        case,
        cases,
        documents,
        period_aware=method in ("metadata_decomposition", "finplan_period_aware"),
    )
    # Generic adaptive and HiREC-style use natural-language period words
    # parsed from the question; they do NOT receive a structured obligation
    # key with the fiscal period, i.e. their state has entity+year metadata
    # but the period is only a query string, not a filtered slot.
    question_periods = parse_periods(case["question"])
    used: list[dict] = []
    actions: list[str] = []

    def retrieve(
        query: str,
        label: str,
        *,
        code: str | None = None,
        year: int | None = None,
        period: str | None = None,
        k_docs: int = 1,
    ) -> None:
        remaining = budget - len({item["doc_id"] for item in used})
        if remaining <= 0:
            return
        ranked = index.search(
            query,
            security_code=code,
            fiscal_year=year,
            fiscal_period=period,
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
        for period in parse_periods(question):
            retrieve(f"{question} {period} 法定代表人", f"period:{period}")
    elif method == "metadata_decomposition":
        for obligation in obligations:
            retrieve(
                f"{question} {obligation['security_code']} {obligation['fiscal_year']}年 {obligation['fiscal_period']} 法定代表人",
                f"obligation:{obligation['doc_id']}",
                code=obligation["security_code"],
                year=obligation["fiscal_year"],
                period=obligation["fiscal_period"],
            )
    elif method in ("generic_adaptive_period", "hirec_period"):
        # These agents know entity + fiscal year metadata, but the fiscal
        # period enters only as a natural-language query word, not as a
        # structured filtered slot.
        initial_label = "initial-query" if method == "generic_adaptive_period" else "initial-evidence"
        retrieve(question, initial_label, k_docs=2)
        observed = {item["doc_id"] for item in used}
        for period in question_periods:
            period_word = PERIOD_NAMES[period]
            if not any(case["security_code"] in doc_id for doc_id in observed):
                break
            label = f"missing:{period}" if method == "generic_adaptive_period" else f"complementary:{period}"
            retrieve(
                f"{question} 补充检索 {period_word} 法定代表人",
                label,
                code=entity_candidates[0] if entity_candidates else None,
            )
            observed = {item["doc_id"] for item in used}
    elif method == "finplan_v1_entity_only":
        if entity_candidates:
            retrieve(
                f"{question} {entity_candidates[0]} {max(requested_year(question))}年 法定代表人",
                f"entity-only:{entity_candidates[0]}",
                code=entity_candidates[0],
            )
    elif method == "finplan_entity_year":
        for obligation in obligations:
            retrieve(
                f"{question} {obligation['security_code']} {obligation['fiscal_year']}年 法定代表人",
                f"unbound:{obligation['doc_id']}",
                code=obligation["security_code"],
                year=obligation["fiscal_year"],
            )
    elif method == "finplan_period_aware":
        for obligation in obligations:
            retrieve(
                f"{question} {obligation['security_code']} {obligation['fiscal_year']}年 {obligation['fiscal_period']} 报告 法定代表人",
                f"path:{obligation['doc_id']}",
                code=obligation["security_code"],
                year=obligation["fiscal_year"],
                period=obligation["fiscal_period"],
            )
    else:  # pragma: no cover
        raise ValueError(method)

    return {
        "entity_candidates": entity_candidates,
        "planned_obligations": obligations,
        "used": used,
        "used_docs": [item["doc_id"] for item in used],
        "actions": actions,
    }


def requested_year(question: str) -> list[int]:
    return [int(year) for year in re.findall(r"20\d{2}", question)]


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
            if row["method"] == "finplan_period_aware"
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
        "schema": "finplan-china-mixed-period-pilot-results.v1",
        "status": "transparent-period-collision-pilot-not-sota",
        "protocol": {
            "data_sha256": sha256(data_path),
            "runner_sha256": sha256(Path(__file__)),
            "cases": len(data["cases"]),
            "documents": len(data["documents"]),
            "pages": len(chunks),
            "budget": budget,
            "retriever": "Unicode NFKC Chinese character+bigram BM25 over PDF pages",
            "period_collision_rule": (
                "same-company same-year H1/Q3/FY reports are all in the corpus; "
                "only the two question-specified periods are gold obligations"
            ),
            "gold_fields_hidden_from_methods": [
                "gold_doc_ids",
                "gold_evidence",
                "normalized_semantic_same",
                "semantic_gold_available",
            ],
        },
        "interpretation_boundary": [
            "5 cases: 2 development + 3 frozen; sample too small for statistical claims.",
            "Legal-representative values for H1/Q3 are extracted from PDF text and manually audited; FY values cross-checked against FinGLM gold.",
            "HiREC-style and Generic adaptive are transparent approximations, not original implementations.",
            "Field-oracle metrics diagnose page coverage; they are not a real LLM reader result.",
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
