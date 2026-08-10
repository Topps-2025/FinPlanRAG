"""Evaluate non-oracle company-year obligation planning and retrieval cascade.

Planners see only the question, cutoff, and a public SEC/corpus registry. Gold
candidate legs and document IDs are used only after prediction for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

try:
    from .storage_paths import DATA_ROOT, MODELS_ROOT, RESULTS_ROOT
except ImportError:
    from storage_paths import DATA_ROOT, MODELS_ROOT, RESULTS_ROOT

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_lofin_multidoc_pilot import BM25, make_chunks, run_method, score


YEAR_RE = re.compile(r"\b(20\d{2})\b")


def normalize(text: str) -> str:
    text = text.lower().replace("&", " and ").replace("’", "'")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def registry_rule(question: str, companies: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    normalized = normalize(question)
    upper_tokens = set(re.findall(r"\b[A-Z][A-Z0-9.\-]{1,5}\b", question))
    years = sorted({int(year) for year in YEAR_RE.findall(question)})
    matched: List[Mapping[str, object]] = []
    for company in companies:
        ticker = str(company["ticker"])
        aliases = sorted((str(alias) for alias in company["aliases"]), key=len, reverse=True)
        alias_hit = any(len(alias) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized) for alias in aliases)
        ticker_hit = ticker in upper_tokens
        if alias_hit or ticker_hit:
            matched.append(company)

    plans: List[Dict[str, object]] = []
    for company in matched:
        available = sorted(int(year) for year in company["available_years"])
        selected: List[int] = []
        # "as stated/reported in the 2023 10-K" identifies the filing even
        # when another numerical range describes forecast values.
        filing_mentions = [
            int(year)
            for year in re.findall(r"\b(20\d{2})\b[^.]{0,30}\b10\s*-?\s*k\b", question, flags=re.I)
        ]
        if filing_mentions:
            selected = [year for year in filing_mentions if year in available]
        elif len(years) >= 2 and re.search(r"\b(from|between|over|trend|change)\b", normalized):
            lower, upper = min(years), max(years)
            selected = [year for year in available if lower <= year <= upper]
        else:
            selected = [year for year in years if year in available]
        if not selected and available:
            # Questions without an explicit report year use the latest filing
            # available by the frozen corpus/cutoff registry.
            selected = [max(available)]
        plans.append({"ticker": str(company["ticker"]), "years": sorted(set(selected))})
    return sorted(plans, key=lambda leg: str(leg["ticker"]))


class QwenPlanner:
    def __init__(self, model_dir: Path, companies: Sequence[Mapping[str, object]]) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
        self.model.eval()
        self.companies = list(companies)
        self.allowed = {str(company["ticker"]): {int(y) for y in company["available_years"]} for company in companies}
        self.registry_text = "\n".join(
            f"{company['ticker']} | {company['official_name']} | years={','.join(map(str, company['available_years']))}"
            for company in companies
        )

    def predict(self, question: str) -> Tuple[List[Dict[str, object]], str, bool]:
        system = (
            "You plan SEC filing retrieval. Select every company explicitly named in the question from the registry. "
            "Do not add peers. Select filing years, not forecast years. If the question says values are stated/reported "
            "in a YEAR 10-K, use YEAR. For a historical range, select registry filing years needed to cover the range. "
            'Return only JSON: {"legs":[{"ticker":"TICKER","years":[2023]}]}.'
        )
        user = f"Registry:\n{self.registry_text}\n\nQuestion: {question}"
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        parsed_ok = True
        try:
            match = re.search(r"\{.*\}", text, flags=re.S)
            payload = json.loads(match.group(0) if match else text)
            raw_legs = payload.get("legs", [])
        except Exception:
            parsed_ok = False
            raw_legs = []
        legs: List[Dict[str, object]] = []
        for leg in raw_legs:
            ticker = str(leg.get("ticker", "")).upper()
            if ticker not in self.allowed:
                continue
            selected = sorted({int(y) for y in leg.get("years", []) if str(y).isdigit() and int(y) in self.allowed[ticker]})
            if selected:
                legs.append({"ticker": ticker, "years": selected})
        unique = {str(leg["ticker"]): leg for leg in legs}
        return sorted(unique.values(), key=lambda leg: str(leg["ticker"])), text, parsed_ok


def obligations(legs: Sequence[Mapping[str, object]]) -> Set[str]:
    return {f"{leg['ticker']}_{int(year)}_10K" for leg in legs for year in leg["years"]}


def prf(predicted: Set[str], gold: Set[str]) -> Mapping[str, float]:
    overlap = len(predicted & gold)
    return {
        "recall": overlap / len(gold) if gold else 1.0,
        "precision": overlap / len(predicted) if predicted else float(not gold),
        "exact": float(predicted == gold),
        "covers_gold": float(gold <= predicted),
    }


def aggregate(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> Mapping[str, float]:
    return {field: sum(float(row[field]) for row in rows) / max(1, len(rows)) for field in fields}


def run(data_path: Path, registry_path: Path, out_path: Path, model_dir: Path, budget: int) -> Mapping[str, object]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    companies = registry["companies"]
    qwen = QwenPlanner(model_dir, companies)
    chunks = make_chunks(data["documents"])
    index = BM25(chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rows: List[Dict[str, object]] = []
    predictions: List[Dict[str, object]] = []
    for case in data["cases"]:
        planner_outputs = {"registry_rule": (registry_rule(str(case["question"]), companies), "", True)}
        planner_outputs["qwen_0_5b"] = qwen.predict(str(case["question"]))
        gold_docs = set(str(doc) for doc in case["gold_doc_ids"])
        gold_entities = {doc.split("_")[0] for doc in gold_docs}
        for planner, (legs, raw, parsed_ok) in planner_outputs.items():
            predicted_docs = obligations(legs)
            predicted_entities = {str(leg["ticker"]) for leg in legs}
            entity_metrics = prf(predicted_entities, gold_entities)
            obligation_metrics = prf(predicted_docs, gold_docs)
            planned_case = dict(case)
            planned_case["candidate_legs"] = legs
            cascade = run_method(planned_case, index, budget, "finplan_v3_path_bound")
            cascade_metrics = score(case, cascade, chunks_by_id)
            row = {
                "case_id": case["case_id"],
                "template": case["template"],
                "planner": planner,
                "entity_recall": entity_metrics["recall"],
                "entity_precision": entity_metrics["precision"],
                "entity_exact": entity_metrics["exact"],
                "obligation_recall": obligation_metrics["recall"],
                "obligation_precision": obligation_metrics["precision"],
                "obligation_exact": obligation_metrics["exact"],
                "obligation_covers_gold": obligation_metrics["covers_gold"],
                "cascade_leg_recall": cascade_metrics["leg_recall"],
                "cascade_closure": cascade_metrics["closure"],
                "cascade_wrong_doc_rate": cascade_metrics["wrong_doc_rate"],
                "queries": cascade_metrics["queries"],
                "parsed_ok": float(parsed_ok),
            }
            rows.append(row)
            predictions.append(
                {
                    "case_id": case["case_id"],
                    "planner": planner,
                    "predicted_legs": legs,
                    "raw_model_output": raw,
                    "retrieval": cascade,
                }
            )
    fields = (
        "entity_recall",
        "entity_precision",
        "entity_exact",
        "obligation_recall",
        "obligation_precision",
        "obligation_exact",
        "obligation_covers_gold",
        "cascade_leg_recall",
        "cascade_closure",
        "cascade_wrong_doc_rate",
        "queries",
        "parsed_ok",
    )
    summary = {
        planner: aggregate([row for row in rows if row["planner"] == planner], fields)
        for planner in ("registry_rule", "qwen_0_5b")
    }
    result = {
        "schema": "finplan-nonoracle-obligation-cascade.v1",
        "status": "planning-and-document-closure-not-answer-generation",
        "protocol": {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases": len(data["cases"]),
            "companies_in_registry": len(companies),
            "retriever": "shared BM25 plus frozen path-bound cascade",
            "budget": budget,
            "gold_fields_hidden_from_planners": ["candidate_legs", "gold_doc_ids", "gold_pages", "reference_answer"],
        },
        "summary": summary,
        "rows": rows,
        "predictions": predictions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_ROOT / "lofin_multidoc_validation2_v1.json")
    parser.add_argument("--registry", default=DATA_ROOT / "sec_company_registry_validation2_v1.json")
    parser.add_argument("--out", default=RESULTS_ROOT / "nonoracle_obligation_planning_v1.json")
    parser.add_argument("--model-dir", default=MODELS_ROOT / "Qwen2.5-0.5B-Instruct")
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    result = run(Path(args.data), Path(args.registry), Path(args.out), Path(args.model_dir), args.budget)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
