"""Evaluate the frozen v1 transparent registry rule without the crashed Qwen branch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .storage_paths import DATA_ROOT, RESULTS_ROOT
except ImportError:
    from storage_paths import DATA_ROOT, RESULTS_ROOT
from typing import Dict, List, Mapping, Sequence

from run_lofin_multidoc_pilot import BM25, make_chunks, run_method, score
from run_nonoracle_obligation_planning import obligations, prf, registry_rule


def average(rows: Sequence[Mapping[str, object]], name: str) -> float:
    return sum(float(row[name]) for row in rows) / max(1, len(rows))


def run(data_path: Path, registry_path: Path, out_path: Path, budget: int) -> Mapping[str, object]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    chunks = make_chunks(data["documents"])
    index = BM25(chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rows: List[Dict[str, object]] = []
    predictions: List[Dict[str, object]] = []
    for case in data["cases"]:
        legs = registry_rule(str(case["question"]), registry["companies"])
        gold_docs = set(case["gold_doc_ids"])
        gold_entities = {doc.split("_")[0] for doc in gold_docs}
        entity = prf({str(leg["ticker"]) for leg in legs}, gold_entities)
        obligation = prf(obligations(legs), gold_docs)
        planned_case = dict(case)
        planned_case["candidate_legs"] = legs
        cascade = run_method(planned_case, index, budget, "finplan_v3_path_bound")
        metrics = score(case, cascade, chunks_by_id)
        rows.append(
            {
                "case_id": case["case_id"],
                "entity_recall": entity["recall"],
                "entity_precision": entity["precision"],
                "entity_exact": entity["exact"],
                "obligation_recall": obligation["recall"],
                "obligation_precision": obligation["precision"],
                "obligation_exact": obligation["exact"],
                "obligation_covers_gold": obligation["covers_gold"],
                "cascade_leg_recall": metrics["leg_recall"],
                "cascade_closure": metrics["closure"],
                "cascade_wrong_doc_rate": metrics["wrong_doc_rate"],
            }
        )
        predictions.append({"case_id": case["case_id"], "predicted_legs": legs, "retrieval": cascade})
    fields = [name for name in rows[0] if name != "case_id"]
    result = {
        "schema": "finplan-registry-rule-validation3.v1",
        "status": "frozen-v1-rule-baseline-without-qwen",
        "protocol": {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases": len(rows),
            "budget": budget,
        },
        "summary": {field: average(rows, field) for field in fields},
        "rows": rows,
        "predictions": predictions,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_ROOT / "lofin_multidoc_validation3_v1.json")
    parser.add_argument("--registry", default=DATA_ROOT / "sec_company_registry_validation3_v1.json")
    parser.add_argument("--out", default=RESULTS_ROOT / "registry_rule_v1_validation3.json")
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    result = run(Path(args.data), Path(args.registry), Path(args.out), args.budget)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
