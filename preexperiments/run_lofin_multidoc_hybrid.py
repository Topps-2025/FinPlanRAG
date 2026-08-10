"""Frozen BM25+BGE reciprocal-rank-fusion robustness check."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

try:
    from .storage_paths import DATA_ROOT, MODELS_ROOT, RESULTS_ROOT
except ImportError:
    from storage_paths import DATA_ROOT, MODELS_ROOT, RESULTS_ROOT

from run_lofin_multidoc_dense import DenseIndex
from run_lofin_multidoc_pilot import BM25, METHODS, Chunk, aggregate, make_chunks, run_method, score


class HybridIndex:
    def __init__(self, chunks: Sequence[Chunk], model_dir: Path, samples_per_doc: int = 12, rrf_k: int = 60) -> None:
        self.lexical = BM25(chunks)
        self.dense = DenseIndex(chunks, model_dir, samples_per_doc=samples_per_doc)
        self.documents = len({chunk.doc_id for chunk in chunks})
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        cutoff: datetime,
        used: Sequence[str],
        allow_future: bool,
        limit: int = 1,
        required_terms: Sequence[str] = (),
    ) -> List[Chunk]:
        lexical = self.lexical.search(
            query, cutoff, used, allow_future, limit=self.documents, required_terms=required_terms
        )
        dense = self.dense.search(
            query, cutoff, used, allow_future, limit=self.documents, required_terms=required_terms
        )
        scores: Dict[str, float] = {}
        representatives: Dict[str, Chunk] = {}
        for ranking in (lexical, dense):
            for rank, chunk in enumerate(ranking, start=1):
                scores[chunk.doc_id] = scores.get(chunk.doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
                representatives.setdefault(chunk.doc_id, chunk)
        ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        return [representatives[doc_id] for doc_id in ordered[:limit]]


def run(data_path: Path, out_path: Path, model_dir: Path, budget: int, samples_per_doc: int, rrf_k: int) -> Mapping[str, object]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    chunks = make_chunks(data["documents"])
    index = HybridIndex(chunks, model_dir, samples_per_doc=samples_per_doc, rrf_k=rrf_k)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rows: List[Dict[str, object]] = []
    trajectories: List[Dict[str, object]] = []
    for case in data["cases"]:
        for method in METHODS:
            prediction = run_method(case, index, budget, method)
            metrics = score(case, prediction, chunks_by_id)
            rows.append({"case_id": case["case_id"], "split": case["split"], "template": case["template"], "method": method, **metrics})
            trajectories.append({"case_id": case["case_id"], "method": method, **prediction})
    summary = {
        split: {method: aggregate([row for row in rows if row["split"] == split and row["method"] == method]) for method in METHODS}
        for split in sorted({str(case["split"]) for case in data["cases"]})
    }
    result = {
        "schema": "finplan-lofin-multidoc-hybrid-results.v1",
        "status": "document-closure-not-full-answer-generation",
        "protocol": {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "data_cases": len(data["cases"]),
            "data_documents": len(data["documents"]),
            "retriever": "reciprocal rank fusion of shared BM25 and frozen CPU BGE document rankings",
            "samples_per_doc": samples_per_doc,
            "rrf_k": rrf_k,
            "budget": budget,
        },
        "summary": summary,
        "rows": rows,
        "trajectories": trajectories,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_ROOT / "lofin_multidoc_validation2_v1.json")
    parser.add_argument("--out", default=RESULTS_ROOT / "lofin_multidoc_validation2_hybrid_v1.json")
    parser.add_argument("--model-dir", default=MODELS_ROOT / "bge-small-en-v1.5")
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--samples-per-doc", type=int, default=12)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()
    result = run(Path(args.data), Path(args.out), Path(args.model_dir), args.budget, args.samples_per_doc, args.rrf_k)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
