"""CPU BGE dense document-level robustness check for the frozen LOFin pilot.

This is intentionally a document-closure experiment, not end-to-end answer
generation.  To keep the experiment reproducible on a CPU-only workstation,
each SEC filing contributes 12 uniformly spaced passages to a max-similarity
document index.  The planner, cutoff filter, path obligations and budgets are
otherwise imported unchanged from ``run_lofin_multidoc_pilot``.
"""

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

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from run_lofin_multidoc_pilot import METHODS, Chunk, make_chunks, parse_time, run_method, score, aggregate


class DenseIndex:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        model_dir: Path,
        samples_per_doc: int = 12,
        batch_size: int = 16,
        max_length: int = 256,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.model.eval()
        self.samples_per_doc = samples_per_doc
        self.max_length = max_length
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        by_doc: Dict[str, List[Chunk]] = {}
        for chunk in chunks:
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
        self.representatives: List[Chunk] = []
        for doc_id in sorted(by_doc):
            items = by_doc[doc_id]
            if len(items) <= samples_per_doc:
                chosen = items
            else:
                positions = np.linspace(0, len(items) - 1, samples_per_doc, dtype=int)
                chosen = [items[int(pos)] for pos in positions]
            self.representatives.extend(chosen)
        self.vectors = self._encode([chunk.text for chunk in self.representatives], batch_size)
        self.query_cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.unsqueeze(-1).expand(hidden.size()).float()
        return (hidden * expanded).sum(1) / torch.clamp(expanded.sum(1), min=1e-9)

    def _encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        outputs: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    list(texts[start : start + batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                hidden = self.model(**batch).last_hidden_state
                pooled = self._mean_pool(hidden, batch["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def search(
        self,
        query: str,
        cutoff: datetime,
        used: Sequence[str],
        allow_future: bool,
        limit: int = 1,
        required_terms: Sequence[str] = (),
    ) -> List[Chunk]:
        if query not in self.query_cache:
            self.query_cache[query] = self._encode([query], 1)[0]
        q = self.query_cache[query]
        used_set = set(used)
        used_docs = {self.chunk_lookup[chunk_id].doc_id for chunk_id in used_set}
        scored = []
        for vector, chunk in zip(self.vectors, self.representatives):
            if chunk.chunk_id in used_set or chunk.doc_id in used_docs:
                continue
            if not allow_future and chunk.available_at > cutoff:
                continue
            if required_terms and not all(term in chunk.terms for term in required_terms):
                continue
            scored.append((float(np.dot(q, vector)), chunk))
        scored.sort(key=lambda item: (-item[0], item[1].available_at, item[1].chunk_id))
        out: List[Chunk] = []
        docs: set[str] = set()
        for _, chunk in scored:
            if chunk.doc_id in docs:
                continue
            out.append(chunk)
            docs.add(chunk.doc_id)
            if len(out) >= limit:
                break
        return out


def run(data_path: Path, out_path: Path, model_dir: Path, budget: int, samples_per_doc: int) -> Mapping[str, object]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    chunks = make_chunks(data["documents"])
    index = DenseIndex(chunks, model_dir, samples_per_doc=samples_per_doc)
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
        "schema": "finplan-lofin-multidoc-dense-results.v1",
        "status": "document-closure-not-full-answer-generation",
        "protocol": {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "data_cases": len(data["cases"]),
            "data_documents": len(data["documents"]),
            "retriever": "BAAI/bge-small-en-v1.5 mean pooling, cosine, max over 12 uniform passages per document",
            "model_dir": str(model_dir),
            "samples_per_doc": samples_per_doc,
            "max_length": 256,
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
    parser.add_argument("--out", default=RESULTS_ROOT / "lofin_multidoc_validation2_dense_v1.json")
    parser.add_argument("--model-dir", default=MODELS_ROOT / "bge-small-en-v1.5")
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--samples-per-doc", type=int, default=12)
    args = parser.parse_args()
    result = run(Path(args.data), Path(args.out), Path(args.model_dir), args.budget, args.samples_per_doc)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
