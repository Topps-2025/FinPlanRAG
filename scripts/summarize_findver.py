"""Aggregate FinDVer component results by retrieval condition."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    truth = [str(row.get("truth")) for row in rows]
    verdict = [str((row.get("parsed") or {}).get("verdict") or "insufficient") for row in rows]
    labels = sorted(set(truth) | set(verdict))
    recalls = [float(row.get("evidence_recall") or 0.0) for row in rows]
    paragraphs = [float(row.get("num_paragraphs") or 0.0) for row in rows]
    usage = [row.get("usage") or {} for row in rows]
    latency = [float(row.get("latency_seconds") or 0.0) for row in rows]
    return {
        "n": len(rows),
        "accuracy": float(accuracy_score(truth, verdict)),
        "macro_f1": float(f1_score(truth, verdict, labels=labels, average="macro", zero_division=0)),
        "evidence_recall": float(np.mean(recalls)),
        "mean_paragraphs": float(np.mean(paragraphs)),
        "mean_input_tokens": float(np.mean([float(item.get("input_tokens") or 0.0) for item in usage])),
        "mean_output_tokens": float(np.mean([float(item.get("output_tokens") or 0.0) for item in usage])),
        "mean_latency_seconds": float(np.mean(latency)),
        "insufficient_rate": float(np.mean([value == "insufficient" for value in verdict])),
        "verdicts": {label: int(sum(value == label for value in verdict)) for label in labels},
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: summarize_findver.py <run_dir> [<run_dir> ...]")
    for value in sys.argv[1:]:
        directory = Path(value)
        rows = load(directory / "per_example.jsonl")
        conditions = sorted({str(row.get("condition")) for row in rows})
        output = {
            "schema": "finplan-findver-summary-v1",
            "n_rows": len(rows),
            "conditions": {condition: summarize([row for row in rows if row.get("condition") == condition]) for condition in conditions},
            "subsets": {
                subset: {
                    condition: summarize([
                        row for row in rows
                        if row.get("condition") == condition and str(row.get("subset")) == subset
                    ])
                    for condition in conditions
                }
                for subset in sorted({str(row.get("subset")) for row in rows})
            },
        }
        target = directory / "summary.json"
        target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(target)
        for condition, metrics in output["conditions"].items():
            print(f"  {condition:18s} acc={metrics['accuracy']:.3f} macro={metrics['macro_f1']:.3f} "
                  f"recall={metrics['evidence_recall']:.3f} paras={metrics['mean_paragraphs']:.2f} "
                  f"tokens={metrics['mean_input_tokens']:.0f}/{metrics['mean_output_tokens']:.0f}")


if __name__ == "__main__":
    main()
