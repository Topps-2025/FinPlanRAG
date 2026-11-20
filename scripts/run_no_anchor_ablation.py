"""Architecture-level ablation: point-in-time LLM scores without anchor state."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from collect_cloud_metrics import metric_block  # noqa: E402
from summarize_cloud_results import LABELS  # noqa: E402

ROOT = REPO_ROOT / "paper" / "results" / "cloud_2026-11-20"
LABEL_ORDER = list(LABELS)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scores(row: dict) -> list[float]:
    raw = row.get("bucket_scores") or {}
    values = np.asarray([float(raw.get(label, 0.0)) for label in LABEL_ORDER], dtype=float)
    values = np.maximum(values, 1e-12)
    return (values / values.sum()).tolist()


def tempered(probabilities: list[float], temperature: float) -> list[float]:
    values = np.power(np.asarray(probabilities, dtype=float), 1.0 / temperature)
    return (values / values.sum()).tolist()


def nll(probabilities: list[float], truth: str) -> float:
    return -math.log(max(probabilities[LABEL_ORDER.index(truth)], 1e-12))


def main() -> None:
    output = {"schema": "finplan-no-anchor-ablation-v1", "backbones": {}}
    for backbone in ("deepseek", "glm", "qwen3"):
        validation = [
            row for row in load_jsonl(ROOT / backbone / "evolution" / "anchors_validation.jsonl")
            if row.get("split") == "validation"
        ]
        test = [
            row for row in load_jsonl(ROOT / backbone / "evolution" / "anchors_test.jsonl")
            if row.get("split") == "test"
        ]
        verified = {
            row["claim_id"]
            for row in load_jsonl(ROOT / backbone / "evolution" / "temporal_test_probs_new_cloud.jsonl")
        }
        test = [row for row in test if row["claim_id"] in verified]
        temperatures = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        chosen = min(
            temperatures,
            key=lambda t: np.mean([
                nll(tempered(scores(row), t), LABELS[row["truth"]] if isinstance(row["truth"], int) else row["truth"])
                for row in validation
            ]),
        )
        probabilities = [tempered(scores(row), chosen) for row in test]
        truths = [LABELS[row["truth"]] if isinstance(row["truth"], int) else row["truth"] for row in test]
        output["backbones"][backbone] = {
            "temperature": chosen,
            "n_test": len(test),
            **metric_block(probabilities, truths),
        }
        print(backbone, "T", chosen, output["backbones"][backbone])
    target = ROOT / "no_anchor_ablation.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
