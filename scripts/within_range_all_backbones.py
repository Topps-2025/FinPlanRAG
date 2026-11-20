"""Within-guidance stratified fusion-versus-RF check on all three backbones."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from summarize_cloud_results import LABELS, actual_loss  # noqa: E402

ROOT = REPO_ROOT / "paper" / "results" / "cloud_2026-11-20"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    rf_rows = load_jsonl(REPO_ROOT / "paper" / "results" / "rf_calibrated_repro" / "per_claim.jsonl")
    seeds = sorted({int(row.get("seed", 0)) for row in rf_rows})
    rf_by_seed = {}
    for seed in seeds:
        rf_by_seed[seed] = {
            row["claim_id"]: np.asarray(row["probabilities"], dtype=float)
            for row in rf_rows
            if int(row.get("seed", 0)) == seed
        }
    output = {"schema": "finplan-within-range-all-backbones-v1", "resamples": 10000, "seed": 20261120, "backbones": {}}
    rng = np.random.default_rng(20261120)
    for backbone in ("deepseek", "glm", "qwen3"):
        fusion = json.loads((ROOT / backbone / "evolution" / "temporal_fusion_new_cloud.json").read_text(encoding="utf-8"))
        weight = float(fusion["selected_weight"])
        rows = load_jsonl(ROOT / backbone / "evolution" / "temporal_test_probs_new_cloud.jsonl")
        within = [row for row in rows if (LABELS[row["truth"]] if isinstance(row["truth"], int) else row["truth"]) == "within"]
        per_seed = {}
        for seed in seeds:
            rf = rf_by_seed[seed]
            available = [row for row in within if row["claim_id"] in rf]
            fused_losses = []
            rf_losses = []
            for row in available:
                anchor = np.asarray(row["llm_probabilities"], dtype=float)
                probability = weight * anchor + (1.0 - weight) * rf[row["claim_id"]]
                probability = probability / probability.sum()
                truth = LABELS[row["truth"]] if isinstance(row["truth"], int) else row["truth"]
                fused_losses.append(actual_loss(list(probability), truth)[0])
                rf_losses.append(actual_loss(list(rf[row["claim_id"]]), truth)[0])
            fused_array = np.asarray(fused_losses)
            rf_array = np.asarray(rf_losses)
            diff = fused_array - rf_array
            draws = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
            per_seed[str(seed)] = {
                "n": int(len(diff)),
                "fusion": float(fused_array.mean()),
                "rf": float(rf_array.mean()),
                "difference": float(diff.mean()),
                "ci_low": float(np.percentile(draws, 2.5)),
                "ci_high": float(np.percentile(draws, 97.5)),
            }
        excluding = sum(1 for value in per_seed.values() if value["ci_high"] < 0 or value["ci_low"] > 0)
        output["backbones"][backbone] = {
            "fusion_weight": weight,
            "n_within": len(within),
            "per_seed": per_seed,
            "seeds_excluding_zero": excluding,
            "seeds_total": len(seeds),
        }
        print(backbone, "weight", weight, "n", len(within), "excl", excluding, "/", len(seeds))
        for seed, value in per_seed.items():
            print(f"  seed {seed}: {value['difference']:+.3f} [{value['ci_low']:+.3f}, {value['ci_high']:+.3f}]")
    target = ROOT / "within_range_all_backbones.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
