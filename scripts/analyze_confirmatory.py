"""Analyze the pre-registered wide-width confirmation seeds 11-20."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "paper/results/rf_confirm_s"


def bootstrap_ci(values, n_boot=5000, seed=20261103):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    )
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seeds", default="11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    by_config = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        directory = Path(f"{args.root}{seed}")
        if not (directory / "per_claim.jsonl").exists():
            raise FileNotFoundError(directory)
        for line in (directory / "per_claim.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            by_config[row["config"]][row["claim_id"]].append(row)
    result = {"seeds": seeds, "configs": {}}
    for name, claims in by_config.items():
        planner = np.asarray([np.mean([r["planner_tuned"] for r in items]) for items in claims.values()])
        fixed = np.asarray([np.mean([r["fixed_tuned"] for r in items]) for items in claims.values()])
        difference = planner - fixed
        result["configs"][name] = {
            "n_claims": len(claims),
            "planner": float(np.mean(planner)),
            "fixed": float(np.mean(fixed)),
            "difference": float(np.mean(difference)),
            "difference_ci95": bootstrap_ci(difference),
            "planner_losses": [float(x) for x in planner],
            "fixed_losses": [float(x) for x in fixed],
        }
    wide = result["configs"]["wide_penalty"]
    base = result["configs"]["base"]
    result["primary"] = {
        "config": "wide_penalty",
        "difference": wide["difference"],
        "ci95": wide["difference_ci95"],
        "supported": wide["difference_ci95"][1] < 0.0,
    }
    result["base_non_inferiority"] = {
        "config": "base",
        "difference": base["difference"],
        "ci95": base["difference_ci95"],
        "margin": 0.2,
        "supported": base["difference_ci95"][1] < 0.2,
    }
    out = args.out or REPO_ROOT / "paper/results/rf_confirm_summary.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("configs",)}, indent=2))
    for name, value in result["configs"].items():
        print(
            f"{name:18s} planner={value['planner']:.4f} fixed={value['fixed']:.4f} "
            f"diff={value['difference']:+.4f} CI={value['difference_ci95']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
