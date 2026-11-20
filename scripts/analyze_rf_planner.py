"""Summarize the RF-backbone planner across the ten loss configurations.

Reads the per-seed outputs written by
``scripts/run_backbone_conformal_diagnostics.py``. The comparator is the
oracle fixed policy: minimum of point answer, tuned fixed adjudicator, and
always-abstain at the configuration's abstention cost.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_real_decision_eval import CONFIGS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "paper/results/rf500_all_s"


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
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    configs = {name: config for name, config in CONFIGS}
    by_config = defaultdict(lambda: defaultdict(list))
    for seed in range(1, 11):
        directory = Path(f"{args.root}{seed}")
        if not directory.exists():
            continue
        for line in (directory / "per_claim.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            by_config[row["config"]][row["claim_id"]].append(row)
    rows = {}
    planner_regrets = []
    fixed_regrets = []
    point_regrets = []
    for name, claims in by_config.items():
        config = configs[name]
        planner = np.asarray([np.mean([r["planner_tuned"] for r in items]) for items in claims.values()])
        fixed = np.asarray([np.mean([r["fixed_tuned"] for r in items]) for items in claims.values()])
        point = np.asarray([np.mean([r["point"] for r in items]) for items in claims.values()])
        best_value = min(float(np.mean(fixed)), float(np.mean(point)), config.abstain)
        planner_regret = planner - best_value
        fixed_regret = fixed - best_value
        point_regret = point - best_value
        planner_regrets.append(planner_regret)
        fixed_regrets.append(fixed_regret)
        point_regrets.append(point_regret)
        rows[name] = {
            "n_claims": len(claims),
            "planner": float(np.mean(planner)),
            "fixed": float(np.mean(fixed)),
            "point": float(np.mean(point)),
            "abstain": float(config.abstain),
            "best_fixed_oracle": float(best_value),
            "planner_regret": float(np.mean(planner_regret)),
            "fixed_regret": float(np.mean(fixed_regret)),
            "point_regret": float(np.mean(point_regret)),
            "planner_minus_fixed": {
                "mean": float(np.mean(planner - fixed)),
                "ci95": bootstrap_ci(planner - fixed),
            },
        }
    all_planner_regret = np.concatenate(planner_regrets)
    all_fixed_regret = np.concatenate(fixed_regrets)
    all_point_regret = np.concatenate(point_regrets)
    result = {
        "configs": rows,
        "aggregate": {
            "planner_mean_regret": float(np.mean([v["planner_regret"] for v in rows.values()])),
            "planner_max_regret": float(np.max([v["planner_regret"] for v in rows.values()])),
            "fixed_mean_regret": float(np.mean([v["fixed_regret"] for v in rows.values()])),
            "fixed_max_regret": float(np.max([v["fixed_regret"] for v in rows.values()])),
            "point_mean_regret": float(np.mean([v["point_regret"] for v in rows.values()])),
            "point_max_regret": float(np.max([v["point_regret"] for v in rows.values()])),
            "planner_regret_ci95": bootstrap_ci(all_planner_regret),
            "fixed_regret_ci95": bootstrap_ci(all_fixed_regret),
            "point_regret_ci95": bootstrap_ci(all_point_regret),
        },
    }
    out = args.out or Path(f"{args.root}_summary.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("| config | planner | fixed | point | abstain | best fixed | planner regret |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, value in rows.items():
        print(
            f"| {name} | {value['planner']:.4f} | {value['fixed']:.4f} | "
            f"{value['point']:.4f} | {value['abstain']:.2f} | "
            f"{value['best_fixed_oracle']:.4f} | {value['planner_regret']:+.4f} |"
        )
    print(json.dumps(result["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
