"""Compute regret against an oracle fixed policy for a conformal result.

The oracle fixed comparator is the minimum, per loss configuration, of
always-answer (point), the width-tuned fixed adjudicator, and always-abstain
at that configuration's abstention cost. It is a diagnostic upper bound: test
labels choose the comparator, but never choose the conformal threshold, score,
or alpha.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_real_decision_eval import CONFIGS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "paper/results/deepseek_conformal_realized_verified/summary.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    configs = {name: config for name, config in CONFIGS}
    policies = ("mean_loss", "planner_tuned", "fixed_tuned", "static_set", "point")
    regrets = {policy: [] for policy in policies}
    rows = {}
    for name, values in summary.items():
        config = configs[name]
        best_fixed = min(
            float(values["fixed_tuned"]),
            float(values["point"]),
            float(config.abstain),
        )
        rows[name] = {"best_fixed_oracle": best_fixed}
        for policy in policies:
            regret = float(values[policy]) - best_fixed
            regrets[policy].append(regret)
            rows[name][policy] = {
                "loss": float(values[policy]),
                "regret": regret,
            }
    aggregate = {
        policy: {
            "mean_loss": float(np.mean([summary[name][policy] for name in summary])),
            "mean_regret": float(np.mean(regrets[policy])),
            "max_regret": float(np.max(regrets[policy])),
        }
        for policy in policies
    }
    result = {"rows": rows, "aggregate": aggregate}
    out = args.out or args.summary.parent / "regret_summary.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("policy, mean loss, mean regret, max regret")
    for policy in policies:
        value = aggregate[policy]
        print(
            f"{policy:14s} {value['mean_loss']:.4f} "
            f"{value['mean_regret']:+.4f} {value['max_regret']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
