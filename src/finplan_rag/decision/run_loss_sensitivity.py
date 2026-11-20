"""Loss-parameter sensitivity for the v2 decision pilot.

The pilot's decision ranking should not depend on a single arbitrary loss
vector.  This script reruns the main policies at a fixed cost configuration
(gamma=0.005, actual_cost=2.0) under several asymmetric-loss settings and
reports whether the planner's advantage survives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .belief import (
    current_q_likelihood_from_cases,
    position_distribution_from_cases,
    prior_from_cases,
)
from .loss import LossConfig
from .pilot_data import load_v4_cases, split_cases
from .policies import default_policies
from .run_pilot import (
    DEFAULT_DATA,
    DEFAULT_OUT,
    apply_company_delay,
    company_delay_map,
    run_config,
)

CONFIGS = {
    "default": LossConfig(),
    "symmetric": LossConfig(over=10.0, under=10.0),
    "more_overclaim": LossConfig(over=15.0, under=5.0),
    "high_width": LossConfig(width=3.0),
    "low_width": LossConfig(width=1.0),
    "high_abstain": LossConfig(abstain=8.0),
}
POLICIES = (
    "always_answer",
    "always_adjudicate",
    "planner_no_wait",
    "always_wait",
    "prehoc_router",
    "oracle_router",
    "planner",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gamma", type=float, default=0.005)
    parser.add_argument("--actual-cost", type=float, default=2.0)
    args = parser.parse_args()

    cases = load_v4_cases(args.data)
    dev = split_cases(cases, "dev")
    target = split_cases(cases, "test")
    delay_mapping, global_delay = company_delay_map(dev)
    apply_company_delay(dev, delay_mapping, global_delay, allow_map=False)
    apply_company_delay(target, delay_mapping, global_delay, allow_map=True)
    prior = prior_from_cases(dev)
    position_distribution = position_distribution_from_cases(dev)
    current_q_likelihood = current_q_likelihood_from_cases(dev, prior)
    policies = default_policies()

    rows = []
    for name, loss in CONFIGS.items():
        result = run_config(
            target,
            prior,
            current_q_likelihood,
            position_distribution,
            loss,
            gamma=args.gamma,
            actual_cost=args.actual_cost,
            wait_precision=1.0,
            policies=policies,
        )
        for policy in POLICIES:
            metrics = result["policies"].get(policy)
            if metrics is None:
                continue
            rows.append({
                "loss_config": name,
                "loss": loss.as_dict(),
                "policy": policy,
                "total_loss": metrics.get("total_loss"),
                "decision_loss": metrics.get("decision_loss"),
                "coverage": metrics.get("coverage"),
                "wait_rate": metrics.get("wait_rate"),
            })
        planner = result["policies"].get("planner", {})
        no_wait = result["policies"].get("planner_no_wait", {})
        adjudicate = result["policies"].get("always_adjudicate", {})
        print(
            f"{name:14s} planner={planner.get('total_loss'):.4f} "
            f"no_wait={no_wait.get('total_loss'):.4f} "
            f"adjudicate={adjudicate.get('total_loss'):.4f} "
            f"wait_rate={planner.get('wait_rate'):.2f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "loss_sensitivity.json"
    path.write_text(json.dumps({
        "schema": "finplan-decision-pilot-loss-sensitivity-v1",
        "gamma": args.gamma,
        "actual_cost": args.actual_cost,
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
