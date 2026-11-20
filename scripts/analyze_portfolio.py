"""Evaluate the loss-aware policy portfolio from saved per-claim artifacts.

Validation records provide ``candidate_validation`` losses. Test per-claim
records provide the realized losses of fixed, abstain, and planner. Test
outcomes are never used to select the policy.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_real_decision_eval import CONFIGS
from finplan_rag.decision.portfolio import PortfolioConfig, select_policy

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_validation(root: Path, seeds: list[int]):
    by_config = defaultdict(list)
    for seed in seeds:
        records = json.loads((Path(f"{root}{seed}") / "records.json").read_text(encoding="utf-8"))
        for record in records:
            by_config[record["config"]].append(record)
    return by_config


def load_test(root: Path, seeds: list[int]):
    by_config = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        for line in (Path(f"{root}{seed}") / "per_claim.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            by_config[row["config"]][row["claim_id"]].append(row)
    return by_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--validation-seeds", default="6,7,8,9,10")
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--test-seeds", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261103)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    validation = load_validation(
        args.validation_root, [int(v) for v in args.validation_seeds.split(",")]
    )
    test = load_test(args.test_root, [int(v) for v in args.test_seeds.split(",")])
    configs = {name: config for name, config in CONFIGS}
    rows = {}
    regrets = []
    regret_vectors = {}
    for name, records in validation.items():
        n = sum(record["n"] for record in records)
        validation_loss = {
            policy: sum(record["candidate_validation"][policy] * record["n"] for record in records) / n
            for policy in ("fixed_tuned", "planner_tuned", "abstain")
        }
        selected = select_policy(
            {
                "fixed": validation_loss["fixed_tuned"],
                "planner": validation_loss["planner_tuned"],
                "abstain": validation_loss["abstain"],
            },
            PortfolioConfig(switching_margin=args.margin),
        )
        test_key = {
            "fixed": "fixed_tuned",
            "planner": "planner_tuned",
            "abstain": "abstain",
        }[selected]
        claims = test[name]
        claim_ids = sorted(claims)
        portfolio = np.asarray(
            [np.mean([row[test_key] for row in claims[cid]]) for cid in claim_ids]
        )
        fixed = np.asarray(
            [np.mean([row["fixed_tuned"] for row in claims[cid]]) for cid in claim_ids]
        )
        point = np.asarray(
            [np.mean([row["point"] for row in claims[cid]]) for cid in claim_ids]
        )
        abstain_cost = configs[name].abstain
        abstain = np.full_like(fixed, abstain_cost)
        candidates = {"fixed": fixed, "point": point, "abstain": abstain}
        best_fixed_policy = min(candidates, key=lambda k: float(np.mean(candidates[k])))
        best_fixed = float(np.mean(candidates[best_fixed_policy]))
        regret_vector = portfolio - candidates[best_fixed_policy]
        regret = float(np.mean(regret_vector))
        regrets.append(regret)
        regret_vectors[name] = regret_vector
        rows[name] = {
            "selected": selected,
            "portfolio": float(np.mean(portfolio)),
            "fixed": float(np.mean(fixed)),
            "point": float(np.mean(point)),
            "abstain": abstain_cost,
            "best_fixed_policy": best_fixed_policy,
            "best_fixed_oracle": best_fixed,
            "regret": regret,
            "validation": validation_loss,
        }

    if args.bootstrap > 0:
        rng = np.random.default_rng(args.bootstrap_seed)
        lengths = {len(v) for v in regret_vectors.values()}
        if len(lengths) != 1:
            raise ValueError("bootstrap requires aligned claim counts across configs")
        n_claims = lengths.pop()
        indices = rng.integers(0, n_claims, size=(args.bootstrap, n_claims))
        boot = np.vstack(
            [regret_vectors[name][indices].mean(axis=1) for name in regret_vectors]
        )
        for i, name in enumerate(regret_vectors):
            rows[name]["regret_ci95"] = [
                float(np.quantile(boot[i], 0.025)),
                float(np.quantile(boot[i], 0.975)),
            ]
        aggregate_ci = [
            float(np.quantile(boot.mean(axis=0), 0.025)),
            float(np.quantile(boot.mean(axis=0), 0.975)),
        ]
    else:
        aggregate_ci = None

    result = {
        "margin": args.margin,
        "validation_seeds": args.validation_seeds,
        "test_seeds": args.test_seeds,
        "bootstrap": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "rows": rows,
        "aggregate": {
            "mean_regret": float(np.mean(regrets)),
            "mean_regret_ci95": aggregate_ci,
            "max_regret": float(np.max(regrets)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("| config | selected | portfolio | best fixed | regret | 95% CI |")
    print("|---|---|---:|---:|---:|---:|")
    for name, value in rows.items():
        ci = value.get("regret_ci95", [float("nan"), float("nan")])
        print(
            f"| {name} | {value['selected']} | {value['portfolio']:.4f} | "
            f"{value['best_fixed_oracle']:.4f} | {value['regret']:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |"
        )
    print(json.dumps(result["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
