"""Prototype: minimax-regret planner over a set of loss configurations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_company_disjoint_benchmark import DEFAULT_DATA, FOLD_SEEDS, run_fold_seed
from run_real_benchmark import LABELS, company_key, load_records
from run_real_decision_eval import CONFIGS, evaluate_policy
from finplan_rag.decision.loss import action_loss, credible_set, expected_action_loss


def belief(probability):
    return {label: float(value) for label, value in zip(LABELS, probability)}


def best_fixed_expected(belief_state, config):
    values = [expected_action_loss("answer", belief_state, config, prediction=label) for label in LABELS]
    values.append(
        expected_action_loss(
            "adjudicate", belief_state, config, credible_set=credible_set(belief_state, config.alpha)
        )
    )
    values.append(config.abstain)
    return min(values)


def robust_choice(probability, configs):
    state = belief(probability)
    actions = {}
    for label in LABELS:
        actions[("answer", label, ())] = ("answer", label, ())
    for _, config in configs:
        chosen = credible_set(state, config.alpha)
        actions[("adjudicate", max(LABELS, key=lambda label: state[label]), chosen)] = (
            "adjudicate",
            max(LABELS, key=lambda label: state[label]),
            chosen,
        )
    actions[("abstain", max(LABELS, key=lambda label: state[label]), ())] = (
        "abstain",
        max(LABELS, key=lambda label: state[label]),
        (),
    )
    best = None
    best_regret = float("inf")
    for action in actions.values():
        regrets = []
        for _, config in configs:
            loss = expected_action_loss(
                action[0], state, config, prediction=action[1], credible_set=action[2]
            )
            regrets.append(loss - best_fixed_expected(state, config))
        worst = max(regrets)
        if worst < best_regret:
            best_regret = worst
            best = action
    return best


def main() -> int:
    rows = load_records(DEFAULT_DATA)
    by_company = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    results = []
    for seed in FOLD_SEEDS:
        result = run_fold_seed(rows, by_company, seed, calibrate=True, verified_only=True)
        truths = result["truths"]
        probabilities = np.asarray(result["probabilities"], dtype=float)
        row = {"seed": seed, "matched": [], "robust": []}
        for name, config in CONFIGS:
            fixed = min(
                evaluate_policy(truths, probabilities, policy, config)["mean_loss"]
                for policy in ("always_answer", "always_adjudicate", "always_abstain")
            )
            matched = evaluate_policy(truths, probabilities, "planner", config)["mean_loss"]
            robust_losses = []
            for truth, probability in zip(truths, probabilities):
                action = robust_choice(probability, CONFIGS)
                robust_losses.append(
                    action_loss(action[0], truth, config, prediction=action[1], credible_set=action[2])
                )
            row["matched"].append(float(matched - fixed))
            row["robust"].append(float(np.mean(robust_losses)) - fixed)
            print(
                f"seed {seed} {name}: matched {row['matched'][-1]:+.4f} "
                f"robust {row['robust'][-1]:+.4f}"
            )
        results.append(row)
        print(
            f"seed {seed}: matched mean/max {np.mean(row['matched']):+.4f}/{np.max(row['matched']):+.4f}; "
            f"robust mean/max {np.mean(row['robust']):+.4f}/{np.max(row['robust']):+.4f}"
        )
    Path("paper/results/robust_planner_prototype.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
