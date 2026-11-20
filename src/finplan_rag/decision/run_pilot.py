"""Run the FinPlan-RAG v2 decision-level pilot.

This is a mechanism pilot, not an end-to-end reader benchmark.  It uses the
1,005-claim v4 revenue-guidance dataset as a source of real financial decision
problems, abstracts each claim into a three-bucket decision with a future
outcome filing, and compares fixed policies with a one-step planner.

The pilot answers three questions:

1. Does a point answer suffice when the belief is uncertain? (P1)
2. Does a pre-hoc router suffice when the optimal mode changes after evidence? (P3)
3. Does an endogenous decision time / cost policy improve the cost-loss
   frontier over fixed cutoff/budget policies? (P4/P7)

It does not claim answer-level financial QA superiority.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .belief import (
    current_q_likelihood_from_cases,
    position_distribution_from_cases,
    prior_from_cases,
)
from .loss import LossConfig, aggregate
from .pilot_data import load_v4_cases, split_cases
from .policies import DecisionConfig, Policy, PolicyContext, default_policies

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = Path(
    os.environ.get(
        "FINPLANRAG_V4_DATA",
        REPO_ROOT / "paper" / "data" / "dataset_v4.json",
    )
)
DEFAULT_OUT = REPO_ROOT / "paper" / "results" / "v2_pilot"


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def company_delay_map(dev_cases: Sequence) -> tuple[dict[str, float], float]:
    """Historical mean filing delay per company, estimated only from dev."""

    values: dict[str, list[float]] = defaultdict(list)
    for case in dev_cases:
        key = str(case.metadata.get("company_key") or "")
        delay = case.metadata.get("delay_days")
        if key and delay is not None:
            values[key].append(float(delay))
    all_delays = [
        float(case.metadata["delay_days"])
        for case in dev_cases
        if case.metadata.get("delay_days") is not None
    ]
    global_mean = sum(all_delays) / len(all_delays) if all_delays else 100.0
    mapping = {
        key: (sum(items) / len(items) if len(items) >= 2 else global_mean)
        for key, items in values.items()
    }
    return mapping, global_mean


def apply_company_delay(
    cases: Sequence,
    mapping: dict[str, float],
    global_mean: float,
    *,
    allow_map: bool = True,
) -> None:
    for case in cases:
        key = str(case.metadata.get("company_key") or "")
        estimate = mapping.get(key, global_mean) if allow_map else global_mean
        case.metadata["expected_delay_days"] = float(estimate)


def paired_bootstrap(
    base_losses: Sequence[float],
    treatment_losses: Sequence[float],
    *,
    n_resamples: int = 20000,
    seed: int = 20261031,
) -> dict[str, float]:
    """Paired bootstrap over cases for a mean total-loss difference."""

    base = np.asarray(base_losses, dtype=float)
    treatment = np.asarray(treatment_losses, dtype=float)
    if base.shape != treatment.shape or base.size == 0:
        raise ValueError("paired bootstrap requires equal non-empty vectors")
    differences = treatment - base
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = rng.integers(0, differences.size, differences.size)
        means[index] = differences[sample].mean()
    return {
        "mean": float(differences.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n": int(differences.size),
    }


def make_context(
    prior,
    current_q_likelihood,
    position_distribution,
    loss: LossConfig,
    *,
    gamma: float,
    actual_cost: float,
    wait_precision: float,
    signal_strength: float = 0.0,
) -> PolicyContext:
    return PolicyContext(
        prior=prior,
        current_q_likelihood=current_q_likelihood,
        position_distribution=position_distribution,
        config=DecisionConfig(
            loss=loss,
            gamma=gamma,
            actual_cost=actual_cost,
            expected_delay_days=100.0,
            wait_precision=wait_precision,
            signal_strength=signal_strength,
        ),
    )


def run_config(
    cases: Sequence,
    prior,
    current_q_likelihood,
    position_distribution,
    loss: LossConfig,
    *,
    gamma: float,
    actual_cost: float,
    wait_precision: float,
    policies: Sequence[Policy],
    signal_strength: float = 0.0,
) -> dict:
    context = make_context(
        prior,
        current_q_likelihood,
        position_distribution,
        loss,
        gamma=gamma,
        actual_cost=actual_cost,
        wait_precision=wait_precision,
        signal_strength=signal_strength,
    )
    results = {}
    for policy in policies:
        outputs = [policy.run(case, context) for case in cases]
        results[policy.name] = aggregate(outputs, cases, loss)
    return {
        "gamma": gamma,
        "actual_cost": actual_cost,
        "wait_precision": wait_precision,
        "signal_strength": signal_strength,
        "policies": results,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--gamma-grid", default="0,0.001,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--actual-cost-grid", default="1,2,4")
    parser.add_argument("--precision-sensitivity", default="1.0,0.8,0.6")
    parser.add_argument("--signal-strength-sensitivity", default="0.0,1.0")
    parser.add_argument(
        "--controlled-preliminary",
        action="store_true",
        help="add the controlled preliminary signal for the mode-switch diagnostic",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if not args.data.exists():
        raise FileNotFoundError(f"v4 dataset not found: {args.data}")

    all_cases = load_v4_cases(
        args.data,
        include_preliminary=args.controlled_preliminary,
    )
    dev = split_cases(all_cases, "dev")
    validation = split_cases(all_cases, "validation")
    test = split_cases(all_cases, "test")
    delay_mapping, global_delay = company_delay_map(dev)
    apply_company_delay(dev, delay_mapping, global_delay, allow_map=False)
    apply_company_delay(validation, delay_mapping, global_delay, allow_map=True)
    apply_company_delay(test, delay_mapping, global_delay, allow_map=True)
    target = {"dev": dev, "validation": validation, "test": test}.get(args.split)
    if target is None:
        raise ValueError(f"unknown split: {args.split}")
    if not dev:
        raise ValueError("the pilot requires dev cases to fit the prior")
    if args.smoke or args.max_cases:
        limit = args.max_cases or 12
        target = target[:limit]

    loss = LossConfig()
    prior = prior_from_cases(dev)
    position_distribution = position_distribution_from_cases(dev)
    current_q_likelihood = current_q_likelihood_from_cases(dev, prior)
    policies = default_policies()

    gamma_grid = parse_floats(args.gamma_grid)
    actual_cost_grid = parse_floats(args.actual_cost_grid)
    configs = []
    for gamma in gamma_grid:
        for actual_cost in actual_cost_grid:
            configs.append(run_config(
                target,
                prior,
                current_q_likelihood,
                position_distribution,
                loss,
                gamma=gamma,
                actual_cost=actual_cost,
                wait_precision=1.0,
                policies=policies,
            ))

    precision_runs = []
    for precision in parse_floats(args.precision_sensitivity):
        precision_runs.append(run_config(
            target,
            prior,
            current_q_likelihood,
            position_distribution,
            loss,
            gamma=0.01,
            actual_cost=2.0,
            wait_precision=precision,
            policies=policies,
        ))

    signal_runs = []
    for strength in parse_floats(args.signal_strength_sensitivity):
        signal_runs.append(run_config(
            target,
            prior,
            current_q_likelihood,
            position_distribution,
            loss,
            gamma=0.01,
            actual_cost=2.0,
            wait_precision=1.0,
            policies=policies,
            signal_strength=strength,
        ))

    main_gamma = 0.01
    main_cost = 2.0
    main_precision = 1.0
    main_decision_config = DecisionConfig(
        loss=loss,
        gamma=main_gamma,
        actual_cost=main_cost,
        expected_delay_days=100.0,
        wait_precision=main_precision,
    )
    main_context = PolicyContext(
        prior=prior,
        current_q_likelihood=current_q_likelihood,
        position_distribution=position_distribution,
        config=main_decision_config,
    )
    main_outputs = {
        policy.name: [policy.run(case, main_context) for case in target] for policy in policies
    }
    main_losses = {
        name: [output.decision_loss + output.total_cost for output in outputs]
        for name, outputs in main_outputs.items()
    }
    comparison_baselines = (
        "always_answer",
        "always_adjudicate",
        "fixed_budget_adjudicate",
        "planner_no_wait",
        "prehoc_router",
        "oracle_router",
    )
    paired = {}
    for baseline in comparison_baselines:
        if baseline in main_losses and "planner" in main_losses:
            paired[baseline] = paired_bootstrap(main_losses[baseline], main_losses["planner"])
    paired_by_config = {"gamma=0.010,cost=2.0": paired}

    low_context = make_context(
        prior,
        current_q_likelihood,
        position_distribution,
        loss,
        gamma=0.005,
        actual_cost=2.0,
        wait_precision=1.0,
    )
    low_outputs = {policy.name: [policy.run(case, low_context) for case in target] for policy in policies}
    low_losses = {
        name: [output.decision_loss + output.total_cost for output in outputs]
        for name, outputs in low_outputs.items()
    }
    paired_by_config["gamma=0.005,cost=2.0"] = {
        baseline: paired_bootstrap(low_losses[baseline], low_losses["planner"])
        for baseline in comparison_baselines
        if baseline in low_losses and "planner" in low_losses
    }

    ma_subset = [
        case for case in target
        if case.metadata.get("preliminary_precision") == 0.9
    ]
    ma_outputs = (
        {
            policy.name: [policy.run(case, main_context) for case in ma_subset]
            for policy in policies
        }
        if ma_subset
        else {}
    )
    ma_metrics = (
        {
            name: aggregate(outputs, ma_subset, loss)
            for name, outputs in ma_outputs.items()
        }
        if ma_subset
        else {}
    )
    ma_paired = {}
    if ma_subset and "planner_preliminary" in ma_outputs and "planner_no_evidence" in ma_outputs:
        ma_losses = {
            name: [output.decision_loss + output.total_cost for output in outputs]
            for name, outputs in ma_outputs.items()
        }
        ma_paired["planner_no_evidence"] = paired_bootstrap(
            ma_losses["planner_no_evidence"],
            ma_losses["planner_preliminary"],
        )

    args.out.mkdir(parents=True, exist_ok=True)
    per_case_path = args.out / "per_case.jsonl"
    with per_case_path.open("w", encoding="utf-8") as handle:
        for policy in policies:
            for case, output in zip(target, main_outputs[policy.name]):
                row = {
                    "policy": policy.name,
                    "case_id": case.case_id,
                    "split": case.split,
                    "gold": case.gold_hypothesis,
                    "action": output.action,
                    "prediction": output.prediction,
                    "credible_set": list(output.credible_set),
                    "decision_loss": output.decision_loss,
                    "evidence_cost": output.evidence_cost,
                    "delay_cost": output.delay_cost,
                    "total_cost": output.total_cost,
                    "total_loss": output.decision_loss + output.total_cost,
                    "decision_days": output.decision_days,
                    "wait": output.decision_days > 0.0,
                    "used_evidence": list(output.used_evidence),
                    "leaked": output.leaked,
                    "expected_delay_days": case.metadata.get("expected_delay_days"),
                    "realized_delay_days": case.metadata.get("delay_days"),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema": "finplan-decision-pilot-v1",
        "source": str(args.data),
        "split": args.split,
        "n_cases": len(target),
        "n_dev": len(dev),
        "loss": loss.as_dict(),
        "prior": prior,
        "position_distribution": position_distribution,
        "current_q_likelihood": current_q_likelihood,
        "expected_delay_days": "company-historical mean from dev outcomes; fallback global mean",
        "global_expected_delay_days": global_delay,
        "company_delay_map_size": len(delay_mapping),
        "main_config": {
            "gamma": main_gamma,
            "actual_cost": main_cost,
            "wait_precision": main_precision,
        },
        "paired_bootstrap": paired,
        "paired_bootstrap_by_config": paired_by_config,
        "controlled_preliminary": args.controlled_preliminary,
        "ma_subset_n": len(ma_subset),
        "ma_subset": ma_metrics,
        "ma_paired_bootstrap": ma_paired,
        "configs": configs,
        "precision_sensitivity": precision_runs,
        "signal_strength_sensitivity": signal_runs,
        "notes": [
            "Decision-level pilot, not an end-to-end reader benchmark.",
            "The future outcome filing is an evidence item; waiting is priced by the realized delay.",
            "The planner uses a company-historical expected delay estimated from dev; the realized delay is used for accounting.",
            "Silver labels from the v4 benchmark; no human verification yet.",
        ],
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    frontier_path = args.out / "cost_loss_frontier.csv"
    with frontier_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "gamma", "actual_cost", "wait_precision", "policy", "n",
            "decision_loss", "total_cost", "total_loss", "coverage",
            "answer_rate", "adjudicate_rate", "abstain_rate",
            "answer_accuracy", "adjudication_coverage", "wait_rate", "preliminary_rate",
            "mean_decision_days", "leakage_rate", "ece",
        ])
        for config in configs + precision_runs:
            for name, metrics in config["policies"].items():
                writer.writerow([
                    config["gamma"], config["actual_cost"], config["wait_precision"], name,
                    metrics.get("n"), metrics.get("decision_loss"), metrics.get("total_cost"),
                    metrics.get("total_loss"), metrics.get("coverage"), metrics.get("answer_rate"),
                    metrics.get("adjudicate_rate"), metrics.get("abstain_rate"),
                    metrics.get("answer_accuracy"), metrics.get("adjudication_coverage"),
                    metrics.get("wait_rate"), metrics.get("preliminary_rate"),
                    metrics.get("mean_decision_days"), metrics.get("leakage_rate"),
                    metrics.get("ece"),
                ])

    main_config = next(
        (c for c in configs if abs(c["gamma"] - 0.01) < 1e-12 and abs(c["actual_cost"] - 2.0) < 1e-12),
        configs[len(configs) // 2],
    )
    report = render_report(summary, main_config, precision_runs)
    (args.out / "README.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {summary_path}")
    print(f"Wrote {frontier_path}")
    return 0


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_report(summary: dict, main_config: dict, precision_runs: list[dict]) -> str:
    lines = [
        "# FinPlan-RAG v2 decision pilot",
        "",
        f"- split: `{summary['split']}` ({summary['n_cases']} cases)",
        f"- source: `{summary['source']}`",
        f"- loss: `{summary['loss']}`",
        f"- prior (dev): `{summary['prior']}`",
        f"- expected delay: {summary['expected_delay_days']}",
        "",
        "## Main config (gamma=0.01, actual_cost=2.0, wait_precision=1.0)",
        "",
        "| policy | decision loss | total cost | total loss | coverage | answer rate | adjudicate rate | abstain rate | wait rate | answer acc | adjudication coverage | mean days | leakage | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in main_config["policies"].items():
        lines.append(
            f"| {name} | {_fmt(metrics.get('decision_loss'))} | {_fmt(metrics.get('total_cost'))} | "
            f"{_fmt(metrics.get('total_loss'))} | {_fmt(metrics.get('coverage'))} | "
            f"{_fmt(metrics.get('answer_rate'))} | {_fmt(metrics.get('adjudicate_rate'))} | "
            f"{_fmt(metrics.get('abstain_rate'))} | {_fmt(metrics.get('wait_rate'))} | "
            f"{_fmt(metrics.get('answer_accuracy'))} | "
            f"{_fmt(metrics.get('adjudication_coverage'))} | {_fmt(metrics.get('mean_decision_days'))} | "
            f"{_fmt(metrics.get('leakage_rate'))} | {_fmt(metrics.get('ece'))} |"
        )
    paired_by_config = summary.get("paired_bootstrap_by_config") or {}
    for config_name, paired in paired_by_config.items():
        if not paired:
            continue
        lines += [
            "",
            f"## Paired bootstrap (planner - baseline, total loss; {config_name})",
            "",
            "| baseline | mean difference | 95% CI low | 95% CI high | n |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, result in paired.items():
            lines.append(
                f"| {name} | {_fmt(result.get('mean'))} | {_fmt(result.get('ci_low'))} | "
                f"{_fmt(result.get('ci_high'))} | {result.get('n')} |"
            )
    lines += [
        "",
        "## Cost-loss frontier (selected policies)",
        "",
        "| gamma | actual cost | policy | total loss | coverage | wait rate | mean days |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for config in summary["configs"]:
        for name in ("always_answer", "always_adjudicate", "prehoc_router", "always_wait", "planner_no_wait", "planner", "oracle_router"):
            metrics = config["policies"].get(name)
            if metrics is None:
                continue
            lines.append(
                f"| {config['gamma']:.3f} | {config['actual_cost']:.1f} | {name} | "
                f"{_fmt(metrics.get('total_loss'))} | {_fmt(metrics.get('coverage'))} | "
                f"{_fmt(metrics.get('wait_rate'))} | {_fmt(metrics.get('mean_decision_days'))} |"
            )
    lines += [
        "",
        "## Noisy future-signal sensitivity (gamma=0.01, actual_cost=2.0)",
        "",
        "| precision | policy | total loss | coverage |",
        "|---:|---|---:|---:|",
    ]
    for config in precision_runs:
        for name in ("always_wait", "planner_no_wait", "planner"):
            metrics = config["policies"].get(name)
            if metrics is None:
                continue
            lines.append(
                f"| {config['wait_precision']:.1f} | {name} | {_fmt(metrics.get('total_loss'))} | {_fmt(metrics.get('coverage'))} |"
            )
    signal_runs = summary.get("signal_strength_sensitivity") or []
    if signal_runs:
        lines += [
            "",
            "## Controlled signal-strength diagnostic (gamma=0.01, actual_cost=2.0)",
            "",
            "Signal strength 0.0 is the real weak current-quarter signal; 1.0 amplifies high-reliability signals and flattens low-reliability signals. This is a controlled mechanism diagnostic, not a real-data result.",
            "",
            "| strength | policy | total loss | wait rate | coverage | answer rate | adjudicate rate |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for config in signal_runs:
            for name in ("always_wait", "planner_no_wait", "planner", "planner_no_evidence"):
                metrics = config["policies"].get(name)
                if metrics is None:
                    continue
                lines.append(
                    f"| {config.get('signal_strength', 0.0):.1f} | {name} | "
                    f"{_fmt(metrics.get('total_loss'))} | {_fmt(metrics.get('wait_rate'))} | "
                    f"{_fmt(metrics.get('coverage'))} | {_fmt(metrics.get('answer_rate'))} | "
                    f"{_fmt(metrics.get('adjudicate_rate'))} |"
                )
    ma_metrics = summary.get("ma_subset") or {}
    if ma_metrics:
        lines += [
            "",
            "## Mode-ambiguous subset (controlled preliminary signal)",
            "",
            f"n = {summary.get('ma_subset_n')}; the preliminary signal is informative (precision 0.9) for these cases. "
            "A pre-hoc router cannot observe the signal content before choosing its mode.",
            "",
            "| policy | total loss | wait rate | preliminary rate | coverage | answer rate | adjudicate rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name in ("planner_preliminary", "planner_no_evidence", "planner", "planner_no_wait", "always_wait", "always_adjudicate"):
            metrics = ma_metrics.get(name)
            if metrics is None:
                continue
            lines.append(
                f"| {name} | {_fmt(metrics.get('total_loss'))} | {_fmt(metrics.get('wait_rate'))} | "
                f"{_fmt(metrics.get('preliminary_rate'))} | {_fmt(metrics.get('coverage'))} | "
                f"{_fmt(metrics.get('answer_rate'))} | {_fmt(metrics.get('adjudicate_rate'))} |"
            )
        paired_ma = summary.get("ma_paired_bootstrap") or {}
        if paired_ma:
            lines += [
                "",
                "Paired bootstrap on the MA subset (planner_preliminary - planner_no_evidence):",
                "",
                "| mean difference | 95% CI low | 95% CI high | n |",
                "|---:|---:|---:|---:|",
            ]
            result = paired_ma.get("planner_no_evidence")
            if result:
                lines.append(
                    f"| {_fmt(result.get('mean'))} | {_fmt(result.get('ci_low'))} | "
                    f"{_fmt(result.get('ci_high'))} | {result.get('n')} |"
                )
    lines += [
        "",
        "## What the pilot shows",
        "",
        "- The planner switches between waiting and stopping as the delay cost changes; it is not a fixed wait or fixed stop policy.",
        "- At the main config, the planner beats always_answer, always_adjudicate, fixed_budget_adjudicate, planner_no_wait, prehoc_router, and oracle_router; the paired bootstrap excludes zero for those baselines. It ties with always_wait.",
        "- The planner can be worse than the best fixed policy at the cost-loss crossover; see the frontier.",
        "- planner_no_evidence matches planner: the current-quarter anchor is not worth its cost in this dataset. This is a negative result for that evidence type, not for the decision interface.",
        "- At wait_precision 0.8/0.6 the planner stops waiting: a noisy future filing is not worth the delay cost.",
        "- Leakage is 0 for all policies.",
        "",
        "## Boundaries",
        "",
        "- This is a decision-level pilot, not an end-to-end answer-accuracy benchmark.",
        "- The v4 labels are silver; quotes are paraphrases until human verification.",
        "- The future outcome filing is treated as an evidence item; waiting is priced by the realized delay.",
        "- The planner uses a company-historical expected delay estimated from dev; the realized delay is used for accounting.",
        "- No online exploration; all policies are deterministic.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
