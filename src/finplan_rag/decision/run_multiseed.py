"""Multi-seed evaluation of the v2 framework with standard metrics.

The experiment has two tracks:

1. **Controlled mode-switch diagnostic**: v4 cases plus a controlled
   preliminary signal whose precision assignment is seeded.  The primary
   metric is top-1 accuracy under an evidence/delay cost budget of 1.0 on the
   informative (precision 0.9) subset.
2. **Real v4**: the same cases without the preliminary signal.  This is the
   real-data external-validity check.

Each seed changes the controlled signal assignment.  The script reports
mean/std/95% CI across seeds and paired differences between
``planner_preliminary`` and the pre-hoc cost-aware router
(``planner_no_evidence``).

No asymmetric decision-loss weights are used for the primary metric.  The
budget-accuracy curve is the main result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .belief import current_q_likelihood_from_cases, position_distribution_from_cases, prior_from_cases
from .loss import LossConfig
from .metrics import aggregate_seeds, evaluate
from .pilot_data import load_v4_cases, split_cases
from .policies import (
    AlwaysAdjudicate,
    AlwaysAnswer,
    AlwaysWait,
    DecisionConfig,
    Planner,
    PlannerNoEvidence,
    PlannerNoWait,
    PlannerPreliminary,
    PolicyContext,
)
from .run_pilot import DEFAULT_DATA, apply_company_delay, company_delay_map

OUT = Path(__file__).resolve().parents[3] / "paper" / "results" / "v2_multiseed"
BUDGETS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, end = part.split(":", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def make_context(prior, likelihood, position_distribution, *, gamma: float, actual_cost: float, stop_margin: float) -> PolicyContext:
    return PolicyContext(
        prior=prior,
        current_q_likelihood=likelihood,
        position_distribution=position_distribution,
        config=DecisionConfig(
            loss=LossConfig(),
            gamma=gamma,
            actual_cost=actual_cost,
            expected_delay_days=100.0,
            wait_precision=1.0,
            allow_wait=True,
            stop_margin=stop_margin,
        ),
    )


def policies() -> list:
    return [
        AlwaysAnswer(),
        AlwaysAdjudicate(),
        AlwaysWait(),
        PlannerNoWait(),
        PlannerNoEvidence(),
        Planner(),
        PlannerPreliminary(),
    ]


def run_seed(
    data: Path,
    seed: int,
    *,
    include_preliminary: bool,
    gamma: float,
    actual_cost: float,
    stop_margin: float,
) -> dict:
    cases = load_v4_cases(data, include_preliminary=include_preliminary, seed=seed)
    dev = split_cases(cases, "dev")
    test = split_cases(cases, "test")
    delay_map, global_delay = company_delay_map(dev)
    apply_company_delay(dev, delay_map, global_delay, allow_map=False)
    apply_company_delay(test, delay_map, global_delay, allow_map=True)
    prior = prior_from_cases(dev)
    position = position_distribution_from_cases(dev)
    likelihood = current_q_likelihood_from_cases(dev, prior)
    context = make_context(
        prior,
        likelihood,
        position,
        gamma=gamma,
        actual_cost=actual_cost,
        stop_margin=stop_margin,
    )
    ma_subset = [case for case in test if case.metadata.get("preliminary_precision") == 0.9]
    results = {"seed": seed, "n_test": len(test), "n_ma": len(ma_subset), "policies": {}}
    for policy in policies():
        outputs = [policy.run(case, context) for case in test]
        ma_cases = ma_subset
        ma_outputs = (
            [policy.run(case, context) for case in ma_cases]
            if include_preliminary
            else []
        )
        results["policies"][policy.name] = {
            "full": evaluate(outputs, test, BUDGETS),
            "ma": evaluate(ma_outputs, ma_cases, BUDGETS) if ma_outputs else None,
        }
    return results


def flatten(metrics: dict, prefix: str = "") -> dict[str, float]:
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}."))
        elif isinstance(value, (int, float)):
            flat[f"{prefix}{key}"] = float(value)
    return flat


def aggregate(per_seed: Sequence[dict], subset: str, policy: str) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {}
    for seed_result in per_seed:
        metrics = seed_result["policies"][policy][subset]
        if metrics is None:
            continue
        for key, value in flatten(metrics).items():
            values.setdefault(key, []).append(value)
    return {key: aggregate_seeds(items) for key, items in values.items()}


def paired_difference(
    per_seed: Sequence[dict],
    subset: str,
    metric: str,
    treatment: str,
    baseline: str,
) -> dict:
    differences = []
    for seed_result in per_seed:
        treat = seed_result["policies"].get(treatment, {}).get(subset)
        base = seed_result["policies"].get(baseline, {}).get(subset)
        if treat is None or base is None:
            continue
        differences.append(flatten(treat).get(metric) - flatten(base).get(metric))
    stats = aggregate_seeds(differences)
    stats["positive_seeds"] = sum(1 for value in differences if value > 0)
    stats["negative_seeds"] = sum(1 for value in differences if value < 0)
    stats["values"] = differences
    return stats


def format_ci(stats: dict, digits: int = 3) -> str:
    return f"{stats['mean']:.{digits}f} [{stats['ci_low']:.{digits}f}, {stats['ci_high']:.{digits}f}]"


def render(controlled: dict, real: dict) -> str:
    seeds = controlled["seeds"]
    lines = [
        "# FinPlan-RAG v2 multi-seed evaluation",
        "",
        f"- seeds: {seeds[0]}..{seeds[-1]} (n={len(seeds)})",
        f"- data: `{controlled['data']}`",
        f"- gamma: {controlled['gamma']}, actual_cost: {controlled['actual_cost']}, stop_margin: {controlled['stop_margin']}",
        f"- primary metric: top-1 accuracy at cost budget 1.0 on the controlled MA subset (precision 0.9)",
        "",
        "## Controlled mode-switch diagnostic: MA subset",
        "",
        "| policy | top-1 acc | macro F1 | coverage | set size | ECE | Brier | log loss | selective@50 | AURC | acc@budget .5 | acc@budget 1.0 | acc@budget 3.0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ("always_answer", "always_adjudicate", "always_wait", "planner_no_wait", "planner_no_evidence", "planner", "planner_preliminary"):
        metrics = controlled["aggregate"]["ma"].get(policy)
        if not metrics:
            continue
        def cell(key: str) -> str:
            return format_ci(metrics[key]) if key in metrics else "n/a"
        lines.append(
            f"| {policy} | {cell('top1_accuracy')} | {cell('macro_f1')} | {cell('coverage')} | "
            f"{cell('mean_set_size')} | {cell('ece')} | {cell('brier')} | {cell('log_loss')} | "
            f"{cell('selective_accuracy')} | {cell('aurc')} | {cell('budgeted_accuracy.0.50')} | "
            f"{cell('budgeted_accuracy.1.00')} | {cell('budgeted_accuracy.3.00')} |"
        )
    primary = controlled["paired"]["ma:budgeted_accuracy.1.00"]
    lines += [
        "",
        "**Paired difference (planner_preliminary - planner_no_evidence, top-1 accuracy at budget 1.0):**",
        "",
        f"- mean: {primary['mean']:.4f}",
        f"- 95% CI: [{primary['ci_low']:.4f}, {primary['ci_high']:.4f}]",
        f"- positive seeds: {primary['positive_seeds']}/{primary['n']}",
        "",
        "## Real v4 (no preliminary signal): MA subset is empty; full test",
        "",
        "| policy | top-1 acc | macro F1 | coverage | set size | ECE | Brier | log loss | acc@budget 0 | acc@budget 1.0 | acc@budget 3.0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ("always_answer", "always_adjudicate", "always_wait", "planner_no_wait", "planner_no_evidence", "planner", "planner_preliminary"):
        metrics = real["aggregate"]["full"].get(policy)
        if not metrics:
            continue
        def cell2(key: str) -> str:
            return format_ci(metrics[key]) if key in metrics else "n/a"
        lines.append(
            f"| {policy} | {cell2('top1_accuracy')} | {cell2('macro_f1')} | {cell2('coverage')} | "
            f"{cell2('mean_set_size')} | {cell2('ece')} | {cell2('brier')} | {cell2('log_loss')} | "
            f"{cell2('budgeted_accuracy.0.00')} | {cell2('budgeted_accuracy.1.00')} | {cell2('budgeted_accuracy.3.00')} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- The controlled MA result is a mechanism result: the planner observes a seeded preliminary signal and stops at low cost; the pre-hoc router cannot.",
        "- The real v4 result is the external-validity check. If it is not positive, the paper must not claim a real-data method win.",
        "- Top-1 accuracy at budget 1.0 is a standard accuracy-under-cost metric; it does not use the asymmetric decision-loss weights.",
        "- The controlled preliminary signal is synthetic; do not present it as real financial evidence.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seeds", default="1:20")
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--actual-cost", type=float, default=2.0)
    parser.add_argument("--stop-margin", type=float, default=1.0)
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    controlled_runs = [
        run_seed(
            args.data,
            seed,
            include_preliminary=True,
            gamma=args.gamma,
            actual_cost=args.actual_cost,
            stop_margin=args.stop_margin,
        )
        for seed in seeds
    ]
    real_runs = [
        run_seed(
            args.data,
            seed,
            include_preliminary=False,
            gamma=args.gamma,
            actual_cost=args.actual_cost,
            stop_margin=args.stop_margin,
        )
        for seed in seeds
    ]

    controlled_aggregate = {
        "full": {policy: aggregate(controlled_runs, "full", policy) for policy in (p.name for p in policies())},
        "ma": {policy: aggregate(controlled_runs, "ma", policy) for policy in (p.name for p in policies())},
    }
    real_aggregate = {
        "full": {policy: aggregate(real_runs, "full", policy) for policy in (p.name for p in policies())},
        "ma": {},
    }
    controlled = {
        "data": str(args.data),
        "seeds": seeds,
        "gamma": args.gamma,
        "actual_cost": args.actual_cost,
        "stop_margin": args.stop_margin,
        "runs": controlled_runs,
        "aggregate": controlled_aggregate,
        "paired": {
            "ma:budgeted_accuracy.1.00": paired_difference(
                controlled_runs,
                "ma",
                "budgeted_accuracy.1.00",
                "planner_preliminary",
                "planner_no_evidence",
            ),
            "ma:top1_accuracy": paired_difference(
                controlled_runs,
                "ma",
                "top1_accuracy",
                "planner_preliminary",
                "planner_no_evidence",
            ),
        },
    }
    real = {
        "data": str(args.data),
        "seeds": seeds,
        "gamma": args.gamma,
        "actual_cost": args.actual_cost,
        "stop_margin": args.stop_margin,
        "runs": real_runs,
        "aggregate": real_aggregate,
        "paired": {},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "controlled.json").write_text(json.dumps(controlled, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "real.json").write_text(json.dumps(real, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(controlled, real)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
