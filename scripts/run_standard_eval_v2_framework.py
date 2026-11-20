"""Standard-metric evaluation of the **new v2 framework** on the v4 pilot.

The previous standard evaluation reused the v1/AnchorEvolve artifacts.  This
script runs the new `src/finplan_rag/decision/` planner and its baselines on the
v4 decision-level cases and reports standard metrics only:

- top-1 accuracy and macro F1 from the planner's belief;
- credible-set coverage;
- ECE (10 fixed bins);
- selective accuracy at 50% coverage and AURC;
- answer rate, wait rate, leakage.

It separates two settings:

1. **forecast-only**: waiting for the outcome filing is disabled; this is the
   fair comparison with a forecast.
2. **wait-allowed**: the planner may wait for the outcome filing; this is a
   decision-timing result, not a forecast.

The output is `paper/results/standard_eval/v2_framework.md/json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import math

from finplan_rag.decision.belief import current_q_likelihood_from_cases, position_distribution_from_cases, prior_from_cases
from finplan_rag.decision.loss import LossConfig
from finplan_rag.decision.pilot_data import load_v4_cases, split_cases
from finplan_rag.decision.policies import (
    AlwaysAdjudicate,
    AlwaysAnswer,
    AlwaysWait,
    DecisionConfig,
    FixedBudgetAdjudicate,
    FixedTopK,
    Planner,
    PlannerNoEvidence,
    PlannerNoWait,
    PolicyContext,
    PrehocRouter,
    PreciseFirst,
    default_policies,
)
from finplan_rag.decision.run_pilot import (
    DEFAULT_DATA,
    apply_company_delay,
    company_delay_map,
)

LABELS = ("below", "within", "above")
OUT = Path(__file__).resolve().parents[1] / "paper" / "results" / "standard_eval"


def _top1(belief: dict[str, float]) -> str:
    return max(LABELS, key=lambda label: (belief.get(label, 0.0), -LABELS.index(label)))


def macro_f1(y_true, y_pred, labels) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else float("nan")


def ece(probs, correct, bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if probs.size == 0:
        return float("nan")
    indices = np.clip((probs * bins).astype(int), 0, bins - 1)
    value = 0.0
    for index in range(bins):
        mask = indices == index
        if not mask.any():
            continue
        value += mask.sum() / probs.size * abs(correct[mask].mean() - probs[mask].mean())
    return float(value)


def selective_metrics(probs, correct, coverage_point: float = 0.5) -> dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if probs.size == 0:
        return {"selective_accuracy": float("nan"), "aurc": float("nan")}
    order = np.argsort(-probs, kind="stable")
    sorted_correct = correct[order]
    n = sorted_correct.size
    risks = []
    for cov in np.linspace(0.1, 1.0, 10):
        k = max(1, int(math.ceil(cov * n)))
        risks.append(1.0 - float(sorted_correct[:k].mean()))
    k = max(1, int(math.ceil(coverage_point * n)))
    return {"selective_accuracy": float(sorted_correct[:k].mean()), "aurc": float(np.mean(risks))}


def evaluate(policy, cases, context) -> dict:
    outputs = [policy.run(case, context) for case in cases]
    truth = [case.gold_hypothesis for case in cases]
    top1 = [_top1(output.belief) for output in outputs]
    correct = [1.0 if pred == gold else 0.0 for pred, gold in zip(top1, truth)]
    confidence = [float(output.belief.get(pred, 0.0)) for output, pred in zip(outputs, top1)]
    coverage = []
    for output, gold in zip(outputs, truth):
        if output.action == "answer":
            coverage.append(1.0 if output.prediction == gold else 0.0)
        elif output.action == "adjudicate":
            coverage.append(1.0 if gold in output.credible_set else 0.0)
        else:
            coverage.append(0.0)
    set_sizes = []
    for output in outputs:
        if output.action == "answer":
            set_sizes.append(1)
        elif output.action == "adjudicate":
            set_sizes.append(len(output.credible_set))
        else:
            set_sizes.append(0)
    answer_outputs = [o for o in outputs if o.action == "answer"]
    answer_accuracy = (
        sum(1.0 for o, gold in zip(outputs, truth) if o.action == "answer" and o.prediction == gold)
        / len(answer_outputs)
        if answer_outputs
        else float("nan")
    )
    return {
        "n": len(cases),
        "top1_accuracy": float(np.mean(correct)),
        "macro_f1": macro_f1(truth, top1, LABELS),
        "coverage": float(np.mean(coverage)),
        "mean_set_size": float(np.mean(set_sizes)),
        "ece": ece(confidence, correct),
        "selective_accuracy": selective_metrics(confidence, correct)["selective_accuracy"],
        "aurc": selective_metrics(confidence, correct)["aurc"],
        "answer_rate": len(answer_outputs) / len(outputs),
        "answer_accuracy": answer_accuracy,
        "adjudicate_rate": sum(1 for o in outputs if o.action == "adjudicate") / len(outputs),
        "abstain_rate": sum(1 for o in outputs if o.action == "abstain") / len(outputs),
        "wait_rate": sum(1 for o in outputs if o.decision_days > 0.0) / len(outputs),
        "leakage_rate": sum(1 for o in outputs if o.leaked) / len(outputs),
        "mean_evidence_cost": float(np.mean([o.evidence_cost for o in outputs])),
        "mean_delay_days": float(np.mean([o.decision_days for o in outputs])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    cases = load_v4_cases(args.data, include_preliminary=False)
    dev = split_cases(cases, "dev")
    test = split_cases(cases, "test")
    delay_map, global_delay = company_delay_map(dev)
    apply_company_delay(dev, delay_map, global_delay, allow_map=False)
    apply_company_delay(test, delay_map, global_delay, allow_map=True)

    prior = prior_from_cases(dev)
    position_distribution = position_distribution_from_cases(dev)
    likelihood = current_q_likelihood_from_cases(dev, prior)

    def context(allow_wait: bool) -> PolicyContext:
        return PolicyContext(
            prior=prior,
            current_q_likelihood=likelihood,
            position_distribution=position_distribution,
            config=DecisionConfig(
                loss=LossConfig(),
                gamma=0.01,
                actual_cost=2.0,
                expected_delay_days=100.0,
                wait_precision=1.0,
                allow_wait=allow_wait,
                stop_margin=1.0,
            ),
        )

    forecast_policies = [
        AlwaysAnswer(),
        AlwaysAdjudicate(),
        FixedTopK(),
        FixedBudgetAdjudicate(),
        PreciseFirst(),
        PrehocRouter(),
        PlannerNoWait(),
    ]
    wait_policies = [
        AlwaysWait(),
        PlannerNoEvidence(),
        Planner(),
    ]

    forecast = {policy.name: evaluate(policy, test, context(False)) for policy in forecast_policies}
    wait = {policy.name: evaluate(policy, test, context(True)) for policy in wait_policies}

    # Add a no-wait no-evidence ablation under the same name space.
    no_wait_no_evidence = Planner(allow_wait=False, allow_evidence=False)
    no_wait_no_evidence.name = "planner_no_wait_no_evidence"
    forecast[no_wait_no_evidence.name] = evaluate(no_wait_no_evidence, test, context(False))

    result = {
        "schema": "finplan-v2-framework-standard-eval-v1",
        "data": str(args.data),
        "split": "test",
        "n": len(test),
        "forecast_only": forecast,
        "wait_allowed": wait,
        "notes": [
            "Forecast-only disables waiting for the outcome filing; it is the fair comparison with a forecast.",
            "Wait-allowed lets the planner wait for the outcome filing; it is a decision-timing result, not a forecast.",
            "Metrics are standard: top-1 accuracy, macro F1, coverage, ECE, selective accuracy/AURC, leakage.",
            "The planner's belief is the credal belief at the decision time.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v2_framework.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(result)
    (args.out / "v2_framework.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def _table(title: str, rows: dict) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| policy | top-1 acc | macro F1 | coverage | mean set size | ECE | selective@50 | AURC | answer rate | answer acc | adjudicate rate | wait rate | leakage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in rows.items():
        lines.append(
            f"| {name} | {metrics['top1_accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['coverage']:.4f} | {metrics['mean_set_size']:.4f} | {metrics['ece']:.4f} | {metrics['selective_accuracy']:.4f} | "
            f"{metrics['aurc']:.4f} | {metrics['answer_rate']:.4f} | {metrics['answer_accuracy']:.4f} | "
            f"{metrics['adjudicate_rate']:.4f} | {metrics['wait_rate']:.4f} | {metrics['leakage_rate']:.4f} |"
        )
    return lines


def render(result: dict) -> str:
    lines = [
        "# New v2 framework: standard-metric evaluation on v4",
        "",
        f"- split: `{result['split']}` (n={result['n']})",
        f"- data: `{result['data']}`",
        "",
        "This is the **new** `src/finplan_rag/decision/` planner, not the v1/AnchorEvolve artifacts.",
        "",
    ]
    lines += _table("Forecast-only (no waiting for the outcome filing)", result["forecast_only"])
    lines += [""]
    lines += _table("Wait-allowed (decision timing; not a forecast)", result["wait_allowed"])
    lines += [
        "",
        "## Boundaries",
        "",
        "- Forecast-only is the fair comparison with a forecast; wait-allowed can see the outcome filing.",
        "- The v4 labels are silver; no human verification.",
        "- The planner's current-quarter signal is weak; this run tests the new decision layer, not a new reader.",
        "- Leakage must be 0; any positive leakage invalidates the run.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
