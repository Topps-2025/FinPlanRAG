"""Decision-layer evaluation on the year-wise company-disjoint benchmark.

The evidence model from ``run_company_disjoint_benchmark.py`` produces a
three-way probability vector for each claim.  This script treats that vector as
the credal state and compares decision policies under the v2 asymmetric loss:

- ``always_answer``: answer with the argmax label;
- ``always_adjudicate``: output the smallest credible set at ``alpha``;
- ``always_abstain``: pay the abstention cost;
- ``fixed_conf_t``: answer when max probability >= t, otherwise abstain;
- ``planner``: choose the action with the lowest expected loss given the
  probabilities (the Bayes action for the loss).

The benchmark is company-disjoint and point-in-time: for test year Y the model
is trained on other company folds using outcomes public before January 1 of Y.
Five fold seeds are run.  We report a loss-parameter grid because the loss
weights are a modeling choice; the planner is compared with fixed policies at
each setting, so the comparison does not depend on one hand-picked weight.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_company_disjoint_benchmark import (
    DEFAULT_DATA,
    FOLD_SEEDS,
    run_fold_seed,
)
from run_real_benchmark import LABELS, company_key, load_records
from finplan_rag.workflow.protocol import BASE, CONFIGS
from finplan_rag.decision.loss import (
    LossConfig,
    action_loss,
    credible_set,
    expected_action_loss,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "paper" / "results" / "real_decision_eval"

THRESHOLDS = (0.40, 0.50, 0.60, 0.70)


def argmax_label(probabilities: np.ndarray) -> str:
    return str(LABELS[int(np.argmax(probabilities))])


def belief_of(probabilities: np.ndarray) -> dict[str, float]:
    return {label: float(value) for label, value in zip(LABELS, probabilities)}


def planner_choice(probabilities: np.ndarray, config: LossConfig) -> tuple[str, str, tuple[str, ...], dict[str, float]]:
    belief = belief_of(probabilities)
    prediction = argmax_label(probabilities)
    adjudication = credible_set(belief, config.alpha)
    values = {
        "answer": expected_action_loss("answer", belief, config, prediction=prediction),
        "adjudicate": expected_action_loss("adjudicate", belief, config, credible_set=adjudication),
        "abstain": config.abstain,
    }
    action = min(("answer", "adjudicate", "abstain"), key=lambda name: values[name])
    return action, prediction, adjudication, values


def choose(action: str, probabilities: np.ndarray, config: LossConfig, threshold: float = 0.0) -> tuple[str, str, tuple[str, ...]]:
    prediction = argmax_label(probabilities)
    if action == "always_answer":
        return "answer", prediction, ()
    if action == "always_adjudicate":
        return "adjudicate", prediction, credible_set(belief_of(probabilities), config.alpha)
    if action == "always_abstain":
        return "abstain", prediction, ()
    if action == "fixed_conf":
        if float(np.max(probabilities)) >= threshold:
            return "answer", prediction, ()
        return "abstain", prediction, ()
    if action == "planner":
        selected, prediction, adjudication, _ = planner_choice(probabilities, config)
        return selected, prediction, adjudication
    raise ValueError(action)


def evaluate_policy(truths, probabilities, action: str, config: LossConfig, threshold: float = 0.0) -> dict:
    losses = []
    covered = 0
    answers = []
    answer_correct = 0
    adjudications = []
    adjudication_covered = 0
    set_sizes = []
    action_counts = defaultdict(int)
    confidences = []
    for truth, probability in zip(truths, probabilities):
        probability = np.asarray(probability, dtype=float)
        selected, prediction, adjudication = choose(action, probability, config, threshold)
        loss = action_loss(selected, truth, config, prediction=prediction, credible_set=adjudication)
        losses.append(float(loss))
        action_counts[selected] += 1
        if selected == "answer":
            answers.append(selected)
            correct = int(prediction == truth)
            answer_correct += correct
            covered += correct
        elif selected == "adjudicate":
            adjudications.append(selected)
            contained = int(truth in adjudication)
            adjudication_covered += contained
            covered += contained
            set_sizes.append(len(adjudication))
        else:
            answers.append("abstain")
        if selected in ("answer", "adjudicate"):
            confidences.append(float(np.max(probability)))
    n = len(truths)
    answer_count = action_counts["answer"]
    adjudicate_count = action_counts["adjudicate"]
    return {
        "n": n,
        "mean_loss": float(np.mean(losses)),
        "coverage": covered / n,
        "answer_rate": answer_count / n,
        "adjudicate_rate": adjudicate_count / n,
        "abstain_rate": action_counts["abstain"] / n,
        "answer_accuracy": (answer_correct / answer_count) if answer_count else None,
        "adjudication_coverage": (adjudication_covered / adjudicate_count) if adjudicate_count else None,
        "mean_set_size": float(np.mean(set_sizes)) if set_sizes else 0.0,
    }


def paired_bootstrap(
    truths,
    probabilities,
    config: LossConfig,
    *,
    baseline: str = "always_answer",
    n_resamples: int = 5000,
    seed: int = 20261031,
) -> dict:
    """Planner minus a fixed baseline, paired over claims."""

    diffs = []
    for truth, probability in zip(truths, probabilities):
        probability = np.asarray(probability, dtype=float)
        planner_action, prediction, adjudication, _ = planner_choice(probability, config)
        planner_loss = action_loss(planner_action, truth, config, prediction=prediction, credible_set=adjudication)
        baseline_action, baseline_prediction, baseline_adjudication = choose(
            baseline, probability, config
        )
        baseline_loss = action_loss(
            baseline_action,
            truth,
            config,
            prediction=baseline_prediction,
            credible_set=baseline_adjudication,
        )
        diffs.append(float(planner_loss) - float(baseline_loss))
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = rng.integers(0, len(diffs), len(diffs))
        values[index] = float(diffs[sample].mean())
    observed = float(diffs.mean())
    flips = rng.choice((-1.0, 1.0), size=(n_resamples, len(diffs)))
    permuted = np.abs((flips * diffs).mean(axis=1))
    return {
        "mean": observed,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "sign_flip_p": float(np.mean(permuted >= abs(observed))),
        "n": len(diffs),
    }


def aggregate_seed(truths, probabilities, configs) -> dict:
    rows = []
    for name, config in configs:
        row = {"config": name, "loss": config.as_dict()}
        for policy in ("always_answer", "always_adjudicate", "always_abstain", "planner"):
            row[policy] = evaluate_policy(truths, probabilities, policy, config)
        for threshold in THRESHOLDS:
            row[f"fixed_conf_{threshold:.2f}"] = evaluate_policy(
                truths, probabilities, "fixed_conf", config, threshold
            )
        row["planner_minus_answer"] = paired_bootstrap(
            truths, probabilities, config, baseline="always_answer"
        )
        row["planner_minus_adjudicate"] = paired_bootstrap(
            truths, probabilities, config, baseline="always_adjudicate"
        )
        rows.append(row)
    return {"rows": rows}


def cross_loss_analysis(truths, probabilities, configs) -> dict:
    """Planner optimized under one loss, scored under another.

    The diagonal is the matched-loss setting used in the main table.  The
    off-diagonal cells test what happens when the analyst's loss is
    misspecified.  Fixed policies do not depend on the nominal loss.
    """
    fixed = {}
    for eval_name, eval_config in configs:
        fixed[eval_name] = min(
            evaluate_policy(truths, probabilities, policy, eval_config)["mean_loss"]
            for policy in ("always_answer", "always_adjudicate", "always_abstain")
        )
    matrix = {}
    for nominal_name, nominal_config in configs:
        row = {}
        for eval_name, eval_config in configs:
            losses = []
            for truth, probability in zip(truths, probabilities):
                probability = np.asarray(probability, dtype=float)
                action, prediction, adjudication, _ = planner_choice(probability, nominal_config)
                losses.append(
                    action_loss(
                        action,
                        truth,
                        eval_config,
                        prediction=prediction,
                        credible_set=adjudication,
                    )
                )
            row[eval_name] = float(np.mean(losses)) - fixed[eval_name]
        matrix[nominal_name] = row
    diagonal = [matrix[name][name] for name, _ in configs]
    off_diagonal = [
        matrix[nominal][eval_name]
        for nominal, _ in configs
        for eval_name, _ in configs
        if nominal != eval_name
    ]
    return {
        "matrix": matrix,
        "diagonal_mean": float(np.mean(diagonal)),
        "off_diagonal_mean": float(np.mean(off_diagonal)),
        "off_diagonal_max": float(np.max(off_diagonal)),
        "per_nominal_mean": {
            nominal: float(np.mean(list(row.values()))) for nominal, row in matrix.items()
        },
        "per_eval_best_nominal": {
            eval_name: min(
                ((matrix[nominal][eval_name], nominal) for nominal, _ in configs)
            )[1]
            for eval_name, _ in configs
        },
    }


def summarize_cross(matrix, configs) -> dict:
    names = [name for name, _ in configs]
    diagonal = [matrix[name][name] for name in names]
    off_diagonal = [
        matrix[nominal][eval_name]
        for nominal in names
        for eval_name in names
        if nominal != eval_name
    ]
    return {
        "matrix": matrix,
        "diagonal_mean": float(np.mean(diagonal)),
        "off_diagonal_mean": float(np.mean(off_diagonal)),
        "off_diagonal_max": float(np.max(off_diagonal)),
        "per_nominal_mean": {
            nominal: float(np.mean(list(matrix[nominal].values()))) for nominal in names
        },
        "per_eval_best_nominal": {
            eval_name: min(
                ((matrix[nominal][eval_name], nominal) for nominal in names)
            )[1]
            for eval_name in names
        },
    }


def render(summary: dict) -> str:
    mode = "calibrated" if summary.get("calibrated") else "raw"
    lines = [
        f"# Decision-layer evaluation on real company-disjoint data ({mode} probabilities)",
        "",
        f"- data: `{summary['data']}`",
        f"- protocol: {summary['protocol']}",
        f"- claims per seed: {summary['n_claims']}",
        "",
        f"- temperature range: {min(t['temperature'] for row in summary['per_seed'] for t in row['temperatures']):.3f}"
        f" to {max(t['temperature'] for row in summary['per_seed'] for t in row['temperatures']):.3f}",
        "",
        "## Base loss configuration",
        "",
        "Loss: wrong={wrong}, over={over}, under={under}, miss={miss}, width={width}, "
        "abstain={abstain}, alpha={alpha}.".format(**BASE.as_dict()),
        "",
        "| seed | policy | mean loss | coverage | answer rate | adjudicate rate | abstain rate | answer acc | set size |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed_row in summary["per_seed"]:
        base = next(row for row in seed_row["rows"] if row["config"] == "base")
        for policy in ("always_answer", "always_adjudicate", "always_abstain", "planner"):
            item = base[policy]
            lines.append(
                f"| {seed_row['seed']} | {policy} | {item['mean_loss']:.4f} | {item['coverage']:.4f} | "
                f"{item['answer_rate']:.4f} | {item['adjudicate_rate']:.4f} | {item['abstain_rate']:.4f} | "
                f"{item['answer_accuracy'] if item['answer_accuracy'] is not None else float('nan'):.4f} | "
                f"{item['mean_set_size']:.4f} |"
            )
        diff = base["planner_minus_answer"]
        lines.append(
            f"| {seed_row['seed']} | **planner - answer** | **{diff['mean']:+.4f}** "
            f"[{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}], p={diff['sign_flip_p']:.3g} | | | | | | |"
        )
        diff_adj = base["planner_minus_adjudicate"]
        lines.append(
            f"| {seed_row['seed']} | **planner - adjudicate** | **{diff_adj['mean']:+.4f}** "
            f"[{diff_adj['ci_low']:+.4f}, {diff_adj['ci_high']:+.4f}], p={diff_adj['sign_flip_p']:.3g} | | | | | | |"
        )
    lines += [
        "",
        "## Loss-parameter grid (mean over five fold seeds)",
        "",
        "| config | always answer | always adjudicate | always abstain | planner | planner - answer [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config_name, _ in CONFIGS:
        values = []
        for seed_row in summary["per_seed"]:
            row = next(item for item in seed_row["rows"] if item["config"] == config_name)
            values.append(row)
        def mean_metric(policy: str) -> float:
            return float(np.mean([item[policy]["mean_loss"] for item in values]))
        def mean_diff(key: str) -> float:
            return float(np.mean([item[key]["mean"] for item in values]))
        diff_low = float(np.mean([item["planner_minus_answer"]["ci_low"] for item in values]))
        diff_high = float(np.mean([item["planner_minus_answer"]["ci_high"] for item in values]))
        lines.append(
            f"| {config_name} | {mean_metric('always_answer'):.4f} | {mean_metric('always_adjudicate'):.4f} | "
            f"{mean_metric('always_abstain'):.4f} | {mean_metric('planner'):.4f} | "
            f"{mean_diff('planner_minus_answer'):+.4f} [{diff_low:+.4f}, {diff_high:+.4f}] |"
        )
    lines += [
        "",
        "## Regret relative to the best fixed policy across the loss grid",
        "",
        "| policy | mean regret | max regret | configs where best |",
        "|---|---:|---:|---:|",
    ]
    config_losses = {}
    for config_name, _ in CONFIGS:
        config_losses[config_name] = {}
        for policy in ("always_answer", "always_adjudicate", "always_abstain", "planner"):
            values = []
            for seed_row in summary["per_seed"]:
                row = next(item for item in seed_row["rows"] if item["config"] == config_name)
                values.append(row[policy]["mean_loss"])
            config_losses[config_name][policy] = float(np.mean(values))
    for policy in ("always_answer", "always_adjudicate", "always_abstain", "planner"):
        regrets = []
        best_count = 0
        for config_name, _ in CONFIGS:
            fixed = min(
                config_losses[config_name]["always_answer"],
                config_losses[config_name]["always_adjudicate"],
                config_losses[config_name]["always_abstain"],
            )
            regret = config_losses[config_name][policy] - fixed
            regrets.append(regret)
            if regret <= 1e-9:
                best_count += 1
        lines.append(
            f"| {policy} | {float(np.mean(regrets)):+.4f} | {float(np.max(regrets)):+.4f} | {best_count}/{len(CONFIGS)} |"
        )
    cross = summary["cross_loss"]
    lines += [
        "",
        "## Loss misspecification",
        "",
        f"- matched-loss diagonal mean regret: {cross['diagonal_mean']:+.4f}",
        f"- off-diagonal mean regret: {cross['off_diagonal_mean']:+.4f}; "
        f"maximum: {cross['off_diagonal_max']:+.4f}",
        "",
        "| nominal loss | mean regret across evaluation losses |",
        "|---|---:|",
    ]
    for nominal, value in cross["per_nominal_mean"].items():
        lines.append(f"| {nominal} | {value:+.4f} |")
    lines += [
        "",
        "## Reading",
        "",
        "- The planner is the Bayes action for the stated loss; it is not tuned on the test labels.",
        "- The model is trained without the test company and only on outcomes public before the test year.",
        "- Negative planner-minus-answer differences mean the planner has lower decision loss.",
        "- The loss grid is reported instead of a single weight vector; conclusions should hold across settings.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="temperature-scale probabilities on internal company-disjoint folds",
    )
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="evaluate only test claims whose outcome label is XBRL-verified",
    )
    args = parser.parse_args()

    rows = load_records(args.data)
    by_company: dict[str, list] = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)

    per_seed = []
    all_truths = []
    all_probabilities = []
    all_keys = []
    cross_by_seed = []
    for seed in FOLD_SEEDS:
        result = run_fold_seed(
            rows,
            by_company,
            seed,
            calibrate=args.calibrate,
            verified_only=args.verified_only,
        )
        truths = result["truths"]
        probabilities = np.asarray(result["probabilities"], dtype=float)
        keys = result["keys"]
        per_seed.append(
            {
                "seed": seed,
                "temperatures": result["temperatures"],
                **aggregate_seed(truths, probabilities, CONFIGS),
            }
        )
        all_truths.extend(truths)
        all_probabilities.extend(probabilities.tolist())
        all_keys.extend(keys)
        cross_by_seed.append(cross_loss_analysis(truths, probabilities, CONFIGS))
        base = next(row for row in per_seed[-1]["rows"] if row["config"] == "base")
        print(
            f"seed {seed}: planner={base['planner']['mean_loss']:.4f} "
            f"answer={base['always_answer']['mean_loss']:.4f} "
            f"adjudicate={base['always_adjudicate']['mean_loss']:.4f} "
            f"abstain={base['always_abstain']['mean_loss']:.4f} "
            f"planner-answer={base['planner_minus_answer']['mean']:+.4f} "
            f"[{base['planner_minus_answer']['ci_low']:+.4f},{base['planner_minus_answer']['ci_high']:+.4f}]"
        )

    cross_names = [name for name, _ in CONFIGS]
    cross_matrix = {
        nominal: {
            eval_name: float(
                np.mean([seed_cross["matrix"][nominal][eval_name] for seed_cross in cross_by_seed])
            )
            for eval_name in cross_names
        }
        for nominal in cross_names
    }
    cross_summary = summarize_cross(cross_matrix, CONFIGS)

    summary = {
        "schema": "finplan-real-decision-eval-v1",
        "data": str(args.data),
        "protocol": {
            "folds": 5,
            "fold_seeds": list(FOLD_SEEDS),
            "test_years": [2022, 2023, 2024, 2025],
            "training": "other company folds, outcome date before January 1 of the test year",
            "company_disjoint": True,
            "point_in_time": True,
            "model": "logistic regression, C=1.0, class_weight=balanced",
            "probability_calibration": (
                "temperature scaling on internal company-disjoint folds"
                if args.calibrate
                else "none"
            ),
            "verified_only": bool(args.verified_only),
        },
        "calibrated": bool(args.calibrate),
        "n_claims": len(all_truths) // len(FOLD_SEEDS),
        "per_seed": per_seed,
        "cross_loss": cross_summary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(summary)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
