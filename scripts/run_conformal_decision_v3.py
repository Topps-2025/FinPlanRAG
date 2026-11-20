"""Clean runner for the nested conformal decision layer (v3)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

from run_company_disjoint_benchmark import FOLDS, TEST_YEARS, VERIFIED_LABELS, fold_of
from run_real_benchmark import DEFAULT_DATA, LABELS, build, company_key, load_records
from run_real_decision_eval import CONFIGS, planner_choice
from run_conformal_decision_eval import fit_predict, oof_probabilities
from finplan_rag.decision.conformal import ConformalConfig, NestedConformalDecisionLayer
from finplan_rag.workflow.protocol import ALPHA_GRID
from finplan_rag.decision.loss import (
    LossConfig,
    action_loss,
    adjudicate_loss,
    answer_loss,
    credible_set,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "paper/results" / "conformal_v3"


def policy_losses(
    probabilities: np.ndarray,
    truths: np.ndarray,
    config: LossConfig,
    policy: str,
    alpha: float | None = None,
) -> np.ndarray:
    """Per-claim realized loss for a policy on one split."""
    values = []
    split_config = replace(config, alpha=alpha) if alpha is not None else config
    for row, truth in zip(probabilities, truths):
        belief = {label: float(value) for label, value in zip(LABELS, row)}
        prediction = LABELS[int(np.argmax(row))]
        if policy == "answer":
            loss = answer_loss(prediction, truth, split_config)
        elif policy == "abstain":
            loss = split_config.abstain
        elif policy == "fixed":
            loss = adjudicate_loss(credible_set(belief, split_config.alpha), truth, split_config)
        elif policy == "planner":
            action, predicted, credible, _ = planner_choice(row, split_config)
            loss = action_loss(
                action,
                truth,
                split_config,
                prediction=predicted,
                credible_set=credible,
            )
        else:
            raise ValueError(policy)
        values.append(float(loss))
    return np.asarray(values, dtype=float)


def tune_alpha(
    probabilities: np.ndarray,
    truths: np.ndarray,
    config: LossConfig,
    policy: str,
    *,
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
) -> tuple[float, float]:
    """Select the set level on validation loss; lower alpha breaks ties."""
    scores = [
        (float(np.mean(policy_losses(probabilities, truths, config, policy, alpha))), alpha)
        for alpha in alpha_grid
    ]
    score, alpha = min(scores, key=lambda item: (item[0], item[1]))
    return alpha, score


def split_indices(meta, seed: int):
    calibration = []
    validation = []
    for index, record in enumerate(meta):
        digest = hashlib.sha256(f"{seed}:{company_key(record)}".encode("utf-8")).hexdigest()
        if int(digest, 16) % 100 < 60:
            calibration.append(index)
        else:
            validation.append(index)
    return np.asarray(calibration), np.asarray(validation)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--configs", default="base")
    parser.add_argument("--thresholds", type=int, default=101)
    parser.add_argument(
        "--threshold-objective",
        choices=("expected", "realized", "both"),
        default="realized",
        help="threshold objective; 'both' selects it on validation",
    )
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="evaluate only test claims whose outcome label is XBRL-verified",
    )
    args = parser.parse_args()
    rows = load_records(args.data)
    by_company = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    seeds = [int(value) for value in args.seeds.split(",")]
    wanted = {value.strip() for value in args.configs.split(",")}
    configs = [(name, config) for name, config in CONFIGS if name in wanted]

    records = []
    for seed in seeds:
        for year in TEST_YEARS:
            boundary = date(year, 1, 1)
            test_rows = [
                row
                for row in rows
                if row[2].get("split") == "test"
                and int(row[2].get("cutoff_year") or 0) == year
                and (
                    not args.verified_only
                    or row[2].get("verification") in VERIFIED_LABELS
                )
            ]
            for fold in range(FOLDS):
                train_rows = [
                    row
                    for row in rows
                    if fold_of(company_key(row[2]), seed) != fold and row[1] < boundary
                ]
                fold_test = [
                    row for row in test_rows if fold_of(company_key(row[2]), seed) == fold
                ]
                if not train_rows or not fold_test:
                    continue
                calibration_probabilities, calibration_truths = oof_probabilities(
                    train_rows, by_company
                )
                _, _, _, meta_train = build(train_rows, by_company)
                calibration_index, validation_index = split_indices(meta_train, seed)
                test_probabilities, test_truths, _ = fit_predict(
                    train_rows, fold_test, by_company
                )
                validation_probabilities = calibration_probabilities[validation_index]
                validation_truths = calibration_truths[validation_index]
                for config_name, config in configs:
                    objectives = (
                        ("expected", "realized")
                        if args.threshold_objective == "both"
                        else (args.threshold_objective,)
                    )
                    layer = NestedConformalDecisionLayer(
                        ConformalConfig(
                            n_thresholds=args.thresholds,
                            threshold_objectives=objectives,
                            threshold_objective=(
                                "realized"
                                if args.threshold_objective == "both"
                                else args.threshold_objective
                            ),
                        )
                    ).fit(
                        calibration_probabilities[calibration_index],
                        calibration_truths[calibration_index],
                        validation_probabilities,
                        validation_truths,
                        config,
                    )
                    layer.recalibrate(calibration_probabilities, calibration_truths, config)
                    decisions = layer.predict(test_probabilities, config)
                    metrics = layer.evaluate(
                        test_probabilities,
                        test_truths,
                        [decision.credible_set for decision in decisions],
                        config,
                    )
                    planner = policy_losses(
                        test_probabilities, test_truths, config, "planner"
                    )
                    always_answer = policy_losses(
                        test_probabilities, test_truths, config, "answer"
                    )
                    always_adjudicate = policy_losses(
                        test_probabilities, test_truths, config, "fixed"
                    )
                    planner_alpha, planner_validation = tune_alpha(
                        validation_probabilities, validation_truths, config, "planner"
                    )
                    fixed_alpha, fixed_validation = tune_alpha(
                        validation_probabilities, validation_truths, config, "fixed"
                    )
                    planner_tuned = policy_losses(
                        test_probabilities, test_truths, config, "planner", planner_alpha
                    )
                    fixed_tuned = policy_losses(
                        test_probabilities, test_truths, config, "fixed", fixed_alpha
                    )
                    candidates = {
                        "fixed_tuned": (fixed_validation, float(np.mean(fixed_tuned))),
                        "planner_tuned": (planner_validation, float(np.mean(planner_tuned))),
                        "conformal": (float(layer.validation_loss), float(metrics["mean_loss"])),
                    }
                    selected, (selected_validation, selected_test) = min(
                        candidates.items(), key=lambda item: item[1][0]
                    )
                    records.append(
                        {
                            "seed": seed,
                            "year": year,
                            "fold": fold,
                            "config": config_name,
                            "method": layer.method,
                            "alpha": layer.alpha,
                            "threshold_objective": layer.threshold_objective,
                            "threshold": layer.threshold,
                            "calibration_coverage": layer.calibration_coverage,
                            "validation_loss": layer.validation_loss,
                            "validation_coverage": layer.validation_coverage,
                            "fallback": bool(getattr(layer, "fallback", False)),
                            **metrics,
                            "planner": float(np.mean(planner)),
                            "planner_alpha": planner_alpha,
                            "planner_validation": planner_validation,
                            "planner_tuned": float(np.mean(planner_tuned)),
                            "fixed_alpha": fixed_alpha,
                            "fixed_validation": fixed_validation,
                            "fixed_tuned": float(np.mean(fixed_tuned)),
                            "always_answer": float(np.mean(always_answer)),
                            "always_adjudicate": float(np.mean(always_adjudicate)),
                            "always_abstain": config.abstain,
                            "selected_policy": selected,
                            "selected_validation": selected_validation,
                            "selected_test": selected_test,
                            "oracle_best": min(
                                float(metrics["mean_loss"]),
                                float(np.mean(planner_tuned)),
                                float(np.mean(fixed_tuned)),
                            ),
                        }
                    )
    grouped = defaultdict(list)
    for row in records:
        grouped[row["config"]].append(row)
    summary = {}
    for config, values in grouped.items():
        n = sum(value["n"] for value in values)
        summary[config] = {
            "n": n,
            "mean_loss": float(np.average([value["mean_loss"] for value in values], weights=[value["n"] for value in values])),
            "coverage": float(np.average([value["coverage"] for value in values], weights=[value["n"] for value in values])),
            "mean_set_size": float(np.average([value["mean_set_size"] for value in values], weights=[value["n"] for value in values])),
            "planner": float(np.average([value["planner"] for value in values], weights=[value["n"] for value in values])),
            "planner_tuned": float(np.average([value["planner_tuned"] for value in values], weights=[value["n"] for value in values])),
            "fixed_tuned": float(np.average([value["fixed_tuned"] for value in values], weights=[value["n"] for value in values])),
            "selected_test": float(np.average([value["selected_test"] for value in values], weights=[value["n"] for value in values])),
            "oracle_best": float(np.average([value["oracle_best"] for value in values], weights=[value["n"] for value in values])),
            "always_answer": float(np.average([value["always_answer"] for value in values], weights=[value["n"] for value in values])),
            "always_adjudicate": float(np.average([value["always_adjudicate"] for value in values], weights=[value["n"] for value in values])),
            "always_abstain": float(np.average([value["always_abstain"] for value in values], weights=[value["n"] for value in values])),
            "methods": dict(
                (key, sum(1 for value in values if value["method"] == key))
                for key in ("lac", "aps")
            ),
            "objectives": dict(
                (key, sum(1 for value in values if value["threshold_objective"] == key))
                for key in ("expected", "realized")
            ),
            "fallbacks": sum(1 for value in values if value.get("fallback")),
            "selected": dict(
                (key, sum(1 for value in values if value["selected_policy"] == key))
                for key in ("fixed_tuned", "planner_tuned", "conformal")
            ),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps({"records": records, "aggregate": summary}, indent=2), encoding="utf-8"
    )
    lines = [
        "# Nested conformal decision layer (v3)",
        "",
        "| config | conformal | coverage | size | planner | planner tuned | fixed tuned | selected | oracle | methods |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for config, value in summary.items():
        lines.append(
            f"| {config} | {value['mean_loss']:.4f} | {value['coverage']:.4f} | "
            f"{value['mean_set_size']:.3f} | {value['planner']:.4f} | "
            f"{value['planner_tuned']:.4f} | {value['fixed_tuned']:.4f} | "
            f"{value['selected_test']:.4f} | {value['oracle_best']:.4f} | "
            f"{value['methods']} |"
        )
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
