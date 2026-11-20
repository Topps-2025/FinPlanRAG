"""Nested conformal decision layer with validation-based score/alpha selection.

For each year/fold:
1. OOF probabilities on training companies.
2. Split training companies into calibration (60%) and validation (40%).
3. For each (method, alpha), choose the conformal threshold on calibration and
   evaluate it on validation. Select the method/alpha with the lowest validation
   loss among those satisfying validation coverage >= 1-alpha.
4. Recompute the threshold on the full OOF calibration set for the selected
   method/alpha and evaluate the test year once.

No test outcome is used for threshold or method selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from run_company_disjoint_benchmark import FOLDS, TEST_YEARS, apply_temperature, fit_temperature, fold_of
from run_real_benchmark import DEFAULT_DATA, LABELS, company_key, load_records
from run_real_decision_eval import CONFIGS, planner_choice
from run_conformal_decision_eval import (
    choose_threshold,
    evaluate,
    fit_predict,
    oof_probabilities,
    sets_from_scores,
)
from finplan_rag.decision.loss import action_loss, adjudicate_loss, answer_loss, credible_set

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "paper/results" / "conformal_selection"
METHODS = ("lac", "aps")
ALPHAS = (0.05, 0.10, 0.20)


def assign_split(meta, seed: int):
    calibrate = []
    validate = []
    for index, record in enumerate(meta):
        digest = hashlib.sha256(f"{seed}:{company_key(record)}".encode("utf-8")).hexdigest()
        if int(digest, 16) % 100 < 60:
            calibrate.append(index)
        else:
            validate.append(index)
    return np.asarray(calibrate), np.asarray(validate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--configs", default="base,symmetric,over_averse,under_averse,cheap_abstain,expensive_abstain,narrow_penalty,wide_penalty,alpha_05,alpha_20")
    args = parser.parse_args()

    rows = load_records(args.data)
    by_company = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    seeds = [int(value) for value in args.seeds.split(",")]
    selected_configs = {value.strip() for value in args.configs.split(",")}
    configs = [(name, config) for name, config in CONFIGS if name in selected_configs]

    records = []
    for seed in seeds:
        for year in TEST_YEARS:
            boundary = date(year, 1, 1)
            test_rows = [
                row
                for row in rows
                if row[2].get("split") == "test"
                and int(row[2].get("cutoff_year") or 0) == year
            ]
            for fold in range(FOLDS):
                train_rows = [
                    row
                    for row in rows
                    if fold_of(company_key(row[2]), seed) != fold
                    and row[1] < boundary
                ]
                fold_test = [
                    row for row in test_rows if fold_of(company_key(row[2]), seed) == fold
                ]
                if not train_rows or not fold_test:
                    continue
                calibration_probabilities, calibration_truths = oof_probabilities(
                    train_rows, by_company
                )
                # Rebuild metadata in the same order as oof_probabilities.
                _, _, meta_train = None, None, None
                from run_real_benchmark import build

                _, _, _, meta_train = build(train_rows, by_company)
                calibration_index, validation_index = assign_split(meta_train, seed)
                if len(calibration_index) == 0 or len(validation_index) == 0:
                    continue
                test_probabilities, test_truths, _ = fit_predict(
                    train_rows, fold_test, by_company
                )
                temperature = fit_temperature(calibration_probabilities, calibration_truths)
                calibrated_test = apply_temperature(test_probabilities, temperature)
                for config_name, config in configs:
                    candidates = []
                    for method in METHODS:
                        for alpha in ALPHAS:
                            threshold, cal_loss, cal_cov = choose_threshold(
                                calibration_probabilities[calibration_index],
                                calibration_truths[calibration_index],
                                config,
                                method,
                                alpha,
                            )
                            validation_sets = sets_from_scores(
                                calibration_probabilities[validation_index], threshold, method
                            )
                            validation = evaluate(
                                calibration_probabilities[validation_index],
                                calibration_truths[validation_index],
                                validation_sets,
                                config,
                            )
                            if validation["coverage"] + 1e-12 < 1.0 - alpha:
                                continue
                            candidates.append(
                                {
                                    "method": method,
                                    "alpha": alpha,
                                    "threshold": threshold,
                                    "calibration_loss": cal_loss,
                                    "calibration_coverage": cal_cov,
                                    "validation_loss": validation["mean_loss"],
                                    "validation_coverage": validation["coverage"],
                                }
                            )
                    if not candidates:
                        continue
                    chosen = min(candidates, key=lambda item: (item["validation_loss"], item["alpha"]))
                    threshold, cal_loss, cal_cov = choose_threshold(
                        calibration_probabilities,
                        calibration_truths,
                        config,
                        chosen["method"],
                        chosen["alpha"],
                    )
                    test_sets = sets_from_scores(test_probabilities, threshold, chosen["method"])
                    test_metrics = evaluate(test_probabilities, test_truths, test_sets, config)
                    planner_losses = []
                    calibrated_planner_losses = []
                    calibrated_adjudicate = []
                    always_answer = []
                    always_adjudicate = []
                    for row, truth in zip(test_probabilities, test_truths):
                        action, prediction, credible, _ = planner_choice(row, config)
                        planner_losses.append(
                            action_loss(action, truth, config, prediction=prediction, credible_set=credible)
                        )
                        belief = {label: float(value) for label, value in zip(LABELS, row)}
                        always_answer.append(
                            answer_loss(LABELS[int(np.argmax(row))], truth, config)
                        )
                        always_adjudicate.append(
                            adjudicate_loss(credible_set(belief, config.alpha), truth, config)
                        )
                        calibrated_belief = {
                            label: float(value)
                            for label, value in zip(LABELS, calibrated_test[len(calibrated_planner_losses)])
                        }
                        c_action, c_prediction, c_credible, _ = planner_choice(
                            calibrated_test[len(calibrated_planner_losses)], config
                        )
                        calibrated_planner_losses.append(
                            action_loss(
                                c_action,
                                truth,
                                config,
                                prediction=c_prediction,
                                credible_set=c_credible,
                            )
                        )
                        calibrated_adjudicate.append(
                            adjudicate_loss(
                                credible_set(calibrated_belief, config.alpha), truth, config
                            )
                        )
                    records.append(
                        {
                            "seed": seed,
                            "year": year,
                            "fold": fold,
                            "config": config_name,
                            "selected_method": chosen["method"],
                            "selected_alpha": chosen["alpha"],
                            "threshold": threshold,
                            "calibration_loss": cal_loss,
                            "calibration_coverage": cal_cov,
                            **test_metrics,
                            "planner": float(np.mean(planner_losses)),
                            "planner_calibrated": float(np.mean(calibrated_planner_losses)),
                            "always_adjudicate_calibrated": float(np.mean(calibrated_adjudicate)),
                            "temperature": temperature,
                            "always_answer": float(np.mean(always_answer)),
                            "always_adjudicate": float(np.mean(always_adjudicate)),
                            "always_abstain": config.abstain,
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
            "planner_calibrated": float(np.average([value["planner_calibrated"] for value in values], weights=[value["n"] for value in values])),
            "always_adjudicate_calibrated": float(np.average([value["always_adjudicate_calibrated"] for value in values], weights=[value["n"] for value in values])),
            "always_answer": float(np.average([value["always_answer"] for value in values], weights=[value["n"] for value in values])),
            "always_adjudicate": float(np.average([value["always_adjudicate"] for value in values], weights=[value["n"] for value in values])),
            "always_abstain": float(np.average([value["always_abstain"] for value in values], weights=[value["n"] for value in values])),
            "selected": dict(Counter(f"{value['selected_method']}:{value['selected_alpha']}" for value in values)),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps({"records": records, "aggregate": summary}, indent=2), encoding="utf-8"
    )
    lines = [
        "# Nested conformal decision layer",
        "",
        "| config | loss | coverage | size | planner raw | planner calibrated | always adj cal. | selected |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for config, value in summary.items():
        lines.append(
            f"| {config} | {value['mean_loss']:.4f} | {value['coverage']:.4f} | "
            f"{value['mean_set_size']:.3f} | {value['planner']:.4f} | "
            f"{value['planner_calibrated']:.4f} | {value['always_adjudicate_calibrated']:.4f} | "
            f"{value['selected']} |"
        )
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
