"""Screen non-LLM backbones for the conformal decision layer.

For each fold, fit a backbone on training companies, generate out-of-fold
probabilities on those companies, split calibration/validation by company,
optionally temperature-scale on validation, and evaluate the conformal layer
against the tuned planner and fixed adjudicator. Test outcomes are never used
for fitting or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from run_classical_baselines import frame, make_model
from run_company_disjoint_benchmark import (
    C_VALUE,
    CLASS_WEIGHT,
    FOLDS,
    TEST_YEARS,
    VERIFIED_LABELS,
    aligned_probabilities,
    apply_temperature,
    fit_temperature,
    fold_of,
)
from run_real_benchmark import (
    DEFAULT_DATA,
    LABELS,
    NUMERIC,
    build,
    company_key,
    load_records,
    make_pipeline,
)
from run_real_decision_eval import CONFIGS, planner_choice
from finplan_rag.decision.conformal import ConformalConfig, NestedConformalDecisionLayer
from finplan_rag.workflow.protocol import ALPHA_GRID, RF_TREES
from finplan_rag.decision.loss import (
    LossConfig,
    action_loss,
    adjudicate_loss,
    answer_loss,
    credible_set,
    expected_action_loss,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "paper/results/backbone_conformal_diagnostics"


def split_indices(meta, seed: int):
    calibration = []
    validation = []
    for index, record in enumerate(meta):
        digest = hashlib.sha256(f"{seed}:{company_key(record)}".encode("utf-8")).hexdigest()
        (calibration if int(digest, 16) % 100 < 60 else validation).append(index)
    return np.asarray(calibration), np.asarray(validation)


def fit_predict_model(name: str, train_rows, test_rows, by_company, seed: int):
    X_train, y_train = frame(train_rows, by_company)
    X_test, y_test = frame(test_rows, by_company)
    model = make_model(name, seed)
    if name == "rf":
        # RF configuration of record for the diagnostic: 200 trees,
        # min_samples_leaf=3, balanced subsampling, one worker per process.
        model.set_params(clf__n_estimators=RF_TREES, clf__n_jobs=1)
    if name == "tfidf":
        model.fit(X_train["text"], list(map(str, y_train)))
        probabilities = aligned_probabilities(model, X_test["text"])
    else:
        model.fit(X_train, list(map(str, y_train)))
        probabilities = aligned_probabilities(model, X_test)
    return probabilities, np.asarray(y_test, dtype=object), [record for _, _, record in test_rows]


def oof_model(name: str, train_rows, by_company, seed: int):
    X, y = frame(train_rows, by_company)
    meta = [record for _, _, record in train_rows]
    groups = np.asarray([company_key(record) for record in meta])
    n_splits = min(3, len(set(groups)))
    probabilities = np.zeros((len(y), len(LABELS)), dtype=float)
    for train_index, test_index in GroupKFold(n_splits=n_splits).split(X, y, groups):
        inner_train = [train_rows[i] for i in train_index]
        inner_test = [train_rows[i] for i in test_index]
        probs, _, _ = fit_predict_model(name, inner_train, inner_test, by_company, seed)
        probabilities[test_index] = probs
    return probabilities, np.asarray(y, dtype=object), meta


def policy_losses(probabilities, truths, config, policy, alpha=None, margin=0.0):
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
        elif policy == "adaptive":
            best = None
            for adaptive_alpha in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
                candidate_set = credible_set(belief, adaptive_alpha)
                expected = {
                    "answer": expected_action_loss(
                        "answer", belief, split_config, prediction=prediction
                    ),
                    "adjudicate": expected_action_loss(
                        "adjudicate",
                        belief,
                        split_config,
                        credible_set=candidate_set,
                    ),
                    "abstain": split_config.abstain,
                }
                action = min(("answer", "adjudicate", "abstain"), key=lambda key: expected[key])
                key = (expected[action], adaptive_alpha)
                if best is None or key < best[0]:
                    best = (key, action, candidate_set)
            action, credible = best[1], best[2]
            loss = action_loss(
                action, truth, split_config, prediction=prediction, credible_set=credible
            )
        elif policy == "planner":
            action, predicted, credible, expected = planner_choice(row, split_config)
            if action == "answer" and margin > 0:
                alternative = min(
                    ("adjudicate", "abstain"),
                    key=lambda key: expected[key],
                )
                if expected["answer"] + margin >= expected[alternative]:
                    action = alternative
            loss = action_loss(
                action, truth, split_config, prediction=predicted, credible_set=credible
            )
        else:
            raise ValueError(policy)
        values.append(float(loss))
    return np.asarray(values, dtype=float)


def tune_alpha(probabilities, truths, config, policy):
    scores = [
        (float(np.mean(policy_losses(probabilities, truths, config, policy, alpha))), alpha)
        for alpha in ALPHA_GRID
    ]
    loss, alpha = min(scores, key=lambda item: (item[0], item[1]))
    return alpha, loss


def confidence_bin(row):
    confidence = float(np.max(row))
    if confidence < 0.40:
        return 0
    if confidence < 0.60:
        return 1
    if confidence < 0.80:
        return 2
    return 3


def tune_binned(probabilities, truths, config):
    probabilities = np.asarray(probabilities, dtype=float)
    bins = np.asarray([confidence_bin(row) for row in probabilities])
    mapping = {}
    for bin_id in range(4):
        index = np.where(bins == bin_id)[0]
        if len(index) == 0:
            mapping[bin_id] = 0.10
            continue
        scores = [
            (
                float(
                    np.mean(
                        policy_losses(
                            probabilities[index],
                            [truths[i] for i in index],
                            config,
                            "planner",
                            alpha,
                        )
                    )
                ),
                alpha,
            )
            for alpha in ALPHA_GRID
        ]
        mapping[bin_id] = min(scores, key=lambda item: (item[0], item[1]))[1]
    return mapping


def binned_losses(probabilities, truths, config, mapping):
    values = []
    for row, truth in zip(probabilities, truths):
        alpha = mapping[confidence_bin(row)]
        split_config = replace(config, alpha=alpha)
        action, prediction, credible, _ = planner_choice(row, split_config)
        values.append(
            action_loss(
                action,
                truth,
                split_config,
                prediction=prediction,
                credible_set=credible,
            )
        )
    return np.asarray(values, dtype=float)


def temperature_transform(probabilities, temperature):
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    logits = np.log(probabilities) / max(temperature, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def tune_decision_transform(validation_probabilities, validation_truths, test_probabilities, test_truths, config, policy):
    scores = []
    for temperature in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
        vp = temperature_transform(validation_probabilities, temperature)
        tp = temperature_transform(test_probabilities, temperature)
        alpha, validation_loss = tune_alpha(vp, validation_truths, config, policy)
        test_loss = float(np.mean(policy_losses(tp, test_truths, config, policy, alpha)))
        scores.append((validation_loss, temperature, alpha, test_loss))
    best = min(scores, key=lambda item: (item[0], item[1]))
    return best[1], best[2], best[3]


def tune_planner(probabilities, truths, config):
    scores = []
    for alpha in ALPHA_GRID:
        for margin in (0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0):
            loss = float(
                np.mean(
                    policy_losses(
                        probabilities, truths, config, "planner", alpha, margin
                    )
                )
            )
            scores.append((loss, alpha, margin))
    loss, alpha, margin = min(
        scores, key=lambda item: (item[0], item[1], item[2])
    )
    return alpha, margin, loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--models", default="logistic")
    parser.add_argument("--configs", default="base")
    parser.add_argument("--thresholds", type=int, default=31)
    parser.add_argument("--verified-only", action="store_true", default=True)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--rf-trees", type=int, default=200)
    args = parser.parse_args()
    global RF_TREES
    RF_TREES = args.rf_trees
    rows = load_records(args.data)
    by_company = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    seeds = [int(value) for value in args.seeds.split(",")]
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    wanted = {value.strip() for value in args.configs.split(",")}
    configs = [(name, config) for name, config in CONFIGS if name in wanted]
    records = []
    per_claim_rows = []
    for seed in seeds:
        for year in TEST_YEARS:
            boundary = date(year, 1, 1)
            test_rows = [
                row
                for row in rows
                if row[2].get("split") == "test"
                and int(row[2].get("cutoff_year") or 0) == year
                and (not args.verified_only or row[2].get("verification") in VERIFIED_LABELS)
            ]
            for fold in range(FOLDS):
                train_rows = [
                    row
                    for row in rows
                    if fold_of(company_key(row[2]), seed) != fold and row[1] < boundary
                ]
                fold_test = [row for row in test_rows if fold_of(company_key(row[2]), seed) == fold]
                if not train_rows or not fold_test:
                    continue
                for model_name in models:
                    calibration_probabilities, calibration_truths, meta_train = oof_model(
                        model_name, train_rows, by_company, seed
                    )
                    calibration_index, validation_index = split_indices(meta_train, seed)
                    validation_probabilities = calibration_probabilities[validation_index]
                    validation_truths = calibration_truths[validation_index]
                    test_probabilities, test_truths, test_meta = fit_predict_model(
                        model_name, train_rows, fold_test, by_company, seed
                    )
                    temperature = 1.0
                    if args.calibrate:
                        temperature = fit_temperature(
                            validation_probabilities, validation_truths
                        )
                        calibration_probabilities = apply_temperature(
                            calibration_probabilities, temperature
                        )
                        validation_probabilities = apply_temperature(
                            validation_probabilities, temperature
                        )
                        test_probabilities = apply_temperature(test_probabilities, temperature)
                    for config_name, config in configs:
                        layer = NestedConformalDecisionLayer(
                            ConformalConfig(n_thresholds=args.thresholds)
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
                        planner_alpha, planner_validation = tune_alpha(
                            validation_probabilities, validation_truths, config, "planner"
                        )
                        fixed_alpha, fixed_validation = tune_alpha(
                            validation_probabilities, validation_truths, config, "fixed"
                        )
                        (
                            planner_margin_alpha,
                            planner_margin,
                            planner_margin_validation,
                        ) = tune_planner(
                            validation_probabilities, validation_truths, config
                        )
                        planner_tuned = policy_losses(
                            test_probabilities, test_truths, config, "planner", planner_alpha
                        )
                        planner_margin_loss = policy_losses(
                            test_probabilities,
                            test_truths,
                            config,
                            "planner",
                            planner_margin_alpha,
                            planner_margin,
                        )
                        fixed_tuned = policy_losses(
                            test_probabilities, test_truths, config, "fixed", fixed_alpha
                        )
                        (
                            planner_temperature,
                            planner_calibrated_alpha,
                            planner_calibrated_test,
                        ) = tune_decision_transform(
                            validation_probabilities,
                            validation_truths,
                            test_probabilities,
                            test_truths,
                            config,
                            "planner",
                        )
                        (
                            fixed_temperature,
                            fixed_calibrated_alpha,
                            fixed_calibrated_test,
                        ) = tune_decision_transform(
                            validation_probabilities,
                            validation_truths,
                            test_probabilities,
                            test_truths,
                            config,
                            "fixed",
                        )
                        binned_mapping = tune_binned(
                            validation_probabilities, validation_truths, config
                        )
                        binned_test = binned_losses(
                            test_probabilities, test_truths, config, binned_mapping
                        )
                        binned_validation = float(
                            np.mean(
                                binned_losses(
                                    validation_probabilities,
                                    validation_truths,
                                    config,
                                    binned_mapping,
                                )
                            )
                        )
                        adaptive_test = policy_losses(
                            test_probabilities, test_truths, config, "adaptive"
                        )
                        adaptive_validation = float(
                            np.mean(
                                policy_losses(
                                    validation_probabilities,
                                    validation_truths,
                                    config,
                                    "adaptive",
                                )
                            )
                        )
                        answer_test = policy_losses(
                            test_probabilities, test_truths, config, "answer"
                        )
                        answer_validation = float(
                            np.mean(
                                policy_losses(
                                    validation_probabilities,
                                    validation_truths,
                                    config,
                                    "answer",
                                )
                            )
                        )
                        candidate_test = {
                            "conformal": float(metrics["mean_loss"]),
                            "planner_tuned": float(np.mean(planner_tuned)),
                            "planner_margin": float(np.mean(planner_margin_loss)),
                            "fixed_tuned": float(np.mean(fixed_tuned)),
                            "answer": float(np.mean(answer_test)),
                            "abstain": float(config.abstain),
                        }
                        candidate_validation = {
                            "conformal": float(layer.validation_loss),
                            "planner_tuned": float(planner_validation),
                            "planner_margin": float(planner_margin_validation),
                            "fixed_tuned": float(fixed_validation),
                            "answer": answer_validation,
                            "abstain": float(config.abstain),
                        }
                        selected_policy = min(
                            candidate_validation, key=lambda key: candidate_validation[key]
                        )
                        conformal_losses = np.asarray(
                            [
                                action_loss(
                                    decision.action,
                                    truth,
                                    config,
                                    prediction=decision.prediction,
                                    credible_set=decision.credible_set,
                                )
                                for decision, truth in zip(decisions, test_truths)
                            ],
                            dtype=float,
                        )
                        point_losses = np.asarray(
                            [
                                answer_loss(
                                    LABELS[int(np.argmax(row))], truth, config
                                )
                                for row, truth in zip(test_probabilities, test_truths)
                            ],
                            dtype=float,
                        )
                        for index, record in enumerate(test_meta):
                            per_claim_rows.append(
                                {
                                    "seed": seed,
                                    "year": year,
                                    "fold": fold,
                                    "model": model_name,
                                    "config": config_name,
                                    "claim_id": str(record.get("claim_id") or ""),
                                    "truth": str(test_truths[index]),
                                    "probabilities": [float(v) for v in test_probabilities[index]],
                                    "conformal": float(conformal_losses[index]),
                                    "planner_tuned": float(planner_tuned[index]),
                                    "fixed_tuned": float(fixed_tuned[index]),
                                    "point": float(point_losses[index]),
                                    "abstain": float(config.abstain),
                                }
                            )
                        records.append(
                            {
                                "seed": seed,
                                "year": year,
                                "fold": fold,
                                "model": model_name,
                                "config": config_name,
                                "temperature": temperature,
                                "method": layer.method,
                                "alpha": layer.alpha,
                                "objective": layer.threshold_objective,
                                "fallback": bool(layer.fallback),
                                **metrics,
                                "planner_alpha": planner_alpha,
                                "planner_tuned": float(np.mean(planner_tuned)),
                                "planner_margin_alpha": planner_margin_alpha,
                                "planner_margin": planner_margin,
                                "planner_margin_loss": float(np.mean(planner_margin_loss)),
                                "fixed_alpha": fixed_alpha,
                                "fixed_tuned": float(np.mean(fixed_tuned)),
                                "adaptive": float(np.mean(adaptive_test)),
                                "binned": float(np.mean(binned_test)),
                                "binned_mapping": binned_mapping,
                                "planner_calibrated": planner_calibrated_test,
                                "planner_calibrated_alpha": planner_calibrated_alpha,
                                "planner_temperature": planner_temperature,
                                "fixed_calibrated": fixed_calibrated_test,
                                "fixed_calibrated_alpha": fixed_calibrated_alpha,
                                "fixed_temperature": fixed_temperature,
                                "candidate_validation": candidate_validation,
                                "candidate_test": candidate_test,
                                "selected_policy": selected_policy,
                                "selected_validation": float(candidate_validation[selected_policy]),
                                "selected_test": float(candidate_test[selected_policy]),
                            }
                        )
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["model"], row["config"])].append(row)
    summary = {}
    for (model_name, config_name), values in grouped.items():
        n = sum(value["n"] for value in values)
        summary[f"{model_name}::{config_name}"] = {
            "n": n,
            "conformal": float(np.average([value["mean_loss"] for value in values], weights=[value["n"] for value in values])),
            "coverage": float(np.average([value["coverage"] for value in values], weights=[value["n"] for value in values])),
            "size": float(np.average([value["mean_set_size"] for value in values], weights=[value["n"] for value in values])),
            "planner_tuned": float(np.average([value["planner_tuned"] for value in values], weights=[value["n"] for value in values])),
            "planner_margin_loss": float(np.average([value["planner_margin_loss"] for value in values], weights=[value["n"] for value in values])),
            "fixed_tuned": float(np.average([value["fixed_tuned"] for value in values], weights=[value["n"] for value in values])),
            "adaptive": float(np.average([value["adaptive"] for value in values], weights=[value["n"] for value in values])),
            "binned": float(np.average([value["binned"] for value in values], weights=[value["n"] for value in values])),
            "planner_calibrated": float(np.average([value["planner_calibrated"] for value in values], weights=[value["n"] for value in values])),
            "fixed_calibrated": float(np.average([value["fixed_calibrated"] for value in values], weights=[value["n"] for value in values])),
            "planner_temperature": float(np.mean([value["planner_temperature"] for value in values])),
            "fixed_temperature": float(np.mean([value["fixed_temperature"] for value in values])),
            "selected_test": float(np.average([value["selected_test"] for value in values], weights=[value["n"] for value in values])),
            "selected": {
                key: sum(1 for value in values if value["selected_policy"] == key)
                for key in ("conformal", "planner_tuned", "planner_margin", "fixed_tuned", "answer", "abstain")
            },
            "fallbacks": sum(1 for value in values if value["fallback"]),
            "temperature": float(np.mean([value["temperature"] for value in values])),
            "alphas": sorted(set(value["alpha"] for value in values)),
            "methods": sorted(set(value["method"] for value in values)),
            "objectives": sorted(set(value["objective"] for value in values)),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (args.out / "per_claim.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_claim_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = [
        "# Backbone conformal diagnostics",
        "",
        "| model::config | conformal | coverage | size | planner | planner+margin | fixed | selected | temp | fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in summary.items():
        lines.append(
            f"| {key} | {value['conformal']:.4f} | {value['coverage']:.3f} | "
            f"{value['size']:.2f} | {value['planner_tuned']:.4f} | "
            f"{value['planner_margin_loss']:.4f} | {value['fixed_tuned']:.4f} | "
            f"{value['selected_test']:.4f} | {value['temperature']:.2f} | "
            f"{value['fallbacks']} |"
        )
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
