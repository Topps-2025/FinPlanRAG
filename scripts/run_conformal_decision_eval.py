"""Conformalized decision interface.

Algorithm:
1. Backbone: point-in-time logistic model on other company folds.
2. Calibration: out-of-fold probabilities on training companies with outcome
   dates before the test year. No test outcomes are used.
3. Feasible set: LAC or APS conformal set at a threshold q. q is selected on
   the calibration set to satisfy a coverage constraint and minimize the
   expected decision loss.
4. Action: answer, adjudicate the conformal set, or abstain, chosen by the
   expected asymmetric loss under the backbone probabilities.

The test year is evaluated once. Threshold selection and score-function
selection are done on calibration data only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from run_company_disjoint_benchmark import (
    C_VALUE,
    CLASS_WEIGHT,
    FOLDS,
    TEST_YEARS,
    aligned_probabilities,
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
from finplan_rag.decision.loss import (
    LossConfig,
    action_loss,
    adjudicate_loss,
    answer_loss,
    credible_set,
    expected_action_loss,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "paper/results" / "conformal_decision"
ALPHAS = (0.05, 0.10, 0.20)


def fit_predict(train_rows, test_rows, by_company):
    X_train, C_train, y_train, _ = build(train_rows, by_company)
    X_test, C_test, y_test, meta_test = build(test_rows, by_company)
    model = make_pipeline(len(NUMERIC), c_value=C_VALUE, class_weight=CLASS_WEIGHT)
    model.fit(np.hstack([X_train, C_train]), list(map(str, y_train)))
    probabilities = aligned_probabilities(model, np.hstack([X_test, C_test]))
    return probabilities, np.asarray(y_test, dtype=object), meta_test


def oof_probabilities(train_rows, by_company, n_splits: int = 3):
    X, C, y, meta = build(train_rows, by_company)
    groups = np.asarray([company_key(record) for record in meta])
    n_splits = min(n_splits, len(set(groups)))
    probabilities = np.zeros((len(y), len(LABELS)), dtype=float)
    for train_index, test_index in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = make_pipeline(len(NUMERIC), c_value=C_VALUE, class_weight=CLASS_WEIGHT)
        model.fit(
            np.hstack([X[train_index], C[train_index]]),
            list(map(str, y[train_index])),
        )
        probabilities[test_index] = aligned_probabilities(
            model, np.hstack([X[test_index], C[test_index]])
        )
    return probabilities, np.asarray(y, dtype=object)


def lac_sets(probabilities: np.ndarray, threshold: float) -> list[tuple[str, ...]]:
    sets = []
    for row in probabilities:
        chosen = tuple(
            label for index, label in enumerate(LABELS) if row[index] >= 1.0 - threshold
        )
        if not chosen:
            chosen = (LABELS[int(np.argmax(row))],)
        sets.append(chosen)
    return sets


def aps_sets(probabilities: np.ndarray, threshold: float) -> list[tuple[str, ...]]:
    sets = []
    for row in probabilities:
        order = np.argsort(-row, kind="stable")
        chosen = []
        mass = 0.0
        for index in order:
            chosen.append(LABELS[index])
            mass += float(row[index])
            if mass >= threshold:
                break
        sets.append(tuple(chosen))
    return sets


def set_scores(probabilities: np.ndarray, truths: np.ndarray, method: str) -> np.ndarray:
    scores = []
    for row, truth in zip(probabilities, truths):
        index = LABELS.index(str(truth))
        if method == "lac":
            scores.append(1.0 - float(row[index]))
        elif method == "aps":
            order = np.argsort(-row, kind="stable")
            mass = 0.0
            for position in order:
                mass += float(row[position])
                if LABELS[position] == str(truth):
                    break
            scores.append(mass)
        else:
            raise ValueError(method)
    return np.asarray(scores, dtype=float)


def sets_from_scores(probabilities, threshold, method):
    if method == "lac":
        return lac_sets(probabilities, threshold)
    return aps_sets(probabilities, threshold)


def policy_loss(probabilities, sets, config: LossConfig) -> float:
    losses = []
    for row, candidate_set in zip(probabilities, sets):
        belief = {label: float(value) for label, value in zip(LABELS, row)}
        prediction = LABELS[int(np.argmax(row))]
        expected = {
            "answer": expected_action_loss("answer", belief, config, prediction=prediction),
            "adjudicate": expected_action_loss(
                "adjudicate", belief, config, credible_set=candidate_set
            ),
            "abstain": config.abstain,
        }
        action = min(("answer", "adjudicate", "abstain"), key=lambda key: expected[key])
        losses.append(expected[action])
    return float(np.mean(losses))


def choose_threshold(
    probabilities,
    truths,
    config: LossConfig,
    method: str,
    alpha: float,
):
    scores = set_scores(probabilities, truths, method)
    candidates = np.unique(np.concatenate(([0.0], np.quantile(scores, np.linspace(0.0, 1.0, 101)), [1.0])))
    best = None
    for threshold in candidates:
        sets = sets_from_scores(probabilities, float(threshold), method)
        coverage = float(np.mean([truth in candidate for truth, candidate in zip(truths, sets)]))
        if coverage + 1e-12 < 1.0 - alpha:
            continue
        loss = policy_loss(probabilities, sets, config)
        key = (loss, float(threshold))
        if best is None or key < best:
            best = key
    if best is None:
        threshold = 1.0
        loss = policy_loss(
            probabilities, sets_from_scores(probabilities, threshold, method), config
        )
        return threshold, loss, 1.0
    threshold, loss = best[1], best[0]
    sets = sets_from_scores(probabilities, threshold, method)
    coverage = float(np.mean([truth in candidate for truth, candidate in zip(truths, sets)]))
    return threshold, loss, coverage


def evaluate(
    probabilities,
    truths,
    sets,
    config: LossConfig,
):
    losses = []
    coverage = 0
    sizes = []
    actions = defaultdict(int)
    for row, truth, candidate_set in zip(probabilities, truths, sets):
        belief = {label: float(value) for label, value in zip(LABELS, row)}
        prediction = LABELS[int(np.argmax(row))]
        expected = {
            "answer": expected_action_loss("answer", belief, config, prediction=prediction),
            "adjudicate": expected_action_loss(
                "adjudicate", belief, config, credible_set=candidate_set
            ),
            "abstain": config.abstain,
        }
        action = min(("answer", "adjudicate", "abstain"), key=lambda key: expected[key])
        actions[action] += 1
        losses.append(
            action_loss(
                action,
                truth,
                config,
                prediction=prediction,
                credible_set=candidate_set,
            )
        )
        coverage += int(truth in candidate_set)
        sizes.append(len(candidate_set))
    n = len(truths)
    return {
        "n": n,
        "mean_loss": float(np.mean(losses)),
        "coverage": coverage / n,
        "mean_set_size": float(np.mean(sizes)),
        "answer_rate": actions["answer"] / n,
        "adjudicate_rate": actions["adjudicate"] / n,
        "abstain_rate": actions["abstain"] / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--methods", default="lac,aps")
    parser.add_argument("--alphas", default="0.05,0.10,0.20")
    parser.add_argument("--configs", default="base")
    args = parser.parse_args()

    rows = load_records(args.data)
    by_company = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    seeds = [int(value) for value in args.seeds.split(",")]
    methods = [value.strip() for value in args.methods.split(",")]
    alphas = [float(value) for value in args.alphas.split(",")]
    config_names = {value.strip() for value in args.configs.split(",")}
    configs = [(name, config) for name, config in CONFIGS if name in config_names]

    per_seed = []
    for seed in seeds:
        seed_results = []
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
                test_probabilities, test_truths, test_meta = fit_predict(
                    train_rows, fold_test, by_company
                )
                for method in methods:
                    for alpha in alphas:
                        for config_name, config in configs:
                            threshold, calibration_loss, calibration_coverage = choose_threshold(
                                calibration_probabilities,
                                calibration_truths,
                                config,
                                method,
                                alpha,
                            )
                            sets = sets_from_scores(test_probabilities, threshold, method)
                            metrics = evaluate(
                                test_probabilities, test_truths, sets, config
                            )
                            # Fixed and planner comparisons on the same folds.
                            planner_losses = []
                            always_answer = []
                            always_adjudicate = []
                            for row, truth in zip(test_probabilities, test_truths):
                                action, prediction, credible, _ = planner_choice(row, config)
                                planner_losses.append(
                                    action_loss(
                                        action,
                                        truth,
                                        config,
                                        prediction=prediction,
                                        credible_set=credible,
                                    )
                                )
                                belief = {
                                    label: float(value)
                                    for label, value in zip(LABELS, row)
                                }
                                always_answer.append(
                                    answer_loss(
                                        LABELS[int(np.argmax(row))], truth, config
                                    )
                                )
                                always_adjudicate.append(
                                    adjudicate_loss(
                                        credible_set(belief, config.alpha), truth, config
                                    )
                                )
                            seed_results.append(
                                {
                                    "seed": seed,
                                    "year": year,
                                    "fold": fold,
                                    "method": method,
                                    "alpha": alpha,
                                    "config": config_name,
                                    "threshold": threshold,
                                    "calibration_loss": calibration_loss,
                                    "calibration_coverage": calibration_coverage,
                                    **metrics,
                                    "planner": float(np.mean(planner_losses)),
                                    "always_answer": float(np.mean(always_answer)),
                                    "always_adjudicate": float(np.mean(always_adjudicate)),
                                    "always_abstain": config.abstain,
                                }
                            )
        per_seed.extend(seed_results)

    # Aggregate by method/alpha/config over seeds, years, folds.
    grouped = defaultdict(list)
    for row in per_seed:
        grouped[(row["method"], row["alpha"], row["config"])].append(row)
    summary = {}
    for key, values in grouped.items():
        method, alpha, config = key
        summary[f"{method}|{alpha}|{config}"] = {
            "n": int(sum(value["n"] for value in values)),
            "mean_loss": float(np.average([value["mean_loss"] for value in values], weights=[value["n"] for value in values])),
            "coverage": float(np.average([value["coverage"] for value in values], weights=[value["n"] for value in values])),
            "mean_set_size": float(np.average([value["mean_set_size"] for value in values], weights=[value["n"] for value in values])),
            "planner": float(np.average([value["planner"] for value in values], weights=[value["n"] for value in values])),
            "always_answer": float(np.average([value["always_answer"] for value in values], weights=[value["n"] for value in values])),
            "always_adjudicate": float(np.average([value["always_adjudicate"] for value in values], weights=[value["n"] for value in values])),
            "always_abstain": float(np.average([value["always_abstain"] for value in values], weights=[value["n"] for value in values])),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps({"per_fold": per_seed, "aggregate": summary}, indent=2),
        encoding="utf-8",
    )
    lines = ["# Conformal decision interface", ""]
    lines += [
        "| method | alpha | config | loss | coverage | size | planner | always adjudicate |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in sorted(summary.items()):
        method, alpha, config = key.split("|")
        lines.append(
            f"| {method} | {alpha} | {config} | {value['mean_loss']:.4f} | "
            f"{value['coverage']:.4f} | {value['mean_set_size']:.3f} | "
            f"{value['planner']:.4f} | {value['always_adjudicate']:.4f} |"
        )
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
