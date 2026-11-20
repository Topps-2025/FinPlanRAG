"""A real pre-hoc mode router on the year-wise benchmark.

The router sees only pre-evidence features (guidance width and history length)
and chooses answer vs adjudicate. The threshold is selected on validation
(2019--2021) and evaluated once on test (2022--2025). The content backend is the
same logistic model used by the planner.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_real_benchmark import (
    LABELS,
    build,
    calibration,
    company_key,
    features,
    load_records,
    make_pipeline,
)
from run_real_decision_eval import CONFIGS, planner_choice
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from finplan_rag.decision.loss import LossConfig, action_loss, adjudicate_loss, answer_loss, credible_set

REPO_ROOT = Path(__file__).resolve().parents[1]
# Released benchmark copy. Override with --data if needed.
DEFAULT_DATA = REPO_ROOT / "paper" / "data" / "dataset_v4.json"
OUT = REPO_ROOT / "paper" / "results" / "router_baseline"


def fit_predict(train_rows, test_rows, by_company):
    X_train, C_train, y_train, _ = build(train_rows, by_company)
    X_test, C_test, y_test, meta_test = build(test_rows, by_company)
    model = make_pipeline(len(features(train_rows[0][2], by_company)) - 1, c_value=1.0, class_weight="balanced")
    model.fit(np.hstack([X_train, C_train]), list(map(str, y_train)))
    probabilities = model.predict_proba(np.hstack([X_test, C_test]))
    order = [list(model.classes_).index(label) for label in LABELS]
    probabilities = probabilities[:, order]
    return probabilities, y_test, meta_test


def router_loss(probabilities, truths, widths, threshold, config, use_history=False, histories=None):
    losses = []
    for index, (probability, truth, width) in enumerate(zip(probabilities, truths, widths)):
        adjudicate = width >= threshold
        if use_history and histories is not None and histories[index] < 2:
            adjudicate = True
        prediction = LABELS[int(np.argmax(probability))]
        if adjudicate:
            belief = {label: float(value) for label, value in zip(LABELS, probability)}
            credible = credible_set(belief, config.alpha)
            losses.append(adjudicate_loss(credible, truth, config))
        else:
            losses.append(answer_loss(prediction, truth, config))
    return float(np.mean(losses))


def oracle_mode_router(train_probabilities, train_truths, train_rows, by_company, test_probabilities, test_truths, test_rows, config):
    train_x = np.asarray([features(record, by_company)[:-1] for record in train_rows], dtype=float)
    test_x = np.asarray([features(record, by_company)[:-1] for record in test_rows], dtype=float)
    labels = []
    for probability, truth in zip(train_probabilities, train_truths):
        prediction = LABELS[int(np.argmax(probability))]
        belief = {label: float(value) for label, value in zip(LABELS, probability)}
        answer = answer_loss(prediction, truth, config)
        adjudicate = adjudicate_loss(credible_set(belief, config.alpha), truth, config)
        labels.append("answer" if answer < adjudicate else "adjudicate")
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))])
    model.fit(train_x, labels)
    modes = model.predict(test_x)
    losses = []
    for mode, probability, truth in zip(modes, test_probabilities, test_truths):
        prediction = LABELS[int(np.argmax(probability))]
        if mode == "answer":
            losses.append(answer_loss(prediction, truth, config))
        else:
            belief = {label: float(value) for label, value in zip(LABELS, probability)}
            losses.append(adjudicate_loss(credible_set(belief, config.alpha), truth, config))
    return float(np.mean(losses)), int(sum(1 for mode in modes if mode == "answer"))


def main() -> int:
    rows = load_records(DEFAULT_DATA)
    by_company = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    dev = [row for row in rows if row[2].get("split") == "dev"]
    validation = [row for row in rows if row[2].get("split") == "validation"]
    test = [row for row in rows if row[2].get("split") == "test"]
    # Train on dev, tune on validation.
    validation_probabilities, validation_truths, validation_meta = fit_predict(
        dev, validation, by_company
    )
    validation_widths = [
        float(features(record, by_company)[0]) for record in validation_meta
    ]
    validation_history = [
        float(features(record, by_company)[3]) for record in validation_meta
    ]
    # Train on dev+validation, evaluate once on test.
    test_probabilities, test_truths, test_meta = fit_predict(
        dev + validation, test, by_company
    )
    test_widths = [float(features(record, by_company)[0]) for record in test_meta]
    test_history = [float(features(record, by_company)[3]) for record in test_meta]

    configs = CONFIGS
    results = {}
    for name, config in configs:
        best = None
        for threshold in np.linspace(0.0, 0.30, 31):
            loss = router_loss(
                validation_probabilities,
                validation_truths,
                validation_widths,
                float(threshold),
                config,
            )
            if best is None or loss < best[0]:
                best = (loss, float(threshold))
        threshold = best[1]
        router = router_loss(
            test_probabilities, test_truths, test_widths, threshold, config
        )
        always_answer = router_loss(
            test_probabilities, test_truths, test_widths, 1e9, config
        )
        always_adjudicate = router_loss(
            test_probabilities, test_truths, test_widths, -1e9, config
        )
        planner_losses = []
        for probability, truth in zip(test_probabilities, test_truths):
            action, prediction, credible, _ = planner_choice(probability, config)
            planner_losses.append(
                action_loss(action, truth, config, prediction=prediction, credible_set=credible)
            )
        oracle_router, oracle_answer_count = oracle_mode_router(
            validation_probabilities,
            validation_truths,
            [row[2] for row in validation],
            by_company,
            test_probabilities,
            test_truths,
            [row[2] for row in test],
            config,
        )
        results[name] = {
            "validation_best_loss": best[0],
            "threshold": threshold,
            "router": router,
            "oracle_router": oracle_router,
            "oracle_answer_count": oracle_answer_count,
            "always_answer": always_answer,
            "always_adjudicate": always_adjudicate,
            "planner": float(np.mean(planner_losses)),
            "n_test": len(test_truths),
        }
        print(name, results[name])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    lines = [
        "# Real pre-hoc router",
        "",
        "| config | threshold | width router | oracle-mode router | always answer | always adjudicate | planner |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in results.items():
        lines.append(
            f"| {name} | {values['threshold']:.3f} | {values['router']:.4f} | "
            f"{values['oracle_router']:.4f} | {values['always_answer']:.4f} | "
            f"{values['always_adjudicate']:.4f} | {values['planner']:.4f} |"
        )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
