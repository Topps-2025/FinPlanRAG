"""Reproduce the calibrated random-forest posterior for the vague branch.

The protocol matches the paper: year-wise company-disjoint folds, five fold
seeds, point-in-time training (outcome before January 1 of the test year), a
random forest on structured features, and temperature scaling fitted on
internal company-disjoint folds.  One JSONL row is written per (claim, seed);
downstream code averages the seeds.

Usage:
    PYTHONPATH=src:scripts python scripts/make_calibrated_rf_probs.py \
      --data paper/data/dataset_v4.json \
      --out paper/results/benchmark_per_claim_verified_calibrated/per_claim.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_company_disjoint_benchmark import (
    FOLDS,
    TEST_YEARS,
    apply_temperature,
    fit_temperature,
    fold_of,
)
from run_real_benchmark import NUMERIC, build, company_key, load_records, parse_date


def make_rf(seed: int, class_weight: str | None = "balanced_subsample", leaf: int = 3) -> Pipeline:
    numeric = list(range(len(NUMERIC)))
    categorical = [len(NUMERIC)]
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric,
            ),
            (
                "categorical",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=leaf,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("prep", preprocess), ("clf", model)])


def align(probabilities: np.ndarray, classes) -> np.ndarray:
    labels = ["below", "within", "above"]
    index = {label: i for i, label in enumerate(labels)}
    out = np.zeros((len(probabilities), len(labels)), dtype=float)
    for column, label in enumerate(classes):
        out[:, index[str(label)]] = probabilities[:, column]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="paper/data/dataset_v4.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--class-weight", default="balanced_subsample")
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    args = parser.parse_args()
    rows = load_records(Path(args.data))
    by_company: dict[str, list] = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    output = []
    for seed in seeds:
        for year in TEST_YEARS:
            boundary = date(year, 1, 1)
            test_rows = [
                row for row in rows
                if row[2].get("split") == "test"
                and int(row[2].get("cutoff_year") or 0) == year
                and row[2].get("verification") in {
                    "xbrl_search", "xbrl_verified", "xbrl_override",
                }
            ]
            for fold in range(FOLDS):
                train_rows = [
                    row for row in rows
                    if fold_of(company_key(row[2]), seed) != fold
                    and row[1] < boundary
                ]
                fold_test = [
                    row for row in test_rows
                    if fold_of(company_key(row[2]), seed) == fold
                ]
                if not train_rows or not fold_test:
                    continue
                X_train, C_train, y_train, meta_train = build(train_rows, by_company)
                X_test, C_test, y_test, meta_test = build(fold_test, by_company)
                train_features = np.hstack([X_train, C_train])
                test_features = np.hstack([X_test, C_test])
                model = make_rf(seed, args.class_weight, args.min_samples_leaf)
                model.fit(train_features, list(map(str, y_train)))
                probabilities = align(model.predict_proba(test_features), model.classes_)

                groups = np.asarray([company_key(record) for record in meta_train])
                n_splits = min(3, len(set(groups)))
                if n_splits >= 2:
                    oof = []
                    truths = []
                    labels = np.asarray(list(map(str, y_train)))
                    for fit_index, cal_index in GroupKFold(n_splits=n_splits).split(
                        train_features, labels, groups
                    ):
                        inner = make_rf(seed, args.class_weight, args.min_samples_leaf)
                        inner.fit(train_features[fit_index], labels[fit_index].tolist())
                        oof.append(align(inner.predict_proba(train_features[cal_index]), inner.classes_))
                        truths.extend(labels[cal_index].tolist())
                    temperature = fit_temperature(np.vstack(oof), truths)
                    probabilities = apply_temperature(probabilities, temperature)
                else:
                    temperature = 1.0

                for record, truth, prob in zip(meta_test, y_test, probabilities):
                    output.append({
                        "seed": seed,
                        "claim_id": record["claim_id"],
                        "truth": str(truth),
                        "probabilities": [float(x) for x in prob],
                        "year": record.get("cutoff_year"),
                        "company": company_key(record),
                        "sector": record.get("sector"),
                        "guidance_width_pct": record.get("guidance_width_pct"),
                        "temperature": float(temperature),
                    })
        print(f"seed {seed}: {len(output)} rows", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", len(output), "rows to", out)


if __name__ == "__main__":
    main()
