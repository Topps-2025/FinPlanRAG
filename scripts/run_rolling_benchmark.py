"""Rolling-origin benchmark over all available guidance claims.

For each cutoff year Y from 2015 onward, train on other company folds with
outcome dates before January 1 of Y and test every claim with cutoff year Y.
This expands the benchmark beyond the 2022-2025 holdout and adds temporal
slices while preserving the point-in-time and company-disjoint constraints.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from run_classical_baselines import frame, make_model
from run_company_disjoint_benchmark import aligned_probabilities, fold_of
from run_real_benchmark import DEFAULT_DATA, LABELS, company_key, load_records

REPO_ROOT = Path(__file__).resolve().parents[1]
START_YEAR = 2015
END_YEAR = 2025


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--models", default="logistic,rf")
    parser.add_argument("--rf-trees", type=int, default=200)
    args = parser.parse_args()
    rows = load_records(args.data)
    by_company = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    seeds = [int(value) for value in args.seeds.split(",")]
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    for model_name in models:
        all_rows = []
        for seed in seeds:
            for year in range(START_YEAR, END_YEAR + 1):
                boundary = date(year, 1, 1)
                test_rows = [
                    row
                    for row in rows
                    if int(row[2].get("cutoff_year") or 0) == year
                ]
                for fold in range(5):
                    train_rows = [
                        row
                        for row in rows
                        if fold_of(company_key(row[2]), seed) != fold
                        and row[1] < boundary
                    ]
                    fold_test = [
                        row
                        for row in test_rows
                        if fold_of(company_key(row[2]), seed) == fold
                    ]
                    if not train_rows or not fold_test:
                        continue
                    X_train, y_train = frame(train_rows, by_company)
                    X_test, y_test = frame(fold_test, by_company)
                    model = make_model(model_name, seed)
                    if model_name == "rf":
                        model.set_params(clf__n_estimators=args.rf_trees, clf__n_jobs=1)
                    model.fit(X_train, list(map(str, y_train)))
                    probabilities = aligned_probabilities(model, X_test)
                    for index, (_, _, record) in enumerate(fold_test):
                        all_rows.append(
                            {
                                "seed": seed,
                                "claim_id": str(record.get("claim_id") or ""),
                                "truth": str(y_test[index]),
                                "probabilities": [float(v) for v in probabilities[index]],
                                "year": year,
                                "company": company_key(record),
                                "sector": record.get("sector"),
                                "split": record.get("split"),
                                "verification": record.get("verification"),
                            }
                        )
        path = args.out / f"{model_name}_per_claim.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(model_name, len(all_rows), "rows", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
