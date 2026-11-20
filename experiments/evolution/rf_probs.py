"""Train the baseline structured RF for a chosen split and save per-claim probabilities.

This mirrors ``scripts/make_calibrated_rf_probs.py`` but makes the evaluation
split configurable so validation-selected fusion weights cannot be tuned on
the test set.

Usage:
    python experiments/evolution/rf_probs.py --split validation --out paper/results/evolution_v1/rf_validation.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for extra in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from experiments.evolution.structured_evolve import (  # noqa: E402
    CATEGORICAL_BY_SET,
    FEATURE_SETS,
    align,
    build_matrix,
    make_rf,
)
from run_company_disjoint_benchmark import (  # noqa: E402
    FOLDS,
    apply_temperature,
    fit_temperature,
    fold_of,
)
from run_real_benchmark import company_key, load_records  # noqa: E402

VERIFIED = {"xbrl_search", "xbrl_verified", "xbrl_override"}
YEARS = {"validation": (2019, 2020, 2021), "test": (2022, 2023, 2024, 2025)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--feature-set", default="baseline", choices=sorted(FEATURE_SETS))
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--leaf", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = load_records(REPO_ROOT / "paper/data/dataset_v4.json")
    by_company: dict[str, list] = defaultdict(list)
    by_sector: dict[str, list] = defaultdict(list)
    for row in records:
        by_company[company_key(row[2])].append(row)
        by_sector[str(row[2].get("sector") or "other")].append(row)
    years = YEARS[args.split]
    test_rows = [
        row for row in records
        if row[2].get("split") == args.split
        and int(row[2].get("cutoff_year") or 0) in years
        and row[2].get("verification") in VERIFIED
    ]
    test_rows.sort(key=lambda item: str(item[2]["claim_id"]))
    output = []
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        for year in years:
            boundary = date(year, 1, 1)
            year_test = [row for row in test_rows if int(row[2].get("cutoff_year") or 0) == year]
            for fold in range(FOLDS):
                fold_test = [row for row in year_test if fold_of(company_key(row[2]), seed) == fold]
                if not fold_test:
                    continue
                train_rows = [
                    row for row in records
                    if row[1] < boundary and fold_of(company_key(row[2]), seed) != fold
                ]
                if not train_rows:
                    continue
                X_train, C_train, y_train, meta_train = build_matrix(train_rows, by_company, by_sector, args.feature_set)
                X_test, C_test, y_test, meta_test = build_matrix(fold_test, by_company, by_sector, args.feature_set)
                train_features = np.hstack([X_train, C_train])
                test_features = np.hstack([X_test, C_test])
                model = make_rf(
                    seed,
                    len(FEATURE_SETS[args.feature_set]),
                    len(CATEGORICAL_BY_SET[args.feature_set]),
                    leaf=args.leaf,
                    trees=args.trees,
                )
                model.fit(train_features, list(map(str, y_train)))
                probs = align(model.predict_proba(test_features), model.classes_)
                groups = np.asarray([company_key(row) for row in meta_train])
                n_splits = min(3, len(set(groups)))
                if n_splits >= 2:
                    oof, truths = [], []
                    labels = np.asarray(list(map(str, y_train)))
                    for fit_index, cal_index in GroupKFold(n_splits=n_splits).split(train_features, labels, groups):
                        inner = make_rf(
                            seed,
                            len(FEATURE_SETS[args.feature_set]),
                            len(CATEGORICAL_BY_SET[args.feature_set]),
                            leaf=args.leaf,
                            trees=args.trees,
                        )
                        inner.fit(train_features[fit_index], labels[fit_index].tolist())
                        oof.append(align(inner.predict_proba(train_features[cal_index]), inner.classes_))
                        truths.extend(labels[cal_index].tolist())
                    temperature = fit_temperature(np.vstack(oof), truths)
                    probs = apply_temperature(probs, temperature)
                for record, truth, prob in zip(meta_test, y_test, probs):
                    output.append({
                        "seed": seed,
                        "claim_id": record["claim_id"],
                        "truth": str(truth),
                        "probabilities": [float(x) for x in prob],
                        "year": record.get("cutoff_year"),
                    })
        print("seed", seed, "rows", len(output), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n", encoding="utf-8")
    print("wrote", out, len(output))


if __name__ == "__main__":
    from sklearn.model_selection import GroupKFold  # noqa: E402

    main()
