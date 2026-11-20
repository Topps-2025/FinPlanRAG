"""Classical (non-LLM) baselines on the year-wise company-disjoint benchmark."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from run_company_disjoint_benchmark import (
    C_VALUE,
    FOLDS,
    TEST_YEARS,
    VERIFIED_LABELS,
    fold_of,
)
from run_real_benchmark import (
    DEFAULT_DATA,
    LABELS,
    NUMERIC,
    balanced_accuracy,
    build,
    company_key,
    load_records,
    macro_f1,
    make_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "paper/results" / "classical_baselines"


def frame(rows, by_company):
    X, C, y, meta = build(rows, by_company)
    df = pd.DataFrame(X, columns=list(NUMERIC))
    df["sector"] = C[:, 0]
    df["text"] = [str(record.get("claim_text") or "") for record in meta]
    return df, list(map(str, y))


def prep(numeric_features, categorical_features, *, sparse=False):
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=sparse),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def make_model(name: str, seed: int):
    numeric = list(NUMERIC)
    categorical = ["sector"]
    if name == "logistic":
        return Pipeline(
            [
                ("prep", prep(numeric, categorical, sparse=True)),
                (
                    "clf",
                    LogisticRegression(
                        C=C_VALUE, class_weight="balanced", max_iter=3000
                    ),
                ),
            ]
        )
    if name == "xgboost":
        return Pipeline(
            [
                ("prep", prep(numeric, categorical, sparse=False)),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if name == "lightgbm":
        return Pipeline(
            [
                ("prep", prep(numeric, categorical, sparse=False)),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=15,
                        min_child_samples=20,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    if name == "catboost":
        return Pipeline(
            [
                ("prep", prep(numeric, categorical, sparse=False)),
                (
                    "clf",
                    CatBoostClassifier(
                        iterations=300,
                        depth=4,
                        learning_rate=0.05,
                        auto_class_weights="Balanced",
                        random_seed=seed,
                        verbose=False,
                        thread_count=1,
                    ),
                ),
            ]
        )
    if name == "histgb":
        return Pipeline(
            [
                ("prep", prep(numeric, categorical, sparse=False)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "rf":
        return Pipeline(
            [
                ("prep", prep(numeric, categorical, sparse=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if name == "tfidf":
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=5000,
                        ngram_range=(1, 2),
                        min_df=2,
                        stop_words="english",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0, class_weight="balanced", max_iter=3000
                    ),
                ),
            ]
        )
    if name == "tfidf_numeric":
        return Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, stop_words="english"), "text"),
                            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
                            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)),
            ]
        )
    raise ValueError(name)


def run_seed(
    rows, by_company, seed: int, models: list[str], verified_only: bool = False
) -> dict:
    def keep(row) -> bool:
        return not verified_only or row[2].get("verification") in VERIFIED_LABELS

    results = {}
    for model_name in models:
        truths = []
        preds = []
        for year in TEST_YEARS:
            boundary = date(year, 1, 1)
            test_rows = [
                row
                for row in rows
                if keep(row)
                and row[2].get("split") == "test"
                and int(row[2].get("cutoff_year") or 0) == year
            ]
            for fold in range(FOLDS):
                train_rows = [
                    row
                    for row in rows
                    if keep(row)
                    and fold_of(company_key(row[2]), seed) != fold
                    and row[1] < boundary
                ]
                fold_test = [
                    row for row in test_rows if fold_of(company_key(row[2]), seed) == fold
                ]
                if not train_rows or not fold_test:
                    continue
                X_train, y_train = frame(train_rows, by_company)
                X_test, y_test = frame(fold_test, by_company)
                model = make_model(model_name, seed)
                if model_name == "tfidf":
                    model.fit(X_train["text"], y_train)
                    predicted = model.predict(X_test["text"])
                elif model_name in ("xgboost", "lightgbm", "catboost"):
                    encoder = LabelEncoder()
                    encoded = encoder.fit_transform(y_train)
                    model.fit(X_train, encoded)
                    predicted = encoder.inverse_transform(model.predict(X_test).astype(int))
                else:
                    model.fit(X_train, y_train)
                    predicted = model.predict(X_test)
                truths.extend(y_test)
                preds.extend(str(value) for value in predicted)
        results[model_name] = {
            "n": len(truths),
            "accuracy": float(np.mean([a == b for a, b in zip(truths, preds)])),
            "balanced_accuracy": balanced_accuracy(truths, preds),
            "macro_f1": macro_f1(truths, preds),
        }
        print(model_name, results[model_name])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--models",
        default="logistic,histgb,rf,xgboost,lightgbm,catboost,tfidf,tfidf_numeric",
    )
    parser.add_argument("--verified-only", action="store_true")
    args = parser.parse_args()
    rows = load_records(args.data)
    by_company = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    result = run_seed(
        rows, by_company, args.seed, models, verified_only=args.verified_only
    )
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "finplan-classical-baselines-v1",
        "data": str(args.data),
        "seed": args.seed,
        "models": result,
        "protocol": "year-wise company-disjoint; other folds, outcome before Jan 1 of test year",
    }
    suffix = "_verified" if args.verified_only else ""
    (args.out / f"summary_seed{args.seed}{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# Classical baselines (seed {args.seed}{suffix})",
        "",
        "| model | n | accuracy | balanced accuracy | macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in result.items():
        lines.append(
            f"| {name} | {values['n']} | {values['accuracy']:.4f} | "
            f"{values['balanced_accuracy']:.4f} | {values['macro_f1']:.4f} |"
        )
    (args.out / f"report_seed{args.seed}{suffix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
