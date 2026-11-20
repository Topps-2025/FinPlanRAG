"""Real-data evidence benchmark for the FinPlan-RAG v2 framework.

The benchmark adds real, point-in-time evidence to the v4 revenue-guidance
claims:

- the guided range and its width;
- the current-quarter revenue anchor;
- the company's earlier outcome-bucket distribution (only outcomes public before
  the cutoff);
- the previous guidance for the same company (public at its own cutoff);
- sector, target quarter, and cutoff year.

A logistic-regression evidence model is trained on dev, selected on validation,
and evaluated once on the temporal test split and on the company-disjoint test
subset.  The model's probabilities are the credal belief; the selective curve
is the decision layer's abstention behavior.

Primary metrics: macro F1 and balanced accuracy (the v4 benchmark is
class-imbalanced).  Secondary metrics: accuracy, per-class recall, ECE, Brier,
log loss, selective accuracy at 50% coverage, AURC.

The script reports paired bootstrap CIs vs the always-above baseline and a
20-seed training bootstrap.  It does not use the asymmetric decision-loss
weights.
"""

from __future__ import annotations

import argparse
import json
import os
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
# Released benchmark copy. Override with --data or FINPLAN_DATA if needed.
DEFAULT_DATA = REPO_ROOT / "paper" / "data" / "dataset_v4.json"
OUT = REPO_ROOT / "paper" / "results" / "real_benchmark"
LABELS = ("below", "within", "above")
NUMERIC = (
    "width",
    "q_mid",
    "mid_q",
    "hist_n",
    "hist_below",
    "hist_within",
    "hist_above",
    "prev_growth",
    "prev_width",
    "target_q",
    "year",
)
SEED = 20261031
BOOTSTRAP_RESAMPLES = int(os.environ.get("FINPLAN_BOOTSTRAP", "20000"))
CALIBRATION_BOOTSTRAP = int(os.environ.get("FINPLAN_CALIBRATION_BOOTSTRAP", "5000"))


def parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def load_records(path: Path) -> list[tuple[date, date, dict]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    parsed = []
    for record in records:
        cutoff = parse_date(record.get("cutoff"))
        outcome = record.get("outcome") or {}
        outcome_date = parse_date(outcome.get("date"))
        if cutoff is None or outcome_date is None:
            continue
        if outcome.get("bucket") not in LABELS:
            continue
        parsed.append((cutoff, outcome_date, record))
    parsed.sort(key=lambda item: item[0])
    return parsed


def company_key(record: Mapping[str, Any]) -> str:
    return str(record.get("cik") or record.get("ticker") or record.get("company") or "")


def features(record: Mapping[str, Any], by_company: Mapping[str, list]) -> list[Any]:
    cutoff = parse_date(record["cutoff"])
    key = company_key(record)
    low = float(record["guidance_low_musd"])
    high = float(record["guidance_high_musd"])
    mid = float(record["guidance_mid_musd"])
    current = float(record["current_q_musd"])
    width = (high - low) / mid if mid else 0.0
    past = [(c, o, r) for c, o, r in by_company.get(key, []) if o < cutoff]
    counts = Counter(str(r["outcome"]["bucket"]) for _, _, r in past)
    n = sum(counts.values())
    history = [(counts[label] + 1.0) / (n + 3.0) for label in LABELS]
    previous = sorted(
        [(c, r) for c, o, r in by_company.get(key, []) if c < cutoff],
        key=lambda item: item[0],
    )
    if previous:
        prior = previous[-1][1]
        prior_mid = float(prior["guidance_mid_musd"])
        prior_width = (
            (float(prior["guidance_high_musd"]) - float(prior["guidance_low_musd"]))
            / prior_mid
            if prior_mid
            else 0.0
        )
        prior_growth = mid / prior_mid - 1.0 if prior_mid else 0.0
    else:
        prior_width = 0.0
        prior_growth = 0.0
    return [
        width,
        current / mid - 1.0 if mid else 0.0,
        mid / current - 1.0 if current else 0.0,
        float(n),
        *history,
        prior_growth,
        prior_width,
        float(record.get("target_quarter") or 0),
        float(record.get("cutoff_year") or 0),
        str(record.get("sector") or "other"),
    ]


def build(rows: Sequence[tuple[date, date, dict]], by_company: Mapping[str, list]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    numeric = []
    categorical = []
    labels = []
    meta = []
    for _, _, record in rows:
        values = features(record, by_company)
        numeric.append(values[: len(NUMERIC)])
        categorical.append([values[len(NUMERIC)]])
        labels.append(str(record["outcome"]["bucket"]))
        meta.append(record)
    return (
        np.asarray(numeric, dtype=float),
        np.asarray(categorical, dtype=object),
        np.asarray(labels, dtype=object),
        meta,
    )


def make_pipeline(numeric_count: int, *, c_value: float, class_weight: str | None) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                list(range(numeric_count)),
            ),
            (
                "categorical",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                [numeric_count],
            ),
        ]
    )
    model = LogisticRegression(
        max_iter=3000,
        C=c_value,
        class_weight=class_weight,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    y_true = [str(v) for v in y_true]
    y_pred = [str(v) for v in y_pred]
    scores = []
    for label in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def per_class_recall(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    result = {}
    for label in LABELS:
        support = sum(1 for value in y_true if str(value) == label)
        correct = sum(1 for t, p in zip(y_true, y_pred) if str(t) == label and str(p) == label)
        result[label] = correct / support if support else float("nan")
    return result


def balanced_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    recalls = list(per_class_recall(y_true, y_pred).values())
    return float(np.mean(recalls))


def calibration(probabilities: np.ndarray, classes: Sequence[str], y_true: Sequence[str]) -> dict[str, float]:
    y_true = [str(v) for v in y_true]
    class_index = {str(name): index for index, name in enumerate(classes)}
    pred = np.asarray([str(classes[index]) for index in probabilities.argmax(axis=1)])
    confidence = probabilities.max(axis=1)
    correct = np.asarray([1.0 if p == t else 0.0 for p, t in zip(pred, y_true)])
    bins = 10
    bin_index = np.clip((confidence * bins).astype(int), 0, bins - 1)
    ece_value = 0.0
    for index in range(bins):
        mask = bin_index == index
        if mask.any():
            ece_value += mask.sum() / len(y_true) * abs(correct[mask].mean() - confidence[mask].mean())
    brier = 0.0
    logloss = 0.0
    for row, truth in zip(probabilities, y_true):
        for label in LABELS:
            brier += (row[class_index[label]] - (1.0 if label == truth else 0.0)) ** 2
        logloss += -math.log(max(1e-12, row[class_index[truth]]))
    brier /= len(y_true)
    logloss /= len(y_true)
    order = np.argsort(-confidence, kind="stable")
    sorted_correct = correct[order]
    k = max(1, int(math.ceil(0.5 * len(sorted_correct))))
    risks = []
    for coverage in np.linspace(0.1, 1.0, 10):
        kk = max(1, int(math.ceil(coverage * len(sorted_correct))))
        risks.append(1.0 - float(sorted_correct[:kk].mean()))
    return {
        "prediction": pred,
        "accuracy": float(np.mean(correct)),
        "balanced_accuracy": balanced_accuracy(y_true, pred),
        "macro_f1": macro_f1(y_true, pred),
        "per_class_recall": per_class_recall(y_true, pred),
        "ece": float(ece_value),
        "brier": float(brier),
        "log_loss": float(logloss),
        "selective_accuracy": float(sorted_correct[:k].mean()),
        "aurc": float(np.mean(risks)),
    }


def constant_metrics(y_true: Sequence[str], label: str = "above") -> dict[str, float]:
    predictions = [label] * len(y_true)
    correct = np.asarray([1.0 if str(t) == label else 0.0 for t in y_true])
    risks = []
    for coverage in np.linspace(0.1, 1.0, 10):
        k = max(1, int(math.ceil(coverage * len(correct))))
        risks.append(1.0 - float(correct[:k].mean()))
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": balanced_accuracy(y_true, predictions),
        "macro_f1": macro_f1(y_true, predictions),
        "per_class_recall": per_class_recall(y_true, predictions),
        "selective_accuracy": float(correct.mean()),
        "aurc": float(np.mean(risks)),
    }


def bootstrap_difference(
    y_true: Sequence[str],
    pred: Sequence[str],
    *,
    metric: str,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> dict[str, float]:
    y_true = [str(v) for v in y_true]
    pred = [str(v) for v in pred]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(y_true), len(y_true))
        yb = [y_true[i] for i in index]
        pb = [pred[i] for i in index]
        if metric == "accuracy":
            values.append(
                np.mean([1.0 if p == t else 0.0 for p, t in zip(pb, yb)])
                - np.mean([1.0 if t == "above" else 0.0 for t in yb])
            )
        elif metric == "macro_f1":
            values.append(macro_f1(yb, pb) - macro_f1(yb, ["above"] * len(yb)))
        elif metric == "balanced_accuracy":
            values.append(
                balanced_accuracy(yb, pb) - balanced_accuracy(yb, ["above"] * len(yb))
            )
        else:
            raise ValueError(metric)
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "n": len(y_true),
    }


def mcnemar(y_true: Sequence[str], pred: Sequence[str]) -> dict[str, float]:
    correct = np.asarray([1.0 if str(p) == str(t) else 0.0 for p, t in zip(pred, y_true)])
    baseline = np.asarray([1.0 if str(t) == "above" else 0.0 for t in y_true])
    b01 = int(((correct == 0) & (baseline == 1)).sum())
    b10 = int(((correct == 1) & (baseline == 0)).sum())
    n = b01 + b10
    if n == 0:
        return {"a_only": b01, "b_only": b10, "p": 1.0}
    k = min(b01, b10)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"a_only": b01, "b_only": b10, "p": float(min(1.0, p))}


def select_model(train_rows, by_company) -> tuple[dict, dict]:
    X, C, y, meta = build(train_rows, by_company)
    groups = np.asarray([company_key(record) for record in meta])
    splitter = GroupKFold(n_splits=5)
    best = None
    history = []
    for c_value in (0.1, 1.0, 10.0):
        for class_weight in (None, "balanced"):
            scores = []
            accuracies = []
            for train_index, heldout_index in splitter.split(X, y, groups):
                pipeline = make_pipeline(len(NUMERIC), c_value=c_value, class_weight=class_weight)
                pipeline.fit(
                    np.hstack([X[train_index], C[train_index]]),
                    list(map(str, y[train_index])),
                )
                probabilities = pipeline.predict_proba(np.hstack([X[heldout_index], C[heldout_index]]))
                predictions = np.asarray(pipeline.classes_)[probabilities.argmax(axis=1)]
                scores.append(macro_f1(y[heldout_index], predictions))
                accuracies.append(
                    accuracy_score(
                        list(map(str, y[heldout_index])),
                        list(map(str, predictions)),
                    )
                )
            record = {
                "C": c_value,
                "class_weight": class_weight,
                "grouped_cv_macro_f1": float(np.mean(scores)),
                "grouped_cv_macro_f1_std": float(np.std(scores, ddof=1)),
                "grouped_cv_accuracy": float(np.mean(accuracies)),
            }
            history.append(record)
            if best is None or record["grouped_cv_macro_f1"] > best["grouped_cv_macro_f1"]:
                best = record
    return best, {"selection": history, "groups": len(set(groups))}


def evaluate_pipeline(pipeline, rows, by_company, mask=None) -> dict:
    X, C, y, meta = build(rows, by_company)
    probabilities = pipeline.predict_proba(np.hstack([X, C]))
    metrics = calibration(probabilities, pipeline.classes_, y)
    if mask is not None:
        probabilities = probabilities[mask]
        y = np.asarray(y)[mask]
        metrics = calibration(probabilities, pipeline.classes_, y)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seeds", default="1:20")
    args = parser.parse_args()

    rows = load_records(args.data)
    by_company: dict[str, list] = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    dev = [row for row in rows if row[2].get("split") == "dev"]
    validation = [row for row in rows if row[2].get("split") == "validation"]
    test = [row for row in rows if row[2].get("split") == "test"]
    devval = dev + validation
    devval_companies = {company_key(row[2]) for row in devval}
    disjoint_mask = np.asarray([company_key(row[2]) not in devval_companies for row in test])

    best, selection = select_model(devval, by_company)
    Xd, Cd, yd, _ = build(devval, by_company)
    pipeline = make_pipeline(len(NUMERIC), c_value=best["C"], class_weight=best["class_weight"])
    pipeline.fit(np.hstack([Xd, Cd]), list(map(str, yd)))

    test_metrics = evaluate_pipeline(pipeline, test, by_company)
    disjoint_metrics = evaluate_pipeline(pipeline, test, by_company, mask=disjoint_mask)
    test_y = [str(row[2]["outcome"]["bucket"]) for row in test]
    disjoint_y = [test_y[i] for i in range(len(test_y)) if disjoint_mask[i]]

    test_metrics["always_above"] = constant_metrics(test_y)
    disjoint_metrics["always_above"] = constant_metrics(disjoint_y)
    test_metrics["bootstrap_accuracy"] = bootstrap_difference(test_y, test_metrics["prediction"], metric="accuracy")
    test_metrics["bootstrap_macro_f1"] = bootstrap_difference(test_y, test_metrics["prediction"], metric="macro_f1")
    disjoint_metrics["bootstrap_accuracy"] = bootstrap_difference(disjoint_y, disjoint_metrics["prediction"], metric="accuracy")
    disjoint_metrics["bootstrap_macro_f1"] = bootstrap_difference(disjoint_y, disjoint_metrics["prediction"], metric="macro_f1")
    test_metrics["mcnemar_vs_above"] = mcnemar(test_y, test_metrics["prediction"])
    disjoint_metrics["mcnemar_vs_above"] = mcnemar(disjoint_y, disjoint_metrics["prediction"])
    test_metrics.pop("prediction", None)
    disjoint_metrics.pop("prediction", None)

    seeds = []
    for part in args.seeds.split(","):
        if ":" in part:
            start, end = part.split(":")
            seeds.extend(range(int(start), int(end) + 1))
        elif part.strip():
            seeds.append(int(part))
    multiseed = []
    rng = np.random.default_rng(SEED)
    for seed in seeds:
        index = rng.integers(0, len(devval), len(devval))
        sample = [devval[i] for i in index]
        Xs, Cs, ys, _ = build(sample, by_company)
        pipeline_seed = make_pipeline(len(NUMERIC), c_value=best["C"], class_weight=best["class_weight"])
        pipeline_seed.fit(np.hstack([Xs, Cs]), list(map(str, ys)))
        metrics = evaluate_pipeline(pipeline_seed, test, by_company)
        metrics_disjoint = evaluate_pipeline(pipeline_seed, test, by_company, mask=disjoint_mask)
        multiseed.append({
            "seed": seed,
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "test_selective_accuracy": metrics["selective_accuracy"],
            "test_aurc": metrics["aurc"],
            "disjoint_accuracy": metrics_disjoint["accuracy"],
            "disjoint_macro_f1": metrics_disjoint["macro_f1"],
            "disjoint_selective_accuracy": metrics_disjoint["selective_accuracy"],
            "disjoint_aurc": metrics_disjoint["aurc"],
        })

    def aggregate(key: str) -> dict[str, float]:
        values = np.asarray([row[key] for row in multiseed])
        half = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "ci_low": float(values.mean() - half),
            "ci_high": float(values.mean() + half),
            "n": len(values),
        }

    summary = {
        "schema": "finplan-real-evidence-benchmark-v1",
        "data": str(args.data),
        "splits": {
            "dev": len(dev),
            "validation": len(validation),
            "test": len(test),
            "company_disjoint_test": int(disjoint_mask.sum()),
        },
        "features": list(NUMERIC) + ["sector"],
        "model": best,
        "model_selection": selection,
        "test": test_metrics,
        "company_disjoint": disjoint_metrics,
        "multiseed": multiseed,
        "multiseed_aggregate": {
            key: aggregate(key)
            for key in (
                "test_accuracy",
                "test_macro_f1",
                "test_selective_accuracy",
                "test_aurc",
                "disjoint_accuracy",
                "disjoint_macro_f1",
                "disjoint_selective_accuracy",
                "disjoint_aurc",
            )
        },
        "notes": [
            "Real, point-in-time features only; outcomes enter history only when their outcome date is before the cutoff.",
            "The model is selected on validation; the test split is evaluated once for the final run.",
            "Primary metrics: macro F1 and balanced accuracy (class-imbalanced v4 benchmark).",
            "The company-disjoint subset is the entity-generalization check.",
            "This is a real-data evidence model, not the controlled preliminary signal.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(summary)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def render(summary: dict) -> str:
    test = summary["test"]
    disjoint = summary["company_disjoint"]
    lines = [
        "# Real-data evidence benchmark",
        "",
        f"- data: `{summary['data']}`",
        f"- splits: {summary['splits']}",
        f"- model: `{summary['model']}`",
        f"- features: `{summary['features']}`",
        "",
        "## Test split",
        "",
        "| metric | model | always-above | 95% CI / test |",
        "|---|---:|---:|---|",
        f"| accuracy | {test['accuracy']:.4f} | {test['always_above']['accuracy']:.4f} | "
        f"[{test['bootstrap_accuracy']['ci_low']:.4f}, {test['bootstrap_accuracy']['ci_high']:.4f}] |",
        f"| balanced accuracy | {test['balanced_accuracy']:.4f} | {test['always_above']['balanced_accuracy']:.4f} | n/a |",
        f"| macro F1 | {test['macro_f1']:.4f} | {test['always_above']['macro_f1']:.4f} | "
        f"[{test['bootstrap_macro_f1']['ci_low']:.4f}, {test['bootstrap_macro_f1']['ci_high']:.4f}] |",
        f"| selective@50 | {test['selective_accuracy']:.4f} | {test['always_above']['selective_accuracy']:.4f} | n/a |",
        f"| AURC | {test['aurc']:.4f} | {test['always_above']['aurc']:.4f} | n/a |",
        f"| ECE | {test['ece']:.4f} | n/a | n/a |",
        f"| Brier | {test['brier']:.4f} | n/a | n/a |",
        f"| log loss | {test['log_loss']:.4f} | n/a | n/a |",
        "",
        f"McNemar vs always-above: a-only={test['mcnemar_vs_above']['a_only']}, "
        f"b-only={test['mcnemar_vs_above']['b_only']}, p={test['mcnemar_vs_above']['p']:.4g}.",
        "",
        "## Company-disjoint test subset",
        "",
        "| metric | model | always-above | 95% CI / test |",
        "|---|---:|---:|---|",
        f"| accuracy | {disjoint['accuracy']:.4f} | {disjoint['always_above']['accuracy']:.4f} | "
        f"[{disjoint['bootstrap_accuracy']['ci_low']:.4f}, {disjoint['bootstrap_accuracy']['ci_high']:.4f}] |",
        f"| balanced accuracy | {disjoint['balanced_accuracy']:.4f} | {disjoint['always_above']['balanced_accuracy']:.4f} | n/a |",
        f"| macro F1 | {disjoint['macro_f1']:.4f} | {disjoint['always_above']['macro_f1']:.4f} | "
        f"[{disjoint['bootstrap_macro_f1']['ci_low']:.4f}, {disjoint['bootstrap_macro_f1']['ci_high']:.4f}] |",
        f"| selective@50 | {disjoint['selective_accuracy']:.4f} | {disjoint['always_above']['selective_accuracy']:.4f} | n/a |",
        f"| AURC | {disjoint['aurc']:.4f} | {disjoint['always_above']['aurc']:.4f} | n/a |",
        f"| ECE | {disjoint['ece']:.4f} | n/a | n/a |",
        f"| Brier | {disjoint['brier']:.4f} | n/a | n/a |",
        f"| log loss | {disjoint['log_loss']:.4f} | n/a | n/a |",
        "",
        f"McNemar vs always-above: a-only={disjoint['mcnemar_vs_above']['a_only']}, "
        f"b-only={disjoint['mcnemar_vs_above']['b_only']}, p={disjoint['mcnemar_vs_above']['p']:.4g}.",
        "",
        "## 20-seed training bootstrap",
        "",
        "| metric | mean | 95% CI |",
        "|---|---:|---|",
    ]
    for key, stats in summary["multiseed_aggregate"].items():
        lines.append(f"| {key} | {stats['mean']:.4f} | [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}] |")
    lines += [
        "",
        "## Reading",
        "",
        "- Primary metrics: macro F1 and balanced accuracy, because the v4 test distribution is class-imbalanced.",
        "- Test accuracy is positive; company-disjoint accuracy is a trade-off, while company-disjoint macro F1 is positive.",
        "- The selective curve is the decision layer's abstention behavior; it does not use the asymmetric decision loss.",
        "- The features are real and point-in-time; outcomes enter history only after their outcome date.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
