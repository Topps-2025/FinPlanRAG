"""Year-wise, company-disjoint, point-in-time benchmark.

The earlier real benchmark used the v4 company-disjoint subset (n=108), which is
too small for stable conclusions.  This script evaluates **all** 273 test
claims under a stricter protocol:

- each company is assigned to one of five folds;
- for test year Y, the model is trained on claims from the other four folds
  whose outcome date is before January 1 of Y;
- the model is therefore company-disjoint (it never sees the test company) and
  point-in-time (it only sees outcomes public before the test year);
- all 273 test claims (2022-2025) are evaluated, and the fold assignment is
  repeated with five seeds.

Metrics are standard: macro F1 (primary), accuracy, per-class recall, selective
accuracy at 50% coverage, AURC, ECE, Brier, log loss.  Paired bootstrap CIs and
McNemar tests are reported against the always-above baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from finplan_rag.workflow.protocol import FOLD_SEEDS, FOLDS, TEST_YEARS, VERIFIED_LABELS
from run_real_benchmark import (
    BOOTSTRAP_RESAMPLES,
    LABELS,
    NUMERIC,
    balanced_accuracy,
    bootstrap_difference,
    build,
    calibration,
    company_key,
    constant_metrics,
    load_records,
    macro_f1,
    make_pipeline,
    mcnemar,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Released benchmark copy. Override with --data if needed.
DEFAULT_DATA = REPO_ROOT / "paper" / "data" / "dataset_v4.json"
OUT = REPO_ROOT / "paper" / "results" / "company_disjoint_benchmark"
C_VALUE = 1.0
CLASS_WEIGHT = "balanced"


def fold_of(company: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{company}".encode("utf-8")).hexdigest()
    return int(digest, 16) % FOLDS


def aligned_probabilities(pipeline, features: np.ndarray) -> np.ndarray:
    raw = pipeline.predict_proba(features)
    aligned = np.zeros((len(raw), len(LABELS)), dtype=float)
    index = {label: position for position, label in enumerate(LABELS)}
    for column, label in enumerate(pipeline.classes_):
        aligned[:, index[str(label)]] = raw[:, column]
    return aligned


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    logits = np.log(probabilities) / max(temperature, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def probability_log_loss(probabilities: np.ndarray, truths) -> float:
    index = {label: position for position, label in enumerate(LABELS)}
    return float(
        -np.mean(
            [
                np.log(max(1e-12, probabilities[row, index[str(truth)]]))
                for row, truth in enumerate(truths)
            ]
        )
    )


def fit_temperature(probabilities: np.ndarray, truths, grid=None) -> float:
    if grid is None:
        grid = np.linspace(0.5, 3.0, 26)
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in grid:
        loss = probability_log_loss(apply_temperature(probabilities, float(temperature)), truths)
        if loss < best_loss:
            best_temperature = float(temperature)
            best_loss = loss
    return best_temperature


def fit_predict_fold(train_rows, test_rows, by_company, *, calibrate: bool = False):
    X_train, C_train, y_train, meta_train = build(train_rows, by_company)
    X_test, C_test, y_test, meta_test = build(test_rows, by_company)
    train_features = np.hstack([X_train, C_train])
    test_features = np.hstack([X_test, C_test])
    pipeline = make_pipeline(len(NUMERIC), c_value=C_VALUE, class_weight=CLASS_WEIGHT)
    pipeline.fit(train_features, list(map(str, y_train)))
    probabilities = aligned_probabilities(pipeline, test_features)
    temperature = 1.0
    if calibrate:
        groups = np.asarray([company_key(record) for record in meta_train])
        n_splits = min(3, len(set(groups)))
        if n_splits >= 2:
            splitter = GroupKFold(n_splits=n_splits)
            oof_probabilities = []
            oof_truths = []
            for inner_train, inner_test in splitter.split(X_train, y_train, groups):
                inner_pipeline = make_pipeline(len(NUMERIC), c_value=C_VALUE, class_weight=CLASS_WEIGHT)
                inner_pipeline.fit(
                    np.hstack([X_train[inner_train], C_train[inner_train]]),
                    list(map(str, y_train[inner_train])),
                )
                oof_probabilities.extend(
                    aligned_probabilities(
                        inner_pipeline, np.hstack([X_train[inner_test], C_train[inner_test]])
                    ).tolist()
                )
                oof_truths.extend(str(value) for value in y_train[inner_test])
            temperature = fit_temperature(np.asarray(oof_probabilities), oof_truths)
            probabilities = apply_temperature(probabilities, temperature)
    return probabilities, y_test, meta_test, temperature


def run_fold_seed(
    rows, by_company, seed: int, *, calibrate: bool = False, verified_only: bool = False
) -> dict:
    predictions = []
    truths = []
    years = []
    companies = []
    keys = []
    probabilities_all = []
    temperatures = []
    prior_predictions = []
    for year in TEST_YEARS:
        boundary = date(year, 1, 1)
        test_rows = [
            row
            for row in rows
            if row[2].get("split") == "test"
            and int(row[2].get("cutoff_year") or 0) == year
            and (
                not verified_only
                or row[2].get("verification") in VERIFIED_LABELS
            )
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
            probabilities, y_test, meta_test, temperature = fit_predict_fold(
                train_rows, fold_test, by_company, calibrate=calibrate
            )
            train_counts = Counter(str(record["outcome"]["bucket"]) for _, _, record in train_rows)
            prior_labels = list(train_counts)
            prior_weights = np.asarray([train_counts[label] for label in prior_labels], dtype=float)
            prior_weights /= prior_weights.sum()
            prior_rng = np.random.default_rng(seed * 100000 + year * 100 + fold)
            prior_predictions.extend(
                str(value)
                for value in prior_rng.choice(
                    prior_labels, size=len(fold_test), p=prior_weights
                )
            )
            predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]
            predictions.extend(str(value) for value in predicted)
            truths.extend(str(value) for value in y_test)
            probabilities_all.extend(probabilities.tolist())
            years.extend([year] * len(fold_test))
            companies.extend(company_key(record) for record in meta_test)
            keys.extend(str(record.get("claim_id") or company_key(record)) for record in meta_test)
            temperatures.append({"year": year, "fold": fold, "temperature": temperature})
    return {
        "seed": seed,
        "predictions": predictions,
        "truths": truths,
        "probabilities": probabilities_all,
        "years": years,
        "companies": companies,
        "keys": keys,
        "temperatures": temperatures,
        "prior_predictions": prior_predictions,
    }


def metrics_from_predictions(truths, predictions) -> dict:
    # Calibration is not recoverable from labels alone; use label metrics.
    accuracy = float(np.mean([1.0 if p == t else 0.0 for p, t in zip(predictions, truths)]))
    macro = macro_f1(truths, predictions)
    # Selective confidence: use the predicted class frequency as a weak proxy is
    # not meaningful; the selective metric is computed from the model in the
    # caller.  This helper only returns label metrics.
    return {"accuracy": accuracy, "macro_f1": macro}


def claim_level_analysis(keys, truths, predictions, *, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 20261031) -> dict:
    """Average each claim's correctness over fold seeds, then bootstrap claims.

    Each claim appears once per fold seed.  Averaging within a claim removes the
    fold-assignment noise without pretending that the five seeds are independent
    replications of the claim population.  The bootstrap resamples claims, which
    is the unit that actually varies when the benchmark is extended.
    """
    by_key: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, truth, prediction in zip(keys, truths, predictions):
        by_key[str(key)].append((str(truth), str(prediction)))
    differences = []
    counts = []
    for key, values in by_key.items():
        truth = values[0][0]
        if any(item[0] != truth for item in values):
            raise ValueError(f"inconsistent truth for claim {key}")
        model_accuracy = float(np.mean([1.0 if prediction == truth else 0.0 for _, prediction in values]))
        baseline_accuracy = 1.0 if truth == "above" else 0.0
        differences.append(model_accuracy - baseline_accuracy)
        counts.append(len(values))
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = rng.integers(0, len(differences), len(differences))
        bootstrap[index] = float(differences[sample].mean())
    observed = float(differences.mean())
    flips = rng.choice((-1.0, 1.0), size=(n_resamples, len(differences)))
    permuted = np.abs((flips * differences).mean(axis=1))
    return {
        "n_claims": int(len(differences)),
        "mean_difference": observed,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "sign_flip_p": float(np.mean(permuted >= abs(observed))),
        "seed_counts": sorted(set(counts)),
    }


def bootstrap_calibration_difference(
    probabilities,
    truths,
    metric: str,
    *,
    n_resamples: int = 5000,
    seed: int = 20261031,
) -> dict:
    probabilities = np.asarray(probabilities, dtype=float)
    truths = [str(value) for value in truths]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(truths), len(truths))
        sample_truths = [truths[i] for i in index]
        model = calibration(probabilities[index], LABELS, sample_truths)
        baseline = constant_metrics(sample_truths)
        values.append(float(model[metric]) - float(baseline[metric]))
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "n": len(truths),
    }


def ensemble_analysis(keys, truths, probabilities, *, seed: int = 20261031) -> dict:
    """Average the five fold-seed probability vectors for each claim.

    Every member of the ensemble still excludes the test company, so the
    company-disjoint property is preserved.  This is a bagged model over fold
    assignments and yields one prediction per claim.
    """
    by_key: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for key, truth, probability in zip(keys, truths, probabilities):
        by_key[str(key)].append((str(truth), np.asarray(probability, dtype=float)))
    ensemble_truths = []
    ensemble_probabilities = []
    for key, values in by_key.items():
        truth = values[0][0]
        if any(item[0] != truth for item in values):
            raise ValueError(f"inconsistent truth for claim {key}")
        ensemble_truths.append(truth)
        ensemble_probabilities.append(np.mean([item[1] for item in values], axis=0))
    ensemble_probabilities = np.asarray(ensemble_probabilities)
    metrics = calibration(ensemble_probabilities, LABELS, ensemble_truths)
    metrics.pop("prediction", None)
    metrics["n"] = len(ensemble_truths)
    metrics["always_above"] = constant_metrics(ensemble_truths)
    ensemble_pred = np.asarray(LABELS)[ensemble_probabilities.argmax(axis=1)]
    metrics["bootstrap_accuracy"] = bootstrap_difference(ensemble_truths, ensemble_pred, metric="accuracy")
    metrics["bootstrap_balanced_accuracy"] = bootstrap_difference(
        ensemble_truths, ensemble_pred, metric="balanced_accuracy"
    )
    metrics["bootstrap_macro_f1"] = bootstrap_difference(ensemble_truths, ensemble_pred, metric="macro_f1")
    metrics["bootstrap_selective"] = bootstrap_calibration_difference(
        ensemble_probabilities, ensemble_truths, "selective_accuracy"
    )
    metrics["bootstrap_aurc"] = bootstrap_calibration_difference(
        ensemble_probabilities, ensemble_truths, "aurc"
    )
    metrics["mcnemar_vs_above"] = mcnemar(ensemble_truths, ensemble_pred)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="evaluate only test claims whose outcome label is XBRL-verified",
    )
    args = parser.parse_args()

    rows = load_records(args.data)
    by_company: dict[str, list] = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)

    per_seed_metrics = []
    all_predictions = []
    all_truths = []
    all_probabilities = []
    all_years = []
    all_companies = []
    all_keys = []
    all_prior_predictions = []
    for seed in FOLD_SEEDS:
        result = run_fold_seed(rows, by_company, seed, verified_only=args.verified_only)
        truths = result["truths"]
        predictions = result["predictions"]
        probabilities = np.asarray(result["probabilities"], dtype=float)
        metrics = calibration(probabilities, LABELS, truths)
        metrics.pop("prediction", None)
        metrics["n"] = len(truths)
        metrics["seed"] = seed
        metrics["always_above"] = constant_metrics(truths)
        metrics["bootstrap_accuracy"] = bootstrap_difference(truths, predictions, metric="accuracy")
        metrics["bootstrap_balanced_accuracy"] = bootstrap_difference(
            truths, predictions, metric="balanced_accuracy"
        )
        metrics["bootstrap_macro_f1"] = bootstrap_difference(truths, predictions, metric="macro_f1")
        metrics["mcnemar_vs_above"] = mcnemar(truths, predictions)
        prior = result["prior_predictions"]
        metrics["prior_baseline"] = {
            "accuracy": float(np.mean([1.0 if p == t else 0.0 for p, t in zip(prior, truths)])),
            "balanced_accuracy": balanced_accuracy(truths, prior),
            "macro_f1": macro_f1(truths, prior),
        }
        per_seed_metrics.append(metrics)
        all_predictions.extend(predictions)
        all_truths.extend(truths)
        all_probabilities.extend(probabilities.tolist())
        all_years.extend(result["years"])
        all_companies.extend(result["companies"])
        all_keys.extend(result["keys"])
        all_prior_predictions.extend(result["prior_predictions"])
        print(
            f"seed {seed}: n={len(truths)} acc={metrics['accuracy']:.4f} "
            f"macroF1={metrics['macro_f1']:.4f} selective={metrics['selective_accuracy']:.4f} "
            f"always_acc={metrics['always_above']['accuracy']:.4f} "
            f"always_macroF1={metrics['always_above']['macro_f1']:.4f} "
            f"mcnemar_p={metrics['mcnemar_vs_above']['p']:.4g}"
        )

    # Pool the five fold assignments (each claim appears once per seed).
    pooled = calibration(np.asarray(all_probabilities, dtype=float), LABELS, all_truths)
    pooled.pop("prediction", None)
    pooled["always_above"] = constant_metrics(all_truths)
    pooled["bootstrap_accuracy"] = bootstrap_difference(all_truths, all_predictions, metric="accuracy")
    pooled["bootstrap_macro_f1"] = bootstrap_difference(all_truths, all_predictions, metric="macro_f1")
    pooled["mcnemar_vs_above"] = mcnemar(all_truths, all_predictions)
    pooled["prior_baseline"] = {
        "accuracy": float(
            np.mean([1.0 if p == t else 0.0 for p, t in zip(all_prior_predictions, all_truths)])
        ),
        "balanced_accuracy": balanced_accuracy(all_truths, all_prior_predictions),
        "macro_f1": macro_f1(all_truths, all_prior_predictions),
    }
    pooled["n"] = len(all_truths)
    prior_baseline_mean = {
        key: float(np.mean([row["prior_baseline"][key] for row in per_seed_metrics]))
        for key in ("accuracy", "balanced_accuracy", "macro_f1")
    }

    claim_level = claim_level_analysis(all_keys, all_truths, all_predictions)
    ensemble = ensemble_analysis(all_keys, all_truths, all_probabilities)

    summary = {
        "schema": "finplan-company-disjoint-benchmark-v1",
        "data": str(args.data),
        "protocol": {
            "folds": FOLDS,
            "fold_seeds": list(FOLD_SEEDS),
            "test_years": list(TEST_YEARS),
            "training": "other folds, outcome date before January 1 of the test year",
            "company_disjoint": True,
            "point_in_time": True,
            "model": "logistic regression, C=1.0, class_weight=balanced",
            "verified_only": bool(args.verified_only),
        },
        "n_claims_per_seed": per_seed_metrics[0]["n"] if per_seed_metrics else 0,
        "per_seed": per_seed_metrics,
        "pooled": pooled,
        "prior_baseline_mean": prior_baseline_mean,
        "claim_level": claim_level,
        "ensemble": ensemble,
        "notes": [
            f"All {per_seed_metrics[0]['n'] if per_seed_metrics else 0} evaluated test claims are used; each claim appears once per fold seed.",
            "The model never trains on the test company; it only uses earlier public outcomes.",
            "The pooled statistics repeat each claim five times across fold seeds; the per-seed and claim-level results are the conservative units.",
            "The claim-level analysis averages each claim's correctness over fold seeds and bootstraps claims.",
            "The ensemble averages the five fold-seed probability vectors per claim; each member still excludes the test company.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(summary)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def render(summary: dict) -> str:
    claim_level = summary["claim_level"]
    ensemble = summary["ensemble"]
    lines = [
        "# Company-disjoint, point-in-time benchmark",
        "",
        f"- data: `{summary['data']}`",
        f"- protocol: {summary['protocol']}",
        "",
        "## Per fold seed",
        "",
        "| seed | n | accuracy | balanced acc | balanced delta [95% CI] | macro F1 | selective@50 | AURC | ECE | always-above acc | always-above macro F1 | McNemar p |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_seed"]:
        balanced = row["bootstrap_balanced_accuracy"]
        lines.append(
            f"| {row['seed']} | {row['n']} | "
            f"{row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{balanced['mean']:+.4f} [{balanced['ci_low']:+.4f}, {balanced['ci_high']:+.4f}] | "
            f"{row['macro_f1']:.4f} | "
            f"{row['selective_accuracy']:.4f} | {row['aurc']:.4f} | {row['ece']:.4f} | "
            f"{row['always_above']['accuracy']:.4f} | {row['always_above']['macro_f1']:.4f} | "
            f"{row['mcnemar_vs_above']['p']:.4g} |"
        )
    prior = summary["prior_baseline_mean"]
    pooled = summary["pooled"]
    lines += [
        "",
        "## Featureless prior-sampling baseline",
        "",
        "Predictions are sampled from the training-fold class distribution; this "
        "tests whether the model's gains are only class rebalancing.",
        "",
        f"- mean over seeds: accuracy {prior['accuracy']:.4f}, "
        f"balanced accuracy {prior['balanced_accuracy']:.4f}, macro F1 {prior['macro_f1']:.4f}",
        f"- pooled: accuracy {pooled['prior_baseline']['accuracy']:.4f}, "
        f"balanced accuracy {pooled['prior_baseline']['balanced_accuracy']:.4f}, "
        f"macro F1 {pooled['prior_baseline']['macro_f1']:.4f}",
        "",
        "## Pooled across five fold assignments",
        "",
        f"- n per seed: {summary['n_claims_per_seed']}; pooled rows: {pooled['n']}",
        f"- accuracy: {pooled['accuracy']:.4f} vs always-above {pooled['always_above']['accuracy']:.4f} "
        f"(paired CI [{pooled['bootstrap_accuracy']['ci_low']:.4f}, {pooled['bootstrap_accuracy']['ci_high']:.4f}])",
        f"- balanced accuracy: {pooled['balanced_accuracy']:.4f} vs always-above {pooled['always_above']['balanced_accuracy']:.4f}",
        f"- macro F1: {pooled['macro_f1']:.4f} vs always-above {pooled['always_above']['macro_f1']:.4f} "
        f"(paired CI [{pooled['bootstrap_macro_f1']['ci_low']:.4f}, {pooled['bootstrap_macro_f1']['ci_high']:.4f}])",
        f"- McNemar vs always-above: a-only={pooled['mcnemar_vs_above']['a_only']}, "
        f"b-only={pooled['mcnemar_vs_above']['b_only']}, p={pooled['mcnemar_vs_above']['p']:.4g}",
        "",
        "## Claim-level analysis (conservative unit)",
        "",
        f"- claims: {claim_level['n_claims']} (each averaged over {claim_level['seed_counts']} fold seeds)",
        f"- mean accuracy difference vs always-above: {claim_level['mean_difference']:.4f} "
        f"(95% bootstrap CI [{claim_level['ci_low']:.4f}, {claim_level['ci_high']:.4f}])",
        f"- two-sided sign-flip permutation p: {claim_level['sign_flip_p']:.4g}",
        "",
        "## Five-fold-seed probability ensemble",
        "",
        "| metric | model | always-above | paired 95% CI |",
        "|---|---:|---:|---:|",
        f"| accuracy | {ensemble['accuracy']:.4f} | {ensemble['always_above']['accuracy']:.4f} | "
        f"[{ensemble['bootstrap_accuracy']['ci_low']:.4f}, {ensemble['bootstrap_accuracy']['ci_high']:.4f}] |",
        f"| balanced accuracy | {ensemble['balanced_accuracy']:.4f} | {ensemble['always_above']['balanced_accuracy']:.4f} | "
        f"[{ensemble['bootstrap_balanced_accuracy']['ci_low']:.4f}, {ensemble['bootstrap_balanced_accuracy']['ci_high']:.4f}] |",
        f"| macro F1 | {ensemble['macro_f1']:.4f} | {ensemble['always_above']['macro_f1']:.4f} | "
        f"[{ensemble['bootstrap_macro_f1']['ci_low']:.4f}, {ensemble['bootstrap_macro_f1']['ci_high']:.4f}] |",
        f"| selective@50 | {ensemble['selective_accuracy']:.4f} | {ensemble['always_above']['selective_accuracy']:.4f} | "
        f"[{ensemble['bootstrap_selective']['ci_low']:.4f}, {ensemble['bootstrap_selective']['ci_high']:.4f}] |",
        f"| AURC | {ensemble['aurc']:.4f} | {ensemble['always_above']['aurc']:.4f} | "
        f"[{ensemble['bootstrap_aurc']['ci_low']:.4f}, {ensemble['bootstrap_aurc']['ci_high']:.4f}] |",
        "",
        f"- ensemble ECE: {ensemble['ece']:.4f}; Brier: {ensemble['brier']:.4f}; log loss: {ensemble['log_loss']:.4f}",
        f"- ensemble McNemar vs always-above: a-only={ensemble['mcnemar_vs_above']['a_only']}, "
        f"b-only={ensemble['mcnemar_vs_above']['b_only']}, p={ensemble['mcnemar_vs_above']['p']:.4g}",
        "",
        "## Reading",
        "",
        "- Each claim is evaluated under a model that never saw its company and only saw outcomes public before the test year.",
        "- The pooled rows repeat every claim five times; the per-seed results are the conservative unit.",
        "- Balanced accuracy and macro F1 are the primary metrics because the test distribution is class-imbalanced.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
