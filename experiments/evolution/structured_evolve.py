"""Point-in-time structured + evolved-prior benchmark.

This is the no-LLM half of the evolution experiment.  It tests whether
point-in-time peer/sector priors, company surprise history and calendar
features add information beyond the current RF feature set, under the same
company-disjoint, year-wise protocol used by the paper.

Usage:
    python experiments/evolution/structured_evolve.py --eval-split validation
    python experiments/evolution/structured_evolve.py --eval-split test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
for extra in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from finplan_rag.decision.loss import (  # noqa: E402
    LossConfig,
    action_loss,
    credible_set,
    expected_action_loss,
)
from run_company_disjoint_benchmark import (  # noqa: E402
    FOLDS,
    apply_temperature,
    fit_temperature,
    fold_of,
)
from run_real_benchmark import (  # noqa: E402
    LABELS,
    balanced_accuracy,
    company_key,
    load_records,
    macro_f1,
    parse_date,
)

VERIFIED = {"xbrl_search", "xbrl_verified", "xbrl_override"}


def make_rf(seed: int, numeric_count: int, categorical_count: int, *, leaf: int, trees: int) -> Pipeline:
    numeric = list(range(numeric_count))
    categorical = list(range(numeric_count, numeric_count + categorical_count))
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
        n_estimators=trees,
        min_samples_leaf=leaf,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("prep", preprocess), ("clf", model)])


def align(probabilities: np.ndarray, classes) -> np.ndarray:
    index = {label: i for i, label in enumerate(LABELS)}
    out = np.zeros((len(probabilities), len(LABELS)), dtype=float)
    for column, label in enumerate(classes):
        out[:, index[str(label)]] = probabilities[:, column]
    return out


def surprise(record: dict) -> float | None:
    try:
        actual = float(record["outcome"]["actual_musd"])
        mid = float(record["guidance_mid_musd"])
    except (KeyError, TypeError, ValueError):
        return None
    return actual / mid - 1.0 if mid else None


def feature_dict(
    record: dict,
    by_company: dict[str, list],
    by_sector: dict[str, list],
    cutoff: date,
) -> dict[str, float | str]:
    key = company_key(record)
    low = float(record["guidance_low_musd"])
    high = float(record["guidance_high_musd"])
    mid = float(record["guidance_mid_musd"])
    current = float(record["current_q_musd"])
    width = (high - low) / mid if mid else 0.0
    past_company = [(c, o, r) for c, o, r in by_company.get(key, []) if o < cutoff]
    counts = Counter(str((r.get("outcome") or {}).get("bucket")) for _, _, r in past_company)
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
            / prior_mid if prior_mid else 0.0
        )
        prior_growth = mid / prior_mid - 1.0 if prior_mid else 0.0
        days_since = (cutoff - previous[-1][0]).days
    else:
        prior_width = 0.0
        prior_growth = 0.0
        days_since = -1.0

    sector = str(record.get("sector") or "other")
    past_peer = [
        (c, o, r) for c, o, r in by_sector.get(sector, [])
        if o < cutoff and company_key(r) != key
    ]
    peer_counts = Counter(str((r.get("outcome") or {}).get("bucket")) for _, _, r in past_peer)
    peer_n = sum(peer_counts.values())
    peer_rates = [(peer_counts[label] + 1.0) / (peer_n + 3.0) for label in LABELS]
    recent = [(c, o, r) for c, o, r in past_peer if (cutoff - o).days <= 365]
    recent_counts = Counter(str((r.get("outcome") or {}).get("bucket")) for _, _, r in recent)
    recent_n = sum(recent_counts.values())
    recent_above = (recent_counts["above"] + 1.0) / (recent_n + 3.0)

    company_surprises = [s for _, _, r in past_company if (s := surprise(r)) is not None]
    peer_surprises = [s for _, _, r in past_peer if (s := surprise(r)) is not None]
    company_surprise_mean = float(np.mean(company_surprises)) if company_surprises else 0.0
    company_surprise_last = float(company_surprises[-1]) if company_surprises else 0.0
    company_surprise_std = float(np.std(company_surprises)) if len(company_surprises) > 1 else 0.0
    peer_surprise_mean = float(np.mean(peer_surprises)) if peer_surprises else 0.0

    target_q = record.get("target_quarter")
    cutoff_month = cutoff.month
    values: dict[str, float | str] = {
        "width": width,
        "q_mid": current / mid - 1.0 if mid else 0.0,
        "mid_q": mid / current - 1.0 if current else 0.0,
        "hist_n": float(n),
        "hist_below": history[0],
        "hist_within": history[1],
        "hist_above": history[2],
        "prev_growth": prior_growth,
        "prev_width": prior_width,
        "target_q": float(target_q or 0),
        "year": float(record.get("cutoff_year") or 0),
        "current_rel_low": current / low - 1.0 if low else 0.0,
        "current_rel_high": current / high - 1.0 if high else 0.0,
        "current_inside": float(low <= current <= high),
        "current_eq_low": float(abs(current - low) < 1e-6),
        "reliability_low": float(str(record.get("current_reliability")) == "low"),
        "target_q_missing": float(target_q is None),
        "cutoff_month": float(cutoff_month),
        "days_since_claim": math.log1p(max(0.0, days_since)),
        "peer_n": float(peer_n),
        "peer_below": peer_rates[0],
        "peer_within": peer_rates[1],
        "peer_above": peer_rates[2],
        "peer_recent_above": recent_above,
        "peer_surprise_mean": peer_surprise_mean,
        "company_surprise_mean": company_surprise_mean,
        "company_surprise_last": company_surprise_last,
        "company_surprise_std": company_surprise_std,
        "sector": sector,
        "reliability": str(record.get("current_reliability") or "unknown"),
    }
    return values


FEATURE_SETS: dict[str, list[str]] = {
    "baseline": [
        "width", "q_mid", "mid_q", "hist_n", "hist_below", "hist_within",
        "hist_above", "prev_growth", "prev_width", "target_q", "year",
    ],
    "peer": [
        "width", "q_mid", "mid_q", "hist_n", "hist_below", "hist_within",
        "hist_above", "prev_growth", "prev_width", "target_q", "year",
        "peer_n", "peer_below", "peer_within", "peer_above",
        "peer_recent_above", "peer_surprise_mean",
    ],
    "company": [
        "width", "q_mid", "mid_q", "hist_n", "hist_below", "hist_within",
        "hist_above", "prev_growth", "prev_width", "target_q", "year",
        "company_surprise_mean", "company_surprise_last", "company_surprise_std",
        "days_since_claim",
    ],
    "all": [
        "width", "q_mid", "mid_q", "hist_n", "hist_below", "hist_within",
        "hist_above", "prev_growth", "prev_width", "target_q", "year",
        "current_rel_low", "current_rel_high", "current_inside",
        "current_eq_low", "reliability_low", "target_q_missing", "cutoff_month",
        "days_since_claim", "peer_n", "peer_below", "peer_within", "peer_above",
        "peer_recent_above", "peer_surprise_mean", "company_surprise_mean",
        "company_surprise_last", "company_surprise_std",
    ],
    "all_no_artifact": [
        "width", "q_mid", "mid_q", "hist_n", "hist_below", "hist_within",
        "hist_above", "prev_growth", "prev_width", "target_q", "year",
        "current_rel_low", "current_rel_high", "current_inside",
        "reliability_low", "target_q_missing", "cutoff_month",
        "days_since_claim", "peer_n", "peer_below", "peer_within", "peer_above",
        "peer_recent_above", "peer_surprise_mean", "company_surprise_mean",
        "company_surprise_last", "company_surprise_std",
    ],
}
CATEGORICAL_BY_SET: dict[str, tuple[str, ...]] = {
    "baseline": ("sector",),
    "peer": ("sector",),
    "company": ("sector",),
    "all": ("sector", "reliability"),
    "all_no_artifact": ("sector", "reliability"),
}


def build_matrix(rows, by_company, by_sector, feature_set: str):
    keys = FEATURE_SETS[feature_set]
    categorical_keys = CATEGORICAL_BY_SET[feature_set]
    numeric, categorical, labels, meta = [], [], [], []
    for _, _, record in rows:
        values = feature_dict(record, by_company, by_sector, parse_date(record["cutoff"]))
        numeric.append([float(values[key]) for key in keys])
        categorical.append([str(values[key]) for key in categorical_keys])
        labels.append(str((record.get("outcome") or {}).get("bucket")))
        meta.append(record)
    return (
        np.asarray(numeric, dtype=float),
        np.asarray(categorical, dtype=object),
        np.asarray(labels, dtype=object),
        meta,
    )


def decide(prob: np.ndarray, truth: int, cfg: LossConfig) -> tuple[float, int, int]:
    belief = {label: float(prob[i]) for i, label in enumerate(LABELS)}
    prediction = LABELS[int(np.argmax(prob))]
    credible = credible_set(belief, cfg.alpha)
    values = {
        "answer": expected_action_loss("answer", belief, cfg, prediction=prediction),
        "adjudicate": expected_action_loss("adjudicate", belief, cfg, credible_set=credible),
        "abstain": cfg.abstain,
    }
    action = min(values, key=values.get)
    truth_label = LABELS[truth]
    loss = action_loss(action, truth_label, cfg, prediction=prediction, credible_set=credible)
    return loss, len(credible), int(truth_label in credible)


def evaluate(probabilities: np.ndarray, truths: list[int]) -> dict[str, float]:
    cfg = LossConfig()
    pred = probabilities.argmax(axis=1).tolist()
    losses, sizes, covered = [], [], []
    for prob, truth in zip(probabilities, truths):
        loss, size, cover = decide(prob, truth, cfg)
        losses.append(loss)
        sizes.append(size)
        covered.append(cover)
    return {
        "n": len(truths),
        "accuracy": float(np.mean([p == t for p, t in zip(pred, truths)])),
        "balanced_accuracy": balanced_accuracy([LABELS[t] for t in truths], [LABELS[p] for p in pred]),
        "macro_f1": macro_f1([LABELS[t] for t in truths], [LABELS[p] for p in pred]),
        "decision_loss": float(np.mean(losses)),
        "mean_credible_size": float(np.mean(sizes)),
        "coverage": float(np.mean(covered)),
    }


def boot_ci(values: np.ndarray, *, resamples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(resamples)])
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(eval_split: str, feature_sets: list[str], *, seeds: list[int], trees: int, leaf: int, out_path: Path) -> dict:
    data = json.loads((REPO_ROOT / "paper/data/dataset_v4.json").read_text(encoding="utf-8"))
    records = load_records(REPO_ROOT / "paper/data/dataset_v4.json")
    by_company: dict[str, list] = defaultdict(list)
    by_sector: dict[str, list] = defaultdict(list)
    for item in records:
        by_company[company_key(item[2])].append(item)
        by_sector[str(item[2].get("sector") or "other")].append(item)

    years = {"validation": (2019, 2020, 2021), "test": (2022, 2023, 2024, 2025)}[eval_split]
    test_rows = [
        row for row in records
        if row[2].get("split") == eval_split
        and int(row[2].get("cutoff_year") or 0) in years
        and row[2].get("verification") in VERIFIED
    ]
    test_rows.sort(key=lambda item: str(item[2]["claim_id"]))
    output: dict[str, object] = {"eval_split": eval_split, "years": years, "n_test": len(test_rows), "feature_sets": {}}

    for feature_set in feature_sets:
        per_claim: dict[str, list[np.ndarray]] = defaultdict(list)
        per_seed_metrics = []
        for seed in seeds:
            seed_probs: dict[str, np.ndarray] = {}
            for year in years:
                boundary = date(year, 1, 1)
                year_test = [
                    row for row in test_rows
                    if int(row[2].get("cutoff_year") or 0) == year
                ]
                if not year_test:
                    continue
                for fold in range(FOLDS):
                    fold_test = [row for row in year_test if fold_of(company_key(row[2]), seed) == fold]
                    if not fold_test:
                        continue
                    train_rows = [
                        row for row in records
                        if row[1] < boundary
                        and fold_of(company_key(row[2]), seed) != fold
                    ]
                    if not train_rows:
                        continue
                    X_train, C_train, y_train, meta_train = build_matrix(train_rows, by_company, by_sector, feature_set)
                    X_test, C_test, y_test, meta_test = build_matrix(fold_test, by_company, by_sector, feature_set)
                    train_features = np.hstack([X_train, C_train])
                    test_features = np.hstack([X_test, C_test])
                    model = make_rf(seed, len(FEATURE_SETS[feature_set]), len(CATEGORICAL_BY_SET[feature_set]), leaf=leaf, trees=trees)
                    model.fit(train_features, list(map(str, y_train)))
                    probs = align(model.predict_proba(test_features), model.classes_)
                    groups = np.asarray([company_key(row) for row in meta_train])
                    n_splits = min(3, len(set(groups)))
                    if n_splits >= 2:
                        oof, truths = [], []
                        labels = np.asarray(list(map(str, y_train)))
                        for fit_index, cal_index in GroupKFold(n_splits=n_splits).split(train_features, labels, groups):
                            inner = make_rf(seed, len(FEATURE_SETS[feature_set]), len(CATEGORICAL_BY_SET[feature_set]), leaf=leaf, trees=trees)
                            inner.fit(train_features[fit_index], labels[fit_index].tolist())
                            oof.append(align(inner.predict_proba(train_features[cal_index]), inner.classes_))
                            truths.extend(labels[cal_index].tolist())
                        temperature = fit_temperature(np.vstack(oof), truths)
                        probs = apply_temperature(probs, temperature)
                    for record, prob in zip(meta_test, probs):
                        seed_probs[str(record["claim_id"])] = prob
            truths = [LABELS.index(str((row[2].get("outcome") or {}).get("bucket"))) for row in test_rows]
            ids = [str(row[2]["claim_id"]) for row in test_rows]
            if all(cid in seed_probs for cid in ids):
                probs = np.vstack([seed_probs[cid] for cid in ids])
                metrics = evaluate(probs, truths)
                metrics["seed"] = seed
                per_seed_metrics.append(metrics)
                for cid, prob in seed_probs.items():
                    per_claim[cid].append(prob)
        if not per_claim:
            continue
        ids = [str(row[2]["claim_id"]) for row in test_rows]
        truths = [LABELS.index(str((row[2].get("outcome") or {}).get("bucket"))) for row in test_rows]
        mean_probs = np.vstack([np.mean(per_claim[cid], axis=0) for cid in ids])
        summary = evaluate(mean_probs, truths)
        loss_values = np.array([
            decide(prob, truth, LossConfig())[0]
            for prob, truth in zip(mean_probs, truths)
        ])
        summary["loss_mean"], summary["loss_ci_low"], summary["loss_ci_high"] = boot_ci(
            loss_values, resamples=10000, seed=20261118
        )
        summary["per_seed"] = per_seed_metrics
        output["feature_sets"][feature_set] = summary
        print(feature_set, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items() if k != "per_seed"}, ensure_ascii=False), flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-split", default="validation", choices=["validation", "test"])
    parser.add_argument("--feature-sets", default="baseline,peer,company,all,all_no_artifact")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--leaf", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    out = Path(args.out) if args.out else REPO_ROOT / "paper/results/evolution_v1" / f"structured_{args.eval_split}.json"
    run(
        args.eval_split,
        [value.strip() for value in args.feature_sets.split(",") if value.strip()],
        seeds=[int(value) for value in args.seeds.split(",")],
        trees=args.trees,
        leaf=args.leaf,
        out_path=out,
    )


if __name__ == "__main__":
    main()
