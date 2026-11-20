"""Validation-trained calibrated meta-readout for AnchorEvolve.

The meta-model is trained only on the 2019-2021 validation split and evaluated
once on 2022-2025.  Features are the calibrated RF posterior, the evolved
AnchorEvolve belief, and compact anchor-state features.  The output posterior
is passed through the same full-action loss-aware gate as the main system.

Usage:
    python experiments/evolution/meta_readout.py --config experiments/evolution/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
for extra in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from experiments.evolution.temporal_fusion import decide, metrics  # noqa: E402
from run_real_benchmark import company_key  # noqa: E402

LABELS = ("below", "within", "above")
VERIFIED = {"xbrl_search", "xbrl_verified", "xbrl_override"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def boot_diff(a: np.ndarray, b: np.ndarray, *, resamples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(a) - np.asarray(b)
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(resamples)])
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def align(classes, probabilities: np.ndarray) -> np.ndarray:
    index = {label: i for i, label in enumerate(LABELS)}
    out = np.zeros((len(probabilities), 3), dtype=float)
    for column, label in enumerate(classes):
        out[:, index[str(label)]] = probabilities[:, column]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/evolution/config.yaml")
    parser.add_argument("--beliefs", default="")
    parser.add_argument("--anchor-source", default="new", choices=["old", "new"])
    args = parser.parse_args()
    config = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    loss = dict(config["anchors"]["loss"])
    loss["alpha"] = 1.0 - float(config["anchors"]["alpha"])

    data = {row["claim_id"]: row for row in json.loads((REPO_ROOT / config["data"]).read_text(encoding="utf-8"))}
    beliefs_path = (
        Path(args.beliefs)
        if args.beliefs
        else REPO_ROOT / config["results_dir"] / f"temporal_beliefs_{args.anchor_source}.jsonl"
    )
    beliefs = {row["claim_id"]: row for row in load_jsonl(beliefs_path)}

    def load_rf(path: Path) -> dict[str, np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for row in load_jsonl(path):
            grouped[row["claim_id"]].append(np.asarray(row["probabilities"], dtype=float))
        return {claim_id: np.mean(values, axis=0) for claim_id, values in grouped.items()}

    rf_validation = load_rf(REPO_ROOT / config["sources"]["rf_validation"])
    rf_test = load_rf(REPO_ROOT / config["sources"]["rf_test"])

    def rows_for(split: str, rf: dict[str, np.ndarray]) -> list[dict]:
        rows = []
        for claim_id, belief in beliefs.items():
            record = data.get(claim_id)
            if not record or record.get("split") != split or claim_id not in rf:
                continue
            if split == "test" and record.get("verification") not in VERIFIED:
                continue
            rows.append({"claim_id": claim_id, "record": record, "belief": belief, "rf": rf[claim_id]})
        rows.sort(key=lambda row: row["claim_id"])
        return rows

    validation = rows_for("validation", rf_validation)
    test = rows_for("test", rf_test)

    def feature_matrix(rows: list[dict], feature_set: str) -> np.ndarray:
        vectors = []
        for row in rows:
            belief = np.asarray(row["belief"]["belief"], dtype=float)
            scores = row["belief"].get("bucket_scores") or {}
            score_vector = np.asarray([float(scores.get(label, 1.0 / 3.0)) for label in LABELS], dtype=float)
            score_vector = score_vector / max(score_vector.sum(), 1e-12)
            base = np.concatenate([row["rf"], belief])
            extra = np.array([
                float(row["belief"].get("n_anchors", 0)),
                float(row["belief"].get("n_hard", 0)),
                float(row["belief"].get("mean_reliability", 0.5)),
                float(row["belief"].get("max_reliability", 0.5)),
                float(row["belief"].get("entropy", 0.0)),
                float(row["belief"].get("margin", 0.0)),
            ])
            if feature_set == "rf_llm":
                vectors.append(base)
            elif feature_set == "anchor":
                vectors.append(np.concatenate([base, extra]))
            else:
                vectors.append(np.concatenate([base, extra, score_vector]))
        return np.vstack(vectors)

    def truth(row: dict) -> int:
        return LABELS.index(row["belief"]["truth"])

    def evaluate(probabilities: np.ndarray, rows: list[dict]) -> tuple[dict, np.ndarray]:
        truths = [truth(row) for row in rows]
        summary = metrics(probabilities, truths, loss, full_action=True)
        losses = np.array([decide(probabilities[i], truths[i], loss, full_action=True)[0] for i in range(len(rows))])
        return summary, losses

    groups = np.asarray([company_key(row["record"]) for row in validation])
    y_validation = np.asarray([LABELS[truth(row)] for row in validation])
    y_test = np.asarray([LABELS[truth(row)] for row in test])
    rf_test_probs = np.vstack([row["rf"] for row in test])
    rf_test_summary, rf_test_losses = evaluate(rf_test_probs, test)

    results = {"n_validation": len(validation), "n_test": len(test), "rf_test": rf_test_summary, "models": {}}
    for feature_set in ("rf_llm", "anchor", "all"):
        X_validation = feature_matrix(validation, feature_set)
        X_test = feature_matrix(test, feature_set)
        best = None
        for c_value in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            fold_losses = []
            n_splits = min(5, len(set(groups)))
            for fit_index, cal_index in GroupKFold(n_splits=n_splits).split(X_validation, y_validation, groups):
                model = Pipeline([
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=c_value, max_iter=2000, class_weight="balanced")),
                ])
                model.fit(X_validation[fit_index], y_validation[fit_index])
                probabilities = align(model.classes_, model.predict_proba(X_validation[cal_index]))
                fold_rows = [validation[i] for i in cal_index]
                _, losses = evaluate(probabilities, fold_rows)
                fold_losses.append(float(np.mean(losses)))
            mean_loss = float(np.mean(fold_losses))
            if best is None or mean_loss < best[1]:
                best = (c_value, mean_loss)
        c_value, cv_loss = best
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=c_value, max_iter=2000, class_weight="balanced")),
        ])
        model.fit(X_validation, y_validation)
        probabilities = align(model.classes_, model.predict_proba(X_test))
        summary, test_losses = evaluate(probabilities, test)
        loss_vs_rf = boot_diff(test_losses, rf_test_losses, resamples=10000, seed=int(config["evaluation"]["seed"]))
        results["models"][feature_set] = {
            "C": c_value,
            "cv_loss": cv_loss,
            "test": summary,
            "loss_vs_rf": {"mean": loss_vs_rf[0], "lo": loss_vs_rf[1], "hi": loss_vs_rf[2]},
        }
        print(feature_set, "C", c_value, "cv", round(cv_loss, 4), "test", json.dumps(summary, ensure_ascii=False), flush=True)

    out = REPO_ROOT / config["results_dir"] / "meta_readout.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
