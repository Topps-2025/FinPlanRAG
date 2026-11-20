"""Feasibility check: do peer outcomes arriving between cutoff and the target
outcome predict the target bucket?

This is a real, point-in-time evidence stream constructed from the released
benchmark itself: for each target claim, peer claims from other companies whose
outcome dates fall strictly after the target cutoff and before the target
outcome date. No human labels are used.

The script is intentionally a prototype for the sequential path-A experiment.
It compares a t0-only logistic model with a model that also sees the peer
summary at a horizon fraction f of the target window.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from run_real_benchmark import (
    DEFAULT_DATA,
    LABELS,
    NUMERIC,
    company_key,
    features,
    load_records,
    parse_date,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def as_date(value: Any):
    return parse_date(value)


def peer_events(rows, target):
    cutoff = as_date(target["cutoff"])
    outcome = as_date(target["outcome"]["date"])
    if cutoff is None or outcome is None:
        return []
    events = []
    for _, _, peer in rows:
        if peer["claim_id"] == target["claim_id"]:
            continue
        if company_key(peer) == company_key(target):
            continue
        p_out = as_date(peer["outcome"]["date"])
        if p_out is None or not (cutoff < p_out < outcome):
            continue
        events.append(
            {
                "date": p_out,
                "peer": peer,
                "sector": peer.get("sector") or "other",
                "bucket": peer["outcome"]["bucket"],
                "surprise": (
                    float(peer["outcome"]["actual_musd"]) - float(peer["guidance_mid_musd"])
                )
                / float(peer["guidance_mid_musd"])
                if float(peer["guidance_mid_musd"])
                else 0.0,
            }
        )
    events.sort(key=lambda item: item["date"])
    return events


def peer_summary(events, target, fraction: float):
    cutoff = as_date(target["cutoff"])
    outcome = as_date(target["outcome"]["date"])
    if cutoff is None or outcome is None:
        return {"n_peer": 0, "peer_below": 0.0, "peer_within": 0.0,
                "peer_above": 0.0, "peer_surprise": 0.0,
                "same_n": 0, "same_above": 0.0}
    horizon = cutoff + (outcome - cutoff) * fraction
    seen = [event for event in events if event["date"] <= horizon]
    n = len(seen)
    if n == 0:
        return {"n_peer": 0, "peer_below": 0.0, "peer_within": 0.0,
                "peer_above": 0.0, "peer_surprise": 0.0,
                "same_n": 0, "same_above": 0.0}
    counts = {label: 0 for label in LABELS}
    for event in seen:
        counts[event["bucket"]] += 1
    sector = target.get("sector") or "other"
    same = [event for event in seen if event["sector"] == sector]
    same_counts = {label: 0 for label in LABELS}
    for event in same:
        same_counts[event["bucket"]] += 1
    return {
        "n_peer": float(n),
        "peer_below": counts["below"] / n,
        "peer_within": counts["within"] / n,
        "peer_above": counts["above"] / n,
        "peer_surprise": float(np.mean([event["surprise"] for event in seen])),
        "same_n": float(len(same)),
        "same_above": (same_counts["above"] / len(same)) if same else 0.0,
    }


PEER_FEATURES = (
    "n_peer",
    "peer_below",
    "peer_within",
    "peer_above",
    "peer_surprise",
    "same_n",
    "same_above",
)


def feature_row(record, by_company, events, fraction):
    base = features(record, by_company)
    peer = peer_summary(events, record, fraction)
    return list(base) + [peer[name] for name in PEER_FEATURES]


def make_model(seed: int):
    # Row layout: 0..10 numeric, 11 sector, 12..18 peer features.
    numeric_indices = list(range(len(NUMERIC))) + list(
        range(len(NUMERIC) + 1, len(NUMERIC) + 1 + len(PEER_FEATURES))
    )
    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric_indices),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             [len(NUMERIC)]),
        ]
    )
    return Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000,
                                   random_state=seed)),
    ])


def evaluate(train, test, fraction, with_peer=True):
    by_company = defaultdict(list)
    rows = train + test
    for item in rows:
        by_company[company_key(item[2])].append(item)
    X_train, y_train = [], []
    for _, _, record in train:
        peers = peer_events(rows, record) if with_peer else []
        X_train.append(feature_row(record, by_company, peers, fraction))
        y_train.append(record["outcome"]["bucket"])
    X_test, y_test = [], []
    for _, _, record in test:
        peers = peer_events(rows, record) if with_peer else []
        X_test.append(feature_row(record, by_company, peers, fraction))
        y_test.append(record["outcome"]["bucket"])
    if not with_peer:
        # Zero the peer block so the comparison is same-dimensional.
        def zero_peer(row):
            return row[:len(NUMERIC)] + [row[len(NUMERIC)]] + [0.0] * len(PEER_FEATURES)

        X_train = [zero_peer(row) for row in X_train]
        X_test = [zero_peer(row) for row in X_test]
    model = make_model(seed=1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "n_train": len(train),
        "n_test": len(test),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "accuracy": float(np.mean(np.asarray(pred) == np.asarray(y_test))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "paper/results/peer_evidence")
    args = parser.parse_args()
    rows = load_records(args.data)
    train = [item for item in rows if item[1].year < 2022]
    test = [item for item in rows if item[1].year >= 2022 and item[2].get("split") == "test"]
    print(f"train={len(train)} test={len(test)}")
    results = {}
    for fraction in [float(value) for value in args.fractions.split(",")]:
        base = evaluate(train, test, fraction, with_peer=False)
        peer = evaluate(train, test, fraction, with_peer=True)
        results[str(fraction)] = {"t0_only": base, "with_peer": peer}
        print(f"fraction={fraction:.2f} t0_only={base} with_peer={peer}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "peer_evidence_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
