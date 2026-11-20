"""Real sequential evidence experiment (path A).

Evidence stream
---------------
For each target claim, a peer claim from another company becomes observable at
the peer's outcome date. The target cutoff and the target outcome date bound the
window. Peer outcomes are public SEC/XBRL facts; no human annotation is used.
The agent may stop, acquire the next peer outcome, or wait for the target
outcome filing. Acquiring a peer outcome costs ``--evidence-cost``; waiting
costs ``--gamma`` per day.

The belief model is a snapshot logistic regression trained on point-in-time
features: target features, peer summary at time t, and days since cutoff. It is
trained on claims before the test period and evaluated on the 2022--2025 test
claims. The peer outcome model predicts a peer's own bucket from its features;
it is used only for one-step lookahead, never for the target label.

This is a prototype for the general workflow. It intentionally keeps the
temporal split simple and reports the policy comparison; the company-disjoint
version can be selected in the workflow config.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from finplan_rag.decision.loss import (
    LABELS,
    LossConfig,
    action_loss,
    credible_set,
    expected_action_loss,
)
from run_real_benchmark import (
    DEFAULT_DATA,
    NUMERIC,
    company_key,
    features,
    load_records,
    parse_date,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def as_date(value):
    return value if isinstance(value, date) else parse_date(value)


def event_date(value):
    if isinstance(value, dict):
        value = value.get("date")
    return as_date(value)

PEER_FEATURES = (
    "n_peer",
    "peer_below",
    "peer_within",
    "peer_above",
    "peer_surprise",
    "same_n",
    "same_above",
)


@dataclass(frozen=True)
class Event:
    date: date
    peer: Mapping[str, Any]
    bucket: str
    surprise: float
    sector: str


def expected_delay_days(train_rows) -> float:
    delays = [
        (row[1] - row[0]).days
        for row in train_rows
        if row[1] is not None and row[0] is not None
    ]
    return float(np.median(delays)) if delays else 90.0


def expected_delay_from_events(train_rows, earnings_events):
    if not earnings_events:
        return None
    values = []
    for _, _, record in train_rows:
        event = earnings_events.get(record["claim_id"])
        if event:
            values.append((event_date(event) - as_date(record["cutoff"])).days)
    return float(np.median(values)) if values else None


def build_events(rows, target, horizon_date=None, earnings_events=None):
    cutoff = as_date(target["cutoff"])
    outcome = event_date(horizon_date or target["outcome"]["date"])
    if not isinstance(cutoff, date) or not isinstance(outcome, date):
        return []
    events = []
    for _, _, peer in rows:
        if peer["claim_id"] == target["claim_id"]:
            continue
        if company_key(peer) == company_key(target):
            continue
        peer_out = event_date((earnings_events or {}).get(peer["claim_id"])) or as_date(peer["outcome"]["date"])
        if not isinstance(peer_out, date) or not (cutoff < peer_out < outcome):
            continue
        mid = float(peer["guidance_mid_musd"])
        actual = float(peer["outcome"]["actual_musd"])
        events.append(
            Event(
                date=peer_out,
                peer=peer,
                bucket=str(peer["outcome"]["bucket"]),
                surprise=(actual - mid) / mid if mid else 0.0,
                sector=str(peer.get("sector") or "other"),
            )
        )
    events.sort(key=lambda event: event.date)
    return events


def summary_vector(
    events: Sequence[Event],
    upto: date,
    sector: str,
    *,
    extra_bucket: str | None = None,
    extra_surprise: float | None = None,
    extra_sector: str | None = None,
) -> list[float]:
    seen = [event for event in events if event.date <= upto]
    if extra_bucket is not None:
        seen.append(
            Event(
                date=upto,
                peer={},
                bucket=extra_bucket,
                surprise=float(extra_surprise or 0.0),
                sector=str(extra_sector or sector),
            )
        )
    n = len(seen)
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    counts = Counter(event.bucket for event in seen)
    same = [event for event in seen if event.sector == sector]
    same_counts = Counter(event.bucket for event in same)
    return [
        float(n),
        counts["below"] / n,
        counts["within"] / n,
        counts["above"] / n,
        float(np.mean([event.surprise for event in seen])),
        float(len(same)),
        (same_counts["above"] / len(same)) if same else 0.0,
    ]


def row_vector(record, by_company, peers, upto, sector):
    base = features(record, by_company)
    summary = summary_vector(peers, upto, sector)
    days = max(0.0, float((upto - as_date(record["cutoff"])).days))
    return list(base[: len(NUMERIC)]) + summary + [days] + [base[len(NUMERIC)]]


def model_pipeline(seed: int):
    # Layout: 0..10 numeric, 11..17 peer, 18 days, 19 sector.
    numeric_indices = list(range(len(NUMERIC) + len(PEER_FEATURES) + 1))
    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_indices,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                [len(NUMERIC) + len(PEER_FEATURES) + 1],
            ),
        ]
    )
    return Pipeline(
        [
            ("prep", preprocess),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    )


def train_models(train_rows, by_company, samples_per_claim=5, seed=1, earnings_events=None):
    delay = expected_delay_from_events(train_rows, earnings_events) or expected_delay_days(train_rows)
    fractions = np.linspace(0.0, 1.0, samples_per_claim)
    X, y, weights = [], [], []
    for _, _, record in train_rows:
        peers = build_events(train_rows, record, (earnings_events or {}).get(record["claim_id"]), earnings_events)
        sector = str(record.get("sector") or "other")
        for fraction in fractions:
            upto = as_date(record["cutoff"]) + timedelta(days=float(fraction) * delay)
            X.append(row_vector(record, by_company, peers, upto, sector))
            y.append(str(record["outcome"]["bucket"]))
            weights.append(1.0 / len(fractions))
    model = model_pipeline(seed)
    model.fit(X, y, clf__sample_weight=np.asarray(weights))
    # A separate model for a peer's own bucket; it sees no peer features.
    X_peer, y_peer = [], []
    for _, _, record in train_rows:
        base = features(record, by_company)
        X_peer.append(list(base[: len(NUMERIC)]) + [0.0] * len(PEER_FEATURES) + [0.0] + [base[len(NUMERIC)]])
        y_peer.append(str(record["outcome"]["bucket"]))
    peer_model = model_pipeline(seed)
    peer_model.fit(X_peer, y_peer)
    bucket_surprise = {}
    for label in LABELS:
        values = [
            (float(r["outcome"]["actual_musd"]) - float(r["guidance_mid_musd"]))
            / float(r["guidance_mid_musd"])
            for _, _, r in train_rows
            if str(r["outcome"]["bucket"]) == label and float(r["guidance_mid_musd"])
        ]
        bucket_surprise[label] = float(np.mean(values)) if values else 0.0
    return model, peer_model, delay, bucket_surprise


def belief(model, record, by_company, peers, upto, sector):
    row = row_vector(record, by_company, peers, upto, sector)
    probabilities = model.predict_proba([row])[0]
    classes = list(model.classes_)
    return {label: float(probabilities[classes.index(label)]) for label in LABELS}


def stop_choice(probabilities, config: LossConfig):
    prediction = max(LABELS, key=lambda label: probabilities[label])
    adjudication = credible_set(probabilities, config.alpha)
    values = {
        "answer": expected_action_loss(
            "answer", probabilities, config, prediction=prediction
        ),
        "adjudicate": expected_action_loss(
            "adjudicate", probabilities, config, credible_set=adjudication
        ),
        "abstain": config.abstain,
    }
    action = min(values, key=lambda key: values[key])
    return action, values[action], prediction, adjudication


def peer_probs(peer_model, peer, by_company):
    base = features(peer, by_company)
    row = list(base[: len(NUMERIC)]) + [0.0] * len(PEER_FEATURES) + [0.0] + [base[len(NUMERIC)]]
    probabilities = peer_model.predict_proba([row])[0]
    classes = list(peer_model.classes_)
    return {label: float(probabilities[classes.index(label)]) for label in LABELS}


def simulate(
    record,
    earnings_date,
    by_company,
    peers,
    model,
    peer_model,
    delay,
    bucket_surprise,
    config,
    gamma,
    evidence_cost,
    outcome_cost,
    max_acquire,
    policy,
    margin=0.0,
):
    sector = str(record.get("sector") or "other")
    acquired = 0
    spent = 0.0
    current = as_date(record["cutoff"])
    next_index = 0
    while True:
        probabilities = belief(model, record, by_company, peers, current, sector)
        action, stop_loss, prediction, adjudication = stop_choice(probabilities, config)
        wait_loss = outcome_cost + gamma * delay + margin
        acquire_loss = math.inf
        if policy in {"planner", "planner_evidence"} and acquired < max_acquire:
            if next_index < len(peers):
                event = peers[next_index]
                probs = peer_probs(peer_model, event.peer, by_company)
                expected = 0.0
                for label in LABELS:
                    updated = summary_vector(
                        peers[:next_index],
                        current,
                        sector,
                        extra_bucket=label,
                        extra_surprise=bucket_surprise.get(label, 0.0),
                        extra_sector=event.sector,
                    )
                    row = (
                        list(features(record, by_company)[: len(NUMERIC)])
                        + updated
                        + [max(0.0, float((event.date - as_date(record["cutoff"])).days))]
                        + [sector]
                    )
                    p = model.predict_proba([row])[0]
                    classes = list(model.classes_)
                    belief_y = {lab: float(p[classes.index(lab)]) for lab in LABELS}
                    expected += probs[label] * stop_choice(belief_y, config)[1]
                acquire_loss = evidence_cost + expected + margin
        choices = [("stop", stop_loss), ("wait", wait_loss)]
        if policy in {"planner", "planner_evidence"}:
            choices.append(("acquire", acquire_loss))
        choice = min(choices, key=lambda item: item[1])
        if choice[0] == "stop":
            decision = action_loss(
                action,
                record["outcome"]["bucket"],
                config,
                prediction=prediction,
                credible_set=adjudication,
            )
            total = decision + spent + gamma * max(0.0, (current - as_date(record["cutoff"])).days)
            return {
                "policy": policy,
                "outcome": "stop",
                "action": action,
                "total_loss": total,
                "decision_loss": decision,
                "evidence_cost": spent,
                "delay_cost": gamma * max(0.0, (current - as_date(record["cutoff"])).days),
                "acquired": acquired,
                "decision_days": max(0.0, (current - as_date(record["cutoff"])).days),
            }
        if choice[0] == "wait":
            delay_days = max(0.0, (as_date(earnings_date) - as_date(record["cutoff"])).days)
            return {
                "policy": policy,
                "outcome": "wait",
                "action": "wait",
                "total_loss": spent + outcome_cost + gamma * delay_days,
                "decision_loss": 0.0,
                "evidence_cost": spent,
                "delay_cost": outcome_cost + gamma * delay_days,
                "acquired": acquired,
                "decision_days": delay_days,
            }
        # acquire
        event = peers[next_index]
        spent += evidence_cost
        acquired += 1
        current = event.date
        next_index += 1


def evaluate(test_rows, train_rows, by_company, model, peer_model, delay, bucket_surprise, config, gamma, evidence_cost, outcome_cost, max_acquire, earnings_events=None, margin=0.0, peer_rows=None, policies=None):
    rows = []
    for _, _, record in test_rows:
        horizon = event_date((earnings_events or {}).get(record["claim_id"])) or as_date(record["outcome"]["date"])
        peers = build_events(peer_rows if peer_rows is not None else train_rows + test_rows, record, horizon, earnings_events)
        selected_policies = policies or ("fixed_answer", "fixed_adjudicate", "fixed_abstain", "always_wait", "planner_no_evidence", "planner_evidence")
        for policy in selected_policies:
            if policy == "fixed_answer":
                result = simulate(record, horizon, by_company, peers, model, peer_model, delay, bucket_surprise, config, gamma, evidence_cost, outcome_cost, 0, "fixed", margin)
                # fixed policies stop at t0; use the requested action.
                probabilities = belief(model, record, by_company, peers, as_date(record["cutoff"]), str(record.get("sector") or "other"))
                decision = action_loss("answer", record["outcome"]["bucket"], config, prediction=max(LABELS, key=lambda label: probabilities[label]))
                result.update(policy=policy, outcome="stop", action="answer", total_loss=decision, decision_loss=decision, evidence_cost=0.0, delay_cost=0.0, acquired=0, decision_days=0.0)
            elif policy == "fixed_adjudicate":
                probabilities = belief(model, record, by_company, peers, as_date(record["cutoff"]), str(record.get("sector") or "other"))
                adjudication = credible_set(probabilities, config.alpha)
                decision = action_loss("adjudicate", record["outcome"]["bucket"], config, credible_set=adjudication)
                result = {"policy": policy, "outcome": "stop", "action": "adjudicate", "total_loss": decision, "decision_loss": decision, "evidence_cost": 0.0, "delay_cost": 0.0, "acquired": 0, "decision_days": 0.0}
            elif policy == "fixed_abstain":
                result = {"policy": policy, "outcome": "stop", "action": "abstain", "total_loss": config.abstain, "decision_loss": config.abstain, "evidence_cost": 0.0, "delay_cost": 0.0, "acquired": 0, "decision_days": 0.0}
            elif policy == "always_wait":
                delay_days = max(0.0, (as_date(horizon) - as_date(record["cutoff"])).days)
                result = {"policy": policy, "outcome": "wait", "action": "wait", "total_loss": outcome_cost + gamma * delay_days, "decision_loss": 0.0, "evidence_cost": 0.0, "delay_cost": outcome_cost + gamma * delay_days, "acquired": 0, "decision_days": delay_days}
            else:
                result = simulate(record, horizon, by_company, peers, model, peer_model, delay, bucket_surprise, config, gamma, evidence_cost, outcome_cost, 0 if policy == "planner_no_evidence" else max_acquire, policy, margin)
            result["claim_id"] = record["claim_id"]
            result["truth"] = record["outcome"]["bucket"]
            result["year"] = record["cutoff_year"]
            rows.append(result)
    return rows


def aggregate(rows):
    by_policy = defaultdict(list)
    for row in rows:
        by_policy[row["policy"]].append(row)
    out = {}
    for policy, items in by_policy.items():
        out[policy] = {
            "n": len(items),
            "mean_total_loss": float(np.mean([item["total_loss"] for item in items])),
            "mean_decision_loss": float(np.mean([item["decision_loss"] for item in items])),
            "mean_evidence_cost": float(np.mean([item["evidence_cost"] for item in items])),
            "mean_delay_cost": float(np.mean([item["delay_cost"] for item in items])),
            "wait_rate": float(np.mean([item["outcome"] == "wait" for item in items])),
            "acquire_rate": float(np.mean([item["acquired"] > 0 for item in items])),
            "mean_acquired": float(np.mean([item["acquired"] for item in items])),
            "mean_decision_days": float(np.mean([item["decision_days"] for item in items])),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "paper/results/sequential")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--evidence-cost", type=float, default=0.5)
    parser.add_argument("--outcome-cost", type=float, default=2.0)
    parser.add_argument("--max-acquire", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--earnings-events", type=Path, default=REPO_ROOT / "paper/data/earnings_events.json")
    parser.add_argument("--margin", type=float, default=0.0)
    args = parser.parse_args()

    rows = load_records(args.data)
    earnings_events = {}
    if args.earnings_events.exists():
        earnings_events = json.loads(args.earnings_events.read_text(encoding="utf-8"))
    if args.verified_only:
        rows = [
            row
            for row in rows
            if row[2].get("verification") in {"xbrl_search", "xbrl_verified", "xbrl_override"}
        ]
    train = [row for row in rows if row[1].year <= args.train_end_year]
    test = [
        row
        for row in rows
        if row[1].year >= args.test_start_year and row[2].get("split") == "test"
    ]
    by_company = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    print(f"train={len(train)} test={len(test)}")
    model, peer_model, delay, bucket_surprise = train_models(
        train, by_company, seed=args.seed, earnings_events=earnings_events
    )
    print(f"expected delay={delay:.1f} days")
    # Belief quality at t0 and at 50% of the expected delay.
    for fraction in (0.0, 0.5, 1.0):
        truths, preds = [], []
        for _, _, record in test:
            peers = build_events(rows, record, earnings_events.get(record["claim_id"]), earnings_events)
            upto = as_date(record["cutoff"]) + timedelta(days=fraction * delay)
            probabilities = belief(model, record, by_company, peers, upto, str(record.get("sector") or "other"))
            truths.append(record["outcome"]["bucket"])
            preds.append(max(LABELS, key=lambda label: probabilities[label]))
        print(
            f"belief@{fraction:.2f} balanced={balanced_accuracy_score(truths, preds):.4f} "
            f"macroF1={f1_score(truths, preds, average='macro'):.4f}"
        )
    config = LossConfig()
    results = evaluate(
        test, train, by_company, model, peer_model, delay, bucket_surprise,
        config, args.gamma, args.evidence_cost, args.outcome_cost, args.max_acquire,
        earnings_events=earnings_events, margin=args.margin,
    )
    summary = aggregate(results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "per_claim.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema": "finplan-sequential-v1",
        "data": str(args.data),
        "train": len(train),
        "test": len(test),
        "train_end_year": args.train_end_year,
        "test_start_year": args.test_start_year,
        "verified_only": args.verified_only,
        "expected_delay_days": delay,
        "gamma": args.gamma,
        "evidence_cost": args.evidence_cost,
        "outcome_cost": args.outcome_cost,
        "max_acquire": args.max_acquire,
        "margin": args.margin,
        "earnings_events": str(args.earnings_events),
        "loss": config.as_dict(),
        "summary": summary,
    }
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
