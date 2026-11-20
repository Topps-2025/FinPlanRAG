"""Sequential policy portfolio over real earnings/peer evidence.

Candidates: fixed answer, fixed adjudicate, fixed abstain, always wait,
planner without peer acquisition, planner with peer acquisition. The selector
starts from fixed adjudication and switches only when a candidate improves
validation loss by a margin. Validation is 2021, test is the 2022--2025
company-disjoint test split, and the model is fit on outcomes up to 2020.

This is the path-A counterpart of the paper's loss-aware portfolio: it tests
whether a policy that can wait and acquire evidence is selected across cost
regimes, using real SEC earnings releases and real peer outcomes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from finplan_rag.decision.loss import LossConfig
from run_real_benchmark import DEFAULT_DATA, company_key, load_records
from run_sequential_evidence import (
    aggregate,
    evaluate,
    train_models,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    "fixed_answer",
    "fixed_adjudicate",
    "fixed_abstain",
    "always_wait",
    "planner_no_evidence",
    "planner_evidence",
)


def select_policy(losses, margin, default="fixed_adjudicate"):
    selected = default
    for name in CANDIDATES:
        if name not in losses:
            continue
        if losses[name] + margin < losses[selected]:
            selected = name
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--earnings-events", type=Path, default=REPO_ROOT / "paper/data/earnings_events.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "paper/results/sequential_portfolio")
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--selection-margin", type=float, default=0.1)
    parser.add_argument("--planner-margin", type=float, default=0.2)
    parser.add_argument("--max-acquire", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gammas", default="0.0,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--evidence-costs", default="0.1,0.5")
    parser.add_argument("--outcome-costs", default="2.0,5.0")
    args = parser.parse_args()

    rows = load_records(args.data)
    if args.verified_only:
        rows = [
            row
            for row in rows
            if row[2].get("verification") in {"xbrl_search", "xbrl_verified", "xbrl_override"}
        ]
    fit = [row for row in rows if row[1].year <= 2020]
    val = [row for row in rows if row[1].year == 2021]
    test = [
        row
        for row in rows
        if row[1].year >= 2022 and row[2].get("split") == "test"
    ]
    by_company = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    earnings_events = {}
    if args.earnings_events.exists():
        earnings_events = json.loads(args.earnings_events.read_text(encoding="utf-8"))
    print(f"fit={len(fit)} val={len(val)} test={len(test)}")
    model, peer_model, delay, bucket_surprise = train_models(
        fit, by_company, seed=args.seed, earnings_events=earnings_events
    )
    print(f"expected delay={delay:.1f} days")
    config = LossConfig()
    fixed_val = evaluate(
        val, fit, by_company, model, peer_model, delay, bucket_surprise,
        config, 0.0, 0.0, 0.0, 0, earnings_events=earnings_events,
        peer_rows=rows, policies=("fixed_answer", "fixed_adjudicate", "fixed_abstain"),
    )
    fixed_test = evaluate(
        test, fit, by_company, model, peer_model, delay, bucket_surprise,
        config, 0.0, 0.0, 0.0, 0, earnings_events=earnings_events,
        peer_rows=rows, policies=("fixed_answer", "fixed_adjudicate", "fixed_abstain"),
    )
    fixed_val_summary = aggregate(fixed_val)
    fixed_test_summary = aggregate(fixed_test)
    results = []
    for gamma in [float(value) for value in args.gammas.split(",")]:
        for evidence_cost in [float(value) for value in args.evidence_costs.split(",")]:
            for outcome_cost in [float(value) for value in args.outcome_costs.split(",")]:
                val_rows = evaluate(
                    val,
                    fit,
                    by_company,
                    model,
                    peer_model,
                    delay,
                    bucket_surprise,
                    config,
                    gamma,
                    evidence_cost,
                    outcome_cost,
                    args.max_acquire,
                    earnings_events=earnings_events,
                    margin=args.planner_margin,
                    peer_rows=rows,
                    policies=("always_wait", "planner_no_evidence", "planner_evidence"),
                )
                test_rows = evaluate(
                    test,
                    fit,
                    by_company,
                    model,
                    peer_model,
                    delay,
                    bucket_surprise,
                    config,
                    gamma,
                    evidence_cost,
                    outcome_cost,
                    args.max_acquire,
                    earnings_events=earnings_events,
                    margin=args.planner_margin,
                    peer_rows=rows,
                    policies=("always_wait", "planner_no_evidence", "planner_evidence"),
                )
                val_summary = {**fixed_val_summary, **aggregate(val_rows)}
                test_summary = {**fixed_test_summary, **aggregate(test_rows)}
                val_losses = {
                    name: val_summary[name]["mean_total_loss"] for name in CANDIDATES
                }
                selected = select_policy(val_losses, args.selection_margin)
                test_loss = test_summary[selected]["mean_total_loss"]
                best_test = min(CANDIDATES, key=lambda name: test_summary[name]["mean_total_loss"])
                row = {
                    "gamma": gamma,
                    "evidence_cost": evidence_cost,
                    "outcome_cost": outcome_cost,
                    "selected": selected,
                    "selected_test_loss": test_loss,
                    "best_test_policy": best_test,
                    "best_test_loss": test_summary[best_test]["mean_total_loss"],
                    "regret": test_loss - test_summary[best_test]["mean_total_loss"],
                    "validation": val_losses,
                    "test": {
                        name: {
                            "total_loss": test_summary[name]["mean_total_loss"],
                            "decision_loss": test_summary[name]["mean_decision_loss"],
                            "evidence_cost": test_summary[name]["mean_evidence_cost"],
                            "delay_cost": test_summary[name]["mean_delay_cost"],
                            "wait_rate": test_summary[name]["wait_rate"],
                            "acquire_rate": test_summary[name]["acquire_rate"],
                            "decision_days": test_summary[name]["mean_decision_days"],
                        }
                        for name in CANDIDATES
                    },
                }
                results.append(row)
                print(
                    f"g={gamma:.3f} c={evidence_cost:.2f} o={outcome_cost:.1f} "
                    f"selected={selected} test={test_loss:.3f} best={best_test} "
                    f"regret={row['regret']:+.3f}"
                )
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "finplan-sequential-portfolio-v1",
        "n_fit": len(fit),
        "n_val": len(val),
        "n_test": len(test),
        "selection_margin": args.selection_margin,
        "planner_margin": args.planner_margin,
        "rows": results,
        "aggregate": {
            "mean_regret": float(np.mean([row["regret"] for row in results])),
            "max_regret": float(np.max([row["regret"] for row in results])),
            "positive_regret_count": int(sum(row["regret"] > 1e-12 for row in results)),
            "n_configs": len(results),
        },
    }
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
