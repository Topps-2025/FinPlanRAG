"""Finite counterexample experiment for the obligation-state research gap.

The experiment is intentionally small and exhaustive. It creates equal-score
confusable filing records and enumerates every presentation order. A
score-only selector cannot identify the intended period, while an
obligation-aware selector uses only the query's period and the cutoff. The
result is a mechanism check for the propositions in the paper, not a claim
about real-corpus accuracy.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable


def worlds(obligations: int, distractors_per_obligation: int, future: bool) -> Iterable[tuple[dict, ...]]:
    records: list[dict] = []
    for i in range(obligations):
        records.append({"obligation": i, "valid": True, "available_at": 1, "score": 1.0})
        for j in range(distractors_per_obligation):
            records.append({"obligation": i, "valid": False, "available_at": 1, "score": 1.0})
        if future:
            records.append({"obligation": i, "valid": False, "available_at": 2, "score": 1.0})
    yield from itertools.permutations(records)


def evaluate(obligations: int, distractors: int, future: bool) -> dict[str, float | int]:
    cases = 0
    score_closed = score_wrong = score_leak = 0
    aware_closed = aware_wrong = aware_leak = 0
    for order in worlds(obligations, distractors, future):
        cases += 1
        # Relevance-only policy: retrieve the top B equal-score records.
        selected = order[:obligations]
        score_valid = {r["obligation"] for r in selected if r["valid"]}
        score_closed += int(len(score_valid) == obligations)
        score_wrong += int(any(not r["valid"] for r in selected))
        score_leak += int(any(r["available_at"] > 1 for r in selected))

        # Obligation-state policy: one admissible record for each obligation.
        chosen = []
        for obligation in range(obligations):
            chosen.append(next(r for r in order if r["obligation"] == obligation and r["valid"] and r["available_at"] <= 1))
        aware_closed += int(len(chosen) == obligations)
        aware_wrong += int(any(not r["valid"] for r in chosen))
        aware_leak += int(any(r["available_at"] > 1 for r in chosen))

    def rate(value: int) -> float:
        return value / cases if cases else 0.0

    return {
        "obligations": obligations,
        "distractors_per_obligation": distractors,
        "future_records": int(future),
        "cases": cases,
        "score_only_closure": rate(score_closed),
        "score_only_wrong_document": rate(score_wrong),
        "score_only_future_leak": rate(score_leak),
        "obligation_state_closure": rate(aware_closed),
        "obligation_state_wrong_document": rate(aware_wrong),
        "obligation_state_future_leak": rate(aware_leak),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("paper/results/gap_proof_small.json"))
    args = parser.parse_args()
    rows = [
        evaluate(1, 1, False),
        evaluate(2, 1, False),
        evaluate(2, 2, False),
        evaluate(2, 1, True),
    ]
    result = {
        "schema": "finplan-research-gap-proof-small.v1",
        "status": "finite-counterexample-mechanism-check",
        "assumption": "all confusable records have identical relevance score; cutoff is 1; valid records are available before cutoff",
        "rows": rows,
        "interpretation": "The score-only selector has no information to break an equal-score fibre. The obligation-state selector uses typed obligation identity and cutoff metadata; this validates the mechanism construction, not real-corpus answer accuracy.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
