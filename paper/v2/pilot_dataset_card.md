# Pilot dataset card: v4 decision-level benchmark

**Status:** pilot, 2026-10-31. This is a decision-level abstraction of the
1,005-claim v4 revenue-guidance dataset. It is not a new raw-text corpus and it
is not an end-to-end reader benchmark.

## Source

- 1,005 quarterly revenue-guidance claims from US SEC 8-K filings, 2010-2025.
- The guidance range is in the filing text; the realized quarter is from SEC
  XBRL (or the first reported filing where XBRL has no standalone quarter).
- Source project: the v4 benchmark in the sibling AnchorEvolve workspace.

## Task

At a cutoff, decide whether the realized revenue will land **below**,
**within**, or **above** the guided range. The later outcome filing is an
evidence item, not a label the policy may use without paying the delay cost.

This is a **decision-timing** task: the policy chooses whether to answer or
adjudicate now, or to wait for the later filing. It is not a forecast-only task.

## Decision-level mapping

| Object | Mapping |
|---|---|
| Hypotheses | `below`, `within`, `above` |
| Guidance evidence | Available at cutoff; cost 0.5; payload low/high/mid/width |
| Current-quarter anchor | Available at cutoff; cost 0.5; payload value/reliability |
| Outcome filing | Available at the outcome date; cost 2.0; payload actual/bucket |
| Gold | Realized outcome bucket |
| Decision time | Endogenous; waiting is priced by delay |
| Expected delay | Company-historical mean from dev outcomes; fallback global mean |
| Realized delay | Outcome date minus cutoff; used for accounting only |

## Counts

| Split | Claims | Above | Within | Below | Current anchor high/low |
|---|---:|---:|---:|---:|---|
| dev | 529 | 298 | 157 | 74 | 168 / 361 |
| validation | 203 | 114 | 40 | 49 | 76 / 127 |
| test | 273 | 126 | 110 | 37 | 97 / 176 |
| all | 1,005 | 538 | 307 | 160 | 341 / 664 |

- Realized delay: 62-180 days; test mean 97.7 days.
- Test years: 2022 (84), 2023 (70), 2024 (67), 2025 (52).
- Test sectors: other 113, semiconductor 83, software 71, hardware 5, health 1.

## Splits and leakage

- Temporal split: dev 2010-2018, validation 2019-2021, test 2022-2025.
- The test distribution is natural, not class-balanced.
- The outcome filing is admissible only at or after its filing date.
- Waiting advances the decision time to that date and incurs the realized
  delay cost. The policy may not observe the outcome filing earlier.
- Audit: leakage rate is 0 for all policies in the pilot.
- The expected delay is estimated from dev outcomes per company; for dev
  evaluation it falls back to the global dev mean to avoid using the same
  split's outcomes. The realized delay is used for accounting, not for the
  planner's decision.

## Evidence strength

The current-quarter anchor is a weak signal in this data. The pilot fits
`P(outcome | current-quarter position)` on dev, smooths it, and blends it with
the dev prior. The empirical table is close to the prior, so the planner often
skips the anchor when its cost is not justified. This is a measured negative
result for this evidence type, not a claim that the decision interface fails.

## Controlled preliminary extension

The mode-switch diagnostic adds a synthetic preliminary evidence item:

- available at cutoff + 30 days, cost 0.5;
- half the cases have precision 0.9 and half 0.5;
- assignment is deterministic from the case id and independent of the query
  features a pre-hoc router can see;
- the MA subset is the 133 test cases with precision 0.9;
- `planner_preliminary` 1.4617 vs `planner_no_evidence` 2.9626, paired
  −1.5009 [−1.875, −1.072].

This is a controlled mechanism diagnostic, not a real-data result. The real
v4 current-quarter anchor remains a negative result.

## Limitations

- Silver labels: the guidance range and outcome are automatically extracted;
  quotes are paraphrases until human verification.
- Revenue guidance only; no EPS, capex, production timing, or mixed queries.
- The future outcome filing is exact in the pilot; waiting can reveal the
  answer. The precision-sensitivity run (0.8/0.6) shows the planner stops
  waiting when the future signal is noisy.
- The current-quarter anchor is weak; evidence-induced mode switching is not
  demonstrated on this dataset.
- No reader/answer-level gold; the pilot is decision-level.
- No human annotation or inter-annotator agreement yet.

## Files

- `src/finplan_rag/decision/pilot_data.py`: loader and mapping.
- `src/finplan_rag/decision/run_pilot.py`: pilot runner.
- `paper/results/v2_pilot/summary.json`: metrics and paired bootstrap.
- `paper/results/v2_pilot/per_case.jsonl`: auditable per-case outputs.
- `paper/results/v2_pilot/cost_loss_frontier.csv`: cost-loss frontier.
- `paper/results/v2_pilot/loss_sensitivity.json`: loss-parameter sensitivity.
