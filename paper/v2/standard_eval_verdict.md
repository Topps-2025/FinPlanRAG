# Standard-metric verdict (2026-10-31)

**Question:** does the current framework have a positive result under metrics
that do not depend on hand-set decision-loss weights?

**Metrics:** accuracy, macro F1, credible-set coverage, ECE (10 fixed bins),
selective accuracy at 50% coverage, AURC, closure, wrong-document rate, future
leakage, paired McNemar, paired bootstrap. No asymmetric loss weights.

**Artifacts:** `scripts/run_standard_eval.py`,
`paper/results/standard_eval/report.md`, `paper/results/standard_eval/standard_eval.json`.

## Verdict table

| Dataset | Task | Framework | Strongest control | Verdict |
|---|---|---|---|---|
| validation8 (24 questions) | typed period retrieval | v9 | metadata_v9_fill | tied (24/24 vs 24/24) |
| LOFin (n=1,572) | typed filing retrieval | v9 | metadata_v9_fill | tied (.8919/.3108 vs .8919/.3108) |
| FinGLM (n=1,829) | typed filing retrieval | v9 | metadata_v9_fill | tied (.9978/.0126 vs .9978/.0126) |
| v4 guidance (test, n=273) | 3-way outcome | full | always_above | tied (.4945 accuracy) |
| v2 production (test, n=7) | timing x stage | full | static | tied (.2857 joint) |
| AnchorEvolve synthetic | timing x stage | full | static | positive (.667 vs .500) |
| controlled planning (5,600) | relation-evidence | finplan_state | generic_adaptive | positive (.3655 vs .3425 closure) |

## New v2 framework on v4 (not the old artifacts)

`scripts/run_standard_eval_v2_framework.py` runs the **new**
`src/finplan_rag/decision/` planner on the v4 test split with standard metrics.

**Forecast-only (no waiting for the outcome filing):**

| policy | top-1 acc | macro F1 | coverage | mean set size | ECE | selective@50 | AURC |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_answer | .4615 | .2105 | .4615 | 1.00 | .1005 | .5036 | .4971 |
| always_adjudicate | .4615 | .2105 | 1.0000 | 3.00 | .1005 | .5036 | .4971 |
| fixed_budget_adjudicate | .4615 | .2105 | .9707 | 2.64 | .2037 | .4380 | .5657 |
| **planner_no_wait (new)** | **.4615** | **.2105** | **.9707** | **2.64** | **.1638** | **.4526** | **.5645** |

- The new planner does **not** improve top-1 accuracy or macro F1 over the
  majority baseline.
- Its ECE is better than the fixed-budget adjudicator (.1638 vs .2037) but
  worse than `always_answer` (.1005); its selective accuracy is worse than
  `always_answer` (.4526 vs .5036).
- It achieves .9707 coverage with a mean set size of 2.64; the always-adjudicate
  baseline gets 1.0 coverage with set size 3.00. This is a coverage/set-size
  trade-off, not a top-1 gain.

**Wait-allowed:** all policies reach 1.0 top-1 accuracy by waiting for the
outcome filing. This is a decision-timing result, not a forecast; it is trivial
and should not be presented as a method win.

## Year-wise company-disjoint benchmark (2026-11-01)

`scripts/run_company_disjoint_benchmark.py` evaluates all 273 test claims under
a company-disjoint, point-in-time protocol with five fold seeds. On all labels,
the evidence model improves **macro F1** to .5042 vs .2105 (ensemble 95% CI
[.2278, .3587]) and **balanced accuracy** to .5656 vs .3333 (CI [.1668,
.2947]); every per-seed macro-F1 and balanced-accuracy CI excludes zero. Plain
accuracy is +.0527 (claim-level CI [-.0308, .1370]), selective@50 is .5839 vs
.4615 (CI [.0020, .2321]), and AURC is .4120 vs .4973 (CI [-.2356, -.0074]).

The XBRL-verified sensitivity (n=227, excluding 46 text-only labels) preserves
both robust gains: macro F1 .4884 vs .2203 (CI [.1942, .3380]) and balanced
accuracy .5635 vs .3333 (CI [.1643, .2937]). It does **not** preserve plain
accuracy (+.0009, CI [-.0907, .0925]), selective (CI [-.0856, .1648]), or AURC
(CI [-.1758, .0857]). The robust real-data claim is therefore macro F1 and
balanced accuracy, not uniform plain-accuracy/selective superiority.

A featureless prior-sampling control (sample labels from the training-fold
class distribution) gets balanced accuracy .3457 and macro F1 .3366 on all
labels, and .3396/.3340 on the verified set. The model remains far above this
class-rebalancing control, so the gain is not merely a change in predicted-class
frequencies.

`scripts/run_real_decision_eval.py` evaluates the decision layer on the same
folds. On calibrated, XBRL-verified probabilities, the planner beats
always-answer by about -.72 decision loss (four of five per-seed CIs exclude
zero; the fifth is borderline), ties always-adjudicate at the base loss (all
per-seed CIs include zero), and has mean regret +.052 (max +.383) across ten
loss configurations, versus +.498 (max +3.063) for always-adjudicate, +1.429
(max +3.685) for always-answer, and +1.710 (max +6.081) for always-abstain.
This is a real-data decision-interface positive: low regret across the loss
frontier, not uniform dominance at one weight vector.

## Overall answer

- **Real-data outcome prediction (v4, v2): no positive result.** The framework
  ties the strongest control.
- **New v2 framework on v4 forecast-only: no positive result.** Top-1 .4615 and
  macro F1 .2105, the same as the majority baseline; ECE and selective accuracy
  are worse than `always_answer`. The only favorable comparison is coverage at
  a smaller set size than always-adjudicate, which is a trade-off, not a
  top-1 gain.
- **New v2 framework wait-allowed: trivial.** Waiting for the outcome filing
  gives 1.0 accuracy; this is decision timing, not a forecast.
- **Real-data retrieval (LOFin, FinGLM): positive vs weak baselines, tied with
  the strongest same-obligation control.** v9 beats single-shot, generic, and
  HiREC on wrong-document rate; metadata_v9_fill ties it exactly. The
  period-metadata comparison is mixed (LOFin: v9 better wrong-doc, worse
  closure; FinGLM: v9 better closure, worse wrong-doc).
- **Real-data parser holdout (validation8): positive vs parser versions and
  weak baselines, tied with metadata_v9_fill.** v9 closes 24/24 vs 4-11/24
  (p <= .00025); metadata_v9_fill also closes 24/24 (p = 1.0).
- **Synthetic / controlled mechanism: positive.** The controlled planning
  diagnostic and the synthetic self-evolution run show the mechanism working;
  these are not real-data claims.
- **Controlled multi-seed (20 seeds): positive.** At cost budget 1.0 on the
  controlled MA subset, `planner_preliminary` top-1 accuracy .8986
  [.8855, .9117] vs the pre-hoc cost-aware router .4695; paired **+0.4290
  [0.4110, 0.4471]**, 20/20 seeds positive. The real v4 control is negative.
- **Real point-in-time evidence benchmark: positive on balanced accuracy and
  macro F1.** A logistic model on guidance/current-quarter/history/previous-
  guidance features improves macro F1 on the temporal test (.4950 vs .2105,
  20-seed CI [.4881, .5020]) and on the year-wise company-disjoint benchmark
  (.5042 vs .2105, CI [.2278, .3587]); balanced accuracy is .5742 vs .3333 on
  the temporal test and .5656 vs .3333 on the year-wise benchmark (CI [.1668,
  .2947]). Both gains survive
  restricting to XBRL-verified labels: macro F1 .4884 vs .2203 (CI [.1942,
  .3380]), balanced accuracy .5635 vs .3333 (CI [.1643, .2937]). Plain
  accuracy/selective/AURC gains on all labels do not survive that restriction.
- **Real decision layer: positive on the loss frontier.** On calibrated,
  XBRL-verified probabilities the planner beats always-answer by about -.72
  (four of five per-seed CIs exclude zero; the fifth is borderline), ties
  always-adjudicate at the base loss, and has mean regret +.052 (max +.383)
  across ten loss configurations versus +.498 (max +3.063) for
  always-adjudicate.

**Conclusion:** the current framework has a robust **representation and
controlled-mechanism** positive, a robust **real-data balanced-accuracy and
macro-F1** positive on a company-disjoint benchmark, and a **real-data
decision-layer** positive (near-optimal across the loss frontier). It does
**not** have a uniform plain-accuracy/selective win: those gains are
label-sensitive and the planner is slightly worse than the best fixed
adjudicator at the base loss (3.997 vs 3.919; fixed wins 5/5 seeds). The empirical package supports a benchmark + decision-interface paper
with explicit boundaries, not a uniform method-win claim.

## What would count as a real-data positive

The year-wise benchmark meets a weaker version of (1): the planner beats
always-answer with confidence intervals excluding zero and has low regret
across the loss grid. It does not meet (2) or (3), and it does not compare
against a pre-hoc cost-aware router on a real mode-ambiguous subset.

1. On a real evidence-rich benchmark, the planner beats a pre-hoc cost-aware
   router on the mode-ambiguous subset (paired CI excludes 0).
2. The framework beats metadata_v9_fill on closure **and** wrong-document rate
   without losing the other metric.
3. A pre-registered answer-level gold with a shared reader shows a gain over
   the best same-obligation control.

## Why this is the right verdict

- It uses held-out labels and standard metrics, not a hand-tuned loss.
- It distinguishes the representation/parser result from the planning-policy
  result.
- It prevents the paper from claiming a real-data method win that the current
  evidence does not support.
- It identifies the exact missing experiment: the evidence-rich,
  mode-ambiguous real benchmark.
