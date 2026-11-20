# Real-data evidence benchmark: result (2026-10-31; feature fix 2026-11-01)

**Status:** confirmatory real-data run. Protocol: `real_benchmark_protocol.json`.
Data: v4 revenue guidance (dev 529 / validation 203 / test 273; old
company-disjoint flag subset 108). Features are real and point-in-time:
guidance width, current-quarter anchor vs guidance, company history (outcomes
public before cutoff), previous guidance, target quarter, year, sector.

**Model:** multinomial logistic regression; hyperparameters selected by
GroupKFold(5) **by company** on dev+validation; final fit on dev+validation;
test evaluated once. Selected: C=1.0, class_weight=balanced, grouped-CV macro
F1 .4492 ± .0527.

**Feature fix (2026-11-01):** the categorical encoder had been selecting column
0 of the concatenated matrix (the width feature) instead of the sector column.
The numbers below are from the corrected pipeline. The bug inflated the
categorical cardinality and discarded sector; the corrected model has a higher
macro F1 and lower ECE.

**Primary metric:** macro F1 (class-imbalanced v4 benchmark). Secondary:
accuracy, per-class recall, selective accuracy@50, AURC, ECE, Brier, log loss.

## Test split (n=273)

| metric | final model | always-above | 95% CI (paired bootstrap) |
|---|---:|---:|---|
| accuracy | .5238 | .4615 | [-.0220, .1429] |
| **balanced accuracy** | **.5742** | **.3333** | n/a |
| **macro F1** | **.5098** | **.2105** | **[.2325, .3615]** |
| selective@50 | .5766 | .4615 | n/a |
| AURC | .4074 | .4971 | n/a |
| ECE | .0763 | n/a | n/a |
| Brier | .6025 | n/a | n/a |
| log loss | 1.0055 | n/a | n/a |

McNemar vs always-above: a-only 57, b-only 74, p=.1619.

**20-seed training bootstrap (test):**

| metric | mean | 95% CI |
|---|---:|---|
| accuracy | .5143 | [.5070, .5216] |
| **macro F1** | **.4950** | **[.4881, .5020]** |
| selective@50 | .5912 | [.5700, .6125] |
| AURC | .4084 | [.3915, .4253] |

Against always-above (.4615 accuracy, .2105 macro F1, .4615 selective, .4971
AURC), the temporal test is **positive on macro F1, and positive on accuracy,
selective accuracy, and AURC under the 20-seed training bootstrap**; the
single-fit accuracy interval includes zero.

## Old company-disjoint test subset (n=108, superseded)

| metric | final model | always-above | 95% CI (paired bootstrap) |
|---|---:|---:|---|
| accuracy | .4630 | .5093 | [-.1944, .1019] |
| **balanced accuracy** | **.5088** | **.3333** | n/a |
| **macro F1** | **.4565** | **.2249** | **[.1173, .3351]** |
| selective@50 | .4259 | .5093 | n/a |
| AURC | .5575 | .5053 | n/a |
| ECE | .1449 | n/a | n/a |
| Brier | .6482 | n/a | n/a |
| log loss | 1.0484 | n/a | n/a |

McNemar vs always-above: a-only 36, b-only 31, p=.6254.

**20-seed training bootstrap (company-disjoint):**

| metric | mean | 95% CI |
|---|---:|---|
| accuracy | .4477 | [.4330, .4624] |
| **macro F1** | **.4176** | **[.4011, .4340]** |
| selective@50 | .4509 | [.4238, .4781] |
| AURC | .5242 | [.4989, .5496] |

Against always-above (.5093 accuracy, .2249 macro F1, .5093 selective, .5053
AURC), the old company-disjoint subset is **positive on macro F1 only**;
accuracy, selective accuracy, and AURC are worse. The subset is small (n=108)
and is superseded by the year-wise benchmark in `company_disjoint_result.md`.

## Verdict

**Real-data positive (qualified):**

- **Temporal test:** positive on macro F1 (.4950 vs .2105, 20-seed CI [.4881,
  .5020]) and balanced accuracy (.5742 vs .3333); positive on accuracy (.5143 vs
  .4615, CI [.5070, .5216]), selective@50 (.5912 vs .4615), and AURC (.4084 vs
  .4971) under the 20-seed training bootstrap. The single-fit test accuracy
  interval includes zero.
- **Year-wise company-disjoint:** macro F1 .5042 vs .2105 (CI [.2278, .3587])
  and balanced accuracy .5656 vs .3333 (CI [.1668, .2947]); both gains survive
  XBRL-verified labels (macro F1 .4884 vs .2203, CI [.1942, .3380]; balanced
  accuracy .5635 vs .3333, CI [.1643, .2937]). Plain accuracy, selective, and
  AURC gains are all-label only and do not survive the verified-only check. See
  `company_disjoint_result.md`.
- **Decision layer:** on calibrated probabilities, the planner beats
  always-answer by about -.72 and has mean regret +.052 (max +.383) across ten
  loss configurations, versus +.498 (max +3.063) for always-adjudicate. See
  `real_decision_result.md`.

**What this supports:** a real, point-in-time evidence model improves macro F1
and balanced accuracy on temporal and company-disjoint splits, and a decision
layer on its probabilities is near-optimal across a loss grid. This is writable
as a **benchmark + decision-interface** paper with explicit boundaries.

**What this does not support:** a uniform entity-generalization win on accuracy
or selective prediction. The all-label accuracy/selective/AURC gains are not
robust to excluding text-only labels, and the old n=108 negative is small and
superseded rather than resolved in favor of a uniform win.

## Why the accuracy trade-off happens

The model uses company history and previous guidance, which improve macro F1
and temporal accuracy but do not transfer cleanly to companies absent from
dev/validation. Company-disjoint companies have less history, and the model
over-predicts `within`/`below` relative to the always-above baseline. The
macro-F1 gain is real; the all-label accuracy gain is positive but
label-sensitive.

## What would strengthen the real-data positive

1. **New real evidence that generalizes across companies**: analyst estimates,
   supplier/customer disclosures, permits, equipment lead times, or
   industry-level aggregates.
2. A hierarchical model with company random effects and strong shrinkage; the
   current model is a flat logistic regression.
3. A larger company-disjoint test set; the year-wise protocol is an improvement,
   but 273 claims and 289 companies are still modest.
4. Human verification of the paraphrase claims.

## Artifacts

- `scripts/run_real_benchmark.py`
- `paper/results/real_benchmark/summary.json`
- `paper/results/real_benchmark/report.md`
- `paper/v2/real_benchmark_protocol.json`
- `paper/v2/company_disjoint_result.md` (supersedes the n=108 subset)
