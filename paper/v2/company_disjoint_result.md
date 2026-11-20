# Year-wise company-disjoint benchmark: result (2026-10-31; feature fix 2026-11-01)

**Status:** confirmatory real-data run. Protocol:
`company_disjoint_protocol.json`. Artifacts:
`paper/results/company_disjoint_benchmark/` (all labels) and
`paper/results/company_disjoint_benchmark_verified/` (XBRL-verified labels).

This benchmark replaces the small `company_disjoint_test` subset (n=108) as the
main entity-generalization test. It evaluates **all 273 test claims** (2022--2025)
under a protocol that is company-disjoint and point-in-time by construction.

## Protocol

- Each company is assigned to one of five folds by a seeded hash.
- For test year \(Y\), the model is trained on claims from the other four folds
  whose **outcome date is before January 1 of \(Y\)**. The test company is
  never in the training set, and no outcome from \(Y\) or later is used.
- Five fold seeds (1--5) are run; each claim is evaluated once per seed.
- Model: multinomial logistic regression, `C=1.0`, `class_weight=balanced`.
- Features: guidance width, current-quarter vs guidance, company outcome
  history before cutoff, previous-guidance growth/width, target quarter, cutoff
  year, sector. (A feature-column indexing bug had one-hot encoded the width
  column instead of sector in the first run; it is fixed in the current
  artifacts. The corrected model is the one reported here.)
- Primary metrics: **macro F1** and **balanced accuracy** (mean per-class
  recall). Both are standard for the class-imbalanced three-way benchmark;
  plain accuracy is dominated by the `above` class and is secondary. Also
  reported: accuracy, per-class recall, selective accuracy@50, AURC, ECE,
  Brier, log loss.

## All test claims (n=273 per seed)

| seed | accuracy | balanced acc | balanced Δ [95% CI] | macro F1 | selective@50 | AURC | ECE | always-above acc | always-above macro F1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | .5238 | .5645 | +.2313 [+.1642, +.2950] | .5101 | .5766 | .4151 | .0819 | .4615 | .2105 |
| 2 | .5018 | .5554 | +.2222 [+.1575, +.2848] | .4912 | .5109 | .4432 | .1266 | .4615 | .2105 |
| 3 | .5165 | .5552 | +.2219 [+.1551, +.2873] | .5009 | .5839 | .4256 | .0836 | .4615 | .2105 |
| 4 | .4982 | .5404 | +.2070 [+.1399, +.2735] | .4849 | .5839 | .4125 | .0986 | .4615 | .2105 |
| 5 | .5311 | .5822 | +.2490 [+.1856, +.3107] | .5185 | .5620 | .4092 | .0781 | .4615 | .2105 |

**Featureless prior-sampling control.** A classifier that samples labels from
the training-fold class distribution (no features) gets accuracy .4103,
balanced accuracy .3457, and macro F1 .3366 on all labels; on the XBRL-verified
set it gets accuracy .4150, balanced accuracy .3396, and macro F1 .3340. The
model's balanced accuracy (.5656) and macro F1 (.5042) remain far above this
class-rebalancing control, so the gain is not only a change in predicted-class
frequencies.

**Pooled across fold assignments (1,365 rows; each claim repeated five times):**

| metric | model | always-above | paired 95% CI |
|---|---:|---:|---:|
| accuracy | .5143 | .4615 | [.0139, .0923] |
| **balanced accuracy** | **.5595** | **.3333** | — |
| **macro F1** | **.5013** | **.2105** | **[.2619, .3199]** |

Pooled McNemar: a-only 337, b-only 409, p=.009291. The pooled p-value treats the
five fold assignments as independent rows, which they are not; it is reported
for completeness, not as the conservative test.

**Claim-level analysis (conservative unit):** average each claim's correctness
over the five fold seeds, then bootstrap the 273 claims.

- mean accuracy difference vs always-above: **+.0527**, 95% CI
  **[-.0308, .1370]**, two-sided sign-flip p=.2201.

**Five-fold-seed probability ensemble:** average the five probability vectors
for each claim; every member still excludes the test company.

| metric | model | always-above | paired 95% CI |
|---|---:|---:|---:|
| accuracy | .5165 | .4615 | [-.0330, .1429] |
| **balanced accuracy** | **.5656** | **.3333** | **[.1668, .2947]** |
| **macro F1** | **.5042** | **.2105** | **[.2278, .3587]** |
| selective@50 | .5839 | .4615 | [.0020, .2321] |
| AURC | .4120 | .4973 | [-.2356, -.0074] |

Ensemble ECE .0766, Brier .5962, log loss .9953; McNemar vs always-above
a-only 66, b-only 81, p=.2481.

**Reading.** Both class-imbalance-robust metrics improve substantially and
significantly: balanced accuracy by about +.23 and macro F1 by about +.29. Every
per-seed balanced-accuracy and macro-F1 confidence interval excludes zero.
Plain accuracy is positive in all five fold seeds (+3.7 to +7.0 points) but its
claim-level confidence interval includes zero; selective accuracy and AURC are
positive on the all-label benchmark, but see the label-verification sensitivity
below.

## Robustness: XBRL-verified test labels only (n=227)

The dataset has 46 test claims whose outcome bucket comes from text-only
extraction (`text_only` or `text_only_mismatch`). Restricting evaluation to
claims whose label is `xbrl_search`, `xbrl_verified`, or `xbrl_override` gives:

| seed | n | accuracy | balanced acc | balanced Δ [95% CI] | macro F1 | selective@50 | AURC | ECE | always-above acc | always-above macro F1 |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 227 | .4934 | .5523 | +.2190 [+.1509, +.2846] | .4825 | .5263 | .4604 | .0963 | .4934 | .2203 |
| 2 | 227 | .4890 | .5597 | +.2266 [+.1613, +.2897] | .4831 | .4825 | .4968 | .1533 | .4934 | .2203 |
| 3 | 227 | .5022 | .5566 | +.2231 [+.1534, +.2899] | .4927 | .5439 | .4570 | .0825 | .4934 | .2203 |
| 4 | 227 | .4802 | .5385 | +.2049 [+.1347, +.2715] | .4716 | .5439 | .4396 | .1097 | .4934 | .2203 |
| 5 | 227 | .5066 | .5766 | +.2434 [+.1797, +.3055] | .4969 | .5351 | .4488 | .0910 | .4934 | .2203 |

- Pooled accuracy .4943 vs .4934, CI [-.0405, .0432]; **pooled balanced
  accuracy .5567 vs .3333**; pooled macro F1 .4855 vs .2203, CI [.2331, .2972];
  pooled McNemar p=1.
- Claim-level accuracy difference **+.0009**, 95% CI **[-.0907, .0925]**,
  sign-flip p=.994.
- Ensemble: accuracy .4978 vs .4934, CI [-.0881, .0969]; **balanced accuracy
  .5635 vs .3333, CI [.1643, .2937]**; **macro F1 .4884 vs .2203, CI
  [.1942, .3380]**; selective .5351 vs .4934, CI [-.0856, .1648]; AURC .4579 vs
  .4610, CI [-.1758, .0857].

**Conclusion.** The **balanced-accuracy and macro-F1 gains are robust** to
excluding text-only labels: balanced accuracy +.230 [.164, .294], macro F1
+.268 [.194, .338]. The plain-accuracy, selective-accuracy, and AURC gains on
the all-label benchmark are **not robust**: their verified-only confidence
intervals include zero. Do not present those three as confirmed
entity-generalization wins.

## Verdict

- **Confirmed (real, company-disjoint, point-in-time):** a simple evidence model
  improves **balanced accuracy** by about +.23 (verified .5635 vs .3333, CI
  [.1643, .2937]) and **macro F1** by about +.27--.29 (verified .4884 vs .2203,
  CI [.1942, .3380]; all-label .5042 vs .2105, CI [.2278, .3587]) across five
  fold seeds. Both gains survive excluding text-only labels.
- **Not confirmed:** a uniform plain-accuracy, selective-prediction, or AURC win
  on company-disjoint data. The all-label point estimates are positive, but the
  verified-only sensitivity does not preserve them.
- **Supersedes:** the old n=108 `company_disjoint_test` subset. That subset was
  small and its accuracy/selective negatives were not stable; the year-wise
  protocol evaluates the full 273-claim test set and separates the robust
  balanced-accuracy/macro-F1 claims from the non-robust plain-accuracy claims.

## Boundaries

1. **Silver-ish labels.** Guidance claims are automatically extracted
   paraphrases (`quote_type=paraphrase` for all 1,005 records). Outcomes are
   mostly XBRL actuals (936/1,005 XBRL-verified/override/search; 69 text-only),
   but the claim text and guidance range are not human-annotated. The
   verified-only sensitivity is the honest check; it preserves balanced
   accuracy and macro F1, not plain accuracy.
2. **Simple model.** The evidence model is a flat logistic regression. The
   result is a lower bound on what a stronger, hierarchical, or peer-feature
   model could do; it is not evidence that the current feature set is optimal.
3. **No reader.** This is a three-way outcome benchmark, not answer-level
   financial QA. It does not measure a reader or an end-to-end answer.
4. **Class imbalance.** Plain accuracy is dominated by the `above` class (.46 of
   the test set). Balanced accuracy and macro F1 are the primary metrics;
   selective accuracy and AURC are decision-layer diagnostics, not the main
   generalization claim.

## Artifacts

- `scripts/run_company_disjoint_benchmark.py`
- `paper/results/company_disjoint_benchmark/report.md`
- `paper/results/company_disjoint_benchmark/summary.json`
- `paper/results/company_disjoint_benchmark_verified/report.md`
- `paper/results/company_disjoint_benchmark_verified/summary.json`
- `paper/v2/company_disjoint_protocol.json`
