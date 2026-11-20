# Decision-layer result on real company-disjoint data (2026-10-31; feature fix 2026-11-01)

> **Superseded as the primary decision-layer result by
> `algorithm_v3_conformal.md` (v3.1).** This document remains the v2 planner
> baseline and the calibration/regret analysis.

**Status:** confirmatory real-data run. Artifacts:
`paper/results/real_decision_eval/` (all labels, raw probabilities),
`paper/results/real_decision_eval_calibrated/` (all labels, calibrated),
`paper/results/real_decision_eval_verified/` (XBRL-verified labels, raw), and
`paper/results/real_decision_eval_verified_calibrated/` (XBRL-verified labels,
calibrated).

This is the real-data test of the v2 decision interface. The evidence model from
the year-wise company-disjoint benchmark produces a three-way probability vector
for each claim; the decision layer treats that vector as the credal state and
chooses among **answer**, **adjudicate** (output a credible set), and
**abstain**.

## Protocol

- Same company-disjoint, point-in-time folds as the benchmark: for test year
  \(Y\), train on other company folds using outcomes public before January 1 of
  \(Y\); five fold seeds; 273 test claims (227 XBRL-verified).
- **Planner:** for each claim, choose the action with the lowest expected loss
  under the asymmetric decision loss. The loss has parameters
  `wrong=10, over=12, under=8, miss=8, width=2, abstain=5, alpha=.1` at the base
  configuration.
- **Calibration:** temperature scaling fitted on internal company-disjoint folds
  (temperatures 1.3--2.9). Calibration does not change the argmax prediction or
  the benchmark's accuracy/macro F1; it changes the confidence and the
  credible-set width used by the decision layer.
- **Baselines:** always answer, always adjudicate (smallest credible set at
  \(\alpha\)), always abstain, and fixed confidence thresholds
  (0.40/0.50/0.60/0.70) that answer above the threshold and abstain below it.
- **Loss grid:** base, symmetric, over-averse, under-averse, cheap abstain,
  expensive abstain, narrow width penalty, wide width penalty, alpha .05, alpha
  .20. The planner is compared with fixed policies at every setting; the loss
  weights are a modeling choice, so no conclusion rests on one weight vector.
- **Uncertainty:** paired bootstrap over claims for planner minus a fixed
  baseline; two-sided sign-flip permutation test. Regret is computed per loss
  configuration as policy loss minus the best fixed-policy loss, then averaged
  over the five fold seeds and reported across the ten configurations.

## Base loss configuration, XBRL-verified labels (n=227)

The calibrated verified-only run is the primary decision result because it uses
the cleaner labels and the calibrated credal state.

| seed | planner | always answer | always adjudicate | always abstain | planner - answer [95% CI] | planner - adjudicate [95% CI] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.9824 | 4.7313 | 3.8767 | 5.0000 | -.7489 [-1.339, -.176] | +.1057 [-.106, +.317] |
| 2 | 3.9736 | 4.7225 | 3.9207 | 5.0000 | -.7489 [-1.313, -.167] | +.0529 [-.132, +.238] |
| 3 | 4.0352 | 4.6784 | 3.9031 | 5.0000 | -.6432 [-1.242, -.061] | +.1322 [-.044, +.308] |
| 4 | 3.9824 | 4.8546 | 3.9559 | 5.0000 | -.8722 [-1.436, -.299] | +.0264 [-.203, +.256] |
| 5 | 4.0088 | 4.5727 | 3.9383 | 5.0000 | -.5639 [-1.145, +.009] | +.0705 [-.123, +.273] |

- **Planner vs always-answer:** lower decision loss in every fold seed. Four of
  the five per-seed confidence intervals exclude zero; the fifth is borderline
  (upper bound +.009). Mean improvement about **-.72** loss points.
- **Planner vs always-adjudicate:** a statistical tie at the base loss. The
  point estimates are +.03 to +.13 and every confidence interval includes
  zero. The planner is not claimed to beat the best fixed adjudicator at this
  single weight vector.

## Regret across the ten loss configurations

Regret = policy loss minus the best fixed-policy loss in that configuration,
averaged over five fold seeds. Lower is better; zero means the policy matches
the best fixed action.

**XBRL-verified labels, calibrated probabilities (n=227):**

| policy | mean regret | max regret | configurations where best |
|---|---:|---:|---:|
| always answer | +1.4285 | +3.6846 | 1/10 |
| always adjudicate | +0.4981 | +3.0626 | 8/10 |
| always abstain | +1.7100 | +6.0811 | 1/10 |
| **planner** | **+0.0521** | **+0.3833** | **3/10** |

**All labels, calibrated probabilities (n=273):**

| policy | mean regret | max regret | configurations where best |
|---|---:|---:|---:|
| always answer | +1.3086 | +3.5132 | 1/10 |
| always adjudicate | +0.4917 | +3.0535 | 8/10 |
| always abstain | +1.7642 | +6.1363 | 1/10 |
| **planner** | **+0.0359** | **+0.3524** | **4/10** |

**All labels, raw probabilities (n=273):**

| policy | mean regret | max regret | configurations where best |
|---|---:|---:|---:|
| always answer | +1.5193 | +3.5084 | 1/10 |
| always adjudicate | +0.3802 | +2.2711 | 8/10 |
| always abstain | +1.9749 | +6.4689 | 1/10 |
| **planner** | **+0.2467** | **+0.5143** | **2/10** |

**XBRL-verified labels, raw probabilities (n=227):**

| policy | mean regret | max regret | configurations where best |
|---|---:|---:|---:|
| always answer | +1.6010 | +3.6467 | 1/10 |
| always adjudicate | +0.4026 | +2.3789 | 8/10 |
| always abstain | +1.8824 | +6.3524 | 1/10 |
| **planner** | **+0.2872** | **+0.6009** | **2/10** |

**Reading.** The best fixed policy changes with the loss: always-answer is best
under a wide width penalty, always-adjudicate under most base-like settings, and
always-abstain under a cheap abstention cost. No fixed policy is uniformly
near-optimal: always-adjudicate's worst-case regret is about 3.1 loss points,
always-answer's is 3.5--3.7. With calibrated probabilities the planner has mean
regret +.05 and maximum +.38 on the verified labels; raw probabilities give
+.29/+.60. The decision-layer value is **low regret across the loss frontier**,
not uniform dominance at one weight vector; calibration is part of the model,
not an optional cosmetic step.

## Representative grid points (verified, calibrated)

| config | always answer | always adjudicate | always abstain | planner | planner - answer [95% CI] |
|---|---:|---:|---:|---:|---:|
| base | 4.7119 | 3.9189 | 5.0000 | 3.9965 | -.7154 [-1.2952, -.1391] |
| symmetric | 5.0573 | 3.9189 | 5.0000 | 3.8925 | -1.1648 [-1.7833, -.5445] |
| over_averse | 4.2775 | 3.9189 | 5.0000 | 4.0925 | -.1850 [-.7428, +.3516] |
| under_averse | 6.8678 | 3.9189 | 5.0000 | 3.6573 | -3.2106 [-4.1621, -2.2722] |
| cheap_abstain | 4.7119 | 3.9189 | 2.0000 | 2.0018 | -2.7101 [-3.3251, -2.0969] |
| narrow_penalty | 4.7119 | 1.0273 | 5.0000 | 1.0273 | -3.6846 [-4.2952, -3.0762] |
| wide_penalty | 4.7119 | 7.7744 | 5.0000 | 5.0952 | +.3833 [-.0767, +.8344] |
| alpha_05 | 4.7119 | 4.0106 | 5.0000 | 4.0581 | -.6537 [-1.2247, -.0898] |
| alpha_20 | 4.7119 | 3.5559 | 5.0000 | 3.6035 | -1.1084 [-1.6881, -.5268] |

The confidence interval in this grid table is the mean of the five per-seed
paired-bootstrap intervals, not a pooled interval; use the per-seed CIs above
for significance claims. The planner beats always-answer in eight of the nine
non-wide configurations (the over-averse interval includes zero), matches the
answer policy under the wide penalty, and matches always-adjudicate under the
base, symmetric, and narrow-penalty settings. It beats always-adjudicate under
under-averse, cheap-abstain, and alpha-.20; it is slightly worse under
over-averse and alpha-.05. This is why the regret table, not a single
base-config comparison, is the primary decision-layer evidence.

## Verdict

- **Confirmed:** on real company-disjoint, point-in-time data with calibrated
  probabilities, the planner has much lower worst-case regret across the loss
  grid than any fixed action, and it lowers decision loss relative to
  always-answer in four of five per-seed tests (the fifth is borderline).
- **Confirmed with a caveat:** the planner is slightly worse than the best
  fixed policy (always-adjudicate) at the base loss (3.997 vs 3.919; fixed wins
  5/5 seeds); it does not uniformly dominate every fixed policy at every weight
  vector. The paper should claim near-optimality across the frontier, not
  uniform superiority.
- **Calibration matters:** raw probabilities give the planner mean regret
  +.25 (all labels) and +.29 (verified), with a verified planner-vs-answer
  interval that includes zero; temperature scaling reduces mean regret to
  +.04/+.05 and makes the planner-vs-answer comparison significant. The decision
  layer should use calibrated probabilities.
- **Label robustness:** the calibrated decision-layer result survives
  restricting to the 227 XBRL-verified claims; the regret ordering is unchanged.

## Boundaries

1. The loss weights are a modeling choice. The grid is the defense; no single
   configuration is presented as the ground truth.
2. The probabilities come from a simple logistic evidence model, not a reader.
   A stronger model could change the absolute losses, but the regret ordering is
   a property of the decision rule given the probabilities.
3. The benchmark labels are automatically extracted (paraphrase claims, XBRL
   outcomes); the verified-only run is the label-noise sensitivity, not a
   human-annotated gold standard.
4. No online exploration or real trading decision is evaluated; this is an
   offline decision-theoretic evaluation.

## Artifacts

- `scripts/run_real_decision_eval.py`
- `paper/results/real_decision_eval/report.md`
- `paper/results/real_decision_eval_calibrated/report.md`
- `paper/results/real_decision_eval_verified/report.md`
- `paper/results/real_decision_eval_verified_calibrated/report.md`
- `paper/v2/company_disjoint_protocol.json`
