# Algorithm v3.1: Loss-Aware Nested Conformal Decision Layer

**Status:** conformal alternative, 2026-11-03. The selected version is
`algorithm_selected_planner_rf.md` (credible-set planner with a non-LLM RF
backbone). This document remains the specification for the conformal layer and
matches `src/finplan_rag/decision/conformal.py` and the two runners named in
the code map. The default threshold objective is `realized`; the `expected` objective is
retained as an ablation. The conformal layer is not the selected primary
method: it loses to the planner on the RF backbone and its DeepSeek advantage
attenuates under company masking.

## 1. What stays from the original idea

The object is unchanged:

- a finite hypothesis space \(H_q=\{below,within,above\}\);
- a credal state \(b_t\) over \(H_q\);
- actions `answer(a)`, `adjudicate(R)`, `abstain`;
- an asymmetric decision loss \(L\) with overclaim, underclaim, miss, width,
  and abstention costs;
- a point-in-time decision at the cutoff; calibration and validation data are
  from other companies and earlier outcomes;
- no fixed cutoff or fixed evidence budget is assumed by the decision rule.
  The real benchmark evaluates one cutoff per claim; the dynamic acquire/wait
  actions are tested separately in the controlled diagnostic.

What changes is the **feasible-set construction** and the **selection of its
threshold**. The v2 planner used the heuristic credible set
\(\{R:b(R)\ge 1-\alpha\}\). v3.1 constructs a split-conformal set \(R_q(x)\)
from a score, selects the score family, the coverage level, and the threshold
on calibration/validation data before the test year, and chooses the action by
the same Bayes rule.

## 2. Backbones

The decision layer consumes a belief over the three labels. It is not a new
reader.

### 2.1 DeepSeek backbone (primary)

The base model is the DeepSeek verification prompt. For each claim it returns a
label \(a\), a confidence \(c\), and a credible set \(C\). The belief is

\[
b(a)=\min(0.95,\max(0.34,c)),\qquad
b(y)=\frac{1-b(a)}{|C\setminus\{a\}|}\quad (y\in C\setminus\{a\}),
\]

with \(b(a)=1\) when \(C=\{a\}\). If the model returns a label outside
\(H_q\), the first credible label is used. If the credible set is empty, the
singleton label is used. This is the same confidence-weighted belief used by
the planner baseline in `scripts/run_backbone_policy_eval.py`.

### 2.2 Logistic backbone (sensitivity)

A multinomial logistic regression on point-in-time structured features,
trained on other company folds with outcomes public before the test year.
Out-of-fold probabilities on the training companies form the calibration pool.
A company-disjoint 60/40 hash split gives calibration and validation sets.

The two backbones are not separate methods. They are two evidence models
consumed by the same decision layer. DeepSeek is the base model of the
framework; the logistic model is the cheap structured sensitivity.

## 3. Conformal state

For a probability vector \(p(x)\in\Delta(H_q)\), a score function
\(s(x,y)\), and a threshold \(q\), the feasible set is

\[
R_q(x)=\{y\in H_q: s(x,y)\le q\}.
\]

Two scores are supported:

- **LAC (least ambiguous set-valued classifier):**
  \(s_{\text{LAC}}(x,y)=1-p(y\mid x)\); equivalently include labels with
  \(p(y\mid x)\ge 1-q\).
- **APS (adaptive prediction sets):** sort labels by decreasing probability
  and set \(s_{\text{APS}}(x,y)\) to the cumulative mass through the true
  label. \(R_q\) includes labels until the cumulative mass reaches \(q\).

Both sets are non-empty by construction: if the score rule would emit an empty
set, the argmax label is inserted.

## 4. Loss-aware action

Given \(p\), a candidate set \(R\), and loss \(L\):

\[
\begin{aligned}
a^\star &= \arg\max_y p(y),\\
\ell_{\text{answer}} &= \sum_y p(y)L(\text{answer}(a^\star),y),\\
\ell_{\text{adjudicate}} &= \sum_y p(y)L(\text{adjudicate}(R),y),\\
\ell_{\text{abstain}} &= \ell_{\text{abstain}},\\
\text{action} &= \arg\min\{\ell_{\text{answer}},\ell_{\text{adjudicate}},\ell_{\text{abstain}}\}.
\end{aligned}
\]

The realized loss uses the true outcome. This is the same decision rule as the
v2 planner; only the set \(R\) changes.

## 5. Threshold selection: realized calibration loss

The v3 implementation selected \(q\) on calibration data by **expected** policy
loss under \(p\). That objective trusts the backbone probabilities, which are
miscalibrated for a long-tail financial label set. v3.1 selects \(q\) by the
**realized** policy loss on the labeled calibration set, subject to the
empirical coverage constraint:

\[
q^\star =
\arg\min_q \frac{1}{n_{\text{cal}}}\sum_{i\in D_{\text{cal}}}
L\bigl(\pi_p(R_q(x_i)),\,y_i\bigr)
\quad\text{s.t.}\quad
\frac{1}{n_{\text{cal}}}\sum_{i\in D_{\text{cal}}}
\mathbf 1[y_i\in R_q(x_i)]\ge 1-\alpha .
\]

The action \(\pi_p(R_q(x_i))\) is still chosen without \(y_i\); the label is
used only to score the threshold. This is empirical risk minimization with a
coverage constraint, not a claim of distribution-free coverage. The
`expected` objective is retained as an ablation. The two objectives are not
selected on test.

## 6. Nested selection (Algorithm 1)

Inputs: calibration probabilities/outcomes \(D_{\text{cal}}\), validation
probabilities/outcomes \(D_{\text{val}}\), loss \(L\), score candidates
\(M=\{\text{LAC},\text{APS}\}\), coverage candidates
\(A=\{0.05,0.10,0.20\}\), threshold objective \(O=\{\text{realized}\}\)
(default; `expected` is an ablation).

```text
for m in M, alpha in A:
    q_cal = argmin_q  realized_policy_loss(D_cal, R_q^m, L)
            subject to  coverage(D_cal, R_q^m) >= 1 - alpha
    evaluate q_cal on D_val -> validation_loss, validation_coverage

valid = candidates with validation_coverage >= 1 - alpha
if valid is non-empty:
    choose (m*, alpha*) with smallest validation_loss
else:
    choose the candidate with the largest validation_coverage (fallback)
    mark the fold as fallback

q* = argmin_q realized_policy_loss(D_cal, R_q^{m*}, L)
     subject to coverage(D_cal, R_q^{m*}) >= 1 - alpha*
return (m*, alpha*, q*)
```

At test time, compute \(R_{q^*}\), choose the action by the expected-loss rule,
and evaluate once. Test outcomes are never used for score selection, alpha
selection, or threshold selection.

## 7. Coverage and distribution shift

Under exchangeability of calibration and test, split conformal with a fixed
quantile gives \(\Pr(y\in R_{q^*})\ge 1-\alpha^*\) up to the finite-sample
correction. Two qualifications are explicit:

1. v3.1 selects \(q\) by realized loss subject to empirical calibration
   coverage. That selection can reduce test coverage relative to a fixed
   quantile; the empirical test coverage is reported for every configuration.
2. The year-wise company-disjoint test is **not** exchangeable: companies and
   years shift. We therefore do not claim a distribution-free guarantee under
   shift. A coverage shortfall is a validity boundary, not a hidden assumption.

The `fallback` flag records folds where no candidate met the validation
coverage constraint. DeepSeek has zero fallbacks in the reported runs; the
logistic backbone has fold-level fallbacks, which are reported rather than
removed.

## 8. Hyperparameters of record

- Score candidates: LAC and APS.
- Coverage candidates: \(\alpha\in\{0.05,0.10,0.20\}\).
- Threshold objective: `realized` (default); `expected` is an ablation.
- Threshold grid: 31 or 101 quantiles of the calibration scores plus 0 and 1;
  the reported DeepSeek runs use 101, the logistic five-seed runs use 31 for
  runtime and are marked as such.
- Calibration/validation split: company-disjoint 60/40 hash.
- DeepSeek: 732 development claims (2010--2021), 227 XBRL-verified test claims
  (2022--2025), five calibration split seeds.
- Logistic: five fold seeds, year-wise company-disjoint folds, verified-only
  test claims; the same 60/40 company split within each training pool.
- Loss: base `wrong=10, over=12, under=8, miss=8, width=2, abstain=5,
  alpha=0.1`, plus the ten-configuration sensitivity grid.

## 9. Complexity

Let \(n\) be calibration size, \(T\) the number of thresholds, \(|M|=2\),
\(|A|=3\), and \(K=3\) labels. Threshold selection is
\(O(|M||A|TnK)\); the backbone fits dominate. Prediction is \(O(nK)\) per
batch. The layer is CPU-only and deterministic. The realized objective adds a
per-threshold action evaluation but no model fit.

## 10. Code map

| Component | File |
|---|---|
| Conformal sets, scores, realized/expected objectives, nested fit | `src/finplan_rag/decision/conformal.py` |
| DeepSeek backbone + conformal layer + tuned planner/fixed baselines | `scripts/run_deepseek_conformal_set_eval.py` |
| Regret against the oracle fixed comparator | `scripts/analyze_conformal_regret.py` |
| Logistic five-seed verified runner | `scripts/run_conformal_decision_v3.py` |
| Earlier exploratory expected-objective grid (not the final class) | `scripts/run_conformal_selection_eval.py` |
| Tests | `tests/test_conformal.py` |

The default `ConformalConfig()` has
`threshold_objectives=("realized",)`, `threshold_objective="realized"`,
`n_thresholds=101`, `methods=("lac","aps")`,
`alphas=(0.05,0.10,0.20)`. The `expected` objective is run by passing
`ConformalConfig(threshold_objectives=("expected",), threshold_objective="expected")`.

## 11. Experiment protocol

### 11.1 DeepSeek (primary, verified labels)

For each of five calibration split seeds:

1. Load the 732 development claims and the 273 test claims. Keep the 227 test
   claims whose outcome is XBRL-verified.
2. Split the development companies 60/40 by a hash of the company key into
   calibration and validation.
3. Fit the layer on calibration/validation: select objective (realized),
   score family, alpha, and threshold.
4. Recalibrate the threshold on the full development calibration set with the
   selected method/alpha/objective.
5. Predict the test claims once.
6. Compare with the same belief and the same loss:
   - planner with alpha tuned on validation over \(\{.05,.10,.20\}\);
   - fixed adjudicator with alpha tuned on validation;
   - the model's reported static credible set;
   - the point label;
   - always-answer, always-abstain.
7. Paired bootstrap over the 227 claims, after averaging each claim's loss
   over the five split seeds; report 95% intervals.

### 11.2 Logistic (sensitivity, verified labels)

For each of five fold seeds and each test year, train on other company folds
with outcomes before January 1 of the test year; generate out-of-fold
probabilities on the training companies; split calibration/validation 60/40
by company; fit, recalibrate, and evaluate the test fold once. The same tuned
planner/fixed baselines are computed. The five seeds are averaged.

Test outcomes are never used for method, alpha, objective, or threshold
selection in either protocol.

## 12. Results

### 12.1 DeepSeek, XBRL-verified, 227 claims, five split seeds

Artifact: `paper/results/deepseek_conformal_realized_verified/` (`summary.json`,
`per_claim.jsonl`, `report.md`, `regret_summary.json`).

| config | conformal | planner tuned | fixed tuned | static set | point |
|---|---:|---:|---:|---:|---:|
| base | **3.1013** | 3.1189 | 3.1225 | 3.1542 | 5.0132 |
| symmetric | **3.1119** | 3.1278 | 3.1225 | 3.1542 | 4.9780 |
| over-averse | **3.1013** | 3.1189 | 3.1225 | 3.1542 | 5.2423 |
| under-averse | **3.1093** | 3.1304 | 3.1225 | 3.1542 | 4.9780 |
| cheap abstain | **1.9471** | 2.5110 | 3.1225 | 3.1542 | 5.0132 |
| expensive abstain | **3.1013** | 3.1189 | 3.1225 | 3.1542 | 5.0132 |
| narrow width | **1.0529** | 1.3172 | 1.2907 | 1.2907 | 5.0132 |
| wide width | **4.4467** | 4.5419 | 5.5330 | 5.6388 | 5.0132 |
| alpha .05 | **3.1013** | 3.1189 | 3.1225 | 3.1542 | 5.0132 |
| alpha .20 | **3.1013** | 3.1189 | 3.1225 | 3.1542 | 5.0132 |

Under cheap abstention, always-abstain costs 2.000 and is the best fixed
comparator; the conformal entry is 1.947. The fixed-tuned column in the table
is a set-valued adjudicator, not the minimum over all fixed policies.

Coverage and fallbacks:

- base: coverage .9163, mean set size 2.203, selected LAC alpha .20,
  fallbacks 0/5.
- cheap abstain: coverage 1.000, size 3.000, LAC alpha .05, fallbacks 0/5.
- narrow width: coverage 1.000, size 3.000, LAC alpha .05, fallbacks 0/5.
- wide width: coverage .8881, size 2.196, LAC/APS alpha .05/.20, fallbacks
  0/5.

Paired differences (conformal minus baseline; negative favors conformal;
95% claim bootstrap):

| config | vs planner tuned | vs fixed tuned | vs static set |
|---|---:|---:|---:|
| base | -0.0176 [-0.0441, 0.0000] | -0.0211 [-0.0758, +0.0300] | -0.0529 [-0.1322, +0.0176] |
| cheap abstain | -0.5639 [-0.8458, -0.3172] | -1.1753 [-1.4608, -0.9093] | -1.2070 [-1.4890, -0.9427] |
| narrow width | -0.2643 [-0.5419, -0.0176] | -0.2379 [-0.5154, +0.0067] | -0.2379 [-0.5154, +0.0067] |
| wide width | -0.0952 [-0.1586, -0.0335] | -1.0863 [-1.5665, -0.6070] | -1.1921 [-1.6846, -0.7004] |

The base difference against the tuned planner is small and its interval
touches zero; the base difference against the tuned fixed adjudicator includes
zero. The material gains are under cheap abstention, the narrow width penalty,
and the wide width penalty. There is no configuration where the conformal
layer is materially worse than the best tuned baseline.

Against an oracle choice among the fixed policies (always-answer, tuned fixed
adjudicator, and always-abstain with its configuration-specific cost), the
conformal layer has mean regret -0.0974 and maximum regret -0.0106. The tuned
planner has mean +0.0075 and maximum +0.5110 (cheap abstention, where
always-abstain is 2.000); the tuned fixed adjudicator +0.1656/+1.1225; the
static set +0.2015/+1.1542; the point label +2.0143/+3.7225. The conformal
layer is the only policy with a negative maximum regret in this comparison.

### 12.2 Ablation: expected versus realized threshold

On DeepSeek, all-label five-split means:

| objective | base | cheap | narrow | wide |
|---|---:|---:|---:|---:|
| expected | 3.0784 | 1.9560 | 1.0440 | 4.5143 |
| realized | 3.0110 | 1.9560 | 1.0440 | 4.5143 |

On the logistic backbone (verified, five fold seeds, 31 thresholds), the
expected objective is more stable at the base loss; the realized objective
helps the DeepSeek backbone more. The exact five-seed means are:

| objective | base | fixed tuned | under-averse | wide width | coverage (base) |
|---|---:|---:|---:|---:|---:|
| expected | 3.3427 | 3.2934 | 3.1850 | 4.7357 | .8247 |
| realized | 3.3956 | 3.2934 | 3.2115 | 4.7401 | .8352 |

The logistic calibration set is small (company-disjoint 60/40 within each
training pool), so realized threshold selection overfits more there. This is a
backbone/calibration-size dependence, not hidden. The
`paper/results/conformal_v3_verified_seed*/` artifacts are the
expected-objective logistic runs; `paper/results/conformal_v3_realized_seed*/`
are the realized-objective runs.

### 12.3 What is not claimed

- The conformal layer does not beat the tuned planner or fixed adjudicator at
  the base loss by a statistically clear margin. It ties them, and it wins in
  three loss regimes.
- The point reader is far worse (base 5.013). The result is about the decision
  layer and the set-valued action, not about a stronger point classifier.
- The DeepSeek test outcomes may be partly exposed in pretraining. The
  masked/point runs are the contamination check; the decision-layer result
  inherits that caveat.
- No distribution-free coverage guarantee under the year-wise company shift.
- The real benchmark evaluates one cutoff per claim; the wait/acquire actions
  are tested only in the controlled diagnostic.

## 13. What would falsify the v3.1 claim

- Test coverage far below the selected nominal level with no fallback flag.
- The conformal layer losing to the tuned planner or fixed adjudicator on most
  loss configurations.
- The gain disappearing when the backbone is changed (DeepSeek vs logistic).
- The threshold/method selection failing on validation and only working when
  chosen on test.
- The realized objective winning only when the threshold is chosen on test.

## 14. Reproduce

```text
PYTHONPATH=src python -m pytest -q tests/test_conformal.py

PYTHONPATH="src;scripts" python scripts/run_deepseek_conformal_set_eval.py \
  --verified-only --out paper/results/deepseek_conformal_realized_verified

PYTHONPATH="src;scripts" python scripts/run_deepseek_conformal_set_eval.py \
  --out paper/results/deepseek_conformal_realized_all

PYTHONPATH="src;scripts" python scripts/analyze_conformal_regret.py \
  --summary paper/results/deepseek_conformal_realized_verified/summary.json

for s in 1 2 3 4 5; do
  PYTHONPATH="src;scripts" python scripts/run_conformal_decision_v3.py \
    --seeds $s --thresholds 31 --verified-only \
    --configs base,symmetric,over_averse,under_averse,cheap_abstain,expensive_abstain,narrow_penalty,wide_penalty,alpha_05,alpha_20 \
    --out paper/results/conformal_v3_realized_seed$s
 done
```

The expected-objective ablation uses the same commands with
`--threshold-objective expected`; the saved
`paper/results/conformal_v3_verified_seed*/` directories are the runs made
before the default changed and are labeled as the expected ablation.
`--threshold-objective both` selects the objective on validation, which is
reported as a sensitivity, not the default.
