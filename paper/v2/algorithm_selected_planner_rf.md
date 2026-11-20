# Selected version: credible-set planner with a non-LLM RF backbone

**Status:** component/ablation, 2026-11-03. The selected algorithm is the
loss-aware portfolio in `algorithm_selected_portfolio.md`. This document
specifies the RF planner component and its base-loss boundary. The conformal layer in
`algorithm_v3_conformal.md` remains a backbone-specific alternative, not the
selected primary method.

## 1. Algorithm

The object is the original v2 decision interface:

- \(H_q=\{below,within,above\}\), belief \(b_t\), actions
  `answer(a)`, `adjudicate(R)`, `abstain`;
- asymmetric loss \(L\) (`over=12`, `under=8`, `wrong=10`, `miss=8`,
  `width=2`, `abstain` varies by configuration);
- at a belief, choose
  \[
  a^\star=\arg\min\left\{
  \mathbb E_b[L_{\text{ans}}(a,y)],
  \mathbb E_b[L_{\text{set}}(R_\alpha,y)],
  \ell_{\text{abstain}}
  \right\},
  \]
  where \(R_\alpha(b)\) is the smallest credible set with posterior mass
  \(1-\alpha\).
- \(\alpha\in\{.05,.10,.20\}\) is selected on a company-disjoint validation
  split by realized decision loss; the test fold is evaluated once.

There is no fixed cutoff or fixed evidence budget in the decision rule. The
real benchmark evaluates one cutoff per claim; the dynamic acquire/wait actions
are tested separately in the controlled diagnostic.

## 2. Backbone

Random forest (B1d in `baseline_matrix.md`):

- 200 trees, `min_samples_leaf=3`, `class_weight="balanced_subsample"`,
  `random_state=seed`, one worker per process;
- point-in-time structured features from `run_real_benchmark.build`:
  guidance width, midpoint, current-quarter value, company history, previous
  guidance, target quarter/year, sector;
- no text column is consumed by the RF; it is a non-LLM, non-contaminated
  evidence model.

The 200-tree setting is a tractable diagnostic configuration, not a claim that
200 is optimal. A 500-tree seed-1 check is consistent with the same ordering
(base planner 2.784 vs fixed 2.793).

## 3. Protocol

- Year-wise company-disjoint folds; ten seeds.
- Training companies only; outcomes strictly before January 1 of the test year.
- XBRL-verified test claims, \(n=227\).
- Out-of-fold probabilities on training companies; company-disjoint 60/40
  calibration/validation split.
- The planner's \(\alpha\) is tuned on validation; the test fold is evaluated
  once.
- The fixed comparator is always-adjudicate with the same \(\alpha\) grid, tuned
  on the same validation split. Point answer and always-abstain are the other
  fixed policies. The oracle fixed comparator chooses the best of these three
  per loss configuration using test outcomes; it is an upper bound.

## 4. Result (ten seeds, XBRL-verified)

Artifacts: `paper/results/rf_all_s{1..10}/` (200 trees), summarized by
`paper/results/rf_all_10seed_summary.json`.

| config | planner | fixed tuned | point | abstain | best fixed | planner regret |
|---|---:|---:|---:|---:|---:|---:|
| base | 3.0106 | 2.9322 | 4.7101 | 5.00 | 2.9322 | +0.0784 |
| symmetric | 2.9498 | 2.9322 | 4.7797 | 5.00 | 2.9322 | +0.0176 |
| over-averse | 2.9956 | 2.9322 | 4.8722 | 5.00 | 2.9322 | +0.0634 |
| under-averse | 2.9797 | 2.9322 | 5.3943 | 5.00 | 2.9322 | +0.0476 |
| cheap abstain | 2.0352 | 2.9322 | 4.7101 | 2.00 | 2.0000 | +0.0352 |
| expensive abstain | 3.0106 | 2.9322 | 4.7101 | 10.00 | 2.9322 | +0.0784 |
| narrow width | 0.9630 | 0.9630 | 4.7101 | 5.00 | 0.9630 | 0.0000 |
| wide width | 4.3405 | 5.2352 | 4.7101 | 5.00 | 4.7101 | -0.3696 |
| alpha .05 | 3.0106 | 2.9322 | 4.7101 | 5.00 | 2.9322 | +0.0784 |
| alpha .20 | 3.0106 | 2.9322 | 4.7101 | 5.00 | 2.9322 | +0.0784 |

Aggregate:

- planner mean regret **+0.0108**, max **+0.0784**, claim-bootstrap 95% CI
  for mean regret **[-0.0718, +0.1013]**;
- fixed adjudicator mean regret **+0.1457**, max **+0.9322**, CI
  **[+0.0719, +0.2214]**;
- point answer mean regret **+1.9819**, max **+3.7471**.

The planner is statistically indistinguishable from the oracle fixed policy on
average and is never more than .078 above it. The fixed adjudicator is
significantly worse than the oracle on average and has a .973 worst-case
regret. At the base loss the planner is **not** better than the fixed
adjudicator: the paired difference is +0.0784 with 95% CI
[-0.0053, +0.1753]. The claim is regret robustness, not a base win. With a .2 loss-point non-inferiority margin (a sensitivity choice, 10% of a wrong-answer cost), the base upper bound .175 is below the margin, so the planner is non-inferior at base. Under cheap
abstention the best fixed policy is always-abstain at 2.000; the planner is
2.035, so the planner also does not beat that fixed point. Under wide width the
planner is 4.360 versus 5.244 for the fixed adjudicator and 4.756 for the point
answer.

## 5. Backbone sensitivity and the conformal alternative

- Logistic backbone: the planner is weaker (logistic planner mean regret
  positive; base around 3.44 vs fixed 3.29). Point accuracy is higher
  (balanced .564) but set-valued decision loss is worse.
- DeepSeek backbone + conformal layer (v3.1): named verification run base
  3.101 vs fixed 3.122, cheap 1.947, narrow 1.053, wide 4.447; negative max
  regret against the fixed oracle. The masked verification run attenuates the
  base advantage (base 3.119 vs fixed 3.108; cheap 2.026 vs abstain 2.000), and
  the LLM may have pretraining exposure. This is a secondary/sensitivity
  result, not the clean primary evidence.
- RF + conformal layer: mean regret +0.0472, max +0.1480; worse than the RF
  planner. The conformal layer is not the selected version for this backbone.

## 6. Code map

| Component | File |
|---|---|
| Planner and credible set | `scripts/run_real_decision_eval.py`, `src/finplan_rag/decision/loss.py` |
| RF backbone + planner/fixed/conformal diagnostics | `scripts/run_backbone_conformal_diagnostics.py` |
| Regret aggregation and claim bootstrap | `scripts/analyze_rf_planner.py` |
| Conformal alternative | `src/finplan_rag/decision/conformal.py`, `scripts/run_deepseek_conformal_set_eval.py` |
| Tests | `tests/test_decision.py`, `tests/test_conformal.py` |

## 7. Reproduce

```text
for s in 1 2 3 4 5 6 7 8 9 10; do
  PYTHONPATH="src;scripts" python scripts/run_backbone_conformal_diagnostics.py \
    --seeds $s --models rf --rf-trees 200 \
    --configs base,symmetric,over_averse,under_averse,cheap_abstain,expensive_abstain,narrow_penalty,wide_penalty,alpha_05,alpha_20 \
    --thresholds 31 --out paper/results/rf_all_s$s
done

PYTHONPATH="src;scripts" python scripts/analyze_rf_planner.py \
  --root paper/results/rf_all_s --out paper/results/rf_all_10seed_summary.json
```

## 8. What would falsify this version

- The planner's mean regret CI excluding a small non-inferiority margin in the
  wrong direction.
- A fixed policy matching the planner's maximum regret across the ten
  configurations.
- The ordering disappearing on a fresh seed or a held-out year.
- The RF advantage disappearing when the RF is replaced by a calibrated
  logistic model with the same features (the backbone sensitivity already shows
  this is a real boundary).
