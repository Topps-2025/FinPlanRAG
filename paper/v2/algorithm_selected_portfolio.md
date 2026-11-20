# Selected version: loss-aware policy portfolio

**Status:** selected algorithm of record, 2026-11-03. Matches
`src/finplan_rag/decision/portfolio.py` and
`scripts/analyze_portfolio.py`. The planner and conformal documents describe
component/alternative policies, not the selected primary method.

## 1. Algorithm

The interface has three candidate output policies at belief \(b_t\):

1. **fixed**: adjudicate the credible set
   \(\mathcal C_\alpha(b_t)=\{R:b_t(R)\ge 1-\alpha\}\);
2. **abstain**: pay the abstention cost;
3. **planner**: choose answer / adjudicate / abstain by expected loss.

The fixed and planner candidates tune \(\alpha\in\{.05,.10,.20\}\) on a
company-disjoint validation split. A conservative selector starts from the
fixed candidate and switches to abstain or planner only when that candidate
improves validation loss by at least \(m=.15\). The selected policy is applied
to the test fold once. Test outcomes are never used for selection. The margin
is a design/simplicity choice; the paper reports the sensitivity.

## 2. Selected-policy pattern

On the ten reported loss configurations:

| loss | selected policy |
|---|---|
| base, symmetric, over-averse, under-averse, expensive abstention, narrow width, alpha .05, alpha .20 | fixed |
| cheap abstention | abstain |
| wide width | planner |

The rule sees the loss configuration and validation losses, not test labels.

## 3. Result (227 XBRL-verified, RF 200 trees, ten seeds)

Artifacts: `paper/results/portfolio_10seed.json`, built from
`paper/results/rf_all_s{1..10}/`.

| config | selected | portfolio | best fixed | regret |
|---|---|---:|---:|---:|
| base | fixed | 2.9322 | 2.9322 | 0.0000 |
| symmetric | fixed | 2.9322 | 2.9322 | 0.0000 |
| over-averse | fixed | 2.9322 | 2.9322 | 0.0000 |
| under-averse | fixed | 2.9322 | 2.9322 | 0.0000 |
| cheap abstain | abstain | 2.0000 | 2.0000 | 0.0000 |
| expensive abstain | fixed | 2.9322 | 2.9322 | 0.0000 |
| narrow width | fixed | 0.9630 | 0.9630 | 0.0000 |
| wide width | planner | 4.3405 | 4.7101 | -0.3696 |
| alpha .05 | fixed | 2.9322 | 2.9322 | 0.0000 |
| alpha .20 | fixed | 2.9322 | 2.9322 | 0.0000 |

Mean regret **-0.0370** (95% claim-level bootstrap CI **[-0.0643, -0.0100]**),
maximum regret **0.0000**. The wide-width regret is -0.3696 (95% CI
[-0.6432, -0.0996]). Intervals are claim-level bootstrap intervals (10,000
resamples, seed 20261103) computed by `analyze_portfolio.py`. The portfolio is
at or below the oracle best fixed policy in every configuration and beats it
under wide width. At base the fixed
adjudicator is the oracle optimum; the portfolio matches it. This is an
oracle-level regret result, not a base-superiority claim.

### Confirmatory seeds 11-20

Artifact: `paper/results/portfolio_confirm.json`.

- base: fixed, 2.8855 vs best fixed 2.8855, regret 0. The planner alone is
  2.984, +0.099 (95% CI [0.001, 0.211]) above fixed; the pre-specified .2
  non-inferiority secondary is not met.
- cheap abstention: abstain, 2.000 vs 2.000, regret 0;
- narrow width: fixed, 0.998 vs 0.998, regret 0;
- wide width: planner, 4.3352 vs 4.6872 oracle fixed, regret -0.3520 (95% CI
  [-0.6441, -0.0647]); the pre-specified primary comparison against the
  width-tuned fixed adjudicator is 4.3352 vs 5.2000, difference -0.8648
  (95% CI [-1.2256, -0.4903]);
- four-cell mean regret -0.0880 (95% CI [-0.1610, -0.0162]), max 0.0000.
  This is not the ten-cell mean.

The wide regime was identified on seeds 1-10 and the protocol was written
before seeds 11-20. This is a seed-level replication on the same claims, not a
new claim sample.

## 4. Backbone and component evidence

- Random forest: 200 trees, `min_samples_leaf=3`, balanced subsampling,
  point-in-time structured features; no text or LLM.
- Planner alone: base 3.011 vs fixed 2.932; it is worse at base. It is used
  only in the wide-width regime by the portfolio.
- Conformal alternative: DeepSeek named base 3.101, cheap 1.947, narrow 1.053,
  wide 4.447; masked base 3.119 vs fixed 3.108. Sensitive to masking; not
  clean primary evidence.
- Logistic backbone: higher point balanced accuracy but worse decision loss;
  does not reproduce the RF result.

## 4b. Named baselines and metrics

The full 11-metric suite (accuracy, balanced accuracy, macro F1, MCC, kappa,
AUROC, AP, ECE, Brier, NLL, selective@50, AURC) is in
`paper/v2/named_baselines_result.md` and
`paper/results/named_baselines/metrics_*.json`. Six named baselines are
adaptations under the same DeepSeek backbone: FinDVer (.551 acc/.461 bal/.674
AUROC), Fin-Fact (.537/.446/.650), Self-RAG (.546/.451/.670), CRAG
(.529/.423/.643), FLARE (.511/.417/.634), and Adaptive-RAG (.493/.406/.584).
No point model dominates: FinDVer has the best accuracy, Self-RAG the best
selective@50/AURC (.684/.333), and FinPlan the best balanced accuracy .561,
MCC .273, AUROC .716, and lowest ECE .048. McNemar does not reject equal
accuracy against FinPlan (p=.17--1.00). The decision portfolio is therefore
selected on regret, not on a single point metric. A 737-claim rolling-origin
backbone check (2015--2025) gives RF .537 accuracy/.503 balanced/.032 ECE and
logistic .487/.535/.092.

## 5. Code map

| Component | File |
|---|---|
| Portfolio selector | `src/finplan_rag/decision/portfolio.py` |
| Portfolio evaluation / validation selection | `scripts/analyze_portfolio.py` |
| RF backbone and component policies | `scripts/run_backbone_conformal_diagnostics.py` |
| Planner / credible set | `scripts/run_real_decision_eval.py`, `src/finplan_rag/decision/loss.py` |
| Conformal alternative | `src/finplan_rag/decision/conformal.py` |
| Tests | `tests/test_portfolio.py`, `tests/test_decision.py`, `tests/test_conformal.py` |

## 6. Reproduce

```text
for s in 1 2 3 4 5 6 7 8 9 10; do
  PYTHONPATH="src;scripts" python scripts/run_backbone_conformal_diagnostics.py \
    --seeds $s --models rf --rf-trees 200 \
    --configs base,symmetric,over_averse,under_averse,cheap_abstain,expensive_abstain,narrow_penalty,wide_penalty,alpha_05,alpha_20 \
    --thresholds 31 --out paper/results/rf_all_s$s
done

PYTHONPATH="src;scripts" python scripts/analyze_portfolio.py \
  --validation-root paper/results/rf_all_s --validation-seeds 6,7,8,9,10 \
  --test-root paper/results/rf_all_s --test-seeds 1,2,3,4,5,6,7,8,9,10 \
  --margin 0.15 --out paper/results/portfolio_10seed.json
```

The confirmatory seeds use `rf_confirm_s11..20`, the same command with
validation/test seeds 11--20, and `portfolio_confirm.json`.

## 7. Falsification

- The portfolio's regret positive in a configuration on a fresh seed set.
- The selector's validation choice disagreeing with the portfolio rule and
  losing to best fixed.
- The wide-width confirmation failing on seeds 11--20.
- The result disappearing with a calibrated non-LLM backbone.

## 8. What is not claimed

- A base-loss win. The fixed adjudicator is the oracle optimum at base; the
  portfolio matches it. No method can beat the oracle by construction.
- A uniform single-policy win. The portfolio is a selector over policies.
- Clean LLM evidence. The DeepSeek result is a secondary sensitivity.
