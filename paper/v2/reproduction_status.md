# Reproduction status (2026-11-03)

## Test suite

```text
PYTHONPATH="src;scripts" python -m pytest -q
33 passed
```

This covers the decision loss, credible sets, planner stopping/waiting,
pilot data schema, preliminary-signal determinism, budgeted accuracy, metrics,
and the real-benchmark categorical-column regression test.

## Selected version: loss-aware portfolio

Algorithm: `paper/v2/algorithm_selected_portfolio.md`. Candidate policies:
fixed adjudicator, abstention, planner; validation selection with a .15
switching margin; RF 200-tree backbone. Ten-seed portfolio (`portfolio_10seed.json`):
mean regret -0.0370 (95% CI [-0.0643, -0.0100]), max 0.0000; base is an exact
match to the best fixed policy, wide width 4.3405 vs 4.7101 (regret -0.3696,
95% CI [-0.6432, -0.0996]). Confirmatory seeds 11-20 (`portfolio_confirm.json`, four cells only):
base exact match (the planner alone is +0.099 [0.001, 0.211] above fixed, so
the pre-specified .2 non-inferiority secondary fails), wide -0.3520 (95% CI
[-0.6441, -0.0647]) against the oracle fixed policy, four-cell mean -0.0880
(95% CI [-0.1610, -0.0162]), max 0.0000. The pre-specified primary comparison
against the width-tuned fixed adjudicator is -0.8648 (95% CI [-1.2256,
-0.4903]). CIs are claim-level bootstrap intervals (10,000 resamples, seed
20261103) stored in both JSON files.

```text
for s in 1 2 3 4 5 6 7 8 9 10; do
  PYTHONPATH="src;scripts" python scripts/run_backbone_conformal_diagnostics.py --seeds $s --models rf --rf-trees 200 --configs base,symmetric,over_averse,under_averse,cheap_abstain,expensive_abstain,narrow_penalty,wide_penalty,alpha_05,alpha_20 --thresholds 31 --out paper/results/rf_all_s$s
done
PYTHONPATH="src;scripts" python scripts/analyze_portfolio.py --validation-root paper/results/rf_all_s --validation-seeds 6,7,8,9,10 --test-root paper/results/rf_all_s --test-seeds 1,2,3,4,5,6,7,8,9,10 --margin 0.15 --out paper/results/portfolio_10seed.json
PYTHONPATH="src;scripts" python scripts/analyze_portfolio.py --validation-root paper/results/rf_confirm_s --validation-seeds 11,12,13,14,15,16,17,18,19,20 --test-root paper/results/rf_confirm_s --test-seeds 11,12,13,14,15,16,17,18,19,20 --margin 0.15 --out paper/results/portfolio_confirm.json
```

## Named baselines, metric suite, rolling benchmark (2026-11-04)

- Named baselines: `scripts/run_named_baselines.py`; six adaptations under
  the same DeepSeek backbone (FinDVer, Fin-Fact, Self-RAG, CRAG, FLARE,
  Adaptive-RAG); `paper/results/named_baselines/`.
- Full metrics: `scripts/compute_metrics_suite.py`; results in
  `paper/results/named_baselines/metrics_*.json`; summary in
  `paper/v2/named_baselines_result.md` and `metrics_suite_result.md`.
- Model comparison: `scripts/compare_models.py`;
  `paper/results/named_baselines/model_comparison_logistic.json`.
- Rolling-origin backbone check: `scripts/run_rolling_benchmark.py`; 737
  claims 2015-2025, 5 seeds; `paper/results/rolling_benchmark/`.
- 227 verified: FinPlan logistic .493/.561/ECE .048/Sel@50 .526; FinDVer
  .551/.461/.092/.675; Fin-Fact .537/.446/.075/.675; Self-RAG
  .546/.451/.102/.684; CRAG .529/.423/.097/.658; FLARE .511/.417/.130/.596;
  Adaptive-RAG .493/.406/.147/.579. No point model dominates.
- Rolling: RF .537 acc/.503 bal/.032 ECE; logistic .487/.535/.092.

## Conformal decision layer (v3.1)

Conformal alternative: `paper/v2/algorithm_v3_conformal.md`. Default
`ConformalConfig()` uses the realized threshold objective and 101 thresholds;
the expected objective is an ablation.

Key verified-label result (`paper/results/deepseek_conformal_realized_verified/`,
227 claims, five calibration split seeds, DeepSeek backbone):

| config | conformal | planner tuned | fixed tuned | static set |
|---|---:|---:|---:|---:|
| base | 3.1013 | 3.1189 | 3.1225 | 3.1542 |
| cheap abstain | 1.9471 | 2.5110 | 3.1225 | 3.1542 |
| narrow width | 1.0529 | 1.3172 | 1.2907 | 1.2907 |
| wide width | 4.4467 | 4.5419 | 5.5330 | 5.6388 |

Base coverage .9163, mean set size 2.203, zero fallback folds. Across the ten
configurations, conformal mean regret is -0.0974 and maximum regret -0.0106
against an oracle choice among fixed policies (always-abstain is 2.000 under
cheap abstention).

Commands:

```text
PYTHONPATH="src;scripts" python scripts/run_deepseek_conformal_set_eval.py \
  --verified-only --out paper/results/deepseek_conformal_realized_verified

PYTHONPATH="src;scripts" python scripts/run_deepseek_conformal_set_eval.py \
  --out paper/results/deepseek_conformal_realized_all

for s in 1 2 3 4 5; do
  PYTHONPATH="src;scripts" python scripts/run_conformal_decision_v3.py \
    --seeds $s --thresholds 31 --verified-only \
    --configs base,symmetric,over_averse,under_averse,cheap_abstain,expensive_abstain,narrow_penalty,wide_penalty,alpha_05,alpha_20 \
    --out paper/results/conformal_v3_realized_seed$s
done
```

Logistic sensitivity: the expected objective on raw out-of-fold probabilities
has base 3.343 vs. 3.293 for the tuned fixed adjudicator; it wins under
under-averse (3.185 vs. 3.293) and wide-width (4.736 vs. 5.508) losses but not
at base. The realized objective is worse at base on this small calibration set
and is reported as an ablation.

## Existing benchmark and decision artifacts

The saved artifacts were regenerated with the current code. Key values match
the documents:

| Artifact | Check |
|---|---|
| `paper/results/company_disjoint_benchmark/summary.json` | n=273; ensemble macro F1 .5042, balanced accuracy .5656 |
| `paper/results/company_disjoint_benchmark_verified/summary.json` | n=227; macro F1 .4884, balanced accuracy .5635 |
| `paper/results/real_decision_eval_verified_calibrated/summary.json` | planner mean regret .0521, max .3833; matched all-cell mean .8208, max 6.0335 |
| `paper/results/robust_planner_prototype.json` | minimax-regret mean .296, max 2.13 |

Commands:

```text
PYTHONPATH=src python scripts/run_company_disjoint_benchmark.py
PYTHONPATH=src python scripts/run_company_disjoint_benchmark.py --verified-only --out paper/results/company_disjoint_benchmark_verified
PYTHONPATH=src python scripts/run_real_decision_eval.py --verified-only --calibrate --out paper/results/real_decision_eval_verified_calibrated
PYTHONPATH="src;scripts" python scripts/prototype_robust_planner.py
```

## Backbone controls (not named baselines)

These are diagnostics for the FinPlan evidence model; the paper's baseline
table uses the six named methods in `named_baselines_result.md`.

```text
PYTHONPATH="src;scripts" python scripts/run_classical_baselines.py --seed 1
PYTHONPATH="src;scripts" python scripts/run_classical_baselines.py --seed 2 --models logistic,tfidf,tfidf_numeric
... seeds 3--5
```

Five-seed means on 273 test claims:

| model | accuracy | balanced | macro F1 |
|---|---:|---:|---:|
| logistic | .514 | .560 | .501 |
| TF-IDF | .512 | .481 | .475 |
| TF-IDF + numeric | .540 | .533 | .509 |
| HistGB (seed 1) | .458 | .400 | .398 |
| Random forest (seed 1) | .528 | .479 | .478 |

The text + numeric model has higher plain accuracy and macro F1 than the
structured logistic model, but lower balanced accuracy. The interface result
does not depend on a single backbone.

## DeepSeek API check

The environment provides `ANTHROPIC_AUTH_TOKEN` and
`ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic`, model
`deepseek-flash[1m]`. The key is not written to any artifact.

A minimal request to `POST /v1/messages` returns 200 with
`thinking: {"type": "disabled"}`. The local proxy configuration breaks
`requests`; the client sets `trust_env=False`.

## DeepSeek as backbone, not a separate method

```text
PYTHONPATH="src;scripts" python scripts/run_backbone_policy_eval.py
```

On 227 verified claims, with the same planner:

| backbone / policy | base loss | max over 10 configs | wins vs other planner |
|---|---:|---:|---:|
| DeepSeek + planner | 3.101 | 4.542 | 8/10 |
| Logistic + planner | 3.982 | 5.110 | 2/10 |
| DeepSeek static set | 3.154 | 5.639 | -- |
| Logistic static set | 3.877 | 7.718 | -- |

DeepSeek supplies label, confidence, and credible set; the planner is the
method. The DeepSeek backbone loses under cheap abstention and the narrow width
penalty, so the result is not a uniform backbone win.

## DeepSeek runs

```text
PYTHONPATH="src;scripts" python scripts/run_deepseek_baselines.py \
  --workers 8 --max-tokens 600 --prompt-version v3 \
  --variants point,selective --out paper/results/deepseek_full_point_selective

PYTHONPATH="src;scripts" python scripts/run_deepseek_baselines.py \
  --workers 8 --max-tokens 600 --prompt-version v3 \
  --variants verification --out paper/results/deepseek_full_verification

PYTHONPATH="src;scripts" python scripts/run_deepseek_baselines.py \
  --workers 8 --max-tokens 600 --prompt-version v3 \
  --variants masked --out paper/results/deepseek_full_masked

PYTHONPATH="src;scripts" python scripts/run_deepseek_baselines.py \
  --workers 8 --max-tokens 600 --prompt-version v3 \
  --variants evidence --out paper/results/deepseek_full_evidence

PYTHONPATH="src;scripts" python scripts/run_deepseek_baselines.py \
  --workers 8 --max-tokens 600 --prompt-version v3 \
  --self-consistency 5 --planner --out paper/results/deepseek_full_selfconsistency

PYTHONPATH="src;scripts" python scripts/analyze_deepseek_adjudication.py
```

Results are in `paper/v2/deepseek_baseline_result.md` and
`paper/results/deepseek_adjudication/report.md`.

## Known gaps

- The LLM point runs are contaminated by possible pretraining exposure to
  2022--2025 outcomes; the masked full run is not yet done.
- B1c/B1d boosting/forest and B1e TF-IDF baselines are not run.
- B5a pre-hoc router is still controlled-only on the real benchmark.
- B6 recent financial RAG systems are not compared on a shared subset.
- DeepSeek credible-set numbers use one prompt; prompt sensitivity is not
  measured.
