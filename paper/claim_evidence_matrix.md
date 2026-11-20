# Claim--evidence matrix

This matrix covers the current paper, which focuses only on the forward-looking
branch.  Router and determinate-branch results are not part of this paper's
claims and should not be cited from this file.  `measured` means produced by
the released code on the released data; `bounded` means the number is measured
but the scope is narrower than a general superiority claim; `design` means an
architectural or protocol claim without a performance estimate.

## Main claims

| Claim | Evidence | Status | Boundary |
|---|---|---|---|
| A forward-looking claim can be represented as a contract with company, target period, outcome space, cutoff and evidence budget | `scripts/anchor_engine.py` `ClaimContract`; `configs/anchor_cases/zhongji_2026.yaml`; `configs/anchor_cases/xinyisheng_2026.yaml` | design | The contract is a modeling choice, not a empirical result |
| The anchor state has three levels and five qualification types, and every accepted anchor carries a source, quote, date and qualification | `scripts/anchor_engine.py` `qualify_anchors`; audit files under `paper/results/anchor_cli/*/audit.json`; `docs/anchor_cli.md` | design | Qualification rules are deterministic checks, not a learned classifier |
| Five-type qualification improves the anchor-only decision over accepting every parsed anchor | `scripts/run_qualification_ablation.py`; `paper/results/qualification_ablation/REPORT.md`: 227 verified claims, full 3.242 vs accept-all 3.445; Δ +0.203, CI [-0.132, +0.537]; acceptance 26.8% vs 93.8% | measured, bounded | Direction favors qualification, but the 95% interval includes zero on 227 claims. Do not claim significance without the 250+ A-share/expanded run. |
| Text-grounded anchors improve the standalone LLM branch over the old structured prompt | `paper/results/evolution_v1/REPORT.md`: accuracy .445 to .502, balanced .307 to .397, macro-F1 .242 to .364 | measured | This is a branch-level comparison; it does not show that the anchor posterior beats a strong RF |
| Numeric anchors can be combined by deterministic bridges rather than an LLM point guess | `scripts/forecast_bridge.py`; `scripts/build_ashare_backtest.py`; `scripts/run_ashare_backtest.py`; expanded A-share data `paper/data/ashare_backtest_large/claims.json` (470 claims, 40 companies, 2023-2025); results `paper/results/ashare_backtest_large/`; case audits `20261120-014606-xinyisheng_2026-2079` and `20261120-013702-zhongji_2026-c5e7` | measured, mixed | DeepSeek full .783 accuracy / 2.179 loss versus run-rate .777 / 2.270 and no-bridge .638 / 2.421. GLM full equals no-bridge (.549 / 3.332) because the bridge did not fire. Qwen is evaluated on a 250-claim stratified subsample. Two prospective cases are workflow evidence only. |
| Full-action loss-aware readout lowers decision loss against the restricted credible-set gate | `paper/results/cloud_2026-11-20/readout_consistency.json`: fusion 3.656 to 2.907 (-0.749 [-1.075, -0.405]), 3.524 to 2.943 (-0.581 [-0.899, -0.264]), 3.621 to 2.925 (-0.696 [-0.987, -0.405]); RF 3.454 to 2.987 (-0.467 [-0.767, -0.159]) | measured | This is the strongest result; it is a decision-rule improvement on the same posterior, not a new classifier |
| Removing the three-layer anchor structure lowers performance substantially | `paper/results/cloud_2026-11-20/no_anchor_ablation.json`; `scripts/run_no_anchor_ablation.py`: point-in-time LLM scores + temperature + full-action readout, no anchors, losses 3.286/3.260/3.824 versus FinPlanRAG 2.907/2.943/2.925 | measured | This isolates the anchor structure from the decision readout; it is the architecture-level ablation requested for the core-structure claim |
| The fused system has the best point estimate against every baseline on all three backbones | `paper/results/cloud_2026-11-20/metrics_table.json`; `paper/results/cloud_2026-11-20/deepseek/evolution/evolved_metrics.json` | measured, bounded | Fusion has the highest accuracy, balanced accuracy, macro-F1 and lowest decision loss on all three backbones. Against the strongest baseline, DeepSeek and GLM intervals include zero; only Qwen excludes zero. |
| Temporal reliability evolution gives a small, consistent loss reduction | `paper/results/evolution_v1/REPORT.md`: new anchors static 2.908 to evolved 2.855; old anchors 2.899 to 2.793 | measured, bounded | The gain is small and not the main driver; do not claim it as a significant accuracy improvement |
| The final system is lower than the RF in point estimate but not significantly different on overall decision loss | Cloud results: DeepSeek 2.907 vs 2.987 (-0.079 [-0.300, 0.141]); GLM 2.943 vs 2.987 (-0.044 [-0.203, 0.115]); Qwen 2.925 vs 2.987 (-0.062 [-0.282, 0.159]) | measured | The anchor layer's value is the auditable semantic state plus the within-range gain, not a uniform point-prediction gain |
| On the within-range stratum, fusion significantly lowers loss relative to RF on all three backbones | `paper/results/cloud_2026-11-20/within_range_all_backbones.json`: 82 claims; DeepSeek 2.780 vs 3.146 (-0.366 [-0.732, -0.049], 5/5 seeds), GLM 2.902 vs 3.146 (-0.244 [-0.537, -0.024], 4/5), Qwen 2.732 vs 3.146 (-0.415 [-0.732, -0.170], 5/5) | measured, stratified | The realized class is unknown at prediction time; this is an error-stratified result, not a routing rule |
| Validation-gated type-library extension improves DeepSeek in point estimate | `paper/results/self_evolve/deepseek_beam/report.json`: validation 3.943 to 3.792 accepted; test accuracy .555 to .577, loss 2.934 to 2.907, paired difference -0.026 [-0.159, 0.097] | measured, bounded | Small directional gain; prompt-only and combined candidates were rejected; single-round and DeepSeek-only |
| The calibrated RF is a strong, reproducible decision head | `paper/results/rf_calibrated_repro/per_claim.jsonl` from `scripts/make_calibrated_rf_probs.py`; five seeds, company-disjoint year-wise folds, temperature scaling; test accuracy .533, balanced .491, macro .486, loss 2.881 | measured | The RF is a diagnostic and baseline, not the paper's novel contribution |
| The LLM anchor posterior alone is not a substitute for the calibrated posterior | `paper/results/evolution_v1/REPORT.md`: AnchorEvolve-LLM balanced accuracy .307 (DeepSeek), .329 (GLM); macro-F1 .242/.295 | measured | The LLM over-predicts the above bucket; this motivates qualification and fusion |
| Validation selects a small anchor weight, and the anchor layer does not replace the RF as a point-metric head | `paper/results/cloud_2026-11-20/*/evolution/temporal_fusion_new_cloud.json`: selected weights .2, .1, .1; fusion .577/.526/.530, .551/.504/.503, .542/.501/.499 versus RF .533/.491/.486 | measured | Fusion is above the RF on all point metrics, but the overall loss difference is not significant; the anchor layer's value is auditability plus the within-range gain |
| The benchmark labels are XBRL-verified or XBRL-overridden | `paper/data/dataset_v4.json`; 227 verified test claims; 46 text-only claims excluded from the main table and used as a sensitivity check | measured | Not a human-annotated gold standard; label noise is a stated limitation |
| The point-in-time protocol excludes future outcomes and test companies | `scripts/run_company_disjoint_benchmark.py`; `experiments/evolution/temporal_fusion.py`; outcome date before January 1 of the test year; other company folds only | measured | Assumes filing and outcome dates are correct |
| Public benchmarks are used only for component-level checks, not as main outcome evidence | `paper/zh/sufe_master_vague/chapters_exp.tex` Section 5; `docs/benchmark_plan.md` | design | FinDVer and Fin-Fact labels are support/refute/insufficient; DocFinQA answers already exist in the document; none is an outcome-prediction benchmark |
| The local CLI and MCP interface preserve the same evidence-to-anchor-to-belief trail | `docs/anchor_cli.md`; `scripts/anchor_cli.py`; `scripts/anchor_mcp.py`; run `20261120-014606-xinyisheng_2026-2079` | measured, case-level | Demonstrates the workflow and audit trail, not statistical accuracy |

## Benchmark plan status

`configs/benchmark_plan.yaml` specifies seven benchmarks, 32 registered
baselines, 34 metrics, eight fairness rules and explicit non-pooling
boundaries.  The validator `scripts/check_benchmark_plan.py` currently reports
0 structural errors and four warnings: FinDVer, Fin-Fact and DocFinQA data are
not present in the local checkout, and the Fin-Fact dataset adapter is not
implemented.  The expanded A-share historical backtest has 470 claims from 40
companies and 2023-2025; Qwen3-8B is evaluated on a 250-claim stratified
subsample.  FinDVer is run as a 150-claim component check on all three
backbones.  Fin-Fact currently exists only as a named baseline on the main
task.  The plan is a specification; it is not a results table.

## Retired claims

The following claims appear in older internal notes but are not part of the
current forward-looking paper.  They should not be reintroduced without a new
experiment:

- Router accuracy or mixed utility on the 573-query workload.
- Determinate-branch DocFinQA gains over fixed-16 or named RAG baselines.
- FinDVer planner/fallback gains.
- "The vague branch lowers decision loss against every named baseline" without
  the paired-interval caveat.  The current evidence supports a point-estimate
  advantage and a significant advantage only on Qwen3-8B.
- "AnchorEvolve alone beats every baseline" or "evolution drives the gain".
  Neither is supported by the current 227-claim evaluation.

## Evidence hygiene

- All main-table numbers come from the 227 XBRL-verified test claims.
- The 46 text-only claims are reported separately and are not mixed into the
  main table.
- Named baselines are same-backbone adaptations with pinned official prompts;
  they are not unmodified original systems.
- External component benchmarks are never pooled with the main benchmark.
- Case results are prospective and unresolved until the 2027 annual reports;
  they support workflow and audit claims, not accuracy claims.
