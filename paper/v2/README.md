# FinPlan-RAG v2: a decision-interface paper

**Status: research design + pilot/confirmatory runs + ACL draft and v3.1
conformal decision layer (2026-11-03).**

This folder contains the v2 research design for FinPlan-RAG. The v2 paper is
not a period-retrieval paper and not a "RAG + adjudication module" paper. It is
a decision-interface paper: a planner over credal evidence states that jointly
decides **when to decide, what evidence to acquire, and which output mode to
use** (precise answer / qualified adjudication / abstention).

## The one assumption removed

v1 assumed a **fixed cutoff** and a **fixed budget**. v2 does not.

- The decision time \(t\), the evidence cost \(\lambda(e)\), and the delay cost
  \(\gamma\) are part of the decision problem. Fixed cutoff/budget is a special
  case, not the problem statement.
- Point-in-time is preserved: at the chosen decision time \(t\), only evidence
  with \(a(e)\le t\) is admissible, and the policy must be
  \(\mathcal F_t\)-measurable. Waiting is an explicit, costly action, not free
  lookahead.
- Evaluation reports **cost--decision-loss frontiers**, not a single budget
  point.

## What happens to v1

v1's evidence is not discarded. Fixed-cutoff, fixed-budget period retrieval is a
special case of v2; the v1 results become a slice/baseline, not the main claim.
The metadata-fill tie is a finding about the fixed-policy slice: under a fixed
period-retrieval task, the representation/parser is the active ingredient, not
the search policy. v2 asks a different question: whether a planner that also
chooses the decision time, evidence acquisition, and output mode improves the
cost--loss frontier when queries are heterogeneous.

## Pilot and confirmatory status (2026-11-01)

A decision-level pilot has been run on the v4 test split (273 cases):

- The planner matches `always-wait` at the main cost configuration and beats
  no-wait, always-answer, always-adjudicate, fixed-budget, and pre-hoc router
  baselines; the planner-vs-no-wait paired difference is significant.
- The current-quarter anchor is not worth its cost: `planner` and
  `planner-no-evidence` are identical at the main configuration.
- The planner is not uniformly better than the best fixed policy at the
  cost-loss crossover, and it loses under a low width penalty.
- **Controlled mode-switch diagnostic:** on the MA subset (n=133),
  `planner_preliminary` beats the pre-hoc cost-aware router
  (`planner_no_evidence`) by −1.5009 total loss, 95% CI [−1.875, −1.072].
  This supports P3 in a controlled setting.
- **Multi-seed controlled result (20 seeds, pre-registered):** at cost budget
  1.0 on the MA subset, `planner_preliminary` top-1 accuracy .8986
  [.8855, .9117] vs the pre-hoc cost-aware router .4695; paired difference
  **+0.4290 [0.4110, 0.4471]**, **20/20 seeds positive**; the planner weakly
  dominates the cost-accuracy frontier. This is the strongest positive result,
  but it is controlled/synthetic, not real-data.
- **Real point-in-time evidence benchmark (temporal):** a logistic evidence
  model on guidance/current-quarter/history/previous-guidance features is
  positive on **macro F1** on the temporal test (.4950 vs .2105, 20-seed CI
  [.4881, .5020]) and on **balanced accuracy** (.5742 vs .3333). The 20-seed
  bootstrap is also positive on accuracy, selective, and AURC; the single-fit
  accuracy interval includes zero. The old n=108 company-disjoint subset was
  small and is superseded by the year-wise benchmark below. See
  `real_benchmark_result.md`.
- **Year-wise company-disjoint benchmark (new):** all 273 test claims, five
  fold seeds, trained only on other companies and outcomes public before the
  test year. Macro F1 .5042 vs .2105 (ensemble 95% CI [.2278, .3587]) and
  balanced accuracy .5656 vs .3333 (CI [.1668, .2947]); every per-seed
  macro-F1 and balanced-accuracy CI excludes zero. Plain accuracy .5165 vs
  .4615 (claim-level CI [-.0308, .1370]) and selective@50 .5839 vs .4615 (CI
  [.0020, .2321]) are positive, but plain accuracy is underpowered. On the 227
  XBRL-verified claims both robust gains survive: macro F1 .4884 vs .2203 (CI
  [.1942, .3380]) and balanced accuracy .5635 vs .3333 (CI [.1643, .2937]);
  plain accuracy/selective/AURC confidence intervals include zero. A
  featureless prior-sampling control gets balanced accuracy .3396 and macro F1
  .3340, so the gain is not only class rebalancing. See
  `company_disjoint_result.md`.
- **Real decision layer (new):** on calibrated probabilities, the planner beats
  always-answer by about -.72 decision loss on XBRL-verified claims (four of
  five per-seed CIs exclude zero; the fifth is borderline), ties
  always-adjudicate at the base loss, and has mean regret +.052 (max +.383)
  across ten loss configurations versus +.498 (max +3.063) for
  always-adjudicate. Raw probabilities give +.287 (max +.601); calibration is
  part of the model. The value is low regret across the loss frontier, not
  uniform dominance at one weight vector. See `real_decision_result.md`.
- **Selected version: non-LLM RF portfolio (2026-11-03):** a loss-aware
  portfolio over fixed adjudication, abstention, and the planner, with a
  random-forest backbone (200 trees, point-in-time structured features) and a
  .15 validation switching margin. Across ten loss configurations on 227
  XBRL-verified claims it matches the oracle fixed policy in nine (regret 0)
  and beats it under wide width (4.340 vs 4.710; regret -0.370, 95% CI
  [-0.643, -0.100]); mean regret -0.037 (95% CI [-0.064, -0.010]), max 0.000.
  The base is an exact match to the optimal fixed policy. Confirmatory seeds
  11-20 give wide -0.352 (95% CI [-0.644, -0.065]) and max 0.000. See
  `algorithm_selected_portfolio.md` and `portfolio_10seed.json`.
- **Named baselines (2026-11-04):** six named methods are adapted under the
  same DeepSeek backbone: FinDVer, Fin-Fact, Self-RAG, CRAG, FLARE, and
  Adaptive-RAG. No point model dominates: FinDVer has the best accuracy .551,
  Self-RAG the best selective@50/AURC (.684/.333), and FinPlan the best
  balanced accuracy/MCC/AUROC (.561/.273/.716) and lowest ECE .048; McNemar
  does not reject equal accuracy for any named baseline (p=.17--1.00).
  A 737-claim rolling-origin backbone check is retained. See
  `named_baselines_result.md` and `metrics_suite_result.md`.
- **Conformal alternative (2026-11-03):** the planner's
  heuristic credible set is replaced by a split-conformal set; the threshold
  is selected on calibration data by realized decision loss subject to a
  coverage constraint, and the score family/alpha on company-disjoint
  validation. On 227 XBRL-verified claims with the DeepSeek backbone, base loss
  is 3.1013 vs 3.1189 for the tuned planner and 3.1225 for the width-tuned
  fixed adjudicator; cheap abstention 1.9471 vs 2.5110/3.1225, narrow width
  1.0529 vs 1.3172/1.2907, wide width 4.4467 vs 4.5419/5.5330. Across ten loss
  configurations mean regret is -0.0974 and max regret -0.0106 against an
  oracle best fixed policy; base coverage .9163, set size 2.203, zero
  fallbacks. The gain is backbone-dependent: on the logistic backbone the
  expected-objective layer has base 3.343 vs 3.293 fixed, winning under
  under-averse and wide width only. See `algorithm_v3_conformal.md` and
  `paper/results/deepseek_conformal_realized_verified/`.
- **DeepSeek backbone and baseline (new):** DeepSeek is the framework's base
  model/reader backend, not a separate method. On 227 XBRL-verified claims, its
  point answers are weaker than the evidence model (balanced accuracy .413 vs
  .564; macro F1 .415 vs .488), and self-consistency does not close the gap.
  Its credible sets are a stronger action: coverage .916 at mean size 2.24,
  base decision loss 3.154 versus 3.877 for the logistic set. With the same
  planner, the DeepSeek backbone has lower loss than the logistic backbone in
  8/10 configurations (base 3.101 vs 3.982; max 4.542 vs 5.110), losing cheap
  abstention and the narrow width penalty. The comparison is not clean
  point-in-time evidence because of pretraining contamination. See
  `deepseek_baseline_result.md` and `baseline_matrix.md`.
- **Backbone controls (not named baselines):** five-seed TF-IDF + numeric
  reaches accuracy .540, balanced .533, macro F1 .509; the structured logistic
  model reaches .514/.560/.501. These are diagnostics only; the paper's
  baseline table uses the six named methods in `named_baselines_result.md`.
- **Real pre-hoc router (new):** a learned oracle-mode router has base loss
  3.516 vs planner 3.927; the two policies each win five of ten configurations
  (the width-threshold router wins five, the planner four, one tie). The
  planner is better under under-averse, cheap abstention, narrow width, wide
  width, and alpha=.20; it is not a uniform method win. See
  `router_baseline_result.md`.
- **Standard-metric check:** no real-data method/policy positive against the
  strongest same-representation control on the *old* artifacts (v4/v2 tie;
  LOFin/FinGLM tie metadata-fill; validation8 ties metadata-fill). The **new v2
  planner** on v4 forecast-only also ties the majority on top-1/macro F1;
  wait-allowed 1.0 is trivial. The representation/parser and the controlled
  mechanism are positive. The year-wise company-disjoint benchmark and the
  decision layer are the new real-data positives; see
  `company_disjoint_result.md` and `real_decision_result.md`, plus
  `standard_eval_verdict.md` and `paper/results/standard_eval/v2_framework.md`.
- **Go/no-go: conditional go.** The controlled mechanism is positive; the
  real-data evidence-rich benchmark is the first task of the full experiment.
  See `pilot_challenge.md` for the two-track protocol.

## Files

- `research_design.md`: master document (background, related-work capability
  map, gap, RQs, formal motivation summary, evaluation, reviewer challenge,
  paper structure).
- `formal_motivation.md`: decision-theoretic setup and propositions P1--P7.
- `experiment_plan.md`: minimal experiment set, baselines, metrics, ablations,
  falsification criteria.
- `pilot_benchmark_spec.md`: 100-query pilot specification, annotation schema,
  and mode-ambiguous construction rules.
- `pilot_dataset_card.md`: the v4 decision-level pilot dataset.
- `metrics_spec.md`: decision loss, ECE, coverage, cost-loss frontier, paired
  bootstrap.
- `algorithm_design.md`: the conservative one-step planner implemented in
  `src/finplan_rag/decision/`.
- `pilot_challenge.md`: pilot results, re-challenge of the gap, and the
  go/no-go decision.
- `standard_eval_verdict.md`: standard-metric evaluation across validation8,
  LOFin, FinGLM, v4, v2, synthetic, and controlled datasets.
- `multiseed_protocol.json`: pre-registered 20-seed controlled experiment.
- `multiseed_result.md`: 20-seed result (primary budget-1.0 metric, cost-accuracy
  frontier, real v4 control).
- `real_benchmark_protocol.json`: pre-registered real evidence benchmark.
- `real_benchmark_result.md`: real point-in-time evidence result (temporal and
  the old n=108 company-disjoint subset, 20-seed training bootstrap).
- `company_disjoint_protocol.json`: year-wise, company-disjoint, point-in-time
  protocol over all 273 test claims.
- `company_disjoint_result.md`: all-label and XBRL-verified company-disjoint
  results (macro F1 robust; accuracy/selective/AURC not robust).
- `real_decision_result.md`: real decision-layer result (planner vs fixed
  policies, loss grid, regret).
- `baseline_matrix.md`: fixed baseline specification and status.
- `deepseek_baseline_result.md`: DeepSeek point, selective, self-consistency,
  hybrid, and credible-set results, including bad-case iterations.
- `router_baseline_result.md`: real pre-hoc router baseline and the boundary
  of the planner's advantage.
- `references_checklist.md`: latest related-work entries, arXiv IDs, and
  verification status.
- `claim_evidence_matrix.md`: claims and evidence status (mostly "to be
  tested"; no claim is silently upgraded).
- `metrics_suite_result.md`: extended metric suite, RAG baseline, paired
  comparisons, and rolling-origin benchmark.
- `algorithm_selected_portfolio.md`: selected version (loss-aware portfolio
  over fixed/abstain/planner with a non-LLM RF backbone; results, confirmatory
  seeds, code map, falsification).
- `algorithm_selected_planner_rf.md`: component/ablation (credible-set planner
  with the same RF backbone; base-loss boundary).
- `algorithm_v3_conformal.md`: conformal alternative (realized threshold,
  nested selection, results, falsification, code map).
- `acl/main.tex`: current ACL-format draft; `acl/FinPlan_ACL_review.pdf` and
  `acl/FinPlan_ACL_preprint.pdf` are the rendered review/preprint versions.
