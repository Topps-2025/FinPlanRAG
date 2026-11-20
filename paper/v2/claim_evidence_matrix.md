# Claim–evidence matrix (v2)

This matrix follows the project's evidence discipline. `proposed` means a formal
statement without empirical support yet; `to be tested` means the experiment is
specified but not run; `measured` means produced by a named deterministic
contract; `partial` means the pilot supports only part of the claim;
`explicitly false` means the claim must not be made.

## Real company-disjoint benchmark and decision layer (2026-11-01)

The year-wise protocol evaluates all 273 test claims (2022--2025), with five
company folds and five fold seeds. For test year \(Y\), training uses only other
company folds and outcomes public before January 1 of \(Y\). The XBRL-verified
sensitivity restricts evaluation to 227 claims with non-text-only labels.

| Claim | Evidence | Status | Boundary / challenge |
|---|---|---|---|
| A real point-in-time evidence model improves macro F1 on a company-disjoint benchmark | All labels (n=273): ensemble macro F1 .5042 vs .2105, CI [.2278, .3587]; per-seed CIs exclude 0. XBRL-verified (n=227): .4884 vs .2203, CI [.1942, .3380] | **measured (real, robust)** | Simple logistic model; silver-ish paraphrase claims; no reader |
| The real model improves balanced accuracy (mean per-class recall) on a company-disjoint benchmark | All labels: .5656 vs .3333, CI [.1668, .2947]; every per-seed Δ CI excludes 0 (+.207 to +.249). XBRL-verified: .5635 vs .3333, CI [.1643, .2937] | **measured (real, robust)** | Balanced accuracy is the accuracy-family metric robust to the majority-class imbalance; plain accuracy is not |
| The real gains are not only class rebalancing | Featureless prior-sampling control: all-label balanced accuracy .3457, macro F1 .3366; verified .3396/.3340. Model: .5656/.5042 all-label, .5635/.4884 verified | **measured (control)** | The prior control predicts rare classes at their training frequency; the model still beats it by a wide margin |
| The real model improves plain company-disjoint accuracy | All labels: +.0527 claim-level, CI [-.0308, .1370], p=.2201; every fold seed positive (+3.7 to +7.0 points). XBRL-verified: +.0009, CI [-.0907, .0925], p=.994 | **partial (all-label point estimate positive, not significant; not robust to verified-only)** | Do not claim a confirmed plain-accuracy win; n=273 is underpowered and the 46 text-only labels drive the gain |
| The real model improves company-disjoint selective accuracy and AURC | All labels: selective .5839 vs .4615, CI [.0020, .2321]; AURC .4120 vs .4973, CI [-.2356, -.0074]. XBRL-verified: selective CI [-.0856, .1648]; AURC CI [-.1758, .0857] | **partial (all-label positive; verified-only not significant)** | The all-label gain is not label-robust; report both |
| The decision layer lowers real decision loss relative to always-answer | XBRL-verified calibrated: planner - answer = -.72 (four of five per-seed CIs exclude 0; the fifth is borderline); all-label calibrated: -.64; all-label raw: -.52 | **measured (real, qualified)** | Loss weights are a modeling choice; grid reported |
| The planner has lower regret across loss configurations than any fixed policy | Verified calibrated: planner mean regret +.052, max +.383; always-adjudicate +.498, max +3.063; always-answer +1.429, max +3.685; always-abstain +1.710, max +6.081 | **measured (real, robust)** | This is the primary decision-interface claim; not uniform dominance at one weight vector |
| The planner uniformly beats always-adjudicate at the base loss | Verified calibrated: planner 3.997 vs adjudicate 3.919 (+.078); the fixed policy wins 5/5 seeds. RF: planner 3.011 vs fixed 2.932 (+.079) | **explicitly false** | The planner is slightly worse than the fixed adjudicator at base; the portfolio matches the fixed policy by selecting it. Claim near-optimality across the frontier, not base superiority |
| The company-disjoint macro-F1 gain survives label verification | 46 text-only test labels excluded; macro F1 remains +.268 [.194, .338] | **measured (robust)** | Accuracy/selective/AURC do not survive the same check |

**Reading.** The robust real-data claims are (i) the company-disjoint macro-F1
gain and (ii) the planner's low regret across the loss grid. The accuracy,
selective-accuracy, and AURC gains on the all-label benchmark are not robust to
excluding text-only labels and must not be presented as confirmed. The planner
ties, rather than beats, always-adjudicate at the base loss.

## DeepSeek baselines (2026-11-02)

| Claim | Evidence | Status | Boundary / challenge |
|---|---|---|---|
| A strong LLM point reader beats the point-in-time evidence model | Verified 227: DeepSeek point accuracy .485, balanced .413, macro F1 .415; logistic ensemble .498/.564/.488. Self-consistency .498/.417/.418; hybrid with logistic prior .520/.547/.503, not better than the prior | **explicitly false** | LLM point answers are a weak baseline; contamination caveat |
| An LLM credible set is a competitive adjudication action | Verified 227: coverage .916, mean size 2.24, base set loss 3.154 vs 3.877 for the logistic set; lower loss in 9/10 configurations, loses under narrow width | measured (LLM, contaminated) | Pretraining may contain later outcomes; one prompt; coverage lower than logistic (.916 vs .996) |
| The LLM is better used as an adjudicator than a point predictor | Point balanced accuracy .413 vs set coverage .916 at size 2.24; the set action has lower base loss than the logistic set | measured (working result) | The set is coarse and coverage is below the logistic set; report the trade-off |
| A text-only TF-IDF baseline is competitive with the structured model | 5 seeds: TF-IDF .512/.481/.475; TF-IDF + numeric .540/.533/.509; logistic .514/.560/.501 | measured (classical) | Text adds plain accuracy/macro, not balanced accuracy; the interface is backend-agnostic |
| The planner uniformly beats a learned pre-hoc router | Real router: base loss 3.516 vs planner 3.927; oracle-mode router and planner each win 5/10 configurations (width router 5, planner 4, one tie) | **explicitly false** | The planner is regime-dependent; report the boundary and the router as a strong baseline |

## Portfolio, named baselines, and extended metrics (2026-11-04)

| Claim | Evidence | Status | Boundary / challenge |
|---|---|---|---|
| The loss-aware portfolio attains oracle-level regret across the loss grid | RF 200, 10 seeds, 227 verified: matches the oracle best fixed policy in 9/10 configurations (regret 0); wide 4.340 vs 4.710, regret -0.370, 95% CI [-0.643, -0.100]; mean -0.037 [-0.064, -0.010], max 0. Confirm seeds 11-20 cover 4 cells: wide -0.352 [-0.644, -0.065] vs oracle fixed; preregistered primary vs width-tuned fixed is -0.865 [-1.226, -0.490]; four-cell mean -0.088 [-0.161, -0.016], max 0. Base non-inferiority secondary fails (+0.099 [0.001, 0.211], margin .2). | **measured (real, robust)** | Oracle-level regret, not base-loss superiority; fixed adjudicator is in the candidate set; confirmatory seeds reuse the same claims and only 4 configs |
| Named strong baselines are competitive but do not dominate | FinDVer acc .551/bal .461/AUROC .674; Fin-Fact .537/.446/.650; Self-RAG .546/.451/.670; CRAG .529/.423/.643; FLARE .511/.417/.634; Adaptive-RAG .493/.406/.584; FinPlan logistic .493/.561/.716; McNemar p=.17-1.00 | measured (adapted, 227 claims) | Prompt-level adaptations under DeepSeek, not official code; LLM contamination; one deterministic seed |
| FinPlan has the strongest balanced accuracy and ranking metrics among the named table | FinPlan balanced .561/MCC .273/AUROC .716/AP .524; named baselines balanced .406-.461, AUROC .584-.674 | measured (descriptive) | FinDVer has higher accuracy .551 and Self-RAG higher selective/AURC; no uniform win. LLM rows elicit a label confidence, not a three-class distribution, so NLL/Brier/ECE are not compared |
| The rolling-origin benchmark preserves the RF/logistic trade-off | 737 claims, 2015-2025, 5 seeds: RF .537 accuracy/.503 balanced/.032 ECE/.593 selective; logistic .487/.535/.092/.501 | measured (stress test) | Same point-in-time construction, not a new claim sample; yearly heterogeneity |

## Pilot update (2026-10-31)

The pilot is a decision-level experiment on the v4 revenue-guidance test split
(273 cases). It is not an answer-level financial QA benchmark. The main
configuration is γ=0.01, outcome cost 2.0, future-signal precision 1.0.

| Claim | Evidence | Status | Boundary / challenge |
|---|---|---|---|
| Financial queries are heterogeneous; fixed-mode systems are insufficient | Pilot: fixed policies lose to the planner at the main config | partial | One track only (revenue guidance); no determinate/mixed/vague tracks yet |
| A credal-state formulation can represent precise answers, qualified adjudications, and abstentions with endogenous decision time and evidence cost | `src/finplan_rag/decision/`; `algorithm_design.md` | proposed / implemented | Representation is not proof of necessity; the pilot is decision-level |
| A point answer is insufficient when the feasible set is non-singleton and the loss is asymmetric (P1) | Pilot: always-answer 5.656 vs planner 2.977; paired CI excludes 0 | measured (pilot) | v4 guidance only; loss parameters are a modeling choice |
| A relevance-only policy is insufficient for typed evidence (P2) | No typed-evidence baseline in the pilot | to be tested | The current-quarter anchor is too weak to test P2 |
| A pre-hoc router is insufficient when the optimal mode changes after evidence (P3) | Real v4: planner = no-evidence planner. Controlled multi-seed (20 seeds), MA subset, budget 1.0: planner_preliminary top-1 .8986 vs no-evidence .4695, paired +.4290 [.4110, .4471], 20/20 seeds positive | **controlled supported / real not supported** | The preliminary signal is synthetic; a real evidence-rich benchmark is required |
| A preliminary signal can induce mid-retrieval mode switching | Controlled 20-seed MA subset: preliminary rate 1.0, set size 1.0, ECE .046, Brier .182, log loss .380; frontier weakly dominates the pre-hoc router | measured (controlled) | Not a real-data result; assignment is synthetic |
| Decision-sufficient stopping improves over fixed top-\(k\) / fixed budget (P4) | Pilot: cost-loss frontier; fixed policies are not uniformly optimal | measured (pilot) | The planner matches the best fixed policy in each regime, not uniformly better |
| Independence and defeat are necessary for sound credence (P5) | No independence/defeat structure in v4 | to be tested | Requires a richer evidence benchmark |
| Endogenous cutoff can be point-in-time measurable (P6) | Pilot: leakage rate 0 for all policies | measured (pilot) | The realized delay is used for accounting; the planner uses a historical estimate |
| A fixed budget is suboptimal when marginal VoI and cost vary across queries (P7) | Pilot: planner matches always-wait at low cost and no-wait at high cost | partial | At the crossover the planner can be worse than the best fixed policy; low-width loss makes waiting unnecessary |
| A joint planner improves the cost–decision-loss frontier over fixed policies and pre-hoc routing | Real v4: planner 2.977 vs no-wait 3.879, always-adjudicate 4.000 at the main config. Controlled MA: planner_preliminary 1.4617 vs no-evidence 2.9626. Year-wise real benchmark: planner regret +.052 vs +.498 for always-adjudicate across ten loss configs | partial (controlled positive; real regret positive; real v4 timing only) | Not uniformly better than always-wait on real v4; loses at the crossover and under low width |
| The planner works on real financial data | Year-wise company-disjoint evidence benchmark + decision layer: macro F1 .5042 vs .2105 (ensemble CI [.2278,.3587]); planner beats always-answer by -.72 (verified, four of five per-seed CIs exclude 0) and ties always-adjudicate at base | **measured (real, qualified)** | The evidence model improves macro F1; the planner is near-optimal across the loss grid but does not uniformly beat the best fixed adjudicator; silver labels; no reader |
| A real point-in-time evidence model improves macro F1 and balanced accuracy on temporal and company-disjoint splits | Temporal macro F1 .4950 vs .2105, 20-seed CI [.4881,.5020], balanced accuracy .5742 vs .3333; year-wise n=273 macro F1 .5042 vs .2105 [.2278,.3587], balanced accuracy .5656 vs .3333 [.1668,.2947]; verified n=227 macro F1 .4884 vs .2203 [.1942,.3380], balanced .5635 vs .3333 [.1643,.2937] | measured (real, robust) | Simple logistic model; real point-in-time features; silver labels |
| The real model improves company-disjoint accuracy and selective prediction | Year-wise all-label: accuracy +.0527, CI [-.0308,.1370]; selective .5839 vs .4615, CI [.0020,.2321]; AURC .4120 vs .4973, CI [-.2356,-.0074]. XBRL-verified n=227: accuracy +.0009, CI [-.0907,.0925]; selective CI [-.0856,.1648]; AURC CI [-.1758,.0857] | **partial (all-label positive; not robust to label verification)** | Do not claim a confirmed accuracy/selective win; the old n=108 subset was small and its negative result is superseded by the year-wise protocol |
| The current-quarter anchor is worth its cost | Pilot: `planner` = `planner-no-evidence` at the main config | **explicitly false (pilot)** | Negative result for this evidence type, not for the decision interface |
| Existing systems already solve the joint decision problem | No evidence | explicitly false | Capability map shows each system fixes at least one of mode, time, budget |
| v1 closure results imply answer-level correctness | No supporting evidence | explicitly false | v1 is file-level closure; reader accuracy is a separate term |
| The mixed benchmark is representative of vague financial claims | Pilot covers v4 revenue guidance + a controlled preliminary signal | to be validated | Real determinate/mixed/vague and real mode-ambiguous subsets are not built yet |
| The framework beats the strongest same-obligation control on real-data retrieval | Standard eval: LOFin/FinGLM v9 ties metadata_v9_fill exactly; period-metadata comparison is mixed | explicitly false | Do not claim a retrieval method win over the strongest control |
| The framework beats weak retrieval baselines | Standard eval: v9 vs single-shot/generic/HiREC wrong-doc McNemar p<.001; validation8 24/24 vs 4-11/24 | measured (bounded) | Baselines are not the strongest control; scope is file-level/period retrieval |
| The parser representation has a real-data positive on the preregistered holdout | validation8: v9 24/24 vs v4 4/24 (p=1.9e-6), v7 10/24, v8 19/24, single 11/24; metadata_v9_fill also 24/24 (p=1.0) | measured / bounded | Tied with the same-obligation metadata control; n=24 |
| The framework beats the strongest control on real-data outcome prediction | Year-wise company-disjoint macro F1 .5042 vs .2105 (all labels, CI [.2278,.3587]) and .4884 vs .2203 (XBRL-verified, CI [.1942,.3380]); balanced accuracy .5656 vs .3333 (all) and .5635 vs .3333 (verified); plain accuracy/selective/AURC are not robust to verified-only labels | partial (macro F1 and balanced accuracy robust; plain accuracy/selective/AURC not robust) | Do not claim a uniform real-data method win; the plain-accuracy/selective/AURC gains are label-sensitive |

## Evidence boundaries

- **v1 vs v2.** v1 is file-level period retrieval under fixed cutoff/budget. v2
  is decision-level output/timing with endogenous time/cost. Do not pool their
  numbers as if they were the same task.
- **Determinate vs vague.** Determinate questions have answer gold; vague
  claims have outcome gold. A unified decision loss must model the asymmetry
  explicitly; do not compare raw accuracy across modes.
- **Silver labels.** The v2/v4 benchmarks are silver; quotes are paraphrases
  until human verification. No claim of human-validated quality.
- **2026 citations.** Entries from 2026 are preprints unless a venue is
  verified; do not claim peer-reviewed status without checking the primary
  record.
- **No online exploration.** Any bandit/RL component is evaluated offline or in
  a simulator; no online exploration on financial data.
- **Pilot scope.** The pilot tests decision timing, not evidence-driven mode
  switching. Do not upgrade "planner matches always-wait" into "the planner
  solves the joint decision problem."
