# FinPlan ACL draft

**Status:** complete ACL-format draft for review, 2026-11-03; updated
2026-11-08. Main text fits in 8 pages; references start on page 9 and the
appendix follows. This is a draft for critique, not a submission-ready camera
copy.

## Files

- `FinPlan_ACL_review.pdf`: ACL review format with line numbers.
- `FinPlan_ACL_preprint.pdf`: clean preprint format without line numbers.
- `main.tex`: source.
- `references.bib`: 32 cited references. Recent 2026 entries are arXiv
  preprints, not venue claims.
- `figures/regret.pdf`, `figures/effects.pdf`: vector result figures.
- `make_figures.py`: regenerates the figures from
  `paper/results/real_decision_eval_verified_calibrated/summary.json`.
- `acl.sty`, `acl_natbib.bst`: official ACL style files from
  `acl-org/acl-style-files`.

Compile with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` from this directory.

## What the draft claims

1. The output interface (answer, credible set, abstain), the decision time,
   and the evidence budget are decision variables, not task constants.
2. Fixed output interfaces are special cases and have high regret in loss
   regimes other than the one they were designed for.
3. On a year-wise company-disjoint benchmark of 273 revenue-guidance claims,
   a logistic evidence model (five-seed raw-probability ensemble) improves
   balanced accuracy (.564 vs .333) and macro F1 (.488 vs .220) on
   XBRL-verified labels; both gains survive the verified-only sensitivity.
4. On a non-LLM random-forest backbone, a loss-aware portfolio over fixed
   adjudication, abstention, and the planner matches the oracle best fixed
   policy in nine of ten loss configurations (regret 0) and beats it under the
   wide-width penalty (4.340 vs 4.710; regret -0.370, 95% CI [-0.643, -0.100]).
   Mean regret is -0.037 (95% CI [-0.064, -0.010]) and maximum regret is 0.000
   on ten seeds. The policy is selected by validation loss with a .15 switching
   margin; test outcomes are never used.
5. A pre-specified seed replication (seeds 11-20, four cells) confirms the
   pattern: the portfolio selects fixed at base and matches it exactly; wide
   is -0.352 (95% CI [-0.644, -0.065]) against the oracle fixed policy, and
   the pre-specified primary against the width-tuned fixed adjudicator is
   -0.865 (95% CI [-1.226, -0.490]); the four-cell mean regret is -0.088
   [-0.161, -0.016], maximum 0.000. The planner alone is +0.099 [0.001, 0.211]
   above fixed at base, so the .2 non-inferiority secondary is not met. This
   is a seed-level replication on the same claims, not a new claim sample.
6. A controlled 20-seed diagnostic isolates evidence-induced mode switching.
7. The six named baselines load prompt templates and decision helpers from
   their official repositories at pinned commits under one shared three-way
   adapter; missing checkpoints/data are documented substitutions. FinPlan
   keeps the best balanced accuracy and AUROC on DeepSeek, GLM 5.3 flash, and a
   locally served open-weight Qwen3-8B (.561/.561/.561 balanced and
   .716/.716/.716 AUROC); the best source-grounded baseline is
   .469/.481/.509 balanced and .689/.664/.630 AUROC. No single point model
   dominates.

## What the draft does not claim

- A uniform plain-accuracy or selective-prediction win. Those gains are not
  robust to the XBRL-verified sensitivity.
- A reader or end-to-end financial QA result. The benchmark is decision-level.
- A state-of-the-art financial RAG system. No retriever or reader is evaluated.
- Human-annotated gold labels. The claims are automatically extracted
  paraphrases; 69 of 1,005 labels are text-only.
- That the planner is robust to an unknown loss without the minimax-regret
  variant.
- A base-loss win. At base the fixed adjudicator is optimal; the portfolio
  matches it. The claim is oracle-level regret, not base superiority.
- A uniform method win. The portfolio matches the oracle in most regimes and
  wins under wide width; it is a selector, not a universally better reader.
- That the conformal layer is the selected method. It loses to the planner on
  the RF backbone and its DeepSeek advantage is sensitive to masking.
- Clean point-in-time LLM evidence. DeepSeek may have seen 2022--2025 outcomes;
  the paper states the contamination caveat.
- Distribution-free coverage under the year-wise company shift. Empirical test
  coverage is reported; the shift violates exchangeability.

## 2026-11-02 baseline additions

- **DeepSeek:** point reader is weaker than the structured evidence model on
  verified labels (balanced .413 vs .564; macro F1 .415 vs .488). Its credible
  sets are competitive: coverage .916 at size 2.24, base decision loss 3.154
  vs 3.877 for the logistic set, lower loss in 9/10 configurations. The
  comparison has a pretraining-contamination caveat.
- **Classical:** five-seed TF-IDF + numeric reaches accuracy .540, balanced
  .533, macro F1 .509; structured logistic reaches .514/.560/.501. The text
  model has higher plain accuracy/macro, the structured model higher balanced
  accuracy.
- **Real pre-hoc router:** in a separate dev/validation-to-test protocol, the
  oracle-mode router has base loss 3.516 vs 3.927 for the raw planner; the two
  policies each win five of ten configurations (the width-threshold router
  wins five, the planner four, and one is a tie). The planner is
  regime-dependent, not a uniform win.
- New files: `../baseline_matrix.md`, `../deepseek_baseline_result.md`,
  `../router_baseline_result.md`, `../reproduction_status.md`; scripts
  `scripts/run_deepseek_baselines.py`, `scripts/deepseek_client.py`,
  `scripts/run_classical_baselines.py`,
  `scripts/analyze_deepseek_adjudication.py`,
  `scripts/run_backbone_policy_eval.py`, `scripts/run_router_baseline.py`.

## 2026-11-03 conformal decision layer

- Algorithm: `paper/v2/algorithm_v3_conformal.md`.
- Code: `src/finplan_rag/decision/conformal.py` (default realized threshold
  objective; expected objective is an ablation).
- DeepSeek verified run:
  `paper/results/deepseek_conformal_realized_verified/` (5 split seeds,
  n=227, per-claim losses, paired bootstrap).
- DeepSeek all-label run:
  `paper/results/deepseek_conformal_realized_all/`.
- Logistic realized five-seed run:
  `paper/results/conformal_v3_realized_seed{1..5}/`.
- Logistic expected-objective ablation:
  `paper/results/conformal_v3_verified_seed{1..5}/`.
- The main table is Table 3; the logistic planner table and the controlled
  diagnostic are in Appendix C.

```text
PYTHONPATH="src;scripts" python scripts/run_deepseek_conformal_set_eval.py \
  --verified-only --out paper/results/deepseek_conformal_realized_verified

PYTHONPATH="src;scripts" python scripts/run_deepseek_conformal_set_eval.py \
  --out paper/results/deepseek_conformal_realized_all
```

## 2026-11-04 named baselines, metrics, benchmark

- Full 11-metric suite on 227 verified claims: accuracy, balanced accuracy,
  macro F1, MCC, Cohen's kappa, AUROC, AP, ECE, Brier, NLL, selective@50,
  AURC, per-class P/R/F1, risk-coverage, bootstrap CIs, McNemar. The paper's
  main table reports classification and ranking metrics; NLL/Brier/ECE are
  not compared across LLM and calibrated-probability rows because the LLM
  baselines elicit a label confidence, not a normalized three-class
  distribution.
- Six named baselines adapted under the same DeepSeek backbone: FinDVer-style,
  Fin-Fact-style, Self-RAG-style, CRAG-style, FLARE-style, Adaptive-RAG-style;
  no exact FinRank/Fin-RATE/HC-RAG/FinSAgent reproduction because no
  compatible code/shared claim labels.
- No point model dominates: FinDVer-style .551 acc/.461 bal/.674 AUROC;
  Fin-Fact-style .537/.446/.650; Self-RAG-style .546/.451/.670; CRAG-style
  .529/.423/.643; FLARE-style .511/.417/.634; Adaptive-RAG-style
  .493/.406/.584; FinPlan logistic .493/.561/.716. McNemar against FinPlan
  p=.17-1.00; Fin-Fact/CRAG/FLARE/Adaptive-RAG AUROC intervals exclude zero,
  FinDVer/Self-RAG include zero.
- Rolling-origin backbone check: 737 claims, 2015-2025; RF .537 acc/.503 bal/
  .032 ECE; logistic .487/.535/.092.
- Artifacts: `paper/results/named_baselines/`,
  `paper/results/rolling_benchmark/`; summaries
  `paper/v2/named_baselines_result.md`, `paper/v2/metrics_suite_result.md`.

## Main review risks

- **ACL fit:** no language reader or retriever; the evidence model is a
  logistic regression over structured features. The draft is closer to a
  finance benchmark/analysis paper than a core NLP method paper.
- **Novelty:** the planner is a Bayes action; the formulation and benchmark are
  the contribution. A reviewer may see the decision rule as standard
  decision theory applied to a new task.
- **Loss misspecification:** the planner is near-optimal under a specified
  loss; the off-diagonal result shows the cost of getting the loss wrong. The
  minimax-regret variant is a prototype, not a full solution.
- **Base tie / router:** the conformal layer ties the tuned planner and
  width-tuned adjudicator at the base loss; its gain is in three loss regimes.
  A learned pre-hoc router in a separate temporal protocol has lower base loss
  than the raw planner (3.516 vs 3.927). The layer is regime-dependent, not a
  uniform method win.
- **Backbone dependence:** the deep gain is on the DeepSeek backbone; the
  logistic backbone does not reproduce the base gain. Reviewers may read the
  result as an LLM-specific effect or a pretraining-contamination artifact.
  The paper states both caveats.
- **Labels:** automatic paraphrase claims, no human IAA; verified-only is a
  sensitivity check, not gold.
- **Baselines:** six named general methods (FinDVer, Fin-Fact, Self-RAG, CRAG,
  FLARE, Adaptive-RAG) are adapted under the same backbone; exact recent
  financial RAG systems (FinRank, Fin-RATE, HC-RAG, FinSAgent) are not run
  because no compatible code or shared labels are available.

## Suggested next steps

1. Add a text-based reader or retrieval baseline, or target a finance/benchmark
   venue rather than the ACL main track.
2. Human-verify a subset (e.g., 100 claims) and report IAA.
3. Integrate the minimax-regret planner into the main method and report it as
   the robust variant; add a loss-elicitation study.
4. Add a real pre-hoc router baseline; the current router comparison is
   controlled only.
5. Compare against at least one recent financial RAG/verification system on a
   shared subset.
6. Fix the minor TikZ overfull box and consider enlarging Table 1.
