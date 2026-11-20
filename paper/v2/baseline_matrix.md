# Baseline matrix for the FinPlan experiments

**Status:** working baseline specification, 2026-11-02. The purpose is to fix
the comparison set before running the DeepSeek experiments, so that prompts and
metrics are not chosen after seeing results. The final paper displays only the
six named baselines in `named_baselines_result.md`; the generic controls below
are retained as historical/backbone diagnostics, not as the paper's baseline
table.

**DeepSeek's role:** DeepSeek is both a baseline and the framework's base
model/reader backend, not a separate method. The method is the decision
interface; DeepSeek supplies point labels, confidence, and credible sets that
the same planner can consume. The backbone comparison is in
`deepseek_baseline_result.md` and `paper/results/backbone_policy/`.

## Evaluation units

- **Benchmark:** year-wise company-disjoint test claims, 273 claims (2022--2025).
- **Verified sensitivity:** 227 claims with XBRL-verified/override/search labels.
- **Trained baselines:** trained only on other company folds with outcome dates
  before January 1 of the test year; five fold seeds.
- **LLM baselines:** no training. The prompt must contain only information
  available at the cutoff. Because the model may have seen 2022--2025 outcomes
  during pretraining, every LLM result is reported with a contamination
  caveat. A company/ticker-masked prompt is run as a robustness check.

## Baselines

### B0. Featureless controls

| ID | Baseline | Definition |
|---|---|---|
| B0a | Majority | Always predict `above` |
| B0b | Prior sampling | Sample a label from the training-fold class distribution |
| B0c | Stratified prior | Predict labels at the training-fold prior frequencies |

### B1. Classical evidence models

| ID | Baseline | Definition |
|---|---|---|
| B1a | Logistic regression | Current model: point-in-time features, C=1.0, balanced |
| B1b | Logistic + calibration | B1a with temperature scaling on internal company-disjoint folds |
| B1c | Gradient boosting | HistGradientBoosting on the same numeric features + one-hot sector |
| B1d | Random forest | Random forest on the same features, balanced class weights |
| B1e | Logistic + text | TF-IDF over `claim_text` concatenated with the numeric features |

### B2. DeepSeek point-answer baselines

| ID | Baseline | Prompt content | Output |
|---|---|---|---|
| B2a | DeepSeek zero-shot, named | Claim text, guidance low/high, current-quarter value, cutoff date, company/ticker | `below`/`within`/`above` + confidence |
| B2b | DeepSeek zero-shot, masked | Same, company/ticker replaced by `Company A` | label + confidence |
| B2c | DeepSeek evidence-grounded | B2a plus the point-in-time evidence list (guidance, current quarter, prior outcomes public before cutoff) | label + confidence + short evidence rationale |
| B2d | DeepSeek verification | Claim + evidence; ask whether the realized outcome will be below/within/above and whether the evidence is sufficient | verdict + confidence + sufficient/insufficient |

### B3. DeepSeek selective / adjudication baselines

| ID | Baseline | Definition |
|---|---|---|
| B3a | DeepSeek answer-or-abstain | Ask for a label or `ABSTAIN`; coverage and selective accuracy |
| B3b | DeepSeek credible set | Ask for the smallest set of labels with at least 90% belief |
| B3c | DeepSeek self-consistency | Five samples at temperature 0.7; majority label; confidence = vote fraction |
| B3d | DeepSeek + planner | Calibrate B3c vote fractions (or verbalized confidence) and feed the planner |

### B4. Fixed decision policies (on B1a/B1b probabilities)

| ID | Policy | Definition |
|---|---|---|
| B4a | Always answer | Argmax label |
| B4b | Always adjudicate | Smallest credible set at alpha=.1 |
| B4c | Always abstain | Pay the abstention cost |
| B4d | Fixed confidence threshold | Answer if max probability >= t, else abstain; t in {.4,.5,.6,.7} |
| B4e | Fixed budget | Stop after a fixed number of evidence items (pilot) |
| B4f | Tuned fixed adjudicator | Smallest credible set with alpha tuned on validation over {.05,.10,.20} |

### B5. Adaptive policies

| ID | Policy | Definition |
|---|---|---|
| B5a | Pre-hoc router | Feature-based mode choice before evidence; logistic router trained on dev/validation |
| B5b | Matched planner | Bayes action for the evaluation loss; calibrated probabilities |
| B5c | Minimax-regret planner | Choose the action minimizing worst-case expected regret over the ten loss configurations |
| B5d | Oracle (upper bound) | Knows the true label; used only to normalize regret, not as a baseline |
| B5e | Nested conformal layer (proposed) | LAC/APS sets; realized threshold on calibration; score/alpha on validation; Bayes action |

### B6. Recent financial RAG / verification systems

If a shared subset can be constructed, compare against at least one of:
FinRank, Fin-RATE, HC-RAG, FinSAgent. These are recent arXiv preprints; use the
authors' released code/prompts and the same claim subset. If a system cannot be
run, report the comparison as not available rather than citing its paper
numbers, which are on different data.

## Status (2026-11-02)

| ID | Status | Key result |
|---|---|---|
| B0a majority | done | verified balanced accuracy .333, macro F1 .220 |
| B0b prior sampling | done | verified balanced .340, macro .334 |
| B1a/B1b logistic | done | verified balanced .5635, macro .4884 (five-seed ensemble) |
| B1c boosting | pilot seed 1 | accuracy .458, balanced .400, macro .398 |
| B1d random forest | pilot seed 1 | accuracy .528, balanced .479, macro .478 |
| B1d RF decision backbone | done, 10 seeds | planner mean regret +.0108, max +.0784; fixed +.1457/+.9322 |
| B1e TF-IDF text only | done, 5 seeds | accuracy .512, balanced .481, macro .475 |
| B1f TF-IDF + numeric | done, 5 seeds | accuracy .540, balanced .533, macro .509 |
| B2a point | done | verified accuracy .4846, balanced .4131, macro .4153 |
| B2b masked | pilot only | 10-claim pilot; not yet full |
| B2c evidence-grounded | done | verified point label .5022/.4326/.4289 |
| B2d verification | done | credible set: coverage .9163, size 2.2423, base loss 3.1542 |
| B3a selective | done | verified accuracy .5088, balanced .4361, coverage .996 |
| B3b credible set | done | see B2d |
| B3c self-consistency | done | 273 claims, balanced .4174; no gain over point |
| B3d LLM + planner | done | self-consistency vote planner base loss 4.725; not competitive |
| B3e DeepSeek backbone + planner | done | confidence-weighted credible-set belief; base loss 3.101, max 4.542; beats logistic backbone + planner in 8/10 configs |
| B4 fixed policies | done | always-answer 4.731, always-adjudicate 3.877, abstain 5.000 (verified, seed 1) |
| B5a pre-hoc router | done | oracle-mode router base loss 3.516 vs planner 3.927; planner wins under under-averse, cheap abstention, wide width; router wins 5/10 configs |
| B5b matched planner | done | mean regret .052, max .383 (calibrated, verified) |
| B5c minimax-regret planner | prototype | mean .296, max 2.13 under misspecification |
| B6 FinDVer-style | done, 227 verified | acc .551, balanced .461, macro .464, AUROC .674, selective@50 .675 |
| B6 Fin-Fact-style | done, 227 verified | acc .537, balanced .446, macro .450, AUROC .650 |
| B6 Self-RAG-style | done, 227 verified | acc .546, balanced .451, macro .456, AUROC .670, selective@50 .684 |
| B6 CRAG-style | done, 227 verified | acc .529, balanced .423, macro .418, AUROC .643 |
| B6 FLARE-style | done, 227 verified | acc .511, balanced .417, macro .411, AUROC .634 |
| B6 Adaptive-RAG-style | done, 227 verified | acc .493, balanced .406, macro .400, AUROC .584 |
| B6 FinRank/Fin-RATE/HC-RAG/FinSAgent | not run | no compatible released code/shared claim labels; no paper-number comparison |
| B7 metric suite | done | AUROC/AP/MCC/kappa/ECE/Brier/NLL/risk-coverage/slices in `paper/results/*_metrics.json` |
| B8 rolling benchmark | done, 5 seeds | 737 claims 2015-2025; RF acc .537/bal .503/ECE .032; logistic acc .487/bal .535/ECE .092 |

**Key finding:** DeepSeek is a weaker point predictor than the point-in-time
logistic model, but its credible sets are a strong adjudication action: 9 of 10
loss configurations have lower decision loss than the logistic credible set,
with lower coverage (.916 vs .996). The loss dependence is the same phenomenon
the paper studies.

**Classical finding:** TF-IDF + numeric features has higher plain accuracy
(.540 vs .514) and macro F1 (.509 vs .501) than the logistic model across five
seeds, but lower balanced accuracy (.533 vs .560). The logistic model remains
the best balanced-accuracy backbone; the text model shows that the interface is
backend-agnostic. A decision-layer run on the text-model probabilities is the
next check.

## Metrics

Primary: balanced accuracy, macro F1.
Secondary: plain accuracy, per-class recall, selective accuracy@50, AURC, ECE,
Brier score, log loss, coverage, abstention rate, decision loss, mean/max regret.

Confidence intervals: paired bootstrap over claims; claim-level average over
fold seeds for trained models. LLM baselines get bootstrap intervals over
claims and a separate run-to-run interval from repeated samples.

## DeepSeek implementation notes

- Endpoint: `ANTHROPIC_BASE_URL` (`https://api.deepseek.com/anthropic`),
  model `deepseek-flash[1m]`.
- Credential: `ANTHROPIC_AUTH_TOKEN` from the environment. Never print it.
- Use `requests.Session(trust_env=False)` because the local proxy configuration
  breaks the direct API call.
- JSON-only output schema; temperature 0 for the primary run, temperature 0.7
  for self-consistency; cache every response by prompt hash.
- Parse failures are counted as invalid and retried once with a repair prompt;
  they are not silently mapped to a label.
- Log token usage and latency per call.

## Bad-case protocol

For every baseline, save per-claim predictions. After the pilot:
1. Stratify errors by true label, sector, guidance width, history length, and
   whether `target_quarter` is present.
2. Read 20--30 errors and label the failure mode: missing context, arithmetic
   error, overclaim, underclaim, parsing/schema failure, or label ambiguity.
3. Fix the prompt or feature code for the dominant mode, rerun the pilot, and
   only then scale to the full 273 claims.
4. Report both the pre-fix and post-fix numbers; do not hide the iteration.
