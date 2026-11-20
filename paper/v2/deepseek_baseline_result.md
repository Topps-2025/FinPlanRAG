# DeepSeek baselines: result (2026-11-02)

> The primary decision-layer result with this backbone is now the v3.1
> conformal layer in `algorithm_v3_conformal.md`; this document remains the
> planner/point/set baseline record.

**Status:** working experiment. Model: `deepseek-flash[1m]` through the
Anthropic-compatible endpoint (`ANTHROPIC_BASE_URL`), credential from
`ANTHROPIC_AUTH_TOKEN`. The key is never written to an artifact.

## Protocol

- Test claims: all 273 test claims (2022--2025); verified-only metrics use the
  227 XBRL-verified/override/search labels.
- The LLM sees only the claim text, guided range, current-quarter value,
  cutoff date, sector, target period, and company history with outcome dates
  before the cutoff. It never sees the outcome.
- Prompts return JSON with a label, confidence, and a short reason. The
  verification prompt also returns a credible set.
- Temperature 0 for point/verification; self-consistency uses five samples at
  temperature 0.7.
- Reasoning is disabled for the primary runs (`thinking: {"type": "disabled"}`).
  A 1,024-token thinking budget was also tested.
- **Contamination caveat:** the model may have seen 2022--2025 outcomes during
  pretraining. A company/ticker-masked prompt is included as a robustness
  check; it does not fully remove contamination. LLM numbers are therefore
  reported separately from the trained evidence model.

## Point-answer and selective results

All 273 claims:

| variant | valid | accuracy | balanced accuracy | macro F1 | coverage | abstain |
|---|---:|---:|---:|---:|---:|---:|
| point (v3) | 273 | .4945 | .4140 | .4171 | 1.000 | .000 |
| masked point (v3) | 273 | .5238 | .4434 | .4478 | 1.000 | .000 |
| evidence-grounded point (v3) | 273 | .4725 | .3973 | .4002 | 1.000 | .000 |
| selective (v3) | 273 | .5018 | .4213 | .4246 | .996 | .004 |
| verification point label (v3) | 273 | .5055 | .4246 | .4240 | 1.000 | .000 |
| self-consistency, 5 samples | 273 | .4982 | .4174 | .4184 | 1.000 | .000 |
| hybrid with logistic prior | 273 | .5201 | .5469 | .5029 | 1.000 | .000 |

Verified-only (227 claims):

| variant | accuracy | balanced accuracy | macro F1 |
|---|---:|---:|---:|
| point | .4846 | .4131 | .4153 |
| masked point | .5022 | .4294 | .4283 |
| evidence-grounded point | .4758 | .4093 | .4088 |
| selective | .5088 | .4361 | .4351 |
| verification point label | .5022 | .4326 | .4289 |

Masking the company/ticker does not hurt the point baseline; the masked
verified accuracy is .5022, balanced .4294, macro .4283, slightly better than
the named run. This weakens, but does not remove, the concern that the named
model uses memorized company context. Adding the raw evidence list lowers the
point result (.4758/.4093/.4088), so the evidence formatting is not the
bottleneck.

For comparison, the five-seed calibrated logistic evidence model has verified
accuracy .4978, balanced accuracy .5635, and macro F1 .4884. DeepSeek is a
**weaker point predictor** than the simple point-in-time evidence model,
including under self-consistency and a hybrid that supplies the logistic
prior. The hybrid does not improve over the prior: it mostly copies it and
slightly degrades balanced accuracy.

The point confusion matrix shows the failure mode:

| truth \ pred | below | within | above |
|---|---:|---:|---:|
| below (37) | 5 | 24 | 8 |
| within (110) | 3 | 65 | 42 |
| above (126) | 4 | 57 | 65 |

The model over-predicts `within` (146 predictions vs 110 true) and
under-predicts `below` (12 predictions vs 37 true). Mean confidence is .618;
correct predictions average .645 and incorrect ones .592, so verbalized
confidence is not a useful abstention signal. This is why the credible-set
mode is the productive use of the LLM.

## Credible-set adjudication

The verification prompt asks for the smallest credible set at roughly 90%
belief. On the 227 verified claims:

- coverage: **.9163**
- mean set size: **2.2423**
- base decision loss (adjudicate action): **3.1542**

The seed-1 calibrated logistic credible set, evaluated on the same claims, has
coverage .9956, mean size 2.9207, and base decision loss 3.8767. DeepSeek's set
is smaller and has lower decision loss at the base loss, at the cost of lower
coverage. The result is loss-dependent:

| config | DeepSeek set | Logistic set | DeepSeek answer | Logistic answer |
|---|---:|---:|---:|---:|
| base | **3.1542** | 3.8767 | 5.0132 | 4.7313 |
| symmetric | **3.1542** | 3.8767 | 4.9780 | 5.0661 |
| over-averse | **3.1542** | 3.8767 | 5.2423 | 4.3172 |
| under-averse | **3.1542** | 3.8767 | 4.9780 | 6.8282 |
| cheap abstain | **3.1542** | 3.8767 | 5.0132 | 4.7313 |
| expensive abstain | **3.1542** | 3.8767 | 5.0132 | 4.7313 |
| narrow width | 1.2907 | **0.9956** | 5.0132 | 4.7313 |
| wide width | **5.6388** | 7.7181 | 5.0132 | 4.7313 |
| alpha=.05 | **3.1542** | 3.9912 | 5.0132 | 4.7313 |
| alpha=.20 | **3.1542** | 3.4361 | 5.0132 | 4.7313 |

DeepSeek's set wins 9 of 10 configurations and loses under the narrow width
penalty, where coverage matters more than set size. This is the clearest LLM
result in the project: the model is a **better adjudicator than a point
predictor** on this task, and the ranking depends on the loss. It supports the
paper's output-interface argument and cautions against evaluating an LLM only
as a point classifier.

## Prompt iteration and bad-case fixes

1. **Reasoning-token exhaustion (fixed).** The endpoint returns a `thinking`
   block; with a 260-token budget the model spent the entire budget on
   reasoning and returned no text. Fix: disable thinking or give a bounded
   thinking budget; the primary run uses `thinking: {"type": "disabled"}`.
2. **Parse-failure reporting (fixed).** A repair call could return valid JSON
   while the record still carried `error=parse_failure`; the error is now
   cleared after a successful repair.
3. **Within-range anchoring (partially fixed).** The v1/v2 models defaulted to
   `within` because the current-quarter value was inside the range. v3 adds:
   - an explicit statement that the final actual is not constrained to the
     range and that `within` is not a default;
   - the company's historical mean surprise versus the guidance midpoint;
   - three dev examples showing that a current value at the range floor can
     end below, within, or above.
   On the 50-claim pilot, point accuracy rose from .30 (v1) to .40 (v3), but
   the model still rarely predicts `below` and remains below the logistic
   model. The credible-set mode is the productive use of the LLM.
4. **Thinking budget.** Enabling thinking with 1,024 tokens did not improve
   point accuracy or the hybrid, and increased latency; the primary runs
   disable it.

## DeepSeek as the framework backbone

DeepSeek is not a separate method. It supplies a label, confidence, and a
credible set; the same decision policies are applied to it and to the logistic
backbone. The DeepSeek planner uses a confidence-weighted belief over the
reported credible set. On 227 verified claims:

| config | DS answer | DS set | DS planner | Logistic answer | Logistic set | Logistic planner |
|---|---:|---:|---:|---:|---:|---:|
| base | 5.0132 | 3.1542 | **3.1013** | 4.7313 | 3.8767 | 3.9824 |
| symmetric | 4.9780 | 3.1542 | **3.1013** | 5.0661 | 3.8767 | 3.8855 |
| over-averse | 5.2423 | 3.1542 | **3.1013** | 4.3172 | 3.8767 | 4.0132 |
| under-averse | 4.9780 | 3.1542 | **3.1233** | 6.8282 | 3.8767 | 3.5683 |
| cheap abstain | 5.0132 | 3.1542 | 2.5110 | 4.7313 | 3.8767 | **1.9824** |
| expensive abstain | 5.0132 | 3.1542 | **3.1013** | 4.7313 | 3.8767 | 3.9824 |
| narrow width | 5.0132 | 1.2907 | 1.3172 | 4.7313 | **0.9956** | **0.9956** |
| wide width | 5.0132 | 5.6388 | **4.5419** | 4.7313 | 7.7181 | 5.1101 |
| alpha=.05 | 5.0132 | 3.1542 | **3.1013** | 4.7313 | 3.9912 | 4.0617 |
| alpha=.20 | 5.0132 | 3.1542 | **3.1013** | 4.7313 | 3.4361 | 3.5507 |

With the same planner, the DeepSeek backbone has lower loss in **8 of 10**
configurations (base 3.101 vs 3.982; maximum 4.542 vs 5.110). The logistic
backbone remains better under cheap abstention and the narrow width penalty.
The DeepSeek backbone's advantage is not its point label; it is the combination
of a set-valued action and the planner. This is the intended framing: DeepSeek
is the base model, and the decision interface is the method.

## Verdict

- **DeepSeek point answer:** negative vs the point-in-time logistic model on
  balanced accuracy and macro F1. Report it as a baseline and as one output
  mode of the backbone, not as the main method.
- **DeepSeek credible set:** positive and decision-relevant; it beats the
  logistic credible set under most loss configurations, but with lower
  coverage and a loss-dependent ranking.
- **DeepSeek as backbone + planner:** with the same planner, the DeepSeek
  backbone has lower loss than the logistic backbone in 8/10 configurations
  (base 3.101 vs 3.982; max 4.542 vs 5.110), losing cheap abstention and the
  narrow width penalty. This is the framework instantiation, not a separate
  DeepSeek method.
- **Contamination:** unresolved; the masked prompt is a partial check, not a
  guarantee. Do not present LLM numbers as clean point-in-time evidence.

## Artifacts

- `scripts/deepseek_client.py`
- `scripts/run_deepseek_baselines.py`
- `scripts/analyze_deepseek_adjudication.py`
- `paper/results/deepseek_pilot*/`
- `paper/results/deepseek_full_point_selective/`
- `paper/results/deepseek_full_verification/`
- `paper/results/deepseek_full_selfconsistency/`
- `paper/results/deepseek_full_hybrid/`
- `paper/results/deepseek_adjudication/`
