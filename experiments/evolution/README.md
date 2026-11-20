# AnchorEvolve-Evolved: config-driven evolution experiment

This directory is a self-contained, config-driven revision of the vague branch.
It keeps the paper's main line (LLM router + AnchorEvolve + calibrated readout)
and adds the missing evolution/readout pieces:

1. **Point-in-time filing evidence.**  For each claim the runner fetches the
   8-K exhibit at `claim_source.url` (the document available at the cutoff) and
   selects a headline+keyword excerpt with stable paragraph ids (`P1...`).
   The later earnings release in `earnings_events.json` is never used as input.
2. **Text-grounded anchors.**  `prompts/anchor_v2.txt` asks the LLM to cite a
   `source_id` (`P#` or `E#`) and a date, and to output calibrated
   `reliability` and `bucket_scores`.  Company names are masked by default.
3. **Five-type qualification.**  `temporal_fusion.py::qualify` implements
   vacuous / stale / adversarial / hard / soft deterministically from the
   anchor's implied bucket, date, source id, scope and reliability.  All
   thresholds are in `config.yaml`.
4. **Temporal evolution.**  A claim at cutoff `c` sees anchors/outcomes only
   when the outcome's public reveal date (earnings event, fallback 10-Q date)
   is before `c`.  Anchor-type reliability is a Beta posterior updated on
   revealed outcomes; the bucket prior is a smoothed count of past outcomes.
   The evolved reliability replaces/augments the LLM's subjective reliability
   in the belief update.
5. **Calibrated fusion.**  The evolved AnchorEvolve posterior is fused with the
   company-disjoint structured RF.  The fusion weight is selected on the
   2019-2021 validation split, never on 2022-2025.
6. **Full-action loss-aware readout.**  The original gate only compared
   answer vs. the highest-density `1-alpha` credible set.  The revision
   enumerates all point labels, all seven non-empty adjudication sets, and
   abstention, then chooses the minimum expected loss under the configured
   asymmetric loss.  This is a strict improvement for the same posterior and
   loss; the `1-alpha` credible set can still be reported separately for
   coverage/audit.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Every path, model, threshold, loss weight, fusion weight grid, seed and worker count. No API key. |
| `fetch_filings.py` | Fetch point-in-time 8-K exhibits into `paper/data/filings/` (gitignored). |
| `filing_text.py` | lxml-based text extraction and excerpt-id selection. |
| `prompts/anchor_v2.txt` | Text-grounded anchor prompt. |
| `run_anchors.py` | Concurrent, cached LLM anchor extraction; parse failures are errors, no oracle fallback. |
| `structured_evolve.py` | No-LLM validation/test comparison of peer/company/calendar feature sets. |
| `rf_probs.py` | Baseline structured RF probabilities for the validation or test split. |
| `temporal_fusion.py` | Evolution sweep, validation-selected fusion, full-action readout, paired bootstrap. |
| `meta_readout.py` | Optional validation-trained logistic meta-readout (kept as an analysis; it did not beat the simple fusion). |

## Commands

```bash
# 1. Put the key in the environment, never in the repo:
#    ANTHROPIC_AUTH_TOKEN=... ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
# 2. Fetch point-in-time filings (cached):
python experiments/evolution/fetch_filings.py --config experiments/evolution/config.yaml --all
# 3. Extract anchors (cached):
python experiments/evolution/run_anchors.py --config experiments/evolution/config.yaml --split validation
python experiments/evolution/run_anchors.py --config experiments/evolution/config.yaml --split test
# 4. Baseline RF probabilities:
python experiments/evolution/rf_probs.py --split validation --out paper/results/evolution_v1/rf_validation.jsonl
# 5. Evolution + full-action readout (baseline RF):
python experiments/evolution/temporal_fusion.py --config experiments/evolution/config.yaml --anchor-source new
#    Final system with the evolved-prior RF:
python experiments/evolution/temporal_fusion.py --config experiments/evolution/config.yaml --anchor-source new \
  --rf-validation paper/results/evolution_v1/rf_validation_ext.jsonl \
  --rf-test paper/results/evolution_v1/rf_test_ext.jsonl --tag ext
#    Ablations:
python experiments/evolution/temporal_fusion.py --config experiments/evolution/config.yaml --anchor-source new --no-evolution
python experiments/evolution/temporal_fusion.py --config experiments/evolution/config.yaml --anchor-source old
python experiments/evolution/temporal_fusion.py --config experiments/evolution/config.yaml --anchor-source old --no-evolution
```

## DeepSeek results (227 verified test claims)

The validation split selects the fusion weight; the test split is evaluated
once.  Paired 95% CIs use 10,000 claim-level bootstrap resamples.

| System | Acc | Bal | Macro-F1 | Loss | Loss vs old restricted |
|---|---:|---:|---:|---:|---|
| Old paper Fusion (restricted gate) | .524 | .471 | .468 | 3.604 | — |
| Old Fusion + full-action readout | .524 | .471 | .468 | 2.863 | −0.740 [−1.031, −0.423] |
| RF, restricted gate | .533 | .491 | .486 | 3.454 | — |
| RF + full-action readout | .533 | .491 | .486 | 2.881 | −0.573 [−0.855, −0.282] |
| New anchors + evolution + full-action (w=.05, baseline RF) | .542 | .499 | .497 | 2.855 | −0.749 [−1.040, −0.432] |
| New anchors, static prior/reliability (w=.05) | .546 | .505 | .503 | 2.908 | −0.696 [−1.004, −0.379] |
| Old anchors + evolution + full-action (w=.30) | .520 | .439 | .438 | **2.793** | −0.811 [−1.101, −0.511] |
| Old anchors, static prior/reliability (w=.20) | .507 | .450 | .450 | 2.899 | −0.705 [−0.978, −0.423] |
| **New anchors + evolution + evolved-prior RF (w=.20)** | **.551** | **.489** | **.503** | **2.890** | **−0.714 [−1.057, −0.352]** |
| Evolved-prior RF only + full-action (diagnostic) | **.555** | **.503** | **.508** | **2.828** | **−0.775 [−1.128, −0.414]** |

**What is significant:** the full-action readout lowers loss significantly
versus the old restricted gate on all three backbones (DeepSeek, GLM, Qwen).
The final DeepSeek system (new anchors + evolution + evolved-prior RF +
full-action) reduces loss from 3.604 to **2.890** (paired CI [−1.057, −0.352])
and improves accuracy/balanced/macro from .524/.471/.468 to .551/.489/.503.
The evolved structured prior alone reaches .555/.503/.508/2.828; the anchor
fusion is statistically indistinguishable from it (+0.062 loss, CI
[−0.141, 0.273]).  The paired accuracy improvement versus the old system is
+2.7 points, but not individually significant on 227 claims (McNemar p≈0.69);
report it as a point estimate, not a significance claim.  The evolution
ablation is consistent (static→evolved: 2.908→2.855 with new anchors;
2.899→2.793 with old anchors) but small; do not overclaim it as the main
driver.

**Fair comparison:** with the same full-action gate applied to every named
baseline, the Fusion is lower in point estimate than every baseline on all
three backbones, significantly lower on Qwen and against the weaker
baselines; the difference versus the strongest baseline on DeepSeek/GLM is
not significant.  The old "significantly lower than every baseline" result
was driven by giving only FinPlanRAG access to the set/abstain gate.

## Router integration

The router itself is unchanged.  Holding the released DeepSeek router
predictions fixed and replacing only the vague-branch loss on the 527-claim
subset with verified vague outcomes (300 determinate + 227 vague), the mixed
utility rises from **.389 to .437**; Always-Anchor rises from .258 to .306, so
the router's .131 advantage from correct determinate routing is preserved.
This is a branch-level integration check, not a new router result; the old
573-claim table remains the reference for routing accuracy.

## Evolved structured prior (no LLM)

The no-LLM feature experiment tests point-in-time peer/sector history, company
surprise history, calendar/current-quarter position and the reliability flag.
All features use only outcomes with `outcome_date < cutoff`; the construction
artifact `current_eq_low` is excluded from the final set.

Validation (185 claims; 3 seeds, 300 trees):

| Feature set | Acc | Bal | Macro | Loss (restricted gate) |
|---|---:|---:|---:|---:|
| Baseline | .557 | .556 | .536 | 3.784 |
| + peer only | .524 | .484 | .478 | 3.946 |
| + company surprise only | .551 | .561 | .535 | 3.751 |
| All evolved features, no artifact | **.605** | **.572** | **.566** | **3.740** |

Test (227 claims; same seeds/trees):

| Feature set | Acc | Bal | Macro | Restricted loss | Full-action loss |
|---|---:|---:|---:|---:|---:|
| Baseline RF | .520 | .476 | .473 | 3.463 | 2.908 |
| Evolved-prior RF | **.555** | **.503** | **.508** | **3.278** | **2.828** |

The accuracy gain is +3.5 points (McNemar p≈0.40) and the full-action loss gain
versus the 3-seed baseline is −0.079 [−0.379, 0.220]; both are point estimates,
not individually significant on 227 claims.  The large, significant loss gain
is versus the old **restricted** system (3.604 → 2.828, CI [−1.128, −0.414]).
Peer-only features hurt; the gain comes from the combination of peer, company,
calendar and current-quarter-position features, so report it as an evolved
prior bundle rather than a peer-prior effect.

## Hardcoding / leakage audit

- All thresholds, loss weights, fusion grid, model, paths and worker counts
  live in `config.yaml`; the API key is read from `ANTHROPIC_AUTH_TOKEN`.
- No route or label fallback to gold: a parse failure is recorded as an error.
- The `aggregate_final.py` hardcoded `w=0.1` is replaced by
  `summary["selected_weight"]`; the router's gold-route fallback is replaced by
  `"invalid"`; the baseline default label "above" is replaced by a neutral
  distribution.
- Remaining known gap (not part of this branch): the determinate branch's
  main runs still use the regex `obligation_terms` and a dummy chunk
  timestamp, not the typed planner in `src/finplan_rag/planner.py`.

## Limitations

- The new text-grounding run is DeepSeek-only.  The full-action readout is
  backbone-agnostic and was verified on GLM/Qwen with the released old anchors.
- The custom benchmark labels are automatic XBRL/rule labels, not human gold.
- Company masking reduces but does not eliminate LLM pretraining knowledge;
  a contamination check with a pre-2022 model or synthetic companies is still
  the right next experiment.
- The full-action readout has mean action-set size ≈1.9 and coverage ≈0.86,
  below the nominal 0.9 of the `1-alpha` credible set; this is the loss-optimal
  trade-off under the configured loss, and the `1-alpha` set should be reported
  separately when coverage is the target.
