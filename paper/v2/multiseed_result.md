# Multi-seed result: controlled mode-switch planner (20 seeds)

**Status:** confirmatory run, 2026-10-31. Pre-registered protocol:
`multiseed_protocol.json`. Seeds 1-20. Data: v4 test split (n=273), controlled
preliminary signal. The 3-seed run was an implementation smoke test only.

**Primary metric:** top-1 accuracy at cost budget 1.0 on the controlled MA
subset (preliminary precision 0.9). This is an accuracy-under-cost metric; it
does not use the asymmetric decision-loss weights.

## Primary result

| policy | top-1 accuracy @ budget 1.0 | coverage | set size | ECE | Brier | log loss | selective@50 | AURC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **planner_preliminary** | **.8986 [.8855, .9117]** | .899 | 1.00 | .046 | .182 | .380 | .928 | .069 |
| planner_no_evidence (pre-hoc cost router) | .4695 [.4576, .4815] | 1.000 | 1.00 | .000 | .000 | .000 | 1.000 | .000 |
| always_wait | .4695 | 1.000 | 1.00 | .000 | .000 | .000 | 1.000 | .000 |
| always_answer | .4695 | .470 | 1.00 | .092 | .620 | 1.008 | .522 | .484 |
| always_adjudicate | .4695 | 1.000 | 3.00 | .092 | .620 | 1.008 | .522 | .484 |

**Paired difference (planner_preliminary - planner_no_evidence, top-1 accuracy
at budget 1.0):**

- mean **+0.4290**
- 95% CI **[0.4110, 0.4471]**
- positive seeds **20/20**

## Cost-accuracy frontier

| budget | planner_preliminary | planner_no_evidence | always_wait | always_answer | always_adjudicate |
|---:|---:|---:|---:|---:|---:|
| 0.0 | .4695 | .4695 | .4695 | .4695 | .4695 |
| 0.5 | .4695 | .4695 | .4695 | .4695 | .4695 |
| 1.0 | **.8986** | .4695 | .4695 | .4695 | .4695 |
| 1.5 | **.8986** | .4695 | .4695 | .4695 | .4695 |
| 2.0 | **.8986** | .4695 | .4695 | .4695 | .4695 |
| 3.0 | **.8986** | .8982 | .8982 | .4695 | .4695 |

The planner **weakly dominates** the pre-hoc cost-aware router on the frontier
and strictly dominates at budgets 1.0-2.0. At budget 3.0 the two are tied
(.8986 vs .8982). The planner's advantage is cost timing, not unbudgeted
accuracy: `planner_no_evidence` reaches 1.0 top-1 accuracy if it waits, but at
cost ~2.98; the planner reaches .899 at cost 0.8.

## Other metrics

- ECE .046, Brier .182, log loss .380: the planner's belief is substantially
  better calibrated than the fixed adjudication baseline (.092/.620/1.008).
- Selective accuracy@50 .928, AURC .069: much better than always-answer
  (.522/.484).
- Mean credible-set size 1.00: the planner's outputs are point answers after
  the preliminary signal, not wide credible sets.
- Leakage 0.

## Real v4 control (no preliminary signal)

| policy | top-1 acc | macro F1 | coverage | ECE | acc@budget 0 | acc@budget 1.0 | acc@budget 3.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_answer | .462 | .211 | .462 | .100 | .462 | .462 | .462 |
| always_adjudicate | .462 | .211 | 1.000 | .100 | .462 | .462 | .462 |
| always_wait | 1.000 | 1.000 | 1.000 | .000 | .462 | .462 | .898 |
| planner_no_wait | .462 | .211 | .971 | .164 | .462 | .462 | .462 |
| planner_no_evidence | 1.000 | 1.000 | 1.000 | .000 | .462 | .462 | .898 |
| planner | 1.000 | 1.000 | 1.000 | .000 | .462 | .462 | .898 |
| planner_preliminary | 1.000 | 1.000 | 1.000 | .000 | .462 | .462 | .898 |

**Real v4 has no positive result.** At budget 1.0 every policy is at .462; at
budget 3.0 the wait-based policies reach .898, but always_wait does the same.
The new planner does not add value on the real v4 evidence.

## Verdict

- **Positive controlled mechanism:** with 20 seeds and a pre-registered
  budget-1.0 metric, `planner_preliminary` beats the pre-hoc cost-aware router
  by +0.429 top-1 accuracy (95% CI [0.411, 0.447], 20/20 seeds), and weakly
  dominates the cost-accuracy frontier.
- **No real-data method positive:** the real v4 control does not reproduce the
  gain. The preliminary signal is synthetic.
- **What this supports:** the planner can observe evidence and switch mode at
  low cost; this is the P3 mechanism result.
- **What this does not support:** a claim that the framework beats strong
  baselines on real financial data. That requires the real evidence-rich
  benchmark.

## Artifacts

- `src/finplan_rag/decision/run_multiseed.py`
- `src/finplan_rag/decision/metrics.py`
- `paper/results/v2_multiseed/controlled.json`
- `paper/results/v2_multiseed/real.json`
- `paper/results/v2_multiseed/report.md`
- Protocol: `paper/v2/multiseed_protocol.json`
