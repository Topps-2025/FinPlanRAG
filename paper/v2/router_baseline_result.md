# Real pre-hoc router baseline (2026-11-02)

**Status:** working experiment. This is the missing real-data router baseline
for the v2 decision-interface claim.

## Protocol

- Content backend: the same point-in-time logistic model used by the planner.
- Train the backend on dev (2010--2018), tune on validation (2019--2021), and
  evaluate once on test (2022--2025).
- Router features: guidance width and prior-outcome count, available at the
  cutoff. The width router chooses adjudicate if width exceeds a threshold
  selected on validation; the oracle-mode router is a logistic classifier
  trained on validation with the ex-post better mode (answer vs adjudicate) as
  the label. It sees features only at test time.
- Both routers choose the mode before seeing the content backend's posterior;
  the content of the chosen mode is supplied by the same logistic model.
- Loss is the v2 asymmetric decision loss at the base configuration; the table
  also reports the ten-configuration grid.

## Base configuration

| policy | mean decision loss |
|---|---:|
| always answer | 4.513 |
| always adjudicate | 3.604 |
| width router | 3.663 |
| oracle-mode router | **3.516** |
| planner (raw probabilities) | 3.927 |

The oracle-mode router is a strong baseline. It answers on 48 of 273 test
claims and adjudicates otherwise. It beats the planner at the base loss by
.411, and it also beats always-adjudicate.

## Ten loss configurations

| config | width router | oracle-mode router | always answer | always adjudicate | planner |
|---|---:|---:|---:|---:|---:|
| base | 3.6630 | **3.5165** | 4.5128 | 3.6044 | 3.9267 |
| symmetric | 3.6996 | **3.5092** | 4.7619 | 3.6044 | 3.6557 |
| over-averse | **3.6081** | 3.5678 | 4.2857 | 3.6044 | 3.8974 |
| under-averse | 3.8828 | 3.5275 | 6.1538 | 3.6044 | **3.3114** |
| cheap abstain | 3.6630 | 3.5165 | 4.5128 | 3.6044 | **2.0513** |
| expensive abstain | 3.6630 | **3.5165** | 4.5128 | 3.6044 | 3.9267 |
| narrow width | **1.2308** | 1.5659 | 4.5128 | 1.2308 | 1.2308 |
| wide width | 5.0549 | 6.2344 | 4.5128 | 6.7692 | **4.6190** |
| alpha=.05 | 3.8535 | **3.7289** | 4.5128 | 3.8535 | 4.2637 |
| alpha=.20 | **3.3993** | 3.4799 | 4.5128 | 3.3993 | 3.4432 |

## Reading

- The planner is **not uniformly better** than a learned pre-hoc router.
  Against the oracle-mode router, the two policies each win five
  configurations: the router wins base, symmetric, over-averse,
  expensive-abstain, and alpha=.05; the planner wins under-averse, cheap
  abstention, narrow width, wide width, and alpha=.20. Against the
  width-threshold router, the router wins five, the planner four, and narrow
  width is a tie.
- The planner's advantage is concentrated in regimes where the loss makes
  waiting or abstention valuable. A router cannot choose those actions; the
  planner can. At the base loss, a router with historical mode labels is a
  stronger policy than the planner.
- The router uses ex-post mode labels on validation, so it is a strong
  supervised baseline, not a strawman. It is the right comparison for the
  paper's P3 claim.
- The paper should report this as a boundary: the planner's case is
  regime-dependent, not a uniform method win.

## Implication for the paper

The current claim "a planner is a better default" needs qualification. The
defensible claim is:

> A fixed output mode is fragile across loss regimes; a learned router is a
> strong baseline at the base loss, while the planner is better when the loss
> makes waiting or abstention valuable. The interface matters, but no single
> policy dominates.

This is consistent with the benchmark's main finding and avoids overclaiming.

## Artifacts

- `scripts/run_router_baseline.py`
- `paper/results/router_baseline/summary.json`
- `paper/results/router_baseline/report.md`
