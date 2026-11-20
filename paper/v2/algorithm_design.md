# Algorithm design: conservative one-step decision planner

**Status:** pilot implementation, 2026-10-31. This document describes the
planner used in `src/finplan_rag/decision/policies.py`. It is a minimal,
auditable instantiation of the v2 decision interface, not the full method.

## 1. Scope

The planner operates on a decision-level case:

- a finite hypothesis space \(H=\{below,within,above\}\);
- evidence items with availability times, costs, source reliability, and
  payloads;
- a cutoff (the query time) and an endogenous decision time \(t\);
- a decision loss with asymmetric over/under-claim costs;
- a future outcome filing that becomes available after the cutoff.

The pilot uses the v4 revenue-guidance dataset. The planner is deliberately
small so that the decision interface, not a large model, is tested.

## 2. State

\[
s_t=(b_t,\ E_t,\ t,\ c_t)
\]

- \(b_t\): belief over \(H\);
- \(E_t\): evidence acquired so far;
- \(t\): current decision time;
- \(c_t\): accumulated evidence and delay cost.

## 3. Evidence and belief updates

- **Guidance** (\(e_g\)): available at cutoff; defines the range; cost 0.5.
- **Current-quarter anchor** (\(e_q\)): available at cutoff; cost 0.5; a weak
  signal of the outcome bucket.
- **Outcome filing** (\(e_a\)): available at the outcome date; cost 2.0; a
  noisy observation of the true bucket with precision \(\pi\).

Prior:

\[
b_0(h)\propto \text{count}_{\text{dev}}(h)+\epsilon.
\]

Current-quarter likelihood:

\[
\ell_q(h)=\text{normalize}\left(
\rho\,\tilde\ell(h)^{\tau(\rho)}
+(1-\rho)\,\frac{1}{|H|}
\right),
\qquad
\tau(\rho)=\max(0.2,\ 1+s(\rho-0.5)2),
\]

where \(\rho\) is the anchor reliability and \(s\) is the controlled signal
strength. The empirical table \(\tilde\ell\) is fitted on dev and blended with
the prior. The pilot sets \(s=0\) for real data; \(s=1\) is a controlled
diagnostic.

Posterior:

\[
b_t(h)\propto b_{t-1}(h)\,\ell_q(h).
\]

Outcome filing:

\[
P(o\mid y)=\begin{cases}\pi & o=y\\ (1-\pi)/(|H|-1) & o\ne y\end{cases}
\]

and the posterior is updated by the observed bucket.

## 4. Actions

- `acquire(guidance)`, `acquire(current_q)`, `acquire(actual)`;
- `wait(until the outcome filing)`;
- `answer(a)`, `adjudicate(R)`, `abstain`.

`R` is restricted to a credible set with posterior mass at least \(1-\alpha\),
so the adjudication action is a credible set, not an arbitrary subset.

## 5. Stop decision

For a belief \(b\), enumerate:

- point answers \(a\in H\);
- credible sets \(R\) with \(b(R)\ge 1-\alpha\);
- abstention.

Let

\[
S(b)=\min\left\{
\mathbb E_b[L(\text{answer}(a),y)],
\mathbb E_b[L(\text{adjudicate}(R),y)],
\ell_{\text{abstain}}
\right\}.
\]

Waiting for the outcome filing costs

\[
W(b)=\lambda_a+\gamma\,\hat d+\mathbb E_o\left[S(b\mid o)\right],
\]

where \(\lambda_a\) is the filing cost, \(\gamma\) the delay cost, \(\hat d\)
the company-historical expected delay, and the expectation is over the noisy
observation \(o\).

The planner waits unless stopping is better by a conservative margin \(m\):

\[
\text{wait} \iff W(b) < S(b)+m.
\]

The margin prevents stopping on a weak signal whose apparent advantage is
smaller than the model's calibration error. It is a design choice, not a
learned parameter.

## 6. Evidence plan

The planner compares two plans:

1. **none**: stop or wait with the prior;
2. **current**: acquire guidance and the current-quarter anchor (cost 1.0),
   update the belief, then stop or wait.

Each plan is scored by its expected total loss plus the conservative margin on
the stop branch. The cheaper plan is executed. This is a one-step
value-of-information approximation, not a full POMDP.

## 6.1 Preliminary plan (controlled P3 diagnostic)

The controlled diagnostic adds a preliminary evidence item:

- available at cutoff + 30 days, cost 0.5;
- half the cases have precision 0.9 and half 0.5; assignment is independent of
  the query features a pre-hoc router can see;
- after acquisition, the planner observes the bucket and precision and updates
  the belief;
- the planner stops at day 30 if the preliminary is informative, otherwise it
  waits for the final filing.

The plan selection compares `none`, `current`, and `preliminary` using an
expected precision of 0.7. The execution uses the observed precision. A
pre-hoc cost-aware router cannot observe the preliminary content before
choosing wait/stop; this is the controlled test for P3.

The preliminary signal is synthetic. It is a mechanism diagnostic, not a
real-data result.

## 7. Pseudocode

```text
input: case, prior, current-q likelihood, costs, loss, margin
plans = {none: (prior, cost 0)}
if guidance and current_q:
    plans["current"] = (posterior(prior, current_q), cost 1.0)
for each plan:
    value = stop_loss(belief) + plan_cost + margin
    if allow_wait:
        value = min(value, plan_cost + actual_cost
                    + gamma * expected_delay
                    + expected_loss_after_wait(belief))
choose the plan with the smallest value
execute:
    if wait is cheaper than stop + margin:
        advance t to the outcome date; observe the noisy bucket
        choose the best stop action on the posterior
    else:
        choose the best stop action on the current belief
return action, prediction/credible set, belief, costs, decision days, leakage
```

## 8. Complexity

- \(|H|=3\); at most 7 subsets, filtered by the credible-set constraint.
- Belief update: \(O(|H|)\).
- Stop decision: \(O(|H|^2)\).
- Wait decision: \(O(|H|^3)\) for the observation expectation.
- Plan selection: constant (two plans).

The pilot is CPU-only and deterministic.

## 9. Why this is not a router

- A router chooses the mode **before** evidence; the planner chooses after the
  belief update and can wait for a later filing.
- The planner's actions include evidence acquisition and waiting, not only
  answer/adjudicate.
- The planner's value is cost-dependent: it matches always-wait when waiting
  is cheap and stops when waiting is expensive. A fixed policy cannot.
- The conservative margin is a calibration device; it makes the planner prefer
  waiting when the signal is weak.

## 10. What the pilot shows

- The planner matches the best fixed policy in most cost regimes and beats
  no-wait, always-answer, always-adjudicate, fixed-budget, and pre-hoc router
  baselines. It is not uniformly better than always-wait: at the cost-loss
  crossover it can be slightly worse because the expected delay is estimated.
- The current-quarter anchor is not worth its cost on v4; the planner skips it.
  This is a negative result for that evidence type, not for the decision
  interface.
- When the future filing is noisy (precision 0.8/0.6), the planner stops
  waiting; when it is exact, it waits.
- The real-data case-level evidence-induced mode switching is not demonstrated
  on v4. The signal is too weak; the planner's decisions are cost thresholds
  rather than per-case evidence-driven switches.
- The **controlled preliminary diagnostic** does demonstrate the mechanism: on
  the MA subset (n=133), `planner_preliminary` 1.4617 vs the pre-hoc cost-aware
  router `planner_no_evidence` 2.9626, paired −1.5009 [−1.875, −1.072]. The
  planner acquires the preliminary signal and switches mode; the router cannot
  observe it. This supports P3 in a controlled setting, not on real data.

## 11. Full-experiment extensions

1. **Richer evidence**: hard anchors (permits, capacity, equipment lead time),
   multiple sources, independence/defeat, and a calibrated likelihood.
2. **Case-level mode switching**: a benchmark where the optimal mode changes
   after evidence, not only after a global cost threshold.
3. **A reader/determinate branch**: answer-level gold for determinate queries,
   with a fixed reader across modes.
4. **A full POMDP or offline policy learning**: the current planner is
   one-step; the full experiment can compare with a multi-step VoI policy.
5. **Human verification and annotation**: the current labels are silver.
