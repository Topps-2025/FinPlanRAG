# Formal motivation: decision interface for heterogeneous financial queries

**Status:** research note, 2026-10-31. This note states the decision-theoretic
motivation for FinPlan-RAG v2. It is not the algorithm. Its purpose is to make
the research question falsifiable and to justify the design choices at the level
of the problem formulation.

**Scope note:** we do **not** assume a fixed cutoff and we do **not** assume a
fixed budget. The decision time \(t\), the evidence cost \(\lambda(e)\), and the
delay cost \(\gamma\) are part of the decision problem. Fixed cutoff and fixed
budget are special cases, used only as baselines.

---

## 1. Objects

Let \(\mathcal X\) be the state space. A query or claim \(q\) induces a set

\[
S(q)\subseteq \mathcal X.
\]

A piece of evidence \(e\) has

- an availability time \(a(e)\),
- an acquisition cost \(\lambda(e)\ge 0\),
- a constraint \(S(e)\subseteq \mathcal X\),
- a source/origin \(\mathrm{org}(e)\).

The agent chooses a decision time \(t\) and an information set
\(\mathcal F_t=\sigma(\{e:a(e)\le t\})\). Only evidence with \(a(e)\le t\) is
admissible at \(t\). Let

\[
V_t=\{e:\ a(e)\le t,\ e\text{ not retracted at }t\}
\]

be the admissible, unretracted evidence. The feasible set is

\[
S_t=S(q)\cap\bigcap_{e\in V_t}S(e).
\]

For a finite hypothesis space \(H_q=\{h_1,\dots,h_K\}\) (a partition of
\(\mathcal X\)), let \(b_t\) be a credal belief over \(H_q\). The agent's output
is chosen from

\[
A=\{\text{answer}(a),\ \text{adjudicate}(R),\ \text{abstain}\},
\]

where \(a\in\mathcal X\) is a point answer and \(R\subseteq\mathcal X\) is a
credible/feasible set. The agent may also take evidence actions

\[
\text{acquire}(e)\quad\text{or}\quad\text{wait}(\Delta),
\]

which advance time, consume cost, and update \(V_t\).

The objective is

\[
J(\pi)=\mathbb E\left[
L(\text{output},x)
+\sum_{e\in \mathrm{acquired}}\lambda(e)
+\gamma\,(t-t_0)
\right],
\]

where \(L\) is a decision loss, \(t_0\) is the query start time, and
\(\gamma\ge 0\) is the per-unit delay cost. There is no hard budget constraint.
A fixed budget is the special case in which the planner is forced to stop after
a fixed number of actions or a fixed cost; a fixed cutoff is the special case in
which \(t\) is given and cannot be chosen.

---

## 2. Assumptions

The propositions below are stated under the following assumptions. They are
deliberately modest; the goal is a falsifiable motivation, not a deep theory.

1. \(H_q\) is finite and \(b_t\) is a probability distribution over \(H_q\).
2. The loss \(L\) is bounded and asymmetric: overclaim is at least as costly as
   underclaim, \(\ell_{\text{over}}\ge \ell_{\text{under}}\).
3. Evidence constraints are truthful: \(x\in S(e)\) whenever \(e\) is valid.
4. Likelihoods are either calibrated or the analysis is worst-case over
   posteriors consistent with \(S_t\).
5. Decision cells \(D_A\subseteq \mathcal X\) are known and measurable.
6. Cost functions \(\lambda(e)\) and delay cost \(\gamma\) are known to the
   planner; the analysis does not require them to be constant across queries.
7. The policy is \(\mathcal F_t\)-measurable at the chosen decision time \(t\).

---

## 3. Propositions

### P1. Point-answer insufficiency

**Statement.** If \(S_t\) is not a singleton and the loss is asymmetric, then
there is no point answer \(a\) that is minimax-optimal for every posterior
consistent with \(S_t\). A set-valued or qualified action can have strictly
lower worst-case risk.

**Sketch.** Consider two states \(x_1,x_2\in S_t\). A point answer \(a=x_1\)
incurs \(\ell_{\text{wrong}}\) on \(x_2\); \(a=x_2\) incurs
\(\ell_{\text{wrong}}\) on \(x_1\). A qualified action
\(R=\{x_1,x_2\}\) incurs a width/qualification cost but no wrong-answer cost.
For \(\ell_{\text{wrong}}\) sufficiently large relative to the width cost, the
set-valued action has lower worst-case risk. The argument extends to any
non-singleton \(S_t\) by taking two states in \(S_t\).

**Design implication.** The output contract must include adjudication and
abstention, not only point answers.

### P2. Relevance insufficiency

**Statement.** If a relevance score \(r(e)\) is constant on a fiber
\(\mathcal F_r\) containing evidence \(e,e'\) with \(S(e)\ne S(e')\), then
\(r\) is not a sufficient statistic for the Bayes action. Any policy that uses
only \(r\) has a worst-case loss gap.

**Sketch.** Construct two worlds with the same relevance observation but
different feasible sets after \(e\) and \(e'\). The Bayes action differs across
the worlds, but an \(r\)-only policy must take the same action. The worst-case
loss is at least the gap between the two Bayes risks.

**Design implication.** Evidence must be typed by its constraint content, not
only by a scalar relevance score.

### P3. Pre-hoc routing insufficiency

**Statement.** If the optimal output mode depends on evidence that is observed
only after an action, then any policy that chooses the mode before observing
that evidence is not optimal for all queries. A state-dependent planner can
improve.

**Sketch.** Construct two queries \(q_1,q_2\) with the same pre-evidence
observation. After evidence \(e\), the optimal mode of \(q_1\) is precise answer
and the optimal mode of \(q_2\) is adjudication. A pre-hoc router must assign the
same mode to both because it sees the same pre-evidence observation; a
state-dependent planner can assign different modes after \(e\). The worst-case
loss gap is positive whenever the loss difference between the two modes is
positive.

**Design implication.** The mode must be a decision after evidence, not a
query-level label. This is the core non-additive capability that distinguishes a
planner from a router.

### P4. Decision-sufficient stopping

**Statement.** Under a one-step (myopic) value-of-information analysis, the
optimal stopping time is the first \(t\) at which either

\[
S_t\subseteq D_A \quad\text{for some output }A,
\]

or

\[
\sup_{e}\ \mathrm{VoI}(e\mid b_t)\le \lambda(e)+\gamma\,\Delta t.
\]

Otherwise, the planner waits or acquires more evidence.

**Sketch.** This is the standard optimal-stopping/VoI argument: stop when the
expected reduction in decision loss from further evidence does not exceed its
cost, including delay. The myopic version is an approximation to the full POMDP;
it is exact when evidence actions are independent and there is no option value
from future evidence beyond the next step.

**Design implication.** Stopping is not a fixed top-\(k\) or a reader-confidence
threshold; it is a decision over \(S_t\), cost, and delay.

### P5. Independence and defeat

**Statement.** Correlated evidence is sub-additive: if \(e_1,e_2\) share an
origin, the information gain of \(e_2\) given \(e_1\) is at most its marginal
information gain. A monotone accumulator that treats sources as independent
overcounts support. A monotone accumulator that does not model retraction is
unsound under revision/restatement.

**Sketch.** The first part is the chain rule for mutual information,
\(I(X;E_1,E_2)=I(X;E_1)+I(X;E_2\mid E_1)\le I(X;E_1)+I(X;E_2)\), with equality
iff \(E_2\perp E_1\mid X\). The second part is a two-world construction: the same
supporting evidence is present in both worlds, but in one world it is retracted
by a later filing; a monotone policy overclaims in that world.

**Design implication.** The evidence state must include origin/independence and
defeat status, not only support.

### P6. Cutoff measurability

**Statement.** At a chosen decision time \(t\), a policy that uses evidence with
\(a(e)>t\) produces a feasible set that is not \(\mathcal F_t\)-measurable. Such
a policy is not a valid point-in-time analyst decision.

**Sketch.** By definition, \(\mathcal F_t\) is generated by evidence with
\(a(e)\le t\). Evidence with \(a(e)>t\) is not in the filtration; any output that
depends on it is not \(\mathcal F_t\)-measurable.

**Design implication.** Endogenous decision time is allowed, but the policy must
be measurable with respect to the filtration at the chosen time. Waiting is an
explicit action with a cost, not free lookahead.

### P7. Fixed-budget suboptimality

**Statement.** When the marginal VoI and marginal cost vary across queries, a
fixed top-\(k\) or fixed-budget stopping rule cannot be optimal for all queries.
An endogenous stopping/acquisition policy can improve the cost–loss frontier.

**Sketch.** Construct two queries: in \(q_1\), the first evidence item resolves
the decision, so stopping after one item is optimal; in \(q_2\), the first item
is uninformative and a second item is needed. A fixed \(k=1\) is suboptimal for
\(q_2\); a fixed \(k=2\) is wasteful for \(q_1\). No single fixed \(k\) is optimal
for both. An endogenous policy can stop at different times.

**Design implication.** The paper must not assume a fixed budget or a fixed
top-\(k\). Evaluation should report cost–loss frontiers, not a single budget
point.

---

## 4. Special cases

The formulation contains the existing interfaces as special cases:

| Special case | Restriction |
|---|---|
| Fixed-cutoff retrieval | \(t\) is given and cannot be chosen; only \(a(e)\le t\) evidence is used |
| Fixed-budget RAG | stop after \(k\) actions or cost \(B\); output mode fixed to answer |
| Claim verification | \(H_q\) is a claim-verdict space; output mode fixed to verdict; evidence given |
| Sufficiency/abstention | output mode fixed to answer/abstain; \(S_t\) collapsed to a scalar confidence |
| Pre-hoc router | mode chosen before evidence; no mid-retrieval switching |
| Point-in-time benchmark | \(t\) fixed; evaluation is a single cutoff |

This table is important for the paper: it shows that the new formulation is a
generalization, not an A+B combination.

---

## 5. Mapping to experiments

| Proposition | Experiment |
|---|---|
| P1 | Mixed benchmark; point-answer vs adjudication/abstention |
| P2 | Relevance-only baseline vs typed evidence |
| P3 | Pre-hoc router vs state-dependent planner |
| P4 | Stopping-rule ablation; cost–loss frontier |
| P5 | Independence/defeat ablation |
| P6 | Leakage audit; endogenous cutoff |
| P7 | Fixed budget/top-\(k\) vs endogenous cost |

Each experiment must report a negative-result criterion: what outcome would
falsify the proposition's relevance to the task.

---

## 6. What this note does not claim

- It does not claim that the propositions are deep or novel as mathematics.
- It does not claim that the planner is implemented or evaluated yet.
- It does not claim that the loss function is the only reasonable one; the
  paper must include a sensitivity analysis.
- It does not claim that the mixed benchmark exists; the benchmark is a design
  requirement, not a result.
- It does not claim that a fixed cutoff or fixed budget is wrong in every
  application; it claims that they are special cases, not the problem
  statement.

The role of the note is to make the motivation **falsifiable**: if the
propositions fail, or if the empirical gaps they predict do not appear, the
paper's framing should be revised.
