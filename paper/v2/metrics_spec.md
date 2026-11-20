# Metrics specification for the v2 decision pilot

**Status:** specification, 2026-10-31. The pilot implements these metrics in
`src/finplan_rag/decision/loss.py`.

## Decision loss

For a three-bucket hypothesis space `{below, within, above}`, a point answer
`a`, a credible set `R`, and the gold bucket `y`:

\[
L(\text{answer}(a), y)=
\begin{cases}
0 & a=y\\
\ell_{\text{over}} & a=\text{above}, y=\text{below}\\
\ell_{\text{under}} & a=\text{below}, y=\text{above}\\
\ell_{\text{wrong}} & \text{otherwise}
\end{cases}
\]

\[
L(\text{adjudicate}(R), y)=
\mathbf{1}[y\notin R]\,\ell_{\text{miss}}
+\ell_{\text{width}}\max(0,|R|-1)
\]

\[
L(\text{abstain}, y)=\ell_{\text{abstain}}
\]

Default parameters: \(\ell_{\text{over}}=12\), \(\ell_{\text{under}}=8\),
\(\ell_{\text{wrong}}=10\), \(\ell_{\text{miss}}=8\), \(\ell_{\text{width}}=2\),
\(\ell_{\text{abstain}}=5\), \(\alpha=0.1\).

The asymmetry reflects the financial cost of overclaiming. The width penalty
prices an uninformative adjudication. All parameters are exposed in
`LossConfig` and a sensitivity run is required.

## Metrics

| Metric | Definition |
|---|---|
| `decision_loss` | Mean \(L\) over cases |
| `evidence_cost` | Mean sum of acquired evidence costs |
| `delay_cost` | Mean \(\gamma \times\) realized delay days |
| `total_loss` | `decision_loss + evidence_cost + delay_cost` |
| `coverage` | Fraction where the answer is correct or the gold is in the credible set |
| `answer_accuracy` | Accuracy among cases where the policy answers |
| `adjudication_coverage` | Fraction where the gold is in the credible set among adjudications |
| `wait_rate` | Fraction of cases with a positive decision delay |
| `preliminary_rate` | Fraction of cases using the controlled preliminary signal |
| `mean_decision_days` | Mean realized delay among waited cases (including zeros) |
| `leakage_rate` | Fraction of outputs using evidence after its availability time |
| `ece` | Expected calibration error over point answers (10 bins) |

## Cost-loss frontier

The pilot does not report a single budget point. It varies:

- delay cost \(\gamma\in\{0,0.001,0.005,0.01,0.02,0.05,0.1\}\);
- outcome-filing cost \(\in\{1,2,4\}\);
- future-signal precision \(\in\{1.0,0.8,0.6\}\).

Each configuration reports total loss, coverage, wait rate, and mean decision
days. The frontier is the primary result; a single configuration is not.

## Paired bootstrap

For the main and low-cost configurations, the pilot compares the planner with
each baseline using a paired bootstrap over cases (20,000 resamples, seed
20261031). The reported quantity is the mean difference in total loss and its
95% percentile interval. A negative difference means the planner is better.

## Sensitivity

- `loss_sensitivity.json` varies the loss parameters: default, symmetric,
  more overclaim, high/low width, high abstain.
- `precision_sensitivity` varies the future-signal precision.
- `signal_strength_sensitivity` is a controlled diagnostic, not a real-data
  result; it amplifies or flattens the current-quarter signal by reliability.

## Boundaries

- The loss is a modeling choice, not a universal financial utility.
- Coverage and answer accuracy are not comparable across modes without the
  decision loss.
- ECE is computed over point answers only; adjudications and abstentions are
  excluded.
- A single mean is not a result; report the frontier and the sensitivity grid.
- No metric is an answer-level financial QA score. The pilot is decision-level.
