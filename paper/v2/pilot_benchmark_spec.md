# Pilot mixed benchmark specification

**Status:** specification, 2026-10-31. This is not a dataset. The examples below
are illustrative and must be replaced or verified against primary sources
before use.

**Purpose:** turn `experiment_plan.md` E1 into a concrete pilot that can be
annotated and run before the full benchmark exists.

---

## 1. Pilot scale

Target: **100 queries**, 25 per type.

| Type | Count | Source |
|---|---:|---|
| Determinate | 25 | LOFin / FinGLM |
| Vague | 25 | v2 production claims / v4 revenue guidance |
| Mixed | 25 | constructed from v4 guidance + outcome |
| Mode-ambiguous | 25 | constructed from v2/v4 evidence pools |

If 100 is too ambitious, start with **40** (10 per type) and report the pilot as
a feasibility study, not a benchmark.

---

## 2. Query types and illustrative examples

> All examples are illustrative. Company/claim facts must be checked against
> the primary filing before inclusion.

### 2.1 Determinate (D)

- *“What was the company's FY2023 revenue?”*
- *“What was the FY2024 first-quarter operating cash flow?”*
- *“How many units were shipped in fiscal 2023?”*

Expected output: precise answer + citation. Evidence: a filing with a point
value. Gold: answer span/number.

### 2.2 Vague (V)

- *“Will the Ohio fab start production in 2025?”*
- *“Is the new line expected to reach mass production in H2?”*
- *“Will the 2026 margin guidance be met?”*

Expected output: credible set / qualified judgment. Evidence: management calls,
permits, supplier disclosures, capacity anchors. Gold: realized outcome (timing,
stage, or below/within/above).

### 2.3 Mixed (M)

- *“Did the company meet its H2 revenue guidance?”*
- *“Was the 2024 capacity target achieved?”*

Expected output: first retrieve the guidance/target (determinate sub-query),
then adjudicate the outcome against it (vague sub-query). Gold: retrieved
guidance + outcome judgment.

### 2.4 Mode-ambiguous (MA)

These are the diagnostic queries. The optimal mode changes after evidence.

- *“Will the plant start production in 2025?”*
  - Before evidence: vague (management claim only).
  - After a permit hard anchor: precise (permit date rules out 2025).
  - Switch trigger: permit filing.
- *“What was FY2024 revenue?”*
  - Before evidence: determinate-looking.
  - After a filing with only a range: adjudication (range, not point).
  - Switch trigger: filing with a range rather than a point.
- *“Is the 2026 margin guidance credible?”*
  - Before evidence: vague.
  - After a supplier cost disclosure: precise enough to answer a narrower
    question (e.g., “margin likely below the low end”).
  - Switch trigger: supplier disclosure.

Each MA query must record `mode_before`, `mode_after`, and `switch_evidence`.

---

## 3. Annotation schema

```json
{
  "query_id": "MA-001",
  "query_text": "...",
  "source": "v2|v4|LOFin|FinGLM|constructed",
  "type": "D|V|M|MA",
  "cutoff_given": "2024-06-30 or null",
  "evidence_pool": ["doc_id_1", "doc_id_2"],
  "evidence_dates": {"doc_id_1": "2024-05-01"},
  "answer_gold": "number/span or null",
  "outcome_gold": "timing/stage/bucket or null",
  "outcome_date": "date or null",
  "oracle_mode_before": "precise|adjudicate|abstain",
  "oracle_mode_after": "precise|adjudicate|abstain",
  "switch_evidence": "doc_id or null",
  "loss_params": {"over": 5.0, "under": 1.0, "wrong": 10.0, "abstain": 1.0},
  "annotator_notes": "..."
}
```

### Required fields

- `type`: D / V / M / MA.
- `evidence_pool`: every admissible document with its date.
- `cutoff_given`: null if the decision time is endogenous.
- `answer_gold` / `outcome_gold`: at least one must be present.
- `oracle_mode_before` / `oracle_mode_after`: for MA queries; for D/V/M, the
  before/after can be identical.
- `switch_evidence`: required for MA queries.

---

## 4. Construction rules for mode-ambiguous queries

1. Start from an existing v2/v4 claim with a real evidence pool.
2. Select two evidence items with different arrival times/costs:
   - \(e_{\text{fast}}\): cheap, weak (management call, news);
   - \(e_{\text{slow}}\): costly, strong (permit, supplier, filing).
3. Define the pre-evidence state \(S_0\) and post-evidence state \(S_1\).
4. Verify that the oracle mode differs between \(S_0\) and \(S_1\) under the
   declared loss.
5. Record the switch trigger and the expected loss if the planner does not
   switch.
6. Reject the query if the mode does not actually change; a false MA query
   weakens the benchmark.

**Anti-gaming check:** an MA query must not be constructed by withholding
information that a normal retriever would always fetch. The planner's advantage
must come from the decision, not from an artificial evidence ordering.

---

## 5. Inter-annotator agreement

- Two annotators label type and oracle mode independently.
- Report Cohen's \(\kappa\) for type and oracle mode.
- Target: \(\kappa\ge 0.7\); below that, refine the guidelines and re-annotate.
- Disagreements are adjudicated by a third annotator; record the reason.
- For MA queries, also report agreement on `switch_evidence`.

---

## 6. Pilot checks

Before scaling to 100 queries, run these checks:

1. **Type balance:** at least 10 per type.
2. **MA switch rate:** at least 50% of MA queries actually change mode.
3. **Evidence coverage:** every query has at least one admissible evidence item
   before the decision time.
4. **Leakage:** no evidence with \(a(e)>t\) is used before \(t\).
5. **Gold availability:** D has answer gold; V/M has outcome gold; MA has both
   mode labels.
6. **Loss sensitivity:** the ranking of modes is stable under at least two
   reasonable loss parameterizations; if not, flag the query.

---

## 7. What the pilot is not

- It is not a representative sample of all financial claims.
- It is not human-verified until the annotation pass is complete.
- It is not a leaderboard; it is a diagnostic for the planner-vs-router
  question.
- It is not a substitute for the full benchmark; it is a feasibility check.
