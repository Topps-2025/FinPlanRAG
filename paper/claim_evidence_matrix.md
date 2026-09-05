# Claim--evidence matrix

This matrix follows the SQCAD evidence discipline. `measured` means produced by
the named deterministic contract; `proved` means valid only under the written
assumptions; `not measured` is not silently upgraded to a result.

| Claim | Evidence | Status | Boundary / challenge |
|---|---|---|---|
| A scalar relevance score can be insufficient for a stopping decision | Score-insufficiency proposition in both drafts; two observationally equivalent worlds | proved | Requires opposite optimal actions within one score fiber; does not prove every ranker fails |
| Point-in-time filtering prevents future documents in the emitted context | Point-in-time theorem plus `test_core.py` and `test_controlled_planning.py` | proved / measured | Assumes timestamps and cutoff are correctly supplied |
| Obligation state improves closure and query efficiency in the controlled task | `controlled_results.json`, SHA-256 in `paper/results/README.md` | measured | Synthetic hidden worlds; not real financial QA or SOTA |
| Counterevidence reduces overclaiming and improves limiting-evidence recall | `finplan_no_counterevidence` paired ablation | measured | It also lowers closure and utility under the declared cost contract |
| Temporal filtering changes the leakage boundary | `finplan_no_time` paired ablation; future leak 0.502 vs 0.000 | measured | Does not quantify real-world revision prevalence |
| Budget 10 is an operating point rather than a universal optimum | `budget_sensitivity.json` over budgets 1--16 | measured | Utility depends on the predeclared query price |
| The tool can be used by code agents | OpenAI-compatible schema, JSONL stdio process, adapter tests | measured | A native MCP transport wrapper remains deployment-specific |
| Closure equals answer correctness | No supporting evidence | explicitly false | Reader accuracy must be evaluated separately on frozen filing data |
