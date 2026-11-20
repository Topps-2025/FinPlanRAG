# FinPlanRAG / AnchorEvolve

This repository implements FinPlanRAG, a point-in-time framework for
forward-looking financial claims.  The current paper focuses on the
forward-looking branch only: a claim contract, three-layer anchor state,
five-type qualification, reveal-date reliability evolution, deterministic
numeric bridges, and a full-action loss-aware readout that returns a point
answer, an adjudication set, or an abstention.

The earlier obligation-aware planner in `src/finplan_rag/` is retained for
provenance.  Its router and determinate-branch results are not part of the
current paper's claims.

## Environment

- Python 3.10 or newer.
- Install the agent extras with `pip install -e ".[agent]"`.
- The DeepSeek key is read from `.env` or the environment.  Never write a key
  into a source file, config, log, audit artifact, or commit.
- The MCP wrapper loads `.env` from the repository root.  If `.env` is absent,
  export `ANTHROPIC_AUTH_TOKEN` before starting Codex.
- The optional model settings are `ANTHROPIC_BASE_URL`,
  `ANTHROPIC_MODEL`, and `ANTHROPIC_AUTH_TOKEN`.

## MCP interface

The Codex project config is `.codex/config.toml`.  It starts the same stdio
server as Claude Code:

```text
python scripts/run_anchor_mcp.py
```

Codex namespaces the tools as `mcp__finplan-anchor__<tool>`:

| Tool | Use |
|---|---|
| `anchor_ask` | Full adjudication of a natural-language forward-looking claim |
| `anchor_case` | Run a named case from `configs/anchor_cases/` |
| `anchor_search` | Retrieve dated web evidence only |
| `anchor_audit` | Read the full evidence-to-anchor-to-belief report |
| `anchor_contract` | Parse a question into a claim contract before running it |

Example:

```text
Use mcp__finplan-anchor__anchor_case with case_id "zhongji_2026" and cutoff
"2026-11-19".  Then read the audit and explain each accepted and rejected
anchor.  Do not invent an outcome.
```

If the tools are missing, run `codex mcp list` inside the repository.  Codex
loads project-local `.codex/config.toml` only when the project is trusted.
After changing the engine, prompts, or a case file, restart Codex so the MCP
process reloads the code.

## Working rules

- Treat every claim as a contract with company, target period, outcome space,
  cutoff, and evidence budget.
- Use only evidence whose publication date is no later than the cutoff.
- Keep the distinction between a company target, analyst consensus, a
  historical actual, and a model forecast.  A target is not a point forecast.
- Every accepted anchor must carry a source, quote, date, and qualification.
  If a quote or number cannot be verified, do not promote the anchor to hard.
- Numeric arithmetic belongs in `scripts/forecast_bridge.py`, not in prose.
- The full-action readout is the decision interface.  The point answer is not
  always the right action; an adjudication set or abstention can have lower
  expected loss.
- Do not change a prospective case's thresholds or labels after its outcome is
  known.
- Do not present FinDVer, Fin-Fact, DocFinQA, FinQA, TAT-QA, or FinanceBench as
  end-to-end forward-looking benchmarks.  They are component-level or control
  checks.
- Do not claim that the anchor layer beats a strong RF on point accuracy.  The
  supported claim is decision-loss competitiveness plus an auditable state.

## Commands

```bash
# named cases
python scripts/anchor_cli.py case zhongji_2026 --cutoff 2026-11-19
python scripts/anchor_cli.py case xinyisheng_2026 --cutoff 2026-11-19

# explicit ablations
python scripts/anchor_cli.py case xinyisheng_2026 --no-bridge
python scripts/anchor_cli.py case xinyisheng_2026 --no-calibration

# audit a prior run
python scripts/anchor_cli.py audit <run_id>

# benchmark plan validation, no model calls
python scripts/check_benchmark_plan.py \
  --report paper/results/benchmark_plan/validation.json

# tests
python -m pytest tests/test_anchor_engine.py -q
```

## Key documents

- `docs/anchor_cli.md`: CLI, MCP, audit format, historical calibration.
- `docs/codex.md`: Codex CLI setup, tool names, example prompts, DeepSeek options.
- `docs/cloud_gpu.md`: 32 GB cloud GPU runbook, Qwen3-8B vLLM, three-backbone suite.
- `docs/benchmark_plan.md`: benchmark expansion, baselines, metrics, fairness.
- `configs/benchmark_plan.yaml`: machine-readable plan.
- `paper/claim_evidence_matrix.md`: claims and supporting artifacts.
- `paper/zh/sufe_master_vague/`: current Chinese master's thesis.
