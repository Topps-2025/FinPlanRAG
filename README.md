# FinPlan-RAG

FinPlan-RAG is a research implementation of **point-in-time, obligation-aware retrieval planning for financial question answering**. It makes the retrieval state explicit: which entity and filing period still require evidence, whether a document was available at the question cutoff, and whether the evidence set is closed enough to answer.

The current paper focuses on the forward-looking branch: a claim contract, source-grounded anchor state, reveal-date reliability evolution, and full-action loss-aware readout. That implementation is `scripts/anchor_engine.py` with the CLI and MCP interface described below. The obligation-aware planner in `src/finplan_rag/` is the earlier determinate branch and is retained for provenance; its router and determinate-branch results are not part of the current paper's claims.

This repository follows the reproducibility layout used by [SQCAD](https://github.com/Topps-2025/SQCAD): an installable `src/` package, deterministic tests, small runnable examples, and external storage for large data and model artifacts.

## Evidence boundary

The code is a research artifact, not a production service or a state-of-the-art claim. The controlled benchmark is useful for mechanism diagnostics. Real-data pilot results remain exploratory, and **document closure is not the same as final answer correctness**. The implementation is intentionally non-oracle: planning never reads gold document IDs, hidden answer labels, or future documents.

## Core algorithm

1. Parse the question into fiscal-period requests (`Q2`, `FY 2024`, first half, and similar phrases).
2. Convert requests into filing obligations using the four-part key `(entity, fiscal year, filing type, fiscal period)`.
3. Retrieve one targeted passage per unresolved obligation with BM25 and a shared query budget.
4. Enforce the point-in-time cutoff during retrieval and expose an auditable evidence state.
5. Let a downstream reader answer only when the retrieved context supports the answer; otherwise it can refuse.

The core modules are:

```text
src/finplan_rag/
├── models.py             # serializable data contracts
├── temporal.py           # fiscal-period parsing and filing obligations
├── retrieval.py          # dependency-free BM25 and cutoff filtering
├── planner.py            # non-oracle obligation-aware retrieval loop
├── answer.py             # deterministic support and reader-output checks
└── controlled_planning.py # synthetic mechanism benchmark and ablations
```

## Quick start

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# POSIX:   source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
python examples/controlled_smoke.py
```

If an older pip rejects editable installation with a PEP 660 message, upgrade
pip first (`python -m pip install --upgrade pip`) or use
`python -m pip install ".[dev]"`.

Run the controlled benchmark from the installed command line:

```bash
finplan-controlled --seeds 2 --cases-per-world 8 --budget 6
```

For Codex, Claude Code, and other tool-using agents, register the vendor-neutral
adapter described in [docs/agent_integrations.md](docs/agent_integrations.md).
`finplan-rag-agent --schema` prints an OpenAI-compatible function schema, while
`python -m finplan_rag.agent_adapter` provides a JSONL stdio protocol.

### Forward-looking AnchorEvolve CLI

The repository also provides a separate CLI and MCP server for claims whose
outcome is not yet disclosed, such as a company's next annual-report metrics or
the expected disclosure date. The engine converts the question into an explicit
claim contract, retrieves point-in-time evidence, extracts source-grounded
anchors, qualifies them, updates a belief, and returns a point answer, an
adjudication set, or an abstention with a full audit trail.

```bash
pip install -e ".[agent]"
python scripts/anchor_cli.py case zhongji_2026 --cutoff 2026-11-19
python scripts/anchor_cli.py case xinyisheng_2026 --cutoff 2026-11-19
python scripts/anchor_cli.py ask "中际旭创2026年年报营业收入增速会落在什么区间？" \
  --company 中际旭创 --ticker 300308.SZ --period 2026年年度报告
```

Claude Code connects through the project `.mcp.json`; the `finplan-anchor`
server exposes `anchor_ask`, `anchor_case`, `anchor_search`, `anchor_audit`,
and `anchor_contract`. See [docs/anchor_cli.md](docs/anchor_cli.md) for the
protocol, case definition, audit format, and limitations.

Codex CLI uses the same server through `.codex/config.toml`; tool names are
namespaced as `mcp__finplan-anchor__anchor_case` and the project instructions
are in `AGENTS.md`. See [docs/codex.md](docs/codex.md) for setup, example
prompts, and optional DeepSeek provider configuration.

The benchmark expansion, named baselines and fairness protocol are specified
in [docs/benchmark_plan.md](docs/benchmark_plan.md) and validated by
`python scripts/check_benchmark_plan.py`.  The cloud-GPU runbook for
DeepSeek + GLM + Qwen3-8B is in [docs/cloud_gpu.md](docs/cloud_gpu.md); the
suite is prepared by `python scripts/run_cloud_suite.py --dry-run`.

The package requires Python 3.10+ and NumPy only. Reader model integrations are deliberately optional; no API key or model weight is needed for the included tests and smoke run.

## Data and storage

Large corpora, downloaded filings, model weights, and generated result tables are not committed to Git. See [DATA_STORAGE.md](DATA_STORAGE.md) for the external-storage contract and `FINPLANRAG_STORAGE_ROOT` override.

## Config-driven workflow

All paper experiments are driven by `configs/paper.json` and
`scripts/workflow.py`; paths, seeds, loss grids, provider/model, and the
sequential cost grid are configuration, not code. The LLM client supports
DeepSeek, GLM 5.3 flash, a locally served open-weight Qwen3-8B, and any
OpenAI-compatible endpoint; keys are read from environment variables
(`GLM_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `QWEN_API_KEY`, ...). See
`paper/reproduce.md` and `paper/remote_run_2026-11-08.md`.

```bash
PYTHONPATH="src;scripts" python scripts/workflow.py --config configs/paper.json --stage data
PYTHONPATH="src;scripts" python scripts/workflow.py --config configs/paper.json --stage benchmark
PYTHONPATH="src;scripts" python scripts/workflow.py --config configs/paper.json --stage sequential_portfolio
GLM_API_KEY=... PYTHONPATH="src;scripts" python scripts/workflow.py --config configs/paper.json --stage named
```

## Paper artifacts

- Current Chinese master's thesis (single-line FinPlanRAG with the AnchorEvolve
  module): `paper/zh/sufe_master_vague/main.tex`, compiled to
  `paper/zh/FinPlanRAG_Chinese_SUFE_Master_2026_AnchorEvolution.pdf`. The
  earlier two-branch thesis is `paper/zh/sufe_master/main.tex`.
- English ACL preprint: `paper/en/finplanrag_2026/main.tex`; conference-style
  rewrite: `paper/en/finplanrag_conference_2026/main.tex`; Chinese SUFE
  undergraduate thesis: `paper/zh/sufe/main.tex`.
- Benchmark: `paper/data/dataset_v4.json` with `paper/data/README.md` and a
  manifest.
- Reproduction commands: `paper/reproduce.md`. Official baseline repositories
  are fetched at pinned commits by `scripts/clone_official_baselines.sh`; the
  source-grounded runner is `scripts/run_official_baselines.py`, with
  provenance in `paper/results/official/*/provenance.json`. The aggregate audit
  is `paper/results/final_remote/final_report.md`; the latest review is
  `paper/review_2026-11-07.md`.

## Repository status

The English package is the maintained public code surface. The `paper/`
directory contains synchronized English and Chinese initial drafts; historical
Chinese research notes and pre-refactor experiment scripts remain in the sibling
directory `FinPlanRAG_legacy_cn` when present.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). The repository is released under the MIT License. Cite the accompanying paper or repository commit when using the benchmark or planner in research.
