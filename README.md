# FinPlan-RAG

FinPlan-RAG is a research implementation of **point-in-time, obligation-aware retrieval planning for financial question answering**. It makes the retrieval state explicit: which entity and filing period still require evidence, whether a document was available at the question cutoff, and whether the evidence set is closed enough to answer.

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

The package requires Python 3.10+ and NumPy only. Reader model integrations are deliberately optional; no API key or model weight is needed for the included tests and smoke run.

## Data and storage

Large corpora, downloaded filings, model weights, and generated result tables are not committed to Git. See [DATA_STORAGE.md](DATA_STORAGE.md) for the external-storage contract and `FINPLANRAG_STORAGE_ROOT` override.

## Repository status

The English package is the maintained public code surface. The `paper/`
directory contains synchronized English and Chinese initial drafts; historical
Chinese research notes and pre-refactor experiment scripts remain in the sibling
directory `FinPlanRAG_legacy_cn` when present.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). The repository is released under the MIT License. Cite the accompanying paper or repository commit when using the benchmark or planner in research.
