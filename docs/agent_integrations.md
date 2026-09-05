# Code-agent integration

FinPlan-RAG exposes one vendor-neutral contract so a code agent can plan
retrieval before asking a reader model to answer.

## Codex or OpenAI-compatible tool calling

Generate the function schema with:

```bash
finplan-rag-agent --schema
```

Register the returned object as a function tool. Pass `question`, `entities`,
`filings`, `documents`, `cutoff` (ISO-8601) and an optional `budget`. The result
contains `obligations`, `covered_document_ids`, `closed`, `future_document_ids`
and `recommendation`. A `closed=false` result should cause the agent to retrieve
more or abstain; it is not permission to invent an answer.

## Claude Code and local MCP wrappers

Claude Code can register the command as a local stdio tool. The process reads
one JSON object per line and writes one JSON object per line:

```json
{"question":"FY 2024 revenue","entities":["ACME"],"filings":[],"documents":[],"cutoff":"2025-03-01T00:00:00Z","budget":4}
```

Start it with `python -m finplan_rag.agent_adapter` (or the installed
`finplan-rag-agent` entry point). A thin MCP server can forward its tool input
and output without changing the schema; no vendor SDK is required.

## Safety contract

The adapter is non-oracle and deterministic. It never accepts hidden labels,
future gold ids or a reader answer. The cutoff is enforced by the planner, and
future ids are returned only as diagnostics. Keep this tool separate from a
reader tool so the agent can audit evidence closure before generation.
