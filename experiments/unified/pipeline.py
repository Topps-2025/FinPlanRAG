"""Single-entry FinPlanRAG pipeline.

The pipeline exposes one call, ``run_item``, that takes a query or claim with a
cutoff, routes it with an LLM, and invokes the determinate or vague branch.
Both branches and the router call an LLM.  The experiment runners import the
branch functions directly; this module is the deployable orchestration layer.

Input item (JSONL):
    {"id": "...", "query": "...", "context": "...", "metadata": {...}}

For a determinate query, ``context`` is the filing or document text.  For a
vague claim, ``claim`` holds the released dataset record (claim_text, cutoff,
guidance range, current quarter, evidence).  The router is called unless the
item sets ``"route"`` to ``"rag"`` or ``"anchor"``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "experiments/unified"))

from deepseek_client import LLMClient  # noqa: E402
from external_common import (  # noqa: E402
    call_json,
    obligation_terms,
    uncovered_terms,
)
from finplan_rag.retrieval import BM25Index  # noqa: E402
from run_docfinqa_external import (  # noqa: E402
    READER_SYSTEM,
    make_doc_chunks,
    parse_reader,
    reader_prompt,
)
from run_router import parse_json as parse_router_json, router_prompt  # noqa: E402
from run_vague_llm import (  # noqa: E402
    BUCKETS,
    anchor_prompt,
    belief_from_anchors,
    gate,
    prior_from_history,
    credible_set,
)


FALLBACK_THRESHOLD = 0.8


class FinPlanRAG:
    def __init__(
        self,
        *,
        provider: str = "deepseek",
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        extra_body: dict[str, Any] | None = None,
        cache_dir: Path | None = None,
        timeout: int = 180,
        history_records: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.history_records = list(history_records or [])
        if self.history_records:
            self.prior, self.sector_priors, _ = prior_from_history(self.history_records)
        else:
            import numpy as np

            self.prior = np.full(3, 1 / 3)
            self.sector_priors = {}
        self.client = LLMClient(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            extra_body=extra_body,
            cache_dir=cache_dir or REPO_ROOT / "paper/results/unified/pipeline_cache",
            timeout=timeout,
            max_retries=4,
            reasoning_effort="low",
        )

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------
    def route(self, query: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        text = query
        if metadata:
            extras = "; ".join(f"{key}: {value}" for key, value in metadata.items())
            if extras:
                text = f"{query} {extras}"
        raw = self.client.text_of(
            self.client.complete(
                router_prompt(text),
                system="Return only the JSON object. Do not add prose.",
                max_tokens=200,
                temperature=0.0,
                thinking=False,
                use_cache=True,
            )
        )
        parsed = parse_router_json(raw) or {}
        route = str(parsed.get("route", "")).lower()
        if route not in {"rag", "anchor"}:
            route = "rag" if not (metadata or {}).get("guidance_low_musd") else "anchor"
        return {
            "route": route,
            "confidence": parsed.get("confidence"),
            "reason": parsed.get("reason"),
        }

    # ------------------------------------------------------------------
    # Determinate branch
    # ------------------------------------------------------------------
    def determinate(
        self,
        query: str,
        context: str,
        *,
        budget: int = 8,
    ) -> dict[str, Any]:
        chunks = make_doc_chunks(context, document_id="query")
        if not chunks:
            return {"answer": "insufficient", "sufficient": False,
                    "missing": "empty context", "evidence_ids": [], "num_paragraphs": 0}
        index = BM25Index(chunks)
        acquired = []
        acquired_ids: set[str] = set()
        last = None
        terms = obligation_terms(query)
        rounds = max(1, budget // 2)
        for round_index in range(rounds):
            target = min(budget, 2 * (round_index + 1))
            needed = target - len(acquired)
            if needed <= 0:
                break
            missing = uncovered_terms(terms, acquired)
            search = query + (" " + " ".join(missing) if missing else "")
            candidates = [
                hit for hit in index.search(search, limit=target + 32, exclude=acquired_ids)
                if hit.chunk.chunk_id not in acquired_ids
            ]
            if missing:
                candidates.sort(
                    key=lambda hit: (
                        -sum(1 for term in missing if term in hit.chunk.text.lower().replace(",", "")),
                        -hit.score,
                    )
                )
            new_chunks = [hit.chunk for hit in candidates[:needed]]
            if not new_chunks:
                break
            acquired.extend(new_chunks)
            acquired_ids.update(chunk.chunk_id for chunk in new_chunks)
            last = call_json(
                self.client,
                reader_prompt(query, acquired, round_index=round_index),
                system=READER_SYSTEM,
            )
            last["parsed"] = parse_reader(last.get("parsed"))
            still_missing = uncovered_terms(terms, acquired)
            answer = (last.get("parsed") or {}).get("answer", "")
            confidence = float((last.get("parsed") or {}).get("confidence", 0.0) or 0.0)
            if (last.get("parsed") or {}).get("sufficient") and not still_missing:
                break
            if answer.lower() not in {"", "insufficient", "unknown"} and confidence >= 0.8 and not still_missing:
                break
        parsed = (last or {}).get("parsed") or {}
        answer = str(parsed.get("answer", "")).strip().lower()
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        keep = bool(parsed.get("sufficient")) and answer not in {
            "", "insufficient", "unknown", "n/a", "none"
        } and confidence >= FALLBACK_THRESHOLD
        if not keep:
            # Escalate to the fixed-16 evidence set when the planner does not
            # close.  This is the determinate branch's fallback rule.
            fallback = [hit.chunk for hit in index.search(query, limit=16)]
            result = call_json(
                self.client,
                reader_prompt(query, fallback, round_index=rounds),
                system=READER_SYSTEM,
            )
            parsed = parse_reader(result.get("parsed")) or {}
            answer = str(parsed.get("answer", "insufficient"))
            return {
                "answer": answer,
                "confidence": parsed.get("confidence"),
                "sufficient": parsed.get("sufficient", False),
                "missing": parsed.get("missing", ""),
                "evidence_ids": parsed.get("evidence_ids", []),
                "num_paragraphs": len(acquired) + len(fallback),
                "rounds": rounds,
                "fallback": True,
            }
        return {
            "answer": parsed.get("answer", "insufficient"),
            "confidence": parsed.get("confidence"),
            "sufficient": parsed.get("sufficient", False),
            "missing": parsed.get("missing", ""),
            "evidence_ids": parsed.get("evidence_ids", []),
            "num_paragraphs": len(acquired),
            "rounds": rounds,
            "fallback": False,
        }

    # ------------------------------------------------------------------
    # Vague branch
    # ------------------------------------------------------------------
    def vague(
        self,
        record: dict[str, Any],
        *,
        prior: Any = None,
        sector_priors: dict[str, Any] | None = None,
        prior_records: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if prior_records:
            prior, sector_priors, _ = prior_from_history(prior_records)
        if prior is None:
            prior = self.prior
        if sector_priors is None:
            sector_priors = self.sector_priors
        sector_prior = None
        if sector_priors:
            sector_prior = sector_priors.get(str(record.get("sector", "other")))
        from run_vague_llm import normalize  # local import keeps the module small

        belief_prior = prior if sector_prior is None else normalize(0.5 * prior + 0.5 * sector_prior)
        raw = self.client.text_of(
            self.client.complete(
                anchor_prompt(record, prior, sector_prior),
                system="Return only the JSON object. Do not add prose.",
                max_tokens=900,
                temperature=0.0,
                thinking=False,
                use_cache=True,
            )
        )
        from run_vague_llm import parse_json as parse_anchor_json

        parsed = parse_anchor_json(raw)
        belief, anchors = belief_from_anchors(belief_prior, parsed)
        action = gate(belief)
        return {
            "pred": BUCKETS[int(max(range(3), key=lambda i: belief[i]))],
            "belief": [float(x) for x in belief],
            "credible": [BUCKETS[i] for i in credible_set(belief)],
            "anchors": anchors,
            "action": action,
            "bucket_scores": (parsed or {}).get("bucket_scores"),
            "rationale": (parsed or {}).get("rationale"),
        }

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_item(self, item: dict[str, Any]) -> dict[str, Any]:
        query = str(item.get("query") or item.get("question") or "")
        route = item.get("route")
        routed = None
        if route not in {"rag", "anchor"}:
            routed = self.route(query, item.get("metadata"))
            route = routed["route"]
        if route == "rag":
            context = item.get("context")
            if context is None and item.get("documents"):
                context = "\n\n".join(str(doc) for doc in item["documents"])
            if not context:
                return {"id": item.get("id"), "route": route, "router": routed,
                        "error": "rag route requires context or documents"}
            output = self.determinate(query, str(context), budget=int(item.get("budget", 8)))
        else:
            record = item.get("claim") or item
            output = self.vague(record)
        return {"id": item.get("id"), "route": route, "router": routed, "output": output}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL items")
    parser.add_argument("--out", required=True)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--extra-body", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--history", default=None, help="optional dataset_v4.json for priors")
    args = parser.parse_args()

    items = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]
    history = json.loads(Path(args.history).read_text(encoding="utf-8")) if args.history else None
    pipeline = FinPlanRAG(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        extra_body=json.loads(args.extra_body) if args.extra_body else None,
        history_records=history,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(items):
            result = pipeline.run_item(item)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if (index + 1) % 10 == 0:
                print(f"{index + 1}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
