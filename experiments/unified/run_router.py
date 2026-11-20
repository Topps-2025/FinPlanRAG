"""Two-layer FinPlanRAG router experiment.

The router sees only the query/claim text (and cutoff metadata) and chooses the
determinate RAG branch or the vague AnchorEvolve branch. Branch scores are read
from the specialist runs; the script reports routing accuracy and mixed-workload
utility under always-RAG / always-AnchorEvolve / router / oracle-router.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from deepseek_client import LLMClient  # noqa: E402
from run_docfinqa_external import sample_examples, stable_id  # noqa: E402

BUCKETS = ["below", "within", "above"]


def parse_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    brace = re.search(r"\{.*\}", text, flags=re.S)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def router_prompt(text: str, kind_hint: str = "") -> str:
    prompt = """You decide whether a financial input is determinate or vague, then route it to one of two specialists.
DETERMINATE (RAG): the input asks for a fact, number, passage, estimate, or projection that a filing states or lets you compute at the cutoff. A question about a future value is determinate when the filing itself states that value or projection.
VAGUE (ANCHOR): the input asserts or asks about a future outcome that is not realized at the cutoff. Examples include whether reported revenue will fall below, within, or above a guidance range, whether a product will launch, or whether a target will be met. A management guidance statement is a forward-looking claim, not a realized fact, even when it contains a number.
Return JSON only: {"vague":true|false,"route":"rag|anchor","confidence":0.0,"reason":"..."}.
Examples:
Input: What was Entergy's net revenue change during 2015? -> {"vague":false,"route":"rag"}
Input: Q1 2024 revenue guidance is $12.2-13.2 billion. -> {"vague":true,"route":"anchor"}
Input: What did the 10-K report for segment operating profit? -> {"vague":false,"route":"rag"}
Input: Will Q3 revenue land within management's guidance range? -> {"vague":true,"route":"anchor"}
Input: Management expects second-half revenue of $1.5-1.6 billion. -> {"vague":true,"route":"anchor"}
"""
    return prompt.replace(chr(10), chr(92) + "n") + f"Input: {text}" + chr(10)


def load_docfinqa_results(path: Path, condition: str = "planner") -> dict[str, dict[str, Any]]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("condition") == condition:
                out[row["example_id"]] = row
    return out


def load_vague_results(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out[row["claim_id"]] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docfinqa", required=True)
    parser.add_argument("--vague", required=True)
    parser.add_argument("--docfinqa-results", required=True)
    parser.add_argument("--vague-results", required=True)
    parser.add_argument("--docfinqa-limit", type=int, default=300)
    parser.add_argument("--vague-limit", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20261109)
    parser.add_argument("--condition", default="planner")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--extra-body", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    docfinqa = json.loads(Path(args.docfinqa).read_text(encoding="utf-8"))
    docfinqa = sample_examples(docfinqa, args.docfinqa_limit, args.sample_seed)
    vague = [row for row in json.loads(Path(args.vague).read_text(encoding="utf-8")) if row.get("split") == "test"]
    if args.vague_limit:
        vague = vague[: args.vague_limit]
    rag_scores = load_docfinqa_results(Path(args.docfinqa_results), args.condition)
    anchor_scores = load_vague_results(Path(args.vague_results))

    items: list[dict[str, Any]] = []
    for example in docfinqa:
        q = str(example.get("Question", ""))
        items.append({
            "id": f"docfinqa-{stable_id(q)}",
            "route": "rag",
            "text": q,
            "f1": float(rag_scores.get(f"docfinqa-{stable_id(q)}", {}).get("f1", 0.0)),
            "correct": bool(rag_scores.get(f"docfinqa-{stable_id(q)}", {}).get("correct", False)),
        })
    for record in vague:
        cid = record.get("claim_id")
        row = anchor_scores.get(cid)
        if not row or row.get("true") not in BUCKETS:
            continue
        action = row.get("action", {})
        y = BUCKETS.index(row["true"])
        if action.get("mode") == "answer":
            a = int(action.get("a", 0))
            loss = 0.0 if a == y else (12.0 if a == 2 and y == 0 else 8.0 if a == 0 and y == 2 else 10.0)
        elif action.get("mode") == "adjudicate":
            R = action.get("R", [0, 1, 2])
            loss = (0.0 if y in R else 8.0) + 2.0 * max(0, len(R) - 1)
        else:
            loss = 5.0
        items.append({
            "id": cid,
            "route": "anchor",
            "text": (
                f"{record.get('claim_text', '')} "
                f"Company: {record.get('company')}; cutoff: {record.get('cutoff')}; "
                f"target: {record.get('target_year')} Q{record.get('target_quarter')}; "
                f"guidance range: ${record.get('guidance_low_musd')}M-${record.get('guidance_high_musd')}M."
            ),
            "loss": float(loss),
            "correct": bool(row.get("pred") == row.get("true")),
        })

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    per_item_path = out / "per_item.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if per_item_path.exists():
        for line in per_item_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[str(row.get("id"))] = row
    todo = [item for item in items if str(item["id"]) not in existing]
    print(f"existing={len(existing)} todo={len(todo)} workers={args.workers}", flush=True)

    local = threading.local()

    def client() -> LLMClient:
        if not hasattr(local, "client"):
            local.client = LLMClient(
                provider=args.provider, model=args.model, base_url=args.base_url,
                api_key_env=args.api_key_env, cache_dir=out / "llm_cache",
                timeout=120, max_retries=4, reasoning_effort="low",
                extra_body=json.loads(args.extra_body) if args.extra_body else None)
        return local.client

    def work(item: dict[str, Any]) -> dict[str, Any]:
        raw = client().text_of(client().complete(
            router_prompt(item["text"]),
            system="Return only the JSON object. Do not add prose.",
            max_tokens=200, temperature=0.0, thinking=False, use_cache=True))
        parsed = parse_json(raw) or {}
        route = str(parsed.get("route", "")).lower()
        if route not in {"rag", "anchor"}:
            # A parse failure is an error.  Never fall back to the gold route:
            # that would leak the evaluation label into the deployed router.
            route = "invalid"
        item["pred_route"] = route
        item["router_confidence"] = parsed.get("confidence")
        item["router_reason"] = parsed.get("reason")
        return item

    routed = list(existing.values())
    written = len(existing)
    with per_item_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, item): item for item in todo}
        for future in as_completed(futures):
            row = future.result()
            routed.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            if written % 25 == 0:
                print(f"{written}/{len(items)}", flush=True)

    def utility(item: dict[str, Any]) -> float:
        if item["route"] == "rag":
            return float(item["f1"])
        return max(0.0, 1.0 - float(item["loss"]) / 10.0)

    rag_items = [x for x in routed if x["route"] == "rag"]
    anchor_items = [x for x in routed if x["route"] == "anchor"]
    correct_route = [x for x in routed if x["pred_route"] == x["route"]]
    wrong_route = [x for x in routed if x["pred_route"] != x["route"]]
    summary = {
        "n": len(routed),
        "n_rag": len(rag_items),
        "n_anchor": len(anchor_items),
        "routing_accuracy": len(correct_route) / len(routed),
        "routing_confusion": {
            "rag_as_rag": sum(x["route"] == "rag" and x["pred_route"] == "rag" for x in routed),
            "rag_as_anchor": sum(x["route"] == "rag" and x["pred_route"] == "anchor" for x in routed),
            "anchor_as_rag": sum(x["route"] == "anchor" and x["pred_route"] == "rag" for x in routed),
            "anchor_as_anchor": sum(x["route"] == "anchor" and x["pred_route"] == "anchor" for x in routed),
        },
        "rag_f1_when_routed": float(np.mean([x["f1"] for x in routed if x["route"] == "rag" and x["pred_route"] == "rag"])) if any(x["route"] == "rag" and x["pred_route"] == "rag" for x in routed) else 0.0,
        "anchor_loss_when_routed": float(np.mean([x["loss"] for x in routed if x["route"] == "anchor" and x["pred_route"] == "anchor"])) if any(x["route"] == "anchor" and x["pred_route"] == "anchor" for x in routed) else 0.0,
        "mixed_utility_router": float(np.mean([utility(x) if x["pred_route"] == x["route"] else 0.0 for x in routed])),
        "mixed_utility_always_rag": float(np.mean([utility(x) if x["route"] == "rag" else 0.0 for x in routed])),
        "mixed_utility_always_anchor": float(np.mean([utility(x) if x["route"] == "anchor" else 0.0 for x in routed])),
        "mixed_utility_oracle": float(np.mean([utility(x) for x in routed])),
        "wrong_route_examples": wrong_route[:20],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
