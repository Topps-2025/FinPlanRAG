"""Named RAG baselines on DocFinQA with a shared reader.

Methods:
  * self_rag: retrieve 8, read, retrieve 4 more on low confidence, read again;
  * crag: retrieve 8, read, expand to 16 on low confidence, read again;
  * flare: retrieve 4, read, expand to 8 on low confidence, read again;
  * adaptive_rag: an LLM complexity classifier selects 2, 8, or 16 paragraphs.

The reader, backbone, paragraph chunks, and scoring are identical to the
FinPlan DocFinQA runner.  These are adaptations of the named strategies, not
unmodified runs of the original systems.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_common import (  # noqa: E402
    call_json,
    format_evidence,
    make_client,
    obligation_terms,
    uncovered_terms,
    write_json,
    write_jsonl,
)
from run_docfinqa_external import (  # noqa: E402
    READER_SYSTEM,
    exact_match,
    gold_numbers,
    load_finqa_map,
    make_doc_chunks,
    normalize_question,
    number_in_text,
    parse_reader,
    reader_prompt,
    sample_examples,
    stable_id,
    token_f1,
)
from finplan_rag.retrieval import BM25Index  # noqa: E402

METHODS = ("self_rag", "crag", "flare", "adaptive_rag")


def adaptive_prompt(question: str) -> str:
    return (
        "Classify the evidence budget needed for this financial document question. "
        "Return JSON only: {\"budget\": 2|8|16}.\n"
        "Use 2 for a single fact, 8 for a comparison or a small table, and 16 for "
        "a multi-step calculation across sections.\n"
        f"Question: {question}\n"
    )


def read(client, question: str, chunks: Sequence[Any]) -> dict[str, Any]:
    result = call_json(client, reader_prompt(question, chunks), system=READER_SYSTEM)
    result["parsed"] = parse_reader(result.get("parsed"))
    return result


def run_example(example, finqa_map, client, method: str) -> dict[str, Any]:
    question = str(example.get("Question", ""))
    gold = str(example.get("Answer", ""))
    finqa = finqa_map.get(normalize_question(question))
    nums = gold_numbers(finqa)
    chunks = make_doc_chunks(str(example.get("Context", "")), document_id=f"doc{stable_id(question)}")
    index = BM25Index(chunks)
    acquired = []
    if method == "self_rag":
        acquired = [hit.chunk for hit in index.search(question, limit=8)]
        result = read(client, question, acquired)
        if (result.get("parsed") or {}).get("confidence", 0) < 0.7:
            query = question + " " + str((result.get("parsed") or {}).get("missing", ""))
            already = {c.chunk_id for c in acquired}
            extra = [hit.chunk for hit in index.search(query, limit=12, exclude=already)
                     if hit.chunk.chunk_id not in already][:4]
            acquired.extend(extra)
            result = read(client, question, acquired)
    elif method == "crag":
        acquired = [hit.chunk for hit in index.search(question, limit=8)]
        result = read(client, question, acquired)
        if (result.get("parsed") or {}).get("confidence", 0) < 0.7:
            already = {c.chunk_id for c in acquired}
            extra = [hit.chunk for hit in index.search(question, limit=16, exclude=already)
                     if hit.chunk.chunk_id not in already][:8]
            acquired.extend(extra)
            result = read(client, question, acquired)
    elif method == "flare":
        acquired = [hit.chunk for hit in index.search(question, limit=4)]
        result = read(client, question, acquired)
        if (result.get("parsed") or {}).get("confidence", 0) < 0.7:
            already = {c.chunk_id for c in acquired}
            extra = [hit.chunk for hit in index.search(question, limit=8, exclude=already)
                     if hit.chunk.chunk_id not in already][:4]
            acquired.extend(extra)
            result = read(client, question, acquired)
    elif method == "adaptive_rag":
        budget = 8
        try:
            raw = client.text_of(client.complete(
                adaptive_prompt(question), system="Return only JSON.",
                max_tokens=50, temperature=0.0, thinking=False, use_cache=True,
            ))
            match = re.search(r"\{.*?\}", raw, flags=re.S)
            if match:
                budget = int(json.loads(match.group(0)).get("budget", 8))
        except Exception:
            budget = 8
        budget = min(16, max(2, budget))
        acquired = [hit.chunk for hit in index.search(question, limit=budget)]
        result = read(client, question, acquired)
    else:
        raise ValueError(method)

    parsed = result.get("parsed")
    answer = parsed["answer"] if parsed else ""
    correct = bool(parsed) and exact_match(answer, gold)
    f1 = token_f1(answer, gold) if parsed else 0.0
    text = " ".join(chunk.text for chunk in acquired)
    found = [number for number in nums if number_in_text(number, text)]
    return {
        "example_id": f"docfinqa-{stable_id(question)}",
        "question": question,
        "gold_answer": gold,
        "condition": method,
        "parsed": parsed,
        "correct": correct,
        "f1": f1,
        "error": result.get("error"),
        "usage": result.get("usage", {}),
        "latency_seconds": result.get("latency_seconds", 0.0),
        "raw_text": result.get("raw_text", "")[:1200],
        "num_paragraphs": len(acquired),
        "silver_gold_numbers": nums,
        "silver_recall": len(found) / len(nums) if nums else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--finqa", required=True)
    parser.add_argument("--provider", required=True, choices=["deepseek", "glm", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--extra-body", default=None)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--sample-seed", type=int, default=20261109)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    data = sample_examples(data, args.limit, args.sample_seed)
    finqa_map = load_finqa_map(Path(args.finqa))
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    local = threading.local()

    def get_client():
        if not hasattr(local, "client"):
            local.client = make_client(
                provider=args.provider, model=args.model, base_url=args.base_url,
                api_key_env=args.api_key_env, cache_dir=Path(args.out) / "llm_cache",
                extra_body=json.loads(args.extra_body) if args.extra_body else None,
            )
        return local.client
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (out / "per_example.jsonl").open("w", encoding="utf-8") as handle, \
            ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_example, example, finqa_map, get_client(), method): (example, method)
            for example in data for method in methods
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0:
                print(f"{index}/{len(futures)}", flush=True)
    write_json(out / "run_config.json", {
        "data": args.data, "finqa": args.finqa, "provider": args.provider,
        "model": args.model, "methods": methods, "limit": args.limit,
        "sample_seed": args.sample_seed, "workers": args.workers,
    })
    print("wrote", len(rows))


if __name__ == "__main__":
    main()
