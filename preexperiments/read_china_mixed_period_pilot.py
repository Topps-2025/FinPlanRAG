"""Answer layer over frozen China mixed-period retrieval traces.

The planner/retriever side is FROZEN: this script only *reads* the frozen
traces in ``china_mixed_period_pilot_v1.json`` (8 methods x 5 cases) and
maps the pages each method actually retrieved to a ``相同 / 不同 / 证据不足``
answer for the fixed-field task "两个报告期的法定代表人是否相同".

Two answer layers:

1. **Deterministic extraction reader (primary)**: reuses the FROZEN
   legal-representative regex extractor from the builder
   (``LEGAL_REPRESENTATIVE_PATTERN`` + blacklist + NFKC canonicalization,
   including the audited ``贇→赟`` variant map).  It reads ONLY the pages the
   method retrieved.  Period attribution comes from the document id suffix
   (``{code}_{year}_{H1|Q3|FY}``) plus the period words parsed from the
   question — no gold fields are used.  If a question-named period has no
   retrieved page with an extractable name, the answer is ``证据不足``
   (refusal).  Equality of the two extracted names is deterministic.

2. **LLM judgment diagnostic (Qwen2.5-0.5B-Instruct, CPU)**: the same pages
   are given to the small LLM for a free-form judgment.  Development
   diagnostics showed its verdicts are unstable across prompt variants (three
   prompts on identical evidence produced contradictory judgments), so this
   layer is reported as a weak-reader robustness diagnostic, NOT the primary
   answer metric.

A7 stop-policy ablation: for the two obligation-closing methods we also build
"fill-to-budget" traces (keep retrieving greedily on the question until the
4-document budget is exhausted) and run both answer layers on them.  The fill
variant re-uses the frozen BM25/planner code via import; its traces are a NEW
ablation, not part of the frozen pilot record.

Frozen-set results are produced once with the final protocol; the development
2 cases are diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from build_china_mixed_period_pilot import (
    LEGAL_REPRESENTATIVE_PATTERN,
    NAME_BLACKLIST,
    canonical_name,
)
from run_china_mixed_period_pilot import (
    BM25,
    build_chunks,
    parse_periods,
    plan_obligations,
    run_method,
)

# ---------------------------------------------------------------------------
# Frozen answer-layer protocol
# ---------------------------------------------------------------------------

# LLM judgment prompt: development variant v1 (judgment tracked the evidence
# best on the dev pair: closing methods answered "不同" correctly on the
# real personnel change case).  Dev prompt search across v1/v3/v4 and the
# resulting instability are recorded in the protocol.
PROMPT_TEMPLATE = """你是金融披露核验助手。给定问题与检索到的披露文件页面,判断两个报告期该公司的法定代表人是否相同。

问题:{question}

【检索到的文件页面】:{pages}

输出 JSON,不要输出其他内容,格式如下:
{{"p1_rep":"第一个报告期法定代表人姓名或'无证据'","p2_rep":"第二个报告期法定代表人姓名或'无证据'","judgment":"相同|不同|证据不足","confidence":0.0到1.0}}

注意:judgment 只能是 "相同"、"不同" 或 "证据不足";只有当页面证据确实不足或互相冲突无法判断时才用 "证据不足"。"""

PROMPT_VARIANTS = {
    # v1 = current PROMPT_TEMPLATE (frozen for the LLM diagnostic layer)
    "v1": "fd65174284c0a17fb27406bd1669ef72a83ec5df0f35c6f14d6f99e01793f8ff",
    "v3": "62d5290a6c9957b69213f3e9c33c6e7fa25227a690be58512f803067b105ef5d",
    "v4": "58b5f742b9643ac0d1ddcbf8b9743e8400ca0caeb13541e8e1b9825d1502ab5a",
}

PAGE_TEMPLATE = "\n=== 文档 {doc_id} 第 {page_number} 页 ===\n{text}"

JUDGMENT_RE = re.compile(r"相同|不同|证据不足")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
DOC_PERIOD_RE = re.compile(r"_(\d{4})_(H1|Q3|FY)$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Layer 1: deterministic extraction reader
# ---------------------------------------------------------------------------

def filter_entity_pages(pages: list[dict], codes: list[str]) -> list[dict]:
    """Keep only pages whose document belongs to one of the target entity
    codes.  Single-shot traces pull other companies' filings (e.g. a second
    company's Q3 report), which must not be attributed to the target entity's
    periods by the reader.  Entity resolution comes from the frozen planner
    (question text + registry aliases), not from gold fields."""
    if not codes:
        return pages
    return [item for item in pages if item["doc_id"].startswith(tuple(codes))]


def extraction_reader(question: str, pages: list[dict]) -> dict:
    """Extract the field value per question-named period from the pages the
    method actually retrieved; judge by canonical equality.

    Returns (judgment, per-period values, evidence detail).
    """
    periods = parse_periods(question)
    by_period: dict[str, list[str]] = {}
    for item in pages:
        match = DOC_PERIOD_RE.search(item["doc_id"])
        if match:
            by_period.setdefault(match.group(2), []).append(item["text"])
    values = {}
    evidence = {}
    for period in periods:
        texts = by_period.get(period, [])
        names = Counter()
        pages_with_field = 0
        for text in texts:
            compact = re.sub(r"\s+", "", text)
            if "法定代表人" not in compact:
                continue
            pages_with_field += 1
            names.update(
                canonical_name(value)
                for value in LEGAL_REPRESENTATIVE_PATTERN.findall(compact)
                if value not in NAME_BLACKLIST
            )
        values[period] = names.most_common(1)[0][0] if names else None
        evidence[period] = {
            "pages_retrieved": len(texts),
            "pages_with_field": pages_with_field,
            "candidates": dict(names.most_common(3)),
        }
    present = {period: values[period] for period in periods if values[period]}
    if len(present) != len(periods):
        judgment = "证据不足"
    elif len(set(present.values())) == 1:
        judgment = "相同"
    else:
        judgment = "不同"
    return {
        "judgment": judgment,
        "values": {period: values[period] for period in periods},
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Layer 2: LLM judgment diagnostic
# ---------------------------------------------------------------------------

def build_reader_input(question: str, pages: list[dict]) -> str:
    blocks = [
        PAGE_TEMPLATE.format(
            doc_id=item["doc_id"],
            page_number=item["page_number"],
            text=item["text"][:1200],
        )
        for item in pages
    ]
    return PROMPT_TEMPLATE.format(question=question, pages="".join(blocks))


def parse_judgment(output: str) -> dict:
    malformed = False
    match = JSON_RE.search(output)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                judgment = str(data.get("judgment", ""))
                if judgment in ("相同", "不同", "证据不足"):
                    return {
                        "judgment": judgment,
                        "p1_rep": str(data.get("p1_rep", "")),
                        "p2_rep": str(data.get("p2_rep", "")),
                        "confidence": data.get("confidence"),
                        "malformed": False,
                    }
        except json.JSONDecodeError:
            pass
    hit = JUDGMENT_RE.search(output)
    if hit:
        return {"judgment": hit.group(0), "p1_rep": "", "p2_rep": "",
                "confidence": None, "malformed": True}
    return {"judgment": "证据不足", "p1_rep": "", "p2_rep": "",
            "confidence": None, "malformed": True}


def run_llm_judgment(model, tokenizer, question: str, pages: list[dict]) -> dict:
    messages = [{"role": "user", "content": build_reader_input(question, pages)}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=96,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0, inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return parse_judgment(text)


# ---------------------------------------------------------------------------
# A7 stop-policy ablation: fill-to-budget traces
# ---------------------------------------------------------------------------

def run_method_fill(case: dict, cases: list[dict], documents: list[dict],
                    index: BM25, budget: int, method: str) -> dict:
    """Run the frozen method, then keep retrieving greedily on the question
    until the budget is exhausted (no stop policy)."""
    trace = run_method(case, cases, documents, index, budget, method)
    used_docs = {item["doc_id"] for item in trace["used"]}
    while len(used_docs) < budget:
        ranked = index.search(case["question"], excluded_docs=used_docs)
        added = set()
        for score, chunk in ranked:
            if chunk["doc_id"] in added or chunk["doc_id"] in used_docs:
                continue
            trace["used"].append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "page_number": chunk["page_number"],
                    "score": score,
                }
            )
            added.add(chunk["doc_id"])
            if len(used_docs) + len(added) >= budget:
                break
        if not added:
            break
        used_docs.update(added)
    return trace


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def score_judgment(gold: bool, judgment: str) -> dict:
    asserted = judgment in ("相同", "不同")
    correct = bool(asserted and (judgment == "相同") == gold)
    return {
        "gold": gold,
        "judgment": judgment,
        "asserted": asserted,
        "correct": correct,
    }


def summarize_eval(rows: list[dict]) -> dict:
    n = max(1, len(rows))
    asserted = [row for row in rows if row["asserted"]]
    correct = sum(row["correct"] for row in asserted)
    return {
        "cases": len(rows),
        "coverage": len(asserted) / n,
        "asserted": len(asserted),
        "refused": len(rows) - len(asserted),
        "accuracy_among_asserted": correct / len(asserted) if asserted else None,
        "risk_among_asserted": 1 - correct / len(asserted) if asserted else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_PATH = r"D:\Engineering\FinPlanRAG\database\preexperiments\models\Qwen2.5-0.5B-Instruct"
RESULTS_ROOT = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\results")
DATA_PATH = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\data\china_mixed_period_page_pilot_v1.json")
FROZEN_RESULT = RESULTS_ROOT / "china_mixed_period_pilot_v1.json"
OUTPUT = RESULTS_ROOT / "china_mixed_period_reader_v1.json"

FILL_METHODS = ("finplan_period_aware", "metadata_decomposition")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "frozen-pilot", "all"),
                        default="all")
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--no-fill", action="store_true", help="skip the A7 fill ablation")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM judgment layer")
    args = parser.parse_args()

    corpus = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_RESULT.read_text(encoding="utf-8"))
    chunks = build_chunks(corpus)
    index = BM25(chunks)

    page_map = {
        (doc["doc_id"], page["page_number"]): page["text"]
        for doc in corpus["documents"]
        for page in doc["pages"]
    }
    traces = {f"{p['qid']}|{p['method']}": p for p in frozen["predictions"]}

    llm = None
    if not args.no_llm:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
        model.eval()
        llm = (model, tokenizer)

    rows = []
    selected = [
        case for case in corpus["cases"]
        if args.split == "all" or case["split"] == args.split
    ]

    for case in selected:
        qid = case["qid"]
        # Entity grounding for the reader: codes from the frozen planner
        # (question text + registry aliases), no gold fields.
        codes, _ = plan_obligations(
            case, corpus["cases"], corpus["documents"], period_aware=False
        )
        for method, trace in traces.items():
            if method.split("|")[0] != qid:
                continue
            raw_pages = [
                {"doc_id": item["doc_id"], "page_number": item["page_number"],
                 "text": page_map.get((item["doc_id"], item["page_number"]), "")}
                for item in trace["used"]
            ]
            pages = filter_entity_pages(raw_pages, codes)
            base = {
                "qid": qid,
                "split": case["split"],
                "method": trace["method"],
                "variant": "frozen_trace",
                "gold": case["normalized_semantic_same"],
                "gold_available": case["semantic_gold_available"],
                "pages_used": [item["doc_id"] + f"::p{item['page_number']}"
                               for item in trace["used"]],
                "entity_codes": codes,
                "pages_filtered": len(raw_pages) - len(pages),
            }
            ext = extraction_reader(case["question"], pages)
            rows.append(
                {
                    **base,
                    "layer": "deterministic_extraction",
                    **score_judgment(case["normalized_semantic_same"], ext["judgment"]),
                    "values": ext["values"],
                    "evidence": ext["evidence"],
                }
            )
            if llm is not None:
                result = run_llm_judgment(llm[0], llm[1], case["question"], pages)
                rows.append(
                    {
                        **base,
                        "layer": "qwen_0_5b_judgment",
                        **score_judgment(case["normalized_semantic_same"], result["judgment"]),
                        "p1_rep": result["p1_rep"],
                        "p2_rep": result["p2_rep"],
                        "confidence": result["confidence"],
                        "malformed": result["malformed"],
                    }
                )
        if not args.no_fill:
            for method in FILL_METHODS:
                trace = run_method_fill(case, corpus["cases"], corpus["documents"],
                                        index, args.budget, method)
                raw_pages = [
                    {"doc_id": item["doc_id"], "page_number": item["page_number"],
                     "text": page_map.get((item["doc_id"], item["page_number"]), "")}
                    for item in trace["used"]
                ]
                pages = filter_entity_pages(raw_pages, codes)
                base = {
                    "qid": qid,
                    "split": case["split"],
                    "method": method,
                    "variant": "fill_to_budget",
                    "gold": case["normalized_semantic_same"],
                    "gold_available": case["semantic_gold_available"],
                    "pages_used": [item["doc_id"] + f"::p{item['page_number']}"
                                   for item in trace["used"]],
                    "entity_codes": codes,
                    "pages_filtered": len(raw_pages) - len(pages),
                }
                ext = extraction_reader(case["question"], pages)
                rows.append(
                    {
                        **base,
                        "layer": "deterministic_extraction",
                        **score_judgment(case["normalized_semantic_same"], ext["judgment"]),
                        "values": ext["values"],
                        "evidence": ext["evidence"],
                    }
                )
                if llm is not None:
                    result = run_llm_judgment(llm[0], llm[1], case["question"], pages)
                    rows.append(
                        {
                            **base,
                            "layer": "qwen_0_5b_judgment",
                            **score_judgment(case["normalized_semantic_same"], result["judgment"]),
                            "p1_rep": result["p1_rep"],
                            "p2_rep": result["p2_rep"],
                            "confidence": result["confidence"],
                            "malformed": result["malformed"],
                        }
                    )

    summary = {}
    for split_name, split_key in (
        ("development", "development"),
        ("frozen_pilot", "frozen-pilot"),
        ("all_descriptive", "all"),
    ):
        split_rows = [row for row in rows if row["split"] == split_key]
        summary[split_name] = {}
        for layer in ("deterministic_extraction", "qwen_0_5b_judgment"):
            summary[split_name][layer] = {}
            for variant in sorted({row["variant"] for row in split_rows}):
                summary[split_name][layer][variant] = {}
                for method in sorted({row["method"] for row in split_rows}):
                    method_rows = [
                        row for row in split_rows
                        if row["layer"] == layer and row["variant"] == variant
                        and row["method"] == method
                    ]
                    if method_rows:
                        summary[split_name][layer][variant][method] = summarize_eval(
                            method_rows
                        )

    result = {
        "schema": "finplan-china-mixed-period-reader.v1",
        "status": (
            "frozen-run-answer-layers"
            if args.split in ("frozen-pilot", "all")
            else "development-diagnostic"
        ),
        "protocol": {
            "primary_layer": (
                "deterministic regex extraction on retrieved pages + canonical equality; "
                "period attribution from doc_id suffix + question period words; "
                "reader is entity-grounded: pages from documents of other entities are "
                "filtered out using the frozen planner's entity resolution "
                "(question text + registry aliases; no gold fields)"
            ),
            "diagnostic_layer": "Qwen2.5-0.5B-Instruct free-form judgment (CPU, greedy, max_new_tokens=96)",
            "llm_prompt_sha256": sha256_text(PROMPT_TEMPLATE),
            "llm_prompt_dev_search": {
                "variants": PROMPT_VARIANTS,
                "finding": (
                    "three dev prompts on identical evidence produced contradictory "
                    "judgments (v1 tracked evidence best; v3 refused correctly on "
                    "single-doc traces but collapsed 不同 to 相同; v4 was noisy). "
                    "Weak-model judgment is unstable -> diagnostic layer only."
                ),
            },
            "extractor_sha256_source": sha256(Path("build_china_mixed_period_pilot.py")),
            "corpus_sha256": sha256(DATA_PATH),
            "frozen_result_sha256": sha256(FROZEN_RESULT),
            "frozen_runner_sha256": sha256(Path("run_china_mixed_period_pilot.py")),
            "reader_sha256": sha256(Path(__file__)),
            "budget": args.budget,
            "fill_ablation_methods": FILL_METHODS,
            "fill_query": "original question, greedy, no stop policy",
        },
        "interpretation_boundary": [
            "Frozen set is 3 cases: no statistical significance claims.",
            "The deterministic layer is objective but regex-anchored: it reads only pages the method retrieved; if a question-named period's retrieved pages lack the field, the answer is refused (证据不足).",
            "LLM layer is a weak-reader diagnostic; do not cite its frozen-set numbers as SOTA or as the primary reader result.",
            "A7 fill-to-budget traces are a new ablation, not part of the frozen pilot record.",
        ],
        "summary": summary,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
