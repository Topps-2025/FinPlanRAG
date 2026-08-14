"""FinGLM 复赛A full-set answer-layer evaluation (answer_accuracy_dense_hybrid_protocol_v1.json, Part A).

L1 (deterministic, full set, n=1829): gold-answer numeric-core support in the
retrieved context, any-variant semantics across the 3 gold variants (answers
file finglm_A_answers.json, joined by filtered-order id - every question text
is asserted equal).  Contexts:
- 4 external baselines: frozen trajectories[].used_docs
- finplan_v4 / finplan_v9_cascade / metadata_v9_fill: reconstructed from
  chinese_internal_obligation_rule doc_ids (marker-deterministic), mirroring
  retrieve() semantics (budget break, metadata dedup, doc-absent skip); the
  per-case count is asserted equal to the frozen rows.documents and
  mismatches are listed explicitly
- oracle: gold_doc_ids

L2 (Qwen2.5-0.5B-Instruct diagnostic, n=150 stratified by answer_type, seed
20260814): contexts capped at <=4 docs x 3000 chars (head).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\preexperiments")
from storage_paths import RESULTS_ROOT, DATA_ROOT, MODELS_ROOT

from answer_layer import (
    load_reader_model, generate_answer, build_prompt, parse_llm_output,
    llm_row, gold_support, normalize,
)
from run_finglm_internal import chinese_internal_obligation_rule
from run_finglm_baselines import precise_path, cn_tokens

PROTOCOL = Path(r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\docs\04-数据与实验\answer_accuracy_dense_hybrid_protocol_v1.json")
CASES_PATH = DATA_ROOT / "finglm_full_cases_v1.json"
CORPUS = DATA_ROOT / "finglm_full_corpus_v1"
ANSWERS = DATA_ROOT / "external" / "china" / "finglm_A_answers.json"
FROZEN_EXT = RESULTS_ROOT / "finglm_full_baselines.json"
FROZEN_INT = RESULTS_ROOT / "finglm_full_internal_v1.json"
MODEL = MODELS_ROOT / "Qwen2.5-0.5B-Instruct"
BUDGET = 4
L2_N = 150
L2_MAX_DOCS = 4
L2_CHARS_PER_DOC = 3000

EXT_METHODS = ("single_shot", "period_metadata_decomposition",
               "generic_adaptive_period", "hirec_period")
INT_METHODS = ("finplan_v4", "finplan_v9_cascade", "metadata_v9_fill")
ALL_METHODS = EXT_METHODS + INT_METHODS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_answers() -> tuple[dict[int, list[str]], dict[int, str]]:
    """id -> (3 gold variants, question text); JSONL file."""
    out: dict[int, list[str]] = {}
    questions: dict[int, str] = {}
    with open(ANSWERS, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["id"])] = [str(v) for v in row["answer"]]
            questions[int(row["id"])] = str(row["question"])
    return out, questions


def validate_mapping(cases: list[dict], ids: list[int],
                     answer_questions: dict[int, str]) -> None:
    """case finglm_N <-> answers id N (filtered order); assert every question
    text matches the answers row's question."""
    mism = 0
    for case, aid in zip(cases, ids):
        if str(case["question"]).strip() != answer_questions[aid].strip():
            mism += 1
    assert mism == 0, f"question-text mapping failed on {mism} cases"


def reconstruct_internal(cases: list[dict], companies: dict,
                         corpus_docs: dict[str, dict]) -> dict[tuple[str, str], list[str]]:
    """Marker-deterministic used_docs for the 3 internal methods, mirroring
    run_finglm_internal.retrieve() (budget break, metadata_v9_fill dedup,
    doc absent from corpus -> no candidate, doc available after cutoff ->
    excluded by allow_future=False).

    The frozen search also returns [] when the marker candidate chunk has no
    query-term overlap with the doc's own text (zero-text stub docs in the
    corpus: markers are injected unconditionally, so postings are non-empty
    while scores stay empty).  Reproduced exactly: the method's query is
    rebuilt as retrieve() builds it and the doc is kept iff some query term
    appears in cn_tokens(doc text)."""
    from run_sec_real_pilot import parse_time
    out: dict[tuple[str, str], list[str]] = {}
    for case in cases:
        q = str(case["question"])
        cutoff = parse_time(str(case["cutoff"]))
        obligations, _ = chinese_internal_obligation_rule(q, companies)
        doc_terms: dict[str, set[str]] = {}  # per-case cache, bounds memory
        for method in INT_METHODS:
            used: list[str] = []
            for ob in obligations:
                if len(used) >= BUDGET:
                    break
                if method == "finplan_v4":
                    marker = f"path{str(ob['ticker']).lower()}{int(ob['fiscal_year'])}"
                    query = f"{q} {ob['ticker']} {ob['fiscal_year']}"
                else:
                    marker = precise_path(ob)
                    query = (f"{q} {ob['ticker']} {ob['fiscal_year']} "
                             f"{ob['fiscal_period']} {ob['filing_type']}")
                doc = corpus_docs.get(str(ob["doc_id"]))
                if doc is None:
                    continue  # marker absent from corpus -> search returns []
                if parse_time(doc["available_at"]) > cutoff:
                    continue  # allow_future=False excludes it
                if method == "metadata_v9_fill" and str(ob["doc_id"]) in used:
                    continue
                did = str(ob["doc_id"])
                text = str(doc.get("text", ""))
                if not text:
                    continue  # zero-text stub: marker candidate, no scores
                terms = doc_terms.get(did)
                if terms is None:
                    terms = doc_terms[did] = set(cn_tokens(text))
                if not set(cn_tokens(query)) & terms:
                    continue  # no query-term overlap -> frozen search returns []
                used.append(did)
            out[(case["case_id"], method)] = used
    return out


def l1_phase(cases: list[dict], answers: dict[int, list[str]], ids: list[int],
             ext_used: dict[tuple[str, str], list[str]],
             int_used: dict[tuple[str, str], list[str]],
             corpus_docs: dict[str, dict]) -> None:
    print(f"corpus docs loaded: {len(corpus_docs)}", flush=True)
    norm_cache: dict[str, str] = {}
    frozen_int = json.loads(FROZEN_INT.read_text(encoding="utf-8"))["rows"]
    rows: list[dict] = []
    mism = []
    checked = 0
    for case, aid in zip(cases, ids):
        cid = case["case_id"]
        gold = answers[aid]
        stratum = str(case.get("answer_type", "unstratified"))
        for method in ALL_METHODS:
            if method in EXT_METHODS:
                used_docs = ext_used[(cid, method)]
            else:
                used_docs = int_used[(cid, method)]
                want = next(r["documents"] for r in frozen_int
                            if r["case_id"] == cid and r["method"] == method)
                checked += 1
                if len(used_docs) != want:
                    mism.append({"case_id": cid, "method": method,
                                 "reconstructed": len(used_docs), "frozen": want})
            parts = []
            for d in used_docs:
                text = corpus_docs.get(str(d), {}).get("text")
                if not text:
                    continue
                key = str(d)
                if key not in norm_cache:
                    norm_cache[key] = normalize(text)
                parts.append(norm_cache[key])
            ctx = "\n\n".join(parts)
            g = gold_support(ctx, gold) if ctx else {"supported": False,
                                                     "scorable": True}
            rows.append({
                "case_id": cid, "method": method, "stratum": stratum,
                "support": int(g["supported"]), "scorable": int(g["scorable"]),
                "n_docs": len(used_docs),
            })
        # oracle
        used_docs = [str(d) for d in case["gold_doc_ids"]]
        parts = []
        for d in used_docs:
            text = corpus_docs.get(str(d), {}).get("text")
            if not text:
                continue
            key = str(d)
            if key not in norm_cache:
                norm_cache[key] = normalize(text)
            parts.append(norm_cache[key])
        ctx = "\n\n".join(parts)
        g = gold_support(ctx, gold) if ctx else {"supported": False,
                                                 "scorable": True}
        rows.append({
            "case_id": cid, "method": "oracle", "stratum": stratum,
            "support": int(g["supported"]), "scorable": int(g["scorable"]),
            "n_docs": len(used_docs),
        })
    out = {
        "schema": "finplan-finglm-full-answers-l1.v1",
        "status": "frozen-benchmark-extension",
        "protocol_sha256": sha256(PROTOCOL),
        "n_cases": len(cases), "n_rows": len(rows),
        "context": "union of used docs full texts (no cap)",
        "internal_reconstruction": {
            "n_asserted": checked,
            "mismatches": mism,
        },
        "rows": rows,
    }
    out_file = RESULTS_ROOT / "finglm_full_answers_l1_v1.json"
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"L1: {len(rows)} rows, internal assert mismatches={len(mism)} "
          f"-> {out_file}", flush=True)


def stratified_sample(cases: list[dict], answers: dict[int, list[str]],
                      ids: list[int], n: int) -> list[tuple[dict, int]]:
    """Deterministic: within each answer_type, ascending case order; quotas
    base + remainder spread over the first strata (sorted type order)."""
    by_type: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    for case, aid in zip(cases, ids):
        by_type[str(case.get("answer_type", "unstratified"))].append((case, aid))
    for v in by_type.values():
        v.sort(key=lambda t: str(t[0]["case_id"]))
    types = sorted(by_type)
    base, rem = divmod(n, len(types))
    picked: list[tuple[dict, int]] = []
    for i, t in enumerate(types):
        quota = base + (1 if i < rem else 0)
        picked.extend(by_type[t][:quota])
    assert len(picked) == n, (len(picked), n)
    return picked


def l2_phase(cases: list[dict], answers: dict[int, list[str]], ids: list[int],
             ext_used: dict[tuple[str, str], list[str]],
             int_used: dict[tuple[str, str], list[str]],
             corpus_docs: dict[str, dict]) -> None:
    sample = stratified_sample(cases, answers, ids, L2_N)
    tokenizer, model = load_reader_model(MODEL)
    rows: list[dict] = []
    for i, (case, aid) in enumerate(sample):
        cid = case["case_id"]
        gold = answers[aid]
        stratum = str(case.get("answer_type", "unstratified"))
        for method in ALL_METHODS:
            used_docs = (ext_used if method in EXT_METHODS else int_used)[(cid, method)]
            ctx_parts = []
            for d in used_docs[:L2_MAX_DOCS]:
                text = corpus_docs.get(str(d), {}).get("text")
                if not text:
                    continue
                ctx_parts.append(text[:L2_CHARS_PER_DOC])
            context = "\n\n".join(ctx_parts)
            prompt = build_prompt(str(case["question"]), context, True)
            gen = parse_llm_output(generate_answer(tokenizer, model, prompt))
            ctx_sup = int(gold_support(context, gold)["supported"]) if context else 0
            rows.append(llm_row(cid, method, stratum, gen, gold, ctx_sup))
        # oracle
        ctx_parts = []
        for d in [str(d) for d in case["gold_doc_ids"]][:L2_MAX_DOCS]:
            text = corpus_docs.get(str(d), {}).get("text")
            if not text:
                continue
            ctx_parts.append(text[:L2_CHARS_PER_DOC])
        context = "\n\n".join(ctx_parts)
        prompt = build_prompt(str(case["question"]), context, True)
        gen = parse_llm_output(generate_answer(tokenizer, model, prompt))
        ctx_sup = int(gold_support(context, gold)["supported"]) if context else 0
        rows.append(llm_row(cid, "oracle", stratum, gen, gold, ctx_sup))
        if (i + 1) % 10 == 0:
            print(f"L2 {i + 1}/{len(sample)} cases", flush=True)
    out = {
        "schema": "finplan-finglm-full-answers-l2.v1",
        "status": "diagnostic-only",
        "protocol_sha256": sha256(PROTOCOL),
        "model": str(MODEL),
        "n_cases": len(sample), "n_rows": len(rows),
        "subsample": "stratified by answer_type, within-type ascending case_id; seed 20260814",
        "context_cap": f"<={L2_MAX_DOCS} docs x {L2_CHARS_PER_DOC} chars (head)",
        "max_new_tokens": 160,
        "rows": rows,
    }
    out_file = RESULTS_ROOT / "finglm_full_answers_l2_v1.json"
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"L2: {len(rows)} rows -> {out_file}", flush=True)


def main(phase: str) -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    ids = [int(c["case_id"].split("_")[1]) for c in cases]
    answers, answer_questions = load_answers()
    assert len(answers) == 2000, len(answers)
    validate_mapping(cases, ids, answer_questions)
    print(f"cases={len(cases)} answers={len(answers)} mapping OK", flush=True)

    ext = json.loads(FROZEN_EXT.read_text(encoding="utf-8"))["trajectories"]
    ext_used: dict[tuple[str, str], list[str]] = {}
    for tr in ext:
        ext_used[(tr["case_id"], tr["method"])] = list(tr["used_docs"])

    registry = json.loads((DATA_ROOT / "finglm_full_registry_v1.json").read_text(encoding="utf-8"))
    # corpus doc index: doc_id -> {available_at, text}; built once and shared
    # by reconstruction + both phases (texts are ~3.5GB in memory on the
    # 14.2GB machine, so only one copy ever exists)
    doc_ids: set[str] = set()
    corpus_docs: dict[str, dict] = {}
    for path in sorted(CORPUS.glob("*.json")):
        for doc in json.loads(path.read_text(encoding="utf-8"))["documents"]:
            doc_ids.add(str(doc["doc_id"]))
            corpus_docs[str(doc["doc_id"])] = {"available_at": doc["available_at"],
                                               "text": str(doc["text"])}
    print(f"corpus doc_ids: {len(doc_ids)}", flush=True)
    int_used = reconstruct_internal(cases, registry["companies"], corpus_docs)

    if phase in ("l1", "all"):
        l1_phase(cases, answers, ids, ext_used, int_used, corpus_docs)
    if phase in ("l2", "all"):
        l2_phase(cases, answers, ids, ext_used, int_used, corpus_docs)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("l1", "l2", "all"), default="all")
    args = ap.parse_args()
    main(args.phase)
