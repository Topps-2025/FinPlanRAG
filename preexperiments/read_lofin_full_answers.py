"""LOFin full-set answer-layer evaluation (answer_accuracy_dense_hybrid_protocol_v1.json, Part A).

L1 (deterministic, full set, n=1572): for every case and every one of the 9
frozen methods plus the oracle, gold reference_answer numeric-core support in
the retrieved context.  The retrieved context is the union of the used docs'
full texts (PEM-stripped) - the retrieval unit was the 180-word window, and
the overlapping window tiling covers the whole document, so the doc-text
union is the window union.

L2 (Qwen2.5-0.5B-Instruct diagnostic, n=150 stratified by stratum, seed
20260814): contexts capped at <=4 docs x 3000 chars (head) per protocol;
generation scored by numeric-core containment + refusal honesty.

used_docs per (case, method) come from:
- frozen.final.json predictions[].retrieval_by_method (v9_cascade, metadata_v9_fill)
- tmp_lofin_full/g*_v4/v7/v8_result.json predictions[].retrieval.used_docs
- tmp_lofin_full/g*_baselines_result.json trajectories[].used_docs (4 baselines)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\preexperiments")
from storage_paths import RESULTS_ROOT, DATA_ROOT, MODELS_ROOT

from answer_layer import (
    load_reader_model, generate_answer, build_prompt, parse_llm_output,
    llm_row, l1_row, strip_pem, normalize,
)

PROTOCOL = Path(r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\docs\04-数据与实验\answer_accuracy_dense_hybrid_protocol_v1.json")
FROZEN = RESULTS_ROOT / "lofin_full_benchmark_frozen.final.json"
CASES = DATA_ROOT / "lofin_full_cases_v1.json"
CORPUS = DATA_ROOT / "lofin_full_corpus_v1"
TMP = RESULTS_ROOT / "tmp_lofin_full"
MODEL = MODELS_ROOT / "Qwen2.5-0.5B-Instruct"
L2_SEED_NOTE = "stratified deterministic within-stratum ascending case_id; seed 20260814 (no RNG needed)"
L2_N = 150
L2_MAX_DOCS = 4
L2_CHARS_PER_DOC = 3000

METHODS = ("single_shot", "period_metadata_decomposition",
           "generic_adaptive_period", "hirec_period",
           "finplan_v4", "finplan_v7", "finplan_v8",
           "finplan_v9_cascade", "metadata_v9_fill")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corpus() -> dict[str, str]:
    """doc_id -> raw text (PEM envelopes still present; stripped on use)."""
    docs: dict[str, str] = {}
    for path in sorted(CORPUS.glob("*.json")):
        for doc in json.loads(path.read_text(encoding="utf-8"))["documents"]:
            docs[str(doc["doc_id"])] = str(doc["text"])
    return docs


def load_used_docs() -> dict[tuple[str, str], list[str]]:
    """(case_id, method) -> used_docs from the frozen run artifacts.

    The frozen combined file carries one prediction entry per case and runner
    (v9_and_metadata with retrieval_by_method for the 2 v9 methods, and
    v8/v7/v4 entries each with their single retrieval dict); the 4 transparent
    baselines live in the per-group baselines trajectories."""
    out: dict[tuple[str, str], list[str]] = {}
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    for pred in frozen["predictions"]:
        if pred["method"] == "v9_and_metadata":
            for method, r in pred["retrieval_by_method"].items():
                out[(pred["case_id"], method)] = list(r["used_docs"])
        else:
            out[(pred["case_id"], f"finplan_{pred['method']}")] = \
                list(pred["retrieval"]["used_docs"])
    for path in sorted(TMP.glob("g*_baselines_result.json")):
        for tr in json.loads(path.read_text(encoding="utf-8"))["trajectories"]:
            out[(tr["case_id"], tr["method"])] = list(tr["used_docs"])
    return out


def l1_support(docs: dict[str, str], used: list[str], gold_variants: list[str],
               norm_cache: dict[str, str]) -> dict:
    """Numeric-core support of gold in the union of used docs' full texts.
    norm_cache caches the normalized PEM-stripped text per doc_id."""
    parts = []
    missing = []
    for d in used:
        text = docs.get(str(d))
        if text is None:
            missing.append(str(d))
            continue
        key = str(d)
        if key not in norm_cache:
            norm_cache[key] = normalize(strip_pem(text))
        parts.append(norm_cache[key])
    ctx = "\n\n".join(parts)
    from answer_layer import gold_support
    g = gold_support(ctx, gold_variants) if ctx else {"supported": False,
                                                      "scorable": True}
    return g, missing


def stratified_sample(cases: list[dict], n: int) -> list[dict]:
    """Deterministic stratified subsample: within each stratum, cases sorted
    by case_id ascending, quotas 150//len(strata) with remainder spread over
    the first strata (sorted stratum order)."""
    from collections import defaultdict
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_stratum[str(c.get("stratum", "unstratified"))].append(c)
    for v in by_stratum.values():
        v.sort(key=lambda c: str(c["case_id"]))
    strata = sorted(by_stratum)
    base, rem = divmod(n, len(strata))
    picked: list[dict] = []
    for i, s in enumerate(strata):
        quota = base + (1 if i < rem else 0)
        picked.extend(by_stratum[s][:quota])
    assert len(picked) == n, (len(picked), n)
    return picked


def main(phase: str) -> None:
    docs = load_corpus()
    print(f"corpus docs: {len(docs)}", flush=True)
    used = load_used_docs()
    print(f"used_docs entries: {len(used)}", flush=True)
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    by_id = {c["case_id"]: c for c in cases}
    assert len(by_id) == 1572

    if phase in ("l1", "all"):
        norm_cache: dict[str, str] = {}
        rows = []
        missing_total = 0
        for case in cases:
            cid = case["case_id"]
            gold = [str(case["reference_answer"])]
            for method in METHODS + ("oracle",):
                if method == "oracle":
                    used_docs = [str(d) for d in case["gold_doc_ids"]]
                else:
                    used_docs = used.get((cid, method))
                    if used_docs is None:
                        raise KeyError((cid, method))
                g, missing = l1_support(docs, used_docs, gold, norm_cache)
                missing_total += len(missing)
                rows.append({
                    "case_id": cid, "method": method,
                    "stratum": str(case.get("stratum", "unstratified")),
                    "support": int(g["supported"]),
                    "scorable": int(g["scorable"]),
                    "n_docs": len(used_docs),
                    "missing_docs": missing,
                })
        out = {
            "schema": "finplan-lofin-full-answers-l1.v1",
            "status": "frozen-benchmark-extension",
            "protocol_sha256": sha256(PROTOCOL),
            "n_cases": 1572, "n_rows": len(rows),
            "context": "union of used docs full texts (PEM-stripped); "
                       "window tiling covers the full document",
            "missing_doc_refs": missing_total,
            "rows": rows,
        }
        out_file = RESULTS_ROOT / "lofin_full_answers_l1_v1.json"
        out_file.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"L1: {len(rows)} rows, missing refs {missing_total} -> {out_file}",
              flush=True)

    if phase in ("l2", "all"):
        sample = stratified_sample(cases, L2_N)
        tokenizer, model = load_reader_model(MODEL)
        rows = []
        for i, case in enumerate(sample):
            cid = case["case_id"]
            gold = [str(case["reference_answer"])]
            for method in METHODS + ("oracle",):
                if method == "oracle":
                    used_docs = [str(d) for d in case["gold_doc_ids"]]
                else:
                    used_docs = used.get((cid, method))
                ctx_parts = []
                for d in (used_docs or [])[:L2_MAX_DOCS]:
                    text = docs.get(str(d))
                    if text is None:
                        continue
                    ctx_parts.append(strip_pem(text)[:L2_CHARS_PER_DOC])
                context = "\n\n".join(ctx_parts)
                prompt = build_prompt(str(case["question"]), context, False)
                gen = parse_llm_output(generate_answer(tokenizer, model, prompt))
                from answer_layer import gold_support
                ctx_sup = int(gold_support(context, gold)["supported"]) if context else 0
                rows.append(llm_row(cid, method, str(case.get("stratum", "unstratified")),
                                    gen, gold, ctx_sup))
            if (i + 1) % 10 == 0:
                print(f"L2 {i + 1}/{len(sample)} cases", flush=True)
        out = {
            "schema": "finplan-lofin-full-answers-l2.v1",
            "status": "diagnostic-only",
            "protocol_sha256": sha256(PROTOCOL),
            "model": str(MODEL),
            "n_cases": len(sample),
            "n_rows": len(rows),
            "subsample": L2_SEED_NOTE,
            "context_cap": f"<={L2_MAX_DOCS} docs x {L2_CHARS_PER_DOC} chars (head)",
            "max_new_tokens": 160,
            "rows": rows,
        }
        out_file = RESULTS_ROOT / "lofin_full_answers_l2_v1.json"
        out_file.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"L2: {len(rows)} rows -> {out_file}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("l1", "l2", "all"), default="all")
    args = ap.parse_args()
    main(args.phase)
