"""Shared answer-layer primitives for the LOFin/FinGLM answer-accuracy eval
(answer_accuracy_dense_hybrid_protocol_v1.json, Part A).

L1 (deterministic, full set): gold-answer numeric-core containment in the
retrieved context.  A gold answer is *supported* if every numeric core (or the
normalized full string, when the answer has no numeric core) occurs in the
context with digit boundaries.  FinGLM uses any-variant semantics across the
three gold variants.

L2 (Qwen2.5-0.5B-Instruct, diagnostic subsample): free-form answer generation
from the retrieved context with an explicit refusal flag; scored by numeric-
core containment of the generated answer plus refusal-honesty statistics.
The weak-model instability caveat (china_mixed_period_reader precedent) makes
this a diagnostic layer only - no significance claims are made on L2.

Both layers normalize via NFKC + whitespace removal + in-number comma removal.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PEM_BLOCK_RE = re.compile(
    r"-----BEGIN PRIVACY-ENHANCED MESSAGE-----.*?"
    r"-----END PRIVACY-ENHANCED MESSAGE-----", re.DOTALL)


def normalize(s: str) -> str:
    """NFKC, lowercase for latin scripts, drop all whitespace and commas."""
    t = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[\s,]", "", t)


def strip_pem(text: str) -> str:
    """Drop EDGAR PEM envelope junk from raw complete-submission texts."""
    return _PEM_BLOCK_RE.sub(" ", text)


def extract_numeric_cores(text: str) -> list[str]:
    """Numeric cores (comma-stripped) plus email literals, in order."""
    cores: list[str] = []
    for m in _NUMBER_RE.finditer(text):
        core = m.group(0).replace(",", "")
        if core not in cores:
            cores.append(core)
    for m in _EMAIL_RE.finditer(text):
        core = m.group(0)
        if core not in cores:
            cores.append(core)
    return cores


def _core_pattern(core: str) -> str:
    if "@" in core:
        return re.escape(core)
    return rf"(?<![0-9]){re.escape(core)}(?![0-9])"


def cores_supported(context_norm: str, cores: Sequence[str]) -> tuple[bool, list[str]]:
    """All cores present (with digit boundaries) in the normalized context."""
    missing = [c for c in cores if re.search(_core_pattern(c), context_norm) is None]
    return not missing, missing


# ---------------------------------------------------------------------------
# L1: gold-answer support (any-variant semantics for multiple gold variants)
# ---------------------------------------------------------------------------

def gold_support(context_text: str, gold_variants: Sequence[str],
                 max_core_len: int = 20) -> dict:
    """Support of the gold answer in a context.

    Returns {supported, scorable, missing_cores, variant_status}.
    A variant is scorable if it has >=1 numeric core (or email) with length
    <= max_core_len (very short numbers like a lone '10' are excluded from
    the core test as non-discriminative but still count as present-only).
    Unscorable variants (no core at all) fall back to normalized full-string
    containment.  Any-variant semantics: supported if ANY variant is fully
    supported.
    """
    context_norm = normalize(context_text)
    variant_status = []
    for variant in gold_variants:
        text = str(variant)
        cores = [c for c in extract_numeric_cores(text) if len(c) <= max_core_len]
        if cores:
            ok, missing = cores_supported(context_norm, cores)
            variant_status.append({
                "mode": "cores",
                "n_cores": len(cores),
                "supported": ok,
                "missing": missing,
            })
        else:
            full = normalize(text)
            ok = bool(full and len(full) > 0 and full in context_norm)
            variant_status.append({
                "mode": "full_string",
                "supported": ok,
                "missing": [] if ok else [full[:60]],
            })
    scorable = any(v["mode"] == "cores" for v in variant_status)
    if scorable:
        ok_variants = [v for v in variant_status if v["mode"] == "cores" and v["supported"]]
        supported = bool(ok_variants)
    else:
        supported = any(v["supported"] for v in variant_status)
    return {
        "supported": supported,
        "scorable": scorable,
        "variant_status": variant_status,
    }


def l1_row(case_id: str, method: str, stratum: str, context_text: str,
           gold_variants: Sequence[str]) -> dict:
    g = gold_support(context_text, gold_variants)
    return {
        "case_id": case_id,
        "method": method,
        "stratum": stratum,
        "support": int(g["supported"]),
        "scorable": int(g["scorable"]),
        "n_cores": sum(v.get("n_cores", 0) for v in g["variant_status"]),
    }


# ---------------------------------------------------------------------------
# L2: Qwen2.5-0.5B-Instruct reader (diagnostic)
# ---------------------------------------------------------------------------

PROMPT_ZH = """你是金融年报问答助手。请只依据【检索到的年报内容】回答问题。
如果检索到的内容不足以回答问题,必须拒绝回答。
问题:{question}

【检索到的年报内容】:
{context}

只输出 JSON,不要输出其他内容,格式:
{{"answer": "你的回答或'证据不足'","support": "yes"或"no"}}
support 为 "no" 表示检索内容不足以回答。"""

PROMPT_EN = """You are a financial annual-report QA assistant. Answer the
question using ONLY the retrieved report content below. If the retrieved
content is insufficient to answer, you MUST refuse.
Question: {question}

[Retrieved report content]:
{context}

Output JSON only, in this format:
{{"answer": "your answer or 'insufficient evidence'", "support": "yes" or "no"}}
"support": "no" means the retrieved content is insufficient. """

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_REFUSAL_RE = re.compile(r"证据不足|insufficient")


# ---------------------------------------------------------------------------
# L2 model: Qwen2.5-0.5B-Instruct (CPU, greedy) - shared by both readers
# ---------------------------------------------------------------------------

def load_reader_model(model_path):
    """Local-only Qwen2.5-0.5B-Instruct load (china_mixed_period_reader
    precedent).  Returns (tokenizer, model) on CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, prompt: str,
                    max_new_tokens: int = 160) -> str:
    """Greedy decode with eos padding, returns raw output text."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def build_prompt(question: str, context_text: str, is_chinese: bool) -> str:
    template = PROMPT_ZH if is_chinese else PROMPT_EN
    return template.format(question=question, context=context_text)


def parse_llm_output(output: str) -> dict:
    malformed = False
    match = _JSON_RE.search(output)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                answer = str(data.get("answer", "")).strip()
                support = str(data.get("support", "")).strip().lower()
                if support in ("yes", "no"):
                    return {"answer": answer, "support": support,
                            "malformed": False}
                if support in ("true", "false"):
                    return {"answer": answer, "support": "yes" if support == "true"
                            else "no", "malformed": False}
        except json.JSONDecodeError:
            pass
    text = output.strip()
    refused = bool(_REFUSAL_RE.search(text))
    return {"answer": text[:200], "support": "no" if refused else "yes",
            "malformed": True}


def score_llm(generated: dict, gold_variants: Sequence[str],
              max_core_len: int = 20) -> dict:
    """Numeric-core containment of the generated answer against any variant,
    plus honesty flags relative to whether the context truly contained the
    gold (caller supplies that via l1_on_context)."""
    text = str(generated.get("answer", ""))
    cores = [c for c in extract_numeric_cores(text) if len(c) <= max_core_len]
    context_norm = normalize(text)
    ok_variant = False
    for variant in gold_variants:
        v_cores = [c for c in extract_numeric_cores(str(variant)) if len(c) <= max_core_len]
        if v_cores:
            ok, _ = cores_supported(context_norm, v_cores)
        else:
            ok = normalize(str(variant)) in context_norm
        if ok:
            ok_variant = True
            break
    return {
        "correct_numeric_core": int(ok_variant),
        "n_generated_cores": len(cores),
        "asserted": int(generated.get("support", "yes") == "yes"),
        "malformed": int(generated.get("malformed", False)),
    }


def llm_row(case_id: str, method: str, stratum: str, generated: dict,
            gold_variants: Sequence[str], context_support: int) -> dict:
    s = score_llm(generated, gold_variants)
    asserted = bool(s["asserted"])
    return {
        "case_id": case_id,
        "method": method,
        "stratum": stratum,
        "context_support": int(context_support),  # L1 support of THIS context
        "support": s["asserted"],                  # model asserted an answer
        "correct": int(asserted and s["correct_numeric_core"] == 1),
        "fabricated": int(asserted and s["correct_numeric_core"] == 0),
        "refused": int(not asserted),
        "correct_refusal": int(not asserted and not context_support),
        "false_refusal": int(not asserted and bool(context_support)),
        "malformed": s["malformed"],
        "n_generated_cores": s["n_generated_cores"],
    }
