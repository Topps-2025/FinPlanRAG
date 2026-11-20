"""Load prompt templates and decision helpers from the vendored official repos.

The vendored repositories are pinned by commit in ``PROVENANCE``.  This module
executes the official prompt-generation code where it is importable; for the
components that require unavailable checkpoints (Self-RAG critic model, CRAG
T5 evaluator, Adaptive-RAG T5 classifier, FLARE token probabilities), the
runner records the substitution in its provenance artifact.

Set ``FINPLAN_OFFICIAL_ROOT`` to the directory that contains ``findver``,
``finfact``, ``selfrag``, ``crag``, ``flare``, and ``adaptiverag``.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "vendor" / "official_baselines"
VENDOR_ROOT = Path(os.environ.get("FINPLAN_OFFICIAL_ROOT", DEFAULT_ROOT))

PROVENANCE = {
    "findver": {
        "repo": "https://github.com/yilunzhao/FinDVer",
        "commit": "e8bb237def4ce555a606a45edba22666e31df248",
        "files": ["utils/model_input_utils.py", "run_llm.py"],
        "reused": "official CoT system prompt and document/statement input format",
        "substituted": "binary entailment output mapped to the three-way outcome by the shared task adapter",
    },
    "finfact": {
        "repo": "https://github.com/IIT-DM/Fin-Fact",
        "commit": "b30387813734d4b461e6fcdd576feb819cc0bcbd",
        "files": ["scripts/models/prompts.py", "scripts/models/experiments.py"],
        "reused": "official open-book fact-checking system prompt and claim/evidence format",
        "substituted": "TRUE/FALSE/NEI label space mapped to below/within/above by the shared task adapter",
    },
    "selfrag": {
        "repo": "https://github.com/AkariAsai/self-rag",
        "commit": "1fcdc420e48f50a7d7ab1ece5494221b93252e99",
        "files": [
            "retrieval_lm/utils.py",
            "data_creation/critic/gpt4_reward/chatgpt_need_retrieval.py",
            "data_creation/critic/gpt4_reward/chatgpt_relevance.py",
            "data_creation/critic/gpt4_reward/chatgpt_groundness.py",
        ],
        "reused": "official instruction/retrieval prompt templates and GPT-4 critic prompts",
        "substituted": "selfrag_llama2_7b special-token critic replaced by API critic calls",
    },
    "crag": {
        "repo": "https://github.com/HuskyInSalt/CRAG",
        "commit": "de7c2961ae624a1483a138c5798e1f6d0c4fb0e0",
        "files": ["scripts/CRAG_Inference.py"],
        "reused": "official format_prompt and correct/incorrect/ambiguous action logic",
        "substituted": "T5 retrieval evaluator weights replaced by an API evaluator with the official criteria",
    },
    "flare": {
        "repo": "https://github.com/jzbjyb/FLARE",
        "commit": "ec4b06b502b5ab54f3f9236b0112a5d28482e7bb",
        "files": ["src/templates.py", "src/openai_api.py", "configs/2wikihop_flare_config.json"],
        "reused": "official low-confidence annotation helper and active retrieval loop shape",
        "substituted": "token log-probabilities are unavailable on the API backbones; self-reported confidence is the trigger",
    },
    "adaptiverag": {
        "repo": "https://github.com/starsuzi/Adaptive-RAG",
        "commit": "0c88670af8707667eb5c1163151bb5ce61b14acb",
        "files": ["classifier/run_classifier.py", "classifier/preprocess/preprocess_utils.py", "prompts/"],
        "reused": "official A/B/C complexity semantics and route actions",
        "substituted": "T5 complexity classifier checkpoint is not released; API zero-shot classifier is used",
    },
}


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def findver_system_prompt() -> str:
    """Execute FinDVer's official CoT prompt builder on a synthetic report."""
    root = VENDOR_ROOT / "findver"
    module = _load_module(
        "finplan_findver_model_input_utils", root / "utils" / "model_input_utils.py"
    )
    with tempfile.TemporaryDirectory() as directory:
        report_path = Path(directory) / "dummy.json"
        report_path.write_text(
            json.dumps({"context": [{"context": "placeholder financial document"}]}),
            encoding="utf-8",
        )
        example = {
            "report": "dummy.json",
            "statement": "placeholder statement",
            "example_id": "dummy",
        }
        system, _ = module.prepare_cot_model_input(example, directory, None)
    return system.strip()


@lru_cache(maxsize=1)
def finfact_prompts() -> dict[str, str]:
    module = _load_module(
        "finplan_finfact_prompts",
        VENDOR_ROOT / "finfact" / "scripts" / "models" / "prompts.py",
    )
    return {
        "open_book_system": module.open_book_system,
        "closed_book_system": module.closed_book_system,
        "chain_of_thought": module.chain_of_thought,
        "self_help": module.self_help,
        "symbolic": module.symbolic,
    }


@lru_cache(maxsize=1)
def selfrag_prompts() -> dict[str, Any]:
    module = _load_module(
        "finplan_selfrag_utils", VENDOR_ROOT / "selfrag" / "retrieval_lm" / "utils.py"
    )
    critic_dir = VENDOR_ROOT / "selfrag" / "data_creation" / "critic" / "gpt4_reward"
    return {
        "prompt_dict": dict(module.PROMPT_DICT),
        "task_inst": dict(module.TASK_INST),
        "control_tokens": list(module.control_tokens),
        "need_retrieval": _literal_dict(critic_dir / "chatgpt_need_retrieval.py"),
        "relevance": _literal_dict(critic_dir / "chatgpt_relevance.py"),
        "groundness": _literal_dict(critic_dir / "chatgpt_groundness.py"),
    }


def _literal_dict(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PROMPT_DICT":
                    value = ast.literal_eval(node.value)
                    if isinstance(value, dict):
                        return {str(key): str(item) for key, item in value.items()}
    raise ValueError(f"PROMPT_DICT not found in {path}")


@lru_cache(maxsize=1)
def crag_format_prompt():
    scripts = VENDOR_ROOT / "crag" / "scripts"
    sys.path.insert(0, str(scripts))
    # The pinned CRAG code was written against an older transformers release;
    # expose the scheduler under the name it imports.
    import transformers

    if not hasattr(transformers, "get_scheduler"):
        try:
            from transformers.optimization import get_scheduler

            transformers.get_scheduler = get_scheduler
        except Exception:
            transformers.get_scheduler = lambda *args, **kwargs: None
    module = _load_module("finplan_crag_inference", scripts / "CRAG_Inference.py")
    return module.format_prompt


@lru_cache(maxsize=1)
def flare_annotate_low_confidence_terms():
    import openai

    if not hasattr(openai, "error"):
        error = types.SimpleNamespace()
        for name in (
            "APIError",
            "Timeout",
            "APIConnectionError",
            "RateLimitError",
            "ServiceUnavailableError",
            "InvalidRequestError",
            "AuthenticationError",
            "PermissionError",
        ):
            setattr(error, name, Exception)
        openai.error = error
    if not hasattr(openai, "ChatCompletion"):
        openai.ChatCompletion = types.SimpleNamespace(create=lambda *args, **kwargs: None)
    sys.path.insert(0, str(VENDOR_ROOT / "flare"))
    module = _load_module(
        "finplan_flare_templates", VENDOR_ROOT / "flare" / "src" / "templates.py"
    )
    return module.CtxPrompt.annotate_low_confidence_terms


@lru_cache(maxsize=1)
def adaptiverag_definitions() -> dict[str, str]:
    return {
        "A": "no retrieval; the structured point-in-time evidence is sufficient",
        "B": "single-step retrieval; use one retrieval pass over earlier claims",
        "C": "multi-step retrieval; retrieve and explicitly verify the evidence",
    }


def prompt_manifest() -> dict[str, Any]:
    """Return source paths, commits, and prompt hashes for the audit artifact."""
    manifest: dict[str, Any] = {"vendor_root": str(VENDOR_ROOT), "entries": {}}
    for name, entry in PROVENANCE.items():
        files = []
        for relative in entry["files"]:
            path = VENDOR_ROOT / name / relative
            if path.exists() and path.is_file():
                files.append(
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        manifest["entries"][name] = {
            "repo": entry["repo"],
            "commit": entry["commit"],
            "reused": entry["reused"],
            "substituted": entry["substituted"],
            "files": files,
        }
    manifest["prompt_hashes"] = {
        "findver_system": _sha256(findver_system_prompt()),
        "finfact_open_book_system": _sha256(finfact_prompts()["open_book_system"]),
        "selfrag_prompt_no_input": _sha256(
            selfrag_prompts()["prompt_dict"]["prompt_no_input"]
        ),
    }
    return manifest
