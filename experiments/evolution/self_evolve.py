"""Recursive self-improvement for anchor types and extraction rules.

The loop uses validation errors only.  It never reads test outcomes.

Round:
1. Evaluate the current anchor library on a held-out validation split.
2. Mine errors on a disjoint validation split.
3. Ask the configured LLM for new anchor types and extraction rules that
   explain those errors.
4. Re-extract validation anchors with the candidate library/prompt.
5. Keep the candidate only if it lowers validation decision loss on the
   held-out split.
6. Freeze the accepted library and evaluate the test split once.

This is a type/prompt evolution loop, not gradient training.  The test set is
used only in the final ``freeze`` step.

Usage:
    python experiments/evolution/self_evolve.py \
      --config experiments/evolution/config.yaml \
      --rounds 2 --max-mine 12 --out paper/results/self_evolve/deepseek
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "experiments" / "evolution"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from deepseek_client import LLMClient  # noqa: E402
from temporal_fusion import (  # noqa: E402
    LABELS,
    canonical_type,
    decide,
    load_config,
    load_jsonl,
    metrics,
    parse_date,
    qualify,
)


def _company_key(row: dict, data: dict[str, dict]) -> str:
    record = data.get(str(row.get("claim_id"))) or {}
    return str(record.get("company") or record.get("ticker") or row.get("claim_id"))


def _split_by_company(rows: list[dict], data: dict[str, dict], seed: int = 20261120) -> tuple[list[dict], list[dict]]:
    companies = sorted({_company_key(row, data) for row in rows})
    mine_companies = {
        company for company in companies
        if int(hashlib.sha1(f"{seed}:{company}".encode()).hexdigest()[:8], 16) % 2 == 0
    }
    mine = [row for row in rows if _company_key(row, data) in mine_companies]
    select = [row for row in rows if _company_key(row, data) not in mine_companies]
    if not mine or not select:
        half = len(rows) // 2
        mine, select = rows[:half], rows[half:]
    return mine, select


def _belief(row: dict, config: dict, *, uniform_prior: bool = True) -> np.ndarray:
    anchor_cfg = config["anchors"]
    prior = np.full(len(LABELS), 1.0 / len(LABELS)) if uniform_prior else np.full(len(LABELS), 1.0 / len(LABELS))
    belief = prior.copy()
    for anchor in row.get("anchors") or []:
        kind = qualify(anchor, row, config)
        if kind not in {"hard", "soft"}:
            continue
        implied = anchor.get("implied_bucket")
        if implied not in LABELS:
            continue
        index = LABELS.index(implied)
        rho = float(anchor.get("reliability", 0.5) or 0.5)
        if kind == "hard":
            belief = np.array([1.0 if i == index else 1e-3 for i in range(3)])
        else:
            likelihood = np.full(3, (1.0 - float(anchor_cfg["soft_p_e"])) / 2.0)
            likelihood[index] = float(anchor_cfg["soft_p_e"])
            belief = belief * np.power(likelihood, rho)
        belief = belief / max(belief.sum(), 1e-12)
    scores = row.get("bucket_scores") or {}
    if all(label in scores for label in LABELS):
        score = np.array([float(scores[label]) for label in LABELS])
        score = score / max(score.sum(), 1e-12)
        mix = float(anchor_cfg["soft_mix"])
        belief = mix * belief + (1.0 - mix) * score
        belief = belief / max(belief.sum(), 1e-12)
    return belief


def evaluate(rows: list[dict], config: dict) -> dict[str, Any]:
    loss = dict(config["anchors"]["loss"])
    loss["alpha"] = 1.0 - float(config["anchors"]["alpha"])
    beliefs = np.vstack([_belief(row, config) for row in rows])
    truths = [LABELS.index(row["truth"]) for row in rows]
    summary = metrics(beliefs, truths, loss, full_action=True)
    summary["accepted_anchors"] = sum(
        sum(1 for anchor in row.get("anchors") or [] if qualify(anchor, row, config) in {"hard", "soft"})
        for row in rows
    )
    summary["total_anchors"] = sum(len(row.get("anchors") or []) for row in rows)
    return summary


def _error_cases(rows: list[dict], config: dict, limit: int) -> list[dict[str, Any]]:
    cases = []
    for row in rows:
        belief = _belief(row, config)
        pred = LABELS[int(np.argmax(belief))]
        truth = row["truth"]
        if pred == truth:
            continue
        cases.append({
            "claim_id": row.get("claim_id"),
            "truth": truth,
            "prediction": pred,
            "claim": (row.get("prompt") or "")[:2200],
            "anchors": [
                {
                    "type": anchor.get("type"),
                    "implied": anchor.get("implied_bucket"),
                    "reliability": anchor.get("reliability"),
                    "date": anchor.get("date"),
                    "quote": str(anchor.get("quote") or "")[:300],
                }
                for anchor in (row.get("anchors") or [])
            ],
            "bucket_scores": row.get("bucket_scores"),
        })
    cases.sort(key=lambda row: (row["prediction"] == row["truth"], row["claim_id"]))
    return cases[:limit]


def propose(
    client: LLMClient,
    config: dict,
    cases: list[dict[str, Any]],
    max_types: int,
) -> dict[str, Any]:
    current = ", ".join(config["anchors"]["types"])
    blocks = []
    for index, case in enumerate(cases, 1):
        blocks.append(
            f"ERROR {index}\n"
            f"claim_id: {case['claim_id']}\n"
            f"truth: {case['truth']}\n"
            f"current prediction: {case['prediction']}\n"
            f"evidence and claim:\n{case['claim']}\n"
            f"current anchors: {json.dumps(case['anchors'], ensure_ascii=False)}\n"
        )
    prompt = f"""You are improving an anchor library for a financial outcome predictor.

The outcome is one of: below, within, above.
Current anchor types: {current}

Below are validation errors. Each error contains the claim, the point-in-time
evidence, the current anchors, the prediction and the true outcome. The test set
is not shown and must not be inferred.

Propose at most {max_types} NEW anchor types that would help correct these
errors. A type must:
- be grounded in a recurring evidence pattern, not in the label alone;
- name the fields to read from the evidence;
- map to a relation among: bound, precedence, lead_time, duration,
  conservation, ratio, unit_conversion, scope;
- be usable at the cutoff without future outcomes;
- be different from the existing types.

Also propose at most 4 short extraction rules that fix recurring failures
without referring to any specific test claim.

Return JSON only:
{{
  "types": [
    {{
      "name": "short_snake_case",
      "relation": "ratio",
      "fields": ["field_1", "field_2"],
      "description": "what to read and how it maps to an outcome label",
      "example": "one sentence from an error case"
    }}
  ],
  "prompt_addendum": "short rules separated by newlines"
}}

Errors:
{chr(10).join(blocks)}
"""
    response = client.complete(prompt, max_tokens=1200, temperature=0.0, thinking=False)
    text = client.text_of(response)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"types": [], "prompt_addendum": ""}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"types": [], "prompt_addendum": ""}
    types = []
    seen = {str(value).lower() for value in config["anchors"]["types"]}
    for item in payload.get("types") or []:
        name = re.sub(r"[^a-zA-Z0-9_]", "_", str(item.get("name") or "").strip().lower())
        if not name or name in seen:
            continue
        seen.add(name)
        types.append({
            "name": name,
            "relation": str(item.get("relation") or "other"),
            "fields": [str(value) for value in (item.get("fields") or [])],
            "description": str(item.get("description") or ""),
            "example": str(item.get("example") or ""),
        })
        if len(types) >= max_types:
            break
    return {"types": types, "prompt_addendum": str(payload.get("prompt_addendum") or "")}


def run_anchors(config_path: Path, split: str, out_path: Path) -> None:
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + __import__("os").pathsep + str(REPO_ROOT / "scripts")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "experiments" / "evolution" / "run_anchors.py"),
         "--config", str(config_path), "--split", split, "--out", str(out_path)],
        cwd=REPO_ROOT, env=env, check=True,
    )


def write_candidate_config(base: dict, out: Path, result_dir: Path, extra_types: list[dict], addendum: str) -> Path:
    config = copy.deepcopy(base)
    config["results_dir"] = str(result_dir)
    config["sources"] = dict(config.get("sources") or {})
    config["sources"]["validation_new"] = str(result_dir / "anchors_validation.jsonl")
    config["sources"]["test_new"] = str(result_dir / "anchors_test.jsonl")
    config["anchors"] = dict(config.get("anchors") or {})
    config["anchors"]["types"] = list(config["anchors"]["types"]) + [item["name"] for item in extra_types]
    config["anchors"]["type_definitions"] = list(config["anchors"].get("type_definitions") or []) + extra_types
    if addendum:
        config["anchors"]["prompt_addendum"] = (str(config["anchors"].get("prompt_addendum") or "") + "\n" + addendum).strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/evolution/config.yaml")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-mine", type=int, default=12)
    parser.add_argument("--max-types", type=int, default=3)
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--out", default="paper/results/self_evolve/deepseek")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    base = load_config(REPO_ROOT / args.config)
    data = {
        row["claim_id"]: row
        for row in json.loads((REPO_ROOT / base["data"]).read_text(encoding="utf-8"))
    }
    validation_path = REPO_ROOT / base["sources"]["validation_new"]
    validation = load_jsonl(validation_path)
    mine_rows, select_rows = _split_by_company(validation, data)
    out_root = REPO_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    current_config = copy.deepcopy(base)
    current_config["results_dir"] = str(out_root / "round0")
    current_config["sources"] = dict(current_config.get("sources") or {})
    current_config["sources"]["validation_new"] = str(out_root / "round0" / "anchors_validation.jsonl")
    current_config["sources"]["test_new"] = str(out_root / "round0" / "anchors_test.jsonl")
    (out_root / "round0").mkdir(parents=True, exist_ok=True)
    # Reuse stored validation anchors as round 0.
    (out_root / "round0" / "anchors_validation.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in validation) + "\n", encoding="utf-8"
    )
    current_rows = validation
    current_select = [row for row in current_rows if row.get("claim_id") in {r.get("claim_id") for r in select_rows}]
    current_score = evaluate(current_select, current_config)
    history = [{"round": 0, "accepted": True, "validation": current_score, "types": []}]

    llm = current_config["llm"]
    client = LLMClient(
        provider=llm["provider"],
        model=llm["model"],
        base_url=llm["base_url"],
        api_key_env=llm["api_key_env"],
        cache_dir=REPO_ROOT / llm["cache_dir"],
        timeout=180,
        max_retries=4,
        extra_body=llm.get("extra_body"),
    )

    for round_index in range(1, args.rounds + 1):
        mine_ids = {row["claim_id"] for row in mine_rows}
        mine_current = [row for row in current_rows if row.get("claim_id") in mine_ids]
        cases = _error_cases(mine_current, current_config, args.max_mine)
        proposal = propose(client, current_config, cases, args.max_types)
        if not proposal["types"] and not proposal["prompt_addendum"]:
            history.append({"round": round_index, "accepted": False, "reason": "no proposal"})
            break
        variants = []
        if proposal["types"]:
            variants.append(("types_only", proposal["types"], ""))
        if proposal["prompt_addendum"]:
            variants.append(("prompt_only", [], proposal["prompt_addendum"]))
        if proposal["types"] and proposal["prompt_addendum"]:
            variants.append(("combined", proposal["types"], proposal["prompt_addendum"]))
        variant_results = []
        best = None
        for variant_name, extra_types, addendum in variants:
            variant_dir = out_root / f"round{round_index}_{variant_name}"
            candidate_config = write_candidate_config(
                current_config,
                out_root / f"candidate_round{round_index}_{variant_name}.yaml",
                variant_dir,
                extra_types,
                addendum,
            )
            run_anchors(candidate_config, "validation", variant_dir / "anchors_validation.jsonl")
            candidate_rows = load_jsonl(variant_dir / "anchors_validation.jsonl")
            select_ids = {row["claim_id"] for row in select_rows}
            candidate_select = [row for row in candidate_rows if row.get("claim_id") in select_ids]
            candidate_cfg = load_config(candidate_config)
            candidate_score = evaluate(candidate_select, candidate_cfg)
            gain = current_score["decision_loss"] - candidate_score["decision_loss"]
            result = {
                "variant": variant_name,
                "candidate_validation": candidate_score,
                "gain": gain,
                "config": str(candidate_config),
                "rows": candidate_rows,
                "select": candidate_select,
            }
            variant_results.append({key: value for key, value in result.items() if key not in {"rows", "select"}})
            if best is None or gain > best["gain"]:
                best = result
        history.append({
            "round": round_index,
            "proposal": proposal,
            "variants": variant_results,
            "current_validation": current_score,
        })
        if best is not None and best["gain"] >= args.min_gain:
            current_config = load_config(Path(best["config"]))
            current_rows = best["rows"]
            current_select = best["select"]
            current_score = best["candidate_validation"]
            history[-1]["accepted"] = True
            history[-1]["accepted_variant"] = best["variant"]
        else:
            history[-1]["accepted"] = False
            break

    final_config_path = out_root / "final_config.yaml"
    final_config_path.write_text(yaml.safe_dump(current_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    test_score = None
    if not args.skip_test:
        run_anchors(final_config_path, "test", out_root / "anchors_test.jsonl")
        test_rows = load_jsonl(out_root / "anchors_test.jsonl")
        test_score = evaluate(test_rows, current_config)
    report = {
        "schema": "finplan-self-evolve-v1",
        "base_config": args.config,
        "rounds": args.rounds,
        "validation_mine": len(mine_rows),
        "validation_select": len(select_rows),
        "history": history,
        "final_config": str(final_config_path),
        "test": test_score,
    }
    (out_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
