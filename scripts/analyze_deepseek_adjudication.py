"""Compare DeepSeek credible sets with logistic credible sets on the same claims."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_company_disjoint_benchmark import DEFAULT_DATA, run_fold_seed
from run_real_benchmark import LABELS, company_key, load_records
from run_real_decision_eval import CONFIGS
from finplan_rag.decision.loss import (
    action_loss,
    adjudicate_loss,
    answer_loss,
    credible_set,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LLM_PATH = REPO_ROOT / "paper/results/deepseek_full_verification/per_claim.jsonl"
OUT = REPO_ROOT / "paper/results/deepseek_adjudication"


def main() -> int:
    records = load_records(DEFAULT_DATA)
    by_company = defaultdict(list)
    for row in records:
        by_company[company_key(row[2])].append(row)

    result = run_fold_seed(records, by_company, 1, calibrate=True, verified_only=True)
    logistic = {
        key: (truth, np.asarray(probabilities, dtype=float))
        for key, truth, probabilities in zip(
            result["keys"], result["truths"], result["probabilities"]
        )
    }
    llm_rows = {}
    if LLM_PATH.exists():
        for line in LLM_PATH.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            llm_rows[row["claim_id"]] = row

    rows = []
    for claim_id, (truth, probabilities) in logistic.items():
        llm = llm_rows.get(claim_id)
        if not llm:
            continue
        parsed = llm.get("parsed") or {}
        credible = parsed.get("credible_set") or []
        if not credible:
            continue
        rows.append(
            {
                "claim_id": claim_id,
                "truth": truth,
                "logistic": probabilities,
                "llm_label": parsed.get("label"),
                "llm_set": credible,
                "llm_confidence": parsed.get("confidence"),
            }
        )

    table = {}
    for name, config in CONFIGS:
        llm_set_loss = []
        llm_answer_loss = []
        logistic_set_loss = []
        logistic_answer_loss = []
        for row in rows:
            truth = row["truth"]
            llm_set_loss.append(adjudicate_loss(row["llm_set"], truth, config))
            llm_answer_loss.append(answer_loss(row["llm_label"] or "above", truth, config))
            belief = {
                label: float(value)
                for label, value in zip(LABELS, row["logistic"])
            }
            credible = credible_set(belief, config.alpha)
            logistic_set_loss.append(adjudicate_loss(credible, truth, config))
            prediction = LABELS[int(np.argmax(row["logistic"]))]
            logistic_answer_loss.append(answer_loss(prediction, truth, config))
        table[name] = {
            "llm_set": float(np.mean(llm_set_loss)),
            "llm_answer": float(np.mean(llm_answer_loss)),
            "llm_abstain": config.abstain,
            "logistic_set": float(np.mean(logistic_set_loss)),
            "logistic_answer": float(np.mean(logistic_answer_loss)),
        }

    coverage = float(
        np.mean([row["truth"] in row["llm_set"] for row in rows])
    )
    size = float(np.mean([len(row["llm_set"]) for row in rows]))
    summary = {
        "schema": "finplan-deepseek-adjudication-v1",
        "n": len(rows),
        "verified_only": True,
        "llm_coverage": coverage,
        "llm_mean_set_size": size,
        "table": table,
        "note": "Logistic probabilities are seed 1, calibrated; LLM sets are from the full 273-claim run filtered to verified labels.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# DeepSeek credible-set adjudication",
        "",
        f"- claims: {len(rows)} verified",
        f"- LLM coverage: {coverage:.4f}",
        f"- LLM mean set size: {size:.4f}",
        "",
        "| config | LLM set | LLM answer | Logistic set | Logistic answer | Abstain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in table.items():
        lines.append(
            f"| {name} | {values['llm_set']:.4f} | {values['llm_answer']:.4f} | "
            f"{values['logistic_set']:.4f} | {values['logistic_answer']:.4f} | "
            f"{values['llm_abstain']:.4f} |"
        )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
