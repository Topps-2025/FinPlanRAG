"""Backbone x policy evaluation: DeepSeek and logistic as base models.

DeepSeek is not a separate method. It supplies a label, confidence, and a
credible set; the same decision policies are applied to it and to the logistic
backbone. The DeepSeek planner uses a confidence-weighted belief over the
reported credible set.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_company_disjoint_benchmark import DEFAULT_DATA, run_fold_seed
from run_real_benchmark import LABELS, company_key, load_records
from run_real_decision_eval import CONFIGS, planner_choice
from finplan_rag.decision.loss import (
    action_loss,
    adjudicate_loss,
    answer_loss,
    credible_set,
    expected_action_loss,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LLM_PATH = REPO_ROOT / "paper/results/deepseek_full_verification/per_claim.jsonl"
OUT = REPO_ROOT / "paper/results/backbone_policy"


def deepseek_belief(parsed: dict) -> tuple[dict[str, float], str, tuple[str, ...]]:
    label = parsed.get("label")
    credible = [value for value in (parsed.get("credible_set") or []) if value in LABELS]
    if label not in LABELS:
        label = credible[0] if credible else "above"
    if not credible:
        credible = [label]
    confidence = min(0.95, max(0.34, float(parsed.get("confidence", 0.5))))
    belief = {value: 0.0 for value in LABELS}
    belief[label] = confidence
    rest = [value for value in credible if value != label]
    if rest:
        for value in rest:
            belief[value] = (1.0 - confidence) / len(rest)
    else:
        belief[label] = 1.0
    return belief, label, tuple(credible)


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
    llm = {}
    for line in LLM_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["claim_id"] in logistic:
            llm[row["claim_id"]] = row

    rows = []
    for claim_id, (truth, probabilities) in logistic.items():
        row = llm.get(claim_id)
        if not row:
            continue
        parsed = row.get("parsed") or {}
        if parsed.get("label") not in LABELS:
            continue
        belief, label, credible = deepseek_belief(parsed)
        rows.append(
            {
                "claim_id": claim_id,
                "truth": truth,
                "logistic": probabilities,
                "ds_belief": belief,
                "ds_label": label,
                "ds_set": credible,
            }
        )

    table = {}
    for name, config in CONFIGS:
        values = defaultdict(list)
        for row in rows:
            truth = row["truth"]
            ds_belief = row["ds_belief"]
            ds_label = row["ds_label"]
            ds_set = row["ds_set"]
            values["ds_answer"].append(answer_loss(ds_label, truth, config))
            values["ds_set"].append(adjudicate_loss(ds_set, truth, config))
            values["ds_abstain"].append(config.abstain)
            expected = {
                "answer": expected_action_loss(
                    "answer", ds_belief, config, prediction=ds_label
                ),
                "adjudicate": expected_action_loss(
                    "adjudicate", ds_belief, config, credible_set=ds_set
                ),
                "abstain": config.abstain,
            }
            action = min(("answer", "adjudicate", "abstain"), key=lambda key: expected[key])
            values["ds_planner"].append(
                action_loss(
                    action,
                    truth,
                    config,
                    prediction=ds_label,
                    credible_set=ds_set,
                )
            )

            logistic_belief = {
                label: float(value)
                for label, value in zip(LABELS, row["logistic"])
            }
            logistic_pred = LABELS[int(np.argmax(row["logistic"]))]
            logistic_set = credible_set(logistic_belief, config.alpha)
            values["lg_answer"].append(answer_loss(logistic_pred, truth, config))
            values["lg_set"].append(adjudicate_loss(logistic_set, truth, config))
            values["lg_abstain"].append(config.abstain)
            action, prediction, credible, _ = planner_choice(row["logistic"], config)
            values["lg_planner"].append(
                action_loss(
                    action,
                    truth,
                    config,
                    prediction=prediction,
                    credible_set=credible,
                )
            )
        table[name] = {key: float(np.mean(value)) for key, value in values.items()}

    wins = sum(
        1
        for name in table
        if table[name]["ds_planner"] < table[name]["lg_planner"] - 1e-9
    )
    summary = {
        "schema": "finplan-backbone-policy-v1",
        "n": len(rows),
        "verified_only": True,
        "deepseek_planner_wins": wins,
        "configs": len(table),
        "table": table,
        "note": "DeepSeek label/confidence/credible set is the backbone; the planner is the same decision rule for both backbones.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Backbone x policy evaluation",
        "",
        f"- claims: {len(rows)} verified",
        f"- DeepSeek planner lower loss than logistic planner in {wins}/{len(table)} configurations",
        "",
        "| config | DS answer | DS set | DS planner | LG answer | LG set | LG planner |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in table.items():
        lines.append(
            f"| {name} | {values['ds_answer']:.4f} | {values['ds_set']:.4f} | "
            f"{values['ds_planner']:.4f} | {values['lg_answer']:.4f} | "
            f"{values['lg_set']:.4f} | {values['lg_planner']:.4f} |"
        )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
