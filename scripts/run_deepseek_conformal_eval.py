"""DeepSeek as the backbone of the nested conformal decision layer.

Calibration/validation: dev+validation claims, split by company hash 60/40.
Test: 2022-2025 claims. The backbone probabilities are DeepSeek self-consistency
vote fractions over the three labels. Test outcomes are never used for
threshold, method, or alpha selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_company_disjoint_benchmark import apply_temperature, fit_temperature
from run_real_benchmark import DEFAULT_DATA, LABELS, company_key, load_records
from run_real_decision_eval import CONFIGS, planner_choice
from finplan_rag.decision.conformal import NestedConformalDecisionLayer
from finplan_rag.decision.loss import (
    LossConfig,
    action_loss,
    adjudicate_loss,
    answer_loss,
    credible_set,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVVAL = REPO_ROOT / "paper/results/deepseek_devval_selfconsistency/per_claim.jsonl"
TEST = REPO_ROOT / "paper/results/deepseek_full_selfconsistency/per_claim.jsonl"
SETS = REPO_ROOT / "paper/results/deepseek_full_verification/per_claim.jsonl"
OUT = REPO_ROOT / "paper/results/deepseek_conformal"


def load_votes(path: Path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        votes = row.get("votes") or {}
        total = sum(votes.get(label, 0) for label in LABELS)
        if total:
            probabilities = np.asarray(
                [votes.get(label, 0) / total for label in LABELS], dtype=float
            )
        else:
            probabilities = np.full(len(LABELS), 1.0 / len(LABELS))
        rows[row["claim_id"]] = {
            "truth": row["truth"],
            "probabilities": probabilities,
            "votes": votes,
        }
    return rows


def split_indices(records, seed: int = 20261103):
    calibration = []
    validation = []
    for index, record in enumerate(records):
        digest = hashlib.sha256(
            f"{seed}:{company_key(record)}".encode("utf-8")
        ).hexdigest()
        (calibration if int(digest, 16) % 100 < 60 else validation).append(index)
    return np.asarray(calibration), np.asarray(validation)


def evaluate_backbone(probabilities, truths, sets, point, config):
    conformal_loss = []
    point_loss = []
    set_loss = []
    planner_loss = []
    for probability, truth, candidate_set, prediction in zip(
        probabilities, truths, sets, point
    ):
        belief = {label: float(value) for label, value in zip(LABELS, probability)}
        set_loss.append(adjudicate_loss(candidate_set, truth, config))
        point_loss.append(answer_loss(prediction, truth, config))
        action, pred, credible, _ = planner_choice(probability, config)
        planner_loss.append(
            action_loss(action, truth, config, prediction=pred, credible_set=credible)
        )
    return {
        "conformal": float(np.mean(conformal_loss)),
        "point": float(np.mean(point_loss)),
        "set": float(np.mean(set_loss)),
        "planner": float(np.mean(planner_loss)),
        "abstain": config.abstain,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    records = {record["claim_id"]: record for _, _, record in load_records(DEFAULT_DATA)}
    devval = load_votes(DEVVAL)
    test = load_votes(TEST)
    sets = {}
    for line in SETS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        parsed = row.get("parsed") or {}
        if parsed.get("credible_set"):
            sets[row["claim_id"]] = tuple(parsed["credible_set"])
    devval_ids = [claim_id for claim_id in devval if claim_id in records]
    test_ids = [claim_id for claim_id in test if claim_id in records and claim_id in sets]
    meta = [records[claim_id] for claim_id in devval_ids]
    calibration_index, validation_index = split_indices(meta)
    calibration_probabilities = np.asarray(
        [devval[devval_ids[i]]["probabilities"] for i in range(len(devval_ids))]
    )
    calibration_truths = np.asarray(
        [devval[devval_ids[i]]["truth"] for i in range(len(devval_ids))]
    )
    test_probabilities = np.asarray([test[claim_id]["probabilities"] for claim_id in test_ids])
    test_truths = np.asarray([test[claim_id]["truth"] for claim_id in test_ids])
    test_sets = [sets[claim_id] for claim_id in test_ids]
    test_point = [LABELS[int(np.argmax(probability))] for probability in test_probabilities]

    summary = {}
    for config_name, config in CONFIGS:
        layer = NestedConformalDecisionLayer().fit(
            calibration_probabilities[calibration_index],
            calibration_truths[calibration_index],
            calibration_probabilities[validation_index],
            calibration_truths[validation_index],
            config,
        )
        layer.recalibrate(calibration_probabilities, calibration_truths, config)
        decisions = layer.predict(test_probabilities, config)
        conformal = layer.evaluate(
            test_probabilities,
            test_truths,
            [decision.credible_set for decision in decisions],
            config,
        )
        baselines = evaluate_backbone(
            test_probabilities, test_truths, test_sets, test_point, config
        )
        summary[config_name] = {
            "n": len(test_ids),
            "method": layer.method,
            "alpha": layer.alpha,
            "fallback": layer.fallback,
            **conformal,
            **baselines,
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# DeepSeek backbone + nested conformal decision layer",
        "",
        "| config | method | alpha | conformal | coverage | size | planner | point | static set |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary.items():
        lines.append(
            f"| {name} | {value['method']} | {value['alpha']} | "
            f"{value['mean_loss']:.4f} | {value['coverage']:.4f} | "
            f"{value['mean_set_size']:.3f} | {value['planner']:.4f} | "
            f"{value['point']:.4f} | {value['set']:.4f} |"
        )
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
