"""DeepSeek verification set + confidence-weighted belief as a conformal backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from run_company_disjoint_benchmark import VERIFIED_LABELS
from run_real_benchmark import DEFAULT_DATA, LABELS, company_key, load_records
from run_real_decision_eval import CONFIGS, planner_choice
from finplan_rag.decision.conformal import ConformalConfig, NestedConformalDecisionLayer
from finplan_rag.workflow.protocol import ALPHA_GRID
from finplan_rag.decision.loss import (
    LossConfig,
    action_loss,
    adjudicate_loss,
    answer_loss,
    credible_set,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVVAL = REPO_ROOT / "paper/results/deepseek_devval_verification/per_claim.jsonl"
TEST = REPO_ROOT / "paper/results/deepseek_full_verification/per_claim.jsonl"
OUT = REPO_ROOT / "paper/results/deepseek_conformal_set"


def belief(parsed):
    label = parsed.get("label")
    credible = [value for value in (parsed.get("credible_set") or []) if value in LABELS]
    if label not in LABELS:
        label = credible[0] if credible else "above"
    if not credible:
        credible = [label]
    confidence = min(0.95, max(0.34, float(parsed.get("confidence", 0.5))))
    values = {value: 0.0 for value in LABELS}
    values[label] = confidence
    rest = [value for value in credible if value != label]
    if rest:
        for value in rest:
            values[value] = (1.0 - confidence) / len(rest)
    else:
        values[label] = 1.0
    return values, label, tuple(credible)


def load(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        parsed = row.get("parsed") or {}
        if parsed.get("label") not in LABELS:
            continue
        values, label, credible = belief(parsed)
        rows[row["claim_id"]] = {
            "truth": row["truth"],
            "probability": np.asarray([values[value] for value in LABELS]),
            "label": label,
            "credible": credible,
        }
    return rows


def split_indices(meta, seed=20261103):
    calibration = []
    validation = []
    for index, record in enumerate(meta):
        digest = hashlib.sha256(f"{seed}:{company_key(record)}".encode("utf-8")).hexdigest()
        (calibration if int(digest, 16) % 100 < 60 else validation).append(index)
    return np.asarray(calibration), np.asarray(validation)




def planner_losses(probabilities, truths, config: LossConfig, alpha: float) -> np.ndarray:
    values = []
    for row, truth in zip(probabilities, truths):
        action, prediction, credible, _ = planner_choice(
            row, replace(config, alpha=alpha)
        )
        values.append(
            action_loss(
                action,
                truth,
                config,
                prediction=prediction,
                credible_set=credible,
            )
        )
    return np.asarray(values, dtype=float)


def fixed_losses(probabilities, truths, config: LossConfig, alpha: float) -> np.ndarray:
    values = []
    for row, truth in zip(probabilities, truths):
        belief = {label: float(value) for label, value in zip(LABELS, row)}
        values.append(
            adjudicate_loss(credible_set(belief, alpha), truth, config)
        )
    return np.asarray(values, dtype=float)


def tune_alpha(probabilities, truths, config: LossConfig, policy: str):
    scores = []
    for alpha in ALPHA_GRID:
        losses = (
            planner_losses(probabilities, truths, config, alpha)
            if policy == "planner"
            else fixed_losses(probabilities, truths, config, alpha)
        )
        scores.append((float(np.mean(losses)), alpha))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--devval", type=Path, default=DEVVAL)
    parser.add_argument("--test", type=Path, default=TEST)
    parser.add_argument("--split-seeds", default="20261103,20261104,20261105,20261106,20261107")
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="evaluate only test claims whose outcome label is XBRL-verified",
    )
    parser.add_argument(
        "--threshold-objective",
        choices=("expected", "realized", "both"),
        default="realized",
        help="threshold objective; 'both' selects it on validation",
    )
    args = parser.parse_args()
    records = {record["claim_id"]: record for _, _, record in load_records(DEFAULT_DATA)}
    devval = load(args.devval)
    test = load(args.test)
    devval_ids = [claim_id for claim_id in devval if claim_id in records]
    test_ids = [
        claim_id
        for claim_id in test
        if claim_id in records
        and (
            not args.verified_only
            or records[claim_id].get("verification") in VERIFIED_LABELS
        )
    ]
    calibration_probabilities = np.asarray([devval[c]["probability"] for c in devval_ids])
    calibration_truths = np.asarray([devval[c]["truth"] for c in devval_ids])
    test_probabilities = np.asarray([test[c]["probability"] for c in test_ids])
    test_truths = np.asarray([test[c]["truth"] for c in test_ids])
    test_sets = [test[c]["credible"] for c in test_ids]
    test_points = [test[c]["label"] for c in test_ids]
    split_seeds = [int(value) for value in args.split_seeds.split(",")]
    split_results = {config_name: [] for config_name, _ in CONFIGS}
    per_claim_rows = []
    for split_seed in split_seeds:
        calibration_index, validation_index = split_indices([records[c] for c in devval_ids], split_seed)
        validation_probabilities = calibration_probabilities[validation_index]
        validation_truths = calibration_truths[validation_index]
        objectives = (
            ("expected", "realized")
            if args.threshold_objective == "both"
            else (args.threshold_objective,)
        )
        for config_name, config in CONFIGS:
            layer = NestedConformalDecisionLayer(
                ConformalConfig(
                    threshold_objectives=objectives,
                    threshold_objective=(
                        "realized"
                        if args.threshold_objective == "both"
                        else args.threshold_objective
                    ),
                )
            ).fit(
                calibration_probabilities[calibration_index],
                calibration_truths[calibration_index],
                validation_probabilities,
                validation_truths,
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
            point = [answer_loss(label, truth, config) for label, truth in zip(test_points, test_truths)]
            static_set = [adjudicate_loss(candidate, truth, config) for candidate, truth in zip(test_sets, test_truths)]
            planner = []
            for probability, truth in zip(test_probabilities, test_truths):
                action, prediction, credible, _ = planner_choice(probability, config)
                planner.append(
                    action_loss(action, truth, config, prediction=prediction, credible_set=credible)
                )
            planner_alpha = tune_alpha(
                validation_probabilities, validation_truths, config, "planner"
            )
            fixed_alpha = tune_alpha(
                validation_probabilities, validation_truths, config, "fixed"
            )
            planner_tuned = planner_losses(
                test_probabilities, test_truths, config, planner_alpha
            )
            fixed_tuned = fixed_losses(
                test_probabilities, test_truths, config, fixed_alpha
            )
            conformal_losses = np.asarray(
                [
                    action_loss(
                        decision.action,
                        truth,
                        config,
                        prediction=decision.prediction,
                        credible_set=decision.credible_set,
                    )
                    for decision, truth in zip(decisions, test_truths)
                ],
                dtype=float,
            )
            for claim_id, truth, c_loss, p_loss, f_loss, s_loss, a_loss in zip(
                test_ids,
                test_truths,
                conformal_losses,
                planner_tuned,
                fixed_tuned,
                static_set,
                point,
            ):
                per_claim_rows.append(
                    {
                        "split_seed": split_seed,
                        "config": config_name,
                        "claim_id": claim_id,
                        "truth": str(truth),
                        "conformal": float(c_loss),
                        "planner_tuned": float(p_loss),
                        "fixed_tuned": float(f_loss),
                        "static_set": float(s_loss),
                        "point": float(a_loss),
                    }
                )
            split_results[config_name].append(
                {
                    "split_seed": split_seed,
                    "method": layer.method,
                    "alpha": layer.alpha,
                    "threshold_objective": layer.threshold_objective,
                    "fallback": layer.fallback,
                    **conformal,
                    "point": float(np.mean(point)),
                    "static_set": float(np.mean(static_set)),
                    "planner": float(np.mean(planner)),
                    "planner_alpha": planner_alpha,
                    "planner_tuned": float(np.mean(planner_tuned)),
                    "fixed_alpha": fixed_alpha,
                    "fixed_tuned": float(np.mean(fixed_tuned)),
                }
            )
    def bootstrap_ci(values, n_boot=5000, seed=20261103):
        values = np.asarray(values, dtype=float)
        rng = np.random.default_rng(seed)
        means = np.asarray(
            [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
        )
        return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]

    claims = sorted({row["claim_id"] for row in per_claim_rows})
    claim_index = {claim: index for index, claim in enumerate(claims)}
    summary = {}
    for config_name, values in split_results.items():
        n = len(test_ids)
        rows = [row for row in per_claim_rows if row["config"] == config_name]
        by_claim = {claim: defaultdict(list) for claim in claims}
        for row in rows:
            for policy in ("conformal", "planner_tuned", "fixed_tuned", "static_set", "point"):
                by_claim[row["claim_id"]][policy].append(row[policy])
        means = {
            policy: np.asarray(
                [np.mean(by_claim[claim][policy]) for claim in claims if by_claim[claim][policy]]
            )
            for policy in ("conformal", "planner_tuned", "fixed_tuned", "static_set", "point")
        }
        differences = {
            policy: means["conformal"] - means[policy]
            for policy in ("planner_tuned", "fixed_tuned", "static_set", "point")
        }
        summary[config_name] = {
            "n": n,
            "splits": len(values),
            "mean_loss": float(np.mean([value["mean_loss"] for value in values])),
            "coverage": float(np.mean([value["coverage"] for value in values])),
            "mean_set_size": float(np.mean([value["mean_set_size"] for value in values])),
            "planner": float(np.mean([value["planner"] for value in values])),
            "planner_tuned": float(np.mean([value["planner_tuned"] for value in values])),
            "fixed_tuned": float(np.mean([value["fixed_tuned"] for value in values])),
            "point": float(np.mean([value["point"] for value in values])),
            "static_set": float(np.mean([value["static_set"] for value in values])),
            "loss_min": float(np.min([value["mean_loss"] for value in values])),
            "loss_max": float(np.max([value["mean_loss"] for value in values])),
            "methods": sorted(set(value["method"] for value in values)),
            "alphas": sorted(set(value["alpha"] for value in values)),
            "objectives": sorted(set(value["threshold_objective"] for value in values)),
            "fallbacks": sum(1 for value in values if value["fallback"]),
            "paired_difference": {
                policy: {
                    "mean": float(np.mean(diff)),
                    "ci95": bootstrap_ci(diff),
                }
                for policy, diff in differences.items()
            },
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out / "per_claim.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_claim_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = [
        "# DeepSeek verification set + conformal layer",
        "",
        "| config | methods | alphas | conformal | coverage | size | planner tuned | fixed tuned | point | static set | range |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, value in summary.items():
        lines.append(
            f"| {name} | {','.join(value['methods'])} | {','.join(str(a) for a in value['alphas'])} | "
            f"{value['mean_loss']:.4f} | {value['coverage']:.4f} | "
            f"{value['mean_set_size']:.3f} | {value['planner_tuned']:.4f} | "
            f"{value['fixed_tuned']:.4f} | {value['point']:.4f} | "
            f"{value['static_set']:.4f} | [{value['loss_min']:.3f}, {value['loss_max']:.3f}] |"
        )
    lines += ["", "Paired differences vs. conformal (negative favors conformal):", ""]
    for name, value in summary.items():
        line = [f"- {name}:"]
        for policy, diff in value["paired_difference"].items():
            line.append(
                f"{policy} {diff['mean']:+.4f} [{diff['ci95'][0]:+.4f}, {diff['ci95'][1]:+.4f}]"
            )
        lines.append(" ".join(line))
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
