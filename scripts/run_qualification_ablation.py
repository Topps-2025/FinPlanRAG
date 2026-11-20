"""Qualification ablation on stored AnchorEvolve anchors.

The paper's H2 claims that five-type qualification and reveal-date evolution
improve calibration and decision loss.  The evolution arm already has a static
vs evolved comparison; this script supplies the missing qualification arm.

It reads the raw anchors already stored by ``experiments/evolution``, applies
several deterministic qualification policies, recomputes the anchor-only
belief and the full-action readout, and reports paired differences against the
full policy.  It makes no LLM calls, so the comparison is reproducible and does
not depend on a new model run.

Policies:

* ``full_qualification``: the paper's date/source/scope checks plus hard/soft.
* ``accept_all``: accept every anchor with an implied label; keep hard/soft.
* ``no_date_check``: skip the future-date rejection.
* ``no_source_check``: skip paragraph/evidence-id scope checks.
* ``no_scope_check``: skip the source and scope checks.
* ``hard_only`` / ``soft_only``: keep only one qualification class.

Usage:
    python scripts/run_qualification_ablation.py \
      --config experiments/evolution/config.yaml \
      --out paper/results/qualification_ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "experiments" / "evolution"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from temporal_fusion import (  # noqa: E402
    LABELS,
    boot_diff,
    canonical_type,
    decide,
    load_config,
    load_jsonl,
    metrics,
    parse_date,
)

VERIFIED_LABELS = {"xbrl_search", "xbrl_verified", "xbrl_override"}


def _base_checks(anchor: dict, row: dict, config: dict, *, date: bool, source: bool, scope: bool) -> str | None:
    implied = anchor.get("implied_bucket")
    if implied not in LABELS:
        return None
    if date:
        anchor_date = parse_date(anchor.get("date"))
        if anchor_date is not None and anchor_date > parse_date(row["cutoff"]):
            return None
    if source:
        source_id = str(anchor.get("source_id") or "")
        if source_id.startswith("P"):
            allowed = set(row.get("excerpt_ids") or [])
            if allowed and source_id not in allowed:
                return None
        elif source_id.startswith("E"):
            if source_id[1:].isdigit() and int(source_id[1:]) > int(row.get("evidence_count") or 0):
                return None
    if scope:
        value = str(anchor.get("scope") or "")
        if value and value not in {"company", "sector", "market"}:
            return None
    reliability = float(anchor.get("reliability", 0.5) or 0.5)
    if bool(anchor.get("hard")) and reliability >= float(config["anchors"]["hard_reliability"]):
        return "hard"
    return "soft"


def policy_full(anchor: dict, row: dict, config: dict) -> str | None:
    return _base_checks(anchor, row, config, date=True, source=True, scope=True)


def policy_accept_all(anchor: dict, row: dict, config: dict) -> str | None:
    if anchor.get("implied_bucket") not in LABELS:
        return None
    reliability = float(anchor.get("reliability", 0.5) or 0.5)
    if bool(anchor.get("hard")) and reliability >= float(config["anchors"]["hard_reliability"]):
        return "hard"
    return "soft"


def policy_no_date(anchor: dict, row: dict, config: dict) -> str | None:
    return _base_checks(anchor, row, config, date=False, source=True, scope=True)


def policy_no_source(anchor: dict, row: dict, config: dict) -> str | None:
    return _base_checks(anchor, row, config, date=True, source=False, scope=True)


def policy_no_scope(anchor: dict, row: dict, config: dict) -> str | None:
    return _base_checks(anchor, row, config, date=True, source=False, scope=False)


def policy_hard_only(anchor: dict, row: dict, config: dict) -> str | None:
    kind = policy_full(anchor, row, config)
    return kind if kind == "hard" else None


def policy_soft_only(anchor: dict, row: dict, config: dict) -> str | None:
    kind = policy_full(anchor, row, config)
    return kind if kind == "soft" else None


POLICIES: dict[str, Callable[[dict, dict, dict], str | None]] = {
    "full_qualification": policy_full,
    "accept_all": policy_accept_all,
    "no_date_check": policy_no_date,
    "no_source_check": policy_no_source,
    "no_scope_check": policy_no_scope,
    "hard_only": policy_hard_only,
    "soft_only": policy_soft_only,
}


def load_anchor_rows(path: Path) -> list[dict]:
    rows = load_jsonl(path)
    for row in rows:
        row["truth"] = LABELS.index(row["truth"])
    return rows


def prior_from(rows: list[dict]) -> np.ndarray:
    counts = np.bincount([row["truth"] for row in rows], minlength=len(LABELS)).astype(float)
    return (counts + 1.0) / (counts.sum() + len(LABELS))


def belief_for(
    row: dict,
    prior: np.ndarray,
    config: dict,
    policy: Callable[[dict, dict, dict], str | None],
) -> tuple[np.ndarray, int, int]:
    anchor_cfg = config["anchors"]
    belief = prior.copy()
    accepted = 0
    for anchor in row.get("anchors") or []:
        kind = policy(anchor, row, config)
        if kind not in {"hard", "soft"}:
            continue
        accepted += 1
        implied = anchor.get("implied_bucket")
        index = LABELS.index(implied)
        rho = float(anchor.get("reliability", 0.5) or 0.5)
        if kind == "hard":
            belief = np.array([1.0 if i == index else 1e-3 for i in range(3)], dtype=float)
        else:
            likelihood = np.full(3, (1.0 - float(anchor_cfg["soft_p_e"])) / 2.0)
            likelihood[index] = float(anchor_cfg["soft_p_e"])
            belief = belief * np.power(likelihood, rho)
        belief = belief / max(belief.sum(), 1e-12)
    scores = row.get("bucket_scores") or {}
    if all(label in scores for label in LABELS):
        score_vector = np.array([float(scores[label]) for label in LABELS])
        score_vector = score_vector / max(score_vector.sum(), 1e-12)
        mix = float(anchor_cfg["soft_mix"])
        belief = mix * belief + (1.0 - mix) * score_vector
        belief = belief / max(belief.sum(), 1e-12)
    return belief, accepted, len(row.get("anchors") or [])


def evaluate(
    rows: list[dict],
    prior: np.ndarray,
    config: dict,
    policy: Callable[[dict, dict, dict], str | None],
) -> tuple[dict[str, Any], np.ndarray, list[float]]:
    loss = dict(config["anchors"]["loss"])
    loss["alpha"] = 1.0 - float(config["anchors"]["alpha"])
    beliefs = []
    accepted = 0
    total = 0
    for row in rows:
        belief, n_accepted, n_total = belief_for(row, prior, config, policy)
        beliefs.append(belief)
        accepted += n_accepted
        total += n_total
    matrix = np.vstack(beliefs)
    truths = [row["truth"] for row in rows]
    summary = metrics(matrix, truths, loss, full_action=True)
    per_row = [
        decide(matrix[i], truths[i], loss, full_action=True)[0]
        for i in range(len(rows))
    ]
    summary["accepted_anchors"] = accepted
    summary["total_anchors"] = total
    summary["acceptance_rate"] = accepted / max(total, 1)
    summary["mean_accepted_per_claim"] = accepted / max(len(rows), 1)
    summary["mean_action_set_size"] = float(np.mean([
        len(decide(matrix[i], truths[i], loss, full_action=True)[2])
        for i in range(len(rows))
    ]))
    return summary, matrix, per_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/evolution/config.yaml")
    parser.add_argument("--out", default="paper/results/qualification_ablation")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()

    config = load_config(REPO_ROOT / args.config)
    validation_path = REPO_ROOT / config["sources"]["validation_new"]
    test_path = REPO_ROOT / config["sources"]["test_new"]
    validation = load_anchor_rows(validation_path)
    test_all = load_anchor_rows(test_path)

    data = {
        row["claim_id"]: row
        for row in json.loads((REPO_ROOT / config["data"]).read_text(encoding="utf-8"))
    }
    verified = {
        claim_id for claim_id, row in data.items()
        if row.get("split") == "test" and row.get("verification") in VERIFIED_LABELS
    }
    test = [row for row in test_all if row["claim_id"] in verified]
    if not test:
        raise SystemExit("no verified test rows")

    prior_rows = [
        row for row in validation
        if data.get(row["claim_id"], {}).get("verification") in VERIFIED_LABELS
    ] or validation
    prior = prior_from(prior_rows)

    results: dict[str, Any] = {}
    per_row_losses: dict[str, list[float]] = {}
    for name, policy in POLICIES.items():
        summary, _, losses = evaluate(test, prior, config, policy)
        results[name] = summary
        per_row_losses[name] = losses

    full_losses = np.asarray(per_row_losses["full_qualification"])
    comparisons: dict[str, Any] = {}
    for name in POLICIES:
        if name == "full_qualification":
            continue
        mean, low, high = boot_diff(
            np.asarray(per_row_losses[name]),
            full_losses,
            resamples=args.bootstrap,
            seed=args.seed,
        )
        comparisons[name] = {
            "mean_loss_difference_vs_full": mean,
            "ci_low": low,
            "ci_high": high,
            "n": len(full_losses),
        }

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "finplan-qualification-ablation-v1",
        "config": str(REPO_ROOT / args.config),
        "validation_anchors": str(validation_path),
        "test_anchors": str(test_path),
        "n_validation": len(validation),
        "n_test_verified": len(test),
        "prior": {label: float(prior[i]) for i, label in enumerate(LABELS)},
        "policies": results,
        "paired_differences": comparisons,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Qualification ablation",
        "",
        f"Verified test claims: {len(test)}; validation anchors: {len(validation)}.",
        "",
        "| Policy | Acc | Bal | Macro-F1 | Loss | Accepted / total | Acceptance | Set size |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in POLICIES:
        row = results[name]
        lines.append(
            f"| {name} | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | "
            f"{row['macro_f1']:.3f} | {row['decision_loss']:.3f} | "
            f"{row['accepted_anchors']}/{row['total_anchors']} | "
            f"{row['acceptance_rate']:.3f} | {row['mean_action_set_size']:.3f} |"
        )
    lines += ["", "| Comparison | Δ loss vs full [95% CI] |", "|---|---:|"]
    for name, row in comparisons.items():
        lines.append(
            f"| {name} | {row['mean_loss_difference_vs_full']:+.3f} "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] |"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "n_test": len(test), "policies": results,
                      "paired_differences": comparisons}, ensure_ascii=False, indent=2)[:5000])


if __name__ == "__main__":
    main()
