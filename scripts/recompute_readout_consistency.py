"""Recompute all readout comparisons under one non-singleton action space."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from summarize_cloud_results import LOSS, LABELS, DISPLAY  # noqa: E402
from anchor_engine import answer_loss, set_loss  # noqa: E402

ROOT = REPO_ROOT / "paper" / "results" / "cloud_2026-11-20"
SETS = [tuple(LABELS[i] for i in combo) for size in (2, 3) for combo in itertools.combinations(range(3), size)]


def expected(belief: list[float]) -> tuple[list[float], list[float]]:
    total = sum(belief)
    probs = [value / total for value in belief]
    answers = [
        sum(probs[j] * answer_loss(label, LABELS[j], LABELS, LOSS) for j in range(3))
        for label in LABELS
    ]
    subsets = [
        sum(probs[j] * set_loss(subset, LABELS[j], LOSS) for j in range(3))
        for subset in SETS
    ]
    return answers, subsets


def loss_for(probabilities: list[float], truth: str, restricted: bool) -> tuple[float, str, int]:
    answers, subsets = expected(probabilities)
    best_answer = min(range(3), key=lambda i: (answers[i], i))
    if restricted:
        total = sum(probabilities)
        probs = [value / total for value in probabilities]
        order = sorted(range(3), key=lambda i: -probs[i])
        credible = []
        mass = 0.0
        for index in order:
            credible.append(index)
            mass += probs[index]
            if mass >= 0.9 - 1e-12:
                break
        credible_labels = tuple(LABELS[index] for index in sorted(credible))
        credible_loss = sum(probs[j] * set_loss(credible_labels, LABELS[j], LOSS) for j in range(3))
        if credible_loss < answers[best_answer]:
            return float(set_loss(credible_labels, truth, LOSS)), "adjudicate", len(credible_labels)
        return float(answer_loss(LABELS[best_answer], truth, LABELS, LOSS)), "answer", 1
    best_subset = min(range(len(SETS)), key=lambda i: (subsets[i], len(SETS[i]), i))
    candidates = [(answers[best_answer], 0, 1, "answer"), (subsets[best_subset], 1, len(SETS[best_subset]), "adjudicate"), (LOSS["abstain"], 2, 0, "abstain")]
    value, _, size, action = min(candidates, key=lambda row: (row[0], row[1]))
    if action == "answer":
        realized = answer_loss(LABELS[best_answer], truth, LABELS, LOSS)
    elif action == "adjudicate":
        realized = set_loss(SETS[best_subset], truth, LOSS)
    else:
        realized = LOSS["abstain"]
    return float(realized), action, int(size)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def boot_diff(a: np.ndarray, b: np.ndarray, seed: int = 20261120, resamples: int = 10000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    diff = a - b
    draws = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(resamples)])
    return {"mean": float(diff.mean()), "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5))}


def main() -> None:
    output: dict[str, Any] = {"schema": "finplan-readout-consistency-v1", "backbones": {}}
    for backbone in ("deepseek", "glm", "qwen3"):
        fusion_rows = read_jsonl(ROOT / backbone / "evolution" / "temporal_test_probs_new_cloud.jsonl")
        truths = [LABELS[row["truth"]] if isinstance(row["truth"], int) else row["truth"] for row in fusion_rows]
        probabilities = {
            "Ours (Fusion)": [row["probabilities"] for row in fusion_rows],
            "LLM-only": [row["llm_probabilities"] for row in fusion_rows],
            "RF": [row["rf_probabilities"] for row in fusion_rows],
        }
        for method in ("findver", "finfact", "selfrag", "crag", "flare", "adaptive"):
            rows = read_jsonl(ROOT / backbone / "named" / f"prob_{method}.jsonl")
            by_id = {row["claim_id"]: row for row in rows}
            ordered = [by_id[row["claim_id"]] for row in fusion_rows]
            probabilities[DISPLAY[method]] = [row["probabilities"] for row in ordered]
            assert [row["truth"] for row in ordered] == truths
        block = {}
        for method, probs in probabilities.items():
            full = [loss_for(row, truth, restricted=False) for row, truth in zip(probs, truths)]
            restricted = [loss_for(row, truth, restricted=True) for row, truth in zip(probs, truths)]
            full_losses = np.array([row[0] for row in full])
            restricted_losses = np.array([row[0] for row in restricted])
            block[method] = {
                "full_action_loss": float(full_losses.mean()),
                "restricted_loss": float(restricted_losses.mean()),
                "difference": float(full_losses.mean() - restricted_losses.mean()),
                "action_counts": {name: sum(row[1] == name for row in full) for name in ("answer", "adjudicate", "abstain")},
                "mean_set_size": float(np.mean([row[2] for row in full])),
                "_full_losses": full_losses,
                "_restricted_losses": restricted_losses,
            }
        # Paired intervals.
        for method in probabilities:
            block[method]["full_vs_restricted"] = boot_diff(block[method]["_full_losses"], block[method]["_restricted_losses"])
        best_named = min(
            [name for name in probabilities if name not in {"Ours (Fusion)", "LLM-only", "RF"}],
            key=lambda name: block[name]["full_action_loss"],
        )
        block["best_named"] = best_named
        block["fusion_vs_rf"] = boot_diff(block["Ours (Fusion)"]["_full_losses"], block["RF"]["_full_losses"])
        block["fusion_vs_best_named"] = boot_diff(block["Ours (Fusion)"]["_full_losses"], block[best_named]["_full_losses"])
        for method in block:
            if isinstance(block[method], dict):
                block[method].pop("_full_losses", None)
                block[method].pop("_restricted_losses", None)
        output["backbones"][backbone] = block
        print(backbone, "best named", best_named)
        for method in ["Ours (Fusion)", "RF", best_named, "LLM-only"]:
            row = block[method]
            print(f"  {method:22s} full={row['full_action_loss']:.3f} restricted={row['restricted_loss']:.3f} "
                  f"diff={row['difference']:+.3f} {row['full_vs_restricted']}")
    (ROOT / "readout_consistency.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
