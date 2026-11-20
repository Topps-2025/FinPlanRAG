"""Aggregate the cloud three-backbone results into one report.

Reads the per-claim probability files and summaries produced by
``run_named_baselines.py``, ``run_official_baselines.py``,
``temporal_fusion.py``, ``run_qualification_ablation.py`` and
``run_ashare_backtest.py``.  Writes ``summary.json`` and ``REPORT.md`` under
``paper/results/cloud``.

The full-action decision loss for named baselines is recomputed from their
probability files under the same loss as the main system.  This keeps the
comparison on one action space instead of scoring baselines as point answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from anchor_engine import full_action_readout  # noqa: E402

LABELS = ("below", "within", "above")
LOSS = {"over": 12.0, "under": 8.0, "wrong": 10.0, "miss": 8.0, "width": 2.0, "abstain": 5.0}
DISPLAY = {
    "findver": "FinDVer-style",
    "finfact": "Fin-Fact-style",
    "selfrag": "Self-RAG-style",
    "crag": "CRAG-style",
    "flare": "FLARE-style",
    "adaptive": "Adaptive-RAG-style",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def actual_loss(probabilities: list[float], truth: str) -> tuple[float, dict[str, Any]]:
    readout = full_action_readout(probabilities, LABELS, LOSS)
    action = readout["action"]
    if action == "answer":
        value = 0.0 if readout["point"] == truth else (
            LOSS["over"] if readout["point"] == LABELS[-1] and truth == LABELS[0]
            else LOSS["under"] if readout["point"] == LABELS[0] and truth == LABELS[-1]
            else LOSS["wrong"]
        )
    elif action == "adjudicate":
        value = (0.0 if truth in readout["set"] else LOSS["miss"]) + LOSS["width"] * max(0, len(readout["set"]) - 1)
    else:
        value = LOSS["abstain"]
    return float(value), readout


def variant_losses(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    grouped: dict[str, list[tuple[list[float], str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("variant")), []).append((row["probabilities"], row["truth"]))
    result = {}
    for variant, values in grouped.items():
        losses = []
        correct = 0
        for probabilities, truth in values:
            loss, readout = actual_loss(probabilities, truth)
            losses.append(loss)
            correct += int(readout.get("point") == truth)
        result[variant] = {
            "n": len(values),
            "accuracy": correct / max(len(values), 1),
            "decision_loss": sum(losses) / max(len(losses), 1),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="paper/results/cloud")
    args = parser.parse_args()
    root = REPO_ROOT / args.root
    payload: dict[str, Any] = {"schema": "finplan-cloud-results-v1", "backbones": {}}
    lines = ["# Cloud three-backbone results", ""]

    for backbone in ("deepseek", "glm", "qwen3"):
        block: dict[str, Any] = {}
        fusion = root / backbone / "evolution" / "temporal_fusion_new_cloud.json"
        if fusion.exists():
            block["fusion"] = load_json(fusion)
        qual = root / backbone / "qualification" / "summary.json"
        if qual.exists():
            block["qualification"] = load_json(qual)
        ashare = root / backbone / "ashare" / "summary.json"
        if ashare.exists():
            block["ashare"] = load_json(ashare)
        for kind in ("named", "official"):
            losses: dict[str, dict[str, Any]] = {}
            for prob in sorted((root / backbone / kind).glob("prob_*.jsonl")):
                losses.update(variant_losses(prob))
            if losses:
                block[kind] = losses
            metrics = root / backbone / kind / "metrics_summary.json"
            if metrics.exists():
                block[f"{kind}_metrics"] = load_json(metrics)
        payload["backbones"][backbone] = block

    (root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines += ["## Full-action decision loss", "", "| Backbone | Fusion | RF | LLM-only | Best named | Best official |", "|---|---:|---:|---:|---:|---:|"]
    for backbone, block in payload["backbones"].items():
        fusion = block.get("fusion") or {}
        named = block.get("named") or {}
        official = block.get("official") or {}
        best_named = min(named.items(), key=lambda item: item[1]["decision_loss"]) if named else ("", {"decision_loss": float("nan")})
        best_official = min(official.items(), key=lambda item: item[1]["decision_loss"]) if official else ("", {"decision_loss": float("nan")})
        lines.append(
            f"| {backbone} | {fusion.get('test_full_action', {}).get('decision_loss', float('nan')):.3f} | "
            f"{fusion.get('test_rf_only', {}).get('decision_loss', float('nan')):.3f} | "
            f"{fusion.get('test_llm_only', {}).get('decision_loss', float('nan')):.3f} | "
            f"{DISPLAY.get(best_named[0], best_named[0])} {best_named[1]['decision_loss']:.3f} | "
            f"{DISPLAY.get(best_official[0], best_official[0])} {best_official[1]['decision_loss']:.3f} |"
        )
    lines += ["", "## Qualification ablation", "", "| Backbone | Full | Accept-all | Δ [95% CI] |", "|---|---:|---:|---:|"]
    for backbone, block in payload["backbones"].items():
        qual = block.get("qualification") or {}
        policies = qual.get("policies") or {}
        diff = (qual.get("paired_differences") or {}).get("accept_all") or {}
        if policies:
            lines.append(
                f"| {backbone} | {policies['full_qualification']['decision_loss']:.3f} | "
                f"{policies['accept_all']['decision_loss']:.3f} | "
                f"{diff.get('mean_loss_difference_vs_full', float('nan')):+.3f} "
                f"[{diff.get('ci_low', float('nan')):+.3f}, {diff.get('ci_high', float('nan')):+.3f}] |"
            )
    lines += ["", "## A-share historical backtest", "", "| Backbone | Full | No bridge |", "|---|---:|---:|"]
    for backbone, block in payload["backbones"].items():
        ashare = block.get("ashare") or {}
        methods = ashare.get("methods") or {}
        if methods:
            lines.append(
                f"| {backbone} | {methods['full']['accuracy']:.3f} / {methods['full']['decision_loss']:.3f} | "
                f"{methods['no_bridge']['accuracy']:.3f} / {methods['no_bridge']['decision_loss']:.3f} |"
            )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
