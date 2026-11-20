"""Aggregate the Tier-1 external results across backbones.

Usage:
    PYTHONPATH="src:scripts" python scripts/summarize_external_tier1.py \
      --root paper/results/external_tier1 \
      --out paper/results/external_tier1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def condition_table(dataset: str, backbone: str, metrics: dict[str, Any]) -> list[str]:
    lines = [
        f"### {dataset} / {backbone}",
        "",
        "| condition | n | acc | 95% CI | coverage | selective acc | macro F1 | AURC | paragraphs | tokens | evidence recall |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    values = metrics.get("metrics", {})
    for name in sorted(key for key in values if not key.startswith("_")):
        item = values[name]
        ci = item.get("accuracy_ci") or [float("nan"), float("nan")]
        recall = item.get("mean_evidence_recall")
        lines.append(
            f"| {name} | {item.get('n', 0)} | {item.get('accuracy', 0):.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {item.get('coverage', 0):.3f} | "
            f"{item.get('selective_accuracy', 0):.3f} | {item.get('macro_f1', 0):.3f} | "
            f"{item.get('aurc', 0):.3f} | {item.get('mean_paragraphs', 0):.2f} | "
            f"{item.get('mean_input_tokens', 0):.0f} | "
            f"{'n/a' if recall is None else f'{recall:.3f}'} |"
        )
    lines.append("")
    return lines


def comparison_table(dataset: str, backbone: str, metrics: dict[str, Any]) -> list[str]:
    lines = [
        f"### {dataset} / {backbone}: planner comparisons",
        "",
        "| comparison | n | accuracy diff | 95% CI | p | paragraph diff | token diff |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for name, item in sorted(metrics.get("paired", {}).items()):
        if not item:
            continue
        ci = item.get("accuracy_ci") or [float("nan"), float("nan")]
        lines.append(
            f"| {name} | {item.get('n_pairs', 0)} | {item.get('accuracy_diff', 0):+.3f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {item.get('accuracy_p_two_sided', float('nan')):.3f} | "
            f"{item.get('cost_diff', 0):+.2f} | {item.get('token_diff', float('nan')):+.0f} |"
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="paper/results/external_tier1")
    parser.add_argument("--out", default="paper/results/external_tier1")
    args = parser.parse_args()
    root = Path(args.root)
    expected = [
        ("FinDVer", "deepseek", "findver/testmini/deepseek"),
        ("FinDVer", "glm", "findver/testmini/glm"),
        ("FinDVer", "qwen3", "findver/testmini/qwen3"),
        ("FinDVer", "deepseek_official_reader", "findver/testmini/deepseek_official_reader"),
        ("DocFinQA", "deepseek", "docfinqa/test300/deepseek"),
        ("DocFinQA", "glm", "docfinqa/test300/glm"),
        ("DocFinQA", "qwen3", "docfinqa/test300/qwen3"),
    ]
    entries = []
    for dataset, backbone, relative in expected:
        data = load_metrics(root / relative / "analysis" / "metrics_summary.json")
        if data:
            entries.append((dataset, backbone, data))
    lines = [
        "# Tier-1 external validity summary",
        "",
        "This report aggregates the FinDVer and DocFinQA adapters.  Conditions",
        "share the same provider, reader, and evidence budget; only the evidence",
        "policy differs.  DeepSeek was the development backbone; GLM and Qwen3",
        "are replication backbones.  Parse failures are scored as incorrect.",
        "",
    ]
    for dataset in ("FinDVer", "DocFinQA"):
        subset = [entry for entry in entries if entry[0] == dataset]
        if not subset:
            continue
        lines.append(f"## {dataset}")
        lines.append("")
        for _, backbone, data in subset:
            lines.extend(condition_table(dataset, backbone, data))
            lines.extend(comparison_table(dataset, backbone, data))
    (Path(args.out) / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "entries": [
            {
                "dataset": dataset,
                "backbone": backbone,
                "config": data.get("config", {}),
                "metrics": data.get("metrics", {}),
                "paired": data.get("paired", {}),
            }
            for dataset, backbone, data in entries
        ]
    }
    (Path(args.out) / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {Path(args.out) / 'summary.md'} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
