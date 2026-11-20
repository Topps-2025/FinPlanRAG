"""Metrics and paired comparisons for the Tier-1 external experiments.

Usage:
    PYTHONPATH="src:scripts" python scripts/analyze_external_tier1.py \
      --kind findver --input paper/results/external_tier1/findver/testmini/deepseek \
      --out paper/results/external_tier1/findver/testmini/deepseek/analysis
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from external_common import (  # noqa: E402
    aurc,
    bootstrap_ci,
    macro_f1,
    paired_bootstrap,
    read_jsonl,
    safe_div,
    write_json,
)

VERDICT_LABELS = ("entailed", "refuted", "insufficient")


def load_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def is_answered(kind: str, row: dict[str, Any]) -> bool:
    parsed = row.get("parsed")
    if not parsed:
        return False
    if kind == "findver":
        return parsed.get("verdict") in {"entailed", "refuted"}
    answer = str(parsed.get("answer", "")).strip().lower()
    return answer not in {"", "insufficient", "unknown", "n/a", "none"}


def is_correct(kind: str, row: dict[str, Any]) -> bool:
    if not row.get("parsed"):
        return False
    if kind == "findver":
        return row["parsed"].get("verdict") == row.get("truth")
    return bool(row.get("correct"))


def prediction(kind: str, row: dict[str, Any]) -> str:
    parsed = row.get("parsed") or {}
    if kind == "findver":
        return str(parsed.get("verdict", "parse_failure"))
    answer = str(parsed.get("answer", "")).strip()
    return answer if answer else "parse_failure"


def condition_metrics(kind: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    parsed_rows = [row for row in rows if row.get("parsed")]
    answered = [row for row in rows if is_answered(kind, row)]
    correct = [is_correct(kind, row) for row in rows]
    confidences = [
        float((row.get("parsed") or {}).get("confidence", 0.0)) if row.get("parsed") else 0.0
        for row in rows
    ]
    answered_mask = [is_answered(kind, row) for row in rows]
    f1_3way = None
    if kind == "findver":
        truths = [row.get("truth") for row in rows]
        preds = [prediction(kind, row) for row in rows]
        binary_preds = [pred if pred in {"entailed", "refuted"} else "__abstain__" for pred in preds]
        f1_all = macro_f1(truths, binary_preds, ("entailed", "refuted"))
        f1_3way = macro_f1(truths, preds, VERDICT_LABELS)
        answered_rows = [row for row in rows if is_answered(kind, row)]
        f1_answered = macro_f1(
            [row.get("truth") for row in answered_rows],
            [prediction(kind, row) for row in answered_rows],
            ("entailed", "refuted"),
        ) if answered_rows else 0.0
    else:
        f1_all = safe_div(sum(float(row.get("f1", 0.0)) for row in rows), n)
        f1_answered = safe_div(
            sum(float(row.get("f1", 0.0)) for row in answered), max(1, len(answered))
        )

    accuracy = safe_div(sum(correct), n)
    selective_accuracy = safe_div(
        sum(is_correct(kind, row) for row in answered), max(1, len(answered))
    )
    coverage = safe_div(len(answered), n)
    acc_values = [1.0 if value else 0.0 for value in correct]
    acc_ci = bootstrap_ci(acc_values)
    recalls = [
        float(row["silver_recall"])
        for row in rows
        if row.get("silver_recall") is not None
    ] if kind == "docfinqa" else [
        float(row["evidence_recall"])
        for row in rows
        if row.get("evidence_recall") is not None
    ]
    input_tokens = [
        float((row.get("usage") or {}).get("input_tokens", 0) or 0) for row in rows
    ]
    output_tokens = [
        float((row.get("usage") or {}).get("output_tokens", 0) or 0) for row in rows
    ]
    num_paragraphs = [float(row.get("num_paragraphs", 0) or 0) for row in rows]
    return {
        "n": n,
        "parse_rate": safe_div(len(parsed_rows), n),
        "coverage": coverage,
        "accuracy": accuracy,
        "accuracy_ci": list(acc_ci),
        "selective_accuracy": selective_accuracy,
        "macro_f1": f1_all,
        "macro_f1_3way": f1_3way,
        "macro_f1_answered": f1_answered,
        "aurc": aurc(confidences, correct, answered_mask),
        "mean_paragraphs": statistics.fmean(num_paragraphs) if num_paragraphs else 0.0,
        "mean_input_tokens": statistics.fmean(input_tokens) if input_tokens else 0.0,
        "mean_output_tokens": statistics.fmean(output_tokens) if output_tokens else 0.0,
        "mean_total_tokens": statistics.fmean(
            [a + b for a, b in zip(input_tokens, output_tokens)]
        ) if input_tokens else 0.0,
        "mean_evidence_recall": statistics.fmean(recalls) if recalls else None,
        "n_evidence_recall": len(recalls),
    }


def aligned(rows: Sequence[dict[str, Any]], condition: str) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("example_id")): row
        for row in rows
        if row.get("condition") == condition
    }


def paired_comparison(
    kind: str,
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    keys = sorted(set(left).intersection(right))
    if not keys:
        return {}
    left_correct = [1.0 if is_correct(kind, left[key]) else 0.0 for key in keys]
    right_correct = [1.0 if is_correct(kind, right[key]) else 0.0 for key in keys]
    left_cost = [float(left[key].get("num_paragraphs", 0) or 0) for key in keys]
    right_cost = [float(right[key].get("num_paragraphs", 0) or 0) for key in keys]
    left_tokens = [
        float((left[key].get("usage") or {}).get("input_tokens", 0) or 0)
        + float((left[key].get("usage") or {}).get("output_tokens", 0) or 0)
        for key in keys
    ]
    right_tokens = [
        float((right[key].get("usage") or {}).get("input_tokens", 0) or 0)
        + float((right[key].get("usage") or {}).get("output_tokens", 0) or 0)
        for key in keys
    ]
    acc_diff = paired_bootstrap(left_correct, right_correct)
    cost_diff = paired_bootstrap(left_cost, right_cost)
    token_diff = paired_bootstrap(left_tokens, right_tokens)
    return {
        "n_pairs": len(keys),
        "accuracy_diff": acc_diff["mean_diff"],
        "accuracy_ci": [acc_diff["lo"], acc_diff["hi"]],
        "accuracy_p_two_sided": acc_diff["p_two_sided"],
        "cost_diff": cost_diff["mean_diff"],
        "cost_ci": [cost_diff["lo"], cost_diff["hi"]],
        "token_diff": token_diff["mean_diff"],
        "token_ci": [token_diff["lo"], token_diff["hi"]],
    }


def cost_accuracy_auc(metrics: dict[str, dict[str, Any]]) -> float | None:
    points = sorted(
        (item["mean_paragraphs"], item["accuracy"])
        for item in metrics.values()
        if item["mean_paragraphs"] > 0
    )
    if len(points) < 2:
        return None
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    span = points[-1][0] - points[0][0]
    return area / span if span else None


def render_report(kind: str, metrics: dict[str, Any], paired: dict[str, Any], config: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Tier-1 external: {kind}")
    lines.append("")
    lines.append(f"- provider: `{config.get('provider')}` / `{config.get('model')}`")
    lines.append(f"- examples: {config.get('examples')}")
    lines.append(f"- conditions: {', '.join(sorted(key for key in metrics if not key.startswith('_')))}")
    lines.append("")
    lines.append("## Condition metrics")
    lines.append("")
    lines.append("| condition | n | acc | acc 95% CI | coverage | selective acc | macro F1 | AURC | paragraphs | input tokens | evidence recall |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in sorted(key for key in metrics if not key.startswith("_")):
        item = metrics[name]
        ci = item["accuracy_ci"]
        recall = item["mean_evidence_recall"]
        lines.append(
            f"| {name} | {item['n']} | {item['accuracy']:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {item['coverage']:.3f} | "
            f"{item['selective_accuracy']:.3f} | {item['macro_f1']:.3f} | "
            f"{item['aurc']:.3f} | {item['mean_paragraphs']:.2f} | "
            f"{item['mean_input_tokens']:.0f} | "
            f"{'n/a' if recall is None else f'{recall:.3f}'} |"
        )
    lines.append("")
    lines.append("## Paired comparisons")
    lines.append("")
    lines.append("| comparison | n | accuracy diff | 95% CI | p (two-sided) | paragraph diff | token diff |")
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    for name in sorted(paired):
        item = paired[name]
        if not item:
            continue
        lines.append(
            f"| {name} | {item['n_pairs']} | {item['accuracy_diff']:+.3f} | "
            f"[{item['accuracy_ci'][0]:+.3f}, {item['accuracy_ci'][1]:+.3f}] | "
            f"{item['accuracy_p_two_sided']:.3f} | {item['cost_diff']:+.2f} | "
            f"{item.get('token_diff', float('nan')):+.0f} |"
        )
    lines.append("")
    auc = metrics.get("_cost_accuracy_auc")
    if auc is not None:
        lines.append(f"Cost-accuracy AUC (accuracy per paragraph): {auc:.4f}")
        lines.append("")
    lines.append("## Boundaries")
    lines.append("")
    lines.append("- Conditions share the same reader, provider, and paragraph budget; only the evidence policy differs.")
    lines.append("- FinDVer evidence recall is computed against the released gold paragraph indices.")
    lines.append("- DocFinQA evidence recall is a silver proxy from FinQA program constants; it is not a human evidence annotation.")
    lines.append("- Parse failures are scored as incorrect, not as abstentions.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=["findver", "docfinqa"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input)
    rows = load_rows(input_dir / "per_example.jsonl")
    config = json.loads((input_dir / "run_config.json").read_text(encoding="utf-8"))
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition"))].append(row)
    metrics = {
        name: condition_metrics(args.kind, group)
        for name, group in sorted(by_condition.items())
    }
    metrics["_cost_accuracy_auc"] = cost_accuracy_auc(
        {name: item for name, item in metrics.items() if not name.startswith("_")}
    )
    paired: dict[str, Any] = {}
    planner = aligned(rows, "planner")
    if planner:
        for name in sorted(by_condition):
            if name == "planner":
                continue
            paired[f"planner_vs_{name}"] = paired_comparison(
                args.kind, planner, aligned(rows, name)
            )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics_summary.json", {"config": config, "metrics": metrics, "paired": paired})
    (out_dir / "report.md").write_text(
        render_report(args.kind, metrics, paired, config), encoding="utf-8"
    )
    print(f"wrote {out_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
