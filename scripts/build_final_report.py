"""Aggregate the frozen experiment artifacts into one auditable report."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "paper" / "results"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def best_row(rows: dict[str, dict], metric: str):
    name, values = max(rows.items(), key=lambda item: item[1].get(metric, float("-inf")))
    return {"model": name, "value": values.get(metric)}


def parse_ensemble(report_path: Path) -> dict:
    """Parse the five-seed ensemble table from a benchmark report."""
    if not report_path.exists():
        return {}
    values = {}
    in_section = False
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if "Five-fold-seed probability ensemble" in line:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line.startswith("|") and "metric" not in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) >= 3 and cells[0] not in ("---",):
                try:
                    values[cells[0]] = {
                        "model": float(cells[1]),
                        "always_above": float(cells[2]),
                    }
                except ValueError:
                    continue
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "paper/data/dataset_v4.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT / "final_remote")
    args = parser.parse_args()
    root = args.root
    args.out.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "schema": "finplan-final-report-v1",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "dataset_sha256": sha256(args.data),
        },
        "benchmark": {},
        "portfolio": {},
        "named_baselines": {},
        "official_named_baselines": {},
        "classical": {},
        "rolling": {},
        "router": {},
        "sequential": {},
        "controlled": {},
    }

    for label, relative in (
        ("all_labels", "workflow/benchmark/summary.json"),
        ("verified", "workflow/benchmark/verified/summary.json"),
    ):
        path = root / relative
        if path.exists():
            data = read_json(path)
            report["benchmark"][label] = {
                "n_claims_per_seed": data.get("n_claims_per_seed"),
                "pooled": data.get("pooled"),
                "ensemble": parse_ensemble(path.parent / "report.md"),
                "per_seed": data.get("per_seed"),
                "protocol": data.get("protocol"),
            }
    report["benchmark"]["verified_metrics"] = (
        read_json(root / "benchmark_metrics_verified_calibrated.json")
        if (root / "benchmark_metrics_verified_calibrated.json").exists()
        else None
    )

    for label, relative in (
        ("ten_seed", "final_remote/portfolio_10seed.json"),
        ("confirmatory", "final_remote/portfolio_confirm.json"),
        ("confirmatory_summary", "final_remote/rf_confirm_summary.json"),
    ):
        path = root / relative
        if path.exists():
            data = read_json(path)
            if "rows" in data:
                regrets = [row["regret"] for row in data["rows"].values()]
                report["portfolio"][label] = {
                    "cells": len(regrets),
                    "mean_regret": float(np.mean(regrets)),
                    "max_regret": float(np.max(regrets)),
                    "selected": {
                        name: sum(1 for row in data["rows"].values() if row["selected"] == name)
                        for name in sorted({row["selected"] for row in data["rows"].values()})
                    },
                    "rows": data["rows"],
                }
            else:
                report["portfolio"][label] = data

    for backbone in ("deepseek", "glm", "qwen3"):
        directory = root / backbone / "named"
        metrics_path = directory / "metrics_summary.json"
        comparison_path = directory / "model_comparison_logistic.json"
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        entry = {
            "metrics": metrics,
            "best_balanced_accuracy": best_row(metrics, "balanced_accuracy"),
            "best_accuracy": best_row(metrics, "accuracy"),
            "best_auroc": best_row(metrics, "auroc_macro"),
            "best_aurc": min(
                metrics.items(), key=lambda item: item[1].get("aurc", float("inf"))
            ),
        }
        if comparison_path.exists():
            comparison = read_json(comparison_path)
            entry["n_common"] = comparison.get("n_common")
            entry["finplan"] = comparison.get("models", {}).get("FinPlan_Logistic")
            entry["comparisons"] = comparison.get("comparisons")
        report["named_baselines"][backbone] = entry

    for backbone in ("deepseek", "glm", "qwen3"):
        directory = root / "official" / backbone
        metrics_path = directory / "metrics_summary.json"
        comparison_path = directory / "model_comparison_logistic.json"
        provenance_path = directory / "provenance.json"
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        entry = {
            "metrics": metrics,
            "best_balanced_accuracy": best_row(metrics, "balanced_accuracy"),
            "best_accuracy": best_row(metrics, "accuracy"),
            "best_auroc": best_row(metrics, "auroc_macro"),
            "best_aurc": min(
                metrics.items(), key=lambda item: item[1].get("aurc", float("inf"))
            ),
        }
        if comparison_path.exists():
            comparison = read_json(comparison_path)
            entry["n_common"] = comparison.get("n_common")
            entry["finplan"] = comparison.get("models", {}).get("FinPlan_Logistic")
            entry["comparisons"] = comparison.get("comparisons")
        if provenance_path.exists():
            entry["provenance"] = read_json(provenance_path)
        report["official_named_baselines"][backbone] = entry

    classical_path = root / "classical/summary_seed1_verified.json"
    if classical_path.exists():
        report["classical"] = read_json(classical_path)

    for model in ("logistic", "rf"):
        path = root / f"rolling_remote/metrics_{model}.json"
        if path.exists():
            data = read_json(path)
            report["rolling"][model] = {
                key: data[key]
                for key in (
                    "n",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "ece",
                    "brier",
                    "nll",
                    "selective_accuracy_50",
                    "aurc",
                )
            }

    router_path = root / "router_baseline/summary.json"
    if router_path.exists():
        report["router"] = read_json(router_path)

    sequential_path = root / "workflow/sequential_portfolio/summary.json"
    if sequential_path.exists():
        data = read_json(sequential_path)
        regrets = [row["regret"] for row in data.get("rows", [])]
        report["sequential"] = {
            "n_test": data.get("n_test"),
            "cells": len(regrets),
            "mean_regret": float(np.mean(regrets)) if regrets else None,
            "max_regret": float(np.max(regrets)) if regrets else None,
            "positive_regret_cells": int(sum(value > 1e-9 for value in regrets)),
            "selected": {
                name: sum(1 for row in data.get("rows", []) if row["selected"] == name)
                for name in sorted({row["selected"] for row in data.get("rows", [])})
            },
            "rows": data.get("rows"),
        }

    controlled_path = root / "controlled_remote.json"
    if controlled_path.exists():
        data = read_json(controlled_path)
        report["controlled"] = {
            "protocol": data.get("protocol"),
            "summary": data.get("summary"),
            "source_sha256": data.get("source_sha256"),
        }

    json_path = args.out / "final_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# FinPlan final experiment report",
        "",
        "## Environment",
        "",
        f"- Python {report['environment']['python']}",
        f"- numpy {report['environment']['numpy']}, scipy {report['environment']['scipy']}, scikit-learn {report['environment']['scikit_learn']}",
        f"- dataset SHA-256 `{report['environment']['dataset_sha256']}`",
        "",
        "## Benchmark",
        "",
        "| split | n | accuracy | balanced | macro F1 | AUROC | AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("all_labels", "verified"):
        data = report["benchmark"].get(label)
        if not data:
            continue
        pooled = data["pooled"]
        metrics = (
            report["benchmark"].get("verified_metrics")
            if label == "verified"
            else None
        ) or {}
        auroc = metrics.get("auroc_macro")
        auprc = metrics.get("auprc_macro")
        lines.append(
            f"| {label} (pooled) | {data['n_claims_per_seed']} | {pooled['accuracy']:.4f} | "
            f"{pooled['balanced_accuracy']:.4f} | {pooled['macro_f1']:.4f} | "
            f"{auroc:.4f} | {auprc:.4f} |"
            if auroc is not None
            else f"| {label} (pooled) | {data['n_claims_per_seed']} | {pooled['accuracy']:.4f} | "
            f"{pooled['balanced_accuracy']:.4f} | {pooled['macro_f1']:.4f} | -- | -- |"
        )
        ensemble = data.get("ensemble") or {}
        if ensemble:
            lines.append(
                f"| {label} (5-seed ensemble) | {data['n_claims_per_seed']} | "
                f"{ensemble.get('accuracy', {}).get('model', float('nan')):.4f} | "
                f"{ensemble.get('balanced accuracy', {}).get('model', float('nan')):.4f} | "
                f"{ensemble.get('macro F1', {}).get('model', float('nan')):.4f} | -- | -- |"
            )

    lines += ["", "## Named baselines", ""]
    lines.append(
        "| backbone | n common | FinPlan Bal | best Bal | FinPlan AUROC | best AUROC | best AURC |"
    )
    lines.append("|---|---:|---:|---|---:|---|---|")
    for backbone, data in report["named_baselines"].items():
        finplan = data.get("finplan") or {}
        lines.append(
            f"| {backbone} | {data.get('n_common', '')} | "
            f"{finplan.get('balanced_accuracy', float('nan')):.3f} | "
            f"{data['best_balanced_accuracy']['model']} {data['best_balanced_accuracy']['value']:.3f} | "
            f"{finplan.get('auroc_macro', float('nan')):.3f} | "
            f"{data['best_auroc']['model']} {data['best_auroc']['value']:.3f} | "
            f"{data['best_aurc'][0]} {data['best_aurc'][1]['aurc']:.3f} |"
        )

    lines += ["", "## Official-source named baselines", ""]
    lines.append(
        "| backbone | n common | FinPlan Bal | best Bal | FinPlan AUROC | best AUROC | best AURC |"
    )
    lines.append("|---|---:|---:|---|---:|---|---|")
    for backbone, data in report["official_named_baselines"].items():
        finplan = data.get("finplan") or {}
        lines.append(
            f"| {backbone} | {data.get('n_common', '')} | "
            f"{finplan.get('balanced_accuracy', float('nan')):.3f} | "
            f"{data['best_balanced_accuracy']['model']} {data['best_balanced_accuracy']['value']:.3f} | "
            f"{finplan.get('auroc_macro', float('nan')):.3f} | "
            f"{data['best_auroc']['model']} {data['best_auroc']['value']:.3f} | "
            f"{data['best_aurc'][0]} {data['best_aurc'][1]['aurc']:.3f} |"
        )

    lines += ["", "## Decision portfolio", ""]
    ten = report["portfolio"].get("ten_seed")
    if ten:
        lines.append(
            f"- Ten-seed portfolio: mean regret {ten['mean_regret']:.4f}, max {ten['max_regret']:.4f}, "
            f"selected {ten['selected']}."
        )
    confirm = report["portfolio"].get("confirmatory")
    if confirm:
        lines.append(
            f"- Confirmatory portfolio: mean regret {confirm['mean_regret']:.4f}, max {confirm['max_regret']:.4f}, "
            f"selected {confirm['selected']}."
        )
    seq = report["sequential"]
    if seq:
        lines.append(
            f"- Real earnings stream: {seq['cells']} cost cells, mean regret {seq['mean_regret']:.4f}, "
            f"max {seq['max_regret']:.4f}, positive-regret cells {seq['positive_regret_cells']}, "
            f"selected {seq['selected']}."
        )

    lines += ["", "## Other checks", ""]
    if report["rolling"]:
        for model, values in report["rolling"].items():
            lines.append(
                f"- Rolling {model}: n={values['n']}, acc {values['accuracy']:.3f}, "
                f"bal {values['balanced_accuracy']:.3f}, ECE {values['ece']:.3f}."
            )
    if report["router"]:
        base = report["router"].get("base", {})
        lines.append(
            f"- Router base: oracle-mode {base.get('oracle_router', float('nan')):.3f} vs "
            f"planner {base.get('planner', float('nan')):.3f}."
        )
    if report["classical"]:
        models = report["classical"].get("models", {})
        if "logistic" in models:
            lines.append(
                f"- Classical diagnostic (seed 1): FinPlan balanced .561; "
                f"logistic {models['logistic']['balanced_accuracy']:.3f}, "
                f"random forest {models['rf']['balanced_accuracy']:.3f}, "
                f"TF-IDF+numeric {models['tfidf_numeric']['balanced_accuracy']:.3f}."
            )
    if report["controlled"]:
        summary = report["controlled"].get("summary", {})
        if "finplan_state" in summary:
            lines.append(
                f"- Controlled: FinPlan state utility {summary['finplan_state']['utility']:.3f}, "
                f"accuracy {summary['finplan_state']['accuracy']:.3f}."
            )

    md_path = args.out / "final_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
