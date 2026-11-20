"""Standard-metric evaluation across the existing and self-built datasets.

This script deliberately avoids the pilot's asymmetric decision loss.  It uses
only standard, auditable metrics:

- accuracy and macro F1 on held-out labels;
- credible-set coverage;
- expected calibration error (10 fixed bins);
- selective accuracy at 50% coverage and AURC;
- closure, wrong-document rate, and future leakage;
- paired McNemar tests and paired bootstrap intervals.

The goal is to answer a simple question: does the framework have a positive
result on real financial data under metrics that do not depend on hand-set
loss weights?

It reads the external FinPlanRAG storage and the sibling AnchorEvolve results.
No data is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path(os.environ.get("FINPLANRAG_STORAGE_ROOT", r"D:\Engineering\FinPlanRAG\database"))
ANCHOR = Path(
    os.environ.get(
        "FINPLANRAG_ANCHOREVOLVE_ROOT",
        str(REPO_ROOT.parent / "AnchorEvolve"),
    )
)

LOFIN = STORAGE / "preexperiments" / "results" / "lofin_full_benchmark_frozen.final.json"
FINGLM_BASE = STORAGE / "preexperiments" / "results" / "finglm_full_baselines.json"
FINGLM_INTERNAL = STORAGE / "preexperiments" / "results" / "finglm_full_internal_v1.json"
VALIDATION8 = STORAGE / "preexperiments" / "results" / "lofin_validation8_statistical_audit_v1.json"
V4 = ANCHOR / "real_data_v4" / "results_guidance_v4" / "results_guidance_v4.json"
V2 = ANCHOR / "real_data_v2" / "results_real_v2" / "results_real_v2.json"
SYNTH = ANCHOR / "results" / "per_claim.csv"
CONTROLLED = REPO_ROOT / "controlled_results.json"
OUT = REPO_ROOT / "paper" / "results" / "standard_eval"

LABELS = ("below", "within", "above")
SEED = 20261031


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else float("nan")


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return mean(1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return mean(scores)


def coverage(credible_sets: Sequence[Sequence[str]], y_true: Sequence[str]) -> float:
    return mean(1.0 if truth in credible else 0.0 for credible, truth in zip(credible_sets, y_true))


def per_class_recall(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, float]:
    result = {}
    for label in labels:
        support = sum(1 for truth in y_true if truth == label)
        correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        result[label] = correct / support if support else float("nan")
    return result


def ece(probs: Sequence[float], correct: Sequence[float], bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if probs.size == 0:
        return float("nan")
    indices = np.clip((probs * bins).astype(int), 0, bins - 1)
    total = probs.size
    value = 0.0
    for index in range(bins):
        mask = indices == index
        if not mask.any():
            continue
        value += mask.sum() / total * abs(correct[mask].mean() - probs[mask].mean())
    return float(value)


def selective_metrics(probs: Sequence[float], correct: Sequence[float], coverage_point: float = 0.5) -> dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if probs.size == 0:
        return {"selective_accuracy": float("nan"), "aurc": float("nan")}
    order = np.argsort(-probs, kind="stable")
    correct_sorted = correct[order]
    n = correct_sorted.size
    coverages = np.linspace(0.1, 1.0, 10)
    risks = []
    for cov in coverages:
        k = max(1, int(math.ceil(cov * n)))
        risks.append(1.0 - float(correct_sorted[:k].mean()))
    k = max(1, int(math.ceil(coverage_point * n)))
    return {
        "selective_accuracy": float(correct_sorted[:k].mean()),
        "aurc": float(np.mean(risks)),
    }


def mcnemar_exact(a_correct: Sequence[int], b_correct: Sequence[int]) -> dict[str, float]:
    a = np.asarray(a_correct, dtype=int)
    b = np.asarray(b_correct, dtype=int)
    b01 = int(((a == 1) & (b == 0)).sum())
    b10 = int(((a == 0) & (b == 1)).sum())
    n = b01 + b10
    if n == 0:
        return {"a_only": b01, "b_only": b10, "p": 1.0}
    if n > 100:
        # Normal approximation; exact binomial overflows for n in the thousands.
        z = abs(b01 - b10) / math.sqrt(n)
        p = math.erfc(z / math.sqrt(2.0))
        return {"a_only": b01, "b_only": b10, "p": float(min(1.0, p))}
    k = min(b01, b10)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"a_only": b01, "b_only": b10, "p": float(min(1.0, p))}


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 20000,
    seed: int = SEED,
) -> dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    diff = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for index in range(n_resamples):
        sample = rng.integers(0, diff.size, diff.size)
        means[index] = diff[sample].mean()
    return {
        "mean": float(diff.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n": int(diff.size),
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _requires(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing artifact: {path}")


def load_v4() -> dict[str, list[dict]]:
    _requires(V4)
    data = json.loads(V4.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in data["split_results"]:
        if row.get("split") == "test":
            grouped[str(row["method"])].append(row)
    return grouped


def load_v2() -> dict[str, list[dict]]:
    _requires(V2)
    data = json.loads(V2.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in data["results"]:
        if row.get("split") == "test":
            grouped[str(row["method"])].append(row)
    return grouped


def load_synthetic() -> dict[str, list[dict]]:
    _requires(SYNTH)
    grouped: dict[str, list[dict]] = defaultdict(list)
    with SYNTH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["round"]) >= 4:
                grouped[str(row["method"])].append(row)
    return grouped


def load_controlled() -> tuple[dict, list[dict]]:
    _requires(CONTROLLED)
    data = json.loads(CONTROLLED.read_text(encoding="utf-8"))
    return data["summary"], data["records"]


def _normalise_retrieval_row(row: Mapping[str, object]) -> dict[str, float]:
    return {
        "closure": float(row.get("cascade_closure", row.get("closure", 0.0)) or 0.0),
        "wrong_doc_rate": float(row.get("cascade_wrong_doc_rate", row.get("wrong_doc_rate", 0.0)) or 0.0),
        "future_leak": float(row.get("future_leak", 0.0) or 0.0),
        "queries": float(row.get("queries", 0.0) or 0.0),
    }


def load_lofin() -> dict[str, dict[str, dict]]:
    _requires(LOFIN)
    data = json.loads(LOFIN.read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in data["rows"]:
        grouped[str(row["method"])][str(row["case_id"])] = _normalise_retrieval_row(row)
    return grouped


def load_finglm() -> dict[str, dict[str, dict]]:
    _requires(FINGLM_BASE)
    _requires(FINGLM_INTERNAL)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in (FINGLM_BASE, FINGLM_INTERNAL):
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["rows"]:
            grouped[str(row["method"])][str(row["case_id"])] = _normalise_retrieval_row(row)
    return grouped


# ---------------------------------------------------------------------------
# Dataset evaluations
# ---------------------------------------------------------------------------


def eval_v4() -> dict:
    grouped = load_v4()
    methods = ("full", "always_above", "always_within", "current_rule", "guidance_mid", "full_no_anchor", "full_no_history")
    table = {}
    for method in methods:
        rows = grouped.get(method, [])
        if not rows:
            continue
        y_true = [str(row["true"]) for row in rows]
        y_pred = [str(row["pred"]) for row in rows]
        probs = [float(row["top_prob"]) for row in rows]
        correct = [1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred)]
        credible = [list(row.get("credible") or []) for row in rows]
        table[method] = {
            "n": len(rows),
            "accuracy": accuracy(y_true, y_pred),
            "macro_f1": macro_f1(y_true, y_pred, LABELS),
            "coverage": coverage(credible, y_true),
            "per_class_recall": per_class_recall(y_true, y_pred, LABELS),
            "ece": ece(probs, correct),
            **selective_metrics(probs, correct),
        }
    return {
        "dataset": "v4 revenue guidance (test, n=273)",
        "task": "3-way outcome classification",
        "methods": table,
        "primary_metric": "accuracy",
        "primary_baseline": "always_above",
        "strongest_control": "always_above",
    }


def eval_v2() -> dict:
    grouped = load_v2()
    methods = ("full", "full_strict", "static", "full_no_anchors", "full_no_independence")
    table = {}
    for method in methods:
        rows = grouped.get(method, [])
        if not rows:
            continue
        timing = [float(row.get("timing_correct", 0.0)) for row in rows]
        stage = [float(row.get("stage_correct", 0.0)) for row in rows]
        joint = [float(row.get("joint_correct", 0.0)) for row in rows]
        probs = [float(row.get("top_prob", 0.0)) for row in rows]
        table[method] = {
            "n": len(rows),
            "timing_accuracy": mean(timing),
            "stage_accuracy": mean(stage),
            "joint_accuracy": mean(joint),
            "coverage": mean(float(row.get("coverage", 0.0)) for row in rows),
            "assert_rate": mean(1.0 if row.get("decision") == "assert" else 0.0 for row in rows),
            "ece": ece(probs, timing),
            **selective_metrics(probs, timing),
        }
    return {
        "dataset": "v2 production milestones (test, n=7)",
        "task": "timing x stage adjudication",
        "methods": table,
        "primary_metric": "joint_accuracy",
        "primary_baseline": "static",
        "strongest_control": "static",
    }


def eval_synthetic() -> dict:
    grouped = load_synthetic()
    methods = ("full", "static", "management", "rag", "full_no_schema", "full_no_anchors", "full_no_independence", "full_no_vacuous")
    table = {}
    for method in methods:
        rows = grouped.get(method, [])
        if not rows:
            continue
        timing = [float(row["timing_correct"]) for row in rows]
        joint = [float(row["joint_correct"]) for row in rows]
        probs = [float(row["top_prob"]) for row in rows]
        table[method] = {
            "n": len(rows),
            "timing_accuracy": mean(timing),
            "joint_accuracy": mean(joint),
            "coverage": mean(float(row["coverage"]) for row in rows),
            "ece": ece(probs, timing),
            **selective_metrics(probs, timing),
        }
    return {
        "dataset": "AnchorEvolve synthetic (late rounds 4-6, n=30/ method)",
        "task": "timing x stage mechanism",
        "methods": table,
        "primary_metric": "timing_accuracy",
        "primary_baseline": "static",
        "strongest_control": "static",
        "synthetic": True,
    }


def eval_controlled() -> dict:
    summary, records = load_controlled()
    methods = ("single_shot", "generic_adaptive", "fixed_decomposition", "finplan_state", "finplan_no_counterevidence", "finplan_no_stop", "finplan_no_time")
    table = {}
    for method in methods:
        if method in summary:
            table[method] = {key: float(value) for key, value in summary[method].items() if isinstance(value, (int, float))}
    paired = {}
    for baseline in ("single_shot", "generic_adaptive", "fixed_decomposition"):
        for metric in ("closure", "overclaim", "counterevidence_recall", "queries", "utility", "future_leak"):
            by_key = {
                (row["scenario"], row["seed"], row["method"]): float(row[metric])
                for row in records
                if metric in row
            }
            keys = sorted({(scenario, seed) for scenario, seed, _ in by_key})
            a = [by_key[(s, seed, "finplan_state")] for s, seed in keys if (s, seed, "finplan_state") in by_key]
            b = [by_key[(s, seed, baseline)] for s, seed in keys if (s, seed, baseline) in by_key]
            paired[f"{baseline}:{metric}"] = paired_bootstrap(a, b)
    return {
        "dataset": "controlled planning diagnostic (5,600 cases, 7 scenarios x 20 seeds)",
        "task": "relation-evidence planning",
        "methods": table,
        "paired": paired,
        "primary_metric": "closure",
        "primary_baseline": "single_shot",
        "strongest_control": "generic_adaptive",
        "synthetic": True,
    }


def _retrieval_table(grouped: Mapping[str, Mapping[str, Mapping[str, float]]], dataset: str, methods: Sequence[str]) -> dict:
    table = {}
    for method in methods:
        rows = grouped.get(method, {})
        if not rows:
            continue
        table[method] = {
            "n": len(rows),
            "closure": mean(row["closure"] for row in rows.values()),
            "wrong_doc_rate": mean(row["wrong_doc_rate"] for row in rows.values()),
            "future_leak": mean(row.get("future_leak", 0.0) for row in rows.values()),
            "queries": mean(row.get("queries", 0.0) for row in rows.values()),
        }
    return {
        "dataset": dataset,
        "task": "typed filing retrieval",
        "methods": table,
        "primary_metric": "wrong_doc_rate",
        "primary_baseline": "single_shot",
        "strongest_control": "metadata_v9_fill" if "metadata_v9_fill" in grouped else "period_metadata_decomposition",
    }


def add_retrieval_paired(result: dict, grouped: Mapping[str, Mapping[str, Mapping[str, float]]], method: str, baselines: Sequence[str]) -> None:
    paired = {}
    method_rows = grouped.get(method, {})
    for baseline in baselines:
        baseline_rows = grouped.get(baseline, {})
        common = sorted(set(method_rows) & set(baseline_rows))
        if not common:
            continue
        a_closure = [1 if method_rows[c]["closure"] >= 0.999999 else 0 for c in common]
        b_closure = [1 if baseline_rows[c]["closure"] >= 0.999999 else 0 for c in common]
        a_wrong = [1 if method_rows[c]["wrong_doc_rate"] <= 1e-9 else 0 for c in common]
        b_wrong = [1 if baseline_rows[c]["wrong_doc_rate"] <= 1e-9 else 0 for c in common]
        paired[f"{baseline}:closure"] = mcnemar_exact(a_closure, b_closure)
        paired[f"{baseline}:wrong_doc"] = mcnemar_exact(a_wrong, b_wrong)
        paired[f"{baseline}:wrong_doc_bootstrap"] = paired_bootstrap(
            [method_rows[c]["wrong_doc_rate"] for c in common],
            [baseline_rows[c]["wrong_doc_rate"] for c in common],
        )
    result["paired"] = paired


def eval_validation8() -> dict:
    _requires(VALIDATION8)
    data = json.loads(VALIDATION8.read_text(encoding="utf-8"))
    table = {
        method: {"n": item["n"], "closure": item["closure"] / item["n"]}
        for method, item in data["closure_by_method"].items()
    }
    paired = {
        name: {
            "a_only": item["n10_a1_b0"],
            "b_only": item["n01_a0_b1"],
            "p": item["p_exact_mcnemar"],
        }
        for name, item in data["paired_closure_tests"].items()
    }
    return {
        "dataset": "validation8 preregistered holdout (24 questions)",
        "task": "typed period retrieval (closure)",
        "methods": table,
        "paired": paired,
        "primary_metric": "closure",
        "primary_baseline": "single_shot",
        "strongest_control": "metadata_v9_fill",
    }


def eval_lofin() -> dict:
    grouped = load_lofin()
    methods = ("finplan_v9_cascade", "metadata_v9_fill", "single_shot", "period_metadata_decomposition", "generic_adaptive_period", "hirec_period", "finplan_v8", "finplan_v7", "finplan_v4")
    result = _retrieval_table(grouped, "LOFin full frozen (n=1,572)", methods)
    add_retrieval_paired(result, grouped, "finplan_v9_cascade", ("single_shot", "period_metadata_decomposition", "generic_adaptive_period", "hirec_period", "metadata_v9_fill"))
    return result


def eval_finglm() -> dict:
    grouped = load_finglm()
    methods = ("finplan_v9_cascade", "metadata_v9_fill", "single_shot", "period_metadata_decomposition", "generic_adaptive_period", "hirec_period")
    result = _retrieval_table(grouped, "FinGLM full (n=1,829)", methods)
    add_retrieval_paired(result, grouped, "finplan_v9_cascade", ("single_shot", "period_metadata_decomposition", "generic_adaptive_period", "hirec_period", "metadata_v9_fill"))
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def verdict(dataset: dict) -> str:
    methods = dataset["methods"]
    primary = dataset["primary_metric"]
    baseline = dataset["strongest_control"]
    if "finplan_v9_cascade" in methods:
        method = "finplan_v9_cascade"
    elif "finplan_state" in methods:
        method = "finplan_state"
    elif "full" in methods:
        method = "full"
    else:
        return "no framework method"
    if method not in methods or baseline not in methods:
        return "missing method or baseline"
    ours = methods[method].get(primary)
    base = methods[baseline].get(primary)
    if ours is None or base is None:
        return "missing primary metric"
    if dataset["primary_metric"] == "wrong_doc_rate":
        better = ours < base - 1e-9
    else:
        better = ours > base + 1e-9
    tie = abs(ours - base) <= 1e-9
    if better:
        return f"positive vs {baseline} on {primary}"
    if tie:
        return f"tied with {baseline} on {primary}"
    return f"negative vs {baseline} on {primary}"


def render_report(results: dict) -> str:
    lines = [
        "# Standard-metric evaluation",
        "",
        "This report uses only standard, auditable metrics: accuracy, macro F1,",
        "coverage, ECE (10 fixed bins), selective accuracy at 50% coverage, AURC,",
        "closure, wrong-document rate, future leakage, paired McNemar tests, and",
        "paired bootstrap intervals. It does not use the pilot's asymmetric",
        "decision-loss weights.",
        "",
        "## Verdicts",
        "",
        "| dataset | task | framework method | strongest control | verdict |",
        "|---|---|---|---|---|",
    ]
    method_names = {
        "validation8 preregistered holdout (24 questions)": "finplan_v9_cascade",
        "LOFin full frozen (n=1,572)": "finplan_v9_cascade",
        "FinGLM full (n=1,829)": "finplan_v9_cascade",
        "v4 revenue guidance (test, n=273)": "full",
        "v2 production milestones (test, n=7)": "full",
        "AnchorEvolve synthetic (late rounds 4-6, n=30/ method)": "full",
        "controlled planning diagnostic (5,600 cases, 7 scenarios x 20 seeds)": "finplan_state",
    }
    for name, dataset in results.items():
        method = method_names.get(name, "")
        lines.append(
            f"| {name} | {dataset['task']} | {method} | {dataset['strongest_control']} | {verdict(dataset)} |"
        )
    lines += [
        "",
        "## Overall verdict",
        "",
        "- **Real-data outcome prediction (v4, v2): no positive result.** The full",
        "  framework ties the strongest control on accuracy / joint accuracy.",
        "- **Real-data retrieval (LOFin, FinGLM): positive vs weak baselines, tied",
        "  with the strongest same-obligation control.** v9 beats single-shot,",
        "  generic, and HiREC on wrong-document rate, but ties the metadata-fill",
        "  control. The period-metadata comparison is mixed (closure vs wrong-doc).",
        "- **Real-data parser holdout (validation8): positive vs parser versions and",
        "  weak baselines, tied with the metadata-fill control.** v9 closes 24/24",
        "  vs 4-11/24 for the baselines (p <= .00025); metadata_v9_fill also closes",
        "  24/24 (p = 1.0).",
        "- **Synthetic / controlled mechanism: positive.** The controlled planning",
        "  diagnostic and the synthetic self-evolution run show the mechanism",
        "  working; these are not real-data claims.",
        "",
        "**Conclusion:** the current framework has a robust **representation and",
        "controlled-mechanism** positive, but not a real-data **method/policy**",
        "positive against the strongest same-representation control. A real-data",
        "method claim requires the evidence-rich benchmark.",
        "",
    ]
    for name, dataset in results.items():
        lines += ["", f"## {name}", "", f"Task: {dataset['task']}.", ""]
        metrics = sorted({key for method in dataset["methods"].values() for key in method})
        display = [key for key in ("n", "accuracy", "macro_f1", "timing_accuracy", "joint_accuracy", "coverage", "closure", "wrong_doc_rate", "future_leak", "ece", "selective_accuracy", "aurc", "assert_rate") if key in metrics]
        lines.append("| method | " + " | ".join(display) + " |")
        lines.append("|---|" + "|".join("---:" for _ in display) + "|")
        for method, values in dataset["methods"].items():
            cells = []
            for key in display:
                value = values.get(key)
                cells.append(f"{value:.4f}" if isinstance(value, float) else ("" if value is None else str(value)))
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        paired = dataset.get("paired") or {}
        if paired:
            lines += ["", "Paired comparisons (framework vs baseline):", "", "| comparison | mean diff / a-only | 95% CI / b-only | p |", "|---|---:|---:|---:|"]
            for key, value in paired.items():
                if "p" in value:
                    lines.append(f"| {key} | {value['a_only']} | {value['b_only']} | {value['p']:.4g} |")
                else:
                    lines.append(f"| {key} | {value['mean']:.4f} | [{value['ci_low']:.4f}, {value['ci_high']:.4f}] | - |")
    lines += [
        "",
        "## Reading",
        "",
        "- A positive result requires the framework method to beat the **strongest control** on the primary metric, not only a weak baseline.",
        "- v4 and v2 are real-data outcome tasks; LOFin and FinGLM are real retrieval tasks; the synthetic and controlled rows are mechanism tests.",
        "- Do not pool synthetic and real rows. A synthetic positive is not a real-data positive.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    results = {
        "validation8 preregistered holdout (24 questions)": eval_validation8(),
        "LOFin full frozen (n=1,572)": eval_lofin(),
        "FinGLM full (n=1,829)": eval_finglm(),
        "v4 revenue guidance (test, n=273)": eval_v4(),
        "v2 production milestones (test, n=7)": eval_v2(),
        "AnchorEvolve synthetic (late rounds 4-6, n=30/ method)": eval_synthetic(),
        "controlled planning diagnostic (5,600 cases, 7 scenarios x 20 seeds)": eval_controlled(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "standard_eval.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = render_report(results)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
