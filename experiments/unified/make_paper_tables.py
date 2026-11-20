"""Generate LaTeX result tables and macros for the FinPlanRAG paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAIN_METHODS = [
    "FinDVer", "Fin-Fact", "Self-RAG", "CRAG", "FLARE",
    "Adaptive-RAG", "FinPlanRAG",
]
METHOD_ORDER = [
    "FinDVer",
    "Fin-Fact",
    "Self-RAG",
    "CRAG",
    "FLARE",
    "Adaptive-RAG",
    "AnchorEvolve-LLM",
    "FinPlan-RF",
    "FinPlanRAG",
]
BACKBONE_ORDER = ["deepseek", "glm", "qwen3"]
BACKBONE_NAME = {"deepseek": "DeepSeek", "glm": "GLM 5.3", "qwen3": "Qwen3-8B"}


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("0.") else text


def best(values: list[float], higher: bool = True) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return max(clean) if higher else min(clean)


def cell(value: float | None, target: float | None, digits: int = 3) -> str:
    text = fmt(value, digits)
    if target is not None and value is not None and abs(value - target) < 1e-12:
        return r"\textbf{" + text + "}"
    return text


def docfinqa_table(results: dict) -> str:
    conditions = [
        ("fixed_k_4", "Fixed-4"),
        ("fixed_k_16", "Fixed-16"),
        ("planner_fallback", r"\textit{FinPlan + fallback}"),
    ]
    named_rows = [
        ("self_rag", "Self-RAG"),
        ("crag", "CRAG"),
        ("flare", "FLARE"),
        ("adaptive_rag", "Adaptive-RAG"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{DocFinQA long-document question answering. All conditions use the same reader and backbone; only the evidence policy changes. ``Paras'' is the mean number of acquired paragraphs and ``Cov.'' the answer coverage.}",
        r"\label{tab:docfinqa}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Condition & Acc. & F1 & Paras & Tokens & Cov. \\",
        r"\midrule",
    ]
    for b in ("deepseek", "glm"):
        entry = results["backbones"].get(b, {})
        metrics = entry.get("docfinqa")
        if not metrics:
            continue
        lines.append(
            r"\multicolumn{6}{l}{\textit{" + BACKBONE_NAME[b] + r"}} \\"
        )
        for key, label in conditions:
            item = metrics.get(key)
            if not item:
                continue
            lines.append(
                f"{label} & {fmt(item.get('accuracy'))} & {fmt(item.get('macro_f1'))} & "
                f"{item.get('mean_paragraphs', 0):.2f} & "
                f"{item.get('mean_input_tokens', 0):.0f} & "
                f"{fmt(item.get('coverage'))} \\\\"
            )
        named = entry.get("docfinqa_named") or {}
        for key, label in named_rows:
            item = named.get(key)
            if not item:
                continue
            lines.append(
                f"{label} & {fmt(item.get('accuracy'))} & {fmt(item.get('macro_f1'))} & "
                f"{item.get('mean_paragraphs', 0):.2f} & "
                f"{item.get('mean_input_tokens', 0):.0f} & "
                f"{fmt(item.get('coverage'))} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def vague_table(results: dict) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\caption{Revenue-guidance adjudication on 227 XBRL-verified test claims. Acc: accuracy; Bal: balanced accuracy; Macro: macro F1; Loss: asymmetric decision loss (lower is better). FinPlanRAG is the full system. Internal controls are reported in the ablation table.}",
        r"\label{tab:vague}",
        r"\resizebox{0.92\textwidth}{!}{%",
        r"\begin{tabular}{l" + "rrrr" * len(BACKBONE_ORDER) + "}",
        r"\toprule",
    ]
    header = ["Method"]
    for b in BACKBONE_ORDER:
        header.extend([BACKBONE_NAME[b] + " Acc", "Bal", "Macro", "Loss"])
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    targets = {}
    for b in BACKBONE_ORDER:
        entry = results["backbones"].get(b, {})
        methods = entry.get("methods", {})
        for metric, higher in (("accuracy", True), ("balanced_accuracy", True),
                               ("macro_f1", True), ("decision_loss", False)):
            values = [methods.get(m, {}).get(metric) for m in MAIN_METHODS]
            targets[(b, metric)] = best(values, higher)
    for method in MAIN_METHODS:
        cells = [method]
        for b in BACKBONE_ORDER:
            item = results["backbones"].get(b, {}).get("methods", {}).get(method, {})
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "decision_loss"):
                cells.append(cell(item.get(metric), targets.get((b, metric))))
        lines.append(" & ".join(cells) + r" \\")
        if method == "Adaptive-RAG":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
    return "\n".join(lines)


def ablation_table(results: dict) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Ablation of the anchor-conditioned belief. $w$ is the weight on the LLM anchor posterior; $w=0$ is FinPlan-RF and $w=1$ is AnchorEvolve-LLM.}",
        r"\label{tab:ablation}",
        r"\resizebox{0.92\textwidth}{!}{%",
        r"\begin{tabular}{l" + "rrrr" * 2 + "}",
        r"\toprule",
        r"Method & DeepSeek Acc & Bal & Macro & Loss & GLM Acc & Bal & Macro & Loss \\",
        r"\midrule",
    ]
    grids = {}
    for b in ("deepseek", "glm"):
        grids[b] = results["backbones"].get(b, {}).get("ensemble_grid", {})
    rows = [
        ("AnchorEvolve-LLM ($w=1$)", "1.0"),
        ("FinPlan-RF ($w=0$)", "0.0"),
        ("Fusion $w=.1$", "0.1"),
        ("Fusion $w=.2$", "0.2"),
        ("Fusion $w=.3$", "0.3"),
        ("Fusion $w=.4$", "0.4"),
    ]
    targets = {}
    for b in ("deepseek", "glm"):
        for metric, higher in (("accuracy", True), ("balanced_accuracy", True),
                               ("macro_f1", True), ("decision_loss", False)):
            values = [grids[b].get(w, {}).get(metric) for _, w in rows]
            targets[(b, metric)] = best(values, higher)
    for label, weight in rows:
        cells = [label]
        for b in ("deepseek", "glm"):
            item = grids[b].get(weight, {})
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "decision_loss"):
                cells.append(cell(item.get(metric), targets.get((b, metric))))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
    return "\n".join(lines)


def router_table(results: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Mixed-workload routing over 300 DocFinQA questions and 273 vague claims. Utility is the task-specific score defined in Section~\ref{sec:metrics}.}",
        r"\label{tab:router}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Backbone & Route acc. & Router & Always RAG & Always anchor & Oracle \\",
        r"\midrule",
    ]
    for b in BACKBONE_ORDER:
        router = results["backbones"].get(b, {}).get("router")
        if not router:
            continue
        lines.append(
            f"{BACKBONE_NAME[b]} & {fmt(router.get('routing_accuracy'))} & "
            f"{fmt(router.get('mixed_utility_router'))} & "
            f"{fmt(router.get('mixed_utility_always_rag'))} & "
            f"{fmt(router.get('mixed_utility_always_anchor'))} & "
            f"{fmt(router.get('mixed_utility_oracle'))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}", ""]
    return "\n".join(lines)


def findver_table(results: dict) -> str:
    conditions = [
        ("fixed_k_4", "Fixed-4"),
        ("fixed_k_8", "Fixed-8"),
        ("fixed_k_16", "Fixed-16"),
        ("planner", r"\textit{FinPlan planner}"),
        ("planner_fallback", r"\textit{FinPlan + fallback}"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{FinDVer testmini transfer with the default reader.}",
        r"\label{tab:findver}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Backbone & Condition & Acc. & Macro F1 & Paras & Coverage \\",
        r"\midrule",
    ]
    for b in BACKBONE_ORDER:
        metrics = results["backbones"].get(b, {}).get("findver")
        if not metrics:
            continue
        lines.append(r"\multicolumn{6}{l}{\textit{" + BACKBONE_NAME[b] + r"}} \\")
        for key, label in conditions:
            item = metrics.get(key)
            if not item:
                continue
            lines.append(
                f"{label} & {fmt(item.get('accuracy'))} & {fmt(item.get('macro_f1'))} & "
                f"{item.get('mean_paragraphs', 0):.2f} & {fmt(item.get('coverage'))} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def macros(results: dict) -> str:
    def macro(name: str, value: str) -> str:
        return "\\newcommand{\\" + name + "}{" + value + "}"

    lines = ["% Generated by make_paper_tables.py"]
    for b, tag in (("deepseek", "DS"), ("glm", "GLM"), ("qwen3", "QW")):
        entry = results["backbones"].get(b, {})
        methods = entry.get("methods", {})
        ours = methods.get("FinPlanRAG", {})
        base = [
            methods.get(m, {}).get("balanced_accuracy", 0)
            for m in METHOD_ORDER[:6]
        ]
        base_i = max(range(len(base)), key=base.__getitem__)
        base_item = methods.get(METHOD_ORDER[base_i], {})
        lines += [
            macro(f"{tag}VAcc", fmt(ours.get("accuracy"))),
            macro(f"{tag}VBal", fmt(ours.get("balanced_accuracy"))),
            macro(f"{tag}VMacro", fmt(ours.get("macro_f1"))),
            macro(f"{tag}VLoss", fmt(ours.get("decision_loss"))),
            macro(f"{tag}VBestAcc", fmt(base_item.get("accuracy"))),
            macro(f"{tag}VBestBal", fmt(base_item.get("balanced_accuracy"))),
            macro(f"{tag}VBestMacro", fmt(base_item.get("macro_f1"))),
            macro(f"{tag}VBestLoss", fmt(base_item.get("decision_loss"))),
            macro(f"{tag}BestBaseName", METHOD_ORDER[base_i]),
        ]
        router = entry.get("router", {})
        lines += [
            macro(f"{tag}RouteAcc", fmt(router.get("routing_accuracy"))),
            macro(f"{tag}RouteUtil", fmt(router.get("mixed_utility_router"))),
            macro(f"{tag}AlwaysRAG", fmt(router.get("mixed_utility_always_rag"))),
            macro(f"{tag}AlwaysAnchor", fmt(router.get("mixed_utility_always_anchor"))),
            macro(f"{tag}Oracle", fmt(router.get("mixed_utility_oracle"))),
        ]
        metrics = entry.get("docfinqa", {})
        planner = metrics.get("planner", {})
        planner16 = metrics.get("planner_fallback") or metrics.get("planner16", planner)
        fixed16 = metrics.get("fixed_k_16", {})
        lines += [
            macro(f"{tag}DocAcc", fmt(planner16.get("accuracy"))),
            macro(f"{tag}DocParas", fmt(planner16.get("mean_paragraphs"), 2)),
            macro(f"{tag}DocFixedAcc", fmt(fixed16.get("accuracy"))),
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_docfinqa.tex").write_text(docfinqa_table(results), encoding="utf-8")
    (out / "table_vague.tex").write_text(vague_table(results), encoding="utf-8")
    (out / "table_ablation.tex").write_text(ablation_table(results), encoding="utf-8")
    (out / "table_router.tex").write_text(router_table(results), encoding="utf-8")
    (out / "table_findver.tex").write_text(findver_table(results), encoding="utf-8")
    (out / "macros.tex").write_text(macros(results), encoding="utf-8")
    print("wrote tables to", out)


if __name__ == "__main__":
    main()
