"""Generate Chinese thesis tables from the final results JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAIN_METHODS = [
    "FinDVer", "Fin-Fact", "Self-RAG", "CRAG", "FLARE",
    "Adaptive-RAG", "FinPlanRAG",
]
METHOD_ORDER = [
    "FinDVer", "Fin-Fact", "Self-RAG", "CRAG", "FLARE",
    "Adaptive-RAG", "AnchorEvolve-LLM", "FinPlanRAG",
    "Shared calibrated RF diagnostic",
]
METHOD_CN = {
    "FinDVer": "FinDVer-style", "Fin-Fact": "Fin-Fact-style",
    "Self-RAG": "Self-RAG-style", "CRAG": "CRAG-style",
    "FLARE": "FLARE-style", "Adaptive-RAG": "Adaptive-RAG-style",
    "AnchorEvolve-LLM": "AnchorEvolve-LLM",
    "FinPlanRAG": r"FinPlanRAG（AnchorEvolve-Fusion, $w=0.1$）",
    "Shared calibrated RF diagnostic": r"共享校准RF（非LLM诊断）",
}
BACKBONES = [("deepseek", "DeepSeek"), ("glm", "GLM 5.3"), ("qwen3", "Qwen3-8B")]


def f(value, digits=3):
    if value is None:
        return "--"
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("0.") else text


def best(values, higher=True):
    clean = [v for v in values if v is not None]
    return (max(clean) if higher else min(clean)) if clean else None


def cell(value, target):
    text = f(value)
    if target is not None and value is not None and abs(value - target) < 1e-12:
        return r"\textbf{" + text + "}"
    return text


def docfinqa(results):
    conds = [
        ("fixed_k_4", "固定4段"), ("fixed_k_16", "固定16段"),
        ("planner_fallback", "规划器+回退"),
    ]
    named_rows = [("self_rag", "Self-RAG-style"), ("crag", "CRAG-style"),
                  ("flare", "FLARE-style"), ("adaptive_rag", "Adaptive-RAG-style")]
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\zihao{5}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{DocFinQA长文档问答结果。所有条件使用同一阅读器和基座，仅证据策略不同。}",
        r"\label{tab:docfinqa}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"条件 & 准确率 & 宏F1 & 平均段落 & 覆盖率 \\", r"\midrule",
    ]
    for b, name in BACKBONES[:2]:
        metrics = results["backbones"].get(b, {}).get("docfinqa")
        if not metrics:
            continue
        lines.append(r"\multicolumn{5}{l}{\textit{" + name + r"}} \\")
        for key, label in conds:
            item = metrics.get(key)
            if not item:
                continue
            lines.append(
                f"{label} & {f(item.get('accuracy'))} & {f(item.get('macro_f1'))} & "
                f"{item.get('mean_paragraphs', 0):.2f} & {f(item.get('coverage'))} \\\\"
            )
        named = results["backbones"].get(b, {}).get("docfinqa_named") or {}
        for key, label in named_rows:
            item = named.get(key)
            if not item:
                continue
            lines.append(
                f"{label} & {f(item.get('accuracy'))} & {f(item.get('macro_f1'))} & "
                f"{item.get('mean_paragraphs', 0):.2f} & {f(item.get('coverage'))} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def vague_main(results) -> str:
    rows = ["FinDVer", "Fin-Fact", "Self-RAG", "CRAG", "FLARE", "Adaptive-RAG",
            "AnchorEvolve-LLM", "FinPlanRAG", "Shared calibrated RF diagnostic"]
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "decision_loss"]
    labels = ["Acc", "Bal", "Macro", "Loss"]
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\zihao{5}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\caption{227条XBRL核验断言上的主结果。FinPlanRAG为包含LLM锚点信号的AnchorEvolve-Fusion，共享校准RF仅作非LLM诊断。Acc为准确率，Bal为平衡准确率，Macro为宏F1，Loss为非对称决策损失（越低越好）。完整指标见附录表\ref{tab:vague-ds}至表\ref{tab:vague-qwen}。}",
        r"\label{tab:vague}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l" + "rrrr" * 3 + "}", r"\toprule",
        r"方法 & DeepSeek Acc & Bal & Macro & Loss & GLM Acc & Bal & Macro & Loss & Qwen Acc & Bal & Macro & Loss \\",
        r"\midrule",
    ]
    targets = {}
    for b, _ in BACKBONES:
        methods = results["backbones"].get(b, {}).get("methods", {})
        for key, hi in zip(metrics, [True, True, True, False]):
            targets[(b, key)] = best([methods.get(m, {}).get(key) for m in rows], hi)
    for method in rows:
        cells = [METHOD_CN[method]]
        for b, _ in BACKBONES:
            item = results["backbones"].get(b, {}).get("methods", {}).get(method, {})
            for key in metrics:
                cells.append(cell(item.get(key), targets.get((b, key))))
        lines.append(" & ".join(cells) + r" \\")
        if method == "Adaptive-RAG":
            lines.append(r"\midrule")
        if method == "AnchorEvolve-LLM":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}", ""]
    return "\n".join(lines)


def vague_full(results, backbone: str, include_finplan: bool) -> str:
    methods = results["backbones"][backbone]["methods"]
    rows = ["FinDVer", "Fin-Fact", "Self-RAG", "CRAG", "FLARE", "Adaptive-RAG"]
    if include_finplan:
        rows.extend(["AnchorEvolve-LLM", "FinPlanRAG", "Shared calibrated RF diagnostic"])
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "mcc",
               "auroc_macro", "auprc_macro", "ece", "brier",
               "decision_loss", "selective_accuracy_50", "aurc"]
    higher = [True, True, True, True, True, True, False, False, False, True, False]
    targets = {}
    for key, hi in zip(metrics, higher):
        targets[key] = best([methods.get(m, {}).get(key) for m in rows], hi)
    label = {"deepseek": "DeepSeek", "glm": "GLM 5.3", "qwen3": "Qwen3-8B"}[backbone]
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\zihao{5}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\caption{" + label + r"基座上的模糊断言结果（227条XBRL核验断言）。Loss越低越好；Sel@50为50\%覆盖率下的选择性准确率，AURC越低越好。}",
        r"\label{tab:vague-" + ("ds" if backbone == "deepseek" else ("qwen" if backbone == "qwen3" else backbone)) + "}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrrr}", r"\toprule",
        r"方法 & Acc & Bal & Macro & MCC & AUROC & AUPRC & ECE & Brier & Loss & Sel@50 & AURC \\",
        r"\midrule",
    ]
    for method in rows:
        item = methods.get(method, {})
        cells = [METHOD_CN[method]] + [cell(item.get(k), targets[k]) for k in metrics]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}", ""]
    return "\n".join(lines)


def ablation(results):
    lines = [
        r"\begin{table*}[htbp]", r"\centering", r"\zihao{5}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{锚点后验融合权重$w$的消融。$w=0$为FinPlan-RF，$w=1$为纯LLM锚点后验。}",
        r"\label{tab:ablation}",
        r"\resizebox{0.96\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r"方法 & DeepSeek Acc & Bal & Macro & Loss & GLM Acc & Bal & Macro & Loss \\",
        r"\midrule",
    ]
    grids = {b: results["backbones"].get(b, {}).get("ensemble_grid", {}) for b, _ in BACKBONES[:2]}
    rows = [("AnchorEvolve-LLM ($w=1$)", "1.0"), ("FinPlan-RF ($w=0$)", "0.0"),
            ("融合 $w=.1$", "0.1"), ("融合 $w=.2$", "0.2"),
            ("融合 $w=.3$", "0.3"), ("融合 $w=.4$", "0.4")]
    targets = {}
    for b in ("deepseek", "glm"):
        for metric, higher in (("accuracy", True), ("balanced_accuracy", True),
                               ("macro_f1", True), ("decision_loss", False)):
            targets[(b, metric)] = best([grids[b].get(w, {}).get(metric) for _, w in rows], higher)
    for label, weight in rows:
        cells = [label]
        for b in ("deepseek", "glm"):
            item = grids[b].get(weight, {})
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "decision_loss"):
                cells.append(cell(item.get(metric), targets.get((b, metric))))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
    return "\n".join(lines)


def router(results):
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\zihao{5}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{混合负载路由结果。效用定义见第\ref{ch:experiments}章。}",
        r"\label{tab:router}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"基座 & 路由准确率 & 路由器 & 始终RAG & 始终Anchor & Oracle \\", r"\midrule",
    ]
    for b, name in BACKBONES:
        r = results["backbones"].get(b, {}).get("router")
        if not r:
            continue
        lines.append(
            f"{name} & {f(r.get('routing_accuracy'))} & {f(r.get('mixed_utility_router'))} & "
            f"{f(r.get('mixed_utility_always_rag'))} & {f(r.get('mixed_utility_always_anchor'))} & "
            f"{f(r.get('mixed_utility_oracle'))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}", ""]
    return "\n".join(lines)


def findver(results):
    conds = [("fixed_k_4", "固定4段"), ("fixed_k_8", "固定8段"),
             ("fixed_k_16", "固定16段"), ("planner", "规划器"),
             ("planner_fallback", "规划器+回退")]
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\zihao{5}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{FinDVer testmini迁移结果（默认阅读器）。}",
        r"\label{tab:findver}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"基座 & 条件 & 准确率 & 宏F1 & 平均段落 \\", r"\midrule",
    ]
    for b, name in BACKBONES:
        metrics = results["backbones"].get(b, {}).get("findver")
        if not metrics:
            continue
        lines.append(r"\multicolumn{5}{l}{\textit{" + name + r"}} \\")
        for key, label in conds:
            item = metrics.get(key)
            if not item:
                continue
            lines.append(
                f"{label} & {f(item.get('accuracy'))} & {f(item.get('macro_f1'))} & "
                f"{item.get('mean_paragraphs', 0):.2f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "docfinqa.tex").write_text(docfinqa(results), encoding="utf-8")
    (out / "vague_main.tex").write_text(vague_main(results), encoding="utf-8")
    (out / "vague_ds.tex").write_text(vague_full(results, "deepseek", True), encoding="utf-8")
    (out / "vague_glm.tex").write_text(vague_full(results, "glm", True), encoding="utf-8")
    (out / "vague_qwen.tex").write_text(vague_full(results, "qwen3", True), encoding="utf-8")
    (out / "ablation.tex").write_text(ablation(results), encoding="utf-8")
    (out / "router.tex").write_text(router(results), encoding="utf-8")
    (out / "findver.tex").write_text(findver(results), encoding="utf-8")
    print("wrote thesis tables to", out)


if __name__ == "__main__":
    main()
