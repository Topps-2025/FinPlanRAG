"""Build the three-panel main-results figure for the Chinese thesis.

Panel A: router accuracy and mixed utility.
Panel B: DocFinQA accuracy versus mean acquired paragraphs.
Panel C: AnchorEvolve balanced accuracy, macro F1, and decision loss
         against the strongest named baseline per backbone.

The figure is grayscale and uses Chinese labels. Data come from the final
results JSON, so the figure can be regenerated with the thesis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

BACKBONES = [("deepseek", "DeepSeek"), ("glm", "GLM"), ("qwen3", "Qwen")]
SHORT = {"DeepSeek": "DS", "GLM": "GLM", "Qwen": "Qwen"}
MAIN = ["FinDVer", "Fin-Fact", "Self-RAG", "CRAG", "FLARE", "Adaptive-RAG"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--font", default="C:/Windows/Fonts/simhei.ttf")
    args = parser.parse_args()
    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    font = FontProperties(fname=args.font)
    plt.rcParams.update({
        "font.family": font.get_name(),
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(7.4, 4.7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.05],
                          height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.42)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c1 = fig.add_subplot(gs[0, 2])
    ax_c2 = fig.add_subplot(gs[1, 2])

    # Panel A: router accuracy and mixed utility.
    acc = [data["backbones"][b]["router"]["routing_accuracy"] for b, _ in BACKBONES]
    util = [data["backbones"][b]["router"]["mixed_utility_router"] for b, _ in BACKBONES]
    oracle = [data["backbones"][b]["router"]["mixed_utility_oracle"] for b, _ in BACKBONES]
    x = np.arange(len(BACKBONES))
    ax_a.bar(x - 0.19, acc, width=0.36, color="#3a3a3a", label="路由准确率")
    ax_a.bar(x + 0.19, util, width=0.36, color="#b0b0b0", label="混合效用")
    ax_a.scatter(x, oracle, marker="_", s=180, color="black", linewidth=2.0, label="Oracle效用")
    for i, (a, u) in enumerate(zip(acc, util)):
        ax_a.text(i - 0.19, a + 0.03, f"{a:.3f}", ha="center", fontsize=6.5)
        ax_a.text(i + 0.19, u + 0.03, f"{u:.3f}", ha="center", fontsize=6.5)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([n for _, n in BACKBONES], fontproperties=font)
    ax_a.set_ylim(0, 1.12)
    ax_a.set_title("(a) 路由与混合负载", fontproperties=font)
    ax_a.grid(axis="y", linestyle=":", color="#cccccc", linewidth=0.6)

    # Panel B: DocFinQA accuracy versus evidence cost.
    budgets = [("fixed_k_4", "固定4段"), ("fixed_k_8", "固定8段"),
               ("fixed_k_16", "固定16段")]
    styles = {"deepseek": ("o", "-", "#222222", "DeepSeek"),
              "glm": ("s", "--", "#777777", "GLM")}
    for b, _ in BACKBONES[:2]:
        metrics = data["backbones"][b]["docfinqa"]
        xs = [metrics[k]["mean_paragraphs"] for k, _ in budgets]
        ys = [metrics[k]["accuracy"] for k, _ in budgets]
        marker, ls, color, label = styles[b]
        ax_b.plot(xs, ys, marker=marker, linestyle=ls, color=color, linewidth=1.1,
                  markersize=4.0, label=label + " 固定预算")
        fb = metrics["planner_fallback"]
        ax_b.scatter([fb["mean_paragraphs"]], [fb["accuracy"]], marker="*", s=80,
                     color=color, zorder=5, label=label + " 回退")
    ax_b.set_xlabel("平均段落数", fontproperties=font)
    ax_b.set_ylabel("精确匹配", fontproperties=font)
    ax_b.set_title("(b) 确定问答的准确率与证据成本", fontproperties=font)
    ax_b.set_xlim(2.5, 17.5)
    ax_b.set_ylim(0.30, 0.52)
    ax_b.legend(frameon=False, prop=font, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax_b.grid(linestyle=":", color="#cccccc", linewidth=0.6)

    # Panel C: dumbbells against the strongest named baseline.
    plan = data["backbones"]["deepseek"]["methods"]["FinPlanRAG"]
    best = {}
    for b, _ in BACKBONES:
        methods = data["backbones"][b]["methods"]
        candidates = [methods[m] for m in MAIN if m in methods]
        best[b] = max(candidates, key=lambda r: r.get("balanced_accuracy", 0))

    rows = []
    for b, name in BACKBONES:
        rows.append((f"{SHORT[name]}-Bal", best[b]["balanced_accuracy"], plan["balanced_accuracy"]))
        rows.append((f"{SHORT[name]}-Macro", best[b]["macro_f1"], plan["macro_f1"]))
    labels = [r[0] for r in rows]
    ypos = np.arange(len(rows))[::-1]
    for y, (_, b_val, p_val) in zip(ypos, rows):
        ax_c1.plot([b_val, p_val], [y, y], color="#9a9a9a", linewidth=1.0, zorder=1)
        ax_c1.scatter([b_val], [y], s=22, color="#bdbdbd", zorder=2)
        ax_c1.scatter([p_val], [y], s=26, color="#222222", zorder=3)
    ax_c1.set_yticks(ypos)
    ax_c1.set_yticklabels(labels, fontproperties=font)
    ax_c1.set_xlim(0.38, 0.54)
    ax_c1.set_title("(c) 平衡准确率与宏F1", fontproperties=font)
    ax_c1.grid(axis="x", linestyle=":", color="#cccccc", linewidth=0.6)

    loss_plan = plan["decision_loss"]
    loss_best = [best[b]["decision_loss"] for b, _ in BACKBONES]
    ypos2 = np.arange(len(BACKBONES))[::-1]
    for y, (b, name), bv in zip(ypos2, BACKBONES, loss_best):
        ax_c2.plot([bv, loss_plan], [y, y], color="#9a9a9a", linewidth=1.0, zorder=1)
        ax_c2.scatter([bv], [y], s=22, color="#bdbdbd", zorder=2)
        ax_c2.scatter([loss_plan], [y], s=26, color="#222222", zorder=3)
    ax_c2.set_yticks(ypos2)
    ax_c2.set_yticklabels([n for _, n in BACKBONES], fontproperties=font)
    ax_c2.set_xlim(3.0, 5.6)
    ax_c2.set_title("(d) 决策损失（越低越好）", fontproperties=font)
    ax_c2.grid(axis="x", linestyle=":", color="#cccccc", linewidth=0.6)

    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
