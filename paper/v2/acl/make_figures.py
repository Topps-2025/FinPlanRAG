"""Generate the two result figures for the ACL draft."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

CONFIG_LABELS = {
    "base": "Base",
    "symmetric": "Symmetric",
    "over_averse": "Over-averse",
    "under_averse": "Under-averse",
    "cheap_abstain": "Cheap abstain",
    "expensive_abstain": "Costly abstain",
    "narrow_penalty": "Narrow width",
    "wide_penalty": "Wide width",
    "alpha_05": r"$\alpha=.05$",
    "alpha_20": r"$\alpha=.20$",
}
POLICIES = [
    ("planner", "Planner", "#1f4e79", "o"),
    ("always_answer", "Always answer", "#c55a11", "s"),
    ("always_adjudicate", "Always adjudicate", "#2e7d32", "^"),
    ("always_abstain", "Always abstain", "#6b7280", "D"),
]


def regret_figure() -> None:
    path = ROOT / "paper/results/real_decision_eval_verified_calibrated/summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    configs = [row["config"] for row in summary["per_seed"][0]["rows"]]
    losses = {}
    for config in configs:
        losses[config] = {}
        for policy, _, _, _ in POLICIES:
            losses[config][policy] = float(
                np.mean(
                    [
                        next(row for row in seed["rows"] if row["config"] == config)[policy][
                            "mean_loss"
                        ]
                        for seed in summary["per_seed"]
                    ]
                )
            )
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    x = np.arange(len(configs))
    for policy, label, color, marker in POLICIES:
        regret = [
            losses[config][policy]
            - min(
                losses[config]["always_answer"],
                losses[config]["always_adjudicate"],
                losses[config]["always_abstain"],
            )
            for config in configs
        ]
        ax.plot(x, regret, marker=marker, markersize=3.2, linewidth=1.1, color=color, label=label)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[config] for config in configs], rotation=32, ha="right")
    ax.set_ylabel("Regret vs. best fixed policy")
    ax.set_ylim(-0.55, 6.55)
    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        handlelength=1.0,
        columnspacing=0.55,
        handletextpad=0.35,
        fontsize=6.5,
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "regret.pdf", bbox_inches="tight")
    plt.close(fig)


def effect_figure() -> None:
    # Ensemble estimates from the company-disjoint benchmark.
    rows = [
        ("Balanced accuracy", "All labels (n=273)", 0.2324, 0.1668, 0.2947),
        ("Balanced accuracy", "XBRL-verified (n=227)", 0.2303, 0.1643, 0.2937),
        ("Macro F1", "All labels (n=273)", 0.2937, 0.2278, 0.3587),
        ("Macro F1", "XBRL-verified (n=227)", 0.2681, 0.1942, 0.3380),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.55), sharey=True)
    for ax, metric in zip(axes, ["Balanced accuracy", "Macro F1"]):
        subset = [row for row in rows if row[0] == metric]
        y = np.arange(len(subset))[::-1]
        values = [row[2] for row in subset]
        low = [row[2] - row[3] for row in subset]
        high = [row[4] - row[2] for row in subset]
        ax.errorbar(
            values,
            y,
            xerr=[low, high],
            fmt="o",
            color="#1f4e79",
            ecolor="#1f4e79",
            capsize=2.5,
            markersize=4,
            linewidth=1.0,
        )
        ax.axvline(0.0, color="black", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([row[1] for row in subset])
        ax.set_title(metric)
        ax.set_xlim(0.10, 0.40)
    fig.supxlabel("Gain over majority baseline", fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 1], pad=0.2)
    fig.savefig(FIG / "effects.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    regret_figure()
    effect_figure()
    print("wrote", FIG / "regret.pdf", FIG / "effects.pdf")
