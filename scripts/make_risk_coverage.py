"""Plot risk-coverage curves from the metric-suite artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "paper/en/finplanrag_2026/figures/risk_coverage_named.pdf"

MONO_STYLES = [
    ("0.00", "-", "o"),
    ("0.20", "--", "s"),
    ("0.35", "-.", "^"),
    ("0.50", ":", "D"),
    ("0.15", "-", "v"),
    ("0.45", "--", "P"),
    ("0.60", "-.", "X"),
]
ZH_LABELS = ("覆盖率", "选择性风险")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="deepseek")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--mono", action="store_true", help="grayscale, print-safe style")
    parser.add_argument("--zh", action="store_true", help="Chinese axis labels")
    args = parser.parse_args()
    root = REPO_ROOT / "paper" / "results"
    input_dir = args.input_dir or (root / args.backbone / "named")
    files = {
        "FinPlan": root / "benchmark_metrics_verified_calibrated.json",
        "FinDVer": input_dir / "metrics_findver.json",
        "Fin-Fact": input_dir / "metrics_finfact.json",
        "Self-RAG": input_dir / "metrics_selfrag.json",
        "CRAG": input_dir / "metrics_crag.json",
        "FLARE": input_dir / "metrics_flare.json",
        "Adaptive-RAG": input_dir / "metrics_adaptive.json",
    }
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    for index, (label, path) in enumerate(files.items()):
        data = json.loads(path.read_text(encoding="utf-8"))
        curve = data["risk_coverage"]
        x = [float(key) for key in curve]
        y = [curve[key] for key in curve]
        if args.mono:
            color, linestyle, marker = MONO_STYLES[index % len(MONO_STYLES)]
            style = {
                "color": color,
                "linestyle": linestyle,
                "marker": marker,
                "markersize": 2.4,
                "linewidth": 1.2,
            }
        else:
            style = {"marker": "o", "markersize": 2.2, "linewidth": 1.1}
        ax.plot(x, y, label=label, **style)
    if args.zh:
        zh_font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc")
        ax.set_xlabel(ZH_LABELS[0], fontproperties=zh_font)
        ax.set_ylabel(ZH_LABELS[1], fontproperties=zh_font)
    else:
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Selective risk")
    ax.set_xlim(0.05, 1.02)
    ax.set_ylim(0.25, 0.75)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
