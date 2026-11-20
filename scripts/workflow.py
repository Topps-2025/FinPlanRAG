"""Config-driven FinPlan experiment workflow.

Examples
--------
    PYTHONPATH="src;scripts" python scripts/workflow.py --config configs/paper.json --stage data
    PYTHONPATH="src;scripts" python scripts/workflow.py --config configs/paper.json --stage benchmark
    PYTHONPATH="src;scripts" python scripts/workflow.py --config configs/paper.json --stage sequential
    GLM_API_KEY=... PYTHONPATH="src;scripts" python scripts/workflow.py \
        --config configs/paper.json --stage named

Stages:
    data        validate the released benchmark, hash it, optionally fetch SEC events
    benchmark   run the company-disjoint logistic benchmark (all + verified)
    sequential  run the real earnings/peer evidence experiment over a gamma grid
    named       run the six named baselines on the configured LLM provider
    report      aggregate the stage summaries into workflow_report.md
    all         data + benchmark + sequential + report (named is separate: it uses an API)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from finplan_rag.workflow import load_config  # noqa: E402
from finplan_rag.workflow import protocol  # noqa: E402


def run(command: list[str], *, dry_run: bool = False) -> None:
    print("+", " ".join(command))
    if dry_run:
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts"), env.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    if completed.returncode != 0:
        raise SystemExit(f"command failed with {completed.returncode}: {command}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ensemble(report_path: Path) -> dict[str, float]:
    """Read the five-seed ensemble table from a benchmark report."""
    if not report_path.exists():
        return {}
    values: dict[str, float] = {}
    in_section = False
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if "Five-fold-seed probability ensemble" in line:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line.startswith("|") and "metric" not in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) >= 2:
                try:
                    values[cells[0]] = float(cells[1])
                except ValueError:
                    continue
    return values


def stage_data(config, args) -> None:
    data = config.data_path
    if not data.exists():
        raise SystemExit(f"missing data: {data}")
    rows = json.loads(data.read_text(encoding="utf-8"))
    print(f"data={data} claims={len(rows)} sha256={sha256(data)}")
    events = config.path("earnings_events")
    if not events.exists() or args.refresh_events:
        run(
            [
                sys.executable,
                "scripts/build_earnings_events.py",
                "--data",
                str(data),
                "--out",
                str(events),
            ],
            dry_run=args.dry_run,
        )


def validate_protocol(config) -> None:
    if tuple(config.section("test_years") or protocol.TEST_YEARS) != protocol.TEST_YEARS:
        raise SystemExit("config test_years disagree with finplan_rag.workflow.protocol")
    if int(config.section("folds") or protocol.FOLDS) != protocol.FOLDS:
        raise SystemExit("config folds disagree with finplan_rag.workflow.protocol")
    if tuple(config.section("fold_seeds") or protocol.FOLD_SEEDS) != protocol.FOLD_SEEDS:
        raise SystemExit("config fold_seeds disagree with finplan_rag.workflow.protocol")


def stage_benchmark(config, args) -> None:
    validate_protocol(config)
    out = config.results_root / "benchmark"
    run(
        [
            sys.executable,
            "scripts/run_company_disjoint_benchmark.py",
            "--data",
            str(config.data_path),
            "--out",
            str(out),
        ],
        dry_run=args.dry_run,
    )
    run(
        [
            sys.executable,
            "scripts/run_company_disjoint_benchmark.py",
            "--data",
            str(config.data_path),
            "--verified-only",
            "--out",
            str(out / "verified"),
        ],
        dry_run=args.dry_run,
    )


def stage_sequential(config, args) -> None:
    settings = config.sequential
    out_root = config.results_root / "sequential"
    for gamma in settings.get("gammas", [0.01]):
        for evidence_cost in settings.get("evidence_costs", [0.5]):
            for seed in settings.get("seeds", [1]):
                name = f"g{gamma}_c{evidence_cost}_s{seed}"
                command = [
                    sys.executable,
                    "scripts/run_sequential_evidence.py",
                    "--data",
                    str(config.data_path),
                    "--earnings-events",
                    str(config.path("earnings_events")),
                    "--out",
                    str(out_root / name),
                    "--gamma",
                    str(gamma),
                    "--evidence-cost",
                    str(evidence_cost),
                    "--outcome-cost",
                    str(settings.get("outcome_cost", 2.0)),
                    "--max-acquire",
                    str(settings.get("max_acquire", 3)),
                    "--seed",
                    str(seed),
                    "--train-end-year",
                    str(settings.get("train_end_year", 2021)),
                    "--test-start-year",
                    str(settings.get("test_start_year", 2022)),
                ]
                if settings.get("verified_only", True):
                    command.append("--verified-only")
                run(command, dry_run=args.dry_run)


def stage_sequential_portfolio(config, args) -> None:
    settings = config.sequential
    out = config.results_root / "sequential_portfolio"
    command = [
        sys.executable,
        "scripts/run_sequential_portfolio.py",
        "--data",
        str(config.data_path),
        "--earnings-events",
        str(config.path("earnings_events")),
        "--out",
        str(out),
        "--selection-margin",
        str(settings.get("selection_margin", 0.1)),
        "--planner-margin",
        str(settings.get("planner_margin", 0.2)),
        "--max-acquire",
        str(settings.get("max_acquire", 3)),
        "--gammas",
        ",".join(str(value) for value in settings.get("gammas", [0.01])),
        "--evidence-costs",
        ",".join(str(value) for value in settings.get("evidence_costs", [0.5])),
        "--outcome-costs",
        ",".join(str(value) for value in settings.get("outcome_costs", [2.0])),
    ]
    if settings.get("verified_only", True):
        command.append("--verified-only")
    run(command, dry_run=args.dry_run)


def stage_named(config, args) -> None:
    llm = config.llm
    out = config.results_root / "named"
    command = [
        sys.executable,
        "scripts/run_named_baselines.py",
        "--data",
        str(config.data_path),
        "--out",
        str(out),
        "--verified-only",
        "--variants",
        "findver,finfact,selfrag,crag,flare,adaptive",
        "--workers",
        str(args.workers),
        "--provider",
        str(llm.get("provider", "glm")),
        "--model",
        str(llm.get("model", "glm-5.3-flash")),
        "--base-url",
        str(llm.get("base_url", "https://open.bigmodel.cn/api/paas/v4")),
        "--api-key-env",
        str(llm.get("api_key_env", "GLM_API_KEY")),
    ]
    if llm.get("reasoning_effort"):
        command.extend(["--reasoning-effort", str(llm["reasoning_effort"])])
    if llm.get("extra_body"):
        command.extend(["--extra-body", json.dumps(llm["extra_body"])])
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    run(command, dry_run=args.dry_run)


def stage_report(config, args) -> None:
    if args.dry_run:
        print("+ would write workflow_report.md")
        return
    root = config.results_root
    lines = ["# FinPlan workflow report", ""]
    benchmark = root / "benchmark" / "summary.json"
    if benchmark.exists():
        payload = json.loads(benchmark.read_text(encoding="utf-8"))
        ensemble = payload.get("ensemble") or read_ensemble(
            root / "benchmark" / "report.md"
        )
        lines += [
            "## Benchmark",
            "",
            f"- claims per seed: {payload.get('n_claims_per_seed')}",
            f"- balanced accuracy (ensemble): {ensemble.get('balanced_accuracy'):.4f}",
            f"- macro F1 (ensemble): {ensemble.get('macro_f1'):.4f}",
            "",
        ]
    seq_root = root / "sequential"
    if seq_root.exists():
        lines += [
            "## Sequential evidence",
            "",
            "| run | gamma | cost | seed | best policy | best total | fixed adjudicate | always wait |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ]
        for summary in sorted(seq_root.glob("*/summary.json")):
            payload = json.loads(summary.read_text(encoding="utf-8"))
            rows = payload.get("summary", {})
            best = min(rows, key=lambda name: rows[name]["mean_total_loss"]) if rows else None
            lines.append(
                "| {name} | {gamma} | {cost} | {seed} | {best} | {best_loss:.3f} | "
                "{fixed:.3f} | {wait:.3f} |".format(
                    name=summary.parent.name,
                    gamma=payload.get("gamma"),
                    cost=payload.get("evidence_cost"),
                    seed=payload.get("seed", ""),
                    best=best,
                    best_loss=rows[best]["mean_total_loss"] if best else float("nan"),
                    fixed=rows.get("fixed_adjudicate", {}).get("mean_total_loss", float("nan")),
                    wait=rows.get("always_wait", {}).get("mean_total_loss", float("nan")),
                )
            )
        lines.append("")
    portfolio = root / "sequential_portfolio" / "summary.json"
    if portfolio.exists():
        payload = json.loads(portfolio.read_text(encoding="utf-8"))
        aggregate = payload.get("aggregate", {})
        lines += [
            "## Sequential portfolio",
            "",
            f"- cost cells: {aggregate.get('n_configs')}",
            f"- mean regret: {aggregate.get('mean_regret'):+.4f}",
            f"- max regret: {aggregate.get('max_regret'):+.4f}",
            f"- cells with positive regret: {aggregate.get('positive_regret_count')}",
            "",
        ]
    named_paths = [root / "named" / "summary.json"]
    for backbone in ("deepseek", "glm", "qwen3"):
        candidate = root.parent / backbone / "named" / "summary.json"
        if candidate.exists():
            named_paths.append(candidate)
    named_payloads = []
    seen = set()
    for path in named_paths:
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("backbone") in {item.get("backbone") for item in named_payloads}:
            continue
        named_payloads.append(payload)
    if named_payloads:
        lines += [
            "## Named baselines",
            "",
            "| backbone | provider | claims |",
            "|---|---|---:|",
        ]
        for payload in named_payloads:
            lines.append(
                f"| {payload.get('backbone')} | {payload.get('provider')} | "
                f"{payload.get('n_claims')} |"
            )
        lines.append("")
    out = root / "workflow_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "data", "benchmark", "sequential", "sequential_portfolio", "named", "report"],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-events", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    stages = {
        "data": stage_data,
        "benchmark": stage_benchmark,
        "sequential": stage_sequential,
        "sequential_portfolio": stage_sequential_portfolio,
        "named": stage_named,
        "report": stage_report,
    }
    order = ["data", "benchmark", "sequential_portfolio", "report"] if args.stage == "all" else [args.stage]
    for stage in order:
        print(f"\n=== {stage} ===")
        stages[stage](config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
