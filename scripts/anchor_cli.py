"""Command-line entry point for AnchorEvolve.

Examples:
    python scripts/anchor_cli.py case zhongji_2026 --cutoff 2026-11-19
    python scripts/anchor_cli.py ask "中际旭创2026年年报营业收入增速会是多少"
    python scripts/anchor_cli.py search "中际旭创 2026年三季报 毛利率"
    python scripts/anchor_cli.py audit 20261119-120000-zhongji_2026-abcd
    python scripts/anchor_cli.py serve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from anchor_engine import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUT,
    audit_markdown,
    load_config,
    load_env,
    make_client,
    parse_question,
    public_summary,
    run_ask,
)
from anchor_web import gather  # noqa: E402


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _print_summary(result: dict, *, as_json: bool) -> None:
    summary = public_summary(result)
    if as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    print("")
    print(f"run_id: {summary.get('run_id')}")
    print(f"公司: {summary.get('company')}  截止时点: {summary.get('cutoff')}")
    print(f"审计目录: {summary.get('audit_dir')}")
    for warning in result.get("warnings") or []:
        print(f"提示: {warning}")
    print("")
    for row in summary.get("claims") or []:
        print(f"[{row.get('claim_id')}] {row.get('claim')}")
        print(f"  动作: {row.get('action')}")
        if row.get("point"):
            print(f"  点答案: {row.get('point')}")
        if row.get("set"):
            print(f"  裁决集: {', '.join(row.get('set'))}")
        if row.get("belief"):
            belief = "  ".join(f"{k}={v:.3f}" for k, v in row["belief"].items())
            print(f"  后验: {belief}")
        date_estimate = row.get("date_estimate")
        if date_estimate:
            print(
                f"  预计披露日: {date_estimate.get('point')} "
                f"({date_estimate.get('low')} 至 {date_estimate.get('high')})"
            )
        gaps = row.get("coverage_gaps") or []
        if gaps:
            print(
                f"  提示: 覆盖缺口 {', '.join(str(gap) for gap in gaps)}；"
                f"后验向先验收缩 {row.get('coverage_penalty') or 0.0}"
            )
        calibration = row.get("calibration")
        if isinstance(calibration, dict) and calibration.get("applied"):
            print(
                f"  历史校准: T={calibration.get('temperature')} "
                f"w={calibration.get('fusion_weight')} "
                f"type_shrink={calibration.get('type_shrinkage')}"
            )
        forecast = row.get("forecast")
        if isinstance(forecast, dict) and forecast.get("point") is not None:
            unit = forecast.get("unit") or ""
            print(
                f"  锚点桥接: {forecast.get('point')} {unit} "
                f"({forecast.get('low')} 至 {forecast.get('high')} {unit})"
            )
            if forecast.get("formula"):
                print(f"    公式: {forecast.get('formula')}")
        numeric = row.get("numeric_estimate")
        if isinstance(numeric, dict) and numeric.get("point") is not None:
            unit = numeric.get("unit") or ""
            print(
                f"  辅助数值估计: {numeric.get('point')} {unit} "
                f"({numeric.get('low')} 至 {numeric.get('high')} {unit})"
            )
        print(f"  锚点: {row.get('n_accepted')}/{row.get('n_anchors')} 被采纳")
        if row.get("error"):
            print(f"  错误: {row.get('error')}")
        print("")
    print(f"完整审计报告: {Path(summary.get('audit_dir') or '') / 'audit.md'}")


def _apply_ablation_flags(config: dict, args: argparse.Namespace) -> dict:
    """Apply explicit, named ablations without editing the base config."""
    if getattr(args, "no_bridge", False):
        config.setdefault("bridge", {})["enabled"] = False
    if getattr(args, "no_calibration", False):
        config.setdefault("calibration", {})["enabled"] = False
    return config


def cmd_ask(args: argparse.Namespace) -> int:
    load_env()
    config = _apply_ablation_flags(load_config(args.config), args)
    if not args.question and not args.case_id and not args.claims_file:
        raise SystemExit("ask 需要问题、--case 或 --claims-file")
    result = run_ask(
        args.question or "",
        config=config,
        cutoff=args.cutoff,
        company=args.company,
        ticker=args.ticker,
        target_period=args.target_period,
        claims_file=args.claims_file,
        case_id=args.case_id,
        out_root=args.out_dir,
        max_workers=args.max_workers,
    )
    _print_summary(result, as_json=args.json)
    return 0


def cmd_case(args: argparse.Namespace) -> int:
    load_env()
    config = _apply_ablation_flags(load_config(args.config), args)
    result = run_ask(
        "",
        config=config,
        cutoff=args.cutoff,
        case_id=args.case_id,
        out_root=args.out_dir,
        max_workers=args.max_workers,
    )
    _print_summary(result, as_json=args.json)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    items = gather(args.query, top_k=args.top_k, fetch_top=args.fetch_top)
    if args.json:
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return 0
    for index, item in enumerate(items, 1):
        print(f"[W{index}] {item.get('title')} | {item.get('date') or 'unknown'}")
        print(f"  {item.get('url')}")
        snippet = (item.get("snippet") or "").replace("\n", " ")
        print(f"  {snippet[:300]}")
        print("")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    print(audit_markdown(args.run_id, out_root=args.out_dir))
    return 0


def cmd_contract(args: argparse.Namespace) -> int:
    load_env()
    config = load_config(args.config)
    client = make_client(config)
    parsed = parse_question(
        args.question,
        config=config,
        client=client,
        company=args.company,
        ticker=args.ticker,
        target_period=args.target_period,
        today=args.cutoff,
    )
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(_: argparse.Namespace) -> int:
    load_env()
    from anchor_mcp import main as mcp_main

    mcp_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AnchorEvolve CLI for forward-looking financial claims"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ask a forward-looking question")
    ask.add_argument("question", nargs="?", default="")
    ask.add_argument("--case", dest="case_id", default=None)
    ask.add_argument("--claims-file", default=None)
    ask.add_argument("--company", default="")
    ask.add_argument("--ticker", default="")
    ask.add_argument("--period", dest="target_period", default="")
    ask.add_argument("--cutoff", default=None)
    ask.add_argument("--max-workers", type=int, default=4)
    ask.add_argument("--config", default=str(DEFAULT_CONFIG))
    ask.add_argument("--out-dir", default=None)
    ask.add_argument("--no-bridge", action="store_true", help="disable the numeric bridge anchor")
    ask.add_argument("--no-calibration", action="store_true", help="disable the historical calibration bundle")
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=cmd_ask)

    case = sub.add_parser("case", help="run a named case from configs/anchor_cases")
    case.add_argument("case_id")
    case.add_argument("--cutoff", default=None)
    case.add_argument("--max-workers", type=int, default=4)
    case.add_argument("--config", default=str(DEFAULT_CONFIG))
    case.add_argument("--out-dir", default=None)
    case.add_argument("--no-bridge", action="store_true", help="disable the numeric bridge anchor")
    case.add_argument("--no-calibration", action="store_true", help="disable the historical calibration bundle")
    case.add_argument("--json", action="store_true")
    case.set_defaults(func=cmd_case)

    search = sub.add_parser("search", help="search the web and show evidence")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=8)
    search.add_argument("--fetch-top", type=int, default=2)
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=cmd_search)

    audit = sub.add_parser("audit", help="print an audit report")
    audit.add_argument("run_id")
    audit.add_argument("--out-dir", default=None)
    audit.set_defaults(func=cmd_audit)

    contract = sub.add_parser("contract", help="parse a question into a claim contract")
    contract.add_argument("question")
    contract.add_argument("--company", default="")
    contract.add_argument("--ticker", default="")
    contract.add_argument("--period", dest="target_period", default="")
    contract.add_argument("--cutoff", default=None)
    contract.add_argument("--config", default=str(DEFAULT_CONFIG))
    contract.set_defaults(func=cmd_contract)

    serve = sub.add_parser("serve", help="run the MCP stdio server")
    serve.set_defaults(func=cmd_serve)
    return parser


def main() -> int:
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
