"""Rebuild the controlled evidence used by the paper tables.

This runner keeps the protocol in one auditable command and writes large
intermediate JSON files to a caller-selected directory (prefer ``tmp/`` or
external storage). It does not alter the paper prose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def run_budget(root: Path, output: Path, seeds: int, cases: int, budget: int) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    command = [
        sys.executable,
        "-m",
        "finplan_rag.controlled_planning",
        "--seeds",
        str(seeds),
        "--cases-per-world",
        str(cases),
        "--budget",
        str(budget),
        "--out",
        str(output),
    ]
    subprocess.run(command, cwd=root, env=env, check=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    row = dict(payload["summary"]["finplan_state"])
    row.update({"budget": budget, "source_sha256": hashlib.sha256(output.read_bytes()).hexdigest().upper()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--cases-per-world", type=int, default=40)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4, 6, 10, 16])
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/paper_contract"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_budget(root, out_dir / f"budget_{budget}.json", args.seeds, args.cases_per_world, budget) for budget in args.budgets]
    summary = {
        "schema": "finplan-paper-contract-runner.v1",
        "protocol": {"seeds": args.seeds, "cases_per_world": args.cases_per_world, "scenarios": 7},
        "finplan_state": rows,
    }
    destination = out_dir / "budget_summary.json"
    destination.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
