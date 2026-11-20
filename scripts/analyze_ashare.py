"""Paired comparisons for the expanded A-share backtest."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def boot(diff: np.ndarray, seed: int = 20261120, resamples: int = 10000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(resamples)])
    return {"mean": float(diff.mean()), "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5))}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: analyze_ashare.py <result_dir>")
    root = Path(sys.argv[1])
    methods = {}
    for method in ("last_year", "run_rate", "full", "no_bridge"):
        path = root / method / "per_claim.jsonl"
        if path.exists():
            methods[method] = {row["claim_id"]: row for row in load(path)}
    output: dict[str, Any] = {"schema": "finplan-ashare-paired-v1", "comparisons": {}}
    for reference in ("last_year", "run_rate", "no_bridge"):
        if reference not in methods or "full" not in methods:
            continue
        common = sorted(set(methods[reference]) & set(methods["full"]))
        full_loss = np.array([methods["full"][cid]["loss"] for cid in common])
        ref_loss = np.array([methods[reference][cid]["loss"] for cid in common])
        full_correct = np.array([methods["full"][cid]["prediction"] == methods["full"][cid]["truth"] for cid in common], dtype=float)
        ref_correct = np.array([methods[reference][cid]["prediction"] == methods[reference][cid]["truth"] for cid in common], dtype=float)
        output["comparisons"][f"full_vs_{reference}"] = {
            "n": len(common),
            "loss_difference": boot(full_loss - ref_loss),
            "accuracy_difference": boot(full_correct - ref_correct),
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
