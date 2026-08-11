"""Verify the frozen full run reproduces the smoke pass exactly (Task #17).

Protocol step1 determinism: every (case_id, method) row and prediction that
appears in the smoke dev output must appear in the frozen full output with
identical content.  The smoke qids/cases are a strict subset of the full set,
so the comparison is one-directional (dev subset -> frozen superset).

Usage: python check_lofin_smoke_determinism.py [frozen.json] [dev.json]
Exit 0 = identical (prints OK), 1 = mismatch with details.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\results")
FROZEN = RESULTS / "lofin_full_benchmark_frozen.json"
DEV = RESULTS / "lofin_full_benchmark_dev.json"


def main() -> None:
    frozen_path = Path(sys.argv[1]) if len(sys.argv) > 1 else FROZEN
    dev_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEV
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    dev = json.loads(dev_path.read_text(encoding="utf-8"))

    frozen_rows = {(r["case_id"], r["method"]): r for r in frozen["rows"]}
    dev_rows = {(r["case_id"], r["method"]): r for r in dev["rows"]}
    missing = [k for k in dev_rows if k not in frozen_rows]
    diffs = [k for k in dev_rows if k in frozen_rows and frozen_rows[k] != dev_rows[k]]

    frozen_preds = {(p.get("case_id"), p.get("method")): p for p in frozen.get("predictions", [])}
    dev_preds = {(p.get("case_id"), p.get("method")): p for p in dev.get("predictions", [])}
    pred_missing = [k for k in dev_preds if k not in frozen_preds]
    pred_diffs = [k for k in dev_preds if k in frozen_preds and frozen_preds[k] != dev_preds[k]]

    print(f"frozen: {len(frozen['rows'])} rows, {len(frozen.get('predictions', []))} predictions")
    print(f"dev:    {len(dev['rows'])} rows, {len(dev.get('predictions', []))} predictions "
          f"(smoke subset)")
    print(f"smoke rows reproduced in frozen: {len(dev_rows) - len(missing)}/{len(dev_rows)}; "
          f"identical: {len(dev_rows) - len(missing) - len(diffs)}")
    if missing or diffs or pred_missing or pred_diffs:
        print(f"MISMATCH: missing rows {len(missing)}, differing rows {len(diffs)}, "
              f"missing predictions {len(pred_missing)}, differing predictions {len(pred_diffs)}")
        for k in (missing + diffs + pred_missing + pred_diffs)[:10]:
            print("  ", k)
        raise SystemExit(1)
    print("OK: all smoke-pass rows and predictions byte-identical in the frozen full run")


if __name__ == "__main__":
    main()
