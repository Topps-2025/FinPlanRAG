"""Compose the planner with a confidence-gated fixed-16 fallback.

The determinate branch runs the obligation planner first.  If the reader is
not sufficient or its confidence is below 0.8, the branch escalates to the
fixed-16 evidence set and uses that reader output.  This script composes the
two already-run conditions into one per-example file so the fallback condition
can be analyzed without extra API calls.  The same rule is implemented in
``experiments/unified/pipeline.py``.

Usage:
    python scripts/apply_planner_fallback.py \
      --planner paper/results/.../planner16/per_example.jsonl \
      --fixed paper/results/.../per_example.jsonl \
      --kind docfinqa --out paper/results/.../planner_fallback/per_example.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

THRESHOLD = 0.8


def valid(kind: str, parsed: dict | None) -> bool:
    if not parsed:
        return False
    if kind == "findver":
        return parsed.get("verdict") in {"entailed", "refuted"}
    answer = str(parsed.get("answer", "")).strip().lower()
    return bool(parsed.get("sufficient")) and answer not in {
        "", "insufficient", "unknown", "n/a", "none"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--kind", choices=["docfinqa", "findver"], required=True)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    planner = [json.loads(line) for line in Path(args.planner).read_text(encoding="utf-8").splitlines() if line.strip()]
    fixed = [json.loads(line) for line in Path(args.fixed).read_text(encoding="utf-8").splitlines() if line.strip()]
    planner_by = {row["example_id"]: row for row in planner if row.get("condition") in {"planner", "planner16"}}
    fixed_by = {row["example_id"]: row for row in fixed if row.get("condition") == "fixed_k_16"}
    rows = []
    used = 0
    for example_id, p in planner_by.items():
        f = fixed_by.get(example_id)
        if f is None:
            continue
        parsed = p.get("parsed") or {}
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        keep = valid(args.kind, parsed) and confidence >= args.threshold
        base = dict(p if keep else f)
        base["condition"] = "planner_fallback"
        base["planner_used"] = keep
        base["num_paragraphs"] = float(p.get("num_paragraphs", 0)) + (
            0.0 if keep else float(f.get("num_paragraphs", 0))
        )
        usage = dict(base.get("usage") or {})
        if not keep:
            usage["input_tokens"] = float((p.get("usage") or {}).get("input_tokens", 0)) + float(
                (f.get("usage") or {}).get("input_tokens", 0)
            )
            usage["output_tokens"] = float((p.get("usage") or {}).get("output_tokens", 0)) + float(
                (f.get("usage") or {}).get("output_tokens", 0)
            )
        base["usage"] = usage
        rows.append(base)
        used += int(keep)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows; planner kept {used/len(rows):.3f}")


if __name__ == "__main__":
    main()
