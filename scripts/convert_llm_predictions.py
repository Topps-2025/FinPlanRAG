"""Convert parsed LLM JSON outputs to per-claim probability rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_company_disjoint_benchmark import VERIFIED_LABELS
from run_real_benchmark import DEFAULT_DATA, LABELS, load_records

REPO_ROOT = Path(__file__).resolve().parents[1]


def belief(parsed):
    label = parsed.get("label")
    credible = [value for value in (parsed.get("credible_set") or []) if value in LABELS]
    if label not in LABELS:
        label = credible[0] if credible else "above"
    if not credible:
        credible = [label]
    confidence = min(0.95, max(0.34, float(parsed.get("confidence", 0.5))))
    values = {value: 0.0 for value in LABELS}
    values[label] = confidence
    rest = [value for value in credible if value != label]
    if rest:
        for value in rest:
            values[value] = (1.0 - confidence) / len(rest)
    else:
        values[label] = 1.0
    return [values[value] for value in LABELS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    records = {record["claim_id"]: record for _, _, record in load_records(args.data)}
    rows = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if args.variant and row.get("variant") != args.variant:
            continue
        parsed = row.get("parsed") or {}
        if parsed.get("label") not in LABELS:
            continue
        record = records.get(row["claim_id"])
        if record is None:
            continue
        if args.verified_only and record.get("verification") not in VERIFIED_LABELS:
            continue
        rows.append(
            {
                "seed": args.seed,
                "claim_id": row["claim_id"],
                "truth": row["truth"],
                "probabilities": belief(parsed),
                "year": int(record.get("cutoff_year") or 0),
                "sector": record.get("sector"),
                "variant": row.get("variant"),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
