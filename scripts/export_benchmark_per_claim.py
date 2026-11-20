"""Export per-claim logistic benchmark probabilities for metric analysis."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from run_company_disjoint_benchmark import run_fold_seed
from run_real_benchmark import LABELS, company_key, load_records, DEFAULT_DATA

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    rows = load_records(args.data)
    by_company = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    records = {str(record.get("claim_id") or company_key(record)): record for _, _, record in rows}
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for seed in [int(v) for v in args.seeds.split(",")]:
        result = run_fold_seed(
            rows, by_company, seed, calibrate=args.calibrate, verified_only=args.verified_only
        )
        for key, truth, probabilities, year in zip(
            result["keys"], result["truths"], result["probabilities"], result["years"]
        ):
            record = records[key]
            all_rows.append(
                {
                    "seed": seed,
                    "claim_id": key,
                    "truth": str(truth),
                    "probabilities": [float(v) for v in probabilities],
                    "year": int(year),
                    "company": company_key(record),
                    "sector": record.get("sector"),
                    "guidance_width_pct": record.get("guidance_width_pct"),
                }
            )
    with (args.out / "per_claim.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", len(all_rows), "rows to", args.out / "per_claim.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
