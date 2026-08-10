"""Paired uncertainty audit for frozen non-oracle obligation planners."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

try:
    from .storage_paths import RESULTS_ROOT
except ImportError:
    from storage_paths import RESULTS_ROOT

from analyze_lofin_multidoc_statistics import exact_mcnemar, wilson


METRICS = ("entity_exact", "obligation_exact", "obligation_covers_gold", "cascade_closure")


def index_rows(payload: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    return {str(row["case_id"]): row for row in payload["rows"]}  # type: ignore[index]


def metric_audit(
    target: Mapping[str, Mapping[str, object]],
    baseline: Mapping[str, Mapping[str, object]],
    cases: Sequence[str],
    metric: str,
) -> Mapping[str, object]:
    target_values = [int(float(target[case][metric])) for case in cases]
    baseline_values = [int(float(baseline[case][metric])) for case in cases]
    target_successes = sum(target_values)
    baseline_successes = sum(baseline_values)
    return {
        "target_successes": target_successes,
        "baseline_successes": baseline_successes,
        "n": len(cases),
        "target_rate": target_successes / len(cases),
        "baseline_rate": baseline_successes / len(cases),
        "target_wilson_95": list(wilson(target_successes, len(cases))),
        "baseline_wilson_95": list(wilson(baseline_successes, len(cases))),
        "paired_exact_mcnemar": exact_mcnemar(target_values, baseline_values),
    }


def run(target_path: Path, baseline_path: Path, out_path: Path) -> Mapping[str, object]:
    target_payload = json.loads(target_path.read_text(encoding="utf-8"))
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    target = index_rows(target_payload)
    baseline = index_rows(baseline_payload)
    if set(target) != set(baseline):
        raise ValueError("Paired audit requires identical case IDs")
    cases = sorted(target)
    result = {
        "schema": "finplan-nonoracle-paired-statistical-audit.v1",
        "status": "post-hoc-small-sample-uncertainty-not-sota",
        "protocol": {
            "target": str(target_path),
            "baseline": str(baseline_path),
            "target_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
            "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            "cases": len(cases),
        },
        "notes": [
            "Target and baseline are evaluated on identical case IDs.",
            "Interpret statistical significance together with the reported discordant-pair count and Wilson intervals.",
            "File closure is not paragraph evidence completeness or answer correctness.",
        ],
        "metrics": {
            metric: metric_audit(target, baseline, cases, metric)
            for metric in METRICS
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=RESULTS_ROOT / "nonoracle_obligation_v4_validation3.json")
    parser.add_argument("--baseline", default=RESULTS_ROOT / "registry_rule_v1_validation3.json")
    parser.add_argument("--out", default=RESULTS_ROOT / "nonoracle_obligation_validation3_statistical_audit_v1.json")
    args = parser.parse_args()
    result = run(Path(args.target), Path(args.baseline), Path(args.out))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
