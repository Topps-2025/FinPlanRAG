"""Post-hoc uncertainty audit for the frozen LOFin multi-document pilot.

This script does not tune or rerun a method.  It reads frozen case-level
closure outcomes and reports Wilson intervals plus exact paired McNemar tests.
Exploratory, holdout, and pooled estimates remain separate because the method
was revised on the exploratory split.
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple


BASELINES = (
    "single_shot",
    "fixed_decomposition",
    "generic_adaptive",
    "hirec_style",
    "finplan_v1",
    "finplan_v3_no_path",
    "metadata_decomposition",
    "generic_path_bound",
    "hirec_path_bound",
)
TARGET = "finplan_v3_path_bound"


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return (center - half, center + half)


def exact_mcnemar(target: Sequence[int], baseline: Sequence[int]) -> Mapping[str, float]:
    target_only = sum(a == 1 and b == 0 for a, b in zip(target, baseline))
    baseline_only = sum(a == 0 and b == 1 for a, b in zip(target, baseline))
    discordant = target_only + baseline_only
    if not discordant:
        p_value = 1.0
    else:
        smaller = min(target_only, baseline_only)
        lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * lower_tail)
    return {
        "target_only_successes": float(target_only),
        "baseline_only_successes": float(baseline_only),
        "discordant_pairs": float(discordant),
        "two_sided_exact_p": p_value,
    }


def read_rows(path: Path) -> Sequence[Mapping[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def case_vectors(rows: Sequence[Mapping[str, object]], method: str) -> Dict[str, int]:
    return {str(row["case_id"]): int(float(row["closure"])) for row in rows if row["method"] == method}


def analyze_split(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    target = case_vectors(rows, TARGET)
    cases = sorted(target)
    methods: Dict[str, object] = {}
    for method in (TARGET, *BASELINES):
        outcomes = case_vectors(rows, method)
        values = [outcomes[case] for case in cases]
        successes = sum(values)
        interval = wilson(successes, len(values))
        item: Dict[str, object] = {
            "successes": successes,
            "n": len(values),
            "closure": successes / len(values),
            "wilson_95": list(interval),
        }
        if method != TARGET:
            item["paired_vs_finplan_v3_path_bound"] = exact_mcnemar(
                [target[case] for case in cases], values
            )
        methods[method] = item
    return {"cases": len(cases), "methods": methods}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="单个已冻结结果 JSON；省略时分析 v3 exploratory/holdout")
    parser.add_argument("--label", default="validation2")
    parser.add_argument("--out")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.input:
        rows = read_rows(root / args.input)
        result = {
            "schema": "finplan-lofin-multidoc-statistical-audit.v1",
            "status": "post-hoc-small-sample-uncertainty-not-sota",
            "notes": [
                "该统计审计在方法和数据冻结后运行，不用于调参。",
                "文件闭包不是端到端答案正确率。",
            ],
            args.label: analyze_split(rows),
        }
        out = root / (args.out or "results/lofin_multidoc_validation2_statistical_audit_v1.json")
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    exploratory = read_rows(root / "results" / "lofin_multidoc_pilot_exploratory_v3.json")
    holdout = read_rows(root / "results" / "lofin_multidoc_pilot_holdout_v1.json")
    result = {
        "schema": "finplan-lofin-multidoc-statistical-audit.v1",
        "status": "post-hoc-small-sample-uncertainty-not-sota",
        "notes": [
            "探索集参与了架构修改，不能作为独立确认性检验。",
            "留出集是冻结后运行，但仅 8 题，配对检验功效有限。",
            "合并结果只作描述性汇总，不能消除探索阶段选择偏差。",
            "文件闭包不是端到端答案正确率。",
        ],
        "exploratory": analyze_split(exploratory),
        "holdout": analyze_split(holdout),
        "pooled_descriptive": analyze_split([*exploratory, *holdout]),
    }
    out = root / "results" / "lofin_multidoc_statistical_audit_v1.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
