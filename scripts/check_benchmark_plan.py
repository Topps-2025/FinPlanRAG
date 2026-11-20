"""Validate the benchmark plan without running any experiment.

The checker answers a narrow question: is the plan structurally complete and
connectable?  It verifies that every named baseline and metric is registered,
that every claim points at a benchmark, that external benchmarks are marked as
component evidence rather than pooled into the main outcome claim, that the
prospective cases exist and are internally consistent, and that data paths are
either present or explicitly marked as requiring a public download.

It does not evaluate a model, read test labels, or produce performance numbers.

Usage:
    python scripts/check_benchmark_plan.py
    python scripts/check_benchmark_plan.py --report paper/results/benchmark_plan/validation.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = REPO_ROOT / "configs" / "benchmark_plan.yaml"
ALLOWED_ROLES = {
    "main_outcome",
    "robustness",
    "historical_backtest",
    "external_component",
    "determinate_control",
    "prospective",
}
METRIC_KINDS = {"level", "growth", "margin", "capacity", "production", "date", "disclosure"}
MODEL_TEMPLATES = {
    "level_bridge",
    "growth_bridge",
    "margin_bridge",
    "capacity_bridge",
    "production_bridge",
    "date_window",
    "disclosure_check",
}


class Checker:
    def __init__(self, plan_path: Path) -> None:
        self.plan_path = plan_path
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks: list[dict[str, Any]] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            self.error(f"{name}: {detail}" if detail else name)

    @staticmethod
    def path_like(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return ("/" in value or "\\" in value) and not value.startswith("http")

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path

    def load_plan(self) -> dict[str, Any]:
        if not self.plan_path.exists():
            raise SystemExit(f"plan not found: {self.plan_path}")
        data = yaml.safe_load(self.plan_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit("plan is not a YAML mapping")
        return data

    def check_registries(self, plan: dict[str, Any]) -> None:
        for key in ("version", "claims", "baseline_registry", "metric_registry", "benchmarks", "fairness_rules"):
            self.check(f"top_level.{key}", key in plan, f"missing {key}")

        baselines = plan.get("baseline_registry") or {}
        self.check("baseline_registry.nonempty", bool(baselines), "no baselines registered")
        for name, entry in baselines.items():
            self.check(f"baseline.{name}.description", bool((entry or {}).get("description")),
                       "missing description")
            self.check(f"baseline.{name}.reproducible", (entry or {}).get("reproducible") is True,
                       "baseline is not marked reproducible")
            runner = (entry or {}).get("runner")
            if runner:
                self.check(f"baseline.{name}.runner_exists", self.resolve(str(runner)).exists(),
                           f"runner not found: {runner}")
            component_runner = (entry or {}).get("component_runner")
            if component_runner:
                self.check(f"baseline.{name}.component_runner_exists",
                           self.resolve(str(component_runner)).exists(),
                           f"component runner not found: {component_runner}")
            artifact = (entry or {}).get("artifact")
            if artifact and not self.resolve(str(artifact)).exists():
                self.warn(f"baseline.{name}.artifact_missing: {artifact} is not generated yet")

        metrics = plan.get("metric_registry") or {}
        self.check("metric_registry.nonempty", bool(metrics), "no metrics registered")
        for name, description in metrics.items():
            self.check(f"metric.{name}.description", bool(description), "missing description")

        rules = plan.get("fairness_rules") or []
        self.check("fairness_rules.minimum", len(rules) >= 5,
                   f"expected at least 5 rules, found {len(rules)}")

    def check_claims(self, plan: dict[str, Any]) -> None:
        claims = plan.get("claims") or {}
        self.check("claims.nonempty", bool(claims), "no claims registered")
        benchmarks = {row.get("id") for row in plan.get("benchmarks") or []}
        baselines = set((plan.get("baseline_registry") or {}).keys())
        metrics = set((plan.get("metric_registry") or {}).keys())
        for claim_id, claim in claims.items():
            self.check(f"claim.{claim_id}.text", bool((claim or {}).get("text")), "missing text")
            names = (claim or {}).get("benchmark")
            if isinstance(names, str):
                names = [names]
            names = names or []
            self.check(f"claim.{claim_id}.benchmark", bool(names), "claim has no benchmark")
            for name in names:
                self.check(f"claim.{claim_id}.benchmark.{name}", name in benchmarks,
                           f"unknown benchmark {name}")
            for name in (claim or {}).get("comparison") or []:
                self.check(f"claim.{claim_id}.comparison.{name}", name in baselines,
                           f"unknown baseline {name}")
            for name in (claim or {}).get("metrics") or []:
                self.check(f"claim.{claim_id}.metric.{name}", name in metrics,
                           f"unknown metric {name}")

        h2 = claims.get("H2") or {}
        h2_comparison = set(h2.get("comparison") or [])
        self.check("claim.H2.qualification_arm",
                   bool(h2_comparison & {"no_qualification", "accept_all", "no_source_check"}),
                   "H2 has no qualification ablation arm")

        bridge = claims.get("BRIDGE") or {}
        bridge_names = bridge.get("benchmark")
        bridge_names = [bridge_names] if isinstance(bridge_names, str) else (bridge_names or [])
        roles = {
            str(row.get("id")): str(row.get("role"))
            for row in plan.get("benchmarks") or []
            if isinstance(row, dict)
        }
        self.check("claim.BRIDGE.historical_outcome_benchmark",
                   any(roles.get(name) in {"historical_backtest", "main_outcome"} for name in bridge_names),
                   "BRIDGE is not bound to a benchmark with realized numeric outcomes")

    def check_benchmark(self, plan: dict[str, Any], row: dict[str, Any], seen: set[str]) -> None:
        benchmark_id = str(row.get("id") or "")
        self.check(f"benchmark.{benchmark_id}.id", bool(benchmark_id), "missing id")
        self.check(f"benchmark.{benchmark_id}.unique", benchmark_id not in seen,
                   f"duplicate id {benchmark_id}")
        seen.add(benchmark_id)

        role = row.get("role")
        self.check(f"benchmark.{benchmark_id}.role", role in ALLOWED_ROLES, f"invalid role {role}")
        self.check(f"benchmark.{benchmark_id}.task", bool(row.get("task")), "missing task")

        status = row.get("status")
        if status == "adapter_required":
            self.warn(f"benchmark.{benchmark_id}.adapter_required: {row.get('adapter_note') or 'no adapter'}")
        adapter = row.get("adapter")
        if adapter:
            self.check(f"benchmark.{benchmark_id}.adapter_exists", self.resolve(str(adapter)).exists(),
                       f"adapter not found: {adapter}")

        baselines = row.get("baselines") or []
        metrics = row.get("metrics") or []
        self.check(f"benchmark.{benchmark_id}.baselines", bool(baselines), "no baselines")
        self.check(f"benchmark.{benchmark_id}.metrics", bool(metrics), "no metrics")
        registered_baselines = set((plan.get("baseline_registry") or {}).keys())
        registered_metrics = set((plan.get("metric_registry") or {}).keys())
        for name in baselines:
            self.check(f"benchmark.{benchmark_id}.baseline.{name}", name in registered_baselines,
                       f"unknown baseline {name}")
        for name in metrics:
            self.check(f"benchmark.{benchmark_id}.metric.{name}", name in registered_metrics,
                       f"unknown metric {name}")
        self.check(f"benchmark.{benchmark_id}.baseline_unique", len(baselines) == len(set(baselines)),
                   "duplicate baseline")
        self.check(f"benchmark.{benchmark_id}.metric_unique", len(metrics) == len(set(metrics)),
                   "duplicate metric")

        for claim_id in row.get("claims") or []:
            self.check(f"benchmark.{benchmark_id}.claim.{claim_id}", claim_id in (plan.get("claims") or {}),
                       f"unknown claim {claim_id}")

        controls = row.get("leakage_controls") or []
        self.check(f"benchmark.{benchmark_id}.leakage_controls", len(controls) >= 1,
                   "no leakage controls")

        if role in {"external_component", "determinate_control"}:
            self.check(f"benchmark.{benchmark_id}.not_for_main_claims",
                       row.get("not_for_main_claims") is True,
                       "external/control benchmark is not marked not_for_main_claims")
            self.check(f"benchmark.{benchmark_id}.pool_with_main",
                       row.get("pool_with_main") is False,
                       "external/control benchmark is pooled with main")
            self.check(f"benchmark.{benchmark_id}.mapping", bool(row.get("mapping")),
                       "missing mapping statement")

        if role == "main_outcome":
            split = row.get("split") or {}
            self.check(f"benchmark.{benchmark_id}.company_disjoint", split.get("company_disjoint") is True,
                       "main benchmark must be company-disjoint")
            self.check(f"benchmark.{benchmark_id}.year_wise", split.get("year_wise") is True,
                       "main benchmark must be year-wise")
            test_years = split.get("test_years") or []
            seeds = split.get("seeds") or []
            self.check(f"benchmark.{benchmark_id}.test_years", len(test_years) >= 2,
                       "main benchmark needs at least two test years")
            self.check(f"benchmark.{benchmark_id}.seeds", len(seeds) >= 3,
                       "main benchmark needs at least three seeds")
            for key in ("n_claims", "n_train", "n_validation", "n_test"):
                self.check(f"benchmark.{benchmark_id}.{key}", isinstance(row.get(key), int) and row[key] > 0,
                           f"missing or invalid {key}")
            if all(isinstance(row.get(key), int) for key in ("n_claims", "n_train", "n_validation", "n_test")):
                total = row["n_train"] + row["n_validation"] + row["n_test"]
                self.check(f"benchmark.{benchmark_id}.split_sum", total == row["n_claims"],
                           f"split sum {total} != n_claims {row['n_claims']}")
            if isinstance(row.get("n_test"), int) and isinstance(row.get("n_verified_test"), int):
                self.check(f"benchmark.{benchmark_id}.verified_test",
                           row["n_verified_test"] <= row["n_test"],
                           "verified test count exceeds test count")
            if isinstance(row.get("n_validation"), int) and isinstance(row.get("n_verified_validation"), int):
                self.check(f"benchmark.{benchmark_id}.verified_validation",
                           row["n_verified_validation"] <= row["n_validation"],
                           "verified validation count exceeds validation count")
            if isinstance(row.get("n_company_disjoint_test_only"), int) and isinstance(row.get("n_test"), int):
                self.check(f"benchmark.{benchmark_id}.test_only_companies",
                           row["n_company_disjoint_test_only"] <= row["n_test"],
                           "test-only company count exceeds test count")

        if role == "prospective":
            self.check(f"benchmark.{benchmark_id}.pre_registration",
                       row.get("pre_registration") is True, "prospective benchmark is not pre-registered")
            self.check(f"benchmark.{benchmark_id}.registered_at", bool(row.get("registered_at")),
                       "missing registered_at")
            self.check(f"benchmark.{benchmark_id}.resolution", bool(row.get("resolution")),
                       "missing resolution")
            self.check(f"benchmark.{benchmark_id}.cutoff", bool(row.get("cutoff")), "missing cutoff")
            cases = row.get("cases") or []
            self.check(f"benchmark.{benchmark_id}.cases", bool(cases), "no case configs")
            for case in cases:
                self.check_case(benchmark_id, str(case))

        if role == "historical_backtest":
            builder = row.get("data_builder")
            self.check(f"benchmark.{benchmark_id}.data_builder", bool(builder),
                       "historical benchmark has no data builder")
            if builder:
                self.check(f"benchmark.{benchmark_id}.data_builder_exists",
                           self.resolve(str(builder)).exists(), f"builder not found: {builder}")
            builder_config = row.get("builder_config")
            if builder_config:
                self.check(f"benchmark.{benchmark_id}.builder_config_exists",
                           self.resolve(str(builder_config)).exists(),
                           f"builder config not found: {builder_config}")
            n_claims = row.get("n_claims")
            target = row.get("target_n_claims")
            self.check(f"benchmark.{benchmark_id}.n_claims", isinstance(n_claims, int) and n_claims > 0,
                       "missing or invalid n_claims")
            if isinstance(n_claims, int) and isinstance(target, int) and n_claims < target:
                self.warn(
                    f"benchmark.{benchmark_id}.smoke_size: {n_claims} claims built; "
                    f"target {target} before the final experiment"
                )

        data = row.get("data")
        if self.path_like(data):
            path = self.resolve(str(data))
            if path.exists():
                self.check(f"benchmark.{benchmark_id}.data_exists", True, str(path))
            elif row.get("requires_download") is True:
                self.warn(f"benchmark.{benchmark_id}.data_missing: {data} requires a public download")
            else:
                self.error(f"benchmark.{benchmark_id}.data_missing: {data}")
        elif role in {"external_component", "determinate_control"}:
            self.check(f"benchmark.{benchmark_id}.requires_download",
                       row.get("requires_download") is True,
                       "external data is neither a local path nor marked requires_download")
            if row.get("requires_download") is True:
                self.warn(f"benchmark.{benchmark_id}.data_missing: public data is not present locally")

        events = row.get("earnings_events")
        if self.path_like(events):
            path = self.resolve(str(events))
            self.check(f"benchmark.{benchmark_id}.earnings_events", path.exists(), str(path))

    def check_case(self, benchmark_id: str, case_path: str) -> None:
        path = self.resolve(case_path)
        label = f"benchmark.{benchmark_id}.case.{case_path}"
        if not path.exists():
            self.error(f"{label}: missing case file")
            return
        case = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self.check(f"{label}.case_id", bool(case.get("case_id")), "missing case_id")
        self.check(f"{label}.stem", str(case.get("case_id") or "") == path.stem,
                   "case_id does not match file stem")
        self.check(f"{label}.company", bool(case.get("company")), "missing company")
        self.check(f"{label}.ticker", bool(case.get("ticker")), "missing ticker")
        self.check(f"{label}.cutoff", bool(case.get("cutoff")), "missing cutoff")
        claims = case.get("claims") or []
        self.check(f"{label}.claims", bool(claims), "no claims")
        for claim in claims:
            self.check_case_claim(label, claim)

    def check_case_claim(self, label: str, claim: dict[str, Any]) -> None:
        claim_id = str(claim.get("claim_id") or "")
        prefix = f"{label}.claim.{claim_id}"
        self.check(f"{prefix}.id", bool(claim_id), "missing claim_id")
        self.check(f"{prefix}.text", bool(claim.get("text")), "missing text")
        kind = claim.get("metric_kind")
        self.check(f"{prefix}.metric_kind", kind in METRIC_KINDS, f"invalid metric_kind {kind}")
        template = claim.get("model_template")
        self.check(f"{prefix}.model_template", template in MODEL_TEMPLATES,
                   f"invalid model_template {template}")
        labels = claim.get("labels") or []
        self.check(f"{prefix}.labels", len(labels) >= 2, "need at least two labels")
        self.check(f"{prefix}.label_unique", len(labels) == len(set(labels)), "duplicate labels")

        thresholds = claim.get("thresholds") or []
        if thresholds:
            self.check(f"{prefix}.threshold_count", len(thresholds) == len(labels),
                       "threshold count must equal label count")
            for index, threshold in enumerate(thresholds):
                self.check(f"{prefix}.threshold.{index}.label",
                           threshold.get("label") in labels,
                           f"threshold label {threshold.get('label')} not in labels")
                low, high = threshold.get("low"), threshold.get("high")
                if low is not None and high is not None:
                    self.check(f"{prefix}.threshold.{index}.order", low < high,
                               f"low {low} is not below high {high}")
            if len(thresholds) == len(labels):
                self.check(f"{prefix}.threshold.first_open", thresholds[0].get("low") is None,
                           "first threshold must have an open lower bound")
                self.check(f"{prefix}.threshold.last_open", thresholds[-1].get("high") is None,
                           "last threshold must have an open upper bound")
                for index in range(len(thresholds) - 1):
                    left = thresholds[index].get("high")
                    right = thresholds[index + 1].get("low")
                    self.check(
                        f"{prefix}.threshold.adjacent.{index}",
                        left is not None and right is not None and math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-9),
                        f"threshold gap or overlap between {left} and {right}",
                    )

        date_windows = claim.get("date_windows") or []
        if kind == "date":
            self.check(f"{prefix}.date_windows", len(date_windows) == len(labels),
                       "date claim needs one window per label")
            for index, window in enumerate(date_windows):
                self.check(f"{prefix}.date_windows.{index}.label", window.get("label") in labels,
                           "date window label not in labels")
                try:
                    start = date.fromisoformat(str(window.get("start")))
                    end = date.fromisoformat(str(window.get("end")))
                    self.check(f"{prefix}.date_windows.{index}.order", start <= end,
                               f"start {start} is after end {end}")
                except ValueError:
                    self.error(f"{prefix}.date_windows.{index}: invalid ISO date")

        if kind in {"growth", "level", "margin", "capacity", "production"}:
            reference = claim.get("reference_value")
            self.check(f"{prefix}.reference_value", isinstance(reference, (int, float)),
                       "numeric metric needs a numeric reference_value")

    def run(self) -> dict[str, Any]:
        plan = self.load_plan()
        self.check_registries(plan)
        self.check_claims(plan)
        seen: set[str] = set()
        for row in plan.get("benchmarks") or []:
            if isinstance(row, dict):
                self.check_benchmark(plan, row, seen)
            else:
                self.error(f"benchmark entry is not a mapping: {row}")

        # Every main-outcome claim should be reachable from the main benchmark.
        main = next((row for row in plan.get("benchmarks") or [] if row.get("role") == "main_outcome"), None)
        if main:
            main_claims = set(main.get("claims") or [])
            for claim_id, claim in (plan.get("claims") or {}).items():
                names = claim.get("benchmark")
                names = [names] if isinstance(names, str) else (names or [])
                if main.get("id") in names:
                    self.check(f"benchmark.main.claim.{claim_id}", claim_id in main_claims,
                               "claim names the main benchmark but it is absent from its claim list")

        return {
            "ok": not self.errors,
            "plan": str(self.plan_path),
            "summary": {
                "benchmarks": len(plan.get("benchmarks") or []),
                "claims": len(plan.get("claims") or {}),
                "baselines": len(plan.get("baseline_registry") or {}),
                "metrics": len(plan.get("metric_registry") or {}),
                "errors": len(self.errors),
                "warnings": len(self.warnings),
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--report", default="")
    parser.add_argument("--show-checks", action="store_true")
    args = parser.parse_args()

    checker = Checker(Path(args.plan))
    result = checker.run()

    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    if result["errors"]:
        print("\nERRORS")
        for message in result["errors"]:
            print(f"  - {message}")
    if result["warnings"]:
        print("\nWARNINGS")
        for message in result["warnings"]:
            print(f"  - {message}")
    if args.show_checks:
        print("\nCHECKS")
        for row in result["checks"]:
            print(f"  [{'ok' if row['ok'] else 'FAIL'}] {row['name']} {row['detail']}")

    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
