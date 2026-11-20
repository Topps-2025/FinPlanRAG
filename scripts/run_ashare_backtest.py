"""Evaluate baselines on the historical A-share annual-report backtest.

The benchmark file is built by ``scripts/build_ashare_backtest.py``.  This
runner supports two deterministic baselines that need no model call and two
AnchorEvolve variants that use the stored evidence directly:

* ``last_year``: carry forward the prior-year growth, margin or disclosure date.
* ``run_rate``: annualize the Q3 year-to-date value; use the Q3 margin; use the
  median historical disclosure date.
* ``full``: AnchorEvolve with qualification, evolution, calibration, bridge and
  full-action readout, using the benchmark evidence only.
* ``no_bridge``: the same system with the numeric bridge disabled.

The deterministic methods can be run now.  The AnchorEvolve methods require the
configured LLM key and write per-claim audits.  No baseline reads the outcome.

Usage:
    python scripts/run_ashare_backtest.py --methods last_year,run_rate
    python scripts/run_ashare_backtest.py --methods full,no_bridge
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from anchor_engine import (  # noqa: E402
    CaseSpec,
    ClaimContract,
    answer_loss,
    load_config,
    run_claim_safe,
    set_loss,
)
from forecast_bridge import label_for_value  # noqa: E402

DEFAULT_DATA = REPO_ROOT / "paper" / "data" / "ashare_backtest" / "claims.json"
DEFAULT_OUT = REPO_ROOT / "paper" / "results" / "ashare_backtest"
LABELS = ("below", "within", "above")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_value(claim: dict[str, Any], period: str) -> float | None:
    for item in claim.get("evidence") or []:
        if str(item.get("period")) == period and item.get("value") is not None:
            try:
                return float(item["value"])
            except (TypeError, ValueError):
                continue
    return None


def _dates(claim: dict[str, Any]) -> list[str]:
    text = " ".join(str(item.get("text") or "") for item in claim.get("evidence") or [])
    return sorted(set(__import__("re").findall(r"\d{4}-\d{2}-\d{2}", text)))


def _target_year(claim: dict[str, Any]) -> int | None:
    window = (claim.get("date_windows") or [{}])[0]
    start = str(window.get("start") or "")
    if start[:4].isdigit():
        return int(start[:4])
    return None


def _shift_date(claim: dict[str, Any], value: str) -> str | None:
    year = _target_year(claim)
    if year is None:
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return f"{year:04d}-{parsed.month:02d}-{parsed.day:02d}"


def _predict_last_year(claim: dict[str, Any]) -> float | None:
    if claim["metric_kind"] == "growth":
        return float(claim["prior_growth"])
    if claim["metric_kind"] == "margin":
        return float(claim["reference_value"])
    if claim["metric_kind"] == "date":
        dates = _dates(claim)
        return _shift_date(claim, dates[-1]) if dates else None
    return None


def _predict_run_rate(claim: dict[str, Any]) -> float | None:
    if claim["metric_kind"] == "growth":
        if "营业收入" in claim["metric"]:
            ytd = _evidence_value(claim, f"{claim['year']}Q1-Q3")
            prior = float(claim["reference_value"])
            if ytd is None or prior == 0:
                return None
            annualized = ytd / 3.0 * 4.0
            return (annualized / prior - 1.0) * 100.0
        if "归母净利润" in claim["metric"]:
            ytd = _evidence_value(claim, f"{claim['year']}Q1-Q3")
            prior = float(claim["reference_value"])
            if ytd is None or prior == 0:
                return None
            annualized = ytd / 3.0 * 4.0
            return (annualized / prior - 1.0) * 100.0
    if claim["metric_kind"] == "margin":
        value = _evidence_value(claim, f"{claim['year']}Q1-Q3")
        if value is not None:
            return value
    if claim["metric_kind"] == "date":
        dates = _dates(claim)
        if dates:
            parsed = [date.fromisoformat(value) for value in dates]
            middle = parsed[len(parsed) // 2]
            return _shift_date(claim, middle.isoformat())
    return None


def _prediction_label(claim: dict[str, Any], prediction: Any) -> str | None:
    if prediction is None:
        return None
    if claim["metric_kind"] == "date":
        value = str(prediction)
        for window in claim.get("date_windows") or []:
            if str(window["start"]) <= value <= str(window["end"]):
                return str(window["label"])
        return None
    return label_for_value(float(prediction), claim.get("thresholds") or [])


def _actual_loss(claim: dict[str, Any], action: str, point: str | None, chosen: list[str], config: dict) -> float:
    truth = claim["outcome"]["label"]
    loss = dict(config.get("loss") or {})
    if action == "abstain":
        return float(loss.get("abstain") or 5.0)
    if action == "adjudicate":
        return set_loss(chosen, truth, loss)
    return answer_loss(point or "", truth, list(claim["labels"]), loss)


def _numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    n = len(rows)
    correct = sum(row["prediction"] == row["truth"] for row in rows)
    recalls = []
    for label in sorted({row["truth"] for row in rows}):
        subset = [row for row in rows if row["truth"] == label]
        if subset:
            recalls.append(sum(row["prediction"] == label for row in subset) / len(subset))
    macro = 0.0
    for label in sorted({row["truth"] for row in rows}):
        tp = sum(row["prediction"] == label and row["truth"] == label for row in rows)
        fp = sum(row["prediction"] == label and row["truth"] != label for row in rows)
        fn = sum(row["prediction"] != label and row["truth"] == label for row in rows)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        macro += 2 * precision * recall / max(precision + recall, 1e-9)

    numeric_rows = []
    date_rows = []
    for row in rows:
        kind = row.get("metric_kind")
        if kind in {"growth", "margin"}:
            prediction = _numeric(row.get("prediction_value"))
            truth = _numeric(row.get("truth_value"))
            if prediction is not None and truth is not None:
                numeric_rows.append((prediction, truth))
        elif kind == "date":
            prediction_date = _date(row.get("prediction_value"))
            truth_date = _date(row.get("truth_value"))
            if prediction_date is not None and truth_date is not None:
                date_rows.append((prediction_date, truth_date))

    numeric_mae = numeric_rmse = numeric_mape = None
    if numeric_rows:
        errors = [prediction - truth for prediction, truth in numeric_rows]
        numeric_mae = sum(abs(error) for error in errors) / len(errors)
        numeric_rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
        percentage = [
            abs(error) / abs(truth) * 100.0
            for error, (_, truth) in zip(errors, numeric_rows)
            if truth != 0
        ]
        numeric_mape = sum(percentage) / len(percentage) if percentage else None
    date_mae_days = None
    if date_rows:
        date_mae_days = sum(abs((prediction - truth).days) for prediction, truth in date_rows) / len(date_rows)

    return {
        "n": n,
        "accuracy": correct / n,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": macro / max(len({row["truth"] for row in rows}), 1),
        "decision_loss": sum(row["loss"] for row in rows) / n,
        "coverage": sum(row["action"] != "abstain" for row in rows) / n,
        "mean_set_size": sum(len(row["set"]) for row in rows) / n,
        "abstain_rate": sum(row["action"] == "abstain" for row in rows) / n,
        "n_numeric": len(numeric_rows),
        "numeric_mae": numeric_mae,
        "numeric_rmse": numeric_rmse,
        "numeric_mape": numeric_mape,
        "n_date": len(date_rows),
        "date_mae_days": date_mae_days,
    }


def _summary_by_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("metric_kind") or "other"), []).append(row)
    return {kind: _summary(subset) for kind, subset in sorted(grouped.items())}


def _run_deterministic(claims: list[dict[str, Any]], method: str, config: dict) -> list[dict[str, Any]]:
    rows = []
    for claim in claims:
        prediction = _predict_last_year(claim) if method == "last_year" else _predict_run_rate(claim)
        label = _prediction_label(claim, prediction)
        action = "answer" if label else "abstain"
        chosen = [label] if label else []
        rows.append({
            "claim_id": claim["claim_id"],
            "method": method,
            "metric_kind": claim.get("metric_kind"),
            "prediction_value": prediction,
            "prediction": label,
            "truth": claim["outcome"]["label"],
            "truth_value": (claim.get("outcome") or {}).get("value"),
            "action": action,
            "set": chosen,
            "loss": _actual_loss(claim, action, label, chosen, config),
        })
    return rows


def _engine_row(claim: dict[str, Any], method: str, config: dict, out_dir: Path) -> dict[str, Any]:
    contract = ClaimContract.from_dict(claim)
    spec = CaseSpec(
        case_id=claim["claim_id"],
        question=claim["metric"],
        company=claim["company"],
        ticker=claim["ticker"],
        target_period=claim["target_period"],
        claims=[contract],
        cutoff=claim["cutoff"],
        source="ashare_backtest",
    )
    run_config = dict(config)
    run_config["bridge"] = dict(run_config.get("bridge") or {})
    if method == "no_bridge":
        run_config["bridge"]["enabled"] = False
    evidence = []
    for item in claim.get("evidence") or []:
        row = dict(item)
        row.setdefault("source_id", row.get("id"))
        row.setdefault("prompt_text", row.get("text"))
        evidence.append(row)
    result = run_claim_safe(
        contract,
        spec,
        run_config,
        date.fromisoformat(claim["cutoff"]),
        out_dir=out_dir / "audits" / claim["claim_id"],
        evidence_override=evidence,
    )
    readout = result.get("readout") or {}
    action = str(readout.get("action") or "abstain")
    point = readout.get("point")
    chosen = list(readout.get("set") or [])
    llm_error = (result.get("llm") or {}).get("error")
    prediction_value = (result.get("forecast") or {}).get("point")
    if claim.get("metric_kind") == "date":
        prediction_value = (result.get("date_estimate") or {}).get("point")
    return {
        "claim_id": claim["claim_id"],
        "method": method,
        "metric_kind": claim.get("metric_kind"),
        "prediction_value": prediction_value,
        "prediction": point,
        "truth": claim["outcome"]["label"],
        "truth_value": (claim.get("outcome") or {}).get("value"),
        "action": action,
        "set": chosen,
        "loss": _actual_loss(claim, action, point, chosen, config),
        "audit_dir": str(out_dir / "audits" / claim["claim_id"]),
        "error": result.get("error") or llm_error,
    }


def _run_engine(
    claims: list[dict[str, Any]],
    method: str,
    config: dict,
    out_dir: Path,
    workers: int = 1,
) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_engine_row(claim, method, config, out_dir) for claim in claims]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_engine_row, claim, method, config, out_dir): claim["claim_id"]
            for claim in claims
        }
        for future in as_completed(futures):
            rows.append(future.result())
    order = {claim["claim_id"]: index for index, claim in enumerate(claims)}
    rows.sort(key=lambda row: order.get(row["claim_id"], 0))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--methods", default="last_year,run_rate")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "anchor_cli.yaml"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    payload = _load(Path(args.data))
    claims = payload.get("claims") or []
    if args.limit:
        claims = claims[: args.limit]
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    config = load_config(args.config)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for method in methods:
        if method in {"last_year", "run_rate"}:
            rows = _run_deterministic(claims, method, config)
        elif method in {"full", "no_bridge"}:
            rows = _run_engine(claims, method, config, out_root / method, workers=args.workers)
        else:
            raise SystemExit(f"unknown method: {method}")
        method_dir = out_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        with (method_dir / "per_claim.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries[method] = {
            **_summary(rows),
            "by_metric": _summary_by_metric(rows),
        }

    (out_root / "summary.json").write_text(
        json.dumps({"schema": "finplan-ashare-backtest-results-v2", "data": args.data,
                    "n_claims": len(claims), "methods": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"n_claims": len(claims), "methods": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
