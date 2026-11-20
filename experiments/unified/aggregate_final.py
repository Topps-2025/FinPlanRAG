"""Aggregate final FinPlanRAG results into a single JSON for paper tables."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from compute_metrics_suite import metric_values  # noqa: E402
from sklearn.metrics import matthews_corrcoef  # noqa: E402
from finplan_rag.decision.loss import (  # noqa: E402
    LossConfig,
    action_loss,
    answer_loss,
    credible_set,
    expected_action_loss,
)

BUCKETS = ["below", "within", "above"]
VARIANTS = ["findver", "finfact", "selfrag", "crag", "flare", "adaptive"]
NAMES = {
    "findver": "FinDVer",
    "finfact": "Fin-Fact",
    "selfrag": "Self-RAG",
    "crag": "CRAG",
    "flare": "FLARE",
    "adaptive": "Adaptive-RAG",
}
METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "auroc_macro",
    "auprc_macro",
    "ece",
    "brier",
    "nll",
    "selective_accuracy_50",
    "aurc",
]


def load_belief(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["claim_id"])] = np.asarray(row["belief"], dtype=float)
    return out


def load_named(path: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["variant"])][str(row["claim_id"])] = row
    return out


def named_belief(row: dict) -> np.ndarray:
    """Match scripts/convert_llm_predictions.py exactly."""
    parsed = row.get("parsed") or {}
    label = parsed.get("label")
    credible = [value for value in (parsed.get("credible_set") or []) if value in BUCKETS]
    if label not in BUCKETS:
        if credible:
            label = credible[0]
        else:
            # Missing label/credible set is an error; do not default to the
            # majority class (that would silently inflate baseline accuracy).
            return np.full(len(BUCKETS), 1.0 / len(BUCKETS), dtype=float)
    if not credible:
        credible = [label]
    confidence = min(0.95, max(0.34, float(parsed.get("confidence", 0.5))))
    values = {value: 0.0 for value in BUCKETS}
    values[label] = confidence
    rest = [value for value in credible if value != label]
    if rest:
        for value in rest:
            values[value] = (1.0 - confidence) / len(rest)
    else:
        values[label] = 1.0
    return np.asarray([values[value] for value in BUCKETS], dtype=float)


def metrics(probs: list[np.ndarray], truths: list[int]) -> dict[str, float]:
    values = metric_values(np.asarray(probs), np.asarray(truths))
    out = {key: float(values[key]) for key in METRICS}
    out["mcc"] = float(matthews_corrcoef(truths, np.asarray(probs).argmax(axis=1)))
    return out



def decision_loss(probs, truths, cfg):
    losses = []
    for prob, truth in zip(probs, truths):
        prob = np.asarray(prob, dtype=float)
        belief = {label: float(val) for label, val in zip(BUCKETS, prob)}
        pred = BUCKETS[int(np.argmax(prob))]
        R = credible_set(belief, cfg.alpha)
        values = {
            "answer": expected_action_loss("answer", belief, cfg, prediction=pred),
            "adjudicate": expected_action_loss("adjudicate", belief, cfg, credible_set=R),
            "abstain": cfg.abstain,
        }
        action = min(values, key=values.get)
        losses.append(action_loss(action, truth, cfg, prediction=pred, credible_set=R))
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--out", default=str(REPO_ROOT / "paper/results/final/results.json"))
    args = parser.parse_args()
    root = Path(args.root)
    data = json.loads(
        (root / "paper/data/dataset_v4.json").read_text(encoding="utf-8")
    )
    truth = {
        str(r["claim_id"]): str(r["outcome"]["bucket"])
        for r in data
        if r.get("split") == "test"
    }
    verified = [
        str(r["claim_id"])
        for r in data
        if r.get("split") == "test"
        and r.get("verification") not in {"text_only", "text_only_mismatch"}
    ]
    verified.sort()
    y = [BUCKETS.index(truth[c]) for c in verified]
    result: dict = {"n_verified": len(verified), "backbones": {}}

    rf_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    rf_path = root / "paper/results/rf_calibrated_repro/per_claim.jsonl"
    if not rf_path.exists():
        rf_path = root / "paper/results/benchmark_per_claim_verified_calibrated/per_claim.jsonl"
    for line in rf_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rf_rows[str(row["claim_id"])].append(row["probabilities"])
    rf = {key: np.mean(value, axis=0) for key, value in rf_rows.items()}
    result["rf"] = metrics([rf[c] for c in verified], y)

    cfg = LossConfig()
    rf_losses = []
    for c in verified:
        prob = rf[c]
        belief = {label: float(val) for label, val in zip(BUCKETS, prob)}
        pred = BUCKETS[int(np.argmax(prob))]
        R = credible_set(belief, cfg.alpha)
        values = {
            "answer": expected_action_loss("answer", belief, cfg, prediction=pred),
            "adjudicate": expected_action_loss("adjudicate", belief, cfg, credible_set=R),
            "abstain": cfg.abstain,
        }
        action = min(values, key=values.get)
        rf_losses.append(
            action_loss(action, truth[c], cfg, prediction=pred, credible_set=R)
        )
    result["rf"]["decision_loss"] = float(np.mean(rf_losses))

    canonical = {
        "deepseek": root / "paper/results/named_baselines/per_claim.jsonl",
        "glm": root / "paper/results/glm/named/per_claim.jsonl",
        "qwen3": root / "paper/results/qwen3/named/per_claim.jsonl",
    }
    for backbone, named_path in canonical.items():
        dir_name = {"deepseek": "deepseek", "glm": "glm", "qwen3": "qwen"}[backbone]
        entry: dict = {"methods": {}}
        named = load_named(named_path)
        for variant in VARIANTS:
            rows = named.get(variant, {})
            ids = [c for c in verified if c in rows]
            if len(ids) < len(verified) * 0.9:
                continue
            probs = [named_belief(rows[c]) for c in ids]
            values = metrics(probs, [BUCKETS.index(truth[c]) for c in ids])
            losses = [
                answer_loss(
                    str((rows[c].get("parsed") or {}).get("label", "")),
                    truth[c],
                    cfg,
                )
                if (rows[c].get("parsed") or {}).get("label") in BUCKETS
                else cfg.wrong
                for c in ids
            ]
            values["decision_loss"] = float(np.mean(losses))
            values["n"] = len(ids)
            entry["methods"][NAMES[variant]] = values

        llm_path = root / f"paper/results/unified/vague_v2_{dir_name}/per_claim.jsonl"
        if llm_path.exists():
            llm = load_belief(llm_path)
            ids = [c for c in verified if c in llm]
            values = metrics([llm[c] for c in ids], [BUCKETS.index(truth[c]) for c in ids])
            values["n"] = len(ids)
            entry["methods"]["AnchorEvolve-LLM"] = values

        ens_path = root / f"paper/results/unified/ensemble_repro_{dir_name}.json"
        ens_claims = (
            root
            / f"paper/results/unified/ensemble_repro_{dir_name}_per_claim.jsonl"
        )
        if ens_path.exists() and ens_claims.exists():
            summary = json.loads(ens_path.read_text(encoding="utf-8"))
            if "AnchorEvolve-LLM" in entry["methods"]:
                entry["methods"]["AnchorEvolve-LLM"]["decision_loss"] = float(
                    summary["grid"]["w=1.0"]["mean_loss"]
                )
            grid = {}
            for weight in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0):
                item = dict(summary["grid"][f"w={weight:.1f}"])
                item["decision_loss"] = item.get("mean_loss")
                grid[f"{weight:.1f}"] = item
            entry["ensemble_grid"] = grid
            # The headline system is the LLM-dependent AnchorEvolve fusion.
            # The calibrated RF is retained as a separate non-LLM diagnostic;
            # it must not be copied into every backbone's FinPlanRAG row.
            fusion = load_belief(ens_claims)
            ids = [c for c in verified if c in fusion]
            fusion_values = metrics(
                [fusion[c] for c in ids],
                [BUCKETS.index(truth[c]) for c in ids],
            )
            selected_weight = float(summary.get("selected_weight", 0.1))
            fusion_values["decision_loss"] = float(
                summary["grid"][f"w={selected_weight:.1f}"]["mean_loss"]
            )
            fusion_values["decision_head"] = f"loss-aware gate on fusion w={selected_weight:.1f}"
            fusion_values["fusion_weight"] = selected_weight
            fusion_values["n"] = len(ids)
            entry["methods"]["FinPlanRAG"] = fusion_values

            rf_values = dict(result["rf"])
            rf_values["decision_head"] = "loss-aware gate on shared calibrated RF"
            rf_values["n"] = result["n_verified"]
            entry["methods"]["Shared calibrated RF diagnostic"] = rf_values
        entry["methods"]["FinPlan-RF"] = dict(result["rf"])
        result["backbones"][backbone] = entry

        docfinqa = (
            root
            / f"paper/results/external_tier1/docfinqa/test300/{backbone}/analysis/metrics_summary.json"
        )
        if docfinqa.exists():
            doc_metrics = json.loads(docfinqa.read_text(encoding="utf-8"))["metrics"]
            for suffix, name, key in (
                ("_planner12", "planner12", "planner"),
                ("_planner16", "planner16", "planner"),
                ("_planner_fallback", "planner_fallback", "planner_fallback"),
            ):
                extra = root / (
                    f"paper/results/external_tier1/docfinqa/test300/{dir_name}{suffix}/analysis/metrics_summary.json"
                )
                if extra.exists():
                    doc_metrics[name] = json.loads(extra.read_text(encoding="utf-8"))["metrics"][key]
            entry["docfinqa"] = doc_metrics
        named_doc = root / (
            f"paper/results/external_tier1/docfinqa/test300/{dir_name}_named/analysis/metrics_summary.json"
        )
        if named_doc.exists():
            entry["docfinqa_named"] = json.loads(named_doc.read_text(encoding="utf-8"))["metrics"]
        findver = (
            root
            / f"paper/results/external_tier1/findver/testmini/{backbone}/analysis/metrics_summary.json"
        )
        if findver.exists():
            findver_metrics = json.loads(findver.read_text(encoding="utf-8"))["metrics"]
            fallback = root / (
                f"paper/results/external_tier1/findver/testmini/{dir_name}_planner_fallback/analysis/metrics_summary.json"
            )
            if fallback.exists():
                findver_metrics["planner_fallback"] = json.loads(
                    fallback.read_text(encoding="utf-8")
                )["metrics"]["planner_fallback"]
            entry["findver"] = findver_metrics
        router = root / f"paper/results/unified/router_v3_{dir_name}/summary.json"
        if not router.exists():
            router = root / f"paper/results/unified/router_{dir_name}/summary.json"
        if router.exists():
            entry["router"] = json.loads(router.read_text(encoding="utf-8"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
