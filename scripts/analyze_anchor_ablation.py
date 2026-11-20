"""Anchor-component ablations on stored AnchorEvolve per-claim outputs.

The script re-runs the deterministic belief update and loss-aware gate on the
LLM anchors already stored in ``vague_v2_{deepseek,glm,qwen}/per_claim.jsonl``.
It does not call an LLM. This isolates the anchor mechanism from the calibrated
random-forest decision head used by the final system.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BUCKETS = ["below", "within", "above"]
LOSS = {"over": 12.0, "under": 8.0, "wrong": 10.0, "miss": 8.0, "width": 2.0, "abstain": 5.0}
P_E = 0.7
FLOOR = 0.02
BACKBONES = [("deepseek", "deepseek"), ("glm", "glm"), ("qwen3", "qwen")]


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.maximum(np.asarray(v, dtype=float), 1e-12)
    return v / v.sum()


def priors(records: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    glob: Counter = Counter()
    sect: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        if rec.get("split") not in {"dev", "validation"}:
            continue
        bucket = (rec.get("outcome") or {}).get("bucket")
        if bucket not in BUCKETS:
            continue
        glob[bucket] += 1
        sect[str(rec.get("sector", "other"))][bucket] += 1
    base = normalize(np.array([glob.get(b, 0) + 1.0 for b in BUCKETS]))
    sectors = {
        s: normalize(np.array([c.get(b, 0) + 1.0 for b in BUCKETS]))
        for s, c in sect.items()
    }
    return base, sectors


def belief(parsed: dict, prior: np.ndarray, drop_types=(), no_hard=False,
           no_scores=False, no_anchors=False) -> np.ndarray:
    b = prior.copy()
    anchors = [] if no_anchors or not parsed else (parsed.get("anchors") or [])
    for anchor in anchors:
        implied = str(anchor.get("implied_bucket", "")).lower()
        if implied not in BUCKETS:
            continue
        if anchor.get("type") in drop_types:
            continue
        rel = max(0.0, min(1.0, float(anchor.get("reliability", 0.5) or 0.5)))
        target = BUCKETS.index(implied)
        hard = bool(anchor.get("hard", False)) and rel >= 0.65 and not no_hard
        if hard:
            b = np.array([1.0 if i == target else FLOOR for i in range(3)], dtype=float)
        else:
            like = np.full(3, (1.0 - P_E) / 2.0)
            like[target] = P_E
            b = b * np.power(like, rel)
        b = normalize(b)
    if not no_scores and parsed:
        scores = parsed.get("bucket_scores")
        if isinstance(scores, dict):
            try:
                marginal = normalize(np.array([float(scores.get(x, 0.0)) for x in BUCKETS]))
                b = normalize(0.5 * b + 0.5 * marginal)
            except Exception:
                pass
    return b


def ans_loss(a: int, y: int) -> float:
    if a == y:
        return 0.0
    if a == 2 and y == 0:
        return LOSS["over"]
    if a == 0 and y == 2:
        return LOSS["under"]
    return LOSS["wrong"]


def set_loss(R: list[int], y: int) -> float:
    return (0.0 if y in R else LOSS["miss"]) + LOSS["width"] * max(0, len(R) - 1)


def gate(b: np.ndarray, lam: float = 0.9) -> tuple[str, object]:
    exp_ans = [sum(float(b[y]) * ans_loss(a, y) for y in range(3)) for a in range(3)]
    a = int(np.argmin(exp_ans))
    order = np.argsort(-b)
    R: list[int] = []
    total = 0.0
    for idx in order:
        R.append(int(idx))
        total += float(b[int(idx)])
        if total >= lam:
            break
    R = sorted(R)
    exp_set = sum(float(b[y]) * set_loss(R, y) for y in range(3))
    if exp_ans[a] <= exp_set and exp_ans[a] <= LOSS["abstain"]:
        return "answer", a
    if exp_set <= LOSS["abstain"]:
        return "adjudicate", R
    return "abstain", R


def loss_of(true: str, b: np.ndarray) -> float:
    y = BUCKETS.index(true)
    mode, payload = gate(b)
    if mode == "answer":
        return ans_loss(int(payload), y)
    if mode == "adjudicate":
        return set_loss(payload, y)
    return LOSS["abstain"]


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    recalls = []
    f1s = []
    for bucket in BUCKETS:
        tp = sum(r["pred"] == bucket and r["true"] == bucket for r in rows)
        pred_pos = sum(r["pred"] == bucket for r in rows)
        true_pos = sum(r["true"] == bucket for r in rows)
        recalls.append(tp / true_pos if true_pos else 0.0)
        f1s.append(2 * tp / max(1, pred_pos + true_pos))
    return {
        "accuracy": sum(r["pred"] == r["true"] for r in rows) / n,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "decision_loss": float(np.mean([r["loss"] for r in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="paper/data/dataset_v4.json")
    parser.add_argument("--results", default="paper/results/unified")
    parser.add_argument("--out", default="paper/results/unified/anchor_ablation.json")
    args = parser.parse_args()

    records = json.loads(Path(args.data).read_text(encoding="utf-8"))
    base, sectors = priors(records)
    by_id = {r["claim_id"]: r for r in records}
    ablations = {
        "full": {},
        "no_hard": {"no_hard": True},
        "no_scores": {"no_scores": True},
        "no_anchors": {"no_anchors": True},
        "drop_guidance_width": {"drop_types": ("guidance_width",)},
        "drop_current_quarter_growth": {"drop_types": ("current_quarter_growth",)},
        "drop_company_history": {"drop_types": ("company_history",)},
        "drop_sector_history": {"drop_types": ("sector_history",)},
    }
    output: dict[str, dict] = {}
    for backbone, dirname in BACKBONES:
        rows = [
            json.loads(line)
            for line in Path(args.results, f"vague_v2_{dirname}", "per_claim.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        output[backbone] = {}
        for name, kwargs in ablations.items():
            ablated = []
            for row in rows:
                rec = by_id[row["claim_id"]]
                sector_prior = sectors.get(str(rec.get("sector", "other")))
                prior = base if sector_prior is None else normalize(0.5 * base + 0.5 * sector_prior)
                b = belief(row.get("llm"), prior, **kwargs)
                ablated.append({
                    "true": row["true"],
                    "pred": BUCKETS[int(np.argmax(b))],
                    "loss": loss_of(row["true"], b),
                })
            output[backbone][name] = metrics(ablated)
            if name == "full":
                output[backbone]["baseline_pred_match"] = sum(
                    1 for a, r in zip(ablated, rows) if a["pred"] == r["pred"]
                ) / len(rows)
    Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
