"""Compare AnchorEvolve-LLM, FinPlan-RF, and their ensemble on the vague track."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_vague_llm import BUCKETS, ALL_SETS, ans_loss, set_loss, gate, credible_set, LOSS  # noqa: E402


def load_llm(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out[row["claim_id"]] = row
    return out


def load_rf(path: Path) -> dict[str, np.ndarray]:
    by: dict[str, list[np.ndarray]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by.setdefault(row["claim_id"], []).append(np.asarray(row["probabilities"], dtype=float))
    return {key: np.mean(values, axis=0) for key, values in by.items()}


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    truth = [row["true"] for row in rows]
    pred = [row["pred"] for row in rows]
    acc = sum(a == b for a, b in zip(pred, truth)) / n
    recalls, f1s = [], []
    for bucket in BUCKETS:
        tp = sum(a == bucket and b == bucket for a, b in zip(pred, truth))
        pp = sum(a == bucket for a in pred)
        tt = sum(b == bucket for b in truth)
        recalls.append(tp / tt if tt else 0.0)
        f1s.append(2 * tp / max(1, pp + tt))
    losses = []
    for row in rows:
        y = BUCKETS.index(row["true"])
        action = row["action"]
        if action["mode"] == "answer":
            losses.append(ans_loss(int(action["a"]), y))
        elif action["mode"] == "adjudicate":
            losses.append(set_loss(action["R"], y))
        else:
            losses.append(LOSS["abstain"])
    mean_loss = float(np.mean(losses))
    fixed_set_losses = []
    for row in rows:
        y = BUCKETS.index(row["true"])
        fixed_set_losses.append(set_loss([BUCKETS.index(b) for b in row["credible"]], y))
    fixed_set_loss = float(np.mean(fixed_set_losses))
    best = min(float(np.mean([ans_loss(a, BUCKETS.index(t)) for t in truth])) for a in range(3))
    for subset in ALL_SETS:
        best = min(best, float(np.mean([set_loss(subset, BUCKETS.index(t)) for t in truth])))
    best = min(best, LOSS["abstain"])
    return {
        "n": n,
        "accuracy": acc,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "mean_loss": mean_loss,
        "oracle_fixed_loss": best,
        "regret": mean_loss - best,
        "fixed_set_loss": fixed_set_loss,
        "fixed_set_regret": fixed_set_loss - best,
        "coverage": float(np.mean([row["true"] in row["credible"] for row in rows])),
        "mean_set_size": float(np.mean([len(row["credible"]) for row in rows])),
    }


def make_row(claim_id: str, truth: str, belief: np.ndarray) -> dict[str, Any]:
    belief = np.maximum(belief, 1e-12)
    belief = belief / belief.sum()
    action = gate(belief)
    return {"claim_id": claim_id, "true": truth, "pred": BUCKETS[int(np.argmax(belief))],
            "belief": belief.tolist(), "credible": [BUCKETS[i] for i in credible_set(belief)],
            "action": action}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True)
    parser.add_argument("--rf", required=True)
    parser.add_argument("--data", default=str(Path(__file__).resolve().parent / "data/dataset_v4.json"))
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--weight", type=float, default=0.2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    llm = load_llm(Path(args.llm))
    rf = load_rf(Path(args.rf))
    common = sorted(set(llm) & set(rf))
    if args.verified_only:
        records = json.loads(Path(args.data).read_text(encoding="utf-8"))
        verified = {
            str(rec.get("claim_id")) for rec in records
            if rec.get("split") == "test" and rec.get("verification") not in {"text_only", "text_only_mismatch"}
        }
        common = [cid for cid in common if cid in verified]
    print("llm", len(llm), "rf", len(rf), "common", len(common))
    records = []
    for cid in common:
        row = llm[cid]
        if row.get("true") not in BUCKETS:
            continue
        lb = np.asarray(row["belief"], dtype=float)
        rb = np.asarray(rf[cid], dtype=float)
        records.append((cid, row["true"], lb, rb))
    results = {}
    selected_rows = None
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        rows = [make_row(cid, truth, w * lb + (1 - w) * rb) for cid, truth, lb, rb in records]
        results[f"w={w:.1f}"] = evaluate(rows)
        if abs(w - args.weight) < 1e-9:
            selected_rows = rows
    if selected_rows is None:
        selected_rows = [make_row(cid, truth, args.weight * lb + (1 - args.weight) * rb)
                         for cid, truth, lb, rb in records]
    out = {"n": len(records), "selected_weight": args.weight, "grid": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    with (out_path.parent / (out_path.stem + "_per_claim.jsonl")).open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
