"""AnchorEvolve vague-claim branch with LLM-extracted anchors.

The LLM proposes L2-style anchors (hard constraints / soft signals) from the
claim and cutoff evidence. A deterministic belief update and loss-aware gate
turn those anchors into answer / credible set / abstain.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from deepseek_client import LLMClient  # noqa: E402

BUCKETS = ["below", "within", "above"]
LOSS = {"over": 12.0, "under": 8.0, "wrong": 10.0, "miss": 8.0, "width": 2.0, "abstain": 5.0}
ALL_SETS = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
HARD_FLOOR = 0.02
SOFT_P_E = 0.7  # informativeness 0.4: p_e = .5 + .5 * .4


def parse_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    brace = re.search(r"\{.*\}", text, flags=re.S)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.maximum(np.asarray(v, dtype=float), 1e-12)
    return v / v.sum()


def credible_set(b: np.ndarray, alpha: float = 0.9) -> list[int]:
    order = np.argsort(-b)
    out: list[int] = []
    total = 0.0
    for idx in order:
        out.append(int(idx))
        total += float(b[int(idx)])
        if total >= alpha:
            break
    return sorted(out)


def ans_loss(a: int, y: int) -> float:
    if a == y:
        return 0.0
    if a == 2 and y == 0:
        return LOSS["over"]
    if a == 0 and y == 2:
        return LOSS["under"]
    return LOSS["wrong"]


def set_loss(R: Sequence[int], y: int) -> float:
    return (0.0 if y in R else LOSS["miss"]) + LOSS["width"] * max(0, len(R) - 1)


def gate(b: np.ndarray, lam: float = 0.9) -> dict[str, Any]:
    exp_ans = [sum(float(b[y]) * ans_loss(a, y) for y in range(3)) for a in range(3)]
    a = int(np.argmin(exp_ans))
    R = credible_set(b, lam)
    exp_set = sum(float(b[y]) * set_loss(R, y) for y in range(3))
    if exp_ans[a] <= exp_set and exp_ans[a] <= LOSS["abstain"]:
        return {"mode": "answer", "a": a, "R": R}
    if exp_set <= LOSS["abstain"]:
        return {"mode": "adjudicate", "a": a, "R": R}
    return {"mode": "abstain", "a": a, "R": R}


def prior_from_history(records: Sequence[dict[str, Any]]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    glob = Counter()
    sect: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        if rec.get("split") not in {"dev", "validation"}:
            continue
        y = str(rec.get("outcome", {}).get("bucket", ""))
        if y not in BUCKETS:
            continue
        glob[y] += 1
        sect[str(rec.get("sector", "other"))][y] += 1
    base = normalize(np.array([glob.get(b, 0) + 1.0 for b in BUCKETS]))
    sectors = {
        s: normalize(np.array([c.get(b, 0) + 1.0 for b in BUCKETS]))
        for s, c in sect.items()
    }
    return base, sectors, {b: glob.get(b, 0) for b in BUCKETS}


def anchor_prompt(rec: dict[str, Any], prior: np.ndarray, sector_prior: np.ndarray | None) -> str:
    evidence = []
    for item in rec.get("evidence", []):
        keep = {
            key: item.get(key)
            for key in ("kind", "source_type", "origin", "date", "low_musd", "high_musd",
                        "value_musd", "reliability", "informativeness")
            if item.get(key) is not None
        }
        evidence.append(keep)
    payload = {
        "claim": str(rec.get("claim_text", ""))[:1200],
        "company": rec.get("company"),
        "ticker": rec.get("ticker"),
        "sector": rec.get("sector"),
        "cutoff": rec.get("cutoff"),
        "target_year": rec.get("target_year"),
        "target_quarter": rec.get("target_quarter"),
        "guidance_low_musd": rec.get("guidance_low_musd"),
        "guidance_high_musd": rec.get("guidance_high_musd"),
        "guidance_mid_musd": rec.get("guidance_mid_musd"),
        "guidance_width_pct": rec.get("guidance_width_pct"),
        "current_q_musd": rec.get("current_q_musd"),
        "current_q_reliability": rec.get("current_reliability"),
        "observed_bucket_prior": dict(zip(BUCKETS, [round(float(x), 4) for x in prior])),
        "sector_bucket_prior": None if sector_prior is None else dict(
            zip(BUCKETS, [round(float(x), 4) for x in sector_prior])
        ),
        "evidence": evidence,
    }
    return (
        "You are an AnchorEvolve adjudicator for a vague forward-looking financial claim.\n"
        "Use ONLY information available at the cutoff. Do not use the realized outcome.\n"
        "The claim asks whether reported quarterly revenue will land below, within, or above the guidance range.\n"
        "The guidance range defines the three buckets; it is NOT by itself evidence that the outcome will be within.\n"
        "Use the current-quarter value relative to the guidance midpoint, the guidance width, and the observed base rates:\n"
        "- mid_to_current_ratio > 1 means the midpoint requires sequential growth; the larger it is, the more likely below.\n"
        "- mid_to_current_ratio < 1 means the midpoint implies a sequential decline; the smaller it is, the more likely above.\n"
        "- A narrow guidance width makes within more likely; a very wide range makes outside buckets more likely.\n"
        "- Base rates are priors, not destiny; update them with these features.\n"
        "Extract explicit anchors and return JSON only with this schema:\n"
        "{\n"
        '  "anchors": [\n'
        '    {"level":"L2","type":"current_quarter_growth|guidance_width|sector_history|company_history|other",\n'
        '     "relation":"bound|precedence|lead_time|duration|conservation|rate|unit_conversion|scope",\n'
        '     "value":"...", "scope":"company|sector|market", "reliability":0.0,\n'
        '     "hard":false, "implied_bucket":"below|within|above|null", "rationale":"..."}\n'
        "  ],\n"
        '  "bucket_scores":{"below":0.0,"within":0.0,"above":0.0},\n'
        '  "rationale":"..."\n'
        "}\n"
        "Rules: a hard anchor is an explicit constraint that can rule out a bucket "
        "(e.g. a reliable current-quarter value far above the guidance range); most management "
        "guidance is a soft signal. reliability is in [0,1]. bucket_scores must sum to 1.\n"
        "Input:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def belief_from_anchors(prior: np.ndarray, parsed: dict[str, Any] | None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    b = prior.copy()
    anchors: list[dict[str, Any]] = []
    if not parsed:
        return b, anchors
    for anchor in parsed.get("anchors", []) or []:
        if not isinstance(anchor, dict):
            continue
        implied = str(anchor.get("implied_bucket", "")).lower()
        if implied not in BUCKETS:
            continue
        rel = float(anchor.get("reliability", 0.5) or 0.5)
        rel = max(0.0, min(1.0, rel))
        target = BUCKETS.index(implied)
        hard = bool(anchor.get("hard", False)) and rel >= 0.65
        anchors.append({
            "type": anchor.get("type"),
            "relation": anchor.get("relation"),
            "implied_bucket": implied,
            "reliability": rel,
            "hard": hard,
            "scope": anchor.get("scope"),
        })
        if hard:
            b = np.array([1.0 if i == target else HARD_FLOOR for i in range(3)], dtype=float)
        else:
            # The soft-anchor likelihood is fixed by the anchor's
            # informativeness (0.4 for management guidance); reliability enters
            # as the exponent, not as the height of the likelihood.
            like = np.full(3, (1.0 - SOFT_P_E) / 2.0, dtype=float)
            like[target] = SOFT_P_E
            b = b * np.power(like, rel)
        b = normalize(b)
    scores = parsed.get("bucket_scores")
    if isinstance(scores, dict):
        try:
            marginal = normalize(np.array([float(scores.get(x, 0.0)) for x in BUCKETS]))
            b = normalize(0.5 * b + 0.5 * marginal)
        except Exception:
            pass
    return b, anchors


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    accuracy = sum(row["pred"] == row["true"] for row in rows) / n
    recalls = []
    f1s = []
    for bucket in BUCKETS:
        tp = sum(row["pred"] == bucket and row["true"] == bucket for row in rows)
        pred_pos = sum(row["pred"] == bucket for row in rows)
        true_pos = sum(row["true"] == bucket for row in rows)
        recalls.append(tp / true_pos if true_pos else 0.0)
        f1s.append(2 * tp / max(1, pred_pos + true_pos))
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
    best = min(float(np.mean([ans_loss(a, BUCKETS.index(row["true"])) for row in rows])) for a in range(3))
    best_name = "answer"
    for subset in ALL_SETS:
        value = float(np.mean([set_loss(subset, BUCKETS.index(row["true"])) for row in rows]))
        if value < best:
            best, best_name = value, f"set:{subset}"
    if LOSS["abstain"] < best:
        best, best_name = LOSS["abstain"], "abstain"
    return {
        "n": n,
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "mean_loss": mean_loss,
        "oracle_fixed_loss": best,
        "oracle_fixed_action": best_name,
        "regret": mean_loss - best,
        "credible_coverage": float(np.mean([row["true"] in row["credible"] for row in rows])),
        "mean_credible_size": float(np.mean([len(row["credible"]) for row in rows])),
        "abstain_rate": float(np.mean([row["action"]["mode"] == "abstain" for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO_ROOT / "experiments/unified/data/dataset_v4.json"))
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--extra-body", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = json.loads(Path(args.data).read_text(encoding="utf-8"))
    base_prior, sector_priors, counts = prior_from_history(records)
    test = [rec for rec in records if rec.get("split") == args.split]
    if args.limit:
        test = test[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_claim_path = out / "per_claim.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if per_claim_path.exists():
        for line in per_claim_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[str(row.get("claim_id"))] = row
    todo = [rec for rec in test if str(rec.get("claim_id")) not in existing]
    print(f"existing={len(existing)} todo={len(todo)} workers={args.workers}", flush=True)

    local = threading.local()

    def client() -> LLMClient:
        if not hasattr(local, "client"):
            local.client = LLMClient(
                provider=args.provider,
                model=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                cache_dir=out / "llm_cache",
                timeout=180,
                max_retries=4,
                reasoning_effort="low",
                extra_body=json.loads(args.extra_body) if args.extra_body else None,
            )
        return local.client

    def work(rec: dict[str, Any]) -> dict[str, Any]:
        sector_prior = sector_priors.get(str(rec.get("sector", "other")))
        prior = base_prior if sector_prior is None else normalize(0.5 * base_prior + 0.5 * sector_prior)
        prompt = anchor_prompt(rec, base_prior, sector_prior)
        raw = client().text_of(client().complete(
            prompt,
            system="Return only the JSON object. Do not add prose.",
            max_tokens=900,
            temperature=0.0,
            thinking=False,
            use_cache=True,
        ))
        parsed = parse_json(raw)
        belief, anchors = belief_from_anchors(prior, parsed)
        action = gate(belief)
        return {
            "claim_id": rec.get("claim_id"),
            "true": rec.get("outcome", {}).get("bucket"),
            "pred": BUCKETS[int(np.argmax(belief))],
            "belief": [float(x) for x in belief],
            "credible": [BUCKETS[i] for i in credible_set(belief)],
            "action": action,
            "anchors": anchors,
            "llm": parsed,
            "raw_text": raw[:1200],
            "sector": rec.get("sector"),
            "cutoff_year": rec.get("cutoff_year"),
        }

    rows = list(existing.values())
    written = len(existing)
    with per_claim_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, rec): rec for rec in todo}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            if written % 10 == 0:
                print(f"{written}/{len(test)}", flush=True)
    summary = metrics([row for row in rows if row["true"] in BUCKETS])
    summary["prior"] = dict(zip(BUCKETS, [float(x) for x in base_prior]))
    summary["prior_counts"] = counts
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
