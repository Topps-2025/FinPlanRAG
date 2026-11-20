"""Temporal AnchorEvolve: reliability/prior evolution + full-action loss-aware readout.

The state is updated only when a claim's outcome has been publicly revealed
(``earnings_events.json`` date, falling back to the 10-Q date).  For a claim at
cutoff c, the belief uses anchors and outcomes with reveal date < c.

Usage:
    python experiments/evolution/temporal_fusion.py --config experiments/evolution/config.yaml
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[2]
for extra in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

LABELS = ("below", "within", "above")
SUBSETS = [tuple(R) for r in range(2, 4) for R in itertools.combinations(range(3), r)]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_date(value) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def canonical_type(anchor: dict, config: dict) -> str:
    anchor_type = str(anchor.get("type") or "other")
    aliases = {
        "guidance": "guidance_width",
        "current_quarter": "current_quarter_growth",
        "company": "company_history",
        "sector": "sector_history",
        "text": "text_signal",
    }
    anchor_type = aliases.get(anchor_type, anchor_type)
    return anchor_type if anchor_type in config["anchors"]["types"] else "other"


def qualify(anchor: dict, row: dict, config: dict) -> str | None:
    """Return the qualification kind, or None for a rejected anchor.

    Implements the five paper categories deterministically:
    vacuous (no implied bucket), stale (future/unknown source), adversarial
    (scope mismatch/source outside the excerpt), hard, soft.
    """
    implied = anchor.get("implied_bucket")
    if implied not in LABELS:
        return None  # vacuous
    anchor_date = parse_date(anchor.get("date"))
    if anchor_date is not None and anchor_date > parse_date(row["cutoff"]):
        return None  # stale
    source = str(anchor.get("source_id") or "")
    if source.startswith("P"):
        allowed = set(row.get("excerpt_ids") or [])
        if allowed and source not in allowed:
            return None  # adversarial: cites a paragraph outside the excerpt
    elif source.startswith("E"):
        if source[1:].isdigit() and int(source[1:]) > int(row.get("evidence_count") or 0):
            return None  # adversarial: cites a missing structured evidence item
    scope = str(anchor.get("scope") or "")
    if scope and scope not in {"company", "sector", "market"}:
        return None  # adversarial: scope does not match the claim contract
    anchor_type = canonical_type(anchor, config)
    reliability = float(anchor.get("reliability", 0.5) or 0.5)
    if bool(anchor.get("hard")) and reliability >= float(config["anchors"]["hard_reliability"]):
        return "hard"
    return "soft"


def answer_loss(prediction: int, truth: int, loss: dict[str, float]) -> float:
    if prediction == truth:
        return 0.0
    if prediction == 2 and truth == 0:
        return float(loss["over"])
    if prediction == 0 and truth == 2:
        return float(loss["under"])
    return float(loss["wrong"])


def set_loss(chosen: tuple[int, ...], truth: int, loss: dict[str, float]) -> float:
    base = 0.0 if truth in chosen else float(loss["miss"])
    return base + float(loss["width"]) * max(0, len(chosen) - 1)


def decide(belief: np.ndarray, truth: int, loss: dict[str, float], *, full_action: bool) -> tuple[float, str, tuple[int, ...], int]:
    belief = np.asarray(belief, dtype=float)
    belief = belief / max(belief.sum(), 1e-12)
    point = int(np.argmax(belief))
    expected_answer = [sum(belief[j] * answer_loss(k, j, loss) for j in range(3)) for k in range(3)]
    if full_action:
        best_answer = min(range(3), key=lambda k: expected_answer[k])
        expected_sets = [sum(belief[j] * set_loss(subset, j, loss) for j in range(3)) for subset in SUBSETS]
        best_index = int(np.argmin(expected_sets))
        action_set = SUBSETS[best_index]
        values = {
            "answer": expected_answer[best_answer],
            "adjudicate": expected_sets[best_index],
            "abstain": float(loss["abstain"]),
        }
        action = min(values, key=values.get)
        if action == "answer":
            return answer_loss(best_answer, truth, loss), action, action_set, best_answer
        if action == "adjudicate":
            return set_loss(action_set, truth, loss), action, action_set, best_answer
        return float(loss["abstain"]), action, action_set, best_answer
    # original restricted gate: highest-density set with mass >= 1-alpha
    alpha = float(loss.get("alpha", 0.1))
    ordered = np.argsort(-belief)
    credible: list[int] = []
    mass = 0.0
    for index in ordered:
        credible.append(int(index))
        mass += belief[int(index)]
        if mass >= 1.0 - alpha - 1e-12:
            break
    action_set = tuple(sorted(credible))
    expected_set = sum(belief[j] * set_loss(action_set, j, loss) for j in range(3))
    values = {
        "answer": expected_answer[point],
        "adjudicate": expected_set,
        "abstain": float(loss["abstain"]),
    }
    action = min(values, key=values.get)
    if action == "answer":
        return answer_loss(point, truth, loss), action, action_set, point
    if action == "adjudicate":
        return set_loss(action_set, truth, loss), action, action_set, point
    return float(loss["abstain"]), action, action_set, point


def macro_f1(preds: list[int], truths: list[int]) -> float:
    scores = []
    for label in range(3):
        tp = sum(p == label and t == label for p, t in zip(preds, truths))
        fp = sum(p == label and t != label for p, t in zip(preds, truths))
        fn = sum(p != label and t == label for p, t in zip(preds, truths))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def balanced(preds: list[int], truths: list[int]) -> float:
    recalls = []
    for label in range(3):
        n = sum(t == label for t in truths)
        recalls.append(sum(p == label and t == label for p, t in zip(preds, truths)) / n if n else 0.0)
    return float(np.mean(recalls))


def metrics(beliefs: np.ndarray, truths: list[int], loss: dict[str, float], *, full_action: bool) -> dict:
    preds = beliefs.argmax(axis=1).tolist()
    rows = [decide(beliefs[i], truths[i], loss, full_action=full_action) for i in range(len(truths))]
    return {
        "n": len(truths),
        "accuracy": float(np.mean([p == t for p, t in zip(preds, truths)])),
        "balanced_accuracy": balanced(preds, truths),
        "macro_f1": macro_f1(preds, truths),
        "decision_loss": float(np.mean([row[0] for row in rows])),
        "mean_action_set_size": float(np.mean([len(row[2]) for row in rows])),
        "actions": {action: sum(row[1] == action for row in rows) for action in ("answer", "adjudicate", "abstain")},
    }


def boot_diff(a: np.ndarray, b: np.ndarray, *, resamples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(a) - np.asarray(b)
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(resamples)])
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar(a: list[int], b: list[int], truths: list[int]) -> tuple[int, int, float]:
    wins = sum(x == y and z != y for x, z, y in zip(a, b, truths))
    losses = sum(x != y and z == y for x, z, y in zip(a, b, truths))
    if wins + losses == 0:
        return wins, losses, 1.0
    return wins, losses, float(binomtest(min(wins, losses), wins + losses, 0.5).pvalue * 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/evolution/config.yaml")
    parser.add_argument("--anchor-source", default=None, choices=["old", "new"])
    parser.add_argument("--no-evolution", action="store_true")
    parser.add_argument("--rf-validation", default="")
    parser.add_argument("--rf-test", default="")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    anchor_cfg = config["anchors"]
    evolution_cfg = config["evolution"]
    loss = dict(anchor_cfg["loss"])
    loss["alpha"] = 1.0 - float(anchor_cfg["alpha"])

    data = {row["claim_id"]: row for row in json.loads((REPO_ROOT / config["data"]).read_text(encoding="utf-8"))}
    events = json.loads((REPO_ROOT / config["earnings_events"]).read_text(encoding="utf-8"))
    source_kind = args.anchor_source or str(config.get("anchor_source", "old"))
    evolution_enabled = bool(config["evolution"].get("enabled", True)) and not args.no_evolution
    suffix = source_kind + ("_static" if not evolution_enabled else "")
    if args.tag:
        suffix += f"_{args.tag}"
    sources = {
        split: REPO_ROOT / config["sources"][f"{split}_{source_kind}"]
        for split in ("validation", "test")
    }
    rf_sources = {
        "validation": REPO_ROOT / (args.rf_validation or config["sources"]["rf_validation"]),
        "test": REPO_ROOT / (args.rf_test or config["sources"]["rf_test"]),
    }

    def load_anchor_rows(split: str) -> list[dict]:
        rows = []
        for row in load_jsonl(sources[split]):
            record = data.get(row["claim_id"])
            if not record:
                continue
            cutoff = record["cutoff"]
            reveal = events.get(row["claim_id"], {}).get("date") or (record.get("outcome") or {}).get("date")
            truth_value = row.get("true") or row.get("truth")
            if truth_value not in LABELS:
                continue
            rows.append({
                "claim_id": row["claim_id"],
                "split": split,
                "truth": LABELS.index(truth_value),
                "cutoff": cutoff,
                "reveal_date": max(cutoff, reveal or cutoff),
                "anchors": row.get("anchors") or (row.get("parsed") or {}).get("anchors") or [],
                "bucket_scores": row.get("bucket_scores") or (row.get("llm") or {}).get("bucket_scores") or {},
                "excerpt_ids": row.get("excerpt_ids") or [],
                "evidence_count": len(record.get("evidence") or []),
                "old_belief": row.get("belief"),
            })
        return rows

    validation = load_anchor_rows("validation")
    test = load_anchor_rows("test")
    stream = sorted(validation + test, key=lambda row: (row["cutoff"], row["claim_id"]))

    type_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    outcome_counts = np.zeros(3, dtype=float)
    pending: list[tuple[str, str, dict]] = []
    beliefs: dict[str, np.ndarray] = {}

    def reveal(row: dict) -> None:
        outcome_counts[row["truth"]] += 1.0
        for anchor in row["anchors"]:
            if qualify(anchor, row, config) not in {"hard", "soft"}:
                continue
            implied = anchor.get("implied_bucket")
            anchor_type = canonical_type(anchor, config)
            type_stats[anchor_type]["n"] += 1
            type_stats[anchor_type]["correct"] += int(LABELS.index(implied) == row["truth"])

    for row in stream:
        if evolution_enabled:
            while pending and pending[0][0] < row["cutoff"]:
                _, _, past = heapq.heappop(pending)
                reveal(past)
            total = outcome_counts.sum()
            prior = (outcome_counts + 1.0) / (total + 3.0) if total else np.full(3, 1.0 / 3.0)
        else:
            prior = np.full(3, 1.0 / 3.0)
        belief = prior.copy()
        for anchor in row["anchors"]:
            kind = qualify(anchor, row, config)
            if kind not in {"hard", "soft"}:
                continue
            implied = anchor.get("implied_bucket")
            anchor_type = canonical_type(anchor, config)
            llm_rel = float(anchor.get("reliability", 0.5) or 0.5)
            if evolution_enabled:
                stats = type_stats.get(anchor_type, {"n": 0, "correct": 0})
                beta_rel = (stats["correct"] + float(evolution_cfg["beta_alpha"])) / (
                    stats["n"] + float(evolution_cfg["beta_alpha"]) + float(evolution_cfg["beta_beta"])
                )
                rho = (
                    float(evolution_cfg["shrinkage"]) * beta_rel
                    + (1.0 - float(evolution_cfg["shrinkage"])) * llm_rel
                )
            else:
                rho = llm_rel
            hard = kind == "hard"
            index = LABELS.index(implied)
            if hard:
                belief = np.array([1.0 if i == index else 1e-3 for i in range(3)])
            else:
                likelihood = np.full(3, (1.0 - float(anchor_cfg["soft_p_e"])) / 2.0)
                likelihood[index] = float(anchor_cfg["soft_p_e"])
                belief = belief * np.power(likelihood, rho)
            belief = belief / max(belief.sum(), 1e-12)
        scores = row["bucket_scores"]
        if all(label in scores for label in LABELS):
            score_vector = np.array([float(scores[label]) for label in LABELS])
            score_vector = score_vector / max(score_vector.sum(), 1e-12)
            belief = float(anchor_cfg["soft_mix"]) * belief + (1.0 - float(anchor_cfg["soft_mix"])) * score_vector
            belief = belief / max(belief.sum(), 1e-12)
        beliefs[row["claim_id"]] = belief
        if evolution_enabled:
            heapq.heappush(pending, (row["reveal_date"], row["claim_id"], row))

    def load_rf(split: str) -> dict[str, np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for row in load_jsonl(rf_sources[split]):
            grouped[row["claim_id"]].append(np.asarray(row["probabilities"], dtype=float))
        return {claim_id: np.mean(values, axis=0) for claim_id, values in grouped.items()}

    rf_validation = load_rf("validation")
    rf_test = load_rf("test")
    verified_test = {
        claim_id for claim_id, record in data.items()
        if record.get("split") == "test" and record.get("verification") in {"xbrl_search", "xbrl_verified", "xbrl_override"}
    }

    def evaluate_split(rows: list[dict], rf: dict[str, np.ndarray], weight: float) -> tuple[dict, np.ndarray, list[int]]:
        selected = [row for row in rows if row["claim_id"] in rf and (row["split"] != "test" or row["claim_id"] in verified_test)]
        fused = []
        truths = []
        for row in selected:
            belief = beliefs[row["claim_id"]]
            probability = weight * belief + (1.0 - weight) * rf[row["claim_id"]]
            fused.append(probability / max(probability.sum(), 1e-12))
            truths.append(row["truth"])
        return metrics(np.vstack(fused), truths, loss, full_action=True), np.vstack(fused), truths

    weights = [float(value) for value in anchor_cfg["fusion_weights"]]
    validation_grid = {}
    for weight in weights:
        summary, _, _ = evaluate_split(validation, rf_validation, weight)
        validation_grid[weight] = summary
    selected_weight = min(weights, key=lambda weight: validation_grid[weight]["decision_loss"])
    test_summary, test_probs, test_truths = evaluate_split(test, rf_test, selected_weight)
    test_llm_only, llm_probs, _ = evaluate_split(test, rf_test, 1.0)

    # Persist the exact fused test posterior so calibration and ranking metrics
    # can be recomputed without reconstructing the anchor state.
    selected_test = [
        row
        for row in test
        if row["claim_id"] in rf_test and row["claim_id"] in verified_test
    ]
    prob_path = REPO_ROOT / config["results_dir"] / f"temporal_test_probs_{suffix}.jsonl"
    prob_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "claim_id": row["claim_id"],
                    "truth": row["truth"],
                    "probabilities": [float(value) for value in test_probs[index]],
                    "llm_probabilities": [float(value) for value in llm_probs[index]],
                    "rf_probabilities": [float(value) for value in rf_test[row["claim_id"]]],
                    "fusion_weight": selected_weight,
                },
                ensure_ascii=False,
            )
            for index, row in enumerate(selected_test)
        )
        + "\n",
        encoding="utf-8",
    )

    # Reference points: RF-only and the old fusion w=0.1 (from the released ensemble file).
    rf_only, _, _ = evaluate_split(test, rf_test, 0.0)
    old_ensemble = {
        row["claim_id"]: np.asarray(row["belief"], dtype=float)
        for row in load_jsonl(REPO_ROOT / config["sources"]["old_ensemble_test"])
    }
    old_rows = [row for row in test if row["claim_id"] in old_ensemble and row["claim_id"] in verified_test]
    old_probs = np.vstack([old_ensemble[row["claim_id"]] for row in old_rows])
    old_truths = [row["truth"] for row in old_rows]
    old_summary = metrics(old_probs, old_truths, loss, full_action=True)
    old_current = metrics(old_probs, old_truths, loss, full_action=False)
    old_current_losses = np.array([
        decide(old_probs[i], old_truths[i], loss, full_action=False)[0]
        for i in range(len(old_truths))
    ])

    # Paired loss improvement over RF-only and over the old fusion.
    test_losses = np.array([decide(test_probs[i], test_truths[i], loss, full_action=True)[0] for i in range(len(test_truths))])
    rf_only_probs = {row["claim_id"]: rf_test[row["claim_id"]] for row in test if row["claim_id"] in rf_test}
    rf_rows = [row for row in test if row["claim_id"] in rf_only_probs and row["claim_id"] in verified_test]
    rf_losses = np.array([decide(rf_test[row["claim_id"]], row["truth"], loss, full_action=True)[0] for row in rf_rows])
    rf_current_losses = np.array([
        decide(rf_test[row["claim_id"]], row["truth"], loss, full_action=False)[0]
        for row in rf_rows
    ])
    old_losses = np.array([decide(old_probs[i], old_truths[i], loss, full_action=True)[0] for i in range(len(old_truths))])
    loss_vs_rf = boot_diff(test_losses, rf_losses, resamples=10000, seed=int(config["evaluation"]["seed"]))
    loss_vs_old = boot_diff(test_losses, old_losses, resamples=10000, seed=int(config["evaluation"]["seed"]))
    loss_vs_old_restricted = boot_diff(test_losses, old_current_losses, resamples=10000, seed=int(config["evaluation"]["seed"]))
    loss_vs_rf_restricted = boot_diff(test_losses, rf_current_losses, resamples=10000, seed=int(config["evaluation"]["seed"]))

    # Persist per-claim evolved beliefs and compact anchor features so the
    # validation-trained meta-readout can be audited without rerunning the sweep.
    belief_rows = []
    for row in validation + test:
        belief = beliefs[row["claim_id"]]
        probs = belief / max(belief.sum(), 1e-12)
        entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
        ordered = np.sort(probs)[::-1]
        anchor_reliabilities = [float(anchor.get("reliability", 0.5) or 0.5) for anchor in row["anchors"]]
        belief_rows.append({
            "claim_id": row["claim_id"],
            "split": row["split"],
            "truth": LABELS[row["truth"]],
            "cutoff": row["cutoff"],
            "belief": [float(value) for value in probs],
            "bucket_scores": row["bucket_scores"],
            "n_anchors": len(row["anchors"]),
            "n_hard": sum(bool(anchor.get("hard")) for anchor in row["anchors"]),
            "mean_reliability": float(np.mean(anchor_reliabilities)) if anchor_reliabilities else 0.5,
            "max_reliability": float(np.max(anchor_reliabilities)) if anchor_reliabilities else 0.5,
            "entropy": entropy,
            "margin": float(ordered[0] - ordered[1]),
        })
    belief_path = REPO_ROOT / config["results_dir"] / f"temporal_beliefs_{suffix}.jsonl"
    belief_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in belief_rows) + "\n",
        encoding="utf-8",
    )

    output = {
        "evolution_enabled": evolution_enabled,
        "selected_weight": selected_weight,
        "validation_grid": {str(weight): item for weight, item in validation_grid.items()},
        "test_full_action": test_summary,
        "test_llm_only": test_llm_only,
        "test_rf_only": rf_only,
        "test_old_fusion_full_action": old_summary,
        "test_old_fusion_restricted_gate": old_current,
        "loss_diff_vs_rf_only": {"mean": loss_vs_rf[0], "lo": loss_vs_rf[1], "hi": loss_vs_rf[2]},
        "loss_diff_vs_old_fusion": {"mean": loss_vs_old[0], "lo": loss_vs_old[1], "hi": loss_vs_old[2]},
        "loss_diff_vs_old_fusion_restricted": {"mean": loss_vs_old_restricted[0], "lo": loss_vs_old_restricted[1], "hi": loss_vs_old_restricted[2]},
        "loss_diff_vs_rf_restricted": {"mean": loss_vs_rf_restricted[0], "lo": loss_vs_rf_restricted[1], "hi": loss_vs_rf_restricted[2]},
        "n_validation": len([row for row in validation if row["claim_id"] in rf_validation]),
        "n_test": len(test_truths),
    }
    out = REPO_ROOT / config["results_dir"] / f"temporal_fusion_{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
