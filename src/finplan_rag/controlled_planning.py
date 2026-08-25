"""Controlled diagnostic for relation-constrained evidence-state planning.

This experiment isolates *planning policy* rather than giving the proposed
method a richer schema or evidence pool. Every method receives the same
candidate paths, relation templates, documents, cutoff, resolver, and budget.
The hidden world is only available to the environment and evaluator.

The diagnostic is intentionally not evidence of real-world financial value.
Its role is to reject planning components that cannot beat simpler policies
under controlled missingness, conflict, multi-path, and future-revision stress.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from .storage import RESULTS_ROOT
except ImportError:  # pragma: no cover - direct script compatibility
    RESULTS_ROOT = Path("results")
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


RELATION_SLOTS: Mapping[str, Tuple[str, ...]] = {
    "product": ("entity", "concept", "status", "counterevidence"),
    "acquisition": ("entity", "asset", "status", "conditions", "counterevidence"),
    "license": ("entity", "asset", "scope", "status", "counterevidence"),
    "supply": ("entity", "asset", "delivery", "status", "counterevidence"),
    "regulated_project": ("entity", "asset", "approval", "status", "counterevidence"),
}
RELATIONS = tuple(RELATION_SLOTS)
STATES = ("direct", "indirect", "conditional", "none")
METHODS = (
    "single_shot",
    "fixed_decomposition",
    "generic_adaptive",
    "finplan_state",
    "finplan_no_counterevidence",
    "finplan_no_stop",
    "finplan_no_time",
)
SCENARIOS = (
    "balanced",
    "multi_path",
    "counterevidence_dense",
    "sparse",
    "source_conflict",
    "future_revision",
    "random_world",
)
SOURCE_WEIGHT = {"official": 1.00, "filing": 0.90, "company": 0.68, "media": 0.48}


@dataclass(frozen=True)
class PathTruth:
    path_id: str
    relation: str
    state: str
    slot_values: Mapping[str, str]


@dataclass(frozen=True)
class Document:
    doc_id: int
    path_id: str
    relation: str
    slot: str
    value: str
    source: str
    available_at: int
    relevance: float
    is_correct: bool
    is_limiting: bool


@dataclass(frozen=True)
class Case:
    case_id: str
    scenario: str
    cutoff: int
    paths: Tuple[Tuple[str, str], ...]  # visible path_id and relation only
    documents: Tuple[Document, ...]
    hidden_paths: Tuple[PathTruth, ...]
    truth_state: str


@dataclass(frozen=True)
class Prediction:
    state: str
    used_doc_ids: Tuple[int, ...]
    consumed_doc_ids: Tuple[int, ...]
    queries: int
    closed: bool


def stable_seed(name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{name}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def scenario_parameters(name: str, rng: np.random.Generator) -> Dict[str, float]:
    params = {"missing": 0.10, "conflict": 0.10, "future": 0.08, "multi": 0.35}
    if name == "multi_path":
        params.update(multi=0.92, conflict=0.15)
    elif name == "counterevidence_dense":
        params.update(multi=0.55, conflict=0.18)
    elif name == "sparse":
        params.update(missing=0.42, conflict=0.06)
    elif name == "source_conflict":
        params.update(conflict=0.48, missing=0.05)
    elif name == "future_revision":
        params.update(future=0.72, conflict=0.18)
    elif name == "random_world":
        params.update(
            missing=float(rng.uniform(0.03, 0.48)),
            conflict=float(rng.uniform(0.02, 0.50)),
            future=float(rng.uniform(0.00, 0.70)),
            multi=float(rng.uniform(0.15, 0.95)),
        )
    return params


def sample_path(path_id: str, relation: str, rng: np.random.Generator) -> PathTruth:
    allowed = ("direct", "conditional", "none") if relation == "product" else (
        "indirect",
        "conditional",
        "none",
    )
    state = str(rng.choice(allowed, p=(0.38, 0.34, 0.28)))
    slots = {slot: "satisfied" for slot in RELATION_SLOTS[relation]}
    slots["status"] = {
        "direct": "realized",
        "indirect": "realized",
        "conditional": "pending",
        "none": "terminated",
    }[state]
    slots["counterevidence"] = "limiting" if state == "none" else "clear"
    if state == "none" and rng.random() < 0.35:
        semantic_slot = "concept" if relation == "product" else "asset"
        slots[semantic_slot] = "not_satisfied"
    return PathTruth(path_id, relation, state, slots)


def aggregate_truth(paths: Sequence[PathTruth]) -> str:
    states = {p.state for p in paths}
    for state in ("direct", "indirect", "conditional"):
        if state in states:
            return state
    return "none"


def alternative(value: str, slot: str, rng: np.random.Generator) -> str:
    if slot == "status":
        choices = ("realized", "pending", "terminated")
    elif slot == "counterevidence":
        choices = ("clear", "limiting")
    else:
        choices = ("satisfied", "not_satisfied")
    return str(rng.choice([x for x in choices if x != value]))


def make_case(scenario: str, seed: int, index: int) -> Case:
    rng = np.random.default_rng(stable_seed(f"{scenario}:{index}", seed))
    params = scenario_parameters(scenario, rng)
    cutoff = 10
    n_paths = 1 + int(rng.random() < params["multi"]) + int(rng.random() < params["multi"] * 0.22)
    hidden = tuple(sample_path(f"p{j}", str(rng.choice(RELATIONS)), rng) for j in range(n_paths))

    # Counterevidence stress is created by prevalence, not a hand-written case
    # that only one named method can solve.
    if scenario == "counterevidence_dense":
        adjusted: List[PathTruth] = []
        for path in hidden:
            if rng.random() < 0.55:
                slots = dict(path.slot_values)
                slots["status"] = "terminated"
                slots["counterevidence"] = "limiting"
                adjusted.append(PathTruth(path.path_id, path.relation, "none", slots))
            else:
                adjusted.append(path)
        hidden = tuple(adjusted)

    documents: List[Document] = []
    doc_id = 0
    for path in hidden:
        for slot in RELATION_SLOTS[path.relation]:
            true_value = path.slot_values[slot]
            if rng.random() < params["missing"]:
                continue
            copies = int(rng.integers(1, 4))
            for _ in range(copies):
                source = str(rng.choice(tuple(SOURCE_WEIGHT), p=(0.24, 0.28, 0.28, 0.20)))
                p_correct = min(0.96, max(0.52, SOURCE_WEIGHT[source] - 0.06))
                correct = bool(rng.random() < p_correct)
                value = true_value if correct else alternative(true_value, slot, rng)
                relevance = float(np.clip(rng.normal(0.78 if correct else 0.70, 0.13), 0.05, 0.99))
                documents.append(
                    Document(
                        doc_id,
                        path.path_id,
                        path.relation,
                        slot,
                        value,
                        source,
                        int(rng.integers(1, cutoff + 1)),
                        relevance,
                        correct,
                        slot == "counterevidence" and value == "limiting",
                    )
                )
                doc_id += 1
            if rng.random() < params["conflict"]:
                wrong = alternative(true_value, slot, rng)
                documents.append(
                    Document(
                        doc_id,
                        path.path_id,
                        path.relation,
                        slot,
                        wrong,
                        str(rng.choice(("company", "media"))),
                        int(rng.integers(1, cutoff + 1)),
                        float(np.clip(rng.normal(0.84, 0.08), 0.05, 0.99)),
                        False,
                        slot == "counterevidence" and wrong == "limiting",
                    )
                )
                doc_id += 1
            if rng.random() < params["future"]:
                future_value = alternative(true_value, slot, rng)
                documents.append(
                    Document(
                        doc_id,
                        path.path_id,
                        path.relation,
                        slot,
                        future_value,
                        "official",
                        cutoff + int(rng.integers(1, 4)),
                        0.97,
                        False,
                        slot == "counterevidence" and future_value == "limiting",
                    )
                )
                doc_id += 1

    # High-similarity distractors test whether planning can target obligations
    # rather than repeatedly consume semantically attractive mentions.
    for j in range(int(rng.integers(1, 4))):
        documents.append(
            Document(doc_id, f"noise{j}", "mention", "mention", "satisfied", "media",
                     int(rng.integers(1, cutoff + 1)), float(rng.uniform(0.82, 0.99)), False, False)
        )
        doc_id += 1
    return Case(
        f"{scenario}:{seed}:{index}",
        scenario,
        cutoff,
        tuple((p.path_id, p.relation) for p in hidden),
        tuple(documents),
        hidden,
        aggregate_truth(hidden),
    )


def slot_vote(documents: Sequence[Document]) -> Tuple[str | None, float]:
    if not documents:
        return None, 0.0
    scores: Dict[str, float] = defaultdict(float)
    for doc in documents:
        scores[doc.value] += SOURCE_WEIGHT.get(doc.source, 0.35)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    total = sum(v for _, v in ranked)
    confidence = ranked[0][1] / total if total else 0.0
    return ranked[0][0], float(confidence)


def resolve_path(
    path_id: str,
    relation: str,
    observed: Sequence[Document],
    ignore_counterevidence: bool = False,
) -> Tuple[str, bool]:
    slots = tuple(
        slot for slot in RELATION_SLOTS[relation]
        if not (ignore_counterevidence and slot == "counterevidence")
    )
    values: Dict[str, str] = {}
    for slot in slots:
        value, confidence = slot_vote([d for d in observed if d.path_id == path_id and d.slot == slot])
        if value is None or confidence < 0.56:
            return "unknown", False
        values[slot] = value
    if values.get("counterevidence") == "limiting" or values["status"] == "terminated":
        return "none", True
    semantic_slot = "concept" if relation == "product" else "asset"
    if values.get(semantic_slot) == "not_satisfied":
        return "none", True
    if values["status"] == "pending":
        return "conditional", True
    if values["status"] != "realized":
        return "unknown", False
    return ("direct" if relation == "product" else "indirect"), True


def aggregate_prediction(
    case: Case,
    observed: Sequence[Document],
    ignore_counterevidence: bool = False,
) -> Tuple[str, bool]:
    resolved = [
        resolve_path(pid, relation, observed, ignore_counterevidence)
        for pid, relation in case.paths
    ]
    for state in ("direct", "indirect", "conditional"):
        if any(pred == state and closed for pred, closed in resolved):
            return state, True
    if resolved and all(pred == "none" and closed for pred, closed in resolved):
        return "none", True
    return "unknown", False


def next_document(case: Case, target: Tuple[str, str] | None, used: set[int]) -> Document | None:
    candidates = [d for d in case.documents if d.doc_id not in used]
    if target is not None:
        path_id, slot = target
        candidates = [d for d in candidates if d.path_id == path_id and d.slot == slot]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d.relevance, d.available_at, -d.doc_id))


def slot_confidence(observed: Sequence[Document], path_id: str, slot: str) -> float:
    return slot_vote([d for d in observed if d.path_id == path_id and d.slot == slot])[1]


def finplan_target(
    case: Case,
    observed: Sequence[Document],
    attempts: Mapping[Tuple[str, str], int],
    include_counter: bool,
) -> Tuple[str, str] | None:
    candidates: List[Tuple[float, str, str]] = []
    for path_id, relation in case.paths:
        for slot in RELATION_SLOTS[relation]:
            if slot == "counterevidence" and not include_counter:
                continue
            confidence = slot_confidence(observed, path_id, slot)
            n = attempts.get((path_id, slot), 0)
            # Relation slots are shared with every schema-aware baseline. The
            # policy contribution is risk/cost-sensitive ordering and stopping.
            risk = 1.35 if slot in {"status", "counterevidence"} else 1.00
            if slot in {"conditions", "approval", "scope", "delivery"}:
                risk = 1.18
            priority = risk * (1.0 - confidence) / (1.0 + 0.55 * n)
            candidates.append((priority, path_id, slot))
    if not candidates:
        return None
    _, path_id, slot = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    return path_id, slot


def infer(case: Case, method: str, budget: int = 10) -> Prediction:
    used: set[int] = set()
    observed: List[Document] = []
    attempts: Dict[Tuple[str, str], int] = defaultdict(int)
    queries = 0

    fixed_targets = [
        (pid, slot)
        for pid, relation in case.paths
        for slot in RELATION_SLOTS[relation]
    ]

    while queries < budget:
        ignore_counter = method == "finplan_no_counterevidence"
        state, closed = aggregate_prediction(case, observed, ignore_counter)
        # Every FinPlan variant except the explicit no-stop ablation keeps the
        # same adaptive stopping rule. This prevents the no-time ablation from
        # conflating temporal filtering with a forced full-budget rollout.
        if method in {"finplan_state", "finplan_no_time"} and closed:
            break
        if method == "finplan_no_counterevidence" and closed:
            break

        if method == "single_shot":
            target = None
        elif method == "fixed_decomposition":
            target = fixed_targets[queries % len(fixed_targets)] if fixed_targets else None
        elif method == "generic_adaptive":
            generic = [
                (slot_confidence(observed, pid, slot), attempts[(pid, slot)], pid, slot)
                for pid, relation in case.paths
                for slot in RELATION_SLOTS[relation]
            ]
            _, _, pid, slot = min(generic, default=(0.0, 0, "", ""))
            target = (pid, slot) if pid else None
        else:
            target = finplan_target(
                case,
                observed,
                attempts,
                include_counter=method != "finplan_no_counterevidence",
            )

        doc = next_document(case, target, used)
        queries += 1
        if target is not None:
            attempts[target] += 1
        if doc is None:
            continue
        used.add(doc.doc_id)
        if method == "finplan_no_time" or doc.available_at <= case.cutoff:
            observed.append(doc)

    ignore_counter = method == "finplan_no_counterevidence"
    state, closed = aggregate_prediction(case, observed, ignore_counter)
    return Prediction(
        state,
        tuple(sorted(used)),
        tuple(sorted(d.doc_id for d in observed)),
        queries,
        closed,
    )


def score(case: Case, prediction: Prediction) -> Dict[str, float]:
    answered = prediction.state != "unknown"
    correct = answered and prediction.state == case.truth_state
    realized = prediction.state in {"direct", "indirect"}
    overclaim = realized and case.truth_state in {"conditional", "none"}
    future_leak = any(
        d.doc_id in prediction.consumed_doc_ids and d.available_at > case.cutoff
        for d in case.documents
    )
    limiting = {d.doc_id for d in case.documents if d.is_limiting and d.available_at <= case.cutoff}
    retrieved_limiting = limiting.intersection(prediction.used_doc_ids)
    counter_recall = len(retrieved_limiting) / len(limiting) if limiting else 1.0
    utility = (1.0 if correct else -0.15 if not answered else -1.0) - 0.03 * prediction.queries
    if overclaim:
        utility -= 1.0
    return {
        "accuracy": float(correct),
        "coverage": float(answered),
        "selective_correct": float(correct) if answered else math.nan,
        "overclaim": float(overclaim),
        "false_none": float(prediction.state == "none" and case.truth_state != "none"),
        "closure": float(prediction.closed),
        "counterevidence_recall": float(counter_recall),
        "future_leak": float(future_leak),
        "queries": float(prediction.queries),
        "utility": float(utility),
    }


def evaluate_world(scenario: str, seed: int, cases: int, budget: int) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    worlds = [make_case(scenario, seed, i) for i in range(cases)]
    for method in METHODS:
        metrics = [score(case, infer(case, method, budget)) for case in worlds]
        row: Dict[str, object] = {"scenario": scenario, "seed": seed, "method": method}
        for name in metrics[0]:
            values = np.asarray([m[name] for m in metrics], dtype=float)
            row[name] = float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
        records.append(row)
    return records


def summarize(records: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        metrics: Dict[str, float] = {}
        for name in (
            "accuracy", "coverage", "selective_correct", "overclaim", "false_none",
            "closure", "counterevidence_recall", "future_leak", "queries", "utility",
        ):
            values = np.asarray([float(r[name]) for r in rows], dtype=float)
            finite = values[np.isfinite(values)]
            metrics[name] = float(finite.mean()) if len(finite) else math.nan
            metrics[f"{name}_ci95"] = float(1.96 * finite.std(ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else 0.0
        out[method] = metrics
    return out


def paired_delta(records: Sequence[Mapping[str, object]], with_method: str, without: str) -> Dict[str, float]:
    lookup = {(r["scenario"], r["seed"], r["method"]): r for r in records}
    deltas: Dict[str, float] = {}
    pairs = sorted({(str(r["scenario"]), int(r["seed"])) for r in records})
    for metric in ("accuracy", "overclaim", "closure", "counterevidence_recall", "queries", "utility"):
        values = np.asarray([
            float(lookup[(s, seed, with_method)][metric]) - float(lookup[(s, seed, without)][metric])
            for s, seed in pairs
        ])
        deltas[metric] = float(values.mean())
        deltas[f"{metric}_ci95"] = float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))
    return deltas


def run(seeds: int, cases_per_world: int, budget: int) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    for scenario in SCENARIOS:
        for seed in range(seeds):
            records.extend(evaluate_world(scenario, seed, cases_per_world, budget))
    protocol = {
        "states": list(STATES),
        "relations": list(RELATIONS),
        "methods": list(METHODS),
        "scenarios": list(SCENARIOS),
        "seeds": seeds,
        "cases_per_world": cases_per_world,
        "budget": budget,
        "shared_information": "paths, relation templates, documents, cutoff, resolver, budget",
        "utility": "correct=1; abstain=-0.15; other_error=-1; overclaim_extra=-1; query=-0.03",
    }
    return {
        "schema": "finplan-controlled-planning.v1",
        "status": "controlled-mechanism-only",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "non_oracle_source_check": {
            "infer_mentions_hidden_paths": "hidden_paths" in __import__("inspect").getsource(infer),
            "infer_mentions_truth_state": "truth_state" in __import__("inspect").getsource(infer),
            "finplan_target_mentions_hidden_paths": "hidden_paths" in __import__("inspect").getsource(finplan_target),
        },
        "summary": summarize(records),
        "module_deltas": {
            "state_planning_vs_single_shot": paired_delta(records, "finplan_state", "single_shot"),
            "state_planning_vs_fixed": paired_delta(records, "finplan_state", "fixed_decomposition"),
            "state_planning_vs_generic": paired_delta(records, "finplan_state", "generic_adaptive"),
            "counterevidence": paired_delta(records, "finplan_state", "finplan_no_counterevidence"),
            "adaptive_stop": paired_delta(records, "finplan_state", "finplan_no_stop"),
            "point_in_time": paired_delta(records, "finplan_state", "finplan_no_time"),
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--cases-per-world", type=int, default=80)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--out", default=RESULTS_ROOT / "finplan_controlled_planning_v1.json")
    args = parser.parse_args()
    result = run(args.seeds, args.cases_per_world, args.budget)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "module_deltas": result["module_deltas"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
