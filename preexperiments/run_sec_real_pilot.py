"""Run a transparent retrieval-planning diagnostic on the SEC real pilot.

This is not a claim of end-to-end LLM SOTA. It uses one shared lexical
retriever and a deterministic evidence resolver so that differences are due to
query ordering, stopping, temporal filtering, and explicit obligations. The
pilot is deliberately small and reports lineage-level rather than pretending
to be a statistically powered benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from .storage_paths import DATA_ROOT, RESULTS_ROOT
except ImportError:
    from storage_paths import DATA_ROOT, RESULTS_ROOT


TOKEN_RE = re.compile(r"[a-z0-9]+")
METHODS = (
    "single_shot",
    "fixed_decomposition",
    "generic_adaptive",
    "generic_path_bound",
    "hirec_style",
    "hirec_path_bound",
    "finplan_v1",
    "finplan_v2",
    "finplan_v3_path_bound",
    "finplan_no_time",
    "finplan_no_limit",
)
SLOT_QUERIES = {
    "entity": "{company} {target}",
    "agreement": "{company} {target} definitive agreement merger acquire",
    "status": "{company} {target} completed acquisition transaction merger closed",
    "limit": "{company} {target} terminated termination withdrawn abandoned merger agreement",
}
V3_SLOT_QUERIES = {
    "entity": "{company} {target}",
    "agreement": "{company} {target} item 1.01 entry material definitive agreement entered into agreement plan merger",
    "status": "{company} {target} item 2.01 completion acquisition on completed transaction merger",
    "limit": "{company} {target} item 1.02 termination material definitive agreement mutually agreed terminate abandoned",
}


def parse_time(value: str) -> datetime:
    value = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    lineage_id: str
    role: str
    available_at: datetime
    text: str
    terms: Tuple[str, ...]


class BM25:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.df = Counter()
        self.tf: List[Counter[str]] = []
        for chunk in self.chunks:
            counts = Counter(chunk.terms)
            self.tf.append(counts)
            self.df.update(counts.keys())
        self.avgdl = sum(len(c.terms) for c in self.chunks) / max(1, len(self.chunks))

    def search(
        self,
        query: str,
        cutoff: datetime,
        used: Sequence[str],
        allow_future: bool,
        limit: int = 1,
        required_terms: Sequence[str] = (),
    ) -> List[Chunk]:
        qterms = tokens(query)
        used_set = set(used)
        n = len(self.chunks)
        scored: List[Tuple[float, Chunk]] = []
        for chunk, counts in zip(self.chunks, self.tf):
            if chunk.chunk_id in used_set:
                continue
            if not allow_future and chunk.available_at > cutoff:
                continue
            if required_terms and not all(term in counts for term in required_terms):
                continue
            dl = len(chunk.terms)
            score = 0.0
            for term in qterms:
                f = counts.get(term, 0)
                if not f:
                    continue
                idf = math.log(1.0 + (n - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5))
                score += idf * (f * 2.2) / (f + 1.2 * (0.25 + 0.75 * dl / max(1.0, self.avgdl)))
            if score:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].available_at, item[1].chunk_id))
        # Document-level de-duplication prevents one long filing from consuming
        # the whole budget and approximates passage retrieval in HiREC.
        out: List[Chunk] = []
        docs: set[str] = set()
        for _, chunk in scored:
            if chunk.doc_id in docs:
                continue
            out.append(chunk)
            docs.add(chunk.doc_id)
            if len(out) >= limit:
                break
        return out


def make_chunks(documents: Sequence[Mapping[str, object]], window: int = 160, stride: int = 120) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in documents:
        words = str(doc["text"]).split()
        if not words:
            continue
        for start in range(0, len(words), stride):
            part = " ".join(words[start : start + window])
            if not part:
                break
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['doc_id']}::c{start}",
                    doc_id=str(doc["doc_id"]),
                    lineage_id=str(doc["lineage_id"]),
                    role=str(doc["role"]),
                    available_at=parse_time(str(doc["available_at"])),
                    text=part,
                    terms=tuple(tokens(part)),
                )
            )
            if start + window >= len(words):
                break
    return chunks


def target_present(text: str, target: str) -> bool:
    low = text.lower()
    significant = [term for term in tokens(target) if len(term) > 2]
    return target.lower() in low or (bool(significant) and all(term in low for term in significant))


def evidence_flags(chunk: Chunk, company: str, target: str) -> Dict[str, bool]:
    low = chunk.text.lower()
    target_ok = target_present(low, target)
    company_ok = company.lower() in low
    dated_event = r"on\s+[a-z]+\s+\d{1,2},\s+\d{4}"
    strong_completed = target_ok and bool(
        re.search(
            rf"(?:item\s*2\.01[^.]{{0,180}}completion[^.]{{0,220}}{dated_event}[^.]{{0,220}}completed|"
            rf"{dated_event}[^.]{{0,260}}completed[^.]{{0,180}}(?:acquisition|transaction|merger))",
            low,
        )
    )
    strong_terminated = target_ok and bool(
        re.search(
            rf"(?:termination of a material definitive agreement[^.]{{0,420}}{dated_event}[^.]{{0,260}}"
            rf"(?:agreed to terminate|terminated)|{dated_event}[^.]{{0,300}}(?:agreed to terminate|terminated)|"
            r"mutual termination[^.]{0,180}(?:approved|effective))",
            low,
        )
    )
    pending = target_ok and bool(
        re.search(
            r"(?:entered into|announc\w*|issued[^.]{0,100}press release)[^.]{0,260}"
            r"(?:definitive agreement|agreement and plan of merger|will acquire|acquire)",
            low,
        )
    )
    # Explicit limiting cues are a separate obligation from status.
    limiting = target_ok and bool(re.search(r"subject to|condition|regulatory approval|termination|terminated|withdrawn|abandon", low))
    return {
        "entity": target_ok and company_ok,
        "agreement": pending,
        "completed": strong_completed,
        "terminated": strong_terminated,
        "pending": pending,
        "limiting": limiting,
    }


def infer_state(observed: Sequence[Tuple[Chunk, Dict[str, bool]]]) -> str:
    if any(flags["terminated"] for _, flags in observed):
        return "none"
    if any(flags["completed"] for _, flags in observed):
        return "indirect"
    if any(flags["pending"] for _, flags in observed):
        return "conditional"
    return "unknown"


def query_for(case: Mapping[str, object], slot: str, method: str) -> str:
    templates = V3_SLOT_QUERIES if method == "finplan_v3_path_bound" else SLOT_QUERIES
    return templates[slot].format(company=case["company"], target=case["target"])


def run_method(case: Mapping[str, object], method: str, index: BM25, budget: int) -> Dict[str, object]:
    cutoff = parse_time(str(case["cutoff"]))
    allow_future = method == "finplan_no_time"
    used: List[str] = []
    observed: List[Tuple[Chunk, Dict[str, bool]]] = []
    actions: List[str] = []
    searched_limit = False
    path_bound = method in {"generic_path_bound", "hirec_path_bound", "finplan_v3_path_bound"}
    required_terms = [term for term in tokens(str(case["target"])) if len(term) > 2] if path_bound else ()

    def retrieve(slot: str, k: int = 1) -> None:
        nonlocal searched_limit
        if slot == "limit":
            searched_limit = True
        actions.append(slot)
        query = query_for(case, slot, method)
        for chunk in index.search(query, cutoff, used, allow_future, limit=k, required_terms=required_terms):
            used.append(chunk.chunk_id)
            observed.append((chunk, evidence_flags(chunk, str(case["company"]), str(case["target"]))))

    if method == "single_shot":
        retrieve("status", budget)
    elif method == "fixed_decomposition":
        for slot in ("entity", "agreement", "status", "limit"):
            if len(actions) >= budget:
                break
            retrieve(slot)
    elif method in {"generic_adaptive", "generic_path_bound", "hirec_style", "hirec_path_bound"}:
        is_hirec = method in {"hirec_style", "hirec_path_bound"}
        retrieve("status", 2 if is_hirec else 1)
        while len(actions) < budget:
            state = infer_state(observed)
            if state != "unknown" and is_hirec:
                break
            if state == "conditional":
                retrieve("status")
            elif state == "indirect":
                retrieve("limit")
            elif state == "none":
                break
            else:
                retrieve("agreement")
            if not is_hirec and infer_state(observed) != "unknown":
                break
    else:
        order = ("status", "limit", "agreement", "entity")
        if method in {"finplan_v2", "finplan_v3_path_bound"}:
            order = ("status", "agreement", "limit", "entity")
        if method == "finplan_no_limit":
            order = tuple(slot for slot in order if slot != "limit")
        for slot in order:
            if len(actions) >= budget:
                break
            retrieve(slot, 2 if method in {"finplan_v2", "finplan_v3_path_bound"} and slot == "status" else 1)
            state = infer_state(observed)
            if state == "none":
                break
            if state == "indirect" and (searched_limit or method == "finplan_no_limit"):
                break
            if state == "conditional" and searched_limit:
                break

    state = infer_state(observed)
    if state == "indirect" and not searched_limit and method not in {
        "single_shot", "generic_adaptive", "generic_path_bound",
        "hirec_style", "hirec_path_bound", "finplan_no_limit",
    }:
        # The proposed policy treats an unsearched limiting branch as an open
        # obligation even when the status cue looks complete.
        state = "unknown"
    if state == "conditional" and method.startswith("finplan") and not searched_limit and method != "finplan_no_limit":
        state = "unknown"

    return {"state": state, "used": used, "actions": actions, "searched_limit": searched_limit}


def score(case: Mapping[str, object], prediction: Mapping[str, object], chunks_by_id: Mapping[str, Chunk]) -> Dict[str, object]:
    truth = str(case["truth_state"])
    state = str(prediction["state"])
    used_chunks = [chunks_by_id[x] for x in prediction["used"]]
    correct = state == truth
    overclaim = state == "indirect" and truth in {"conditional", "none"}
    future = any(chunk.available_at > parse_time(str(case["cutoff"])) for chunk in used_chunks)
    gold = set(case["gold_doc_ids"])
    limiting_gold = set(case.get("limiting_doc_ids", []))
    used_docs = {chunk.doc_id for chunk in used_chunks}
    closure = bool(gold & used_docs) and state != "unknown"
    limiting_recall = None if not limiting_gold else float(bool(limiting_gold & used_docs))
    limit_action_recall = None if not limiting_gold else float(bool(limiting_gold & used_docs) and bool(prediction["searched_limit"]))
    wrong_lineage = float(any(chunk.lineage_id != case["lineage_id"] for chunk in used_chunks))
    return {
        "accuracy": float(correct),
        "answered": float(state != "unknown"),
        "overclaim": float(overclaim),
        "closure": float(closure),
        "limiting_recall": limiting_recall,
        "limit_action_recall": limit_action_recall,
        "future_leak": float(future),
        "wrong_lineage": wrong_lineage,
        "queries": float(len(prediction["actions"])),
        "state": state,
    }


def aggregate(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    names = ("accuracy", "answered", "overclaim", "closure", "limiting_recall", "limit_action_recall", "future_leak", "wrong_lineage", "queries")
    out: Dict[str, float] = {}
    for name in names:
        values = [float(row[name]) for row in rows if row[name] is not None]
        out[name] = sum(values) / max(1, len(values))
    return out


def run(data_path: Path, budget: int) -> Mapping[str, object]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    chunks = make_chunks(data["documents"])
    index = BM25(chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rows: List[Dict[str, object]] = []
    trajectories: List[Dict[str, object]] = []
    for case in data["cases"]:
        for method in METHODS:
            prediction = run_method(case, method, index, budget)
            metric = score(case, prediction, chunks_by_id)
            row = {"case_id": case["case_id"], "lineage_id": case["lineage_id"], "slice": case["slice"], "method": method, **metric}
            rows.append(row)
            trajectories.append({"case_id": case["case_id"], "method": method, **prediction})
    summary = {method: aggregate([row for row in rows if row["method"] == method]) for method in METHODS}
    return {
        "schema": "finplan-sec-real-pilot-results.v1",
        "status": "small-real-data-pilot-not-sota",
        "protocol": {"data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": len(data["cases"]), "lineages": len({c["lineage_id"] for c in data["cases"]}), "budget": budget, "chunk_window": 160, "chunk_stride": 120},
        "summary": summary,
        "rows": rows,
        "trajectories": trajectories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_ROOT / "sec_real_pilot_v1.json")
    parser.add_argument("--out", default=RESULTS_ROOT / "sec_real_pilot_v1.json")
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    result = run(Path(args.data), args.budget)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
