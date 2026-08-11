"""Inventory the full LOFin public benchmark for the frozen protocol (Task #14).

Pure data-description pass over the 5 original question files (no method
tuning): SHA-256 of every frozen input, question-phrasing coverage for the
planner's period grammar, qid-year vs evidence-year relation, single- vs
multi-doc counts per subset, and the 8K/EARNINGS exclusion face.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

DATA = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\data")
LOFIN = DATA / "external" / "lofin"
SUBSETS = ["finqa_test.jsonl", "numeric_table_test.jsonl", "numeric_text_test.jsonl",
           "secqa_test.jsonl", "textual_test.jsonl"]

PATTERNS = {
    "fiscal_year_N": re.compile(r"fiscal year\s+(?:of\s+)?(20\d\d)", re.I),
    "FY_N": re.compile(r"\bfy\s+((?:20)?\d\d)\b", re.I),
    "first_half": re.compile(r"first\s+half", re.I),
    "second_half": re.compile(r"second\s+half", re.I),
    "first_n_quarters": re.compile(r"first\s+(?:two|three|four)\s+quarters", re.I),
    "explicit_Qn": re.compile(r"\bq[1-4]\b", re.I),
    "quarter_N": re.compile(r"(?:quarter|quarters?)\s+(?:of\s+)?(?:the\s+)?(?:fiscal\s+)?year\s+(?:of\s+)?(20\d\d)", re.I),
    "plain_year": re.compile(r"(?:^|\D)(20\d\d)(?:\D|$)"),
    "relative": re.compile(r"prior\s+year|previous\s+year|last\s+year|year\s+over\s+year|compared\s+to\s+the\s+prior", re.I),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    rows: list[dict] = []
    for name in SUBSETS:
        for line in (LOFIN / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append((name, json.loads(line)))

    total = len(rows)
    pattern_hits: Counter[str, int] = Counter()
    per_subset: Counter[str, int] = Counter()
    multi_doc: Counter[str, int] = Counter()
    excluded: list[dict] = []
    year_mismatch: Counter[str, int] = Counter()
    ev_year_range = Counter()
    n_evidences = Counter()
    no_year_questions = []

    for name, row in rows:
        q = str(row["question"])
        subset = name.replace("_test.jsonl", "")
        per_subset[subset] += 1
        for key, pat in PATTERNS.items():
            if pat.search(q):
                pattern_hits[key] += 1
        if not re.search(r"20\d\d", q):
            no_year_questions.append(row["qid"])

        qid = str(row["qid"])
        m = re.match(r"([A-Z0-9.\-]+)/(20\d\d)/", qid)
        qy = int(m.group(2)) if m else None

        evs = row.get("evidences", [])
        n_evidences[len(evs)] += 1
        if len(evs) > 1:
            multi_doc[subset] += 1
        years = set()
        is_8k_earn = False
        for ev in evs:
            doc = str(ev["doc_name"])
            for token in doc.split("_"):
                mm = re.fullmatch(r"(20\d\d)(?:Q[1-4])?", token)
                if mm:
                    years.add(int(mm.group(1)))
            if doc.endswith("_8K") or doc.endswith("_EARNINGS"):
                is_8k_earn = True
        if qy is not None and years and years != {qy}:
            year_mismatch[subset] += 1
        ev_year_range[len(sorted(years))] += 1
        if is_8k_earn:
            excluded.append({"subset": subset, "qid": qid, "evidences": [str(e["doc_name"]) for e in evs]})

    out = {
        "schema": "finplan-lofin-full-inventory.v1",
        "generated_at": "2026-08-11",
        "n_questions_total": total,
        "n_questions_excluded_8k_earnings": len(excluded),
        "n_questions_main": total - len(excluded),
        "excluded_by_subset": Counter(e["subset"] for e in excluded),
        "per_subset": dict(per_subset),
        "multi_doc_by_subset": dict(multi_doc),
        "n_evidences_distribution": dict(n_evidences),
        "pattern_coverage": dict(pattern_hits),
        "qid_year_vs_evidence_year_mismatch_by_subset": dict(year_mismatch),
        "evidence_year_range_width": dict(ev_year_range),
        "n_questions_no_year": len(no_year_questions),
        "excluded": excluded,
        "sha256": {
            "finqa_test.jsonl": sha256(LOFIN / "finqa_test.jsonl"),
            "numeric_table_test.jsonl": sha256(LOFIN / "numeric_table_test.jsonl"),
            "numeric_text_test.jsonl": sha256(LOFIN / "numeric_text_test.jsonl"),
            "secqa_test.jsonl": sha256(LOFIN / "secqa_test.jsonl"),
            "textual_test.jsonl": sha256(LOFIN / "textual_test.jsonl"),
        },
    }
    out_path = DATA / "lofin_full_inventory_v1.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("excluded", "sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
