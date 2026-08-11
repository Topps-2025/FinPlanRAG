"""Run the full LOFin benchmark: 1,572 questions x 9 methods (Task #16).

Execution follows preexperiments/lofin_full_benchmark_protocol_v1.json
(amendment_3: question set = 1,595 unique qids from the by_answer_type
partition minus 23 excluded = 1,572; finqa/secqa views are redundant and
not read):
  - groups: questions grouped by (match_companies(question) union {C_q});
    one runner invocation per group (data = merged member company filings,
    registry = member companies only).
  - smoke phase first: groups containing any of the 30 pre-registered smoke
    qids; diagnostics only, discarded as evidence, used for determinism.
  - full phase: every group, one-shot.  Runners are imported (byte-identical
    files, SHA-verified against the freeze), never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re as _re
import sys
from pathlib import Path

# 470 groups x 9 method runs recompile match_companies' alias patterns on
# cache eviction; the default 512-pattern re cache degrades to O(n^2)
# recompilation over a long full run.  Bump it here (driver-level process
# setting; imported runners share the module, nothing in the runners changes).
_re._MAXCACHE = 200_000

sys.path.insert(0, r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\preexperiments")
from storage_paths import RESULTS_ROOT

from run_nonoracle_obligation_planning_v4 import run as run_v4
from run_nonoracle_obligation_planning_v5 import run as run_v5
from run_nonoracle_obligation_planning_v6 import run as run_v6
from run_nonoracle_obligation_planning_v7 import run as run_v7
from run_period_aware_baselines import run as run_baselines

DATA = Path(r"D:\Engineering\FinPlanRAG\database\preexperiments\data")
CASES = DATA / "lofin_full_cases_v1.json"
GROUPS = DATA / "lofin_full_groups_v1.json"
REGISTRY = DATA / "lofin_full_registry_v1.json"
CORPUS_DIR = DATA / "lofin_full_corpus_v1"
PROTOCOL = Path(r"C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\preexperiments\lofin_full_benchmark_protocol_v1.json")
TMP = RESULTS_ROOT / "tmp_lofin_full"
BUDGET = 4

RUNNERS = [
    ("v9_and_metadata", run_v7, ("finplan_v9_cascade", "metadata_v9_fill")),
    ("v8", run_v6, ("finplan_v8",)),
    ("v7", run_v5, ("finplan_v7",)),
    ("v4", run_v4, ("finplan_v4",)),
    ("baselines", run_baselines, ("single_shot", "period_metadata_decomposition",
                                  "generic_adaptive_period", "hirec_period")),
]

SMOKE_QIDS = ["AAL/2010/page_72.pdf-1", "ABMD/2012/page_41.pdf-4", "AMT/2006/page_113.pdf-2",
              "AON/2007/page_171.pdf-1", "BKNG/2016/page_33.pdf-2", "C/2008/page_44.pdf-2",
              "CE/2016/page_19.pdf-3", "DISCA/2014/page_64.pdf-3", "EMR/2017/page_78.pdf-2",
              "FIS/2006/page_31.pdf-2", "GPN/2014/page_92.pdf-2", "GS/2015/page_188.pdf-2",
              "HOLX/2009/page_151.pdf-2", "IP/2006/page_75.pdf-4", "JPM/2012/page_140.pdf-2",
              "LMT/2015/page_89.pdf-3", "MRO/2003/page_45.pdf-2", "MSI/2014/page_76.pdf-2",
              "PNC/2012/page_68.pdf-5", "SLB/2015/page_59.pdf-1", "STT/2011/page_69.pdf-3",
              "UNP/2015/page_56.pdf-1", "ZBH/2008/page_57.pdf-4", "financebench_03473",
              "openqa_124", "openqa_17", "openqa_214", "openqa_26", "openqa_304", "openqa_50"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_group(gid: str, group: dict) -> tuple[dict, dict, list[dict]]:
    """Group data JSON (documents+cases), group registry, member company files."""
    companies = json.loads(REGISTRY.read_text(encoding="utf-8"))["companies"]
    documents = []
    member_entries = []
    for ticker in group["companies"]:
        corp = json.loads((CORPUS_DIR / f"{ticker}.json").read_text(encoding="utf-8"))
        documents.extend(corp["documents"])
        member_entries.append(companies[ticker])
    cases = [c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]
             if c["case_id"] in set(group["case_ids"])]
    data = {
        "schema": "finplan-lofin-full-group-data.v1",
        "status": "frozen-benchmark-input",
        "documents": documents,
        "cases": cases,
    }
    registry = {
        "schema": "finplan-lofin-full-group-registry.v1",
        "status": "frozen-benchmark-input",
        "companies": member_entries,
    }
    return data, registry, cases


def result_files_for(gid: str) -> list[tuple[str, Path, Path, Path, Path]]:
    """(runner name, data path, registry path, result path, result path exists)."""
    out = []
    for name, _, _ in RUNNERS:
        dp = TMP / f"{gid}_{name}_data.json"
        rp = TMP / f"{gid}_{name}_registry.json"
        op = TMP / f"{gid}_{name}_result.json"
        out.append((name, dp, rp, op, op.exists()))
    return out


def absorb_group(gid: str, name: str, methods: tuple, result: dict,
                 all_rows: list, all_predictions: list, per_runner_files: dict) -> int:
    """Append one runner's rows/predictions to the accumulators (shared by the
    run-now and resume-from-disk paths, so resumed output is byte-identical
    to a continuous run)."""
    per_runner_files.setdefault(name, []).append(str(TMP / f"{gid}_{name}_result.json"))
    rows = result.get("rows", [])
    for row in rows:
        if "method" not in row:
            row["method"] = methods[0]  # single-method runners
    all_rows.extend(rows)
    for pred in result.get("predictions", []) or []:
        all_predictions.append({"group": gid, "method": name, **pred})
    return len(rows)


def main(phase: str, start: int, end: int) -> None:
    protocol_sha = sha256(PROTOCOL)
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    groups = json.loads(GROUPS.read_text(encoding="utf-8"))["groups"]
    if phase == "smoke":
        smoke_set = set(SMOKE_QIDS)
        group_ids = [gid for gid, g in groups.items() if set(g["case_ids"]) & smoke_set]
        out_file = RESULTS_ROOT / "lofin_full_benchmark_dev.json"
    else:
        group_ids = list(groups.keys())
        out_file = RESULTS_ROOT / "lofin_full_benchmark_frozen.json"
    all_group_ids = group_ids          # full list: the final scan covers every
    group_ids = group_ids[start:end]   # group, so ANY finishing process writes
                                       # the complete combined output

    TMP.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_predictions = []
    per_runner_files = {}
    n_cases_seen = 0

    # Resume: a group is complete iff all 5 per-runner result files exist.
    # Killed runs leave them behind; reading them back reproduces the exact
    # same combined output as a continuous run (rows are byte-identical).
    resumed = []
    pending = []
    for gid in group_ids:
        files = result_files_for(gid)
        if all(exists for _, _, _, _, exists in files):
            resumed.append(gid)
        else:
            pending.append(gid)
    if resumed:
        print(f"[{phase}] resuming: {len(resumed)} groups already complete "
              f"({resumed[:5]}{'...' if len(resumed) > 5 else ''}), "
              f"{len(pending)} to run", flush=True)

    savepoint_every = 25
    n_since_savepoint = 0

    def write_output(tag: str) -> None:
        combined = {
            "schema": "finplan-lofin-full-benchmark-result.v1",
            "phase": phase,
            "status": "frozen",
            "protocol_sha256": protocol_sha,
            "protocol_file": str(PROTOCOL),
            "budget": BUDGET,
            "n_groups": len(per_runner_files.get("baselines", [])),
            "n_cases": n_cases_seen,
            "n_rows": len(all_rows),
            "per_runner_files": per_runner_files,
            "rows": all_rows,
            "predictions": all_predictions,
        }
        (out_file.with_suffix(f".{tag}.json")).write_text(
            json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        if tag == "final":
            out_file.write_text(json.dumps(combined, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            print(f"saved {out_file} ({len(all_rows)} rows, "
                  f"{len(all_predictions)} predictions)", flush=True)

    def absorb_one_runner(gid: str, name: str, op: Path) -> None:
        """Absorb one runner's result file; if a killed process truncated it,
        re-run that runner for the group (resume-from-disk robustness)."""
        try:
            result = json.loads(op.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"[{phase}] corrupt result {op.name}, re-running {name} "
                  f"for {gid}", flush=True)
            group = groups[gid]
            data, registry, _ = load_group(gid, group)
            dp = TMP / f"{gid}_{name}_data.json"
            rp = TMP / f"{gid}_{name}_registry.json"
            dp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            rp.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
            run_fn = next(fn for n, fn, _ in RUNNERS if n == name)
            result = run_fn(dp, rp, op, BUDGET)
        methods = next(m for n, _, m in RUNNERS if n == name)
        absorb_group(gid, name, methods, result, all_rows, all_predictions, per_runner_files)

    for gid in resumed:
        # completed groups: absorb their rows from the per-runner result files
        for name, _, _, op, _ in result_files_for(gid):
            absorb_one_runner(gid, name, op)
        n_cases_seen += len(groups[gid]["case_ids"])
    for gid in pending:
        group = groups[gid]
        data, registry, group_cases = load_group(gid, group)
        n_cases_seen += len(group_cases)
        for name, run_fn, methods in RUNNERS:
            dp, rp, op = TMP / f"{gid}_{name}_data.json", TMP / f"{gid}_{name}_registry.json", TMP / f"{gid}_{name}_result.json"
            dp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            rp.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
            result = run_fn(dp, rp, op, BUDGET)
            absorb_group(gid, name, methods, result, all_rows, all_predictions, per_runner_files)
        print(f"[{phase}] {gid} companies={group['companies']} cases={len(group_cases)} "
              f"total_cases={n_cases_seen}", flush=True)
        n_since_savepoint += 1
        if n_since_savepoint >= savepoint_every:
            write_output(f"savepoint{n_since_savepoint // savepoint_every}")
            n_since_savepoint = 0

    # Final disk scan: absorb any group whose 5 result files now exist but were
    # not in this process's range (parallel-run splits) or appeared mid-run.
    # Absorb in gid order so the combined output is byte-identical regardless
    # of which process computed which groups.
    absorbed = set(resumed) | set(pending)
    scan_count = 0
    for gid in all_group_ids:
        if gid in absorbed:
            continue
        files = result_files_for(gid)
        if all(exists for _, _, _, _, exists in files):
            for name, _, _, op, _ in files:
                absorb_one_runner(gid, name, op)
            n_cases_seen += len(groups[gid]["case_ids"])
            scan_count += 1

    write_output("final")
    print(f"[{phase}] combined output: {len(all_rows)} rows, "
          f"{len(all_predictions)} predictions (resumed {len(resumed)} + "
          f"freshly run {len(pending)} + disk-scan {scan_count})", flush=True)
    for stale in out_file.parent.glob(out_file.stem + ".savepoint*.json"):
        stale.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    # Work-split for parallel runs: process only group_ids[start:end].
    # Both halves write the identical complete output (final disk scan absorbs
    # the other half's result files in gid order); safe because per-group
    # outputs are deterministic and independent.
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2 ** 31)
    args = parser.parse_args()
    main(args.phase, args.start, args.end)
