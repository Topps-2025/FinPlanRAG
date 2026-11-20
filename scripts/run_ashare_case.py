"""Run one historical A-share claim at its cutoff and save the full audit."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from anchor_engine import CaseSpec, ClaimContract, load_config, run_claim_safe  # noqa: E402


def load_claims(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")).get("claims") or []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="paper/data/ashare_backtest_large/claims.json")
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--config", default="paper/results/cloud/configs/deepseek_anchor_cli.yaml")
    parser.add_argument("--out", default="paper/results/ashare_case")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    claim = next((row for row in load_claims(Path(args.data)) if row["claim_id"] == args.claim_id), None)
    if claim is None:
        raise SystemExit(f"claim not found: {args.claim_id}")
    config = load_config(args.config)
    contract = ClaimContract.from_dict(claim)
    spec = CaseSpec(
        case_id=claim["claim_id"],
        question=claim["metric"],
        company=claim["company"],
        ticker=claim["ticker"],
        target_period=claim["target_period"],
        claims=[contract],
        cutoff=claim["cutoff"],
        source="ashare_case",
    )
    evidence = []
    for item in claim.get("evidence") or []:
        row = dict(item)
        row.setdefault("source_id", row.get("id"))
        row.setdefault("prompt_text", row.get("text"))
        evidence.append(row)
    out_dir = Path(args.out) / (args.tag or claim["claim_id"])
    result = run_claim_safe(
        contract,
        spec,
        config,
        date.fromisoformat(claim["cutoff"]),
        out_dir=out_dir,
        evidence_override=evidence,
    )
    payload = {
        "claim": claim,
        "result": result,
        "outcome": claim.get("outcome"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    readout = result.get("readout") or {}
    print(json.dumps({
        "claim_id": claim["claim_id"],
        "cutoff": claim["cutoff"],
        "prediction": readout.get("point") or readout.get("set"),
        "action": readout.get("action"),
        "belief": (result.get("belief") or {}).get("final"),
        "outcome": claim.get("outcome"),
        "anchors": [
            {
                "type": anchor.get("type"),
                "qualification": anchor.get("qualification"),
                "reliability": anchor.get("reliability"),
                "implied_label": anchor.get("implied_label"),
            }
            for anchor in (result.get("anchors") or [])
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
