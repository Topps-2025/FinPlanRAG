import json
import subprocess
import sys
from pathlib import Path


def test_gap_proof_exhaustive_counterexample(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "gap.json"
    subprocess.run(
        [sys.executable, "scripts/run_gap_proof.py", "--out", str(output)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = json.loads(output.read_text(encoding="utf-8"))["rows"]
    two_obligation = next(row for row in rows if row["obligations"] == 2 and row["distractors_per_obligation"] == 1 and not row["future_records"])
    assert two_obligation["score_only_closure"] == 1 / 6
    assert two_obligation["obligation_state_closure"] == 1.0
    future = next(row for row in rows if row["future_records"])
    assert future["score_only_future_leak"] > 0
    assert future["obligation_state_future_leak"] == 0.0
