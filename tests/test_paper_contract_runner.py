from pathlib import Path

from scripts.run_paper_contract import run_budget


def test_paper_contract_runner_uses_the_controlled_entrypoint(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    row = run_budget(root, tmp_path / "budget_1.json", seeds=1, cases=2, budget=1)
    assert row["budget"] == 1
    assert len(row["source_sha256"]) == 64
    assert 0.0 <= row["future_leak"] <= 1.0
