# Reproducibility checklist

- Use Python 3.10 or newer and install with `pip install -e ".[dev]"`.
- Run `python -m pytest -q` before changing the planner or parser.
- Run `python examples/controlled_smoke.py` after changes to retrieval or stopping logic.
- Keep question cutoff, document availability, and the four-part obligation key in every benchmark record.
- Freeze benchmark inputs and protocol hashes before interpreting comparisons.
- Report document closure, answer correctness, overclaiming, future leakage, and query cost as separate metrics.
- Keep large data and model artifacts outside Git as described in `DATA_STORAGE.md`.

To rebuild the paper's budget-sensitivity evidence with one command, run
`python scripts/run_paper_contract.py --seeds 20 --cases-per-world 40` and
inspect the generated `budget_summary.json`. The script records a SHA-256 for
each full JSON run; compact values are committed in `paper/results/`.
