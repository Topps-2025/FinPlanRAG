# Reproducibility checklist

- Use Python 3.10 or newer and install with `pip install -e ".[dev]"`.
- Run `python -m pytest -q` before changing the planner or parser.
- Run `python examples/controlled_smoke.py` after changes to retrieval or stopping logic.
- Keep question cutoff, document availability, and the four-part obligation key in every benchmark record.
- Freeze benchmark inputs and protocol hashes before interpreting comparisons.
- Report document closure, answer correctness, overclaiming, future leakage, and query cost as separate metrics.
- Keep large data and model artifacts outside Git as described in `DATA_STORAGE.md`.
