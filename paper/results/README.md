# Paper evidence

The summary artifact was generated from the repository's deterministic
controlled benchmark:

```text
PYTHONPATH=src python -m finplan_rag.controlled_planning \
  --seeds 20 --cases-per-world 40 --budget 10 \
  --out controlled_results.json
```

Source SHA-256 (the local full JSON run):
`E5D207A28136A91CA74B85D0B373C71E5C5D7346AD497E16799B6DB4E42CAA28`.

The benchmark is a controlled mechanism diagnostic. It has synthetic hidden
worlds and should not be read as a financial filing QA benchmark or an SOTA
comparison.

Budget sensitivity is in `budget_sensitivity.json` (budgets 1, 2, 4, 6, 10,
16 with the same seeds and scenarios). It shows the quality-cost frontier and
prevents the main budget from being treated as a universally optimal setting.
