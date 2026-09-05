# Reproducibility and challenge checklist

## Completed in this draft

- [x] Fixed seeds, scenario list, case count and query budget are recorded.
- [x] All methods share paths, documents, cutoff, resolver and utility contract.
- [x] Non-oracle source inspection tests reject hidden-label access.
- [x] Point-in-time behavior, parser behavior, answer support parsing and agent
  JSONL behavior have deterministic tests.
- [x] Full controlled run and six budget runs have source SHA-256 records.
- [x] English and Chinese sources compile with XeLaTeX and are UTF-8 clean.
- [x] Claims, evidence types and non-claims are listed in
  `claim_evidence_matrix.md`.
- [x] Audited real-data artifacts have read-only paths, SHA-256 hashes, and
  interpretation boundaries in `results/real_data_summary.json`.
- [x] The drafts separate file-level closure, document correctness, field-oracle
  diagnostics, and answer accuracy.

## Required before a top-journal empirical claim

- [ ] Freeze a licensed financial filing corpus with publication timestamps.
- [ ] Add a shared reader and official answer scorer; report answer correctness
  separately from closure.
- [ ] Reproduce strong sparse, dense and hybrid retrieval baselines under the
  same cutoff, reader, budget and storage contract.
- [ ] Add held-out entities and filing periods, confidence intervals at the
  declared independent unit, and preregistered multiple-comparison handling.
- [ ] Audit model/API versions and external data manifests before release.

The open items are deliberate scope boundaries, not missing prose. The current
paper is a rigorous mechanism draft, not a claim of completed top-journal
external validity.
