# Audited Real-Data Provenance

The real-data values in the drafts are a compact audit of read-only artifacts
under `D:/Engineering/FinPlanRAG/database/preexperiments/results`. The source
corpora and model outputs are intentionally not copied into this repository.
The machine-readable record is `paper/results/real_data_summary.json`.

The evidence is separated into four tracks: public LOFin (1,572 questions),
public FinGLM annual reports (1,829 questions over 11,587 documents), SEC
point-in-time lineage replay (17 exploratory and 12 holdout lineages), and a
Chinese mixed-period mechanism pilot (5 cases, 3 frozen). All reported values
remain file-level closure, document correctness, leakage, or field-oracle
diagnostics unless explicitly labelled otherwise. None is an answer-accuracy
leaderboard result.

The external directory is read-only under the storage contract. To audit a
value, compute SHA-256 for the named artifact and compare it with the record in
`real_data_summary.json`; the original protocol and legacy research notes are
also retained outside this repository.
