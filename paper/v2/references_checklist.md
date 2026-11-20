# References checklist for v2 related work

**Status:** working checklist, 2026-10-31. Entries marked `arXiv-verified` were
checked against the arXiv API on 2026-10-25 in the AnchorEvolve research note.
Venue/DOI must be re-checked against the primary record before submission. Do
not claim a venue for a preprint.

## Answer-centric RAG / adaptive RAG

| Short name | Record | Status |
|---|---|---|
| Lewis et al., RAG | arXiv:2005.11401 | classic; verify venue |
| FLARE | Jiang et al. 2023 | verify venue |
| Self-RAG | Asai et al. 2024 | verify venue |
| DRAGIN | Su et al. 2024 | verify venue |
| Adaptive-RAG | Jeong et al. 2024 | verify venue |
| ReAct | Yao et al. 2023 | verify venue |
| Reflexion | Shinn et al. 2023 | verify venue |

## Structured financial RAG

| Short name | Record | Status |
|---|---|---|
| FinRank | Mansouri et al. 2026 | verify arXiv/venue |
| Fin-RATE | Jiang et al. 2026 | verify arXiv/venue |
| HiREC | Choe et al. 2025 | verify arXiv/venue |
| HC-RAG | arXiv:2608.12335 | arXiv-verified |
| FinCARDS | Zhou et al. 2026 | verify arXiv/venue |
| FinSAgent | Chi et al. 2026 | verify arXiv/venue |

## Claim verification / evidence ledger

| Short name | Record | Status |
|---|---|---|
| FinGround | arXiv:2604.23588 | arXiv-verified |
| CLAIR-Fin | arXiv:2608.13706 | arXiv-verified |
| VeriFin | arXiv:2608.10213 | arXiv-verified |
| EvidenceLens | arXiv:2606.23724 | arXiv-verified |
| Evidence-Ledger Adjudication | arXiv:2607.26512 | arXiv-verified |
| FinRCA-Bench | arXiv:2608.18534 | arXiv-verified |
| FinFIRST | arXiv:2609.25192 | arXiv-verified |
| FinVerBench | arXiv:2605.29586 | arXiv-verified |

## Sufficiency / selective prediction / abstention

| Short name | Record | Status |
|---|---|---|
| RealFin | arXiv:2602.07096 | arXiv-verified |
| SAFE-MR | arXiv:2610.06708 | arXiv-verified |
| FailSafe-RAG | 2026 | verify |
| GBFR | 2026 | verify |
| AWARE-FX | 2026 | verify |
| Chow reject option | Chow 1970 | classic |
| Selective prediction | Geifman & El-Yaniv 2017 | classic |

## Point-in-time

| Short name | Record | Status |
|---|---|---|
| FinanceGym | project site | verify |
| FinTrace/PITfall | GitHub | verify |
| Look-Ahead-Bench | arXiv:2601.13770 | arXiv-verified |
| AhaSignals PIT benchmark | GitHub | verify |
| Lookahead bias | Sarkar & Vafa 2025 | verify |

## VoI / evidence acquisition

| Short name | Record | Status |
|---|---|---|
| LAVOIR | arXiv:2609.30706 | arXiv-verified |
| REVOIR | arXiv:2609.37588 | arXiv-verified |
| VOILA | arXiv:2602.03007 | arXiv-verified |
| VoI framework for human-agent communication | arXiv:2601.06407 | arXiv-verified |
| RAVEL | arXiv:2609.21924 | arXiv-verified |
| POMDP / controlled sensing | Krishnamurthy 2017 | classic |
| Active sensing | Satsangi et al. 2018 | classic |

## Independence / defeasible reasoning

| Short name | Record | Status |
|---|---|---|
| InfoGatherer | arXiv:2603.05909 | arXiv-verified |
| Split the Labor | arXiv:2608.14509 | arXiv-verified |
| Dung argumentation | Dung 1995 | classic |
| Trust, but Don't Verify | arXiv:2606.05403 | arXiv-verified |

## Event forecasting / guidance tracking

| Short name | Record | Status |
|---|---|---|
| RICE-Alpha | arXiv:2609.34004 | arXiv-verified |
| PolyBench | arXiv:2604.14199 | arXiv-verified |
| SEC Filings to Financial Signals | arXiv:2610.24268 | arXiv-verified |
| Marvin Labs Guidance Scorecard | industry product | not peer-reviewed |
| NOSIBLE FLS detector | industry model | not peer-reviewed |

## Benchmarks

| Short name | Record | Status |
|---|---|---|
| FinDVer | arXiv:2411.05764, EMNLP 2024 | verify |
| Fin-Fact | arXiv:2309.08793, WWW 2025 | verify |
| FinBen | arXiv:2402.12659, NeurIPS 2024 | verify |
| FinMRAGBench | ACL Findings 2026, 2026.findings-acl.187 | verify |
| FinQA / TAT-QA / FinanceBench | classic financial QA | verify |

## Citation discipline

- Use `@misc` + `eprint` + `url` for arXiv preprints; do not invent venues.
- Verify author lists, titles, and identifiers against the primary record.
- Do not cite a paper for a specific number unless the number is in that paper.
- Industry products are context/baselines, not peer-reviewed evidence.
- Re-run the verification pass before camera-ready; 2026 venues may change.
