# Reference audit (2026-11-08)

## Scope

- English source: `paper/en/finplanrag_2026/main.tex` and
  `paper/en/finplanrag_2026/references.bib`.
- Chinese source: `paper/zh/sufe/main.tex` and
  `paper/zh/sufe/references.bib`.
- The English and Chinese bibliographies are derived from the same
  `paper/v2/acl/references.bib`, with `FinCARDS` added only to the Chinese
  related-work section.

## Citation completeness

- English: 32 distinct citation keys, 32 bibliography entries after adding
  `qwen2025qwen3`. No missing key, no unused entry.
- Chinese: 33 entries after adding `chen2026fincards` and `qwen2025qwen3`, all
  cited through `biblatex` + `gb7714-2015`; the biber run reports no missing
  entries.
- Both PDFs compile without undefined references.

## 2026 financial-RAG entries

The following entries were checked against arXiv/DOI records on 2026-11-04:

| Key | Record | Status |
|---|---|---|
| `mansouri2026finrank` | arXiv:2608.07400, FinRank, SEC-filing evidence benchmark | verified |
| `jiang2026finrate` | arXiv:2602.07294; KDD '26, DOI 10.1145/3770855.3817528 | verified |
| `chen2026hcrag` | arXiv:2608.12335, HC-RAG | verified |
| `chi2026finsagent` | arXiv:2607.18102, FinSAgent | verified |
| `chen2026fincards` | arXiv:2601.06992; Findings of ACL 2026, pp. 24836--24852 | verified |
| `qwen2025qwen3` | arXiv:2505.09388, Qwen3 Technical Report | verified |

These are recent preprints or conference papers. The thesis cites them as
related systems and does not claim to reproduce their code or paper numbers.
FinRank, Fin-RATE, HC-RAG, and FinSAgent are not run on the claim-outcome
benchmark because no compatible released code or shared labels are available.

## Standard references

The remaining entries are standard RAG, financial QA, selective prediction,
conformal prediction, calibration, and decision-theory references. Their
authors, venues, and years were checked against the bibliography; no
fabricated DOI or invented author list was found in the current files.
`krishnamurthy2017pomdp` was additionally checked as arXiv:1701.00179.

## What the audit does not establish

- It does not establish that every numerical claim in a cited paper is
  reproduced. It only checks that the bibliographic record exists and is
  cited in the right context.
- It does not make the 2026 preprints peer-reviewed. The Chinese and English
  texts call them preprints or conference papers where appropriate.
