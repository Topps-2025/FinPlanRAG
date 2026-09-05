# Reference verification report (2026-09-05)

This report records the first metadata pass for the expanded bibliography used by both drafts. It is deliberately conservative: a bibliography entry is not treated as evidence for a result unless the cited paper is primary and the claim is within its scope.

## Crossref-checked DOI records

The following DOI records returned matching title and publication metadata from the Crossref API on 2026-09-05:

| Key | DOI | Result |
|---|---|---|
| `park2023generative` | 10.1145/3586183.3606763 | title and 2023 publication record match |
| `memorybank` | 10.1609/aaai.v38i17.29946 | title and 2024 AAAI record match |
| `maharana2024locomo` | 10.18653/v1/2024.acl-long.747 | title and ACL 2024 record match |
| `satsangi2018active` | 10.1007/s10514-017-9666-5 | title and 2018 journal record match |
| `gutierrez2024hipporag` | 10.52202/079017-1902 | title and 2024 proceedings record match |

The Crossref endpoint did not resolve the ACL Anthology DOI strings for `jeong2024adaptiverag` and `jiang2023flare` in this environment; their BibTeX entries retain the official ACL Anthology DOI URLs and should be rechecked against ACL Anthology before submission.

## URL/arXiv records

`memorybank`, `maharana2024locomo`, `wu2024longmemeval`, `packer2023memgpt`, `rasmussen2025zep`, `yuan2026retrieval`, `zhou2025mem1`, `kim2025premem`, `cai2025fracom`, `tan2025membench`, `hu2025memoryagentbench`, `yao2023react`, `shinn2023reflexion`, and `wang2023voyager` retain explicit arXiv or publisher URLs in `paper/en/references.bib`. The final submission pass should record the version/date actually cited for preprints.

## Provisional or project-supplied entries

`simplemem`, `actmem`, `oblivion`, `memoryworth`, `trivium`, `govmem`, and `zou2026demem` are included because they are relevant to the supplied SQCAD research context, but their current records are `@misc`/early-release entries. They must not be used as sole support for a novelty or superiority claim. Before submission, replace each with a primary, citable version or remove it from the main text.

## Citation hygiene

- The drafts use author--year citations and an expanded bibliography; `plainnat` is a compile-time fallback, not the final QF style.
- Every empirical number in the drafts points to a local result JSON or provenance file, not to a literature citation.
- No SQCAD result is presented as a FinPlan-RAG result. SQCAD is used only as a protocol and evidence-ladder reference.
- Before upload, run a final DOI/title/author/year/page audit and freeze the `.bib` hash together with the PDF.
