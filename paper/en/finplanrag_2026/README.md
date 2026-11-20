# English manuscript

- Source: `main.tex`
- Bibliography: `references.bib`
- Figures: `figures/`
- PDF: `../FinPlanRAG_English_2026.pdf`

Compile:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The manuscript is the ACL preprint build of the current FinPlanRAG paper. It
uses the same narrative as the conference build: research background and
related work, research gap, framework and contributions, complete algorithm
design for obligation retrieval and AnchorEvolve, multi-benchmark and
multi-backbone experiments, discussion, and conclusion. Main text is about 11
pages; references start on page 12; the full PDF is 15 pages. The benchmark is
released in `../../data/`; reproduction commands are in `../../reproduce.md`.
