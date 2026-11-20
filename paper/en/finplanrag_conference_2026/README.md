# FinPlan conference manuscript

Top-conference version of the current FinPlanRAG paper.

- Source: `main.tex`
- PDF: `../FinPlanRAG_Conference_2026.pdf`
- Template: ACL two-column style (`acl.sty`, `acl_natbib.bst`)
- Layout: about 11 pages of main text; references start on page 12; the full PDF is 15 pages
- `main_review.tex` and `../FinPlanRAG_Conference_Review.pdf` are the earlier v2 decision-interface build; they are not part of the current router paper.

The manuscript follows one narrative:

1. Financial text analysis mixes determinate questions with forward-looking claims, and existing systems fix the output contract before seeing evidence.
2. FinPlanRAG routes each query to obligation retrieval or AnchorEvolve, with a shared point-in-time constraint and different output contracts.
3. Obligation retrieval uses typed obligations, coverage-driven acquisition, and a confidence-gated fallback; AnchorEvolve extracts L0/L1/L2 anchors, qualifies evidence, updates a belief, and returns a point answer, credible set, or abstention under an asymmetric loss.
4. Experiments cover DocFinQA, 227 XBRL-verified revenue-guidance claims, FinDVer, and a 573-query mixed workload, with DeepSeek, GLM 5.3, and Qwen3-8B, six named baselines, and classification, calibration, evidence-cost, decision-loss, and routing metrics.

Compile:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

For the review build, use `main_review.tex`.
