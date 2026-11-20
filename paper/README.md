# FinPlan-RAG paper drafts

This directory contains two generations of bilingual drafts derived from the
maintained FinPlan-RAG implementation and the SQCAD evidence discipline. The
v2 decision-interface research design is in `v2/`; it is the current research
direction, while the v1 conference drafts below remain the completed paper
artifacts.

## Current pre-defense thesis (2026-11-20)

The current pre-defense thesis is the two-branch FinPlanRAG version at
`zh/sufe_master/`, compiled to
`zh/FinPlanRAG_Chinese_SUFE_Master_2026_ContractRouting.pdf`. The title is
"FinPlanRAG: Contract Routing and Anchor Evolution for Mixed Financial
Queries". The router distinguishes determinate questions from forward-looking
claims; the determinate branch uses obligation retrieval, and the vague branch
uses the latest AnchorEvolve implementation with full-action readout,
validation-gated type-library extension, numeric bridges, and a 470-claim
A-share annual-report backtest. The vague branch also reports a 150-claim
FinDVer component check and a no-anchor architecture ablation.

The single-branch version at `zh/sufe_master_vague/` remains the source for the
forward-looking branch and is retained as a backup. It is 70 pages and focuses
only on the forward-looking claim contract, anchor state, qualification,
temporal evolution, and full-action readout. The two-branch version is the
submission candidate because it preserves the contract-routing contribution
that distinguishes the framework from selective-forecasting work such as
FinAbstain.

The forward-looking conference paper is `en/FinPlanRAG_Conference_2026_Forward.pdf`,
with LaTeX source in `en/finplanrag_conference_2026/main_forward.tex`. It is a
six-page ACL-format paper focused on the unresolved-claim decision problem, not
the older router version.

## Current bilingual artifacts (2026-11-09)

The mainline paper is now FinPlanRAG, an LLM-router framework with an obligation-retrieval branch and a forward-looking AnchorEvolve branch. The English conference paper is 15 pages including references and appendix, with about 11 pages of main text; the Chinese master's thesis at `zh/sufe_master/` is 64 pages, organized as seven chapters (introduction, literature review, research design, obligation retrieval, AnchorEvolve, empirical results, conclusion), including the appendix, acknowledgments, personal resume, and back cover. Both versions follow the same narrative: background and related work, research gap, framework and contributions, complete algorithm design, multi-benchmark and multi-backbone experiments, discussion, and conclusion.

## Earlier v2 artifacts (2026-11-08)

- English preprint PDF: `en/FinPlanRAG_English_2026.pdf`; self-contained source
  in `en/finplanrag_2026/`.
- Conference-style rewrite: `en/FinPlanRAG_Conference_2026.pdf` and
  `en/FinPlanRAG_Conference_Review.pdf`; source in
  `en/finplanrag_conference_2026/`.
- Chinese SUFE undergraduate thesis PDF:
  `zh/FinPlanRAG_Chinese_SUFE_2026.pdf`; source in `zh/sufe/`.
- Chinese SUFE master's thesis PDF:
  `zh/FinPlanRAG_Chinese_SUFE_Master_2026.pdf`; source in `zh/sufe_master/`.
- Reference audit: `reference_audit.md`; format/style review:
  `review_2026-11-04.md`.
- The current paper uses the loss-aware policy portfolio and six named
  baselines (FinDVer, Fin-Fact, Self-RAG, CRAG, FLARE, Adaptive-RAG), adapted
  under DeepSeek, GLM 5.3 flash, and a locally served open-weight Qwen3-8B.
  They load prompt templates and decision helpers from the pinned official
  repositories, with missing checkpoints/data substituted and documented; the
  paper and the artifact README state this. FinPlan keeps the best balanced
  accuracy and AUROC on all three backbones.
- The 1,005-claim benchmark is released in `data/` with a manifest and
  construction notes. Reproduction commands are in `reproduce.md`.
- The latest audit is `review_2026-11-07.md`; `review_2026-11-04.md` is the
  earlier format review.
- The config-driven workflow is `configs/paper.json` +
  `scripts/workflow.py`; provider/model/paths are not hardcoded. The
  provider-agnostic client supports DeepSeek, GLM 5.3 flash, a local Qwen3-8B,
  and any OpenAI-compatible endpoint; keys are read from environment variables.
- Real sequential evidence (path A): `paper/data/earnings_events.json` records
  the earliest 8-K Item 2.02 earnings release for 993/1,005 claims; all 273 test
  claims have one. `scripts/run_sequential_evidence.py` and
  `scripts/run_sequential_portfolio.py` run the real decision-time experiment.
  The 24-cell portfolio matches the oracle best policy with zero regret,
  selecting wait when delay is cheap and fixed adjudication when it is
  expensive; peer acquisition is not selected.
- Named-baseline artifacts for the three backbones are in
  `paper/results/official/{deepseek,glm,qwen3}/`; the aggregate audit is
  `paper/results/final_remote/final_report.md`. FinPlan keeps the best balanced
  accuracy and AUROC on all three backbones.
- The older v1 conference drafts below use the earlier obligation-closure
  framing.

## Conference drafts (v1)

- `en/conference/main.tex`: English ICLR 2027 conference draft, 9 pages of main
  text excluding references and appendix. The compiled PDF is
  `en/FinPlanRAG_English_ICLR2027.pdf`.
- `zh/conference/main.tex`: Chinese conference version with the same sections,
  evidence, and appendix. The compiled PDF is
  `zh/FinPlanRAG_Chinese_ICLR2027.pdf`.
- `en/conference/references.bib`: reference set used by both conference
  versions. Every entry used in the text is a primary, verifiable record; the
  four 2026 financial-RAG papers and FinCARDS were checked against their
  arXiv/ACL Anthology records. `references_verification.md` records the earlier
  Crossref pass.
- `results/reanalysis_v1.py` and `results/reanalysis_v1.json`: paired bootstrap
  re-analysis of the frozen per-case artifacts. The conference drafts use these
  intervals (20,000 resamples, seed 20261021).

The conference narrative is organized as background -> related work and
research gap -> contributions -> contract and theory -> method -> experiments ->
discussion -> conclusion. Its central claim is an identification result: on
LOFin, FinGLM, and the preregistered 24-question holdout, a metadata-fill
control with the same typed obligations but no risk-ranked cascade is exactly
tied with FinPlan-RAG. The gain is therefore attributed to the obligation
representation (period-aware parsing and hard typed binding), not to the search
policy. The controlled diagnostic still shows a cascade gain under a noisier
synthetic index, which is reported as a scope condition rather than a
contradiction.

Build commands:

```bash
# English (run from paper/en/conference)
pdflatex main && bibtex main && pdflatex main && pdflatex main

# Chinese (run from paper/zh/conference)
xelatex main && bibtex main && xelatex main && xelatex main
```

## Earlier drafts

- `en/main.tex`: English quantitative-finance-compatible article draft.
- `zh/main.tex`: Chinese `ctexbook` thesis-style draft.
- `results/controlled_summary.json`: frozen summary of the local controlled
  mechanism run. The full run is generated with the command in
  `results/README.md` and is not a claim of real-world QA superiority.
- `results/budget_sensitivity.json`: same-contract runs across six query
  budgets, used as a robustness and cost-frontier check.

Both generations use the same evidence boundary: controlled evidence
demonstrates the planning mechanism; document closure is not answer
correctness; the financial corpora provide file-level rather than
reader-level gold. The SQCAD-style audit is in `claim_evidence_matrix.md`;
remaining empirical gates are listed in `reproducibility_checklist.md`.

## Related paper workspace

The broader vague-claim adjudication line, AnchorEvolve, now lives outside this
repository as a sibling paper workspace: `../AnchorEvolve/`. Its ACL
Introduction draft, prototype, and 1,020-claim benchmark are a separate claim
set and are not part of the FinPlan-RAG evidence boundary.

## v2 research design (current direction)

`v2/` contains the research design for the new FinPlan-RAG paper. The v2 paper
is a decision-interface paper: a planner over credal evidence states that
jointly decides when to decide, what evidence to acquire, and which output mode
to use. It does **not** assume a fixed cutoff or a fixed budget; fixed
cutoff/budget is a special case/baseline. The v1 period-retrieval evidence is
kept as a special case, not discarded.

Start with `v2/README.md`; the master argument is `v2/research_design.md`.
The current ACL draft is `v2/acl/main.tex` (review and preprint PDFs in the
same directory). The selected algorithm is the loss-aware portfolio with a non-LLM RF backbone
in `v2/algorithm_selected_portfolio.md`; the planner and conformal layer are
reported components/alternatives in `v2/algorithm_selected_planner_rf.md` and
`v2/algorithm_v3_conformal.md`.
