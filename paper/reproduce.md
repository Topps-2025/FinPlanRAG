# Reproducing the FinPlan paper

**Current forward-looking paper:** the thesis in `paper/zh/sufe_master_vague/`
focuses on the forward-looking branch only.  Its benchmark specification,
named baselines and fairness protocol are in `docs/benchmark_plan.md` and
`configs/benchmark_plan.yaml`; run `python scripts/check_benchmark_plan.py`
to validate the plan.  The v2 workflow below is retained for provenance.  It
produces the older decision-interface and router results, which are not part
of the current paper's claims.

## Benchmark plan

The plan is a specification, not a result.  It covers the 1,005-claim revenue
guidance benchmark, its year-wise company-disjoint variant, FinDVer and
DocFinQA component checks, and the pre-registered A-share 2026 cases.  It
registers 20 baselines and 30 metrics and records which claims each metric can
support.

```bash
python scripts/check_benchmark_plan.py \
  --report paper/results/benchmark_plan/validation.json
```

The validator checks registries, data paths, split fields, adapters and
prospective-case thresholds without reading test labels or running a model.
External datasets are not committed; the validator reports them as warnings
until they are downloaded.

For the cloud-GPU run with DeepSeek, GLM and Qwen3-8B, see
`docs/cloud_gpu.md` and generate the command sheet with:

```bash
python scripts/run_cloud_suite.py --dry-run
```

## Cloud results (2026-11-20)

The three-backbone cloud run is summarized in
`paper/results/cloud_2026-11-20/README.md`. It contains the full-action
readout, qualification ablation, A-share smoke backtest, anchor-type
self-evolution and the within-range stratified result. The raw per-claim
files and summaries are in the same directory; large anchor caches are
excluded.

The type self-evolution loop is reproducible with:

```bash
python experiments/evolution/self_evolve.py   --config paper/results/cloud/configs/deepseek_evolution.yaml   --rounds 1 --max-mine 6 --max-types 2   --out paper/results/self_evolve/deepseek_beam
```

## Config-driven workflow

The general workflow is in `scripts/workflow.py`; all paths, seeds, loss
configs, provider/model, and the sequential cost grid are in
`configs/paper.json`. Backbone-specific named-baseline configs are
`configs/deepseek.json`, `configs/glm.json`, and `configs/qwen3.json`. No path,
model name, or API key is hardcoded in the runner.

```bash
# validate the released data and build SEC earnings events if missing
PYTHONPATH="src;scripts" python scripts/workflow.py --config configs/paper.json --stage data

# company-disjoint logistic benchmark (all + verified)
# FINPLAN_BOOTSTRAP=300 gives a quick point-estimate check; omit for 20k CIs.
FINPLAN_BOOTSTRAP=300 PYTHONPATH="src;scripts" python scripts/workflow.py --config configs/paper.json --stage benchmark

# real sequential evidence (peer outcomes + earnings release)
PYTHONPATH="src;scripts" python scripts/workflow.py --config configs/paper.json --stage sequential
PYTHONPATH="src;scripts" python scripts/workflow.py --config configs/paper.json --stage sequential_portfolio

# six named baselines on the configured provider (GLM by default)
GLM_API_KEY=... PYTHONPATH="src;scripts" python scripts/workflow.py   --config configs/paper.json --stage named

# aggregate the stage summaries
PYTHONPATH="src;scripts" python scripts/workflow.py --config configs/paper.json --stage report
```

The protocol constants are centralized in
`src/finplan_rag/workflow/protocol.py`; the workflow checks that
`configs/paper.json` agrees with them. Provider response caches are under
`paper/results/llm_cache/` and `paper/data/sec_cache/` and are gitignored; the
derived per-claim outputs and `paper/data/earnings_events.json` are kept.

## Providers

- DeepSeek: `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`, `ANTHROPIC_AUTH_TOKEN`.
- GLM 5.3 flash: `GLM_API_KEY`; default base URL
  `https://open.bigmodel.cn/api/paas/v4`; model `glm-5.3-flash`; the client
  uses `reasoning_effort=low` because GLM 5.x cannot disable thinking.
- Any OpenAI-compatible endpoint: `OPENAI_API_KEY`, `OPENAI_BASE_URL`,
  `OPENAI_MODEL`.
- Local Qwen3-8B via vLLM: `QWEN_API_KEY` (can be `dummy`),
  `QWEN_BASE_URL=http://127.0.0.1:8000/v1`, model `qwen3-8b`; disable the
  default thinking trace with
  `chat_template_kwargs={"enable_thinking": false}`.

The key is read from the environment and never written to an artifact.

The reported numbers use the pinned experiment environment in
`requirements-reproduce.txt` (`scikit-learn==1.7.2`, `scipy==1.15.3`,
`numpy==2.2.6`). Newer scikit-learn versions converge to slightly different
logistic coefficients; the per-claim artifacts and the commands above reproduce
the reported point estimates under the pinned environment.

## Real sequential evidence (path A)

`scripts/build_earnings_events.py` finds the earliest 8-K Item 2.02 earnings
release after each cutoff. Coverage is 993/1,005 claims; all 273 test claims
have one; median lead over the 10-Q is 2 days; median delay from cutoff to the
earnings release is 91 days. `scripts/run_sequential_evidence.py` runs one cost
configuration; `scripts/run_sequential_portfolio.py` runs the cost grid and the
validation-selected portfolio. The portfolio matches the oracle best candidate
in all 24 cost cells (mean/max regret 0), selecting wait when delay is cheap and
fixed adjudication when it is expensive.

This file lists the commands and artifacts for the results in the English and
Chinese drafts. The dataset is in `paper/data/`; the code is in `scripts/` and
`src/finplan_rag/`. All commands are run from the repository root with
`PYTHONPATH="src;scripts"` on Windows (use `src:scripts` on POSIX).

## Environment

```text
Python 3.10+
numpy, scipy, scikit-learn, pandas, requests
```

The controlled and decision-layer tests use no API key. The named baselines
use the Anthropic-compatible DeepSeek endpoint:

```text
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_MODEL=deepseek-flash[1m]
ANTHROPIC_AUTH_TOKEN=<set locally; never commit>
```

The model identifier is recorded in
`paper/results/named_baselines/summary.json`. It is a provider alias, not a
frozen snapshot; exact regeneration depends on the provider. The per-claim
outputs and raw response text are released so the analysis can be checked
without the API.

## Main benchmark and decision layer

```bash
# Table 2: logistic five-seed ensemble, all labels and verified-only
PYTHONPATH="src;scripts" python scripts/run_company_disjoint_benchmark.py \
  --data paper/data/dataset_v4.json \
  --out paper/results/company_disjoint_benchmark

PYTHONPATH="src;scripts" python scripts/run_company_disjoint_benchmark.py \
  --data paper/data/dataset_v4.json --verified-only \
  --out paper/results/company_disjoint_benchmark_verified

# Table 3: RF 200-tree portfolio, exploratory seeds 1-10
for s in 1 2 3 4 5 6 7 8 9 10; do
  PYTHONPATH="src;scripts" python scripts/run_backbone_conformal_diagnostics.py \
    --data paper/data/dataset_v4.json --seeds $s --models rf --rf-trees 200 \
    --configs base,symmetric,over_averse,under_averse,cheap_abstain,expensive_abstain,narrow_penalty,wide_penalty,alpha_05,alpha_20 \
    --thresholds 31 --out paper/results/rf_all_s$s
done

PYTHONPATH="src;scripts" python scripts/analyze_portfolio.py \
  --validation-root paper/results/rf_all_s \
  --validation-seeds 6,7,8,9,10 \
  --test-root paper/results/rf_all_s \
  --test-seeds 1,2,3,4,5,6,7,8,9,10 \
  --margin 0.15 --out paper/results/portfolio_10seed.json

# Confirmatory seeds 11-20 (four cells)
for s in 11 12 13 14 15 16 17 18 19 20; do
  PYTHONPATH="src;scripts" python scripts/run_backbone_conformal_diagnostics.py \
    --data paper/data/dataset_v4.json --seeds $s --models rf --rf-trees 200 \
    --configs base,cheap_abstain,narrow_penalty,wide_penalty \
    --thresholds 31 --out paper/results/rf_confirm_s$s
done

PYTHONPATH="src;scripts" python scripts/analyze_portfolio.py \
  --validation-root paper/results/rf_confirm_s \
  --validation-seeds 11,12,13,14,15,16,17,18,19,20 \
  --test-root paper/results/rf_confirm_s \
  --test-seeds 11,12,13,14,15,16,17,18,19,20 \
  --margin 0.15 --out paper/results/portfolio_confirm.json

PYTHONPATH="src;scripts" python scripts/analyze_confirmatory.py
```

Expected: `portfolio_10seed.json` mean regret -0.0370 (95% CI
[-0.0643, -0.0100]), max 0.0000; `portfolio_confirm.json` four-cell mean
-0.0880 (95% CI [-0.1610, -0.0162]); `rf_confirm_summary.json` primary
wide-width difference -0.8648 (95% CI [-1.2256, -0.4903]) against the
width-tuned fixed adjudicator.

## Switching-margin sensitivity (Appendix A.6)

```bash
for m in 0.0 0.05 0.10 0.15 0.20 0.30; do
  PYTHONPATH="src;scripts" python scripts/analyze_portfolio.py \
    --validation-root paper/results/rf_all_s \
    --validation-seeds 6,7,8,9,10 \
    --test-root paper/results/rf_all_s \
    --test-seeds 1,2,3,4,5,6,7,8,9,10 \
    --margin $m --out paper/results/portfolio_margin_$m.json
done
```

Expected: margins below .15 select the planner under under-averse and give
mean regret -0.0322, max +0.0476; margins .15 and above select fixed and give
-0.0370, max 0.000.

## Named baselines (Appendix A.4)

The runner is provider-agnostic. Set the key in the environment; never put it in
a command line, a config file, or an artifact. The frozen comparison uses three
backbones: DeepSeek (`deepseek-flash[1m]`), GLM 5.3 flash
(`glm-5.3-flash`), and a locally served open-weight Qwen3-8B.

```bash
# DeepSeek (Anthropic-compatible endpoint)
ANTHROPIC_AUTH_TOKEN=... PYTHONPATH="src;scripts" python scripts/run_named_baselines.py \
  --data paper/data/dataset_v4.json --verified-only \
  --variants findver,finfact,selfrag,crag,flare,adaptive --workers 8 \
  --provider deepseek --model 'deepseek-flash[1m]' \
  --base-url https://api.deepseek.com/anthropic \
  --out paper/results/deepseek/named

# GLM 5.3 flash (OpenAI-compatible endpoint)
GLM_API_KEY=... PYTHONPATH="src;scripts" python scripts/run_named_baselines.py \
  --data paper/data/dataset_v4.json --verified-only \
  --variants findver,finfact,selfrag,crag,flare,adaptive --workers 8 \
  --provider glm --model glm-5.3-flash --base-url https://open.bigmodel.cn/api/paas/v4 \
  --out paper/results/glm/named

# Local Qwen3-8B (vLLM OpenAI-compatible server on port 8000)
QWEN_API_KEY=dummy PYTHONPATH="src;scripts" python scripts/run_named_baselines.py \
  --data paper/data/dataset_v4.json --verified-only \
  --variants findver,finfact,selfrag,crag,flare,adaptive --workers 6 \
  --provider openai --model qwen3-8b --base-url http://127.0.0.1:8000/v1 \
  --api-key-env QWEN_API_KEY \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --out paper/results/qwen3/named
```

The local server is started from the shared model directory, never from a
duplicate download:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/models/Qwen3-8B \
  --served-model-name qwen3-8b --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.80 --max-model-len 16384 --trust-remote-code
```

For each backbone, the evaluator converts parsed labels to probability rows,
computes the full metric suite, and runs the paired comparison against the
frozen FinPlan logistic reference:

```bash
for backbone in deepseek glm qwen3; do
  PYTHONPATH="src;scripts" python scripts/evaluate_named_backbone.py \
    --input paper/results/${backbone}/named/per_claim.jsonl \
    --reference paper/results/benchmark_per_claim_verified_calibrated/per_claim.jsonl \
    --out paper/results/${backbone}/named --no-bootstrap
done
```

The prompts, retrieval recipe, FLARE threshold, Adaptive-RAG routing, parse
repair, and caching are in `scripts/run_named_baselines.py` and
`scripts/deepseek_client.py`. The released `per_claim.jsonl` contains the
parsed label, confidence, credible set, rationale, raw response text, token
usage, latency, route, and retrieved ids for every claim and variant. The GLM
artifacts in `paper/results/glm/named/` were produced with an authenticated
GLM endpoint; the repository stores the outputs, not the key.

### Official-source baseline runs

The source-grounded tables load prompt templates and decision helpers from the
vendored official repositories. Clone them at the pinned commits recorded in
`scripts/official_prompts.py` under `vendor/official_baselines/`, or set
`FINPLAN_OFFICIAL_ROOT` to the directory containing `findver`, `finfact`,
`selfrag`, `crag`, `flare`, and `adaptiverag`.

```bash
bash scripts/clone_official_baselines.sh
```

```bash
for backbone in deepseek glm qwen3; do
  PYTHONPATH="src;scripts" python scripts/run_official_baselines.py \
    --data paper/data/dataset_v4.json --verified-only --workers 6 \
    --provider "$BACKBONE_PROVIDER" --model "$BACKBONE_MODEL" \
    --base-url "$BACKBONE_BASE_URL" \
    --out paper/results/official/${backbone}
  PYTHONPATH="src;scripts" python scripts/evaluate_named_backbone.py \
    --input paper/results/official/${backbone}/per_claim.jsonl \
    --reference paper/results/benchmark_per_claim_verified_calibrated/per_claim.jsonl \
    --out paper/results/official/${backbone} --no-bootstrap
done
```

Each run writes `provenance.json`, which records the official repository URL,
commit, file hashes, reused components, and substitutions. The runner does not
pretend that the original systems are unmodified: the original datasets,
retrievers, and some checkpoints are unavailable or incompatible with the
claim-outcome task.

The comparison environment is pinned to `scikit-learn==1.7.2`,
`scipy==1.15.3`, and `numpy==2.2.6`. Newer scikit-learn versions converge to
slightly different logistic coefficients and change the third decimal of some
metrics; the released per-claim probabilities and the commands above reproduce
the reported point estimates under the pinned environment.

## Rolling-origin benchmark

```bash
PYTHONPATH="src;scripts" python scripts/run_rolling_benchmark.py \
  --data paper/data/dataset_v4.json --models logistic,rf \
  --out paper/results/rolling_benchmark
```

Expected: n=737 (273 test claims plus 464 earlier claims), RF .537
accuracy/.503 balanced/.032 ECE, logistic .487/.535/.092.

Classical backbone diagnostic (not a named baseline):

```bash
PYTHONPATH="src;scripts" python scripts/run_classical_baselines.py \
  --data paper/data/dataset_v4.json --verified-only --seed 1 \
  --models logistic,histgb,rf,xgboost,lightgbm,catboost,tfidf,tfidf_numeric \
  --out paper/results/classical
```

Expected: calibrated FinPlan logistic balanced .561; classical logistic .549,
random forest .497, TF-IDF + numeric .516. The script also writes
`summary_seed1_verified.json` and `report_seed1_verified.md`.

## Real pre-hoc router

```bash
PYTHONPATH="src;scripts" python scripts/run_router_baseline.py \
  --data paper/data/dataset_v4.json
```

Expected: oracle-mode router base 3.5165 vs planner 3.9267; the two policies
each win five of the ten configurations (see
`paper/results/router_baseline/summary.json`).

## Tests

```bash
PYTHONPATH="src;scripts" python -m pytest -q
```

The test suite covers the decision loss, credible sets, planner stopping and
waiting, pilot data schema, preliminary-signal determinism, budgeted accuracy,
metrics, and the real-benchmark categorical-column regression.

## Artifact map

| Paper item | Artifact |
|---|---|
| Table 2 | `paper/results/company_disjoint_benchmark*/summary.json` |
| Table 3 | `paper/results/portfolio_10seed.json`, `rf_all_s{1..10}/` |
| Confirmatory paragraph | `paper/results/portfolio_confirm.json`, `rf_confirm_summary.json` |
| Official-source baselines, DeepSeek | `paper/results/official/deepseek/metrics_summary.json`, `per_claim.jsonl`, `provenance.json` |
| Official-source baselines, GLM | `paper/results/official/glm/metrics_summary.json`, `per_claim.jsonl` |
| Official-source baselines, Qwen3 | `paper/results/official/qwen3/metrics_summary.json`, `per_claim.jsonl` |
| Named baselines, prompt-only DeepSeek | `paper/results/deepseek/named/metrics_summary.json`, `per_claim.jsonl` |
| Named baselines, GLM | `paper/results/glm/named/metrics_summary.json`, `per_claim.jsonl` |
| Named baselines, Qwen3 | `paper/results/qwen3/named/metrics_summary.json`, `per_claim.jsonl` |
| Per-class table | `paper/results/deepseek/named/metrics_*.json` |
| Aggregate audit | `paper/results/final_remote/final_report.{json,md}` |
| Rolling benchmark (regenerated) | `paper/results/rolling_remote/metrics_*.json` |
| Table 8 rolling | `paper/results/rolling_benchmark/*_metrics.json` |
| Table 9 logistic decision | `paper/results/real_decision_eval_verified_calibrated/summary.json` |
| Table 10 router | `paper/results/router_baseline/summary.json` |
| Table 11 controlled | `paper/v2/multiseed_result.md`, `paper/results/v2_multiseed/` |
| Tables 12-13 conformal | `paper/results/deepseek_conformal_realized_verified/`, `conformal_v3_realized_seed*/` |
| Loss grid | `scripts/run_real_decision_eval.py:48-60` |
| Dataset | `paper/data/dataset_v4.json`, `paper/data/README.md` |

## What is not reproduced bit-for-bit

- The LLM outputs depend on the provider's model alias and serving stack.
  Temperature 0 reduces but does not eliminate run-to-run variation.
- The full raw response cache is 91 MB and is not committed. The released
  per-claim files contain the parsed outputs and raw response text for the
  reported runs.
- The benchmark labels are automatic; a human-annotated gold set would require
  a new annotation pass.

## Thesis figures and ablations (2026-11-09)

- Main-results figure: `python scripts/make_main_results_figure.py --results paper/results/final/results.json --out paper/zh/sufe_master/figures/main_results_mono.pdf`
- Anchor-component ablations: `python scripts/analyze_anchor_ablation.py` (writes `paper/results/unified/anchor_ablation.json`)
- Thesis tables: `python experiments/unified/make_thesis_tables.py --results paper/results/final/results.json --out paper/zh/sufe_master/tables_thesis`
- Compilation: see `paper/zh/sufe_master/README.md`.
