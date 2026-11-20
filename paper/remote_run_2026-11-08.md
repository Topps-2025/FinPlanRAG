# Remote reproduction run (2026-11-08)

This run used a shared AutoDL GPU container. No other user's process was
terminated, no existing model directory was overwritten, and the Qwen3-8B
weights were reused from the shared data disk rather than downloaded again.

## Host and storage

- Host: `autodl-container-g1f8p6tjj9-71341b74` (RTX 4080 32 GB, 128 CPU, 503 GB RAM)
- Data disk: `/root/autodl-tmp` (150 GB; project at `/root/autodl-tmp/FinPlanRAG`)
- Shared models: `/root/autodl-tmp/models/Qwen3-8B` (16 GB, reused read-only)
- Shared vLLM endpoint: `http://127.0.0.1:8000/v1`, model name `qwen3-8b`
- Project archive: `paper/data/dataset_v4.json`, SHA-256
  `cf24c311806cc15d204b19b8a52c8f3bae6e91bbee563f448cf6643d40923f24`

## Environment

```text
Python 3.12.3
numpy 2.2.6
scipy 1.15.3
scikit-learn 1.7.2
torch 2.8.0+cu128
vllm 0.11.0
```

The venv is `/root/autodl-tmp/FinPlanRAG/.venv` with `--system-site-packages`.
`scikit-learn==1.7.2` and `scipy==1.15.3` are pinned because the frozen paper
artifacts use that environment; newer versions converge to slightly different
logistic coefficients.

## Commands

```bash
cd /root/autodl-tmp/FinPlanRAG
export PYTHONPATH=src:scripts

# Benchmark (all and verified labels)
.venv/bin/python scripts/workflow.py --config configs/paper.json --stage benchmark

# Sequential real earnings-release portfolio (24 cost cells)
.venv/bin/python scripts/workflow.py --config configs/paper.json --stage sequential_portfolio

# Named baselines: DeepSeek, GLM 5.3 flash, local Qwen3-8B
.venv/bin/python scripts/run_named_baselines.py ... --out paper/results/deepseek/named
.venv/bin/python scripts/run_named_baselines.py ... --out paper/results/glm/named
.venv/bin/python scripts/run_named_baselines.py ... --out paper/results/qwen3/named

# Convert, metric suite, paired comparison
for backbone in deepseek glm qwen3; do
  .venv/bin/python scripts/evaluate_named_backbone.py \
    --input paper/results/${backbone}/named/per_claim.jsonl \
    --reference paper/results/benchmark_per_claim_verified_calibrated/per_claim.jsonl \
    --out paper/results/${backbone}/named --no-bootstrap
done

# Rolling benchmark and aggregate report
.venv/bin/python scripts/run_rolling_benchmark.py --data paper/data/dataset_v4.json \
  --models logistic,rf --out paper/results/rolling_remote
.venv/bin/python scripts/build_final_report.py --root paper/results \
  --out paper/results/final_remote
```

The exact provider flags for each backbone are in `paper/reproduce.md`.

## GLM endpoint note

The first GLM key supplied for this run returned HTTP 401 (`error.code=1000`).
A rotated key authenticated on the same endpoint
(`https://open.bigmodel.cn/api/paas/v4`), so the six GLM adaptations were rerun
in full on 227 verified claims. The new outputs were written to
`paper/results/glm_new/named/`, validated, and promoted to
`paper/results/glm/named/`; the previous frozen outputs are retained at
`paper/results/glm_frozen_2026-11-07/named/`. The key was read from a root-only
environment file outside the repository, was never printed or written to an
artifact, and the file was removed after the run.

## Main outcomes

- Benchmark verified five-seed ensemble: accuracy .4978, balanced .5635, macro
  F1 .4884, ECE .0912. All-label ensemble: .5165/.5656/.5042.
- Three named-baseline backbones: FinPlan balanced accuracy .561/.561/.561 and
  AUROC .716/.716/.716 for DeepSeek/GLM/Qwen3 (GLM paired test: .558/.715 on
  226 common claims); the best baseline is .459/.439/.477 balanced and
  .671/.642/.609 AUROC.
- Ten-seed decision portfolio: mean regret -0.0370, max 0.0000.
- Confirmatory four-cell portfolio: mean regret -0.0880, max 0.0000.
- Real earnings-release sequential portfolio: 24/24 zero regret; fixed
  adjudication selected in 16 cells and always-wait in 8; peer acquisition and
  the planner are never selected.
- Rolling 737-claim benchmark: RF .537/.503/.032 (acc/balanced/ECE), logistic
  .487/.535/.092.
- Classical diagnostic (seed 1): FinPlan calibrated logistic balanced .561;
  classical logistic .549, random forest .497, TF-IDF+numeric .516.
- Tests: 41 passed.

## Official-source baseline run (2026-11-08)

The six official repositories were cloned at pinned commits under
`vendor/official_baselines/`:

| Method | Repository | Commit |
|---|---|---|
| FinDVer | `yilunzhao/FinDVer` | `e8bb237` |
| Fin-Fact | `IIT-DM/Fin-Fact` | `b303878` |
| Self-RAG | `AkariAsai/self-rag` | `1fcdc42` |
| CRAG | `HuskyInSalt/CRAG` | `de7c296` |
| FLARE | `jzbjyb/FLARE` | `ec4b06b` |
| Adaptive-RAG | `starsuzi/Adaptive-RAG` | `0c88670` |

`scripts/official_prompts.py` loads the official prompt templates and decision
helpers from these files; `scripts/run_official_baselines.py` runs the shared
three-way adapter. Each output directory contains `per_claim.jsonl`,
`metrics_*.json`, `model_comparison_logistic.json`, `summary.json`, and
`provenance.json`.

```bash
for backbone in deepseek glm qwen3; do
  PYTHONPATH="src;scripts" python scripts/run_official_baselines.py \
    --data paper/data/dataset_v4.json --verified-only --workers 6 \
    --provider "$PROVIDER" --model "$MODEL" --base-url "$BASE_URL" \
    --out paper/results/official/${backbone}
  PYTHONPATH="src;scripts" python scripts/evaluate_named_backbone.py \
    --input paper/results/official/${backbone}/per_claim.jsonl \
    --reference paper/results/benchmark_per_claim_verified_calibrated/per_claim.jsonl \
    --out paper/results/official/${backbone} --no-bootstrap
done
```

DeepSeek was run in one batch with 8 workers; no further DeepSeek requests were
made after that batch. GLM and Qwen3 were run at 6 workers. Qwen3-8B was served
locally on port 8100 from the shared model directory; the model was not
downloaded again and no other CLI process was terminated. The local vLLM server
was stopped after the run; both API key files were removed from the remote host.

Official-source results (227 verified claims, FinPlan row fixed):

| Backbone | FinPlan Bal | best baseline Bal | FinPlan AUROC | best baseline AUROC |
|---|---:|---|---:|---|
| DeepSeek | .561 | .469 (CRAG) | .716 | .689 (CRAG) |
| GLM 5.3 flash | .561 | .481 (FinDVer) | .716 | .664 (FinDVer) |
| Qwen3-8B | .561 | .509 (FLARE) | .716 | .630 (FLARE) |

All six methods parse all 227 rows on every backbone; the earlier prompt-only
GLM CRAG parse failure does not apply to the official-source run.
