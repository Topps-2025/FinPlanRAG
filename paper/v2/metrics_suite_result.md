# Extended metric suite and named baseline comparison (2026-11-04)

## Purpose

The earlier evaluation reported accuracy, balanced accuracy, macro F1,
selective accuracy, AURC, ECE, and decision loss. This run adds AUROC, average
precision, MCC, Cohen's kappa, Brier, NLL, classwise ECE, calibration slope,
risk-at-80/90, year/sector/truth slices, claim-bootstrap intervals, and paired
McNemar comparisons.

## Named baseline table (227 XBRL-verified claims)

The six named baselines are adaptations under the same DeepSeek backbone:
FinDVer, Fin-Fact, Self-RAG, CRAG, FLARE, and Adaptive-RAG. Details and
boundaries are in `named_baselines_result.md`.

| Model | Acc. | Bal. | Macro F1 | MCC | AUROC | AP | Sel@50 | AURC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FinPlan (logistic) | .493 | .561 | .485 | .273 | .716 | .524 | .526 | .468 |
| FinDVer-style | .551 | .461 | .464 | .233 | .674 | .494 | .675 | .337 |
| Fin-Fact-style | .537 | .446 | .450 | .201 | .650 | .475 | .675 | .338 |
| Self-RAG-style | .546 | .451 | .456 | .216 | .670 | .491 | .684 | .333 |
| CRAG-style | .529 | .423 | .418 | .177 | .643 | .473 | .658 | .349 |
| FLARE-style | .511 | .417 | .411 | .164 | .634 | .467 | .596 | .362 |
| Adaptive-RAG-style | .493 | .406 | .400 | .143 | .584 | .425 | .579 | .415 |

No point model dominates. FinDVer-style has the best accuracy; Self-RAG-style
the best selective accuracy and AURC; FinPlan the best balanced accuracy,
macro F1, MCC, AUROC, and AP. The LLM rows return a label confidence, not a
normalized three-class distribution; NLL/Brier/ECE are not comparable and are
not shown.

## Paired comparisons against FinPlan logistic

| Model | Accuracy difference | McNemar p | AUROC difference [95% CI] |
|---|---:|---:|---:|
| FinDVer | +.057 | .165 | -.043 [-.099,.004] |
| Fin-Fact | +.044 | .295 | -.067 [-.124,-.018] |
| Self-RAG | +.053 | .207 | -.047 [-.104,.003] |
| CRAG | +.035 | .440 | -.074 [-.127,-.021] |
| FLARE | +.018 | .734 | -.083 [-.141,-.034] |
| Adaptive-RAG | +.000 | 1.000 | -.132 [-.187,-.072] |

McNemar does not reject equal accuracy for any named baseline. Fin-Fact, CRAG,
FLARE, and Adaptive-RAG have lower AUROC with intervals excluding zero; the
FinDVer and Self-RAG intervals include zero. The result is a multi-metric
Pareto front, not a single-model win.

## Rolling-origin benchmark

| Model | $n$ | Acc. | Bal. | Macro F1 | AUROC | AP | ECE | Sel@50 | AURC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RF | 737 | .537 | .503 | .498 | .675 | .482 | .032 | .593 | .414 |
| Logistic | 737 | .487 | .535 | .480 | .683 | .484 | .092 | .501 | .480 |

The rolling set trains on other company folds with outcomes before January 1
of each cutoff year and tests every claim in 2015--2025. Yearly accuracy is
heterogeneous (logistic .32 in 2021, .63 in 2024); slices are in
`paper/results/rolling_benchmark/*_metrics.json`.

## Named baselines

- FinDVer, Fin-Fact, Self-RAG, CRAG, FLARE, and Adaptive-RAG are adapted to the
  claim-outcome task under the same DeepSeek backbone; they are not official
  code releases.
- FinRank, Fin-RATE, HC-RAG, and FinSAgent are not run because no compatible
  released code/shared claim labels were available; their paper numbers are on
  different data.
- Generic controls (logistic, RF, TF-IDF, XGBoost, LightGBM, CatBoost) are
  retained only as backbone diagnostics, not as named baselines.

## Artifacts and commands

```text
PYTHONPATH="src;scripts" python scripts/compute_metrics_suite.py \
  --per-claim paper/results/benchmark_per_claim_verified_calibrated/per_claim.jsonl \
  --out paper/results/benchmark_metrics_verified_calibrated.json

PYTHONPATH="src;scripts" python scripts/run_named_baselines.py \
  --verified-only --variants findver,finfact,selfrag,crag,flare,adaptive \
  --workers 8 --out paper/results/named_baselines

PYTHONPATH="src;scripts" python scripts/compare_models.py \
  --inputs FinPlan_Logistic=paper/results/benchmark_per_claim_verified_calibrated/per_claim.jsonl \
    FinDVer=paper/results/named_baselines/prob_findver.jsonl \
    FinFact=paper/results/named_baselines/prob_finfact.jsonl \
    SelfRAG=paper/results/named_baselines/prob_selfrag.jsonl \
    CRAG=paper/results/named_baselines/prob_crag.jsonl \
    FLARE=paper/results/named_baselines/prob_flare.jsonl \
    AdaptiveRAG=paper/results/named_baselines/prob_adaptive.jsonl \
  --out paper/results/named_baselines/model_comparison_logistic.json

PYTHONPATH="src;scripts" python scripts/run_rolling_benchmark.py \
  --models logistic,rf --out paper/results/rolling_benchmark
```

## Boundaries

- The named LLM baselines inherit the pretraining-contamination caveat; the
  RF/logistic rolling benchmark is clean.
- The rolling benchmark is not a new claim sample; it expands the cutoff years
  and test claims under the same point-in-time construction.
- The metric suite is a descriptive comparison; the selected decision layer is
  still evaluated by decision loss/regret, not by a single point metric.
