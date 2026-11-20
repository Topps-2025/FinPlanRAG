# Named strong baselines (2026-11-04)

## Purpose

The earlier baseline table mixed generic controls (logistic, TF-IDF, random
forest) with LLM readers. This run replaces that comparison with six named
methods adapted to the claim-outcome task under the same DeepSeek backbone:

- FinDVer-style claim verification \citep{zhao2024findver};
- Fin-Fact-style evidence-grounded fact checking \citep{rangapur2025finfact};
- Self-RAG-style self-reflective retrieval \citep{asai2024selfrag};
- CRAG-style corrective retrieval \citep{yan2024crag};
- FLARE-style confidence-triggered active retrieval \citep{jiang2023flare};
- Adaptive-RAG-style complexity routing \citep{jeong2024adaptiverag}.

These are adaptations, not official code releases. Each receives the same
point-in-time evidence; retrieval methods retrieve six earlier claims from
other companies using TF-IDF plus guidance metadata. The retrieval-based
variants see the same retrieved block; FLARE and Adaptive-RAG control when it
is used. The comparison therefore isolates the named decision/retrieval
strategy, not the base model.

## Main table (227 XBRL-verified claims)

Artifacts: `paper/results/named_baselines/metrics_*.json`,
`paper/results/named_baselines/metrics_summary.json`,
`paper/results/named_baselines/per_claim.jsonl`,
`paper/results/named_baselines/model_comparison_logistic.json`.

| Model | Acc. | Bal. | Macro F1 | MCC | AUROC | AP | Sel@50 | AURC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FinPlan (logistic) | .493 | .561 | .485 | .273 | .716 | .524 | .526 | .468 |
| FinDVer-style | .551 | .461 | .464 | .233 | .674 | .494 | .675 | .337 |
| Fin-Fact-style | .537 | .446 | .450 | .201 | .650 | .475 | .675 | .338 |
| Self-RAG-style | .546 | .451 | .456 | .216 | .670 | .491 | .684 | .333 |
| CRAG-style | .529 | .423 | .418 | .177 | .643 | .473 | .658 | .349 |
| FLARE-style | .511 | .417 | .411 | .164 | .634 | .467 | .596 | .362 |
| Adaptive-RAG-style | .493 | .406 | .400 | .143 | .584 | .425 | .579 | .415 |

No method dominates. FinDVer-style has the best accuracy; Self-RAG-style the
best selective accuracy and AURC; FinPlan the best balanced accuracy, macro F1,
MCC, AUROC, and AP. The LLM rows return a label, a confidence, and a credible
set, not a normalized three-class distribution; NLL, Brier, and ECE are not
comparable across rows and are not shown.

## Paired tests against FinPlan

Reference: FinPlan calibrated logistic, 227 claims.

| Model | Acc. difference | McNemar p | AUROC difference [95% CI] |
|---|---:|---:|---:|
| FinDVer | +.057 | .165 | -.043 [-.099, .004] |
| Fin-Fact | +.044 | .295 | -.067 [-.124, -.018] |
| Self-RAG | +.053 | .207 | -.047 [-.104, .003] |
| CRAG | +.035 | .440 | -.074 [-.127, -.021] |
| FLARE | +.018 | .734 | -.083 [-.141, -.034] |
| Adaptive-RAG | +.000 | 1.000 | -.132 [-.187, -.072] |

McNemar does not reject equal accuracy for any named baseline. Fin-Fact,
CRAG, FLARE, and Adaptive-RAG have lower AUROC with intervals excluding zero;
FinDVer and Self-RAG intervals include zero.

## Routing behavior

- FLARE: 181/227 claims triggered retrieval at confidence < .70; 46 kept the
  no-retrieval answer.
- Adaptive-RAG: 210/227 routed to simple (no retrieval), 17 to moderate
  (three retrieved claims), 0 to complex. The router is conservative, so the
  row is close to a no-retrieval reader.

## Per-class

| Model | Below P/R/F1 | Within P/R/F1 | Above P/R/F1 |
|---|---|---|---|
| FinPlan (logistic) | .29/.82/.43 | .63/.39/.48 | .63/.47/.54 |
| FinDVer | .56/.15/.24 | .46/.60/.52 | .63/.63/.63 |
| Fin-Fact | .50/.15/.23 | .46/.52/.49 | .60/.66/.63 |
| Self-RAG | .50/.15/.23 | .46/.52/.49 | .61/.68/.64 |
| CRAG | .43/.09/.15 | .45/.50/.47 | .59/.68/.63 |
| FLARE | .43/.09/.15 | .41/.56/.48 | .61/.60/.61 |
| Adaptive-RAG | .43/.09/.15 | .39/.57/.47 | .62/.55/.58 |

The named baselines trade minority-class recall for within/above performance;
FinPlan has high below recall but low below precision.

## Boundaries

- These are prompt-level adaptations under one base model, not official
  reproductions. The paper labels them `-style` in the text.
- FinRank, Fin-RATE, HC-RAG, and FinSAgent remain not run: no compatible
  released code or shared claim labels, and their published numbers are on
  different data.
- DeepSeek may have seen 2022--2025 outcomes; all LLM rows inherit the
  pretraining-contamination caveat.
- One deterministic seed (temperature 0) per named baseline; the FinPlan
  reference averages five fold seeds.
- The claim is not point-metric SOTA. The decision portfolio remains the
  selected method and is evaluated separately by decision regret.

## Commands

```text
PYTHONPATH="src;scripts" python scripts/run_named_baselines.py \
  --verified-only --variants findver,finfact,selfrag,crag,flare,adaptive \
  --workers 8 --out paper/results/named_baselines

for v in findver finfact selfrag crag flare adaptive; do
  PYTHONPATH="src;scripts" python scripts/convert_llm_predictions.py \
    --input paper/results/named_baselines/per_claim.jsonl \
    --out paper/results/named_baselines/prob_${v}.jsonl \
    --variant $v --verified-only --seed 1
  PYTHONPATH="src;scripts" python scripts/compute_metrics_suite.py \
    --per-claim paper/results/named_baselines/prob_${v}.jsonl \
    --out paper/results/named_baselines/metrics_${v}.json --no-bootstrap
done

PYTHONPATH="src;scripts" python scripts/compare_models.py \
  --inputs FinPlan_Logistic=paper/results/benchmark_per_claim_verified_calibrated/per_claim.jsonl \
    FinDVer=paper/results/named_baselines/prob_findver.jsonl \
    FinFact=paper/results/named_baselines/prob_finfact.jsonl \
    SelfRAG=paper/results/named_baselines/prob_selfrag.jsonl \
    CRAG=paper/results/named_baselines/prob_crag.jsonl \
    FLARE=paper/results/named_baselines/prob_flare.jsonl \
    AdaptiveRAG=paper/results/named_baselines/prob_adaptive.jsonl \
  --out paper/results/named_baselines/model_comparison_logistic.json
```
