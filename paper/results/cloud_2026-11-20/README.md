# Cloud three-backbone results (2026-11-20)

This directory contains the results transferred from the 32 GB cloud GPU run.
The code and data live in the repository; the large anchor caches are excluded.

## Main decision loss (227 verified claims)

| Backbone | Fusion | RF | LLM-only | Best named |
|---|---:|---:|---:|---:|
| DeepSeek Flash | 2.907 | 2.987 | 3.445 | 3.004 (CRAG-style) |
| GLM 5.3 Flash | 2.943 | 2.987 | 3.427 | 3.181 (Self-RAG-style) |
| Qwen3-8B | 2.925 | 2.987 | 5.498 | 4.881 (Fin-Fact-style) |

DeepSeek uses the validation-gated expanded type library; GLM and Qwen use the
base library. Fusion has the highest accuracy, balanced accuracy, macro-F1 and
lowest decision loss on all three backbones. Against the strongest baseline,
the DeepSeek and GLM paired intervals include zero; the Qwen interval excludes
zero. Against RF, all three intervals include zero.

## Full-action readout versus restricted gate

| Backbone | Fusion restricted | Fusion full | Delta [95% CI] |
|---|---:|---:|---:|
| DeepSeek | 3.656 | 2.907 | -0.749 [-1.075, -0.405] |
| GLM | 3.524 | 2.943 | -0.581 [-0.899, -0.264] |
| Qwen3 | 3.621 | 2.925 | -0.696 [-0.987, -0.405] |

RF moves from 3.454 to 2.987 (-0.467 [-0.767, -0.159]). All six intervals
exclude zero. This is the strongest result.

## Qualification ablation

| Backbone | Full | Accept-all | Delta [95% CI] |
|---|---:|---:|---:|
| DeepSeek | 3.304 | 3.507 | +0.203 [-0.106, +0.520] |
| GLM | 3.339 | 3.427 | +0.088 [-0.176, +0.361] |
| Qwen3 | 4.599 | 4.652 | +0.053 [-0.141, +0.256] |

The direction favors qualification, but none of the intervals excludes zero.

## A-share historical backtest (470 claims, 40 companies, 2023-2025)

| Method | n | Acc. | Bal. | Macro | Loss |
|---|---:|---:|---:|---:|---:|
| Prior-year carry-forward | 470 | .491 | .402 | .401 | 5.094 |
| Run-rate annualization | 470 | .777 | .786 | .785 | 2.270 |
| DeepSeek full | 470 | .783 | .778 | .776 | 2.179 |
| DeepSeek no bridge | 470 | .638 | .591 | .590 | 2.421 |
| GLM full | 470 | .549 | .485 | .483 | 3.332 |
| GLM no bridge | 470 | .549 | .485 | .483 | 3.332 |
| Qwen3 run-rate (250 subset) | 250 | .772 | .789 | .789 | 2.324 |
| Qwen3 full (250 subset) | 250 | .580 | .523 | .522 | 4.208 |
| Qwen3 no bridge (250 subset) | 250 | .564 | .500 | .499 | 3.864 |

DeepSeek full versus no bridge: loss -0.243 [-0.404, -0.077], accuracy +0.145
[0.109, 0.183]. Qwen3 full versus no bridge: accuracy +0.016 [0.004, 0.032],
loss +0.344 [0.064, 0.640]. GLM bridge does not fire.

## FinDVer component check (150 claims)

| Condition | DeepSeek acc | GLM acc | Qwen3 acc |
|---|---:|---:|---:|
| Fixed 4 paragraphs | .767 | .760 | .733 |
| Fixed 8 paragraphs | .847 | .860 | .700 |
| Gold evidence | .913 | .933 | .707 |
| State-conditioned | .873 | .913 | .660 |

State-conditioned acquisition uses about half the paragraphs of fixed-8 and
improves accuracy on DeepSeek and GLM; Qwen3 is better with fixed 4.

## No-anchor architecture ablation

Point-in-time LLM label scores, temperature scaling and the same full-action
readout, with all anchor objects removed:

| Backbone | Acc. | Macro-F1 | Loss |
|---|---:|---:|---:|
| DeepSeek | .524 | .398 | 3.286 |
| GLM | .463 | .347 | 3.260 |
| Qwen3 | .405 | .257 | 3.824 |

Compared with FinPlanRAG at .577/.530/2.907, .551/.504/2.943 and
.542/.501/2.925. See `no_anchor_ablation.json`.

## Anchor-type extension (DeepSeek)

Validation selection: base 3.943, type-only candidate 3.792, accepted. Test
accuracy .555 to .577, loss 2.934 to 2.907; paired difference -0.026
[-0.159, 0.097].

## Within-range stratum

On the 82 test claims whose realized outcome is inside the guidance range
(first RF seed):

| Backbone | Fusion | RF | Delta [95% CI] | Seeds excluding zero |
|---|---:|---:|---:|---:|
| DeepSeek | 2.780 | 3.146 | -0.366 [-0.732, -0.049] | 5/5 |
| GLM | 2.902 | 3.146 | -0.244 [-0.537, -0.024] | 4/5 |
| Qwen3 | 2.732 | 3.146 | -0.415 [-0.732, -0.170] | 5/5 |

This is an error-stratified result. The realized class is unknown at
prediction time, so it is not a routing rule.

## Files

- `deepseek/`, `glm/`, `qwen3/`: per-backbone fusion, named, official,
  qualification and A-share outputs.
- `metrics_table.json`: comparable metrics for every method.
- `readout_consistency.json`: paired readout comparisons under one action space.
- `within_range_all_backbones.json`: seed-level within-range check.
- `rf/`: structured RF benchmark.
- `summary.json`, `REPORT.md`: aggregate tables.
