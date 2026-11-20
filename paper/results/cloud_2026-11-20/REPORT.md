# Cloud three-backbone results

## Full-action decision loss

| Backbone | Fusion (base library) | RF | LLM-only | Best named | Best official |
|---|---:|---:|---:|---:|---:|
| deepseek | 2.934 | 2.987 | 3.445 | CRAG-style 3.004 | Fin-Fact-style 3.207 |
| glm | 2.943 | 2.987 | 3.427 | Self-RAG-style 3.181 | FLARE-style 3.648 |
| qwen3 | 2.925 | 2.987 | 5.498 | Fin-Fact-style 4.881 | not run |

DeepSeek main-table result uses the validation-gated expanded type library:
fusion 2.907, accuracy .577, balanced .526, macro-F1 .530.

## Qualification ablation

| Backbone | Full | Accept-all | Δ [95% CI] |
|---|---:|---:|---:|
| deepseek | 3.304 | 3.507 | +0.203 [-0.106, +0.520] |
| glm | 3.339 | 3.427 | +0.088 [-0.176, +0.361] |
| qwen3 | 4.599 | 4.652 | +0.053 [-0.141, +0.256] |

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
