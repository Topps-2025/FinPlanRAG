# 云 GPU 实验方案

这版把实验迁到一台 32 GB 显存的云 GPU 上。三个基座模型是：

- DeepSeek Flash，走 Anthropic 兼容 API；
- GLM 5.3 Flash，走智谱 API；
- Qwen3-8B，在云 GPU 上用 vLLM 本地部署。

这样只有 Qwen3-8B 占显存，DeepSeek 和 GLM 不占 GPU。32 GB 对 8B 模型足够，但需要控制 `max-model-len` 和并发数。

## 1. 云主机准备

建议镜像：Ubuntu 22.04，Python 3.10，CUDA 12.x，单卡 32 GB。

```bash
git clone <repo-url> FinPlanRAG
cd FinPlanRAG
python -m venv .venv
source .venv/bin/activate
pip install -e ".[agent]"
pip install -r requirements-cloud.txt
```

`requirements-reproduce.txt` 固定了论文使用的 scikit-learn、scipy 和 numpy 版本。vLLM 的 wheel 与 CUDA/PyTorch 版本相关，建议使用云镜像预装的 vLLM，或按 vLLM 官方索引安装对应 CUDA 版本，并把实际版本写进运行记录。

## 2. 环境变量

不要把 key 写进代码、YAML、日志或 Git。云主机上导出：

```bash
export ANTHROPIC_AUTH_TOKEN="<deepseek-key>"
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_MODEL="deepseek-flash[1m]"

export GLM_API_KEY="<glm-key>"

# vLLM 的 OpenAI 兼容接口不校验 key，但客户端要求变量存在。
export QWEN_API_KEY="dummy"
```

如果不想用 shell export，可以在仓库根目录放 `.env`，但必须确认 `.gitignore` 已经忽略它。MCP 包装器和 CLI 会读取 `.env`，云端批处理脚本只读环境变量。

## 3. 启动 Qwen3-8B

```bash
bash scripts/serve_qwen3_vllm.sh
```

默认参数：

```text
model: Qwen/Qwen3-8B
served name: qwen3-8b
port: 8000
dtype: bfloat16
max-model-len: 32768
gpu-memory-utilization: 0.90
max-num-seqs: 16
```

32 GB 显存的经验值：

- 8B bf16 权重约 16 GB，剩余空间给 KV cache 和激活。
- 如果启动时报 OOM，先把 `MAX_MODEL_LEN` 降到 16384，或把 `GPU_MEMORY_UTILIZATION` 降到 0.85。
- 如果并发高时 OOM，把 `MAX_NUM_SEQS` 降到 8。
- 如果需要更长上下文，优先增加显存而不是强行拉高 `max-model-len`。

健康检查：

```bash
curl http://127.0.0.1:8000/v1/models
```

应返回 `qwen3-8b`。

Qwen3 默认可能输出 thinking trace。客户端通过：

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

关闭 thinking。`configs/cloud/backbones.yaml` 已经把这个参数写进 `extra_body`，锚点抽取、具名基线和 A 股回测都会带上。

## 4. 数据准备

需要上传或生成：

```text
paper/data/dataset_v4.json
paper/data/earnings_events.json
paper/data/filings/                 # 8-K 文本缓存，若使用 run_anchors
paper/data/ashare_backtest/claims.json
configs/
experiments/
scripts/
```

A 股回测数据可以云端重新生成：

```bash
python scripts/build_ashare_backtest.py
```

当前是 5 家公司、40 条 smoke 数据。正式实验前把 `configs/ashare_backtest.yaml` 扩到 30 至 50 家公司，目标 250 至 400 条。

## 5. 一键生成命令

先 dry run，不调用任何模型：

```bash
python scripts/run_cloud_suite.py --dry-run
```

它会：

- 从 `configs/cloud/backbones.yaml` 读取三个基座；
- 生成 `paper/results/cloud/configs/{backbone}_evolution.yaml`；
- 生成 `paper/results/cloud/configs/{backbone}_anchor_cli.yaml`；
- 把全部命令写到 `paper/results/cloud/commands.sh`；
- 写 `paper/results/cloud/manifest.json`，包含环境检查和命令清单。

确认命令后执行：

```bash
python scripts/run_cloud_suite.py --execute
```

执行前会检查：

- DeepSeek 和 GLM 的 key 是否存在；
- Qwen 的 vLLM 是否在 `127.0.0.1:8000` 可访问。

## 6. 分阶段执行

不建议一次跑全量。按阶段跑，失败可以从断点继续。

| 阶段 | 内容 | 输出 |
|---|---|---|
| `anchors` | 每个基座抽取验证集和测试集锚点 | `paper/results/cloud/{backbone}/evolution/anchors_*.jsonl` |
| `rf` | 结构化 RF 基线，与基座无关，只跑一次 | `paper/results/cloud/rf/` |
| `fusion` | 锚点后验、RF、静态/演化可靠度、全动作读出 | `paper/results/cloud/{backbone}/evolution/temporal_fusion_cloud.json` |
| `named` | 六个具名 RAG/核验方法 | `paper/results/cloud/{backbone}/named/` |
| `official` | 官方提示词/判定逻辑适配 | `paper/results/cloud/{backbone}/official/` |
| `qualification` | 资格判定消融 | `paper/results/cloud/{backbone}/qualification/` |
| `ashare` | A 股历史回测的完整系统和去桥系统 | `paper/results/cloud/{backbone}/ashare/` |

示例：

```bash
# 先跑 20 条 smoke，确认链路
python scripts/run_cloud_suite.py --execute --limit 20 \
  --stages anchors,named,official --backbones qwen3

# 再跑全量锚点和融合
python scripts/run_cloud_suite.py --execute \
  --stages anchors,rf,fusion

# 最后跑基线和 A 股回测
python scripts/run_cloud_suite.py --execute \
  --stages named,official,qualification,ashare
```

`--limit` 会传给支持它的脚本。A 股回测的 `--limit` 也已接入。

## 7. 运行量与缓存

粗略规模：

- 主基准测试 227 条，验证 203 条；锚点抽取约 430 条/基座。
- 六个具名方法各 227 条，约 1,362 次调用/基座。
- 六个官方适配各 227 条，约 1,362 次调用/基座。
- A 股回测 40 条 smoke，每条 `full` 和 `no_bridge` 各一次，约 80 次调用/基座。

实际调用量取决于缓存命中。所有客户端都有 LLM cache：

```text
paper/results/cloud/{backbone}/llm_cache/
paper/results/cloud/{backbone}/anchor_cache/
```

重跑时先复用缓存，不要清空。API 并发建议：

- DeepSeek：`workers=8`
- GLM：`workers=6`
- Qwen3-8B：`workers=8`，`max-num-seqs=16`

如果 API 返回限流，降低 workers，不要修改 prompt 或标签。

## 8. 输出目录

```text
paper/results/cloud/
├── configs/                 # 自动生成的三套配置
├── commands.sh              # dry-run 生成的命令
├── manifest.json            # 环境、命令、问题清单
├── rf/                      # 结构化 RF
├── deepseek/
│   ├── evolution/
│   ├── named/
│   ├── official/
│   ├── qualification/
│   └── ashare/
├── glm/
└── qwen3/
```

论文表格从 `paper/results/cloud/*/.../summary.json` 和 `metrics_summary.json` 读取。不要把三个基座的结果混成一个平均数，按基座分别报告。

## 9. 常见问题

**vLLM 启动 OOM。** 降低 `MAX_MODEL_LEN` 到 16384，或 `GPU_MEMORY_UTILIZATION` 到 0.85，或 `MAX_NUM_SEQS` 到 8。

**Qwen 输出 thinking。** 检查请求里是否有 `chat_template_kwargs.enable_thinking=false`。`configs/cloud/backbones.yaml` 已经配置；如果手工调用，记得带上。

**`/v1/models` 能通但生成很慢。** 32 GB 单卡上 8B 模型可用，但并发不要开太高。先 `--limit 20` 看单条延迟，再放大。

**GLM key 缺失。** `run_cloud_suite.py` 在执行前会报错。导出 `GLM_API_KEY` 后重跑。

**DeepSeek 走错端点。** DeepSeek 在 Codex/Claude Code 里用的是 Anthropic 兼容端点 `https://api.deepseek.com/anthropic`；在 OpenAI 兼容脚本里不能直接用这个端点。`configs/cloud/backbones.yaml` 已按 provider 分开。

**重跑后结果不一致。** 检查是否清空了缓存、是否换了模型版本、是否改了温度。所有实验温度固定为 0.0，但 API 端仍可能有细微非确定性，报告时保留运行日期和模型版本。

## 10. 本地与云端的分工

本地只保留：

- 代码修改和小样本 smoke；
- 基准计划校验；
- 确定性基线（`last_year`、`run_rate`）；
- 论文写作。

云端负责：

- 三个基座的锚点抽取；
- 六个具名基线和六个官方适配；
- 资格判定消融；
- A 股历史回测的完整系统；
- 所有需要 API 或 GPU 的全量实验。

本地已经验证过 DeepSeek + MCP 的完整工作流；云端只需要把同一套脚本按基座配置重跑，不需要改算法代码。
