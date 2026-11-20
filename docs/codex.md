# 在 Codex CLI 中使用 FinPlanRAG

这个接口把同一个 AnchorEvolve MCP 服务接到 OpenAI Codex CLI。Codex 和 Claude Code 调用的是 `scripts/run_anchor_mcp.py`，因此两边生成同样的审计文件。区别只在客户端配置：Claude Code 读 `.mcp.json`，Codex 读 `.codex/config.toml`。

## 1. 项目内直接使用

仓库根目录已经提供：

```text
.codex/config.toml
AGENTS.md
```

`.codex/config.toml` 注册 `finplan-anchor`：

```toml
[mcp_servers.finplan-anchor]
command = "python"
args = ["scripts/run_anchor_mcp.py"]
cwd = "."
startup_timeout_sec = 30
tool_timeout_sec = 900
enabled = true
supports_parallel_tool_calls = false
default_tools_approval_mode = "approve"

[mcp_servers.finplan-anchor.env]
PYTHONIOENCODING = "utf-8"
PYTHONUTF8 = "1"
```

在仓库根目录启动 Codex：

```bash
codex
```

进入会话后可以用 `/mcp` 查看服务器，或者先在终端确认：

```bash
codex mcp list
codex mcp get finplan-anchor
```

Codex 会把工具名加上服务器前缀：

```text
mcp__finplan-anchor__anchor_ask
mcp__finplan-anchor__anchor_case
mcp__finplan-anchor__anchor_search
mcp__finplan-anchor__anchor_audit
mcp__finplan-anchor__anchor_contract
```

## 2. 直接对 Codex 说的话

运行中际旭创案例：

```text
用 mcp__finplan-anchor__anchor_case 运行 zhongji_2026，
截止时点 2026-11-19。读取审计报告，逐条说明哪些锚点被采纳、
哪些被拒绝，以及最终为什么选择点答案、裁决集或弃权。
不要自己补充没有出现在证据链里的数字。
```

运行新易盛案例：

```text
用 mcp__finplan-anchor__anchor_case 运行 xinyisheng_2026，
截止时点 2026-11-19。分别解释营收增速、产能变化和披露日期
三个断言的锚点、数值桥和最终动作。
```

先检查契约再运行：

```text
用 mcp__finplan-anchor__anchor_contract 把
“中际旭创2026年年报营业收入同比增速”转成契约。
先只返回标签、阈值、参考值和查询面，不要运行完整裁决。
```

读取已有运行的审计：

```text
用 mcp__finplan-anchor__anchor_audit 读取 run_id
20261120-014606-xinyisheng_2026-2079，并总结证据缺口。
```

## 3. Codex 自己的模型与 MCP 内部模型

推荐用法是：Codex 使用它当前配置的模型负责对话和工具编排，MCP 服务内部继续用 DeepSeek 完成问题解析和锚点抽取。这条路径不需要修改 Codex 的模型提供方，也不需要把 DeepSeek key 暴露给 Codex 配置。MCP 包装器会从仓库根目录的 `.env` 读取 `ANTHROPIC_AUTH_TOKEN`。

如果希望 Codex 本身也走 DeepSeek，需要把 provider 配在用户级 `~/.codex/config.toml`，因为项目级配置会过滤掉模型提供方和 base URL 这类机器级设置。示例：

```toml
model = "deepseek-chat"
model_provider = "deepseek"

[model_providers.deepseek]
name = "DeepSeek"
base_url = "https://api.deepseek.com"
env_key = "DEEPSEEK_API_KEY"
wire_api = "responses"
```

注意两点。第一，当前 Codex 只接受 Responses API，`wire_api = "chat"` 会报错。第二，DeepSeek 的 Responses 兼容性取决于账号和模型；如果直连返回 404，保留 Codex 默认模型，让 MCP 服务内部调用 DeepSeek，或者使用本地 Responses 桥接。不要把 key 写进 `.codex/config.toml`，用 `env_key` 指向环境变量。

## 4. 与 Claude Code 的对应关系

| 项目 | Claude Code | Codex |
|---|---|---|
| MCP 配置 | `.mcp.json` | `.codex/config.toml` |
| 服务器类型 | `stdio` | `stdio` |
| 启动命令 | `python scripts/run_anchor_mcp.py` | 同上 |
| 项目说明 | 无自动读取的 `AGENTS.md` | `AGENTS.md` |
| 工具名 | `anchor_case` 等 | `mcp__finplan-anchor__anchor_case` 等 |
| 审批 | 由 Claude Code 权限模式控制 | `default_tools_approval_mode` 控制 |

## 5. 常见问题

**`/mcp` 里看不到 finplan-anchor。** 确认是在仓库根目录启动 Codex，并且项目已被信任。Codex 只有在项目受信任时才加载项目级 `.codex/config.toml`。也可以先运行 `codex mcp list` 检查。

**工具调用被取消。** 非交互式 `codex exec` 默认会等待审批。项目配置里已经给这五个只读/审计工具设置 `default_tools_approval_mode = "approve"`；如果仍然被取消，检查是否有用户级配置覆盖了它。

**找不到 `python`。** Windows 上通常是 `python`，部分环境是 `py -3` 或 `python3`。把 `.codex/config.toml` 的 `command` 改成当前环境的解释器路径。

**没有 DeepSeek key。** 在仓库根目录创建 `.env`，写入 `ANTHROPIC_AUTH_TOKEN`、`ANTHROPIC_BASE_URL` 和 `ANTHROPIC_MODEL`；或者在启动 Codex 前导出同名环境变量。不要把 key 写进 `AGENTS.md`、`.codex/config.toml` 或任何提交文件。

**修改代码后行为没有变化。** Codex 启动时加载 MCP 进程。修改 `anchor_engine.py`、提示词或案例后，退出并重新启动 Codex。

**案例运行很慢。** 一个案例可能检索多份公告并多次调用模型。`tool_timeout_sec = 900` 给了足够时间；如果网络检索不可用，审计会显式记录证据缺口，不会用标准答案回填。

## 6. 验证命令

```bash
# 配置是否被 Codex 读到
codex mcp list
codex mcp get finplan-anchor

# 不经过 Codex 的 MCP 烟测
python scripts/anchor_cli.py case xinyisheng_2026 --cutoff 2026-11-19

# 计划校验
python scripts/check_benchmark_plan.py \
  --report paper/results/benchmark_plan/validation.json
```

当前已经验证：`codex mcp list` 能看到 `finplan-anchor`；`codex exec` 成功调用 `anchor_contract` 并返回中际旭创营收增速契约；`anchor_case` 成功运行新易盛 2026 案例，run_id 为 `20261120-021829-xinyisheng_2026-5990`，审计目录在 `paper/results/anchor_cli/20261120-021829-xinyisheng_2026-5990`。交互式 Codex 会话更适合阅读长审计报告。
