# 基准扩充、具名基线与公平性方案

这份文档说明论文实验部分的基准组合、每个基准能够证明什么、不能证明什么，以及具名基线如何在同一接口下复现。它不记录新的实验结果，也不把外部基准的结果并入主结果。机器可读的版本在 `configs/benchmark_plan.yaml`，结构检查脚本是 `scripts/check_benchmark_plan.py`。

## 1. 为什么需要扩充基准

论文的主张不是“在某一个案例上给出一个答案”，而是“在结果尚未披露时，锚点层、资格判定、时序演化和全动作裁决能够形成一个可审计的判断”。这个主张至少需要三类证据：

1. 在规模足够、结果可验证的主基准上，比较完整系统与具名基线、结构基线、消融系统。
2. 在时间切分和公司切分下，确认结论不是由某一个年份或某几家公司驱动。
3. 在外部声明核验和长文档问答基准上，确认证据资格判定与阅读器接口可以接入公开任务；这些任务不用于证明前瞻性结果预测能力。

A 股前瞻案例单独作为预注册检验。它的作用不是提供统计显著性，而是在结果尚未发生时就冻结断言、阈值和证据截止日，等到 2027 年年报披露后再判定。这样，案例研究不再是“看到结果后解释”，而是可复核的前瞻记录。

## 2. 基准矩阵

| 基准 | 角色 | 数据与规模 | 任务与标签 | 能证明 | 不能证明 |
|---|---|---|---|---|---|
| `revenue_guidance_main` | 主结果基准 | `paper/data/dataset_v4.json`，1,005 条断言；训练 529、验证 203、测试 273；测试中 XBRL 可验证 227 条 | 季度营业收入相对指引区间：低于、位于、高于 | 锚点增量、资格判定与演化、全动作裁决、与 RF 的竞争力 | 不能证明对所有行业和所有指标都成立 |
| `revenue_guidance_temporal` | 时间稳健性 | 同一数据，按 2022、2023、2024、2025 逐年评估 | 同上 | 结论不依赖单一年份；演化只使用披露日早于截止日的旧结果 | 不能替代跨市场外部验证 |
| `findver_component` | 外部组件基准 | FinDVer 公开测试集，需下载 | 声明核验：entailed、refuted、insufficient | 证据资格判定、引文核验、证据召回 | 标签空间不是 below/within/above，不能当作结果预测基准 |
| `finfact_component` | 外部组件基准，适配待补 | Fin-Fact 公开数据，需下载；当前只有主基准上的 `finfact_style` 具名基线 | 多模态金融事实核验 | 接入后可以检验证据驱动的核验接口和解释链 | 当前没有数据集适配器，不能把主基准上的 `finfact_style` 结果当作 Fin-Fact 数据集结果 |
| `docfinqa_control` | 确定性问答控制 | DocFinQA 固定 300 题样本，需下载 | 长文档抽取式问答 | 阅读器、证据预算和检索接口没有因框架改造而失效 | 答案已在文档中，不能证明前瞻判断能力 |
| `ashare_annual_backtest_2023_2025` | 历史结果基准 | `paper/data/ashare_backtest/claims.json`，当前 40 条 smoke，目标 250 至 400 条 | A 股年报收入增速、归母净利润增速、毛利率、披露日期 | 数值桥、非收入指标、跨市场复现、资格判定 | 当前样本较小，不能替代主基准的统计功效 |
| `ashare_prospective_2026` | 前瞻预注册 | 中际旭创、新易盛 2026 年报断言，截止日 2026-11-19 | 收入增速、毛利率、产能变化、披露日期 | 数值桥、证据可审计性、结果未披露时的完整工作流 | 两个案例没有统计功效，不能单独支撑总体准确率主张 |

主基准的 273 条测试断言全部进入更严格的评估协议：每家公司按五折划分，评估某一年时，训练集只包含其他折中结果披露日早于该年 1 月 1 日的断言。这样既保证公司不交叉，也保证时间不泄漏。测试集中有 108 条来自训练期完全未出现的公司，通常单独报告；227 条是 XBRL 可验证子集，作为标签噪声较小的主表。验证集 203 条中 185 条可验证，温度和融合权重只在验证集上选择。

## 3. 主基准与时间稳健性

主基准的任务是判断实际收入落在公司指引区间的哪一侧。这个任务适合检验锚点层，因为证据不是单一数字，而是公司指引、历史实际、行业趋势、分析师预期和披露文本的组合。标签只有三类，避免把结论建立在细粒度数值误差上。

时间稳健性基准不改变任务，只改变评估切片。每个测试年份单独报告准确率、平衡准确率、宏 F1、决策损失、ECE、Brier、裁决覆盖率、平均集合大小和弃权率。这样可以看出某一年份的市场环境是否让方法失效，而不是只报告一个混合平均数。

主基准的泄漏控制包括：

- 只使用披露日不晚于截止日的证据。
- 只使用公开披露日早于截止日的结果更新先验和锚点类型可靠度。
- 测试公司不进入任何训练折。
- 融合权重、温度、阈值只在验证集选择。
- 解析失败、检索为空、证据缺失都计入结果，不用标准答案回填。

## 4. 外部组件基准的接入边界

FinDVer 的标签空间是声明核验，不是结果预测。本文把它的文档、声明和证据段落映射到锚点资格判定接口，输出 entailed、refuted、insufficient。这样可以检验三件事：引文是否能在原文中找到，证据日期是否满足截止日约束，资格判定是否与人工证据状态一致。适配入口是 `scripts/run_findver_external.py`，公开数据需要下载到 `vendor/official_baselines/findver/`。

Fin-Fact 的情况不同。它目前只作为主基准上的一个具名基线 `finfact_style` 使用，数据集本身还没有适配器。计划中保留 `finfact_component` 条目是为了记录接入目标，不能把主基准上的 `finfact_style` 结果写成 Fin-Fact 数据集的结果。补齐适配器之前，这一项不进入组件结果表。

这类基准不能回答“2026 年收入会高于还是低于指引”。因此论文中只把它们放在组件有效性部分，不与主基准合并，也不把核验准确率写成结果预测准确率。外部数据的路径、下载来源和适配脚本在计划中单独标注；当前仓库没有存放这些公开数据，运行前需要下载到 `vendor/official_baselines/` 下。

DocFinQA 是确定性问答控制。它的答案是文档中已经存在的事实，阅读器和检索接口必须在这种任务上保持正常。这个控制的作用是排除一种替代解释：主基准上的差异不是因为框架把阅读器改坏了。适配入口是 `scripts/run_docfinqa_external.py`，它同样不进入主结果表。

## 5. A 股历史回测与前瞻预注册

### 5.1 历史回测

`ashare_annual_backtest_2023_2025` 用已披露年报构造有 ground truth 的样本。对每个公司年度 Y，问题在 Y+1 年 1 月 31 日提出，此时三季报已公开、年报尚未披露。`scripts/build_ashare_backtest.py` 从公开财务报表读取收入、归母净利润和毛利率，从巨潮资讯网读取年报披露日期和公告链接。当前 smoke 数据包含 5 家公司、2024 和 2025 两个年度、40 条断言；目标是扩展到 30 至 50 家公司、250 至 400 条。

每条断言只使用截止日前的证据：

- 上年年报的收入、利润和毛利率；
- 目标年度三季报的累计值；
- 上年三季报的累计值；
- 历史年报披露日期。

标签在读取结果前固定：收入增速以上年增速为中心加减 10 个百分点，净利润增速加减 15 个百分点，毛利率以上年毛利率为中心加减 1 个百分点，披露日期按 3 月 31 日、4 月 15 日两个窗口切分。结果标签和数值只在回测评估时读取，不进入证据或提示词。

这个基准检验 `growth_bridge`、`margin_bridge`、资格判定和披露日期预测，并提供 `last_year_actual`、`run_rate`、`ashare_full_system`、`ashare_no_bridge` 四个可复现基线。当前确定性基线已经可以运行：

```bash
python scripts/build_ashare_backtest.py
python scripts/run_ashare_backtest.py --methods last_year,run_rate
```

### 5.2 前瞻预注册

两个案例在 2026-11-19 冻结：

- 中际旭创：2026 年营业收入同比增速、归母净利润同比增速、综合毛利率、年报披露日期。
- 新易盛：2026 年营业收入同比增速、光互联产品产能相对 2026 年半年报的变化、年报披露日期。

每个断言的标签、阈值、参考值和证据查询在 `configs/anchor_cases/` 中固定。阈值来自已披露的 2025 年实际值、公司公开目标和法定披露窗口，不在看到 2026 年报后调整。结果在 2027 年年报披露后判定。

这个基准能够证明的是：系统在结果未知时就能给出点答案、裁决集合或弃权，并且每个锚点都能回溯到原文、日期和资格判定。它不能证明总体准确率，因为案例数太少。论文中应把它写成前瞻可行性检验，而不是统计结论。

## 6. 具名基线与复现接口

所有基线共享同一基座模型、同一截止日、同一证据预算和同一标签空间。外部方法的原始代码、提示词和决策规则来自固定提交；无法直接复用的部分在 `paper/results/official/deepseek/provenance.json` 中逐项记录。命名基线是“同接口适配”，不是“原系统原样运行”，论文中必须这样表述。

| 基线 | 作用 | 复现入口 | 已有产物 |
|---|---|---|---|
| `llm_only` | 无锚点的同一基座 | `scripts/run_deepseek_baselines.py` | `paper/results/deepseek` |
| `anchor_llm` | 只用锚点后验 | `experiments/evolution/run_anchors.py` | `paper/results/evolution_v1/temporal_beliefs_new.jsonl` |
| `anchor_fusion` | 锚点与 RF 融合 | `experiments/evolution/temporal_fusion.py` | `paper/results/evolution_v1/temporal_fusion_new.json` |
| `full_system` | 锚点、RF、桥、校准、全动作裁决 | `experiments/evolution/temporal_fusion.py` | `paper/results/evolution_v1/meta_readout.json` |
| `no_anchors` | 去掉锚点 | `scripts/analyze_anchor_ablation.py` | `paper/results/evolution_v1` |
| `old_prompt` | 旧结构化证据提示词 | `experiments/unified/run_vague_llm.py` | `paper/results/evolution_v1` |
| `static_reliability` | 固定可靠度 | `experiments/evolution/temporal_fusion.py` | `paper/results/evolution_v1/temporal_beliefs_new_static.jsonl` |
| `evolved_reliability` | 按披露日演化 | `experiments/evolution/temporal_fusion.py` | `paper/results/evolution_v1/temporal_beliefs_new.jsonl` |
| `restricted_gate` | 点答案或最高密度可信集 | `experiments/evolution/temporal_fusion.py` | `paper/results/evolution_v1` |
| `full_action_readout` | 点答案、裁决集、弃权 | `experiments/evolution/temporal_fusion.py` | `paper/results/evolution_v1/meta_readout.json` |
| `rf` | 结构随机森林 | `scripts/run_company_disjoint_benchmark.py` | `paper/results/company_disjoint_benchmark_verified/summary.json` |
| `evolved_rf` | 带时点特征的 RF | `experiments/evolution/structured_evolve.py` | `paper/results/evolution_v1/rf_test_ext.jsonl` |
| `no_bridge` | 去掉数值桥 | `scripts/anchor_cli.py --no-bridge` | `paper/results/anchor_cli` |
| `full_qualification` | 完整五类资格判定 | `scripts/run_qualification_ablation.py` | `paper/results/qualification_ablation/summary.json` |
| `no_qualification` | 接受全部锚点 | `scripts/run_qualification_ablation.py` | `paper/results/qualification_ablation/summary.json` |
| `no_source_check` | 去掉来源检查 | `scripts/run_qualification_ablation.py` | `paper/results/qualification_ablation/summary.json` |
| `last_year_actual` | 上年值延续 | `scripts/run_ashare_backtest.py` | `paper/results/ashare_backtest/last_year` |
| `run_rate` | 三季报年化 | `scripts/run_ashare_backtest.py` | `paper/results/ashare_backtest/run_rate` |
| `ashare_full_system` | A 股回测完整系统 | `scripts/run_ashare_backtest.py` | `paper/results/ashare_backtest/full` |
| `ashare_no_bridge` | A 股回测去桥 | `scripts/run_ashare_backtest.py` | `paper/results/ashare_backtest/no_bridge` |
| `findver_style` | FinDVer 提示词与判定 | `scripts/run_official_baselines.py` | `paper/results/official/deepseek/metrics_findver.json` |
| `finfact_style` | Fin-Fact 提示词与判定 | `scripts/run_official_baselines.py` | `paper/results/official/deepseek/metrics_finfact.json` |
| `selfrag_style` | Self-RAG 检索反思 | `scripts/run_named_baselines.py` | `paper/results/named_baselines/metrics_selfrag.json` |
| `crag_style` | CRAG 纠错检索 | `scripts/run_named_baselines.py` | `paper/results/named_baselines/metrics_crag.json` |
| `flare_style` | FLARE 低置信触发检索 | `scripts/run_named_baselines.py` | `paper/results/named_baselines/metrics_flare.json` |
| `adaptiverag_style` | Adaptive-RAG 复杂度路由 | `scripts/run_named_baselines.py` | `paper/results/named_baselines/metrics_adaptive.json` |
| `fixed_k_reader` | 固定证据预算阅读器 | `scripts/run_docfinqa_external.py` | `paper/results/external_tier1/docfinqa/test300/deepseek` |
| `fixed_k_verifier` | 固定证据预算核验器 | `scripts/run_findver_external.py` | `paper/results/external_tier1/findver/testmini/deepseek` |

`--no-bridge` 与 `--no-calibration` 是命令行显式消融开关，不修改基础配置，也不改变其他模块。这样 `no_bridge` 和 `no_calibration` 的差异只来自被关闭的模块。

## 7. 指标与主张的对应

| 主张 | 主要指标 | 比较方式 | 解释边界 |
|---|---|---|---|
| H1 锚点增量 | 准确率、平衡准确率、宏 F1 | `anchor_llm` 对 `no_anchors`、`old_prompt` | 说明锚点层有增量，不说明每个行业都有效 |
| H2 资格与演化 | 决策损失、NLL、ECE、宏 F1、接受率、引文核验率 | `full_qualification` 对 `no_qualification`、`no_source_check`；`evolved_reliability` 对 `static_reliability` | 资格判定和演化都要单独拆开；演化增益可能较小，不能写成主要驱动 |
| H3 全动作裁决 | 决策损失、裁决覆盖率、集合大小、弃权率 | `full_action_readout` 对 `restricted_gate` | 同一后验下的严格改进，不涉及后验本身 |
| H4 与 RF 的竞争力 | 决策损失、准确率、平衡准确率 | `full_system` 对 `rf`、`evolved_rf` | 竞争力而非全面优越；RF 是很强的决策头 |
| 数值桥 | MAE、RMSE、MAPE、pinball loss、区间覆盖率、缺失输入率 | `ashare_full_system` 对 `ashare_no_bridge`、`run_rate`、`last_year_actual`；前瞻案例作为补充 | 历史 A 股回测提供已实现结果；前瞻案例只检验工作流，不提供准确率 |
| 可审计性 | 引文核验率、证据缺口率、来源日期合规率、审计完整率 | `full_system` 对 `llm_only` | 过程指标，不等于预测准确率 |

外部基准的指标单独报告：核验准确率、宏 F1、证据召回、引文核验率、资格一致性和 ECE。它们不进入主结果表，也不与主基准做加权平均。

## 8. 公平性协议

1. 同一基座、同一提示词模板、同一证据预算、同一标签空间。不同方法之间只改变被检验的模块。
2. 所有超参数只在验证集选择。测试集只评估一次，不因结果反复调整。
3. 主基准使用公司不交叉、年份不交叉的折分。每家公司只出现在一个折中，测试年份只使用此前已披露的结果。
4. 外部方法的提示词、判定规则和决策逻辑来自固定提交。替代组件在 provenance 中记录，不能把适配结果写成原系统成绩。
5. 解析失败、检索失败和证据缺失都保留在分母中。不能用标准答案、人工补写或 oracle 回退。
6. 报告全部具名基线，不只报告最弱的一个。配对 bootstrap 使用 10,000 次重采样，分类差异使用 McNemar 检验。
7. 外部组件基准不与主基准合并。主基准也不与前瞻案例合并。
8. 前瞻案例的阈值和标签在结果披露前冻结。结果披露后不修改阈值，不更换案例。

## 9. 当前可接入状态

本地主基准数据、事件日期、已有基线产物和演化实验结果都在仓库中。`scripts/check_benchmark_plan.py` 会检查配置完整性、基线注册、指标注册、数据路径、适配器路径、折分字段、前瞻案例阈值和日期窗口。当前检查结果是 7 个基准、6 个主张、32 个具名基线、34 个指标、0 个结构错误；警告包括外部数据未下载、Fin-Fact 适配器未实现、A 股历史回测当前只有 40 条 smoke 数据（目标 250 条）。

外部数据尚未放入仓库：

- FinDVer：需要下载公开测试集到 `vendor/official_baselines/findver/`；适配器 `scripts/run_findver_external.py` 已存在。
- Fin-Fact：需要下载公开数据到 `vendor/official_baselines/finfact/`；数据集适配器尚未实现，当前只有主基准上的 `finfact_style` 具名基线。
- DocFinQA：需要下载公开数据；适配器 `scripts/run_docfinqa_external.py` 已存在，本地已有 300 题样本的运行产物，但没有原始数据。

因此，当前阶段可以确认“主基准与 FinDVer/DocFinQA 接口可接入、具名基线有复现入口、指标与主张对应”，但不能声称 Fin-Fact 数据集或全部外部基准已经跑通。外部数据下载后再执行对应脚本，结果仍按组件基准单独报告。

## 10. 不能从这套基准得到的结论

- 不能用 FinDVer 或 Fin-Fact 的核验准确率证明前瞻结果预测能力。
- 不能用 DocFinQA 的抽取式问答成绩证明模糊分支的裁决能力。
- 不能用两个 A 股案例证明总体准确率或统计显著性。
- 不能把 `full_system` 与 `rf` 的差异写成全面优越。现有证据支持的是决策损失上的竞争力，以及锚点层提供的可审计状态。
- 不能把“演化”写成主要增益来源。现有结果显示演化方向一致但幅度较小，主要显著改进来自全动作裁决。
- 不能把同接口适配的具名基线写成原系统的官方成绩。

## 11. 执行清单

- [x] 主基准、时间稳健性、外部组件、确定性控制和前瞻案例写入 `configs/benchmark_plan.yaml`。
- [x] 32 个具名基线登记复现入口和已有产物。
- [x] A 股历史回测 builder、smoke 数据和确定性基线接入。
- [x] 6 个主张绑定基准、比较对象和指标。
- [x] 新易盛 2026 前瞻案例写入 `configs/anchor_cases/xinyisheng_2026.yaml`。
- [x] `--no-bridge` 与 `--no-calibration` 消融开关接入 CLI。
- [x] 结构检查脚本和报告生成。
- [ ] 下载 FinDVer、Fin-Fact、DocFinQA 公开数据。
- [ ] 补齐 Fin-Fact 数据集适配器；在此之前不使用 `finfact_component` 结果。
- [ ] 在主基准上运行全部基线并生成配对区间。
- [ ] 2027 年年报披露后判定前瞻案例。
- [ ] 如需要对毛利率和产能做历史校准，单独训练对应指标类型的校准包；当前校准包只作用于 level 和 growth。
