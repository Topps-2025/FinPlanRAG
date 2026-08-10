---
type: real-data-pilot-and-architecture-revision
status: completed-small-pilot-with-document-level-second-rail
updated: 2026-08-10
---

# 真实数据 Pilot 与 FinPlan 架构迭代

## 1. 本轮回答的问题

本轮不尝试用小样本宣布 SOTA，而是回答两个更基础的问题：

1. research gap 所描述的多证据、路径串线和点时泄漏是否能在真实公开数据中观察到；
2. 当 FinPlan v1 低于通用自适应/HiREC-style 诊断时，框架是否能根据失败轨迹做出可解释的结构修改，并在未参与修改的新 lineage 上保留增益。

## 2. 数据获取与访问问题

### 2.1 Hugging Face

主站 `huggingface.co` 在当前环境连接失败，没有因此略过数据。改用 `https://hf-mirror.com/` 成功下载：

- LOFin/HiREC 五个测试子集，共 3,031 题；
- FinSearchComp 全量 635 题；
- 两个数据集 README 与许可信息。

所有文件保存在 `preexperiments/data/external/`，结果中记录 SHA-256。LOFin 标注许可为 CC BY-NC-ND 4.0，FinSearchComp 为 CC BY 4.0；本项目只做读取和统计，不改写再发布原数据。

### 2.2 SEC EDGAR

未设置联系信息的默认 User-Agent 收到 HTTP 403。构建器保留 `SEC_USER_AGENT` 环境变量，实验下载时使用非用户的测试占位联系字符串；正式复现必须由作者替换为真实研究联系邮箱。未读取或发送本机 Git 邮箱。

## 3. 外部 benchmark 对 gap 的支持

### 3.1 LOFin/HiREC

| 子集 | 题数 | 多证据率 | 跨文档率 |
|---|---:|---:|---:|
| FinQA | 1,103 | 0.9% | 0.0% |
| Numeric Table | 925 | 8.3% | 5.0% |
| Numeric Text | 273 | 0.7% | 0.0% |
| SECQA | 333 | **94.3%** | **94.3%** |
| Textual | 397 | **70.5%** | **67.5%** |

这说明金融 benchmark 内部存在明显不同的证据机制：普通数值题主要是单证据，会计/开放 SECQA 与文本比较却经常需要多页、多文档。它支持“金融规划不能用一个固定槽位序列覆盖全部任务”，但本统计本身不证明 FinPlan 优于 HiREC。

### 3.2 FinSearchComp

- T1 时间敏感数据获取：244 题；
- T2 简单历史查找：219 题；
- T3 复杂历史调查：172 题；
- 68.5% 的题面显式出现年份、截止或历史时间表达。

这为点时指标和复杂比较两条金融轨道提供真实任务依据；由于数据未提供路径级检索轨迹，本轮不把它当作 FinPlan 性能证据。

## 4. SEC 点时关系回放

### 4.1 数据

探索集包含 9 条独立交易 lineage、17 个点时 case；在 FinPlan v3 冻结后，另采集 6 条新 lineage、12 个 prospective holdout case。共覆盖：

- 完成：Activision Blizzard、VMware、Cerner、Splunk、Informatica、Nuance、Xilinx、Pioneer、Discover、Ansys；
- 终止：Figma、iRobot、Spirit、Arm；
- 仍处于条件状态的历史切片：Wiz 及每条交易的终局前切片。

语料均为 SEC 原始 filing。`available_at` 使用 submissions API 的 `acceptanceDateTime`；每条 lineage 构造终局前一日和终局披露后的历史回放。

### 4.2 公平协议

- 共享同一个 BM25 实现、160-token passage、4 次查询预算和确定性 resolver；
- 同一查询、cutoff、文档集合；
- 基线包括 single-shot、固定分解、通用自适应和 HiREC-style answerability/complementary query；
- FinPlan v1、v2、v3、`–time`、`–limit` 共享证据解析器；
- 这是检索规划诊断，不是对原版 HiREC、Search-R1 或真实 LLM 的完整复现。

## 5. Gap 门结果

### 探索集

- single-shot 在 100% case 的四文档结果中至少混入一个其他 lineage 文档；这衡量检索轨迹污染，不等于所有最终答案都错；
- 移除 `available_at` 后 future leakage 为 **58.8%**，过度断言为 **11.8%**；
- 保留时间过滤的所有方法 future leakage 为 0。

### Prospective holdout

- single-shot 的轨迹污染仍为 100%；
- `–time` future leakage 为 **41.7%**，过度断言为 **8.3%**。

因此 G5“真实点时错误门”得到小规模直接支持；路径污染也能观察到。但样本只有 15 条 lineage，G1 仍是 pilot 级通过，尚不足以报告总体发生率或统计显著性。

## 6. 原框架失败与架构修改

### 6.1 探索集触发条件

| 方法 | Accuracy | Closure | 错误 lineage 轨迹率 | 查询数 |
|---|---:|---:|---:|---:|
| 通用自适应 | .765 | 1.000 | .118 | 2.18 |
| HiREC-style | .765 | 1.000 | .588 | 1.65 |
| **FinPlan v1** | **.706** | .824 | .588 | 2.59 |

v1 低于两种现有改进路线，说明“显式状态＋风险优先顺序”本身不足。失败轨迹显示两个具体问题：

1. 槽位查询仍会从其他交易取回高相似 passage，状态没有在检索入口约束路径；
2. `completed/terminated` 在协议风险条款中可能是条件或假设表述，通用关键词查询不能区分真实终局事件。

### 6.2 FinPlan v3 修改

v3 不是单纯调整权重，而是两项结构修改：

- **path-bound retrieval**：把目标主体/资产标识作为检索阶段的硬约束，而不是检索后才做状态归属；
- **mechanism-specific action query**：收购模板分别使用 Item 1.01/协议形成、Item 2.01/完成、Item 1.02/终止查询，并要求带日期的实际事件语言，过滤 forward-looking 风险条款。

探索集结果：v3 Accuracy **.824**、Closure **1.000**、错误 lineage 轨迹率 **.118**，但查询数升至 2.94。相对 v1 是准确率/闭包提升与成本增加的权衡。

### 6.3 冻结后的 prospective holdout

| 方法 | Accuracy | Closure | 错误 lineage 轨迹率 | 查询数 |
|---|---:|---:|---:|---:|
| 通用自适应 | .417 | .667 | .000 | 2.83 |
| HiREC-style | .417 | .667 | .500 | 2.50 |
| FinPlan v1 | .333 | .583 | .167 | 3.17 |
| **FinPlan v3** | **.583** | **.833** | **.000** | 3.00 |

新的 6 条 lineage 没有参与 v3 设计，结果方向与探索集一致。不过 12 个 case 太小，且 resolver 是规则型，因此只能说架构修改值得进入更大规模实验，不能声称超过 HiREC 或达到 SOTA。

## 7. 金融域通用性

框架现已改为四模板注册表：生命周期关系、会计推导、点时指标、比较调查。四类共享路径/义务/冲突/双时间/谱系/预算状态，但拥有不同闭包义务和合法动作。

当前证据状态：

- 生命周期关系：已完成小规模真实方法 pilot；
- 会计推导和比较调查：已完成第一批 17 题/38 份与第二批 30 题/67 份完全独立真实 SEC 10-K 的文件级闭包 pilot；路径绑定需求复现，但 FinPlan 与仅元数据分解持平；
- 点时指标：FinSearchComp 已完成数据需求审计，尚未完成方法对比；
- 所有非收购轨道仍缺少非 oracle 义务生成和真实 reader/LLM 的端到端实验。

所以现在可以主张“框架设计为金融域多轨道，并在第二条真实文件级轨道的独立验证批次复现了路径绑定需求”，不能主张“已经证明完整 FinPlan 在金融域通用”。G7 只有部分支持，尚未通过。

## 8. 下一轮必做实验

1. 把 SEC lineage 扩展至至少 30 条，并对 passage 状态做双人标注（当前文件级 LOFin 第二批已达 30 题，但尚非生命周期 lineage）；
2. 继续解决 LOFin 原 PDF 文档包的 Google Drive 访问问题，在 SECQA/textual 上复现原版 HiREC 与 same-state generic agent；当前 SEC HTML 回放只完成文件级闭包；
3. 为 accounting derivation 实现定义、组件、期间、单位、重述和计算义务；
4. 在 FinSearchComp T1/T3 各抽取冻结子集，标注 vintage、独立实体腿和来源冲突；
5. 使用真实 LLM 和 dense/hybrid retriever 重跑，报告 company/time/template OOD；
6. 若 v3 在强 HiREC/Search-R1 复现下失去优势，继续修改模板路由、状态更新或学习策略，或者把论文收缩为 benchmark/审计协议。

## 9. 可复现文件

- `preexperiments/build_sec_real_pilot.py`
- `preexperiments/run_sec_real_pilot.py`
- `preexperiments/audit_financial_gap_datasets.py`
- `preexperiments/financial_task_templates.json`
- `preexperiments/sec_pilot_manifest.json`
- `preexperiments/sec_pilot_holdout_manifest.json`
- `preexperiments/results/sec_real_pilot_v1.json`
- `preexperiments/results/sec_real_pilot_holdout_v1.json`
- `preexperiments/results/financial_gap_audit_v1.json`
- `preexperiments/build_lofin_multidoc_pilot.py`
- `preexperiments/run_lofin_multidoc_pilot.py`
- `preexperiments/lofin_multidoc_freeze_v1.json`
- `preexperiments/results/lofin_multidoc_pilot_exploratory_v3.json`
- `preexperiments/results/lofin_multidoc_pilot_holdout_v1.json`
