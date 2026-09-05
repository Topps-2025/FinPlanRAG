# FinPlan-RAG 0905 QF 稳定接收路线方案规划

日期：2026-09-05  
目标：以 Quantitative Finance（QF）为目标期刊，形成可审计、可复现、不过度声称的英文稿与上海财经大学格式中文稿。  
状态：**初稿审稿版；尚未达到“稳定接收”**。本文档中的门槛是路线标准，不是已经观测到的结果。

## 1. 当前核心论点与证据分层

**一句话论点。** FinPlan-RAG 将金融 RAG 从“相关性排序”重述为“带实体、申报、期间和截止时间约束的义务闭合”，并通过反证搜索、状态更新和保守停止提高可审计性。

| 证据层 | 当前结果 | 可以声称 | 不能声称 |
|---|---|---|---|
| 理论 | 分数纤维下界、Bellman 递推、截止时间安全、闭合—正确率边界 | 相关性单分数不足；时点过滤在假设下安全 | 已证明答案正确或经济收益 |
| 受控机制 | 5,600 案例；FinPlan 闭合 .3655、查询 8.8721、效用 -.3583 | H1/H2/H3 的过程级证据 | 真实金融问答 SOTA |
| 外部审计 | LOFin、FinGLM、SEC、中国混合期间试验 | 文件级闭合、未来泄漏与迁移性诊断 | 答案准确率优越、投资回报 |
| 工程集成 | `finplan_rag_plan` JSONL/OpenAI-compatible；Codex/Claude Code 可调用 | 供应商中立接口 | 不同 agent 的性能比较 |

LOFin 中强时段元数据填充基线闭合率约为 .895，而 FinPlan v9 为 .8919；这是一条必须正面回应的反证。FinGLM 的 gold 是自行构造的文件级标签，SEC 和中国试验样本很小，均应在摘要、结果和限制中明确。

## 2. QF 投稿适配

英文稿采用 12pt、Times 风格、双倍行距、1 英寸页边距，并包含标题、摘要、关键词、JEL 分类、正文、图表和参考文献。QF 的作者—年份引用与期刊缩写要求需在最终版本换用官方 `.bst`，当前 `plainnat` 仅用于可编译初稿。格式依据：[Taylor & Francis manuscript layout guide](https://authorservices.taylorandfrancis.com/publishing-your-research/writing-your-paper/journal-manuscript-layout-guide/)、[format-free submission guide](https://authorservices.taylorandfrancis.com/publishing-your-research/making-your-submission/format-free-manuscript-submission/)、[Quantitative Finance journal page](https://www.tandfonline.com/journals/rquf20/about-this-journal) 与 [QF reference guide](https://files.taylorandfrancis.com/ref_rquf.pdf)。

提交包应分为匿名稿和作者信息稿；图表单独导出；附数据可得性、代码可得性、伦理、利益冲突、资助和版本哈希声明。正文中的完整框架图保留用于总览/附录，主文使用简化三面板图。

## 3. 严格审稿攻击矩阵

| 审稿攻击 | 当前薄弱点 | 必须补的证据 |
|---|---|---|
| 新颖性不足 | 与通用 adaptive RAG、HiREC、metadata-fill 的差异尚未在同一 reader/预算下隔离 | 统一协议下的强基线与组件交互消融 |
| 闭合不是正确 | 当前真实数据没有答案级 gold | 数值答案准确率、引用完整性、unsupported assertion、盲审 reader |
| metadata baseline 平局 | LOFin v9 与强 period-aware baseline 闭合率接近 | 错误文件率、跨文档比较、答案准确率和成本 Pareto |
| 基线过弱 | 当前受控主表没有稠密/混合检索 | BM25、BGE/E5、hybrid RRF、single-shot、fixed、generic、HiREC、metadata-fill |
| 外部有效性不足 | self-derived gold、小样本、中国试验探索性 | 许可语料、公司/期间留出、跨语言 OOD |
| 理论假设不落地 | Markov、元数据正确、reader soundness 未做 stress test | 时间戳噪声、元数据扰动、reader 校准与独立证明审计 |
| 金融价值不足 | 未连接比率、预测或决策 | 经济决策敏感性任务；不宣称投资回报 |
| 图表过载 | 当前图有五列、九步循环、后端、多个 callout | 主文三面板；完整图移附录 |

## 4. 必须补齐的理论

### 4.0 重新定义 gap（必须先改）

截至 2026 年，FinRank 已将金融证据检索与答案正确性分开并使用跨期间 hard negatives；Fin-RATE 覆盖纵向/跨实体分析；FinSAgent、HC-RAG 和 FinCARDS 分别从语料对齐、类型化证据路径和模式约束重排角度推进了相邻问题。因此“现有方法只做相关性排序”已经不是可防守的 gap。正文应改为：**尚缺一个统一的有限预算控制契约，将类型化义务、point-in-time cutoff、反证、闭合停止/拒答和共享 reader 的答案级验证放进同一个可审计目标。** 这个 gap 是“统一可检验属性”的缺口，不是“组件从未被提出”的缺口。

### 4.1 已加入的 strong-proof 链条

正文现在包含四个逻辑层：

1. 分数纤维不可识别性：构造相同 relevance、相反 obligation validity 的两个世界，得到 score-only error >= 1/2。
2. 多义务闭合下界：每个义务一个有效记录和 `k` 个同分无效记录时，score-only closure <= `(1/(k+1))^m`。
3. 安全完备性与反证必要性：没有 admissible support 必须 abstain；不搜索 counterevidence 时，存在两个支持证据相同但正确输出不同的世界，过度主张至少 1/2。
4. Bellman + cutoff safety + closure--accuracy boundary：把上述必要信息映射为状态字段、动作集合、停止规则和 reader gate。

### 4.2 Gap proof 小实验

`scripts/run_gap_proof.py` 对等分数展示顺序做穷举，输出 `paper/results/gap_proof_small.json`。当前结果：

| 设置 | score-only closure | obligation-state closure | score-only future leak |
|---|---:|---:|---:|
| 1 obligation, 1 distractor | .5000 | 1.0000 | .0000 |
| 2 obligations, 1 distractor each | .1667 | 1.0000 | .0000 |
| 2 obligations, 2 distractors each | .0667 | 1.0000 | .0000 |
| 2 obligations + future record | .0667 | 1.0000 | .6000 |

这是命题的**机制验证**，不是现实语料准确率。Strong proof 的下一步是把 FinRank/FinanceBench 的真实 hard negatives 按同样方式做 permutation test，固定 reader、top-k、budget 和 cutoff，报告 paired bootstrap CI；若真实 hard-negative 试验不复现，下界仍成立但“现实重要性”需要收窄。

实现一致性要求：输入文档必须提供 `entity`、`fiscal_year`、`filing_type`、`fiscal_period`、`available_at`，并在可解析时提供 `document_id`。检索器已对这些字段执行硬过滤；若数据缺少字段，结果应记为未解决/拒答，不得把语义命中当成理论中的 typed binding。

1. **读者声音性桥接。** 形式化“闭合上下文 + 引用”到数值答案正确的条件，给出 unsupported assertion 和引用遗漏的上界。
2. **预算化 value-of-information。** 把每次检索的预期闭合增益、查询成本和反证价值写成可估计的停止阈值，而不只给 Bellman 形式。
3. **解析与时间戳鲁棒性。** 对期间解析错误率、可得时间误差和修订文件重复给出风险界，配套扰动实验。
4. **状态抽象条件。** 证明在何种充分统计量下可将完整证据历史压缩为 $E_t=(O_t,U_t,C_t,n_t,c)$，并说明何时不成立。
5. **独立证明审计。** 让不参与实现的研究者复核命题、假设和反例；正文只保留可复查的证明，复杂推导放附录。

## 5. 必须补齐的实验

### E1. 冻结真实语料与答案金标准

- 至少两套可再分发或可审计的真实金融数据；公司和财年留出，避免同公司相邻期间泄漏。
- 至少 5,000 个可回答问题，显式标注文件、期间、数值答案、允许的截止时间和不可回答/应拒答样本。
- 发布文档时间戳、修订版、下载清单、问题生成脚本、hash 和许可证。

### E2. 同协议强基线

BM25、BGE/E5 或 Contriever 稠密检索、hybrid RRF、single-shot、fixed decomposition、generic adaptive、HiREC-style、metadata-fill、FinPlan v4/v7/v8/v9；同一 reader、top-k、cutoff、budget、prompt 和后处理。额外报告 oracle-context 上限，区分 retriever 与 reader 瓶颈。

### E3. 预注册消融与 stress

移除 obligation parser、period binding、counterevidence、stopping、cutoff、reader gate；entity-only/entity-year；metadata-fill；期间打乱；时间戳加噪；修订文件重复；缺失申报；跨语言和非标准财年。每个声称有因果作用的模块必须有配对置信区间。

### E4. Reader 与人工锚点

至少两个 reader（一个本地开源、一个独立强 reader），报告数值答案准确率、单位/符号错误、引用完整性、unsupported assertion、拒答精度、校准误差。对不少于 100 个分层样本做盲审人工锚点，报告一致性（如 Cohen's $\kappa$）。

### E5. OOD 与跨语言

SEC、LOFin、FinGLM、中国混合期间、不同财年制度和公司规模做 firm/time split；报告闭合、答案、错误文件、future leak 和成本，不把小样本 pilot 与主结果合并。

### E6. 经济翻译

至少一个比率、预测输入或决策敏感性任务，例如用检索到的收入/利润计算财务比率并测量下游决策改变率；不得直接声称投资收益提升。

### E7. 统计与复现

预注册随机种子和推断单位（问题或公司—期间），使用 paired bootstrap、McNemar 或 permutation，执行 Holm/FDR 校正，报告效应量与 95% CI；发布 prompts、reader 版本、失败案例和完整日志。

## 6. 稳定接收的建议门槛

| 维度 | 建议门槛 |
|---|---|
| 答案准确率 | 两套冻结数据上较最强基线至少 +3 个百分点，95% CI 不跨 0；简单层不下降超过 1 个百分点 |
| 证据闭合 | closure >= .95；exact obligation match >= .85 |
| 错误与安全 | wrong-document <= .20；future leak = 0 |
| 读者可靠性 | unsupported assertion <= .05；ECE <= .05 或人工 $\kappa >= .60$ |
| 成本 | 查询、token 暴露和货币/API 成本不劣于最强基线，或形成清晰 Pareto 改善 |
| 消融 | 每个核心模块至少 2 个百分点或可解释的安全—成本权衡，CI 不跨零 |
| OOD | 留出集达到域内结果的至少 90% |
| 理论 | 假设明确、证明可独立复核、扰动实验与风险界一致 |

以上是路线标准，不是当前结果。若无法达到，应将论文定位为“表示感知的金融检索与保守拒答”，不要写“稳定接收”或“投资决策提升”。

## 7. 0905 后执行顺序

- [ ] T0：冻结术语、主张和 claim-evidence matrix；把 closure、answer accuracy、economic value 分开。
- [ ] T1：补齐正式来源和 DOI；将 QF 官方 `.bst`、匿名/作者版和图表规范接入构建。
- [ ] T2：锁定许可语料、时间戳、firm/time split 和问题金标准。
- [ ] T3：运行稠密、混合、HiREC、metadata-fill 以及统一 reader 基线。
- [ ] T4：运行 reader grid、人工 100 例锚点、引用和 unsupported assertion 评测。
- [ ] T5：完成组件、期间打乱、时间戳噪声、缺失文件和跨语言消融。
- [ ] T6：完成比率/预测/决策敏感性任务与成本 Pareto。
- [ ] T7：更新中英文稿、附录证明、匿名包、数据/代码声明和预注册统计报告。

## 8. 框架图挑战与 GPT Image 提示词

当前图片作为工程总览很强，但主文双栏宽度下会出现字体不可读、视觉层级竞争和“核心算法被外围接口稀释”三个问题。建议：主文三面板，附录保留当前全图；主图只保留一个 callout“Relevance score is not evidence closure”，不在主图放后端比较、九步展开和 Codex/Claude 标识。

**英文生成提示词（小于 4,000 字符）：**

> Create a publication-quality, three-panel scientific diagram for “FinPlan-RAG: Point-in-Time Obligation-State Planning for Financial RAG.” White background, landscape orientation, crisp vector-like lines, readable at two-column journal width, restrained navy/teal/orange palette, no gradients, no 3D effects, no decorative icons, no vendor logos. Panel A, “Query to obligations”: show one financial question flowing to a compact parser and the canonical tuple (e,f,y,p,d); show two separate obligations for two fiscal periods. Panel B, the dominant panel, “Obligation-state planning loop”: show only four numbered phases in a loop: (1) select highest-risk unresolved obligation, (2) retrieve with entity, filing, period, and cutoff constraints, (3) verify supporting and counterevidence, (4) update the evidence state and stop or abstain. Include only the equation E_t=(O_t,U_t,C_t,n_t,c) and one callout “Relevance score is not evidence closure.” Panel C, “Evidence ledger and reader gate”: show ledger fields for document ID, entity, filing, period, availability time, support, and counterevidence, followed by a binary gate to “grounded answer with citations” or “structured abstention.” Add a small footer: “finplan_rag_plan | JSONL/OpenAI-compatible interface.” Use concise labels, generous whitespace, clear arrows, and visually emphasize the central loop. Do not include the nine-step version, backend comparison boxes, long explanatory paragraphs, or Codex/Claude branding.

## 9. 交付物映射

- 英文初稿：`paper/en/main.tex`；中文初稿：`paper/zh/main.tex`。
- 本轮理论增强编译 PDF：`paper/en/FinPlanRAG_English_Draft_0905_v2.pdf`、`paper/zh/FinPlanRAG_Chinese_Draft_0905_v2.pdf`（原交付 PDF 保留，避免覆盖正在打开的文件）。
- 框架图：`paper/FinPlanRAG.png`；正文已插图并标注复杂度边界。
- 结果和溯源：`paper/results/`、`paper/real_data_provenance.md`、`paper/claim_evidence_matrix.md`。
- 参考文献：`paper/en/references.bib`；条目状态见 `paper/references_verification.md`。
- 适配器和测试：`src/finplan_rag/agent_adapter.py`、`tests/`。

## 10. 最终审稿清单

先通过 LaTeX 编译、引用和图表溢出检查，再执行未来泄漏审计、答案级评分、统计校正和人工锚点。提交前必须在摘要和结论中保留一句边界声明：**当前证据支持的是可审计的规划与检索过程，不是已经证明的金融答案正确率或投资收益。**
