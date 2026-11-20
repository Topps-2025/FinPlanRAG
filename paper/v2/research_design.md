# FinPlan-RAG v2 研究设计：从期间检索到决策接口

**状态**：研究设计（2026-10-31）。本文档只讨论立论、related work 边界、gap、RQ 与评测设计，不展开算法实现。

**关键修正**：不假设固定 cutoff，也不假设固定预算。决策时间 \(t\)、证据成本 \(\lambda(e)\)、等待/延迟成本 \(\gamma\) 是决策问题的一部分；固定 cutoff/budget 只是特例。Point-in-time 仍然保留：在选定决策时间 \(t\)，只能使用 \(a(e)\le t\) 的证据，策略必须 \(\mathcal F_t\)-可测。

---

## 0. 一句话论点

金融查询是异质的；现有系统在证据之前就固定了输出接口（RAG 假设答案存在，verification 假设 claim 与证据已给定，sufficiency head 在固定任务上判断是否回答）。本文把**何时做判断、获取什么证据、输出哪种模式**统一为一个 credal evidence state 上的决策问题；RAG 与 adjudication 是这个决策问题的动作实例，而不是两个被路由的模块。

### 0.1 Pilot 状态（2026-10-31）

已在 v4 revenue-guidance 的 test split 上完成 decision-level pilot（273 cases），并加了一个 controlled preliminary 诊断。结论：

- **支持（real）**：决策时机（wait vs stop）在 cost-loss frontier 上是有价值的；固定 wait/stop 策略不是一致最优；planner 相对 no-wait 的 paired bootstrap 显著。
- **支持（controlled）**：controlled preliminary MA subset（n=133）上，`planner_preliminary` 1.4617 vs pre-hoc cost-aware router 2.9626，paired −1.5009 [−1.875, −1.072]；planner 观察 evidence 后切换模式。
- **不支持（real）**：current-quarter anchor 没有证据价值（`planner` = `planner-no-evidence`）；真实数据上没有 case-level evidence-induced mode switching；planner 在 crossover 和 low-width 下会输给最优固定策略。
- **Go/No-Go**：**conditional go**。Track A（controlled mechanism）已通过；Track B（real evidence-rich benchmark）是完整实验的第一项任务。如果 Track B 复现 MA 结果，论文可声称真实数据上的联合决策；如果不复现，收缩为 controlled mechanism + benchmark/measurement。详见 `pilot_challenge.md`。

---

## 1. 研究背景

### 1.1 金融查询不是同一种对象

- **determinate**：*“FY2023 revenue 是多少？”*——有唯一答案；证据是申报文件中的数值。
- **vague forward-looking**：*“新产线会不会在 H2 投产？”*——没有唯一答案；输出应是可行集/置信判断。
- **mixed**：*“公司是否达成了 H2 guidance？”*——先检索 guidance，再判断 outcome 是否落在区间内。
- **quantitative-vague**：*“2026 年 margin guidance 可信吗？”*——既有数值，又有前向不确定性。

这个异质性不是噪声。它决定了什么证据可采纳、什么输出是正当的。当前系统把这个问题压成了“一个问题 → 一个答案”或“一个 claim → 一个 verdict”。

### 1.2 现有接口在证据之前固定了三件事

现有系统通常固定：

1. **输出模式**：answer 或 verdict 或 abstain；
2. **决策时间**：query time / cutoff 由用户给定；
3. **检索预算**：top-\(k\)、固定轮数、固定 token/查询预算。

- **RAG / adaptive RAG**（Lewis 2020；FLARE 2023；Self-RAG 2024；DRAGIN 2024；Adaptive-RAG 2024；ReAct 2023；Reflexion 2023）假设答案存在，检索 passage，生成 answer；停止规则是 reader confidence 或 query complexity。
- **Claim verification**（FinGround 2026；CLAIR-Fin 2026；VeriFin 2026；EvidenceLens 2026；Evidence-Ledger Adjudication 2026；FinRCA-Bench 2026；FinFIRST 2026）假设 claim 与证据已给定，输出 supported/contradicted/unverifiable 或 evidence ledger。
- **Sufficiency / abstention**（RealFin 2026；GBFR；FailSafe-RAG；SAFE-MR 2026；FinVerBench 2026；AWARE-FX；Chow 1970；Geifman & El-Yaniv 2017）在固定任务上判断“够不够答”，输出二值/三值拒答。
- **Point-in-time**（FinanceGym；FinTrace/PITfall；Look-Ahead-Bench 2026；AhaSignals；Sarkar & Vafa 2025）限制文档时间，是 validity condition，不是 decision semantics。
- **VoI / evidence acquisition**（LAVOIR 2026；REVOIR 2026；VOILA 2026；VoI Framework 2026；RAVEL 2026；Krishnamurthy 2017；Satsangi 2018）决定“何时/问什么”，但 hypothesis space 通常不是金融 claim，且不包含 hard/soft evidence、independence、defeat、输出模式选择。
- **Independence / defeasible reasoning**（echo chamber/influence-aware aggregation；InfoGatherer 2026；Split the Labor 2026；Dung 1995；CLAIR-Fin 2026）处理来源可靠性与非单调更新，但不是 claim-level credal state，也不决定输出模式。
- **Event forecasting / guidance tracking**（RICE-Alpha 2026；PolyBench 2026；dynamic financial knowledge graphs；SEC Filings to Financial Signals 2026；Marvin Labs；NOSIBLE）做事件状态/预测/beat-miss，不是 cutoff 下的 claim adjudication。

### 1.3 代价

- 对 vague claim 给点答案 → **overclaim**；
- 对 determinate 问题给 verdict → **under-informative**；
- 证据不足仍回答 → **校准失败**；
- 证据足够但不敢答 → **过度拒答**；
- 在错误的时点做判断 → **过早/过晚决策**；
- 在错误的成本下检索 → **预算浪费或证据不足**。

金融场景的损失不对称：overclaim 的代价通常高于 abstain；过早决策的代价可能高于等待一个 filing。现有接口没有表示“证据支持 H2，但不支持 Q3/Q4”，也没有表示“等到下一个 10-Q 再判断”的决策。

### 1.4 能力缺口

现有系统要么**按任务固定模式**（QA vs verification），要么**在证据之前按 query 分类**（router），并且通常固定 cutoff/budget。没有一个 formulation 同时决定：

1. 何时做判断（decision time）；
2. 获取什么证据（evidence acquisition）；
3. 输出哪种模式（answer / adjudicate / abstain）。

因此它们无法表达：

> 先按 precise 检索 → 发现证据只支持 qualified judgment → 等到下一个 filing 或切换 adjudication → 若 credible set 不落入决策单元则 abstain。

这是 **capability gap**，不是缺少组件。组件都有人做；缺的是把组件连接起来的**决策接口**。

### 1.5 本文的问题

> 一个在 credal evidence state 上联合选择决策时间、证据获取与输出模式的 planner，能否比 fixed-mode、fixed-cutoff、fixed-budget policy 和 pre-hoc router 获得更好的 cost–decision-loss frontier？

这是一个 formulation + evaluation 问题，不是一个 retriever/reader 问题。

---

## 2. Related Work 能力边界

> 组织原则：不按论文列表写，按**能力**写；每节结尾写“它不能表示什么”。这样 gap 自然浮现。2026 条目按仓库已核验的 arXiv/ACL 列表引用；正式投稿前逐条核对 venue/DOI，未核验的按 preprint 处理。

| 能力 | 最新代表工作 | 能做什么 | 不能表示什么 | 对本文的含义 |
|---|---|---|---|---|
| **Answer-centric RAG / adaptive RAG** | Lewis 2020；FLARE 2023；Self-RAG 2024；DRAGIN 2024；Adaptive-RAG 2024；ReAct 2023；Reflexion 2023 | 检索、重排、生成、自适应停止、工具调用 | 非 singleton 可行集；输出模式选择；typed evidence；内生决策时间 | 假设答案存在；不能表达“证据只支持合格判断” |
| **结构化金融 RAG** | FinRank 2026；Fin-RATE 2026；HiREC 2025；HC-RAG 2026；FinCARDS 2026；FinSAgent 2026 | typed evidence、period/entity 约束、provenance、层级检索 | vague 前向 claim；credal state；outcome adjudication；模式/时间/成本联合决策 | 在固定任务内提升检索，不改变输出接口 |
| **Claim verification / decomposition** | FinGround 2026；CLAIR-Fin 2026；VeriFin 2026；EvidenceLens 2026；Evidence-Ledger Adjudication 2026；FinRCA-Bench 2026；FinFIRST 2026 | claim 分解、verdict、citation、evidence ledger、argumentation | determinate 检索/回答；未来 outcome；模式/时间/成本联合决策 | 假设 claim 与证据已给定；输出是 verdict，不是可行集 |
| **Sufficiency / selective prediction** | RealFin 2026；GBFR；FailSafe-RAG；SAFE-MR 2026；FinVerBench 2026；AWARE-FX；Chow 1970；Geifman & El-Yaniv 2017 | 校准拒答、sufficiency head、selective risk | 非二值可行集；decision cell；模式/时间/成本联合决策 | sufficiency 是固定任务上的二/三值判断 |
| **Point-in-time** | FinanceGym；FinTrace/PITfall；Look-Ahead-Bench 2026；AhaSignals；Sarkar & Vafa 2025 | cutoff 检索、泄漏审计、as-of query | 未来 outcome 的 adjudication；decision sufficiency；内生决策时间 | 是 validity condition，不是 decision semantics |
| **VoI / evidence acquisition** | LAVOIR 2026；REVOIR 2026；VOILA 2026；VoI Framework 2026；RAVEL 2026；Krishnamurthy 2017；Satsangi 2018 | 何时/问什么；决策论停止；active sensing | 金融 hypothesis space；hard/soft evidence；independence/defeat；输出模式选择 | 通用决策理论，不是金融 claim state |
| **Independence / defeasible reasoning** | echo chamber/influence-aware aggregation；InfoGatherer 2026；Split the Labor 2026；Dung 1995；CLAIR-Fin 2026 | 来源可靠性、独立性折扣、belief fusion、argumentation | 金融证据图；claim-level credal state；模式/时间/成本联合决策 | 通用融合，不是决策接口 |
| **Event forecasting / guidance tracking** | RICE-Alpha 2026；PolyBench 2026；dynamic financial knowledge graphs；SEC Filings to Financial Signals 2026；Marvin Labs；NOSIBLE | 事件状态、预测、beat/miss、guidance 抽取 | cutoff 下的 claim adjudication；feasible set；decision sufficiency | 预测事件 ≠ 裁决已披露 claim |
| **Benchmarks** | FinQA/TAT-QA/FinanceBench；FinDVer 2024；Fin-Fact 2025；FinBen 2024；FinMRAGBench 2026 | 静态 QA/verification、数值推理、检索评测 | 异质 claim 类型；模式/时间/成本联合决策；decision-loss 评测 | 固定任务/标签，无法评测混合查询 |

**结论**：每一行都有人做；缺的是它们的并集上的**决策接口**。并集 gap 只有在你能证明“并集改变了系统能表达/能决定的东西”时才成立；否则就是 A+B。

---

## 3. Research Gap

### 3.1 人无我有 vs 人有我强

- **人无我有（capability gap）**：现有 formulation 表达不了某个对象；这个“表达不了”有可测代价。
- **人有我强（performance gap）**：现有方法存在，但在某个重要维度上更差；你更强。

顶会最稳的是 **capability gap 为主、performance gap 为辅**。单纯“我们比 X 强”脆弱；“没人做过 X”容易被问 so what。

### 3.2 三个候选 gap

**候选 1（弱，A+B）**：*“No prior work combines RAG and adjudication for financial claims.”*
→ 审稿人：这就是把两个系统接起来；组合本身不是贡献。**拒绝。**

**候选 2（弱，router）**：*“No prior work routes between precise and vague financial queries.”*
→ 审稿人：query routing / adaptive RAG / mixture-of-experts 都是已知模式；分类器不是贡献。**拒绝。**

**候选 3（强，capability gap）**：

> 现有金融 QA 与 verification 系统在证据之前就固定了输出接口、决策时间和检索预算：RAG 假设答案存在并在固定 top-\(k\)/预算下检索；verification 假设 claim 与证据已给定并返回 verdict；sufficiency head 在固定任务上决定是否回答。**没有一个 formulation 把“何时做判断、获取什么证据、输出哪种模式”表示为 credal evidence state 上的联合决策**，因此无法表达“先 precise 检索、发现只支持 qualified judgment、等到下一个 filing 或切换 adjudication、若 credible set 不落入决策单元则 abstain”。这是 capability gap，不是缺少组件。

→ **接受。** 它不依赖“没人做 RAG”或“没人做 verification”，而是指出**决策接口**缺失。

### 3.3 “人有我强”的部分

即使有人做了 pre-hoc router，router 在证据之前决定模式；planner 可以在证据到达后切换，并且可以决定是否等待/继续检索。我们可以证明/测量：

> 在最优模式或最优决策时间依赖于检索后才观察到的证据的 query 上，pre-hoc routing 与 fixed-cutoff/fixed-budget policy 存在 worst-case loss gap；联合 planner 可以避免。

这是一个**有机制的 performance gap**，不是“我们的分类器更准”。

### 3.4 Gap 的可证伪标准

- 如果 **oracle pre-hoc router** 在 mixed benchmark 上追平 planner，则 mid-retrieval switching 无价值；
- 如果 planner 的增益在固定 backend、固定 cutoff、固定成本后消失，则增益来自这些固定切片，不是 planner；
- 如果 mixed benchmark 没有“模式/决策时间在证据后改变”的 query，则 planner 退化为 router；
- 如果 loss function 在合理替代下结果翻转，则 claim 不稳健。

**论文必须在 Intro 或 RQ 里明确写出这些否定条件。**

### 3.5 为什么不是 A+B

A+B 的定义是：两个现有系统 + 一个 glue，增益可分解为“A 管 A、B 管 B”。本文的反 A+B 论证是：

1. **统一对象**：一个 credal state 覆盖 answer/adjudication/abstention，RAG 与 adjudication 是动作实例；
2. **联合决策**：决策时间、证据获取、输出模式在同一个 objective 下选择，而不是先分类再路由；
3. **非可加能力**：mid-retrieval switching + endogenous stopping 是 router 与 fixed policy 无法表达的；
4. **识别**：oracle router vs planner、fixed vs endogenous、state/action ablation，证明增益来自联合决策本身。

四条缺一，A+B 攻击就成立。

---

## 4. RQ 与设计难点

### 4.1 RQ1：Formulation

> 一个 credal-state formulation 能否同时表示 precise answer、qualified adjudication 和 abstention，并在内生决策时间与证据成本下保持 point-in-time 可测？

- **可证伪性**：如果三种输出无法在同一个 hypothesis space/输出契约下定义，或无法保证 \(\mathcal F_t\)-可测，则 formulation 失败。
- **设计难点**：answer spans 与 future states 如何放进同一个 \(H_q\)？输出契约如何比较 answer 与 judgment？决策时间 \(t\) 如何进入状态？
- **最小证据**：形式定义 + 一个接口原型 + 一个小型 mixed set。

### 4.2 RQ2：Necessity

> fixed-mode、fixed-cutoff、fixed-budget policy 和 pre-hoc router 在异质 query 上是否存在可测的 loss gap，而联合 planner 能避免？

- **可证伪性**：如果 oracle router 追平 planner，或 fixed-mode 在 mixed benchmark 上不差，则必要性不成立。
- **设计难点**：构造“最优模式/决策时间在证据后改变”的 query；定义跨模式可比的 loss；避免 benchmark 偏向 planner。
- **最小证据**：mixed benchmark + fixed policies + pre-hoc router + oracle planner + worst-case 例子。

### 4.3 RQ3：Identification

> 增益来自 planner 的 state/action（时间、证据、模式），还是来自 backend 强度、检索预算、query classifier？

- **可证伪性**：固定 backend 与成本模型后，如果 planner 的增益消失，则增益不在 planner。
- **设计难点**：oracle router vs learned planner；state ablation（query-only vs credal）；action ablation（route-only vs full）；backend 控制。
- **最小证据**：factorial 实验 + 误差分解。

### 4.4 RQ4：Empirical

> 在变化的 cutoff 与成本下，planner 是否在 cost–decision-loss frontier 上优于最强 fixed-mode baseline、pre-hoc routing 和 fixed-cutoff/budget policy？

- **可证伪性**：如果 planner 在 mixed/ambiguous 子集上不优于 router，或在 precise 子集上劣于 always-RAG，则 empirical claim 不成立。
- **设计难点**：loss 的定义与敏感性；answer accuracy 与 adjudication accuracy 的可比性；calibration、cost、time-to-decision 的联合报告。
- **最小证据**：主比较 + 置信区间 + 敏感性分析 + 失败案例。

### 4.5 设计难点清单

| 难点 | 为什么重要 | 一个 solid 的论文要做什么 |
|---|---|---|
| **common hypothesis space** | answer spans vs future states | 定义 typed \(H_q\)；证明输出契约 well-defined |
| **comparable loss** | 错答案 vs 过度自信的裁决 | 定义 decision loss；做敏感性分析 |
| **mixed benchmark** | 模式/决策时间在证据后改变 | 构造 + 标注协议 + IAA；避免偏向 planner |
| **oracle policy** | 最优模式/停止时间是什么 | 模拟环境算 oracle；真实数据做 external validity |
| **backend asymmetry** | RAG 有 reader，adjudication 是确定性更新 | 固定 reader/extractor，或显式建模不对称 |
| **mid-retrieval switching** | planner 的非可加能力 | 必须包含“证据后模式改变”的 query |
| **endogenous decision time** | 不假设固定 cutoff | 等待是动作；延迟有成本；评测 cost–loss frontier |
| **endogenous evidence cost** | 不假设固定预算 | 证据有成本；停止由边际 VoI vs 边际成本决定 |
| **safety/exploration** | 如果使用 bandit/RL | 安全约束或离线评估；金融场景不应在线探索 |
| **external validity** | claim 类型/语言/行业 | 多 claim 类型；时间切分；scope 声明 |
| **annotation** | precise/vague/mixed 的边界 | 协议、IAA、adjudication |
| **non-stationarity** | 金融 regime 变化 | temporal split；报告 drift |

### 4.6 否定条件

论文必须明确：什么结果会让我们放弃假设。例如：

- oracle router 追平 planner → mid-retrieval switching 不必要；
- fixed-cutoff/fixed-budget 追平 planner → 内生时间/成本不必要；
- planner 在 precise 子集上显著劣于 always-RAG → 联合决策的代价不可接受；
- mixed benchmark 的模式切换率过低 → planner 与 router 不可区分；
- loss 的合理替代导致结论翻转 → claim 不稳健。

---

## 5. 形式化 Motivation（摘要）

> 详细定义、假设与证明义务见 `formal_motivation.md`。

### 5.1 设置

- 状态空间 \(\mathcal X\)；query/claim \(q\) 诱导 \(S(q)\subseteq\mathcal X\)。
- 证据 \(e\) 有可用时间 \(a(e)\)、成本 \(\lambda(e)\)、约束 \(S(e)\)。
- 决策时间 \(t\) 是决策变量；在 \(t\) 可用的证据集
  \[
  V_t=\{e:a(e)\le t,\ \text{not retracted}\}.
  \]
- 可行集
  \[
  S_t=S(q)\cap\bigcap_{e\in V_t}S(e).
  \]
- 有限 hypothesis space \(H_q\)；belief \(b_t\) over \(H_q\)。
- 动作集
  \[
  A=\{\text{acquire}(e),\ \text{wait}(\Delta),\ \text{answer}(a),\ \text{adjudicate}(R),\ \text{abstain}\}.
  \]
- 目标
  \[
  J(\pi)=\mathbb E\left[L(\text{output},x)+\sum_{e}\lambda(e)+\gamma\,(t-t_0)\right],
  \]
  其中 \(\gamma\) 是单位时间延迟成本，\(t_0\) 是 query 起始时间；没有硬预算约束。固定 cutoff/budget 是特例。

### 5.2 命题

- **P1（点答案不充分）**：若 \(S_t\) 非 singleton 且 \(L\) 对 overclaim 不对称，则对任意点答案 \(a\)，存在 \(x\in S_t\) 使 \(L(a,x)>\min_A R(A)\)；set-valued/qualified action 可以有更低 worst-case risk。
- **P2（相关性不充分）**：若 relevance score \(r(e)\) 在一个 fiber 上相同，但 fiber 内证据的 \(S(e)\) 不同，则 \(r\) 不是 Bayes action 的充分统计量；只用 \(r\) 的 policy 存在 worst-case loss gap。
- **P3（pre-hoc routing 不充分）**：若最优模式依赖于检索后才观察到的证据，则任何在观察前决定模式的 policy 都不是对所有 \(q\) 最优；state-dependent planner 可以改进。
- **P4（决策充分性停止）**：最优停止时间由 \(S_t\subseteq D_A\) 或边际 VoI \(\le\) 边际成本（含延迟成本）决定。
- **P5（独立性与击败）**：相关证据对可行集的收缩是次可加的；单调累积在 revision/restatement 下不健全。
- **P6（cutoff 可测性）**：在选定决策时间 \(t\)，使用 \(a(e)>t\) 的证据产生的可行集不是 \(\mathcal F_t\)-可测的。
- **P7（固定预算次优）**：当边际 VoI 与边际成本随 query 变化时，固定 top-\(k\)/固定预算的停止规则不能对所有 query 最优；内生停止可以改善 cost–loss frontier。

### 5.3 假设与边界

有限 \(H_q\)、有界 loss、证据真实、likelihood 校准或 worst-case 分析、决策单元已知。命题不追求数学深度；作用是**把 motivation 从故事变成可证伪的论证**。论文不要把它们卖成“重大理论突破”，要卖成“单接口不可能性 + 联合决策的必要性”。

### 5.4 命题到实验

| 命题 | 设计含义 | 实验 |
|---|---|---|
| P1 | answer/adjudicate/abstain 输出契约 | mixed benchmark；point-answer vs adjudication/abstention |
| P2 | typed evidence / hard constraints | relevance-only baseline vs typed evidence |
| P3 | planner state + mid-retrieval switching | pre-hoc router vs planner |
| P4 | decision gate + stopping | stopping-rule ablation；cost–loss frontier |
| P5 | independence/defeat | independence/defeat ablation |
| P6 | cutoff | leakage audit；endogenous cutoff |
| P7 | endogenous cost | fixed budget vs endogenous cost |

---

## 6. 评测设计（摘要）

> 详细实验矩阵见 `experiment_plan.md`。

### 6.1 Mixed benchmark

四类 query：

1. **determinate**：有唯一答案（LOFin/FinGLM 现有资产）。
2. **vague**：前向 claim，未来 outcome 可观察（v2 生产里程碑、v4 revenue guidance）。
3. **mixed**：先检索再裁决（“是否达成 guidance”）。
4. **mode-ambiguous**：最优模式/决策时间在证据后改变——这是 planner 与 router 的分水岭。

每一类都要有标注协议、IAA、以及“为什么这个 query 需要/不需要 mode switching”的记录。

### 6.2 Decision loss

需要跨模式可比的 loss：

\[
L(\text{output},x)=
\begin{cases}
0 & \text{correct answer / correct judgment}\\
\ell_{\text{over}} & \text{overclaim}\\
\ell_{\text{under}} & \text{underclaim / abstain despite sufficiency}\\
\ell_{\text{wrong}} & \text{wrong answer/judgment}
\end{cases}
\]

\(\ell_{\text{over}}>\ell_{\text{under}}\) 反映金融场景的不对称。必须做敏感性分析，不能只报一个数。

### 6.3 Baselines

- **Fixed-mode**：always-RAG、always-adjudicate、always-abstain。
- **Fixed-policy**：precise-first、fixed top-\(k\)、fixed budget、fixed cutoff。
- **Pre-hoc router**：learned query classifier + oracle mode label。
- **Oracle**：模拟环境中的 Bayes-optimal planner。
- **Planner**：联合决策（时间、证据、模式）；具体实现不在本文档讨论。

### 6.4 无固定 cutoff/budget 的评测

- 不报单一预算点；报 **cost–decision-loss frontier**。
- 改变证据成本 \(\lambda(e)\) 与延迟成本 \(\gamma\)，观察最优停止时间与输出模式如何变化。
- 对每个 query，记录“如果早一个 filing 判断/晚一个 filing 判断”的反事实，验证等待的价值。
- 保留 point-in-time 审计：任何使用 \(a(e)>t\) 的证据都算泄漏。

### 6.5 Metrics

- Answer accuracy（determinate）；
- Adjudication accuracy / coverage / calibration（vague）；
- Abstention precision / recall；
- Decision loss（统一）；
- Evidence cost、time-to-decision；
- Cost–loss AUC；
- Mode-switching accuracy（是否在正确的证据后切换）；
- Router accuracy（pre-hoc baseline）；
- Leakage rate。

---

## 7. 顶会审稿人 Challenge

### 7.1 模拟审稿意见

**Summary**：The paper studies heterogeneous financial queries and proposes a planner over credal evidence states. The problem is timely and the formulation is potentially interesting. However, the current argumentation does not yet establish a new capability.

**Strengths**：
- Important problem: financial queries are heterogeneous; fixed-mode systems are insufficient.
- Credal-state formulation is a plausible unifying object.
- Removing fixed cutoff/budget is realistic and opens a genuine decision problem.
- Real-data assets and a prototype exist.

**Weaknesses — fatal**：
1. **A+B**：If the planner is RAG + adjudication + router, it is engineering. The paper must show a capability that a pre-hoc router cannot express.
2. **Motivation not formal**：The paper asserts vague claims matter but does not prove that a fixed interface is insufficient. A decision-theoretic necessity argument is missing.
3. **RQ not falsifiable**：The RQ mixes formulation/routing/adjudication; no negative-result criterion.
4. **Vague branch negative**：The full adjudicator ties always-above on v4; the central claim is unsupported on real data.
5. **Endogenous cutoff/budget is underconstrained**：Removing fixed cutoff/budget can make the problem ill-posed; the planner could wait forever or acquire unlimited evidence. The paper must define delay/evidence costs and a stopping rule, and audit for lookahead.
6. **Mixed benchmark validity**：If constructed by the authors, the planner may win by construction; no oracle/annotation protocol.

**Weaknesses — major**：
7. Related work is a list, not a capability map; strongest baselines (FinRank, HiREC, FinCARDS, HC-RAG, RealFin, FinVerBench) are not compared.
8. Evaluation asymmetry: answer accuracy vs adjudication accuracy; no unified decision loss.
9. Theory disconnected: propositions are not tested; assumptions are not stated.
10. Bandit/RL underspecified: why bandit? reward? exploration? non-stationarity? delayed outcomes?
11. Data/label quality: silver labels, paraphrases, no human verification.

**Questions**：
- What can the planner express that a pre-hoc router cannot? Give a concrete query and a trace.
- What is the decision loss? Why is it the right loss?
- What prevents the planner from waiting forever or acquiring unlimited evidence?
- How do you ensure the endogenous cutoff does not become lookahead?
- What would falsify the planner's advantage?
- How do you ensure the mixed benchmark does not favor the planner?
- What is the oracle policy, and how do you compute it?
- Why bandit instead of a model-based planner? What is the exploration cost?
- How do you handle the reader asymmetry?

**Recommendation**：Major revision / reject, depending on whether the authors can provide the formal necessity argument, the mixed-benchmark experiment, and the endogenous-cost/stopping analysis.

### 7.2 攻击面优先级

| 攻击 | 严重度 | 为什么会打中 | 修复 |
|---|---|---|---|
| A+B / 无新能力 | fatal | 两个模块 + router | 统一对象 + 非可加能力 + 必要性命题 |
| Motivation 不 formal | fatal | 故事代替论证 | 决策论命题 + 单接口不可能性 |
| RQ 不可证伪 | fatal | 混了 formulation/routing/adjudication | RQ1–RQ4 + 否定条件 |
| Vague 分支负结果 | fatal | v4 与 always-above 持平 | 正结果，或明确 benchmark/measurement 定位 |
| 内生 cutoff/budget 无约束 | fatal | 可能无限等待/检索 | 成本模型 + 停止规则 + 泄漏审计 |
| Mixed benchmark 偏向 | major | 作者构造 | oracle + 标注协议 + 反例 |
| baseline 不够强 | major | 没有最强金融 RAG/verification | 同语料/同预算/同 reader |
| 评测不对称 | major | answer vs judgment | 统一 decision loss |
| 理论不落地 | major | P1–P7 无实验 | 每条 proposition 映射一个 ablation |
| Bandit 过度 | major | 无在线数据/模拟 | 先 model-based，bandit 可选 |
| 数据/标签质量 | major | silver label | 人工样本 + IAA |
| 可复现性 | minor | 代码/数据 | manifest/hash/scripts |

### 7.3 什么会让我接受

1. 一个清晰的 formal decision problem，包含 decision time、evidence cost、output mode；
2. P1–P3 + P7 的证明或严格论证；
3. 一个 mixed benchmark，包含 mode-ambiguous queries，且有标注协议；
4. planner vs pre-hoc router vs fixed-policy 的 cost–loss frontier，显示非可加增益；
5. 一个泄漏审计，证明内生 cutoff 没有变成 lookahead；
6. 一个诚实的 negative-result 分析：planner 在哪些 query 上不如 fixed policy，为什么。

---

## 8. 论文结构

### 8.1 Title candidates

- *FinPlan-RAG: Deciding When to Decide, What to Retrieve, and How to Answer for Financial Claims*
- *FinPlan-RAG: A Decision-Theoretic Planner for Heterogeneous Financial Queries*
- *From Answers to Adjudication: Joint Evidence Acquisition and Output-Mode Selection for Financial Claims*

### 8.2 Section outline

1. **Introduction**：异质查询 → 现有接口固定三件事 → capability gap → formulation → contributions → scope。
2. **Related Work**：按能力表组织；每节写 “what it leaves open”。
3. **Problem Formulation**：\(\mathcal X\)、\(S(q)\)、\(\mathcal F_t\)、\(S_t\)、\(H_q\)、动作集、目标函数；固定 cutoff/budget 作为特例。
4. **Theory**：P1–P7；假设、证明、设计含义。
5. **Planner**：联合决策的 instantiation（本文档不展开）。
6. **Action Backends**：RAG（typed obligations）与 adjudication（credal/AnchorEvolve）作为动作实例。
7. **Experiments**：mixed benchmark、baselines、cost–loss frontier、ablations、identification、leakage audit。
8. **Discussion**：什么结果支持/不支持联合决策；什么时候 fixed policy 更好。
9. **Limitations**：silver labels、claim coverage、no reader、non-stationarity、annotation。
10. **Appendix**：proofs、dataset protocol、reproducibility。

### 8.3 Contributions

1. **Formulation**：把“何时判断、获取什么证据、输出什么模式”统一为 credal evidence state 上的决策问题；固定 cutoff/budget 是特例。
2. **Theory**：P1–P3 证明 fixed interface / pre-hoc routing 的不充分性；P4/P7 给出 stopping/cost 的必要性；P5/P6 连接 independence/defeat/cutoff。
3. **Benchmark**：mixed precise/vague/mode-ambiguous benchmark，带 cost–loss 评测与泄漏审计。
4. **Identification**：planner vs router vs fixed policy；证明增益来自联合决策，而不是 backend 或更多检索。

---

## 9. 下一步（不涉及算法）

按优先级：

1. **完成 formal note**：`formal_motivation.md` 的 P1–P7 证明义务与假设。
2. **完成 capability map**：本文第 2 节；逐条核对最新论文与 venue。
3. **定义 mixed benchmark**：四类 query、标注协议、IAA、mode-ambiguous 的构造规则。
4. **定义 decision loss**：跨模式可比；敏感性分析。
5. **写 experiment plan**：fixed policies、pre-hoc router、oracle、planner 对比；cost–loss frontier。
6. **构造必要性实验**：先用小规模/模拟环境验证 P3/P7，再上真实数据。
7. **最后才写算法**：在立论与评测站住之前，不写具体 planner 实现。

---

## 10. 边界与未决问题

- **Claim coverage**：当前资产覆盖 period retrieval、production timing/stage、revenue guidance；是否能代表“vague financial claims”全体？需要在 scope 中明确。
- **Reader asymmetry**：determinate 分支有 reader/answer gold；vague 分支只有 outcome gold。统一 loss 需要显式建模。
- **Oracle**：真实数据没有反事实；oracle policy 只能在模拟环境或受限假设下计算。
- **Annotation**：precise/vague/mixed 的边界需要人工标注；IAA 是硬门槛。
- **Non-stationarity**：金融 regime 变化；temporal split 与 drift 报告是必需的。
- **Ethics**：金融建议的边界；系统输出是 evidence adjudication，不是投资建议。
