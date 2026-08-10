---
type: method-specification
status: proposed-real-data-gated
updated: 2026-08-10
---

# FinPlan-RAG 完整框架

## 1. 方法总览

FinPlan-RAG 是 RAG 上层的受约束序贯规划器。它不修改文档分块或基础检索器，而是在每轮检索后更新显式证据状态，并根据仍未闭合的金融关系义务选择下一动作。基础模型和检索器可替换；算法贡献位于状态、动作约束、策略学习和停止目标。

## 1.1 金融域通用，而非域泛化

FinPlan-RAG 的“通用”限定为金融研究中反复出现的证据机制，不是让 agent 面对任意领域、任意工具自由规划。模板注册表当前包含四条金融任务轨道：

| 金融轨道 | 典型任务 | 关系机制特有义务 |
|---|---|---|
| 生命周期关系 | 收购/控制、许可、供应合同、受监管项目 | 签署/批准/交割/终止、主体链、限制证据 |
| 会计推导 | 财务比率、表格计算、叙述数字推导 | 指标定义、分子/分母、期间、单位、调整/重述 |
| 点时指标 | 市场价格、宏观指标、历史财务值 | 观测日、发布 vintage、修订状态、来源 |
| 比较调查 | 公司比较、复杂历史调查、多源核验 | 独立主体分支、共同定义、期间/单位对齐、冲突 |

四条轨道共享路径/义务/冲突/双时间/谱系/预算状态，但每条轨道的合法动作、查询策略和最低闭包条件不同。这样可以在金融内部迁移，同时避免把“金融模板约束”误称为通用 Agent 能力。真实 10-K 实验确认了公司—年份路径绑定的作用，但混合 10-K/10-Q 冻结实验进一步否定了“公司—年份就是充分原子义务”：同一公司同一年可包含 Q1、Q2、Q3 和 FY 多份独立 filing，旧状态会发生键碰撞。现行 v7 因此采用 filing-aware 义务；它在错误驱动开发回放有效，但在新冻结 validation6 仅闭包 6/8，与元数据分解持平并低于 Generic/HiREC-style period 的 7/8，不能冒充端到端金融域通用性或策略优势证据。

## 2. M0：查询与冻结合同

每个任务冻结公司/关系对象、认知截止时点、概念/模板版本、允许来源、检索器、生成模型、最大调用数、token 与费用。测试期不得更新模板、停止阈值或来源优先级。

## 3. M1：点时证据账本

文档保存 `published_at`、`available_at`、`valid_from/to`、主体、来源、谱系、原文跨度和版本。进入状态更新的必要条件是 `available_at ≤ cutoff`；未来文档即使检索得分高也只能记录为被过滤项。

对监管申报额外保存 filing type、fiscal year 与 fiscal period。财政期间不是简单的日历年份：10-Q 的 Q1–Q3 与 10-K 的 FY 必须保留为不同节点，Q4 数值通常由 FY 10-K 闭合，且非自然年公司不能用日历季度替代财政季度。

同一接口在中国轨道映射为证券代码、公告类型、报告年度/期间和公告可得时间。FinGLM 年报 pilot 只覆盖公司—年度报告义务；巨潮/交易所补建集才覆盖半年报/季报、问询函—回复、并购进展和限制证据。两类数据不得用同一闭包标签混合评分。

## 4. M2：候选路径初始化

RAG/LLM 从问题和已检索材料提出候选主体、交易、资产、合同或项目路径。候选提出不授予关系状态。路径必须具有 `path_id`，并绑定一个关系模板；无法匹配模板时进入 `unknown_template`。

## 5. M3：关系约束的证据状态

对每条路径 \(\pi\) 维护：

\[
S_i^\pi=(O_i^\pi,C_i^\pi,L_i^\pi,T_i^\pi,P_i^\pi),
\]

- \(O\)：必需槽位及 `missing/partial/closed`；
- \(C\)：同槽位冲突及待核验来源；
- \(L\)：终止、撤回、过期、不控制或不蕴含等限制证据；
- \(T\)：可得时间、有效时间和先后约束；
- \(P\)：转载谱系与独立来源计数。

文件级义务的最小键写为

\[
o=(e,r,f,y,p,\tau),
\]

其中 \(e\) 是主体，\(r\) 是关系或指标槽位，\(f\) 是 filing/source type，\(y\) 是财政年度，\(p\in\{Q1,Q2,Q3,FY\}\) 是财政期间，\(\tau\) 是认知截止时点。只维护 \((e,y)\) 会把同年多季度折叠；只维护文档 ID 又无法在检索前表达缺失义务。注册表提出可见 filing 候选，规划器从问题解析期间和关系槽位，闭包校验逐个检查该六元键。

产品、收购控制、许可、供应和受监管项目拥有不同义务，但共享点时、主体解析、谱系和“缺失不等于否定”等硬约束。

## 6. M4：受约束动作空间

合法动作集合为：

\[
\mathcal A(S_i)=\{
\texttt{retrieve-slot},
\texttt{retrieve-limit},
\texttt{expand-entity},
\texttt{expand-path},
\texttt{verify-time},
\texttt{resolve-conflict},
\texttt{assert},
\texttt{abstain}
\}.
\]

关系模板只屏蔽非法或无意义动作，不指定动作顺序。这样，方法与固定金融规则的差异可由“相同模板、不同策略”消融识别。

## 7. M5：规划策略

第一阶段实现透明 value-of-information 策略：优先选择预期降低错误风险最多且成本最低的未闭合义务。第二阶段训练 \(\pi_\theta\)：

\[
\pi_\theta(a_i\mid q,S_i),
\]

训练信号来自公开真实轨迹、受控扰动轨迹和人工审计的动作偏好。可采用监督行为克隆加离线偏好优化；只有公开训练规模足够且相对规则策略有外推增益时，才进入在线/强化学习。

综合目标不只奖励最终答案：

\[
R=\lambda_1 R_{answer}
+\lambda_2 R_{closure}
+\lambda_3 R_{limit}
+\lambda_4 R_{stop}
-\lambda_5 R_{overclaim}
-\lambda_6 R_{leakage}
-\lambda_7 Cost.
\]

损失系数必须以开发集业务风险或多组敏感性报告，不使用单一主观效用证明优越。

## 8. M6：证据验证与状态更新

检索结果先经过实体、槽位、时间、来源和谱系验证，再写入状态。LLM 可以提出字段值，但必须保留原文跨度。冲突不通过全局来源权重直接覆盖，而是生成 `resolve-conflict` 动作或拒答。

## 9. M7：停止、断言与拒答

满足以下任一条件才停止：

1. 至少一条正向路径达到模板闭包，且高风险限制证据已检索；
2. 所有候选路径均达到负向闭包；
3. 证据冲突、模板不适用或预算耗尽，输出 `abstain`。

停止策略用 coverage–risk 曲线评价，不把拒答当错误类别或通过任意效用权重掩盖覆盖损失。

## 10. M8：可审计输出

输出包括最终状态、每条路径、已闭合/缺失义务、支持与限制证据、时间过滤记录、来源谱系、动作轨迹、停止原因、模型/索引/模板版本和成本。

## 11. 伪代码

```python
def finplan(query, cutoff, contract):
    state = initialize_paths_and_obligations(query, contract.templates)
    while state.budget > 0:
        if can_assert(state):
            return audit_card("assert", state)
        if must_abstain(state):
            return audit_card("abstain", state)

        legal_actions = mask_actions(state, contract.templates)
        action = planner.select(query, state, legal_actions)
        evidence = retrieve(action, contract.index)
        verified = validate_entity_time_lineage(evidence, cutoff)
        state = update_evidence_state(state, action, verified)

    return audit_card("abstain", state, reason="budget_exhausted")
```

## 12. 方法边界

- 如果 planner 与固定分解准确率相同但成本更低，只主张效率增益；
- 如果通用 agent 加相同结构化状态后追平，删除“关系约束策略”贡献；
- 如果路径绑定 FinPlan 仅与元数据分解持平，不把过滤器包装成完整状态机的算法增益；
- 如果真实数据没有路径/限制证据错误，删除任务 gap；
- 如果模板人工成本高于收益，论文收缩为 benchmark 或审计工具。
