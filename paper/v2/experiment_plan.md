# FinPlan-RAG v2 实验计划

**状态**：设计稿（2026-10-31）。本计划只定义实验、基线、指标与否定条件；planner 的具体实现不在本文件讨论。

**核心约束**：不假设固定 cutoff，也不假设固定预算。决策时间 \(t\)、证据成本 \(\lambda(e)\)、延迟成本 \(\gamma\) 是实验变量；固定 cutoff/budget 只作为 baseline。Point-in-time 审计仍然保留。

---

## 0. 现有资产

| 资产 | 规模 | 用途 |
|---|---|---|
| LOFin period questions | 1,572 | determinate 分支 |
| FinGLM period questions | 1,829 | determinate 分支 |
| v2 production claims | 15 | vague timing/stage pilot |
| v4 revenue guidance | 1,005 | vague quantity benchmark |
| validation8 holdout | 24 | parser/obligation component |
| SEC lineage holdout | 12 | point-in-time pilot |
| Chinese mixed-period pilot | 3 | cross-lingual pilot |
| AnchorEvolve prototype | code | adjudication action backend |
| FinPlanRAG typed RAG | code | answer action backend |

**需要新建**：mixed benchmark、decision loss、cost model、planner interface、oracle simulator、pre-hoc router baselines。

---

## 1. E1：Mixed benchmark

具体 pilot 规格（100 条、标注 schema、MA 构造规则、IAA）见 `pilot_benchmark_spec.md`。

### 1.1 四类 query

1. **determinate**：来自 LOFin/FinGLM；有唯一答案；预期输出 precise answer。
2. **vague**：来自 v2/v4；未来 outcome 可观察；预期输出 qualified adjudication。
3. **mixed**：构造“是否达成 guidance”类 query；先检索 guidance，再判断 outcome。
4. **mode-ambiguous**：最优模式/决策时间在证据后改变；planner 与 router 的分水岭。

### 1.2 mode-ambiguous 的构造

必须显式构造“证据后模式改变”的 query，例如：

- 先给一个看似 precise 的问题，但检索到的 evidence 只支持区间/多候选；
- 先给一个看似 vague 的 claim，但 hard anchor（许可、设备交付、产能）唯一确定 timing；
- 多义务 query：一部分义务精确、一部分模糊；
- 证据成本/延迟不同的两条证据路径：一条快速但弱，一条慢但强。

每条 mode-ambiguous query 记录：

- 证据到达前的 oracle mode；
- 证据到达后的 oracle mode；
- 模式切换的证据触发点；
- 如果 planner 不切换，预期 loss。

### 1.3 标注协议

- 类型标签：determinate / vague / mixed / mode-ambiguous；
- oracle mode：在给定证据池与 cost model 下的最优输出模式；
- 证据触发点：哪条证据使最优模式改变；
- IAA：至少两人标注，报告 Cohen's \(\kappa\) 或 Krippendorff's \(\alpha\)；
- 争议样本进入 adjudication，记录裁决理由。

**否定条件**：如果 mode-ambiguous 子集的模式切换率过低（例如 <10%），planner 与 router 不可区分，benchmark 不具备诊断力。

---

## 2. E2：Decision loss

定义跨模式可比的 loss：

\[
L(\text{output},x)=
\begin{cases}
0 & \text{correct answer / correct judgment}\\
\ell_{\text{over}} & \text{overclaim}\\
\ell_{\text{under}} & \text{underclaim / unnecessary abstain}\\
\ell_{\text{wrong}} & \text{wrong answer/judgment}\\
\ell_{\text{abstain}} & \text{abstain when decision-sufficient}
\end{cases}
\]

- 默认 \(\ell_{\text{over}}>\ell_{\text{under}}\)，反映金融场景的不对称；
- 做敏感性网格：\(\ell_{\text{over}}/\ell_{\text{under}}\in\{1,2,5,10\}\)；
- 报告归一化 loss 与原始指标；
- 如果合理替代导致结论翻转，claim 不稳健。

**否定条件**：如果 planner 的优势只在某一个极端 loss 参数下成立，不能作为主结论。

---

## 3. E3：Baselines

### 3.1 Fixed-mode

- always-RAG；
- always-adjudicate；
- always-abstain。

### 3.2 Fixed-policy

- precise-first（先 RAG，失败再 adjudication）；
- fixed top-\(k\) RAG（\(k=1,2,4,8\)）；
- fixed-budget adjudication（固定证据条数）；
- fixed-cutoff policy（只在给定 cutoff 判断，不等待）。

### 3.3 Pre-hoc router

- learned query classifier + mode label；
- **oracle pre-hoc router**：在证据前就知道最优模式，作为上界/对照。如果 oracle router 追平 planner，则 mid-retrieval switching 无价值。

### 3.4 Oracle planner

在模拟环境中计算 Bayes-optimal planner；真实数据只做 external validity。Oracle 用来回答“planner 离最优有多远”，不是主结果。

### 3.5 Planner

联合决策 时间 / 证据 / 模式。具体实现另文讨论；本计划只要求接口可替换。

**注意**：如果 planner 使用 bandit/RL，必须提供离线评估或模拟环境，不得在金融数据上在线探索。

---

## 4. E4：Necessity experiments

| 实验 | 命题 | 设计 | 否定条件 |
|---|---|---|---|
| E4.1 | P1 | 非 singleton 子集：point-answer vs adjudication/abstention | point-answer 不劣于 adjudication → P1 对本任务不成立 |
| E4.2 | P2 | relevance-only vs typed evidence | relevance-only 追平 → typed evidence 不必要 |
| E4.3 | P3 | pre-hoc router vs planner on mode-ambiguous | oracle router 追平 planner → P3 不成立 |
| E4.4 | P4 | stopping-rule ablation（固定轮数 vs VoI/决策充分性） | 固定轮数追平 → 决策充分性停止不必要 |
| E4.5 | P5 | independence/defeat ablation | 去掉后不差 → 独立性/击败不必要 |
| E4.6 | P6 | leakage audit（内生 cutoff） | 出现泄漏 → 形式化失败 |
| E4.7 | P7 | fixed budget/top-\(k\) vs endogenous cost | fixed budget 追平 → 内生成本不必要 |

**E4.3 是主实验**。如果 planner 不能在中途切换模式上赢过 router，planning 贡献为空。

**Pilot 结果（2026-10-31）**：v4 decision-level pilot 没有通过这个门。planner 的决策是全局成本阈值，`planner` 与 `planner-no-evidence` 持平；没有 case-level evidence-induced mode switching。E4.3 必须增加两个东西才能重跑：

1. 一个 **pre-hoc cost-aware router** baseline：它知道成本 regime，选择 wait/stop，但不观察证据内容；planner 必须在 mode-ambiguous 子集上显著优于它。
2. 一个 **evidence-rich、mode-ambiguous** 的 benchmark：证据必须能改变 per-case 的最优模式，而不是只改变全局成本阈值。

---

## 5. E5：Endogenous time/cost

### 5.1 成本模型

- 证据成本 \(\lambda(e)\)：按来源/文档类型/检索次数设定；
- 延迟成本 \(\gamma\)：按天/季度设定；
- 等待动作：等待下一个可用 filing，产生延迟成本；
- 不设硬预算；报告 cost–loss frontier。

### 5.2 评测

- 变化 \(\lambda\) 与 \(\gamma\)，观察最优停止时间与输出模式；
- 对每个 query 记录“早一个 filing / 晚一个 filing”的反事实；
- 报告 cost–loss AUC、time-to-decision、leakage rate；
- 与 fixed-cutoff、fixed-budget、fixed-top-\(k\) 对比。

### 5.3 泄漏审计

- 任何使用 \(a(e)>t\) 的证据都算泄漏；
- 审计等待动作是否真的推进了 \(t\)，而不是偷偷使用未来证据；
- 报告每个决策的 \(\mathcal F_t\)-可测性检查。

**否定条件**：如果 planner 的优势来自“等待到未来信息”而不是“更优的停止/模式决策”，则 claim 不成立。

---

## 6. E6：Ablations / Identification

| Ablation | 目的 |
|---|---|
| state：query-only vs credal state | 证明 state 不是 query 标签 |
| action set：route-only vs full（acquire/wait/answer/adjudicate/abstain） | 证明联合动作集有价值 |
| mode：fixed vs adaptive | 证明模式选择有价值 |
| cost：fixed vs endogenous | 证明内生成本有价值 |
| backend：RAG-only / adjudication-only / both | 证明增益不是来自 backend |
| oracle vs learned planner | 证明 planner 接近最优 |
| pre-hoc router vs planner | 证明 mid-retrieval switching 有价值 |

**识别目标**：增益来自联合决策（时间、证据、模式），而不是来自更强 reader、更多检索、或 query classifier。

---

## 7. E7：Real-data external validity

- **determinate**：LOFin/FinGLM，按模板/公司/期间分层；
- **vague**：v2（15 条）与 v4（1,005 条），按时间/公司/行业分层；
- **mixed**：构造集，报告构造协议；
- **cross-lingual**：中文 pilot 只作描述性证据，不与主结果合并；
- **temporal split**：dev/validation/test 按时间切分；报告 drift；
- **company-disjoint**：v4 的 108 条子集作为实体泛化检查。

**不要**把不同 track 的数字直接平均；分别报告，并说明可比性。

---

## 8. E8：Safety / Reproducibility

- 泄漏审计：证据日期、outcome 日期、history 日期；
- 确定性：固定 seed；报告模型/数据 hash；
- 发布：manifest、accessions、audit scripts、prompt/reader 版本；
- 安全：金融建议边界；系统输出是 evidence adjudication，不是投资建议；
- 如果使用 bandit/RL：离线评估或模拟环境，报告 exploration cost 与 wrong-answer rate。

---

## 9. 否定条件（总表）

| 结果 | 含义 |
|---|---|
| oracle pre-hoc router 追平 planner | P3 不成立；mid-retrieval switching 无价值 |
| fixed-cutoff/fixed-budget 追平 planner | 内生时间/成本不必要 |
| fixed backend 下 planner 增益消失 | 增益来自 backend，不是 planner |
| mode-ambiguous 子集切换率过低 | benchmark 无诊断力 |
| loss 敏感性导致结论翻转 | claim 不稳健 |
| planner 在 precise 子集显著劣于 always-RAG | 联合决策的代价不可接受 |
| 等待动作产生泄漏 | 形式化/实现失败 |

---

## 10. 依赖与顺序

1. **E1/E2**：mixed benchmark + decision loss（可先用 100 条 pilot）。
2. **E3**：fixed baselines + pre-hoc router（不依赖 planner 实现）。
3. **E4.1/E4.2/E4.7**：可用现有 backend 先做。
4. **E4.3/E4.4/E5**：依赖 planner 实现；先做模拟环境，再做真实数据。
5. **E6**：在 planner 实现后做。
6. **E7/E8**：收尾与复现。

**当前可以立刻做的**：

- 写 mixed benchmark 的构造协议与 100 条 pilot；
- 定义 loss 并在现有 v1/v4 结果上做 post-hoc 敏感性；
- 实现 always-RAG / always-adjudicate / precise-first / fixed-k / oracle router；
- 在模拟环境里验证 P3/P7 的 worst-case 例子。

**暂不做的**：

- 不在真实金融数据上在线探索；
- 不把 fixed-cutoff/fixed-budget 的结果写成主结论；
- 不在没有 mixed benchmark 的情况下声称 planner 优于 router。

---

## 11. Pilot gate (2026-10-31): conditional go

**Track A (controlled mechanism) passed.** With a controlled preliminary signal
(available at cutoff+30, cost 0.5; half the cases precision 0.9, half 0.5), on
the MA subset (n=133):

- `planner_preliminary` total loss 1.4617 vs the pre-hoc cost-aware router
  `planner_no_evidence` 2.9626;
- paired bootstrap −1.5009, 95% CI [−1.875, −1.072];
- preliminary rate 1.0, coverage .925, leakage 0.

This is the P3 mechanism result and the first non-additive A+B answer.

**Track B (real-data external validity) did not pass.** On the real v4 pilot:

- `planner` = `planner_no_evidence`; the current-quarter anchor is not worth
  its cost;
- there is no case-level evidence-induced mode switching;
- the planner can be worse than the best fixed policy at the cost-loss
  crossover and under a low width penalty.

**Decision:** enter the full experiment with a two-track protocol.

1. **Track A:** expand the controlled preliminary diagnostic (multiple
   precisions/costs/delays) as mechanism evidence.
2. **Track B:** build a real evidence-rich benchmark (v2 production expansion,
   hard anchors, multi-source, defeat), repeat the MA test, and report the
   result honestly. If Track B is negative, the paper becomes a controlled
   mechanism + benchmark/measurement paper, not a method win.

See `pilot_challenge.md` for the full analysis.
