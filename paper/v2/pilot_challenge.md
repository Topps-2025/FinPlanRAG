# Pilot challenge: 预实验结果对立论、gap 与 A+B 的裁决

**状态**：预实验完成（2026-10-31）。结论先行：**还不能进入完整实验。** 这版 pilot 验证了“成本感知的决策时机”这一半，但没有验证“证据驱动的模式切换”这一半；planner 目前更像一个成本感知的 wait/stop 阈值，而不是证据驱动的 planner。A+B 攻击没有被完全回应。

---

## 1. Pilot 设置

- **数据**：v4 revenue-guidance，1,005 claims；dev 529 / validation 203 / test 273。
- **任务**：cutoff 时判断 below/within/above；未来 outcome filing 作为一条带成本的证据，可用时点为其 filing date。
- **证据**：guidance（cost 0.5）、current-quarter anchor（cost 0.5）、outcome filing（cost 2.0，未来可用）。
- **决策**：answer / adjudicate / abstain；可等待；等待按 realized delay 计成本。
- **损失**：over 12、under 8、wrong 10、miss 8、width 2、abstain 5；alpha 0.1。
- **先验/似然**：dev prior；current-quarter likelihood 在 dev 上拟合并用 reliability 调和；outcome filing 用 precision 控制噪声。
- **基线**：always-answer、always-adjudicate、fixed top-k、fixed-budget adjudicate、precise-first、pre-hoc router、always-wait、planner-no-wait、planner-no-evidence、oracle-router、oracle。
- **统计**：paired bootstrap（20,000 resamples，seed 20261031）；cost-loss frontier；loss/precision/signal-strength 敏感性。

---

## 2. 主结果（γ=0.01, outcome cost=2.0, precision=1.0）

| 方法 | total loss | coverage | wait rate | answer rate | adjudicate rate | ECE |
|---|---:|---:|---:|---:|---:|---:|
| always-answer | 5.656 | .462 | .00 | 1.00 | .00 | .101 |
| always-adjudicate | 4.000 | 1.000 | .00 | .00 | 1.00 | n/a |
| fixed-budget adjudicate | 4.524 | .971 | .00 | .00 | 1.00 | n/a |
| planner-no-wait | 3.879 | .971 | .00 | .00 | 1.00 | n/a |
| pre-hoc router | 6.656 | .462 | .00 | 1.00 | .00 | .203 |
| oracle-router | 5.381 | .648 | .00 | .645 | .355 | .124 |
| always-wait | **2.977** | 1.000 | 1.00 | 1.00 | .00 | .000 |
| **planner** | **2.977** | 1.000 | 1.00 | 1.00 | .00 | .000 |
| planner-no-evidence | **2.977** | 1.000 | 1.00 | 1.00 | .00 | .000 |
| oracle | 0.000 | 1.000 | .00 | 1.00 | .00 | .602 |

**Paired bootstrap（planner − baseline, total loss）：**

| baseline | mean diff | 95% CI |
|---|---:|---|
| always-answer | −2.679 | [−3.305, −2.040] |
| always-adjudicate | −1.023 | [−1.043, −1.001] |
| fixed-budget adjudicate | −1.547 | [−1.725, −1.380] |
| planner-no-wait | −0.902 | [−1.069, −0.755] |
| pre-hoc router | −3.679 | [−4.305, −3.040] |
| oracle-router | −2.404 | [−2.965, −1.859] |

**关键观察**：planner 与 always-wait 在主配置**完全持平**，与 planner-no-evidence 也完全持平。planner 的增益来自“等待”这个动作，不是来自 current-quarter 证据。current-quarter anchor 在 v4 上不是 worth its cost。

---

## 3. Cost-loss frontier：planner 不是一致占优

| γ | outcome cost | planner | always-wait | planner-no-wait | planner wait rate |
|---:|---:|---:|---:|---:|---:|
| .000 | 1 | 1.000 | 1.000 | 3.879 | 1.00 |
| .005 | 2 | 2.489 | 2.489 | 3.879 | 1.00 |
| .010 | 2 | 2.977 | 2.977 | 3.879 | 1.00 |
| .010 | 4 | 3.967 | 4.977 | **3.879** | .10 |
| .020 | 2 | 3.954 | 3.954 | **3.879** | 1.00 |
| .050 | 2 | 3.879 | 6.885 | **3.879** | .00 |

- 在低 delay cost 区域，planner 匹配 always-wait；
- 在高 delay cost 区域，planner 匹配 no-wait；
- 在 crossover 区域，planner 可能**略差于** no-wait（例如 γ=.01/cost=4: 3.967 vs 3.879；γ=.02/cost=2: 3.954 vs 3.879；γ=.005/cost=4: 4.247 vs 3.879）。原因是 company-historical expected delay 有估计误差，保守 margin 让它等得过久。

**裁决**：planner 的价值是“在未知/变化的成本 regime 下接近最优固定策略”，不是“一致优于最优固定策略”。如果成本 regime 已知且固定，一个固定策略就够了。

---

## 4. 证据层：current-quarter anchor 没有用

- `planner-no-evidence` 与 `planner` 在主配置完全持平（2.977）。
- `oracle-router`（使用 current-quarter 证据后选最优 stop）5.381，反而比 no-wait 3.879 更差：证据成本没有被收益抵消。
- Signal-strength 诊断（controlled，非真实数据）：把 high-reliability 信号放大到 strength=1.0，planner 仍然 wait all 2.977；`planner-no-wait` 反而恶化到 5.154（过信强信号导致错误答案）。没有出现 case-level mode switching。
- Precision 敏感性：
  - precision=1.0: planner 2.977, no-wait 3.879, always-wait 2.977；
  - precision=0.8: planner 4.272, no-wait 3.879, always-wait 4.354（planner 比 no-wait 差）；
  - precision=0.6: planner 3.879 = no-wait, always-wait 5.724。
  - 说明 planner 能对**未来信号质量**做全局调整，但不是 per-case 证据驱动。

**裁决**：v4 的 current-quarter anchor 太弱，无法支撑“证据改变模式”的主张。pilot 只能支持“决策时机”，不能支持“证据获取/模式切换”。

### 4.1 Controlled mode-switch diagnostic（2026-10-31）

为了单独测试 P3，加了一个 **controlled preliminary signal**：

- 每条 case 在 cutoff+30 天有一条 preliminary evidence，cost 0.5；
- 一半 case 的 precision=0.9，一半 precision=0.5；分配由 case_id 决定，**与 pre-hoc router 能看到的 query 特征独立**；
- planner 可以获取 preliminary，观察 signal 后再决定 stop/wait；pre-hoc cost-aware router 不能观察 signal 内容。

**MA subset（precision=0.9，n=133）结果：**

| policy | total loss | decision loss | total cost | coverage | answer rate | adjudicate rate | preliminary rate | leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| planner_preliminary | **1.4617** | 0.6617 | 0.8000 | .9248 | .5564 | .4436 | 1.00 | 0 |
| planner_no_evidence (pre-hoc cost router) | 2.9626 | 0.0000 | 2.9626 | 1.0000 | 1.0000 | .0000 | .00 | 0 |
| planner_no_wait | 3.9098 | 3.5188 | 0.3910 | .9624 | .0000 | 1.0000 | .00 | 0 |
| always_wait | 2.9626 | 0.0000 | 2.9626 | 1.0000 | 1.0000 | .0000 | .00 | 0 |
| always_adjudicate | 4.0000 | 4.0000 | 0.0000 | 1.0000 | .0000 | 1.0000 | .00 | 0 |

**Paired bootstrap（planner_preliminary − planner_no_evidence）：**
**−1.5009，95% CI [−1.8753, −1.0722]，n=133。**

**全 test split（273 cases）**：planner_preliminary 2.4745 vs planner_no_evidence 2.9771。

**裁决**：
- 在 controlled setting 下，planner 观察 evidence 后切换模式，显著优于 pre-hoc cost-aware router。**P3 在 controlled diagnostic 中成立。**
- 这是 A+B 的“非可加能力”证据：增益来自“观察 signal 后决策”，不是全局成本阈值。
- **边界**：preliminary signal 是合成的；v4 真实 current-quarter 证据仍然是负结果。完整实验必须用 real evidence-rich benchmark 复现这个 MA 结果。

---

## 5. Loss 敏感性：planner 会输

| loss config | planner | no-wait | always-adjudicate |
|---|---:|---:|---:|
| default | 2.489 | 3.879 | 4.000 |
| symmetric | 2.489 | 4.319 | 4.000 |
| more overclaim | 2.489 | 3.879 | 4.000 |
| high width | 2.489 | 6.011 | 6.000 |
| **low width** | **2.489** | **2.000** | **2.000** |
| high abstain | 2.489 | 3.879 | 4.000 |

- 在 default / symmetric / high-width / high-abstain 下 planner 赢；
- 在 **low width** 下，no-wait adjudication 只要 2.0，planner 的等待反而多余，planner 输。
- 这说明 planner 的优势依赖 adjudication 的宽度成本；如果 adjudication 很便宜，等待没有必要。

---

## 6. 重新 challenge 立论与 gap

### 6.1 Pilot 支持什么

- **P1（点答案不充分）**：always-answer 5.656 vs 等待/裁决 2.977–4.0；点答案在不确定时明显更差。
- **P4（决策充分性停止）**：cost-loss frontier 显示固定 top-k/固定预算不是最优；等待/停止应随成本变化。
- **P7（固定预算次优）**：在变化 γ / outcome cost 下，没有单一固定策略占优；planner 能接近每个 regime 的最优固定策略。
- **泄漏审计**：所有策略 leakage=0。
- **指标/数据集构建**：decision-level 数据卡、损失、ECE、coverage、cost-loss frontier、paired bootstrap 均可运行。

### 6.2 Pilot 不支持什么

- **P2（相关性不充分）**：没有 typed evidence 的对照；真实 v4 current-quarter 信号太弱。
- **P5（独立性/击败）**：v4 证据池没有 independence/defeat 结构，未测。
- **真实数据上的证据获取价值**：current-quarter 证据不被使用；只有 controlled preliminary signal 有正价值。
- **A+B 的完整回应**：controlled setting 下已回应（planner 观察 signal 后切换模式）；真实数据上仍未回应。
- **Loss 稳健性**：low width 下 no-wait 胜出；需要完整实验的敏感性报告。

### 6.3 Gap 还成立吗

- **Capability gap 的陈述仍然成立**：现有系统固定输出接口/时间/预算，没有一个联合决策。
- **实证支持现在是两层**：
  - controlled mode-switch diagnostic：planner_preliminary 显著优于 pre-hoc cost-aware router（−1.5009 [−1.875, −1.072]，n=133），支持 P3；
  - real-data v4 pilot：支持决策时机（P1/P4/P7），不支持 current-quarter 证据（P2/P3 在真实数据上仍负）。
- **A+B 裁决**：controlled setting 下 planner 不是 router——它观察 preliminary signal 后切换模式；真实数据上仍是成本阈值。**A+B 的首关已过，external-validity 关未过。**
- **Motivation 裁决**：P1/P3/P4/P7 有 controlled/real 支持；P2/P5 未测。数学动机可以立住“接口不足 + 联合决策必要”，但不能声称已经在真实金融证据上全部验证。

### 6.4 RQ 裁决

- **RQ1（formulation）**：支持。
- **RQ2（necessity）**：部分支持。controlled MA subset 上 planner 显著优于 pre-hoc cost-aware router；real v4 上 planner 不是一致优于 best fixed policy。
- **RQ3（identification）**：**controlled 正结果，real v4 负结果**。controlled 增益来自 preliminary evidence；real v4 增益来自 wait/stop，不来自 current-quarter 证据。
- **RQ4（cost–loss frontier）**：部分支持。planner 接近但不一致占优；crossover/low-width 下会输。

---

## 7. Go / No-Go：Conditional Go

**结论：可以进入完整实验，但必须采用 two-track protocol，且主 claim 条件化。**

### Track A：Controlled mechanism（已通过）

- preliminary signal 在 cutoff+30 天可用，cost 0.5；50% precision 0.9，50% precision 0.5；分配与 router 可见特征独立。
- MA subset（n=133）：planner_preliminary 1.4617 vs pre-hoc cost-aware router 2.9626；paired −1.5009 [−1.875, −1.072]。
- 这证明：planner 观察 evidence 后切换模式，显著优于“只看成本、不看证据”的 router。
- **A+B 的非可加能力在 controlled setting 下成立。**

### Track B：Real-data external validity（未通过，完整实验的第一项任务）

- v4 current-quarter anchor 是负结果：planner = planner_no_evidence。
- 真实数据上没有 case-level evidence-induced mode switching。
- 完整实验必须构建 evidence-rich real benchmark（v2 expansion/hard anchors/multi-source/defeat），并复现 MA 结果。

### 主 claim 的条件化

- 如果 Track B 复现 controlled MA 结果：论文可以声称“联合决策在真实金融证据上有效”。
- 如果 Track B 不复现：论文只能声称“controlled mechanism + real-data benchmark/measurement”；不能声称 method win。这仍是可发表的 negative/measurement 贡献，但主张要收缩。

### 为什么现在是 Go 而不是 No-Go

之前的 No-Go 理由中，以下已经解决：

1. **pre-hoc cost-aware router baseline**：已实现（`planner_no_evidence`），planner 在 MA subset 上显著优于它。
2. **mode-ambiguous subset**：controlled MA subset 已建，n=133，preliminary rate 1.0。
3. **非可加能力**：controlled diagnostic 已证明。
4. **数据集/指标/算法设计**：已落盘、可运行、测试通过。

仍未解决的：

1. real evidence-rich benchmark（Track B）；
2. v2 production claims 扩到 30–50 条；
3. determinate/mixed tracks；
4. 人工核验 + IAA；
5. loss 敏感性在全实验中的稳健性。

这些是**完整实验的工作项**，不是立论 blocker。

---

## 8. 完整实验 protocol（two-track）

### 8.1 Track A：controlled mechanism

- 保留 preliminary signal 诊断；
- 扩展为多精度（0.9/0.7/0.5）、多成本、多 delay；
- 报告 planner vs pre-hoc cost-aware router 的 paired bootstrap；
- 作为 P3 的机制证据，不作为真实数据主张。

### 8.2 Track B：real evidence-rich benchmark

1. **v2 production expansion**：15 → 30–50 claims，加入 permit/capacity/equipment/qualification hard anchors；
2. **v4 evidence enrichment**：multi-source、independence、defeat；或换用有 filing-level anchors 的 guidance 数据；
3. **mode-ambiguous subset**：记录 `mode_before`、`mode_after`、`switch_evidence`；≥50% 真切换；IAA κ ≥ 0.7；
4. **pre-hoc cost-aware router**：同成本模型，不观察 evidence；planner 必须显著优于它；
5. **determinate/mixed tracks**：LOFin/FinGLM + guidance/outcome 联合查询；统一 decision loss；
6. **human verification**：≥100 条分层样本；
7. **stats**：paired bootstrap + Holm；loss 敏感性必报；
8. **leakage audit**：0 泄漏。

### 8.3 决策门槛

- **Method paper**：Track B MA subset 上 planner 显著优于 pre-hoc cost-aware router；loss 敏感性不翻转；leakage 0。
- **Benchmark/measurement paper**：Track B 无显著优势，但 benchmark + controlled mechanism + negative result 完整；主张收缩。
- **No-go**：Track A 也失败（当前 not）。

---

## 9. 对 v2 文档的修订（已执行/待执行）

- `research_design.md`：已加 Pilot 状态；需加 controlled positive + two-track。
- `algorithm_design.md`：已加 conservative planner；需加 preliminary plan + controlled result。
- `claim_evidence_matrix.md`：C5/P3 改为 “controlled supported / real not supported”；C10 改为 partial。
- `experiment_plan.md`：E4.3 增加 pre-hoc cost-aware router（已实现 `planner_no_evidence`）；Pilot gate 改为 conditional go。
- `pilot_dataset_card.md`：增加 controlled preliminary extension 一节。
- `metrics_spec.md`：增加 `preliminary_rate`；loss 敏感性必报。

---

## 10. 一句话

**Conditional Go：controlled mechanism 已过 A+B 的首关（planner 观察 evidence 后切换模式，显著优于 pre-hoc cost-aware router）；real-data evidence-rich benchmark 是完整实验的第一项任务，不是立论 blocker。**
