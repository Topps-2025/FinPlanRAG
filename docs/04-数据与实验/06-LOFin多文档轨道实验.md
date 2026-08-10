---
type: lofin-real-sec-multidocument-pilot
status: multi-batch-gap-validated-v7-development-not-end-to-end
updated: 2026-08-10
---

# LOFin 多文档轨道：真实 SEC 文件级规划实验

## 1. 研究目的

本实验把 FinPlan 从收购生命周期关系扩展到两条非收购金融轨道：

- 比较调查：对多个公司分别取得同期间证据，再进行可比性判断；
- 会计推导：对同一公司多个年份分别取得证据，再执行计算。

实验只回答文件级规划问题：在相同 BM25、语料、预算和已知候选公司—年份义务下，各方法能否检索齐全部标注 10-K。它不评价段落级证据、数值计算或最终答案生成，不能用作端到端 SOTA 证据。

## 2. 数据取得与未解决的访问问题

主站 Hugging Face 在当前环境不可达；使用 `https://hf-mirror.com/` 成功取得 LOFin/HiREC 的问题与证据文件名标注。LOFin 完整文档包所指向的 Google Drive 在当前环境仍不可达，因此没有静默跳过文档：构建器根据 LOFin 的公司、报告年份和文件名标注，通过 SEC submissions/archives API 回溯公开 10-K HTML。

最终得到：

- 17 个真实问题；
- 38 份 SEC 10-K；
- 13 个比较调查、4 个会计推导；
- 探索集 9 题、冻结留出集 8 题；
- 两个分割的 gold document ID 完全不重叠；
- 每题需要 2–3 份文件。

`XOM_2023_10K` 无法按报告年度规则解析，因此对应候选题未纳入；失败被记录在数据协议中，没有用其他年份替换。SEC HTML 与 LOFin 原 PDF 的页边界不同，所以保留了 LOFin 页码标注，但本轮不冒充页级复现。

在初始 8 题留出集之后，又按预先冻结的 qid 升序和文档不重叠规则构建第二验证批次。第二批原计划 30 题、67 份文件；AEP 的历史 CIK 和 BlackRock 的历史申报主体经过 SEC submissions 记录核验后修复，最终实际取得 30 题、67 份文件，失败记录为空。65 份文件由缓存复用，但逐份重新验证文本 SHA-256；第二批与第一批文件零重叠。

## 3. 公平协议

- 共享 38 份文件构成的检索库；
- 共享 BM25、180 词窗口、135 词步长；
- 每种方法最多 4 次动作、读取最多 4 份不同文件；
- 已读文件不能以另一个 passage 重复占用预算；
- 对时间感知方法，只有 `available_at ≤ cutoff` 的文件可进入轨迹；
- 候选公司—年份义务对全部方法可见，以隔离文件级规划，故本轮不评价义务抽取；
- `oracle_reader_correct` 仅等于完整文件闭包，不代表真实 reader 已回答正确。

比较方法包括：single-shot、固定分解、仅元数据分解、Generic Adaptive、HiREC-style，以及为 Generic/HiREC 加相同路径绑定的强基线；另比较 FinPlan v1、v3 无路径消融、v3 路径绑定和无时间消融。HiREC-style 是透明的 answerability/complementary-query 近似，不是原作者代码复现。

## 4. 探索阶段与架构修改

第一轮实现暴露并修复了两个评价错误：Generic Adaptive 的覆盖判断使用了错误 ID 格式；跨轮检索可能让同一文件的不同 passage 重复占预算。修复后，FinPlan v1 仅闭包 4/9，低于固定分解、Generic Adaptive 和 HiREC-style 的 5/9。

探索阶段还发现，统一追加 `definition/period/unit/reported` 会把 FinPlan v3 无路径消融的闭包压低至 2/9。这不是金融域共享状态的合理效果，而是跨任务强行共享查询词。最终架构据此改为：

1. 共享路径、义务、时间和预算状态接口；
2. 把每个公司—年份展开为独立证据义务；
3. 在检索入口执行公司—年份路径绑定；
4. 不强迫比较调查和会计推导使用同一查询扩展词；
5. 不检索 LOFin 未单独标注的“共同定义文件”，避免无法评价的动作污染指标。

修改后运行探索集 v3，并以脚本、数据和结果 SHA-256 写入 `preexperiments/lofin_multidoc_freeze_v1.json`；随后才运行留出集，未根据留出结果继续调参。

## 5. 结果

### 5.1 探索集（9 题）

| 方法 | 完整闭包 | Leg Recall | 错误文档率 | 平均动作 | 平均文件 |
|---|---:|---:|---:|---:|---:|
| Single-shot | 4/9 | .593 | .667 | 1.00 | 4.00 |
| 固定分解 | 5/9 | .704 | .296 | 2.33 | 2.33 |
| Generic Adaptive | 5/9 | .741 | .389 | 3.00 | 3.00 |
| HiREC-style | 5/9 | .741 | .463 | 2.56 | 3.44 |
| FinPlan v1 | 4/9 | .630 | .296 | 1.89 | 1.89 |
| FinPlan v3 无路径 | 5/9 | .704 | .296 | 2.33 | 2.33 |
| HiREC-style + 路径绑定 | 8/9 | .963 | .324 | 2.56 | 3.44 |
| Generic + 路径绑定 | 9/9 | 1.000 | .204 | 3.00 | 3.00 |
| 仅元数据分解 | 9/9 | 1.000 | .000 | 2.33 | 2.33 |
| **FinPlan v3 路径绑定** | **9/9** | **1.000** | **.000** | **2.33** | **2.33** |

### 5.2 冻结留出集（8 题）

| 方法 | 完整闭包 | Leg Recall | 错误文档率 | 平均动作 | 平均文件 |
|---|---:|---:|---:|---:|---:|
| Single-shot | 6/8 | .917 | .406 | 1.00 | 4.00 |
| 固定分解 | 4/8 | .792 | .208 | 2.63 | 2.63 |
| Generic Adaptive | 4/8 | .813 | .104 | 2.38 | 2.38 |
| HiREC-style | 6/8 | .896 | .198 | 2.00 | 3.00 |
| FinPlan v1 | 4/8 | .708 | .229 | 2.25 | 2.25 |
| FinPlan v3 无路径 | 4/8 | .792 | .208 | 2.63 | 2.63 |
| HiREC-style + 路径绑定 | 8/8 | 1.000 | .146 | 2.13 | 3.13 |
| Generic + 路径绑定 | 8/8 | 1.000 | .031 | 2.75 | 2.75 |
| 仅元数据分解 | 8/8 | 1.000 | .000 | 2.63 | 2.63 |
| **FinPlan v3 路径绑定** | **8/8** | **1.000** | **.000** | **2.63** | **2.63** |

### 5.3 第二验证批次（30 题、67 份完全独立 SEC 文件）

该批次在方法冻结后运行，未依据结果调参。27 题属于比较调查，3 题属于会计推导。

| 方法 | 完整闭包 | Leg Recall | 错误文档率 | 平均动作 | 平均文件 |
|---|---:|---:|---:|---:|---:|
| Single-shot | 24/30 | .911 | .475 | 1.00 | 3.93 |
| 固定分解 | 22/30 | .856 | .144 | 2.23 | 2.23 |
| Generic Adaptive | 22/30 | .861 | .144 | 2.30 | 2.30 |
| HiREC-style | 27/30 | .950 | .183 | 1.73 | 2.73 |
| FinPlan v1 | 20/30 | .800 | .161 | 2.10 | 2.10 |
| FinPlan v3 无路径 | 22/30 | .856 | .144 | 2.23 | 2.23 |
| Generic + 路径绑定 | 30/30 | 1.000 | .056 | 2.40 | 2.40 |
| HiREC-style + 路径绑定 | 30/30 | 1.000 | .153 | 1.73 | 2.73 |
| 仅元数据分解 | 30/30 | 1.000 | .000 | 2.23 | 2.23 |
| **FinPlan v3 路径绑定** | **30/30** | **1.000** | **.000** | **2.23** | **2.23** |

路径绑定后的 FinPlan 与三个路径绑定/元数据强基线都达到文件闭包上限；这再次说明当前方法的独立贡献尚未被文件级实验识别。相反，无路径的 FinPlan v1、Generic Adaptive、固定分解和 FinPlan v3 无路径分别只有 20/30、22/30、22/30、22/30，提供了比 8 题留出更稳定的 gap 诊断。

按模板分层后，第二批的差异并非由单一轨道造成：

| 模板 | 题数 | FinPlan v1 闭包 | FinPlan v3 无路径闭包 | FinPlan v3 路径绑定闭包 | HiREC-style 闭包 |
|---|---:|---:|---:|---:|---:|
| 比较调查 | 27 | 20/27（.741） | 20/27（.741） | 27/27（1.000） | 24/27（.889） |
| 会计推导 | 3 | 0/3 | 2/3 | 3/3 | 3/3 |

会计推导样本仍然太少，不能据此声称跨模板统计稳定性；但它提示 v1 的“实体级而非实体—年份级义务”在多年份财务任务上尤其脆弱。比较调查则提供了主要的独立验证样本，说明无路径规划缺口不是收购关系特例。

所有时间感知方法 future leakage 为 0。`FinPlan –time` 也为 0，因为精确公司—年份路径键已经唯一确定文件；因此本轨道的时间消融不可识别，不能把“无泄漏”解释成时间字段无用。点时必要性由另一个 SEC 生命周期回放实验识别。

### 5.4 不确定性与配对检验

对冻结 case-level 闭包结果进行事后 Wilson 区间和 exact McNemar 审计：

| 数据 | FinPlan v3 闭包及 Wilson 95% CI | 相对 v1 的配对 p | 相对 HiREC-style 的配对 p | 相对元数据分解的配对 p |
|---|---:|---:|---:|---:|
| 探索集 | 9/9；[.701, 1.000] | .0625 | .125 | 1.000 |
| 冻结留出 | 8/8；[.676, 1.000] | .125 | .500 | 1.000 |
| 合并描述性 | 17/17；[.816, 1.000] | .0039 | .0313 | 1.000 |

探索集参与了架构修改，不能作为独立确认性检验；合并结果也不能消除这种选择偏差。初始冻结留出只有 8 题，功效不足。第二验证批次的 30 题与第一批文件完全独立，FinPlan v1 相对路径绑定 v3 的 exact McNemar (p=.00195)，Generic Adaptive 为 (p=.00781)，但 HiREC-style 为 (p=.25)，元数据分解为 (p=1)。因此本轮可以较有力地说“无路径的文件闭包缺口在独立批次中复现”，仍不能说 FinPlan 相对强 HiREC 或元数据上界已获得确认性算法优势。

### 5.5 Dense 与 Hybrid 稳健性

为检验 BM25 特异性，使用 Hugging Face 镜像下载冻结 revision 的 `BAAI/bge-small-en-v1.5`。当前机器只有 CPU，因此预注册的 dense 协议对每份 10-K 均匀抽取 12 个 passage，使用 attention-mask mean pooling、余弦相似度和文档内 max score；hybrid 用 RRF（k=60）融合全 passage BM25 排名与该 BGE 文档排名。两者均保持 4 次动作/4 份文件预算、同一 cutoff 和路径约束。

| 方法 | BM25 闭包 | BGE dense 闭包 | RRF hybrid 闭包 |
|---|---:|---:|---:|
| Single-shot | 24/30 | 22/30 | 27/30 |
| 固定分解 | 22/30 | 18/30 | 21/30 |
| Generic Adaptive | 22/30 | 14/30 | 22/30 |
| HiREC-style | 27/30 | 22/30 | 27/30 |
| FinPlan v1 | 20/30 | 16/30 | 19/30 |
| FinPlan v3 无路径 | 22/30 | 18/30 | 21/30 |
| Generic + 路径绑定 | 30/30 | 30/30 | 30/30 |
| HiREC-style + 路径绑定 | 30/30 | 29/30 | 29/30 |
| 仅元数据分解 | 30/30 | 30/30 | 30/30 |
| **FinPlan v3 路径绑定** | **30/30** | **30/30** | **30/30** |

三种检索设置均复现无路径闭包缺口，说明现象不是 BM25 独有。Dense/Hybrid 中 FinPlan v3 相对未绑定 Generic 的配对 p<.01，但相对元数据分解、Generic 路径绑定和 HiREC 路径绑定均不能拒绝等价结果。由于 dense 只抽取 12 个 passage，且没有 cross-encoder reranker，该实验是检索器稳健性诊断，不是完整 dense/hybrid SOTA 复现。

### 5.6 从 oracle 路径到非 oracle 义务规划

第二批 30 题首先用于检验“给定正确公司—年份路径”与“系统自己生成路径”之间的差距。Oracle 路径闭包为 30/30；透明 registry rule 为 19/30，Qwen2.5-0.5B 单体自由规划仅为 3/30。Qwen 的 JSON 解析率为 93.3%，但主要失败是漏主体、错 ticker 和错年份；在第三批运行时还输出了整数 years 字段并触发旧解析器崩溃。该结果说明 oracle 30/30 不能外推为可部署系统能力，schema 约束也只能修复格式，不能自动补齐金融义务。

据此形成的 v4 把规划拆为注册表候选提出、稀有 token/全名覆盖校验、filing-year 义务生成和路径绑定检索。它在参与开发的第二批达到 30/30；在随后冻结的第三批 company/file-OOD 数据上结果如下：

| 方法 | 主体 exact | 义务 exact/覆盖 | 级联闭包 | 错误文档率 |
|---|---:|---:|---:|---:|
| v1 registry rule | 18/22 | 15/22 | 15/22 | 0 |
| 冻结 v4 | 21/22 | 18/22 | 18/22 | 0 |

v4 新增解决 3 题且没有回退，但只有 3 个不一致对，双侧 exact McNemar p=.25；闭包率的 Wilson 95% 区间为 [.615, .927]，证据不足以声称稳定算法优势。失败分别来自事件时间到 filing 年份映射、损坏年份 20222024、过去三年展开和 O'Reilly 别名覆盖。

### 5.7 v5/v6 时间规划迭代及新的冻结反证

v5 增加 cutoff 可见 filing registry、损坏年份修复、相对窗口和标点别名规范化。在第三批错误驱动回放中由 18/22 达到 22/22，在第二批回归保持 30/30，但这些都不是新确认性结果。随后从从未使用的旧年份 LOFin 问题构建 validation4：6 题、12 份 SEC 10-K，与前三批文件零重叠，并在运行方法前冻结。

在 validation4 上，v1、v4 和冻结 v5 的闭包均为 3/6，v5 相对 v4 的不一致对为 0，p=1；95% Wilson 区间为 [.188, .812]。失败来自 recent N-year period、N years ago 和 ExxonMobil 有空格/无空格别名。该结果直接否定了“v5 已形成通用金融时间规划”的主张。

v6 增加相对时间边界和金融别名等价，在 validation4 的错误驱动开发回放达到 6/6，并在第二、三批回归保持完整闭包。由于修改使用了 validation4 失败，6/6 只证明机制能够解释已观察错误，仍需新数据。

### 5.8 混合 10-K/10-Q：实体—年份表示的真实碰撞

为检验框架是否只适用于年度报告，validation5 按 qid 升序选择 8 个此前未使用的问题、21 份真实 SEC 文件和 14 家公司；题间公司与文件均不重叠，至少包含一份 10-Q。构建器用相邻两份 10-K 的财政年度边界定位 Q1–Q3，避免把 Oracle、惠普等非自然年公司的财政季度误映射为日历季度。数据、注册表、构建器与 v6 源码在运行前均写入冻结哈希。

冻结非 oracle v6 的主体 exact 为 8/8，但显式义务召回为 0，级联闭包为 4/8。原因不是主体漏识别，而是旧义务键只有“公司—年份”：同年 Q1、Q2、Q3 与 FY 被折叠成一个节点。跨公司且每家公司只有一份季度文件的题目有时可被路径键偶然检全；同公司多季度和年度/季度混合问题则无法表达完整闭包。

同批透明方法结果为：

| 方法 | 文件闭包 | Leg Recall | 错误文档率 |
|---|---:|---:|---:|
| FinPlan v1 | 2/8 | .521 | .229 |
| FinPlan v3 无路径 | 2/8 | .583 | .229 |
| FinPlan v3 路径绑定 | 5/8 | .792 | 0 |
| Generic path-bound | 5/8 | .875 | .125 |
| Single-shot | 6/8 | .854 | .438 |
| HiREC-style | 6/8 | .917 | .323 |
| HiREC-style path-bound | 7/8 | .958 | .177 |

FinPlan v3 路径绑定闭包的 Wilson 95% 区间为 [.306, .863]，HiREC-style path-bound 为 [.529, .978]；两者只有 2 个不一致对，双侧 exact McNemar p=.5。样本不足以确认 HiREC 的总体优势，但这组结果满足“原框架比自身早期版本好、却在该批描述性结果上弱于现有改进思路”的架构重做条件。HiREC-style 仍是本项目的透明近似，不是原作者复现；因此不能据此排序真实 SOTA，但足以否定当前 FinPlan 已领先的内部主张。

v7 将文件义务扩展为“主体、filing type、财政年度、财政期间”，并把 Q4 映射为 FY 10-K 义务。在 validation5 的错误驱动开发回放中，主体、义务和闭包均为 8/8，错误文档与 future leakage 均为 0。由于 v7 正是根据该冻结集失败开发，8/8 不是确认性结果。

### 5.9 新冻结 validation6：v7 未获得策略优势

在 v7 冻结后构建 validation6：8 题、14 份前五批从未使用的 SEC 文件、8 家公司；题间公司和文件均不重叠。分层为 1 道年度多文件题、2 道季度多文件题和 5 道季度单文件 company-OOD 题。

| 方法 | 文件闭包 | 错误文档率 |
|---|---:|---:|
| v7 period-aware FinPlan | 6/8 | 0 |
| Single-shot | 8/8 | .5625 |
| Period metadata decomposition | 6/8 | 0 |
| Generic adaptive period | 7/8 | .15625 |
| HiREC-style period | 7/8 | .40625 |

三道多 filing 题上所有方法均为 3/3。五道季度单文件 company-OOD 中，v7/metadata 为 3/5，Generic/HiREC-style 为 4/5；single-shot 虽为 5/5，但通过读取固定 4 份文件取得，错误文档率高达 .5625。样本太少，不能检验多 filing 总体优势；总体结果也明确说明 v7 没有超过同元数据强基线。

两个失败需要区别对待：`financebench_00288` 题面只有 “FY2023 与 FY2024 Q2”，没有公司名称或 ticker，开放域 planner 不应利用 gold 反推 Best Buy；`financebench_00407` 已正确识别 MGM，但 v7 未把 H1 FY2023 映射到 Q2 10-Q。冻结结果的 entity 指标从最终义务反推主体，因此后者被误计为 entity failure；冻结文件保持不变，后续审计必须把 entity extraction 与 period mapping 分开。

若加入 H1→Q2、H2→FY 的 v8，它只能是 validation6 错误驱动开发版，必须再用新来源冻结验证。当前裁决是：period-aware 表示必要，但 FinPlan 的动作策略尚未证明比同元数据 Generic/HiREC-style 更优。

### 5.10 自建 validation7：冻结 14 题上 v8（半年度别名）闭包 11/14，首次显著优于元数据分解

LOFin 测试子集的全部同公司多 filing 题（48 行/24 题）公司与前六批的 114 个 ticker 全部重叠，不存在新鲜来源，因此按方案自行构建 validation7：6 家新公司（MDLZ/TMUS/CAT/RTX/UPS/CTVA，均 Dec-31 财年、有收入线）× (Q1/Q2/Q3 2024 10-Q + FY2024 10-K + Q1 2025 10-Q) = 30 份此前从未使用的 SEC 文件。问题模板在读取任何金标数字之前按分析师自然措辞固定（h1 上半年→Q2 10-Q、h1_9m 上半年与前三季度→Q2+Q3 10-Q、q1q2 一季与二季→Q1+Q2 10-Q、h2 下半年=全年−前三季度→Q3 10-Q+FY 10-K、fy_q1n 全年与下年一季→FY 10-K+Q1 10-Q）；gold 文件由 XBRL companyfacts 事实的 accn 必须匹配所下载文件的 accession，且数值必须在语料文本收入标签 ±250 字符内可定位，双验证决定，不按任何检索方法的行为设计。18 题 = dev 4 + frozen 14；manifest 在方法运行前冻结，builder 产出的数据、两个 registry 与四个 runner 的 SHA-256 全部先于任何方法写入 `nonoracle_obligation_v7_validation7_freeze.json`。

v8 = v7 + 半年度别名映射（H1→Q2 10-Q、H2→Q3 10-Q + FY 10-K），该映射与 v7 在别名题上的预期失败均先在 manifest 的 interpretation_boundary 中预声明，v8 runner 在冻结运行前写完哈希，不是看到冻结结果后的调优。

冻结 14 题闭包（gold ⊆ 检索文件集，各方法定义一致）：

| 方法 | 文件闭包 | 错误文档题数 | 平均查询 |
|---|---:|---:|---:|
| v8（v7+半年度别名） | **11/14** | 2 | 1.71 |
| v7 period-aware | 5/14 | 2 | 1.29 |
| entity-year（v6） | 3/14 | 7 | 1.21 |
| Single-shot | 8/14 | 14 | 4.00 |
| Period metadata decomposition | 5/14 | 2 | 1.29 |
| Generic adaptive period | 8/14 | 8 | 2.00 |
| HiREC-style period | 8/14 | 13 | 2.79 |

配对精确 McNemar（双尾，n=14，探索性、未多重校正）：v8 闭包显著高于 v7（6 个差异对全部偏向 v8，p=0.031）、entity-year（p=0.022）与 metadata 分解（p=0.031）；与 Generic/HiREC-style/Single-shot 的差异（11 对 8）为方向性（p=0.25）。错误文档出现次数上 v8 显著低于 Single-shot（12 对 0，p<0.001）、HiREC-style（11 对 0，p=0.001）与 Generic（6 对 0，p=0.031），与 metadata 持平（均 2 题）。

v8 的 6 个新闭包恰好全部落在预声明的别名层：3 道 h1 题（TMUS/CAT/UPS-h1）与 3 道 h2 题（TMUS/CAT/UPS-h2）均从 0 到 1；h2 推导（H2=FY−9M 需同时取 Q3 10-Q 与 FY 10-K）对所有非 v8 方法都是失败面（Generic/HiREC/metadata 0/3，Single-shot 仅 UPS-h2 靠 4 文件全取侥幸 1/3）。显式季度题（h1_9m/q1q2）v7 与 v8 均为 5/5。因此 validation6 预声明的 H1→Q2、H2→FY 映射在冻结真实数据上被确认，这是本框架首次在同一冻结集上闭包显著超过同元数据分解基线。

必须同时报告的负结果：

1. **fy_q1n 层（跨财政年度边界）全部方法失败**：v8 0/3、baselines 0/3（v4 的 1/3 靠年腿侥幸）。题面 "fiscal year 2024 … first quarter of 2025" 中 "fiscal year 2024" 没有被任何 period 解析规则识别（annual fallback 只在无任何 period 命中时触发），FY 10-K 义务系统性丢失。"fiscal year N" 短语进入义务键是明确的下一步缺口。
2. **v8 在 h1_9m 题上的过度检索**：既有 "first n quarters" 规则把 Q1 一并纳入计划，导致 TMUS/RTX-h1_9m 各 1 个错误文档（闭包仍为 1.0）。累积期应只取 Qn 所在文件；该修正未预声明，不在本冻结集上改动。
3. 文件闭包仍不是端到端答案正确率；自建集与 LOFin 原题分开报告；样本 14 题不支撑 SOTA 声明。

结论收紧为：period-aware 表示必要；在"同公司同年多 filing 期间碰撞 + 半年度自然措辞"这一自建冻结面上，预声明的期间别名映射使 FinPlan 首次在同一冻结集上以显著差异超过元数据分解与自身 v7，但相对 Generic/HiREC-style 的闭包差异仍不显著（方向有利），且跨年度边界与答案层仍未解决。

### 5.11 A4（–Available time）消融：移除 `available_at ≤ cutoff` 后，v8 在冻结 14 题上完全不变，三个基线显著泄漏未来文件

消融矩阵（方案 v2 第 4 节）的 A4 曾标为"SEC 回放完成；LOFin 当前不可识别"——因为此前 LOFin 各批语料不含 cutoff 之后的文件。validation7 语料天然含未来文件（相对 2024 年 cutoff：各公司 Q3-2024 10-Q、FY2024 10-K 与 Q1-2025 10-Q），使 A4 在该冻结集上可执行。

操作化（预声明，`lofin_validation7_ablation_time_freeze.json` 在运行前写入）：no_time 变体只移除 `available_at ≤ cutoff` 约束——planning 层 `visible_filings` 以 cutoff=None 调用、retrieval 层 `index.search(..., allow_future=True)`；语料、索引、模板、预算、义务/期间解析规则、BM25 均不变。time 变体重放冻结管线，必须逐字段复现冻结结果（确定性检查：0 差异，全部 5 个方法 × 3–4 字段逐一核对）。`future_leak` 为严格大于：available_at 恰等于 cutoff 的 gold filing（fy_q1n 的 Q1-2025 10-Q）不算泄漏。frozen 14 题一次性运行，dev 4 题仅诊断。预声明预测 P1–P4 及裁决如下。

冻结 14 题上 no_time vs time（配对精确 McNemar / sign test）：

| 方法 | 闭包 time→no_time | future_leak no_time | 泄漏 p | 闭包损失 p |
|---|---|---|---:|---:|
| v8 cascade | 11/14 → 11/14 | 0/14 | 1.0（无变化） | 1.0（无变化） |
| Single-shot | 8/14 → 4/14 | 10/14 | 0.002 | 0.125 |
| Period metadata decomposition | 5/14 → 5/14 | 3/14 | 0.25 | 1.0 |
| Generic adaptive period | 8/14 → 6/14 | 7/14 | 0.016 | 0.50 |
| HiREC-style period | 8/14 → 7/14 | 7/14 | 0.016 | 1.0 |

v8 的 no_time 行在 case 级全部字段与 time 行逐字段相同（含 planned_obligations、documents、wrong_doc_rate），即 A4 在 v8 上是完全 no-op。三个基线在去掉时间门后未来泄漏显著（Single-shot p=0.002、Generic/HiREC p=0.016）；Single-shot 的错误文档在全部 6 个非平局配对中增加（sign test p=0.031）。no_time 体制下的跨方法对比：v8 泄漏 0/14 对 Single-shot 10/14（p=0.002）、Generic 7/14（p=0.016）、HiREC 7/14（p=0.016）；闭包 11 对 4（p=0.016）、6（p=0.063）、7（p=0.125）。

机制与预声明裁决：

- **P1（v8 不变）确认**：v8 的防泄漏是结构性的，不靠时间门本身——显式期间解析（含别名）使义务计划不需要 annual fallback 去"看见"未来的 10-K；precise_path 绑定使 retrieval 只能在义务自身文档内命中。
- **P2（Single-shot 泄漏）确认**：Q1-2025 10-Q 含 Q1-2024 同期比较与 FY2024 全年度数据，去掉时间门后 BM25 把未来文件取回，10/14 例泄漏，且闭包 8→4（未来文件挤占预算）。
- **P3（metadata 分解无泄漏）被证伪**：其检索保持路径绑定，但 planning 层 no_time 使共享 v7 规则的 annual fallback 能看见未来 10-K——"first half of 2024" 在 v5 规则中未解析 → 无显式期间 → annual fallback → FY2024 10-K 仅在 no_time 下成为义务 → 路径绑定检索忠实地取回该未来文件（3/14）。泄漏发生在 planning 层而非 retrieval 层，我的预声明推理不完整。
- **P4（Generic/HiREC 仅 initial 泄漏）部分被证伪**：泄漏同时来自无绑定 initial 查询与基于 fallback 义务的 fill_missing。

解读边界（诚实分层）：

1. 这是**稳健性消融**，不是新的 SOTA 主张：时间门开启时 v8 相对 Generic/HiREC-style 的闭包差异仍是方向性（p=0.25）。no_time 下的显著差异（p=0.016 对 Single-shot）说明 v8 的优越性不以时间过滤为前提，但正式优势主张仍受主对比约束。
2. 该消融把"时间门承担了多少保护"从 v8 中剥离：对 v8 承担 0（结构性保护），对三个基线承担显著保护（3 个方法泄漏 p≤0.016）。这与 SEC 真实 pilot 的 A4 结果方向一致（finplan_no_time 在探索/留出集上 future_leak 0.35/0.40、overclaim 增加），且首次在预声明冻结集上给出配对检验。
3. A3（–Limiting evidence）在本轨道不可执行：period 轨道的义务管线（v7/v8 与全部基线）均无 L 机制，没有可删除对象；A3 的真实载体是 SEC lineage pilot（finplan_no_limit vs finplan_v3_path_bound），该 pilot 未走冻结清单协议，本会话对其追加了**事后配对统计审计**（`sec_real_pilot_ablation_audit_v1.json`，明确标注 post-hoc、仅描述性）：探索集（17 个 lineage×slice 计分单元）闭包 17→15（p=0.5）、limit_action_recall 0.5→0.0、limiting_recall 不变；留出集（12）闭包 10→7（p=0.25）、limit_action_recall 0.5→0.0、limiting_recall 1.0→0.5。A4 在同一审计中：探索集 future_leak 0→10/17（p=0.002，即文献记录的 58.8%）、留出集 0→5/12（p=0.0625，即 41.7%）——与 README 已记录的泄漏率逐字吻合，配对检验后探索集显著、留出集方向性。同一审计的 A5 邻近证据（wrong_lineage）：路径绑定方法跨谱系率 0–0.12，而 single_shot 17/17（1.0）、HiREC-style 10/17（0.59）探索集、6/12（0.5）留出集——但这不是完整的 –Lineage 消融（pilot 没有单独删除 P 维度而其余不变的对照），仍只作描述性证据。A6（–Conflict action）因载体数据（比较调查轨道、来源冲突案例）未构建，在真实冻结集上不可执行。

## 6. 对 research gap 的支持强度

本实验提供两项直接证据：

1. 在探索和冻结留出中，固定分解、Generic Adaptive、HiREC-style 及 FinPlan v1 均不能稳定闭合全部真实 SEC 文件；即使候选公司—年份义务已给定，未绑定路径的检索仍会遗漏或串入其他文件。
2. 把义务原子化并在检索入口绑定公司—年份后，10-K 文件闭包和轨迹纯度明显改善，说明“独立路径/义务必须进入检索约束”是实际机制需求，而非只存在于概念图中。
3. 混合 10-K/10-Q 冻结集进一步显示，公司—年份仍不是充分原子：主体识别完整时，旧义务召回仍为 0，并在同公司多季度上发生闭包失败。由此支持的是 filing/period-aware 状态表示缺口，而不是笼统的“Agent 不会规划”。

但实验也明确否定了一个更强主张：FinPlan v3 没有超过“仅元数据分解”上界；Generic/HiREC 获得相同路径绑定后也闭包全部留出题。因此真实、可辩护的结论是：

> 无路径绑定的代表性规划实现，在该真实金融多文档任务和文件级表示下出现了可复现的证据闭包缺口；路径绑定可以消除该缺口，但当前实验尚未证明完整 FinPlan 状态机相对所有强路径绑定基线具有额外算法增益。

在混合 filing 轨道上还必须追加：

> 实体—年份义务无法区分同年多个财政期间；该表示缺口在冻结真实 SEC 数据中直接出现。period-aware v7 能在开发回放闭合这些文件，但尚未在新任务上证明超过强补充检索基线。

validation6 运行后应进一步收紧为：

> period-aware v7 在新冻结数据上闭合 6/8，与元数据分解持平并低于 Generic/HiREC-style period 的 7/8；三道多 filing 题均闭合，但数量不足。故当前只确认表示问题真实存在，不确认 FinPlan 策略增益。

validation7（自建冻结 14 题）运行后再次收紧为：

> v8（v7 + 预声明半年度别名 H1→Q2、H2→Q3+FY）闭包 11/14，显著高于自身 v7 与元数据分解（p≈.03），错误文档显著低于 Single-shot/HiREC-style；相对 Generic/HiREC-style 的闭包差异仍只有方向性。别名层是 v8 全部增益的来源；跨年度边界（fy_q1n）三法全败。表示必要已确认，策略优势在"自然期间措辞"这一面上首次部分确认，仍不是 SOTA。

A4 消融（预声明，frozen 14 一次性运行）后追加：

> 移除 available_at 约束后，v8 在 case 级完全不变（0 未来泄漏、闭包 11/14 不变、错误文档不变），三个基线的未来泄漏显著上升（Single-shot 10/14、Generic/HiREC 7/14，配对 p≤0.016）且 Single-shot 闭包 8→4。v8 的防泄漏是结构性（期间解析+路径绑定），不依赖时间门；no_time 体制下 v8 闭包显著超过 Single-shot（11 对 4，p=0.016），但正式优势主张仍受时间门开启时主对比（p=0.25）约束。预声明 P3 被证伪（metadata 分解在 planning 层经 annual fallback 泄漏未来 10-K），如实记录。A3 在本轨道无 L 机制可删（SEC pilot 已有 finplan_no_limit 证据）；A5/A6 载体数据未构建。

这仍是“在该任务和表示下验证”的结论，不是“已有方法做不到”。特别是，强通用 agent 完全可能学会或调用相同的元数据过滤器。

## 7. 对论文框架的实际约束

- FinPlan v1 必须淘汰：它在两条非收购轨道上没有把多年份义务原子化。
- 公司—年份版 v3–v6 也不能作为最终框架：它们无法表达同公司同年多季度或年度/季度混合义务。
- 路径绑定是必要工程机制，但不能单列为足够的新算法贡献；现行候选方法必须使用 filing/period-aware 义务键。
- 金融域通用性应定义为共享状态接口加任务模板，而不是共享一个查询模板。
- 下一阶段必须在新的混合 filing 冻结集评价 planner 是否能从问题和公共注册表生成、修订并闭合主体—类型—年度—期间—指标义务。
- 必须加入真实 reader 或 LLM，评价段落证据、计算、引用和继续/回答/拒答；文件闭包只是必要条件。
- 正式比较仍需复现原版 HiREC/Search-R1 或得到足够接近的公开强实现，并使用 dense/hybrid retriever。
- 若在非 oracle、端到端实验中 FinPlan 仍只与元数据分解持平，论文应收缩为金融证据状态 benchmark/审计协议，而非 SOTA 规划算法。

## 8. 可复现文件

- `preexperiments/build_lofin_multidoc_pilot.py`
- `preexperiments/run_lofin_multidoc_pilot.py`
- `preexperiments/data/lofin_multidoc_pilot_v1.json`
- `preexperiments/lofin_multidoc_freeze_v1.json`
- `preexperiments/results/lofin_multidoc_pilot_exploratory_v1.json`
- `preexperiments/results/lofin_multidoc_pilot_exploratory_v2.json`
- `preexperiments/results/lofin_multidoc_pilot_exploratory_v3.json`
- `preexperiments/results/lofin_multidoc_pilot_holdout_v1.json`
- `preexperiments/lofin_multidoc_validation2_manifest.json`
- `preexperiments/lofin_multidoc_validation2_freeze_v1.json`
- `preexperiments/data/lofin_multidoc_validation2_v1.json`
- `preexperiments/results/lofin_multidoc_validation2_v1.json`
- `preexperiments/results/lofin_multidoc_validation2_statistical_audit_v1.json`
- `preexperiments/analyze_lofin_multidoc_statistics.py`
- `preexperiments/results/lofin_multidoc_statistical_audit_v1.json`
- `preexperiments/run_lofin_multidoc_dense.py`
- `preexperiments/lofin_multidoc_dense_freeze_v1.json`
- `preexperiments/results/lofin_multidoc_validation2_dense_v1.json`
- `preexperiments/results/lofin_multidoc_validation2_dense_statistical_audit_v1.json`
- `preexperiments/run_lofin_multidoc_hybrid.py`
- `preexperiments/lofin_multidoc_hybrid_freeze_v1.json`
- `preexperiments/results/lofin_multidoc_validation2_hybrid_v1.json`
- `preexperiments/results/lofin_multidoc_validation2_hybrid_statistical_audit_v1.json`
- `preexperiments/test_lofin_multidoc_pilot.py`
- `preexperiments/lofin_validation6_manifest.json`
- `preexperiments/build_lofin_validation6.py`
- `preexperiments/data/lofin_validation6_v1.json`
- `preexperiments/nonoracle_obligation_v7_validation6_freeze.json`
- `preexperiments/results/nonoracle_obligation_v7_validation6_frozen.json`
- `preexperiments/run_period_aware_baselines.py`
- `preexperiments/results/period_aware_baselines_validation6.json`
- `preexperiments/lofin_validation7_manifest.json`
- `preexperiments/build_lofin_validation7.py`
- `preexperiments/data/lofin_validation7_v1.json`
- `preexperiments/results/nonoracle_obligation_v7_validation7_freeze.json`
- `preexperiments/results/nonoracle_obligation_v7_validation7_frozen.json`
- `preexperiments/results/nonoracle_obligation_v8_validation7_frozen.json`
- `preexperiments/results/nonoracle_obligation_v6_validation7_frozen.json`
- `preexperiments/results/period_aware_baselines_validation7.json`
- `preexperiments/results/lofin_validation7_statistical_audit_v1.json`
- `preexperiments/run_validation7_ablation_time.py`（A4 消融 runner，预声明）
- `preexperiments/audit_validation7_ablation_time.py`（A4 统计审计）
- `preexperiments/results/lofin_validation7_ablation_time_freeze.json`（运行前冻结）
- `preexperiments/results/validation7_ablation_time_dev.json`（dev 诊断）
- `preexperiments/results/validation7_ablation_time_frozen.json`（frozen 14 一次性运行）
- `preexperiments/results/validation7_ablation_time_audit_v1.json`（统计审计）
- `preexperiments/audit_sec_real_pilot_ablations.py` + `preexperiments/results/sec_real_pilot_ablation_audit_v1.json`（SEC pilot A3/A4 事后配对审计 + A5 邻近 wrong_lineage 证据）
- `preexperiments/run_nonoracle_obligation_planning_v6.py`
