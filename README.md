---
title: FinPlan-RAG：关系约束的证据状态规划金融 RAG
status: empirical-gap-validated-method-not-confirmed
version: 1.1
updated: 2026-08-10
---

# FinPlan-RAG：关系约束的证据状态规划金融 RAG

本项目研究一个经过现有工作收窄的可检验问题：主动 RAG 已能决定何时检索，Agentic RAG 已能进行多轮搜索，金融 HiREC 已能判断证据是否充分并生成补充问题；但在点时金融关系判断中，规划状态通常仍隐含在自然语言轨迹或二元 answerability 中，尚未被系统建模为多条关系路径、路径级证据义务、限制证据、来源谱系与可得时间共同约束的检索决策状态。

FinPlan-RAG 将任务表述为受约束序贯决策：规划器在同一语料、索引、关系模板和预算下，根据显式证据状态选择下一检索动作，并决定继续、断言或拒答。关系模板只定义合法动作和最低证据义务；可学习策略负责动作排序和停止，最终主张必须由真实公开金融文档回放建立。

## 一句话论证

> 在点时金融关系问答中，我们检验关系约束的显式证据状态是否能使 RAG 规划器比单次检索、固定问题分解、通用动态 RAG 和自由搜索智能体更有效地闭合多跳证据、发现限制证据，并以更低的过度断言和检索成本决定继续、断言或拒答。

## 当前证据边界

- 文献层面：通用“会规划、会多轮检索”不是 gap；本文只主张一个任务与表示层面的待验证差异。
- 受控预实验：22,400 个合成 case 中，显式状态规划相对固定分解提高闭包并减少约 1.11 次查询；相对公平通用自适应基线的准确率差为 −0.00018，不能主张准确率优势。限制证据降低过度断言但牺牲覆盖与诊断效用。
- 当前不得声称：FinPlan-RAG 已优于通用 Agentic RAG、原版 HiREC 或强时间 RAG；也不得把文件闭包等同于最终答案正确率。
- 真实数据门：SEC EDGAR 点时关系回放已在小样本上复现 future leakage、轨迹污染及 v1 失败后的结构改进；仍需扩大样本和复现原版强基线。
- 小规模真实 pilot：15 条 SEC 交易 lineage 上，`–available_at` 产生 41.7%–58.8% future leakage；原 FinPlan v1 低于通用自适应/HiREC-style 诊断，加入路径绑定与机制化动作查询的 v3 在探索集和新 lineage 留出集上改善准确率与闭包。2026-08-11 追加事后配对审计（描述性，pilot 未走冻结清单协议）：A4 future_leak 探索 0→10/17（p=0.002）、留出 0→5/12（p=0.0625）；A3 移除限制证据动作后 limit_action_recall 0.5→0、闭包留出 10→7（p=0.25）；A5 邻近 wrong_lineage 路径绑定 0–0.12 vs single_shot 1.0/HiREC-style 0.5–0.59。该结果仍不是 SOTA 证据。
- 金融域多轨道：LOFin 关联实验现已覆盖五批真实 SEC 文件。第二批独立 10-K 验证中，无路径闭包缺口在 BM25、BGE dense 和 RRF hybrid 上复现，但路径绑定 FinPlan 与元数据/Generic 强基线持平。第三批 company/file-OOD 非 oracle 规划中，旧规则闭包 15/22，冻结 v4 为 18/22，exact McNemar p=.25，方向改善但证据不足。
- 新的真实表示缺口：公司和文件均不重叠的混合 10-K/10-Q 冻结集含 8 题、21 份文件、14 家公司。冻结 v6 的主体识别为 8/8，但显式义务召回为 0、闭包仅 4/8；实体—年份状态把同年不同季度折叠。同期 FinPlan v1、FinPlan v3 路径绑定和 HiREC-style 路径绑定闭包分别为 2/8、5/8、7/8。因此原框架虽优于自身 v1，却弱于强改进基线，不能声称 SOTA。
- 当前方法状态：把义务扩展为“主体、filing type、财政年度、财政期间”四元键的 v7 在 validation5 错误驱动回放达到 8/8；新冻结 validation6 只达 6/8，未超过同元数据强基线。validation7 自建冻结集上，v8（v7 + 预声明半年度别名映射 H1→Q2、H2→Q3+FY）闭包 11/14，首次在同一冻结集上显著超过元数据分解与自身 v7（p≈.03），但相对 Generic/HiREC-style 仅方向有利；同一冻结集 A4 消融（预声明一次性）显示 v8 在移除 available_at 后 0 未来泄漏、case 级不变（防泄漏结构性），Single-shot/Generic/HiREC 显著泄漏（p≤.016）；A3 在本轨道无 L 机制可删（SEC pilot 已有 finplan_no_limit 证据），A5/A6 载体数据未构建；"fiscal year N" 跨年短语解析与真实 reader/LLM、页级证据、原版 HiREC/Search-R1、停止/拒答仍未完成。
- 新冻结 validation6：8 题、14 份此前未使用 SEC 文件、8 家公司。v7 文件闭包 6/8；同元数据 period-aware decomposition、Generic、HiREC-style 分别为 6/8、7/8、7/8。三道多 filing 题均为 3/3，但样本过少；两个失败分别来自题面缺失主体和 H1→Q2 期间映射。该结果确认 v7 尚无策略优势，并暴露现有 entity 指标把“主体已识别、期间失败”误计为主体失败的问题。
- 自建 validation7（与 LOFin 原题分开报告）：LOFin 测试子集全部同公司多 filing 题的公司与前六批重叠，故用 6 家新公司 × 5 份同期文件 = 30 份此前未用 SEC 文件自建 18 题（dev 4 / frozen 14），gold 由 XBRL accn 匹配 + 语料文本定位双验证。冻结 14 题中，v8（v7 + 预声明半年度别名 H1→Q2、H2→Q3+FY）闭包 11/14，显著高于 v7 的 5/14、元数据分解的 5/14 与 entity-year 的 3/14（配对精确 McNemar p≈.02–.03），错误文档显著低于 Single-shot/HiREC-style（p≤.001）；相对 Generic/HiREC-style 的闭包 11 vs 8 只有方向性（p=0.25）。v8 的 6 个新闭包全部来自别名层；h2（H2=FY−9M 推导）对全部非 v8 方法仍是失败面。负结果：跨年度边界层（fy_q1n，3 题）所有方法全败（"fiscal year N" 短语未进入义务键），h1_9m 因既有 "first n quarters" 规则多取 Q1 产生 2 个错误文档。同一冻结集上的 A4（–Available time）消融（预声明、一次性运行）：移除 available_at 约束后 v8 在 case 级完全不变（0 未来泄漏、闭包不变，防泄漏是结构性而非依赖时间门），Single-shot 10/14 泄漏（p=0.002）且闭包 8→4、Generic/HiREC 7/14（p=0.016）、metadata 分解 3/14（planning 层 annual fallback 取未来 10-K，预声明 P3 被证伪并如实记录）；no_time 下 v8 闭包显著超过 Single-shot（11 对 4，p=0.016）。A3 在本轨道无 L 机制可删（SEC pilot 的 finplan_no_limit 已有证据）；A5/A6 载体数据（中国问询/并购链、比较调查轨道）未构建，仍不可执行。
- 中国公开数据轨道：FinGLM 的 9 道跨年题、18 份巨潮官方年报和 3,567 页语料已完成冻结 pilot。测试 5 题中，原始 entity-only FinPlan 闭包 0/5，扩展为 entity-year 后为 5/5；但 metadata decomposition、Generic 和 HiREC-style 也均为 5/5，故只支持义务原子化的必要性，不支持 FinPlan 的独有策略优势。Single-shot 虽为 5/5，错误文档率达 0.4667。另有两条姓名标签仅因简繁字形或大小写被标为“不相同”，已保留原标签并另报规范化语义标签。FinGLM2 因数据库和答案未公开只作规划诊断；混合期间、问询—回复、并购进展和处罚整改将从巨潮/交易所同步补建，公开原题与自建机制集分别报告。
- 中国机制轨道：5 家公司的同年 H1/Q3/FY 期间碰撞协议已在方法运行前冻结，巨潮官方接口取得 10 份 H1/Q3 报告且失败为 0；页级构建与冻结运行已于 2026-08-11 完成（15 文档/1,894 页）。冻结 3 题中，period-aware 义务闭包 3/3，而 entity-only 与 entity-year 均 0/3（配对 exact McNemar p=0.25，仅方向性）；但 fixed/metadata decomposition、Generic 与 HiREC-style 也均为 3/3，且 Fixed decomposition 零错误文档，因此该 pilot 支持“财政期间必须进入义务键”的表示必要性，不支持 FinPlan 策略独立优势。答案层首跑：确定性抽取 reader（只读方法检索到的页面、主体接地）断言 2/3 全对、缺证据全拒答，entity-only/entity-year 覆盖为 0，直接暴露“文件闭包 ≠ 答案正确率”（全部方法在单一案例上因页级伪影拒答）；Qwen2.5-0.5B 自由判定在开发集三种 prompt 下矛盾、冻结集上过度断言，只作弱模型鲁棒性诊断，不作主指标。

## 推荐阅读

1. [研究总图](docs/00-研究总图.md)
2. [真实 Research Gap](docs/02-相关工作与Research%20Gap/02-Research%20Gap与能力矩阵.md)
3. [研究问题](docs/03-FinPlan-RAG方法/01-任务定义与研究问题.md)
4. [完整方法](docs/03-FinPlan-RAG方法/02-FinPlan-RAG完整框架.md)
5. [中文框架图](docs/03-FinPlan-RAG方法/03-中文框架图.md)
6. [数据与 Benchmark](docs/04-数据与实验/01-数据来源与Benchmark.md)
7. [基线与指标](docs/04-数据与实验/02-基线指标与公平协议.md)
8. [预实验报告](docs/04-数据与实验/04-预实验报告.md)
9. [真实数据 Pilot 与框架迭代](docs/04-数据与实验/05-真实数据Pilot与框架迭代.md)
10. [LOFin 多文档轨道实验](docs/04-数据与实验/06-LOFin多文档轨道实验.md)
11. [中国公开数据轨道](docs/04-数据与实验/07-中国公开数据轨道.md)
12. [Introduction](docs/05-论文写作/02-Introduction统一版.md)
13. [当前进度与后续实验](docs/06-当前进度与后续实验.md)
14. [外置数据与 GitHub 迁移说明](DATA_STORAGE.md)

## 引文核验状态

核心 BibTeX 已通过离线语法与重复项检查；12 条均为 `PARSED_ONLY`，正式投稿前仍需在获得作者授权的联系邮箱后执行 Crossref/DBLP/Semantic Scholar/arXiv 在线核验。离线工具原始结论：`PASS: 0/12 verified, 0 errors, 0 warnings (0 checks skipped)`。

## 旧版本

旧版方案、参考文件和生成缓存已迁移到 `D:\Engineering\FinPlanRAG\database\archive\`。代码仓库根目录只保留 FinPlanRAG 框架、核心代码、测试和冻结协议。
