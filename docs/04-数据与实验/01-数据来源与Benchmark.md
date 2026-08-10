---
type: data-and-benchmark-specification
status: real-data-gated
updated: 2026-08-09
---

# 数据来源与 Benchmark

## 1. 数据设计原则

论文不以私有终端或不可复核的券商数据作为主证据。核心实验同时使用美国 SEC 与中国交易所/巨潮公开披露；现成 benchmark 用于验证通用金融问答、多跳和时间能力，新建数据只补充现有数据没有提供的“关系路径—证据义务—限制证据—可得时间—动作轨迹”标签。美国与中国轨道分别报告，不用一国结果代替另一国验证。

必须分别保存：

- `valid_time`：事实、合同或权利何时生效；
- `available_at`：该材料何时进入公众认知集合；
- `cutoff`：问题允许使用信息的截止时点。

任何 `available_at > cutoff` 的材料均不能进入回答证据，即使其回溯描述了截止日前事件。

## 2. 金融域多轨道设计

FinPlan 不锁定收购。核心状态接口计划在四条金融轨道上复用：

1. SEC 收购/控制生命周期：签署—条件—完成/终止；
2. LOFin/HiREC 会计与 SEC 问答：定义—组件—期间—单位—计算；
3. FinSearchComp 的时间敏感/历史查找：观测日—发布 vintage—修订；
4. FinSearchComp 复杂历史调查与 LOFin textual：独立实体路径—共同定义—跨源冲突。
5. 中国公开披露：FinGLM 跨年年报问答，以及交易所/巨潮的混合期间、问询回复、并购进展和监管整改链。

只有在至少两条非同源轨道完成非 oracle、端到端冻结测试并显示状态接口的独立增益后，论文才可以使用“金融域通用”。当前 LOFin 真实 10-K 文件级 pilot 是第二轨道的机制诊断，但由于给定候选腿且与元数据分解持平，不单独通过该门。

## 3. 可直接获得的现成数据

| 数据 | 获得方式 | 用途 | 不能替代的部分 |
|---|---|---|---|
| LOFin / HiREC | [GitHub](https://github.com/deep-over/LOFin-bench-HiREC)；SEC 文档 | 金融长文档检索、answerability 与补充检索基线 | 没有本文的独立关系路径、限制证据和历史回放标签 |
| FinSearchComp | [Hugging Face](https://huggingface.co/ByteSeedXpert/FinSearchComp/) | 真实多源、时间敏感金融搜索的外部效度 | 主要是结果级评价，不能训练路径闭包状态 |
| MultiHop-RAG | [GitHub](https://github.com/yixuantt/MultiHop-RAG/) | 2–4 项证据的通用多跳检索迁移测试 | 非金融关系机制，缺少点时与限制证据 |
| TimeR4 / MultiTQ 等时间 QA | [TimeR4 GitHub](https://github.com/qianxinying/TimeR4) | 检验事实时间约束和强时间基线 | 时间知识图谱不等于文档的历史可得时间 |
| FinGLM | [GitHub](https://github.com/MetaGLM/FinGLM)；[ModelScope](https://modelscope.cn/datasets/modelscope/chatglm_llm_fintech_raw_dataset/summary) | 中国上市公司年报、人工问答、中文跨年义务规划 | 只有年度报告，缺少证据跨度、严格发布时间和混合期间/公告链 |
| FinGLM2 | [GitHub](https://github.com/MetaGLM/FinGLM2) | 101 组链式中文金融问题，用于问题规划与上下文状态诊断 | 比赛数据库和金标答案未随仓库公开，不能直接作为完整 RAG benchmark |

现成数据只承担辅助实验。核心主张必须在下面的真实点时关系回放集上成立。

国内公开数据的来源、许可、实际下载状态和冻结 pilot 见 [[07-中国公开数据轨道]]。公开原题与自行构建题必须分开；FinGLM 单文档答案分数不等于多文档闭包或点时正确性。

## 4. 核心数据：FinRel-PT

### 3.1 MVP 关系范围

首版聚焦“收购/控制链”而不是同时覆盖所有金融关系：

> 截至认知截止日 (t_c)，主体 A 是否通过交易或子公司直接、间接或有条件地控制资产/公司 B？

标签为 `direct / indirect / conditional / none`；`abstain` 是系统行为而非真值标签。只在收购/控制关系上完成可靠数据后，才扩展许可、供应合同、产品和受监管项目。这样可避免用少量关系样例支持过宽结论。

### 3.2 公开来源与字段

主来源为 [SEC EDGAR](https://www.sec.gov/edgar/search/) 的结构化提交元数据、原始 filing 与 exhibits：

- 8-K Item 1.01：签订重大协议；
- 8-K Item 2.01：完成资产收购或处置；
- 8-K Item 1.02：重大协议终止；
- 10-K / 10-Q：子公司、业务与后续状态；
- Exhibit 2.x / 10.x：合并、购买、许可或重大合同原文；
- `acceptanceDateTime`：主要 `available_at`；若页面或附件实际稍后可得，取较晚时间并记录抓取证据。

下载遵守 SEC fair-access 规则，固定 User-Agent、限速、原始文件哈希、accession number 与抓取清单。语料快照按 accession 固定，避免测试时在线搜索结果变化。

可选第二轨是 [Drugs@FDA](https://www.accessdata.fda.gov/scripts/cder/daf/) 与 [openFDA](https://open.fda.gov/) 的批准、撤回和公司主体材料；它只用于关系外推，不能在主实验未完成时分散数据建设。

### 3.3 样本与标注单位

一个样本由 `query_id, entity_a, target_b, cutoff, candidate_paths, obligations, evidence_spans, limiting_spans, provenance, label` 组成。每条候选路径至少标注：

1. 主体解析：法定主体、曾用名与子公司；
2. 交易/合同标识：accession、exhibit 与交易对象；
3. 关系机制：直接持有、子公司链、待交割协议或未成立；
4. 生命周期：签署、条件、完成、终止、处置或失效；
5. 限制证据：终止、未交割、剥离、不控制、对象不一致；
6. 时间：事件有效时间、公众可得时间与 cutoff；
7. 来源谱系：原始监管文件、公司转载、新闻转载及独立性。

正例与负例都必须由证据闭包构造。`none` 不能仅由“没有搜到”得到；优先采用终止、处置、主体不一致或时间未到等可证实负例。证据不足的样本保留为 answerability/abstention 测试，不强行赋予 `none`。

### 3.4 规模与拆分

最小可投稿门槛：

- 不少于 300 条唯一关系路径、100 个公司组；
- 若声称跨关系泛化，则至少 3 类关系且每类均有足够训练/验证/测试样本；否则标题与结论明确限定为收购/控制；
- 按公司集团、交易 lineage 和时间成组切分，任何同一交易的修订、完成或终止文件不得跨集合；
- 设置 `company-OOD`、`time-OOD`，扩展后再设置 `relation-OOD`；
- 测试集双人独立标注，分歧由第三人裁决；报告标签、路径、槽位与证据跨度的一致性。

### 3.5 防止 benchmark 泄漏

- 查询文本不暴露 filing 类型或最终状态；
- 候选路径对所有 schema-aware 方法一致可见；另设“需发现路径”的完整任务；
- 测试时间晚于训练时间，且冻结模板、检索索引和阈值；
- 同一新闻转载与同一 SEC 原件视为同一 provenance family；
- 记录模型版本及其可能预训练记忆，用仅编号/改写实体的受控子集检查记忆依赖。

## 5. 两阶段 benchmark

### 阶段 A：检索上限诊断

给定金标主体与候选路径，分别测原子证据 Recall@k、槽位召回和时间过滤。若检索器连必要原子证据都找不到，不能把最终失败归因于规划。

### 阶段 B：端到端点时规划

模型自行扩展路径、选择动作并停止。主比较在相同语料、候选初始化、模板、检索器和预算下进行；完整开放路径发现只作为额外难度。阶段 A 先确认“证据可取”，阶段 B 才检验“问题规划是否有效”。

## 6. 数据可行性预检

正式标注前抽取 30 个交易 lineage，检查：关键生命周期事件是否有明确 accession、`acceptanceDateTime` 是否稳定、终止/完成材料是否可配对、同一主体是否存在多条易混路径。若 30 例中不能稳定构造至少 20 例闭包轨迹，应先收窄关系定义，而不是扩大抓取规模。
