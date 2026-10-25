# GSU实验技术总报告

版本：2026-10-25研究快照。用途：集中理解项目全部实验脉络、算法、伪代码、结果和失败分析；不是已验证最优方法的论文定稿，也不是自动刷新仪表盘。

本报告按RQ与方法演变组织；表格注明来源和可比性。原始JSON/JSONL优先于手工报告，冻结run优先于当前共享代码，实际完成状态优先于计划。当前源文件新增的修复不会被追认成旧run已经使用。

## 目录

1. [结论速览与证据地图](#s01)
2. [术语、研究主线与数据权限](#s02)
3. [数据集、拆分和评测样本构成](#s03)
4. [环境、模型、参数化和比较轨道](#s04)
5. [指标公式、方向和计分差异](#s05)
6. [历史范围扩展与因果探测](#s06)
7. [GSU方法版本、公式和伪代码](#s07)
8. [baseline方法和复现配方审计](#s08)
9. [本地G组、早期M组与恢复实验结果](#s09)
10. [逐步诊断与A/B/C组结果](#s10)
11. [本地固定预算baseline与独立实现](#s11)
12. [云端小模型完整结果](#s12)
13. [8B三种子大表、逐seed小表与排序](#s13)
14. [配对差值、PPL和机制解释](#s14)
15. [故障、复现缺口和资源管理](#s15)
16. [exp017修复、随机对照及调参计划](#s16)
17. [RQ证据判定与论文可写范围](#s17)
18. [复现操作、检查表和项目职责](#s18)
19. [历史报告全部表格汇编](#s19)
20. [来源文件、哈希与快照说明](#s20)

<a id="s01"></a>
## 1. 结论速览与证据地图

### 1.1 截止时间与完成状态

云端状态观察：2026-10-25 12:38:28（北京时间）；磁盘/exp017复核至12:40:53。随后回收exp015/exp016汇总文件，文件SHA256见末尾；远程仍在运行，文件回收不是整台机器的事务性时间快照。

- exp016_8b_multiseed：42/162次正式训练已完成，44个完成能力评测的端点包括两个Original；Llama-8B的DPOmis seed42在评测阶段。
- C1/C2、GA、GradDiff、NPO、SimNPO、RMU：每种均已完成两个8B模型的seed42/43/44。后续方法仍按队列执行。
- KLUE原生8B：两个smoke反向传播失败；其6次正式训练被前置门跳过，不可当作6次正式性能失败或给0分。
- exp015小模型：26个正式端点完成，包括24个matched-LoRA与两个KLUE原生；其余原生轨因smoke错误跳过。
- exp017_decisive_validation：进程已登记，等待exp016完成；修复KLUE、匹配随机对照、扩展监督和开发集搜索尚无新成绩。
- 本地独立baselines/：训练完成不代表统一评测完成，任务可能在补评。其不同配方及评分不混入旧114步面板。

### 1.2 当前可以说什么

1. 多跳残留与直接问答残留是不同的测量对象；G组、A/B/C组和云端baseline给出了不同权衡的实际观察。
2. 当前C1/C2与GA在8B全89路径回忆上的均值差很小；平均PPL相对GA降低，但仍高于Original，Llama种子方差较大。
3. GradDiff/NPO在已测配置下更好地保持retain/PPL，同时多跳回忆较高。这是权衡而非“GSU全面超过baseline”。
4. 组件指导的因果必要性尚缺匹配随机对照；历史C3R不能代替C1R。
5. 早期局部锚点概率改善已被逐步日志观察到，但不自动推出最终独立保持改善；累计线性保护曾被实际非线性轨迹否定。

### 1.3 证据层级

| 层级 | 能证明什么 | 不能据此跳到什么结论 |
|---|---|---|
| 论文/代码审查 | 方法定义、公式来源、差异 | 已运行或达到作者成绩 |
| toy损失/梯度/恢复测试 | 被覆盖实现合同 | 所有真实模型端到端等价 |
| 实模smoke | 加载、前反向、部分更新/保存恢复可用 | 正式114步、三seed或最优性能 |
| 单seed正式run | 指定模型/协议/预算的点估计 | 显著性、全局最优、跨数据泛化 |
| 三seed完成 | 均值、波动和逐seed差值 | 自动证明显著性或关联定位因果 |
| 匹配随机/独立确认 | 对具体替代解释的检验 | 所有恢复途径和所有模型均消失 |

<a id="s02"></a>
## 2. 术语、研究主线与数据权限

### 2.1 主线

从关联知识中定位多跳回忆风险，将其转化为遗忘更新指导，使GSU在直接遗忘损失仅来自F0时，也能抑制关联多跳回忆。

这是一项待验证的性能与机制主张。它不等于“只访问F0”，也不等于“必须保住一切关联知识”。论文应区分遗忘监督、指导信息、保持目标和测试数据。

### 2.2 术语表

| 记号/名称 | 定义 | 常见误读 |
|---|---|---|
| F0 | 本次指定的原始直接遗忘事实QA | 所有多跳测试问题 |
| F1 | 选中的关联支持事实，当前历史固定5条 | F1_hop2题目本身 |
| F1_hop2 | 被指定支持事实覆盖的历史多跳切片75题 | 全89路径或75条独立路径 |
| Q_multi | 目标多跳查询集合 | 额外的直接训练标签 |
| retain_train | 实际参与保护/训练的保持样本 | 独立保持测试 |
| retain_heldout | 对该run未进入保护的保持样本 | 与目标语义绝对无关 |
| related/unrelated | 预先冻结图规则下的关联分层 | 已证明所有语义联系的真值 |
| answer LP | 答案token平均log概率 | 正确选项的历史全串分数 |
| actual step | 优化器提交后真实参数位移 | 当前梯度或学习率乘梯度 |
| C1 | 组件路由、实际锚点反馈和F0护栏 | 无梯度信号或原始GA |
| C2 | C1基础上增加组件内F1共享子空间过滤 | 直接增加F1遗忘损失 |
| matched LoRA | 显式限制参数域的统一比较轨 | 与原始全参数方法严格等价 |
| native | 明示方法原生更新域，可能使用CPU执行适配 | 必然与作者优化器/数值轨迹一致 |
| Original | 每模型不训练参照 | 一个训练算法或3次独立seed |
| Pareto frontier | 当前有限实测点的非支配集合 | 全局最优或统计显著优势 |

### 2.3 权限合同

定位集Q_loc用于探测，开发集Q_dev用于冻结候选/超参数，确认集Q_test只产出最终评价。若题目参与过F1筛选、候选观察或调参，它是探索数据，不能事后改名为未见确认。

F0-only：每步直接遗忘损失只来自F0。C1/C2仍访问F1直接事实标签与路径元数据，使用retain锚点及额外前反向。F0+F1：扩展监督明确在两组事实上产生遗忘梯度，必须单列。

原生轨使用全部retain66时，名字叫retain_heldout的历史JSON字段仍可能存在，但解释必须改为训练重合。保持分层需在看方法输出前按完整实体/三元组/答案字段冻结；不能把字符串按字符展开统计共享答案。

<a id="s03"></a>
## 3. 数据集、拆分和评测样本构成

### 3.1 FaithUn本地合同

| 项目 | 数量/规则 |
|---|---|
| 数据主体 | 200个实体 |
| 事实簇 | 664 |
| 官方forget | 33簇，F0一条直接QA/簇 |
| 官方retaining | 66簇；旧保护轨前33训练、后33审计 |
| 官方test | 464簇，不是465 |
| 官方三split已分配 | 563簇，互斥 |
| 未分配 | 101簇，不自动并入任一集合 |
| forget改写 | 99题，33×3 |
| 全hop2 | 89路径，267改写问题 |
| F1覆盖hop2 | 25路径，75改写问题 |
| 非覆盖hop2 | 64路径，192改写问题 |
| 历史same_object | 144题；不同历史受损子集可能为24，不能混用分母 |
| 历史D_hop2 | 24题，其中12题与F1设计内目标重合，12题不重合 |

数据原文：[FaithUn_Dataset.json](../data/faithun/FaithUn_Dataset.json)；split与许可在同目录。数据冻结SHA256为d570cc3e30c083357ca45ab4dd9df807abbd800f51a657d89629e4fe7b3adbb0。字段包括hop1_unlearn、hop1_test、hop2_test、hop3_test、same_object；本项目主比较不因字段存在就声称三跳已完成。

### 3.2 输入形状与接口

完整FaithUn数据以data字段保存实体到簇列表的映射；必须保持原始插入顺序展开，官方索引是展开后的簇索引。

~~~json
{
  "data": {
    "entity_id": [{
      "hop1_unlearn": {
        "questions": ["直接问题"],
        "answers": "目标答案",
        "false_answer_options": ["错误项1", "错误项2"],
        "graphs": ["主体", "关系", "客体"]
      },
      "hop1_test": {"questions": ["改写1", "改写2", "改写3"]},
      "hop2_test": {
        "questions": [["路径1改写1", "路径1改写2", "路径1改写3"]],
        "answers": ["路径1答案"],
        "false_answer_options": [["错误项1", "错误项2"]],
        "graphs": [[["主体", "关系", "桥接实体"], ["桥接实体", "关系", "答案"]]]
      }
    }]
  }
}
~~~

上例只展示字段层级，不是合法的完整训练数据；不能拿缺少答案的改写示例直接喂入。内部标准item含id/source/question/answer/false_options；F1另带fact_key/path_count。模型加载、F0/F1构造、旧集合归约由各冻结版本的load_inputs决定。

### 3.3 外部数据的角色

| 资源 | 角色 | 导入前必做 |
|---|---|---|
| MQuAKE-CF-3k-v2 | 外部原始多跳链适配 | 使用target_true、orig链和原答案，不用target_new/new_answer冒充遗忘；按主体/共享边分组隔离 |
| RWKU | 实体级遗忘与邻居/攻击探针 | target、合法训练语料、level测试权限分开；不从测试探针取F0后称原协议 |
| Mirage/llm2graph | 相关图式压力测试和生成工具 | 冻结生成器、提示、质检、攻击预算，工具不是训练baseline |
| MMLU | 通用学科知识评测 | 固定shot、subject/sample计划，不能训练测试题 |
| WikiText2 | 一般文章PPL语料 | 固定revision、text哈希、拼接、tokenizer和窗口 |

MQuAKE原任务是知识编辑；改造后应称derived unlearning evaluation。RWKU各配置字段和许可证需逐一核实，不假定全任务接口一致。新增数据实验另建卡，不混入当前已冻结FaithUn结果。


<a id="s04"></a>
## 4. 环境、模型、参数化和比较轨道

### 4.1 模型与执行环境

| 轨道 | 模型 | 执行位置与用途 |
|---|---|---|
| 历史主探索 | Qwen2.5-1.5B-Instruct | 本地，范围扩展、损伤/恢复、A/B/C与历史baseline |
| 历史规模扩展 | Qwen2.5-3B-Instruct | 本地独立clean G1–G5，不与1.5B拼前沿 |
| 云端小模型 | Llama-3.2-3B-Instruct、Qwen3.5-4B | exp008–015，架构不同，不单独识别参数规模效应 |
| 云端8B | Meta-Llama-3.1-8B-Instruct、Qwen3-8B | ModelScope BF16权重，exp016三seed及exp017补充 |
| 新独立baseline | 本地Qwen2.5-1.5B及其独立配方 | baselines/下的训练/补评，不与历史114步不经审核合表 |

本地指定解释器D:/Anaconda/envs/cuda126_env/python.exe；历史环境记录为Python3.10、PyTorch2.6.0+cu126、RTX4060Ti16GB。云端历史快照记录Python3.12.3、PyTorch2.8.0+cu128、Transformers5.20.2、PEFT0.21.2、单RTX4090约24GB；具体版本以run/environment为准，不把当前安装版本倒填历史。

当前云端8B权重路径：Llama在/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct，Qwen在/root/models8b/Qwen3-8B。两者根级权重/配置/tokenizer有下载manifest及哈希；不下载重复original格式，也不用量化权重冒充BF16 8B。

### 4.2 LoRA与组件粒度

W'=W+(alpha/r)BA，底座W冻结，A/B可训练。rank8、alpha16、dropout0是多数固定预算探索的设置，不代表各论文默认。初始化B=0时A的梯度可以为0，不能据此永久封死A。

一个组件通常是一块文本线性矩阵对应的LoRA-A/B张量对。Qwen2.5-1.5B为28层×7矩阵=196组件；32层×7=224只适用于相应结构。Qwen3.5包含linear_attn，不能强行用self_attn正则漏掉其模块。视觉、MTP和lm_head是否排除由具体目标枚举记录。

旧云端小模型通用LoRA：Llama196组件约1216万参数，Qwen3.5为248组件约1623万；RMU早期只覆盖三层down_proj，参数量不同。8B组件数/参数量以其modules.json为准，不能复制小模型数字。

### 4.3 三条不同的比较合同

| 合同 | 保持什么 | 不保证什么 |
|---|---|---|
| 固定114步matched-LoRA | 同模型、F0、LoRA、训练更新预算；8B进一步固定同seed F0顺序 | 等计算、等信息、同UA终点或作者最佳调参 |
| 原生重实现 | 明示原生权重/神经元域、论文或上游配方 | 与作者隐藏细节、提示、数值、优化器逐位一致 |
| baselines独立来源对齐 | 来源锁、作者社区说明、不同早停/批次/评分约定 | 自动与exp015/016旧轨可互换 |

8B原生全参数用CPU权重/梯度流式执行与FP32动量，BF16权重储存；没有额外完整FP32 master。KLUE采用原生选中神经元更新的单独合同。它们是执行/数值适配，不是LoRA替代，也不能隐瞒与原配方差异。

<a id="s05"></a>
## 5. 指标公式、方向和计分差异

### 5.1 概率、遗忘与选择题

答案平均log概率：

~~~text
ell(q,a) = (1/Ta) * sum_t log p(a_t | q, a_<t)
L_GA = mean_F0 ell(q,a)      # 优化器最小化，故目标答案概率下降
NLL_mean = -ell
NLL_sum = -sum_t log p(a_t | q, a_<t)
~~~

NPO使用序列总NLL的参考差，SimNPO使用长度归一化NLL，不能互换后仍说同公式。完整答案的序列概率是exp(-NLL_sum)，与exp(ell)这个token几何均值不同。

历史MCQA全串分数对q+候选答案的全部next-token取均值，含公共题干及长度分母。答案条件序列概率只评分答案区间；再除答案长度又是第三种分数。正确率ACC与margin取决于候选排序，不能由某个分数变低直接推出答案LP下降。

~~~text
correct_i = 1[ score(correct) > max score(wrong options) ]   # 历史strict-tie规则
margin_i = score(correct) - max score(wrong)
R_all89 = (1/89) * sum_path mean_paraphrases(correct_i)
~~~

每路径恰3改写时R_all89与267题微均值相等；统计单元仍相关，不能据此宣称267个独立实验。三选一1/3是随机猜测期望，不是确定模型下界或知识消除证明。

### 5.2 指标读表

| 项目 | 方向 | 含义/限制 |
|---|---|---|
| F0原问/改写ACC | ↓ | 指定知识直接残留；原问不替代改写 |
| 全89路径宏回忆 | ↓ | 主多跳结果；覆盖75题不能替代 |
| F1_hop2 ACC | ↓ | 被历史F1覆盖的路径切片 |
| F1直接事实 | 审计 | 反映支持事实被改变，不强制当作保持或遗忘成功 |
| 独立retain/test ACC | ↑ | 解释需看实际训练是否重合 |
| D/K、same_object | 随任务 | 旧损伤/恢复子集与当前遗忘目标可重合，不能用错方向 |
| 遗忘答案LP/margin | ↓ | 概率抑制/选项相对排序分开 |
| 保持答案LP/margin | ↑ | 与漂移绝对值接近0是不同目标 |
| MMLU-10/科目 | ↑ | 570题多学科子集，不是完整MMLU或全部推理 |
| MMLU delta pp | ↑ | 当前ACC减Original，再乘100 |
| WikiText2 token PPL | ↓ | 固定一般语料下token预测质量，非错误率 |
| PPL比值/meanNLL差 | ↓ | 同模型与Original比较，不能跨tokenizer按绝对PPL排名 |
| 时间、显存、参数量 | ↓为成本 | 成本低不单独代表遗忘好 |

### 5.3 MMLU实际配置

57个科目，各自固定seed42无放回取10测试题，5-shot来自允许的开发示例；无chat template、batch1。训练seed42/43/44仍使用同一测试样本计划；保存subject和逐题结果。小样本全科smoke通常每科1题，不参加正式排名。

MMLU大白话是跨学科选择题考试；数据集/基准是MMLU，报告的是其ACC。MMLU稳定不代表一般文本生成没有退化，必须同时看PPL和独立保持。

### 5.4 PPL实际配置与伪代码

~~~text
PPL = exp(total_NLL / scored_target_tokens)
PPL_ratio = PPL_method / PPL_original
delta_NLL = meanNLL_method - meanNLL_original
~~~

WikiText2 raw-v1 test，固定dataset revision b08601e04326c79dfdd32d625aee71d232d685c3及文本哈希；text字段双换行拼接，tokenizer有BOS时加一次；context≤2048，stride512；每个有效next-token标签只计一次。

~~~python
ids = tokenize(join_official_text_with_double_newlines())
ids = prepend_bos_once_if_available(ids)
total_nll, targets = 0.0, 0
for window in frozen_sliding_windows(ids, context=2048, stride=512):
    labels = mask_already_scored_prefix(window)
    logits = model(window.input_ids, use_cache=False).logits.float()
    total_nll += shifted_cross_entropy_sum(logits, labels)
    targets += number_of_valid_shifted_labels(labels)
assert targets == len(ids) - 1
return exp(total_nll / targets)
~~~

不平均每句PPL；不把harness的word_perplexity/byte_perplexity写成这里的token PPL。异常大PPL先查底座/adapter、mask、窗口计数与dtype，再解释退化；不能只因MMLU没掉就忽略。

### 5.5 FaithUn综合指标的额外口径

论文综合分可能结合遗忘侧与保持侧，例如对其明确指定的forget/test多跳准确率采用(1−forget_multi+test_multi)/2。必须同时说明提示、选项、聚合和题目范围。旧执行器的AA/MA字段不能凭名字冒充独立baselines/已核对的论文UA/TA/MA/SA/Score。

本报告主表优先展示直接残留与多跳残留，不用综合分掩盖一侧变化。旧恢复研究里的rec/rb等是相对LP标尺百分数，不能与MCQA百分比并列当同义指标。

<a id="s06"></a>
## 6. 历史范围扩展与因果探测

### 6.1 探测驱动支持事实选择

由F0路径的非目标边构造候选支持事实，按模型掌握过滤，再在预算B=5内覆盖活跃路径。历史池为89次支持边出现、61条去重事实；1.5B掌握41条、活跃56路径；3B掌握45条、活跃50路径。候选掌握用两个直接模板和同关系干扰项，阈值tau=.75。

~~~python
candidates = deduplicate(non_target_edges(paths_containing(F0)))
known = [f for f in candidates if probe_consistency(f) >= 0.75]
active = [p for p in paths if base_model_passes_frozen_path_probe(p)]
selected, covered = [], set()
for _ in range(5):
    fact = deterministic_argmax(known - selected,
        key=lambda f: weight(paths_supported_by(f) - covered))
    selected.append(fact)
    covered |= paths_supported_by(fact)
return selected
~~~

这是固定集合上的预算覆盖启发式；标准单调次模最大覆盖条件满足时，贪心有1−1/e近似背景，但不据此推导模型真实推理机制或全路径安全。公共USA/LosAngeles事实的高覆盖暴露了共享知识代价和R禁区设计需求。

### 6.2 五种范围对照

| 组 | 直接遗忘监督 | 回答的问题 |
|---|---|---|
| G1 | F0 | 不扩展能否阻止多跳 |
| G2 | F0+贪心5条支持事实 | 挑选有信息量的事实是否有效 |
| G3 | F0+全部61候选 | 数据规模上限参考与代价 |
| G4 | F0+随机5条支持事实 | 同规模随机对照 |
| G5 | 仅5条支持事实 | 支持路径本身是否影响目标查询 |

旧训练使用GA-LoRA及KL保持，不能与后来的无KL纯GA-F0简单当相同算法。G5的影响只给局部路径依赖线索，不证明所有自然推理严格沿预设链进行。


<a id="s07"></a>
## 7. GSU方法版本、公式和伪代码

### 7.1 为什么经历多个版本

| 阶段 | 改什么 | 必须区分的结论 |
|---|---|---|
| M1 | 局部高层MLP更新+手术 | 损伤定位位置不必等于有效遗忘更新位置 |
| M2/M3 | 全层LoRA、部分/完整梯度保护 | 梯度保护不保证AdamW实际位移也保护 |
| M3C/M3U | 改梯度或改真实位移 | 一阶局部保护与逐锚点非线性保护不同 |
| M3A | 独立锚点半空间交集 | 局部预测约束不保证累计保持 |
| COV16/CUM16 | 扩大覆盖/累计线性保护 | 覆盖不足与线性化失真是不同假设 |
| FB16 | 对实际答案LP反馈 | 测到当步改进不自动迁移最终retain |
| A组 | 训练前组件一致性/路径分数，Adam前缩放 | 缩放可能被优化器归一化抵消，需实测 |
| B组 | A信号叠加旧损伤先验 | B2不是完全没用A，旧先验也不是跨模型因果真值 |
| C组 | Adam后组件/子空间路由+F0护栏 | 更新指导、一般缩放、反馈和护栏各自需对照 |

### 7.2 M3及事后位移修正

设遗忘梯度g_f、保持梯度g_r，以当前符号约定检查是否对保持不利；原M3在梯度空间减去冲突投影，再交给AdamW。由于自适应尺度与动量，最终d不必平行于被投影的梯度。

~~~python
# M3概念流程；具体符号和聚合以冻结实现为准
forget_grad = gradient(forget_objective)
retain_grad = gradient(retain_log_probability)
safe_grad = remove_conflicting_projection(forget_grad, retain_grad)
optimizer.step_with(safe_grad)

# M3U概念流程：对象换为优化器真实位移
before = clone_trainable_parameters()
optimizer.step()
d_raw = parameters() - before
if dot(retain_grad, d_raw) < 0:
    d_safe = d_raw - dot(retain_grad, d_raw) / norm2(retain_grad) * retain_grad
    write_parameters(before + d_safe)
~~~

这是线性近似：dot(g_r,d)非负不保证每条锚点LP都升，也不保证有限步真实模型响应。聚合梯度可能让某些锚点获益、另一些受损。

M3A把约束按锚点分开，求多个半空间交集；COV16扩大每步采样覆盖，CUM16增加相对初始化的线性累计下限。原始求解器残差、写回误差和实际LP轨迹都需审计。

### 7.3 实际LP反馈FB16

~~~python
raw_anchor_lp = answer_logprobs(model_after_raw_update, anchors)
proposal = current_displacement()
for feedback_round in range(2):
    risks = [a for a in anchors if current_lp(a) < reference_lp(a) - tolerance]
    if not risks:
        break
    gradients = [grad_answer_lp(a) for a in risks]
    correction = solve_bounded_anchor_correction(gradients, lp_deficits(risks))
    proposal = enforce_shared_trust_radius(proposal + correction,
        center=raw_displacement, radius=0.30 * norm(raw_displacement))
    write_parameters(before + proposal)
    measure_actual_lp_again()
~~~

FB16是反馈机制概括，不把线性可行性称成全局最优；保护锚点、最终独立retain与多跳抑制必须分别评价。

### 7.4 A组组件评分的精确定义

组件j合并其LoRA-A/B张量的梯度。g_ij来自第i个F0；h_mj来自第m个F1；c_m为该支持事实历史活跃路径数。

~~~text
gbar_j = (1/n) sum_i g_ij
A_j = ||sum_i g_ij|| / sum_i ||g_ij||,  0 <= A_j <= 1
P_j = sum_m c_m*cos(gbar_j,h_mj) / sum_m c_m
S_j = A_j*max(P_j,0)
~~~

A体现F0梯度内部一致性，P体现与支持记忆的路径加权对齐，S不是组件真实因果重要性的已识别量。零范数、被跳过梯度和F1 path_count都必须记录。

~~~python
for fact in F0:
    snapshots_f0.append(cpu_gradient_snapshot(fact))
for memory in F1:
    snapshots_f1.append(cpu_gradient_snapshot(memory))
assert parameter_hash_before == parameter_hash_after
for component in text_lora_components:
    A = norm(sum_f0_gradients(component)) / sum_gradient_norms(component)
    P = weighted_mean_cosine(mean_f0_gradient(component), f1_gradients(component), path_counts)
    S[component] = A * max(P, 0)
freeze(S)
~~~

A2软引导用S/maxS；A3硬引导用S≥median(S)，偶数中位数取中两值均值、ties全保留；再恢复整体梯度范数。随机组置换分数保留直方图。A4在扩展监督下用引导；不能只看最优切片决定有效性。

### 7.5 B组先验

~~~text
damage_prior(layer) = sqrt(normalized_D_G1(layer) * normalized_D_G2(layer))
S_B(component) = S_A(component) * damage_prior(layer)   # MLP
attention prior = 1                                  # 无相同粒度先验时
~~~

B0/B1是无指导参照；B2/B3及其随机对应组按各冻结卡解释。历史B2保留了A型F0/F1/路径信号，只是叠加先验；新旧信号不是二选一。旧Qwen层号不能直接迁移到Llama或混合注意力。

### 7.6 C1/C2的数学更新

~~~text
w_j = 0.5 + 0.5*S_j/max_k(S_k)    # 全零分数时全1
P_j = U_j U_j^T                  # 不构造稠密投影矩阵
v_j(C1) = w_j*d0_j
v_j(C2) = w_j*(d0_j + P_j*d0_j)/2
route = d0 + min(1, 0.30*||d0||/||v-d0||)*(v-d0)
~~~

U_j由5条F1梯度列的SVD/正交基给出，rank≤5，相对阈值1e-6，更新1/39/77刷新。零列空间identity fallback，不把初始化A零梯度当永久不可训练。非零列缩放不改变完整列空间，路径权重不是额外独立方向旋转。

实际位移预算共享：最终||d_final−d0||≤0.30||d0||。反馈修正也占用这个半径。Adam状态只按原F0梯度提议推进一次，不把改过的位移伪装成新梯度回写二阶矩。

### 7.7 C组完整伪代码

~~~python
set_seed(train_seed)
model = frozen_base_plus_fresh_lora(task_type="CAUSAL_LM")
model.train()
S = score_components(F0, F1, frozen_path_metadata)
weights = soft_floor_normalization(S, floor=0.5)
forget_plan = deterministic_f0_schedule(F0, steps=114, seed=train_seed)
anchor_plan = deterministic_16_of_33_schedule(seed=train_seed)

for update, item in enumerate(forget_plan, start=1):
    anchors = anchor_plan[update]
    if variant == "C2" and update in (1, 39, 77):
        U = per_lora_tensor_basis(gradients_of_f1_facts(F1), cutoff=1e-6)
    elif variant == "C1":
        U = identity_fallback_for_every_tensor()

    before = clone_trainable_parameters()
    lp_before = target_answer_lp(item)
    anchor_before = actual_anchor_lps(anchors)
    backward(mean_answer_logprob(item))
    optimizer.step()                         # 恰好一次
    raw = parameters() - before
    lp_raw = target_answer_lp(item)
    route = bounded_component_and_span_route(raw, weights, U, radius=0.30)
    write(before + route)
    feedback_candidate = bounded_actual_lp_feedback(anchors, anchor_before, raw)

    raw_progress = lp_before - lp_raw
    if raw_progress <= 0:
        final = raw
        label_no_positive_progress_guarantee()
    else:
        for alpha in (1.0, 0.5, 0.25, 0.0):
            trial = raw + alpha*(feedback_candidate - raw)
            write(before + trial)
            if target_answer_lp(item) <= lp_before - 0.9*raw_progress + 1e-6:
                final = trial
                break
    remeasure_final_anchor_lps_and_f0()
    assert optimizer_step_count() == update
    assert norm(final - raw) <= 0.30*norm(raw) + numerical_tolerance
    save_step_diagnostics()
    if update in (38, 76, 114):
        evaluate_frozen_groups_without_selecting_best_checkpoint()

save_final_checkpoint()
run_all89_mmlu_wikitext_audit()
~~~

### 7.8 本地与云端对齐

旧云端C1从C2入口关闭span，仍与本地train/eval模式、CAUSAL_LM包装、smoke步数和中途评测不同；不能把旧结果称为严格重放。后来云端直接调用本地worker，仅保留模型加载/组件枚举的架构适配。8B再增加真实seed参数和非重入checkpointing；核验同seed F0顺序，不更换评测题。

train/eval差异存在不等于因果已证；dropout0也不能替代完整一整步数值核验。模型、tokenizer、CUDA/包版本不同，所以对齐实现后的跨模型实验仍不是同模型逐位重放。

<a id="s08"></a>
## 8. Baseline方法和复现配方审计

### 8.1 方法目标与伪代码对应

| 方法 | 遗忘核心 | 参考/保持 | 需要避免的移植错误 |
|---|---|---|---|
| GA | -CE_f | 无强制保持 | 符号写反、用LP下降解释成概率上升 |
| GradDiff | gamma*(-CE_f)+alpha*CE_r | retain监督 | NLL/KL、权重和预算混为同设置 |
| NPO | -(2/beta)*logsigmoid(beta*(NLL_sum−NLL_ref)) | 冻结参考+retain项 | 错用平均NLL、参考随训练变化 |
| SimNPO | -(2/beta)*logsigmoid(beta*(NLL_mean−delta)) | 无参考+retain项 | beta/gamma导致梯度弱不等于公式错误 |
| RMU | 忘记表示推随机控制向量 | retain表示匹配底模 | hook层、mask、归一化和参数域不同 |
| DPOmis | 失配答案作为preferred，原答案dispreferred | 实际retain目标按来源配置 | 把问题改写当失配、retain NLL冒充反向偏好 |
| DPOrej | 拒答preferred、原答案dispreferred | 同上 | 拒答原文、模板和beta不披露 |
| CEU | 正确token抑制的detach分布蒸馏 | 来源实现不额外加retain | 首token排除后单token空mask |
| PDU | logit spread与保持约束的对偶优化 | epsilon/dual | no-dual配置冒充完整PDU |
| WGA | detach(p^beta)加权负CE | retain项 | 低置信梯度很小、权重未detach |
| SatImp | 饱和/重要性加权负CE | retain项 | gamma/alpha类默认与作者说明混用 |
| UNDIAL | 冻结teacher降低正确token logit后蒸馏 | 配方可设retain权重0 | -100标签错误索引、提示token目标错误 |
| KLUE | 正则化知识归因、筛样、原生神经元掩码 | 训练retain | LoRA代理、概率换logprob、掩码只屏grad却让动量移动 |

### 8.2 旧默认、上游YAML与作者配方

- 旧统一GAMMA/ALPHA/BETA=1、DELTA=0是构造默认移植，不等于论文配方。后续上游YAML NPO beta=.1，SimNPO beta4.5/gamma.125，SatImp gamma.1/alpha1，UNDIAL beta10/alpha0；独立baselines来源进一步核对了不同作者文件。
- baselines/记录SimNPO作者来源默认beta2.5/gamma.1375、seed1001和warmup等；与exp016统一训练seed42/43/44不是同一合同，必须各自报告配置而非拼接。
- RMU早期restricted LoRA c20/alpha1与上游YAML c2/EMBED_DIFF/full-text不同，FaithUn说明还有c20及retain权重20/50/100/150/200/300搜索。
- FaithUn相关方法batch4、最高150epochs、UA≤.33停止、选择可达阈值的较低学习率，与batch1/114步固定终点不同。统一LoRA不是全部公平性的充分条件。

### 8.3 KLUE完整流程伪代码

~~~python
for epoch in range(max_epochs):
    remaining = select_still_remembered_training_facts(F0, frozen_criterion)
    if training_UA(remaining) <= stop_threshold:
        break
    for batch in batches(remaining):
        for fact in batch:
            attribution = activation_times_gradient(full_answer_probability(fact))
            mismatches = sample_5_training_questions_with_same_target_answer()
            regularizer = mean(clip_positive(attribution_of(m)) for m in mismatches)
            importance[fact] = token_max_pool(attribution) - 10*regularizer
        scores = fixed_batch_aggregation(importance)
        selected = stable_global_top5_percent(scores)
        loss = -0.7*forget_CE(batch) + retain_CE(training_retain)
        backward(loss)
        mask_gated_mlp_rows_and_columns(selected)
        native_masked_optimizer_step_without_unselected_momentum_motion()
~~~

gated MLP约定：pre-down激活是中间神经元，gate/up对应行、down对应列；全局5%而非每层任意5%。这些是重实现的公开约定，不冒充未公开作者代码。CPU原生优化器与BF16精度有边界；double概率标量不代表整个反向链FP64。

### 8.4 两项有反例的上游修复

RMU批次MSE按每序列有效token数归一化再平均；避免batch>1时错误广播成B×B，batch1不变。UNDIAL仅对有效答案label构造惩罚目标并计算CE，排除-100负索引与提示位置。两者有单独反例测试；修复版不声称损失逐位等同有问题的上游原样。

### 8.5 端到端验收顺序

~~~python
freeze_source_data_model_config()
check_loss_formula_and_independent_gradients()
check_actual_tokenizer_masks_and_variable_answer_lengths()
check_trainable_modules_reference_cache_and_random_streams()
run_actual_model_forward_backward_one_step()
check_unselected_weights_and_momentum_do_not_move()
save_reload_and_compare_logits()
run_smoke_on_actual_architecture()
if smoke_passed:
    run_predeclared_formal_training_and_all_metrics()
else:
    retain_failure_and_skip_formal_scores()
~~~

这些门检并不等于已调优或达到作者成绩。完整公平比较还需开发集搜索、同F0强度/等计算对照、独立确认和合理统计。


<a id="s09"></a>
## 9. 本地G组、早期M组与恢复实验结果

### 9.1 预实验原始摘要

| 模型 | 簇数 | hop1一致率 | hop2整体 | hop1正确条件下hop2 | hop2条目 | 随机期望 |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 133 | 66.92% | 67.52% | 70.31% | 353 | 33.33% |
| Qwen2.5-3B | 133 | 73.87% | 61.00% | 62.52% | 353 | 33.33% |

来源：[exp/results/pre_experiment_summary.json](../exp/results/pre_experiment_summary.json)、[exp/results/pre_experiment_summary_3B.json](../exp/results/pre_experiment_summary_3B.json)。这不是正式33簇全89路径主表，不把不同口径的数值拼成规模因果结论。

### 9.2 G组正式结果大表

| 模型 | 组 | F0↓ | 改写↓ | hop2↓ | same_object审计 | retain↑ |
|---|---|---|---|---|---|---|
| 1.5B | orig | 63.64% | 64.65% | 64.42% | 75.69% | 75.76% |
| 1.5B | G1 | 18.18% | 20.20% | 36.33% | 31.94% | 39.39% |
| 1.5B | G2 | 18.18% | 19.19% | 25.47% | 25.69% | 39.39% |
| 1.5B | G3 | 36.36% | 29.29% | 22.47% | 29.86% | 60.61% |
| 1.5B | G4 | 18.18% | 20.20% | 31.09% | 31.25% | 36.36% |
| 1.5B | G5 | 60.61% | 63.64% | 55.81% | 75.00% | 69.70% |
| 3B clean | orig | 69.70% | 70.71% | 63.30% | 85.42% | 81.82% |
| 3B clean | G1 | 12.12% | 14.14% | 25.47% | 22.92% | 36.36% |
| 3B clean | G2 | 15.15% | 13.13% | 23.60% | 18.75% | 33.33% |
| 3B clean | G3 | 21.21% | 24.24% | 15.73% | 22.22% | 54.55% |
| 3B clean | G4 | 18.18% | 18.18% | 23.60% | 21.53% | 36.36% |
| 3B clean | G5 | 66.67% | 61.62% | 46.07% | 63.19% | 63.64% |

来源：[exp/results/eval_results_1p5B_v2.json](../exp/results/eval_results_1p5B_v2.json)、[exp/results/eval_results_3B_v2_clean.json](../exp/results/eval_results_3B_v2_clean.json)。均为历史单seed，LoRA+KL保持，非当前纯GA损失/114步轨。

1.5B的G2−G1 hop2差为-10.86pp，G2−G4为-5.62pp；3B的G2−G1为-1.87pp，G2与G4点估计相同。前者是方向性筛选信号，后者不支持“3B上探测贪心优于随机”。

G3扩大监督后结果并非所有指标单调变化，说明固定训练轮数、目标规模和保持项存在交互；不能只按额外事实数归因。旧3B PEFT重复适配器警告run留档，最终采用独立底座clean版本。


### 9.3 早期GSU和移植baseline五集表

| 历史group | F0↓ | 改写↓ | hop2↓ | same_object | retain↑ |
|---|---|---|---|---|---|
| GSUM1 | 42.42% | 46.46% | 59.55% | 66.67% | 72.73% |
| GSUM3 | 15.15% | 24.24% | 26.97% | 20.14% | 48.48% |
| GSUM2 | 33.33% | 39.39% | 31.09% | 29.17% | 42.42% |
| BLGradDiff | 36.36% | 42.42% | 64.04% | 63.19% | 93.94% |
| BLNPO | 42.42% | 48.48% | 59.18% | 65.28% | 84.85% |
| BLSimNPO | 81.82% | 81.82% | 76.78% | 81.94% | 93.94% |
| BLRMU | 81.82% | 78.79% | 76.03% | 81.94% | 93.94% |
| BLGA | 39.39% | 37.37% | 29.96% | 23.61% | 33.33% |
| BLCEU | 24.24% | 25.25% | 26.22% | 38.19% | 39.39% |
| BLPDU | 24.24% | 25.25% | 22.47% | 27.08% | 42.42% |
| BLWGA | 75.76% | 76.77% | 77.53% | 83.33% | 93.94% |
| BLSatImp | 84.85% | 80.81% | 76.40% | 82.64% | 93.94% |
| BLUNDIAL | 81.82% | 77.78% | 74.16% | 84.72% | 93.94% |
| BLDPO | 69.70% | 78.79% | 71.91% | 80.56% | 90.91% |

来源：[exp/results/eval_results_gsu.jsonl](../exp/results/eval_results_gsu.jsonl)。这些run主要是F0+F1监督、早期LoRA/3epochs及当时移植公式；不与后来的F0-only同名方法视为独立seed或同配方。这张表完整保留负结果，不把未遗忘方法的高retain解释成综合优势。

### 9.4 损伤与定位观察


| 题组 | n | G1平均损伤LP差 | G2平均损伤LP差 | G2负损伤比例 |
|---|---|---|---|---|
| hop1_unlearn | 33 | 5.3059 | 6.3222 | 0.00% |
| hop1_test | 99 | 4.5382 | 5.5660 | 0.00% |
| hop2 | 267 | 2.6104 | 3.7277 | 0.00% |
| same_object | 144 | 3.7509 | 4.8748 | 0.00% |
| retain | 33 | 3.3807 | 4.2753 | 0.00% |

来源：[exp/results/damage_summary.json](../exp/results/damage_summary.json)。损伤定义和来源按原damage脚本；正差不能直接当作因果组件贡献。

历史报告记录：G2 FFN漂移与单层撤销收益Spearman rho=.824，G1→G2留出rho=.830；定向6层MLP撤销恢复34.4%，随机6层13.8%；top12神经元3.8%与随机3.6%接近。它支持层/模块干预信号，不支持“神经元清单足以恢复”，也不外推跨架构。具体原表及来源见第19节，相关训练/归因数据在exp/results/attribution。

### 9.5 恢复—回弹相对LP标尺


| 恢复配置 | 恢复rec↑ | F0回弹rb↓ | F1复活↓ | K漂移↓ | retain恢复 |
|---|---|---|---|---|---|
| R1_rec | 81.97% | 75.54% | 74.03% | 80.20% | 88.88% |
| B_kl | 66.39% | 56.55% | 64.42% | 79.37% | 65.82% |
| R3_l0.1_e80 | 79.83% | 73.14% | 72.28% | 78.00% | 86.50% |
| R3_l0.1_e90 | 79.84% | 73.23% | 72.18% | 77.58% | 86.75% |
| R3_l0.1_e95 | 79.88% | 73.36% | 72.81% | 77.69% | 86.78% |
| R3_l1_e80 | 76.97% | 65.78% | 70.10% | 75.00% | 82.71% |
| R3_l1_e90 | 74.45% | 64.42% | 67.77% | 73.89% | 80.81% |
| R3_l1_e95 | 73.73% | 64.93% | 66.95% | 73.50% | 80.30% |
| R3_l10_e80 | 69.68% | 46.21% | 66.21% | 71.69% | 66.54% |
| R3_l10_e90 | 62.04% | 44.52% | 57.17% | 71.04% | 64.19% |
| R3_l10_e95 | 64.70% | 49.97% | 62.83% | 69.33% | 66.67% |
| R5_full | 11.84% | -6.67% | 15.29% | 4.06% | 6.07% |
| R5_surgery | 31.19% | 16.35% | 21.33% | 4.75% | 26.80% |
| R2_ans | -33.49% | -55.61% | -41.15% | -135.97% | -66.40% |
| R4_combo | -16.13% | -29.37% | -18.30% | -72.10% | -39.67% |

来源：[exp/results/restore_summary.json](../exp/results/restore_summary.json)。以上数值已经是百分比，不再乘100；相对标尺分母为原模型与G2的LP可恢复空间，负数代表超出该标尺方向，不是负准确率。

R5_surgery恢复31.19%、回弹16.35%、F1复活21.33%、K漂移4.75%；满足某个K门不等于同时满足回弹<10%和F1≤5%的历史验收。该恢复任务与当前遗忘训练方向不同，不能拿其低K替代当前GSU的通用能力证明。


<a id="s10"></a>
## 10. 逐步诊断与A/B/C组结果

### 10.1 从局部保护到真实累计结果

| 变体 | D_nonoverlap/12（旧保持主项↑） | 留出/旧retain /33↑ | F1覆盖/75↓ | 局部日志观察 | 推进判定 |
|---|---|---|---|---|---|
| M3复评 | 0 | 16 | 25 | 聚合梯度与实际位移不等同 | 历史参照 |
| M3C | 0 | 17 | 24 | 梯度侧修正 | 未达原门 |
| M3U | 1 | 13 | 9 | 聚合事后修正437/456同批锚点LP变好 | 未达主指标/retain门 |
| M3A | 3 | 11 | 9 | 439/456同批锚点LP相对raw改善 | 主项改善、retain未达14/33 |
| COV16 | 9 | 20 | 29 | 1788/1824相对raw改善，全程覆盖33锚点 | F1与实际残差门未过 |
| CUM16 | 9 | 23 | 41 | 1731/1824相对raw改善 | 累计线性下限不等于实际LP保护 |

此表来自历史报告与exp004事后审计，分母和旧主项方向特意保留；旧D_nonoverlap是保持/恢复研究目标，不是当前“遗忘越强越好”的主指标。

COV16初次run在57/114步、Dykstra300轮上限退出；重跑提高固定求解上限后完成。COV16写回残差最小−3.62e−5、9/1824低于−1e−5；CUM16累计残差最小−4.96e−5、4/1824低于−1e−5，真实LP相对初始化最差−25.5761。该数值直接反驳“累计一阶约束足以保证真实累计保持”。

FB16实际反馈将raw锚点下降1780/1824减少到final838/1824；不是每步完全无损。相对COV16的下降率改善约1.54pp，最终retain19/33，没有通过原推进门。不能只报局部成功率而省略最终结果。

来源：[早期GSU诊断报告](../exp/results/新遗忘方法GSU对比报告.md)、[exp004事后审计](../exp/runs/exp004_gsu_retain_protection/posthoc_audit.md)、[FB16原始metrics](../exp/runs/exp005_gsu_actual_feedback/runs/run_20261020_1714_fb16/metrics.json)。

### 10.2 A/B组完整切片对比

下表从各run的metrics重建，全部为Qwen2.5-1.5B、seed42、114步。A0/B0与A1/B1是同设定重复端点，不当成独立seed。F1直接、覆盖与全hop2分别报告。

| 组 | F0↓ | 五集hop2↓ | F1覆盖↓ | F1直接审计 | 留出retain↑ | F1答案LP |
|---|---|---|---|---|---|---|
| A0 | 24.24% | 44.94% | 69.33% | 100.00% | 45.45% | -10.3263 |
| A1 | 21.21% | 30.34% | 29.33% | 20.00% | 42.42% | -21.0180 |
| A2 | 27.27% | 55.06% | 76.00% | 100.00% | 57.58% | -6.8561 |
| A2R | 30.30% | 55.81% | 85.33% | 100.00% | 60.61% | -8.9229 |
| A3 | 18.18% | 32.21% | 62.67% | 100.00% | 45.45% | -6.5028 |
| A3R | 39.39% | 58.43% | 97.33% | 100.00% | 57.58% | -4.3791 |
| A4 | 9.09% | 25.47% | 28.00% | 20.00% | 27.27% | -21.5135 |
| B0 | 24.24% | 44.94% | 69.33% | 100.00% | 45.45% | -10.3263 |
| B1 | 21.21% | 30.34% | 29.33% | 20.00% | 42.42% | -21.0180 |
| B2 | 39.39% | 49.44% | 58.67% | 80.00% | 42.42% | -8.7399 |
| B2R | 30.30% | 59.55% | 98.67% | 100.00% | 54.55% | -8.3626 |
| B3 | 39.39% | 61.80% | 98.67% | 100.00% | 66.67% | -4.9228 |
| B3R | 36.36% | 60.67% | 100.00% | 100.00% | 57.58% | -4.0470 |

来源：exp006/exp007的完整run_id清单及哈希见末尾。历史F0-only和F0F1行根据各config单独解释，不合成同监督排名。

B2与A2在初始化/计划及原A/P/S上相同，加入的是旧MLP损伤先验。B2相对B0在覆盖切片有12道对变错、4道错变对；12道对变错中只有4道目标answer-LP下降。答案LP从−11.3204变为−10.5503，margin从.4571变为.2143：排序变化不能单独解释为更深概率遗忘。该配对审计是C组转向实际位移的重要动机。

### 10.3 本地C组八变体最终表


| 排名 | 方法 | 全89路径宏回忆 ↓ | 覆盖多跳 /75 ↓ | F0残留 /33 ↓ | F1 /5 | 留出retain /33 ↑ | 护栏回退 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | C1 | 27.72% | 0 | 12 | 0 | 12 | 103 |
| 2 | C4 | 29.59% | 7 | 12 | 0 | 10 | 107 |
| 3并列 | C3R | 29.96% | 7 | 12 | 0 | 10 | 109 |
| 3并列 | C2 | 29.96% | 7 | 12 | 0 | 10 | 109 |
| 5 | C3 | 30.71% | 7 | 13 | 0 | 10 | 108 |
| 6 | C3D | 31.46% | 9 | 13 | 0 | 10 | 105 |
| 7 | Cg | 32.58% | 9 | 12 | 1 | 10 | 98 |
| 8 | C0 | 44.94% | 52 | 8 | 5 | 15 | 0 |

来源：[exp/runs/exp010_gsu_step_routing/final_comparison_20261023.md](../exp/runs/exp010_gsu_step_routing/final_comparison_20261023.md)。前五组来自batch_20261023_092800_Cgroup_restart1，C3R/C3D/C4来自local_continue_20261023_205325；C3R曾中断后从相同初始化完整重跑，不把中断36步当最终。

C1全89=27.72%优于C组其他行，但未支配C0；C1的F1直接0/5，不能说在保持支持事实时独立切断推理。C3是早期事前主候选、C1是后验探索领先项，该时间线不能改写。

### 10.4 本地C组护栏与更新日志小表


| 组 | 日志步数 | alpha0/raw | alpha.25 | alpha.5 | alpha1 | 最终修正比均值 | 状态 |
|---|---|---|---|---|---|---|---|
| C0 | 114 | 不适用 | 不适用 | 不适用 | 不适用 | 0.2053 | completed |
| Cg | 114 | 43 | 33 | 22 | 16 | 0.0705 | completed |
| C1 | 114 | 30 | 43 | 30 | 11 | 0.0967 | completed |
| C2 | 114 | 39 | 44 | 26 | 5 | 0.0763 | completed |
| C3 | 114 | 35 | 37 | 36 | 6 | 0.0875 | completed |
| C3R | 114 | 37 | 42 | 30 | 5 | 0.0803 | completed |
| C3D | 114 | 40 | 40 | 25 | 9 | 0.0829 | completed |
| C4 | 114 | 40 | 36 | 31 | 7 | 0.0829 | completed |

alpha0表示回到原F0 optimizer提议，而不是跳过整个遗忘更新；不能将“护栏回退103步”误写成103步没有训练。alpha1少且修正比小，提示关联路由可能被进展约束削弱；是否解释全部性能差异要用Cg/C1R/C2D检验，不由触发率直接作因果归因。


<a id="s11"></a>
## 11. 本地固定预算baseline与独立实现

### 11.1 旧114步完整比较

Qwen2.5-1.5B、seed42；包含C组历史行，native KLUE单列。原表顺序仅为全89描述性顺序，不是统一原论文调优排名。

| 方法 | 轨道 | 全89回忆↓ | 覆盖正确/75↓ | F0/33↓ | F1/5审计 | retain/33↑ | 独立retain |
|---|---|---|---|---|---|---|---|
| PDU | 历史matched LoRA | 21.72% | 0 | 7 | 1 | 13 | 是，相对该训练池 |
| GA | 历史matched LoRA | 22.85% | 22 | 6 | 3 | 13 | 是，相对该训练池 |
| C1 | 历史matched LoRA | 27.72% | 0 | 12 | 0 | 12 | 是，相对该训练池 |
| C4 | 历史matched LoRA | 29.59% | 7 | 12 | 0 | 10 | 是，相对该训练池 |
| C3R | 历史matched LoRA | 29.96% | 7 | 12 | 0 | 10 | 是，相对该训练池 |
| C2 | 历史matched LoRA | 29.96% | 7 | 12 | 0 | 10 | 是，相对该训练池 |
| C3 | 历史matched LoRA | 30.71% | 7 | 13 | 0 | 10 | 是，相对该训练池 |
| C3D | 历史matched LoRA | 31.46% | 9 | 13 | 0 | 10 | 是，相对该训练池 |
| Cg | 历史matched LoRA | 32.58% | 9 | 12 | 1 | 10 | 是，相对该训练池 |
| CEU | 历史matched LoRA | 35.21% | 37 | 9 | 5 | 18 | 是，相对该训练池 |
| C0 | 历史matched LoRA | 44.94% | 52 | 8 | 5 | 15 | 是，相对该训练池 |
| DPOmis | 历史matched LoRA | 59.18% | 58 | 11 | 5 | 22 | 是，相对该训练池 |
| UNDIAL | 历史matched LoRA | 62.17% | 63 | 15 | 5 | 23 | 是，相对该训练池 |
| RMU | 历史matched LoRA | 64.42% | 71 | 21 | 5 | 26 | 是，相对该训练池 |
| NPO | 历史matched LoRA | 65.17% | 69 | 11 | 5 | 26 | 是，相对该训练池 |
| KLUE | native | 65.54% | 73 | 8 | 5 | 28 | 否，训练重合 |
| DPOrej | 历史matched LoRA | 67.04% | 70 | 14 | 5 | 24 | 是，相对该训练池 |
| GradDiff | 历史matched LoRA | 74.91% | 74 | 13 | 5 | 24 | 是，相对该训练池 |
| SimNPO | 历史matched LoRA | 75.66% | 75 | 28 | 5 | 24 | 是，相对该训练池 |
| SatImp | 历史matched LoRA | 75.66% | 75 | 28 | 5 | 24 | 是，相对该训练池 |
| WGA | 历史matched LoRA | 75.66% | 75 | 26 | 5 | 24 | 是，相对该训练池 |

来源：[exp/runs/exp016_local_baselines/runs/continue_20261024_064547/comparison.json](../exp/runs/exp016_local_baselines/runs/continue_20261024_064547/comparison.json)。PDU在旧预算中未进入5轮后对偶更新，必须称no-dual先导；CEU短答案跳过另查training_summary。MMLU/PPL该表缺失，不以retain代替。

本地C1 27.72%只是在C变体中领先；PDU21.72%、GA22.85%都更低，且F0残留更少。C1的覆盖0/75优于GA22/75并不等于全路径更优。这个反例说明不能先看覆盖子集，再选有利指标宣布最优。

### 11.2 新独立baselines实现的训练与评测分离

| 方法 | queue job状态 | artifact状态 | update字段 | 训练UA字段 | returncode |
|---|---|---|---|---|---|
| GA | completed | training_completed | 684 | 未提供 | 0 |
| GradDiff | completed | training_completed | 1350 | 未提供 | 0 |
| NPO | completed | training_completed | 1350 | 未提供 | 0 |
| SimNPO | completed | training_completed | 90 | 未提供 | 0 |
| RMU | completed | training_completed | 1350 | 未提供 | 0 |
| DPOmis | completed | training_completed | 1350 | 未提供 | 0 |
| DPOrej | completed | training_completed | 1350 | 未提供 | 0 |
| CEU | completed | training_completed | 18 | 未提供 | 0 |
| WGA | completed | training_completed | 15 | 未提供 | 0 |
| SatImp | completed | training_completed | 15 | 未提供 | 0 |
| UNDIAL | completed | training_completed | 20 | 未提供 | 0 |

来源：[baselines/runs/formal_lora_20261024_120150/queue_status.json](../baselines/runs/formal_lora_20261024_120150/queue_status.json)。聚合状态文件可能残留running，因此本表保留原字段，不能等同实时进程；训练完成与能力评测完成单独核验。

| Method | Phase | Updates | Final training UA | Evaluation complete |
|---|---|---:|---:|---|
| GA | training | 684 | 0.30303030303030304 |  |
| GradDiff | training | 1350 | 0.6363636363636364 |  |
| NPO | training | 1350 | 0.7878787878787878 |  |
| SimNPO | training | 90 | None |  |
| RMU | training | 1350 | 0.8181818181818182 |  |
| DPOmis | training | 1350 | 0.7575757575757576 |  |
| DPOrej | training | 1350 | 0.7575757575757576 |  |
| CEU | training | 18 | None |  |
| WGA | training | 15 | None |  |
| SatImp | training | 15 | None |  |
| UNDIAL | training | 20 | None |  |

上表是独立配方旧汇总原样摘录，Evaluation complete为空不填0；其中UA为训练内终点，不是与C1匹配的独立评测。GA684步、GradDiff/NPO等1350步与114步轨不可混排。SimNPO、CEU、WGA等轮数/有效batch不同，更新少不能单独解释为算法更高效。PDU因epsilon校准推迟、KLUE不进入该LoRA轨，均按独立README说明。

### 11.3 本地与云端同名方法的对齐边界

| 维度 | 必须记录的差异 | 审计后动作 |
|---|---|---|
| 原始目标 | NPO总NLL、SimNPO归一化、DPO保持偏好 | 比同公式而非比方法简称 |
| 训练配方 | LR、batch、epoch、warmup、UA早停、search | 固定预算与来源配方分面板 |
| token/prompt | 裸QA、Figure8模板、选项顺序、tie | 同checkpoint交叉评测 |
| 参数域 | 全text LoRA、RMU restricted、native | 原生/适配分轨，参数量单列 |
| 计算资源 | CPU流式、checkpointing、dtype、参考缓存 | 先实模smoke/数值门，再报成绩 |
| 选择规则 | 开发集、早停集、确认集 | 不能按测试最优选赢家 |

独立实现正文和来源锁：[baselines/README.md](../baselines/README.md)、[baselines/source_lock.json](../baselines/source_lock.json)。该目录后续变化不能回填云端旧run的算法合同。


<a id="s12"></a>
## 12. 云端小模型完整结果

### 12.1 exp008/009/011经exp012补评大表

seed42单次，模型内比较。C2行为引用其冻结训练评测，其他baseline统一补全89路径；提示为历史裸QA全串均值。GA-F0F1扩展监督单列。

| 模型 | 方法/监督 | 全89↓ | 覆盖75↓ | F0↓ | 审计retain↑ | MMLU↑ | ΔMMLU pp | PPL↓ | PPL/Original |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | Original | 65.17% | 86.67% | 72.73% | 69.70% | 73.68% | 0.00 | 8.46 | 1.00 |
| Qwen3.5-4B | GA [F0] | 38.20% | 16.00% | 30.30% | 24.24% | 69.30% | -4.39 | 565697.92 | 66898.16 |
| Qwen3.5-4B | GradDiff [F0] | 65.54% | 61.33% | 36.36% | 48.48% | 74.04% | 0.35 | 10.99 | 1.30 |
| Qwen3.5-4B | NPO [F0] | 66.67% | 86.67% | 42.42% | 69.70% | 74.04% | 0.35 | 8.46 | 1.00 |
| Qwen3.5-4B | SimNPO [F0] | 71.54% | 98.67% | 75.76% | 78.79% | 75.09% | 1.40 | 8.49 | 1.00 |
| Qwen3.5-4B | GA [F0F1] | 37.45% | 12.00% | 30.30% | 24.24% | 69.12% | -4.56 | 233689.91 | 27635.64 |
| Qwen3.5-4B | RMU [F0] | 69.29% | 88.00% | 54.55% | 63.64% | 73.68% | 0.00 | 8.64 | 1.02 |
| Qwen3.5-4B | GSU-C2 [F0] | 38.58% | 16.00% | 30.30% | 24.24% | 70.00% | -3.68 | 663393.67 | 78451.44 |
| Llama-3.2-3B-Instruct | Original | 70.41% | 92.00% | 87.88% | 87.88% | 65.26% | 0.00 | 9.62 | 1.00 |
| Llama-3.2-3B-Instruct | GA [F0] | 30.34% | 1.33% | 24.24% | 21.21% | 65.26% | 0.00 | 3407.02 | 354.15 |
| Llama-3.2-3B-Instruct | GradDiff [F0] | 60.67% | 84.00% | 42.42% | 66.67% | 65.09% | -0.18 | 9.84 | 1.02 |
| Llama-3.2-3B-Instruct | NPO [F0] | 71.54% | 92.00% | 72.73% | 87.88% | 65.26% | 0.00 | 9.66 | 1.00 |
| Llama-3.2-3B-Instruct | SimNPO [F0] | 71.54% | 92.00% | 75.76% | 87.88% | 65.09% | -0.18 | 9.70 | 1.01 |
| Llama-3.2-3B-Instruct | GA [F0F1] | 27.72% | 0.00% | 24.24% | 21.21% | 64.74% | -0.53 | 871.76 | 90.62 |
| Llama-3.2-3B-Instruct | RMU [F0] | 69.66% | 90.67% | 84.85% | 90.91% | 64.91% | -0.35 | 9.68 | 1.01 |
| Llama-3.2-3B-Instruct | GSU-C2 [F0] | 30.34% | 1.33% | 24.24% | 21.21% | 65.79% | 0.53 | 1033.64 | 107.44 |

来源：[exp/analysis/cloud_comparison_20261022/evidence/exp012_capability/comparison.json](../exp/analysis/cloud_comparison_20261022/evidence/exp012_capability/comparison.json)；逐题、subject及PPL原始输出在相邻endpoint目录。

小模型C2与GA主要行为接近，而PPL显著高于Original，说明此前只看三四个MCQA数不足以判断保持成功。Llama-C2 PPL约Original107倍、Qwen-C2约78451倍；这不是MMLU的同义指标。也不能把小模型特定极端值复制成8B预测。

### 12.2 exp014旧C1与对齐重跑

| 模型 | 版本 | 全89↓ | F0↓ | retain↑ | MMLU↑ | PPL↓ |
|---|---|---|---|---|---|---|
| llama32_3b | 旧云端C1，seed42 | 30.34% | 24.24% | 21.21% | 65.61% | 1034.74 |
| qwen35_4b | 旧云端C1，seed42 | 38.20% | 30.30% | 24.24% | 70.18% | 468560.98 |

来源：exp/runs/exp014_cloud_c1/runs/batch_20261023_01下两模型audit。batch_20261023_03_local_aligned后来切到本地worker；本次未回收其全部小模型指标，不能用旧C1填新版本行。8B C1/C2表使用独立exp016实际完成结果。

### 12.3 exp015完整小模型来源重实现

两个模型各12个matched-LoRA方法，加各一个native KLUE，共26个完成端点；尚未成功的全原生配置不在性能表里。

| 模型 | 方法 | 轨道 | 全89↓ | F0↓ | retain↑ | retain独立 | MMLU↑ | PPL↓ |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct | KLUE | native | 70.04% | 30.30% | 93.94% | 否，训练重合 | 65.44% | 9.68 |
| Qwen3.5-4B | KLUE | native | 64.04% | 30.30% | 84.85% | 否，训练重合 | 74.04% | 8.42 |
| Llama-3.2-3B-Instruct | GA | matched_lora | 32.58% | 24.24% | 21.21% | 是 | 64.74% | 572.41 |
| Qwen3.5-4B | GA | matched_lora | 37.83% | 30.30% | 24.24% | 是 | 70.18% | 734669.41 |
| Llama-3.2-3B-Instruct | GradDiff | matched_lora | 64.79% | 42.42% | 69.70% | 是 | 65.09% | 9.84 |
| Qwen3.5-4B | GradDiff | matched_lora | 61.80% | 24.24% | 48.48% | 是 | 74.21% | 9.74 |
| Llama-3.2-3B-Instruct | NPO | matched_lora | 67.04% | 45.45% | 84.85% | 是 | 64.74% | 9.69 |
| Qwen3.5-4B | NPO | matched_lora | 68.91% | 27.27% | 60.61% | 是 | 74.56% | 8.70 |
| Llama-3.2-3B-Instruct | SimNPO | matched_lora | 73.03% | 87.88% | 90.91% | 是 | 65.44% | 9.69 |
| Qwen3.5-4B | SimNPO | matched_lora | 72.66% | 87.88% | 87.88% | 是 | 75.61% | 8.49 |
| Llama-3.2-3B-Instruct | RMU | matched_lora | 70.41% | 87.88% | 93.94% | 是 | 64.21% | 9.65 |
| Qwen3.5-4B | RMU | matched_lora | 66.67% | 66.67% | 72.73% | 是 | 72.28% | 9.25 |
| Llama-3.2-3B-Instruct | DPOmis | matched_lora | 67.04% | 30.30% | 81.82% | 是 | 65.44% | 9.71 |
| Qwen3.5-4B | DPOmis | matched_lora | 65.17% | 21.21% | 66.67% | 是 | 74.56% | 8.83 |
| Llama-3.2-3B-Instruct | DPOrej | matched_lora | 70.04% | 57.58% | 87.88% | 是 | 64.56% | 9.68 |
| Qwen3.5-4B | DPOrej | matched_lora | 67.04% | 27.27% | 69.70% | 是 | 75.09% | 8.41 |
| Llama-3.2-3B-Instruct | CEU | matched_lora | 47.19% | 39.39% | 63.64% | 是 | 65.61% | 9.90 |
| Qwen3.5-4B | CEU | matched_lora | 40.82% | 21.21% | 45.45% | 是 | 73.16% | 8.56 |
| Llama-3.2-3B-Instruct | PDU | matched_lora | 36.70% | 21.21% | 42.42% | 是 | 65.79% | 32.14 |
| Qwen3.5-4B | PDU | matched_lora | 38.20% | 33.33% | 36.36% | 是 | 74.74% | 15.77 |
| Llama-3.2-3B-Instruct | WGA | matched_lora | 71.91% | 81.82% | 90.91% | 是 | 65.44% | 9.69 |
| Qwen3.5-4B | WGA | matched_lora | 71.16% | 81.82% | 84.85% | 是 | 75.44% | 8.47 |
| Llama-3.2-3B-Instruct | SatImp | matched_lora | 72.66% | 87.88% | 90.91% | 是 | 65.61% | 9.68 |
| Qwen3.5-4B | SatImp | matched_lora | 72.28% | 87.88% | 87.88% | 是 | 75.09% | 8.49 |
| Llama-3.2-3B-Instruct | UNDIAL | matched_lora | 65.92% | 72.73% | 81.82% | 是 | 64.39% | 9.65 |
| Qwen3.5-4B | UNDIAL | matched_lora | 59.93% | 51.52% | 60.61% | 是 | 73.51% | 8.43 |

来源：[exp/analysis/technical_report_20261025/evidence/exp015/comparison.json](../exp/analysis/technical_report_20261025/evidence/exp015/comparison.json)。完整状态见同目录status.json；原生GA等24个smoke失败、对应正式端点跳过，绝不能把completed_with_failures写成所有方法完成。KLUE停止条件是训练UA，且使用66 retain；其高retain不与LoRA后33独立审计等同。


<a id="s13"></a>
## 13. 8B三种子大表、逐seed小表与排序

### 13.1 批次与计数

两模型×三seed×27配置=162次训练；27=12种matched-LoRA baseline+13种native baseline+C1+C2。Original每模型一次；smoke和C方法额外audit是子任务，不计成额外独立训练。整体230个命令任务不是230次正式训练。

快照来源：[exp/analysis/technical_report_20261025/evidence/exp016/manifest.json](../exp/analysis/technical_report_20261025/evidence/exp016/manifest.json)、[exp/analysis/technical_report_20261025/evidence/exp016/status.json](../exp/analysis/technical_report_20261025/evidence/exp016/status.json)、[exp/analysis/technical_report_20261025/evidence/exp016/comparison.json](../exp/analysis/technical_report_20261025/evidence/exp016/comparison.json)。


| 方法 | 轨道 | 计划正式训练 | 已完成 | 失败/跳过记录 | 说明 |
|---|---|---|---|---|---|
| GA | matched_lora | 6 | 6 | 0 | 已齐或待排队 |
| GA | native | 6 | 0 | 0 | 已齐或待排队 |
| GradDiff | matched_lora | 6 | 6 | 0 | 已齐或待排队 |
| GradDiff | native | 6 | 0 | 0 | 已齐或待排队 |
| NPO | matched_lora | 6 | 6 | 0 | 已齐或待排队 |
| NPO | native | 6 | 0 | 0 | 已齐或待排队 |
| SimNPO | matched_lora | 6 | 6 | 0 | 已齐或待排队 |
| SimNPO | native | 6 | 0 | 0 | 已齐或待排队 |
| RMU | matched_lora | 6 | 6 | 0 | 已齐或待排队 |
| RMU | native | 6 | 0 | 0 | 已齐或待排队 |
| DPOmis | matched_lora | 6 | 0 | 0 | 当前运行 |
| DPOmis | native | 6 | 0 | 0 | 已齐或待排队 |
| DPOrej | matched_lora | 6 | 0 | 0 | 已齐或待排队 |
| DPOrej | native | 6 | 0 | 0 | 已齐或待排队 |
| CEU | matched_lora | 6 | 0 | 0 | 已齐或待排队 |
| CEU | native | 6 | 0 | 0 | 已齐或待排队 |
| PDU | matched_lora | 6 | 0 | 0 | 已齐或待排队 |
| PDU | native | 6 | 0 | 0 | 已齐或待排队 |
| WGA | matched_lora | 6 | 0 | 0 | 已齐或待排队 |
| WGA | native | 6 | 0 | 0 | 已齐或待排队 |
| SatImp | matched_lora | 6 | 0 | 0 | 已齐或待排队 |
| SatImp | native | 6 | 0 | 0 | 已齐或待排队 |
| UNDIAL | matched_lora | 6 | 0 | 0 | 已齐或待排队 |
| UNDIAL | native | 6 | 0 | 0 | 已齐或待排队 |
| KLUE | native | 6 | 0 | 6 | 前置失败/跳过 |
| C1 | matched_lora | 6 | 6 | 0 | 已齐或待排队 |
| C2 | matched_lora | 6 | 6 | 0 | 已齐或待排队 |

“失败/跳过”字段区分smoke故障与未运行正式任务，不能把所有条目当作独立正式失败。当前完成的8B能力表仅来自实际成功端点。

### 13.2 三种子均值和样本标准差

百分比±值单位为百分点。Original只报告单次值，不能写作三seed标准差0。PPL采用三个seed的算术均值/样本标准差，不是把三模型语料重新并成一次评测。


#### Meta-Llama-3.1-8B-Instruct

| 方法 | 轨道 | n/seeds | 全89↓ | F0↓ | 独立retain↑ | MMLU↑ | PPL↓ |
|---|---|---|---|---|---|---|---|
| Original | none | 1/Original | 66.67% | 78.79% | 84.85% | 69.12% | 6.43 |
| GA | matched_lora | 3/42,43,44 | 34.58% ± 0.57pp | 24.24% ± 3.03pp | 28.28% ± 1.75pp | 65.61% ± 5.48pp | 145.66 ± 185.12 |
| GradDiff | matched_lora | 3/42,43,44 | 52.43% ± 2.46pp | 31.31% ± 1.75pp | 83.84% ± 9.26pp | 68.83% ± 0.66pp | 7.48 ± 1.12 |
| NPO | matched_lora | 3/42,43,44 | 57.30% ± 1.63pp | 37.37% ± 3.50pp | 84.85% ± 3.03pp | 69.06% ± 0.27pp | 6.47 ± 0.03 |
| SimNPO | matched_lora | 3/42,43,44 | 75.78% ± 1.32pp | 90.91% ± 3.03pp | 90.91% ± 0.00pp | 68.95% ± 0.18pp | 6.44 ± 0.00 |
| RMU | matched_lora | 3/42,43,44 | 66.79% ± 0.57pp | 80.81% ± 1.75pp | 87.88% ± 0.00pp | 69.65% ± 0.61pp | 6.44 ± 0.01 |
| C1 | matched_lora | 3/42,43,44 | 34.21% ± 1.92pp | 25.25% ± 1.75pp | 27.27% ± 3.03pp | 65.61% ± 4.08pp | 92.08 ± 104.16 |
| C2 | matched_lora | 3/42,43,44 | 34.08% ± 1.35pp | 25.25% ± 1.75pp | 27.27% ± 3.03pp | 65.09% ± 4.92pp | 111.80 ± 123.69 |


#### Qwen3-8B

| 方法 | 轨道 | n/seeds | 全89↓ | F0↓ | 独立retain↑ | MMLU↑ | PPL↓ |
|---|---|---|---|---|---|---|---|
| Original | none | 1/Original | 56.18% | 69.70% | 87.88% | 75.09% | 8.58 |
| GA | matched_lora | 3/42,43,44 | 34.96% ± 0.94pp | 28.28% ± 1.75pp | 25.25% ± 1.75pp | 75.96% ± 0.30pp | 51.57 ± 17.59 |
| GradDiff | matched_lora | 3/42,43,44 | 57.43% ± 5.11pp | 36.36% ± 3.03pp | 76.77% ± 1.75pp | 75.44% ± 0.00pp | 9.38 ± 0.07 |
| NPO | matched_lora | 3/42,43,44 | 52.18% ± 5.42pp | 31.31% ± 3.50pp | 74.75% ± 6.31pp | 75.38% ± 0.27pp | 8.78 ± 0.01 |
| SimNPO | matched_lora | 3/42,43,44 | 71.29% ± 1.08pp | 90.91% ± 3.03pp | 91.92% ± 1.75pp | 76.02% ± 0.27pp | 8.34 ± 0.00 |
| RMU | matched_lora | 3/42,43,44 | 56.80% ± 0.22pp | 72.73% ± 0.00pp | 84.85% ± 0.00pp | 75.61% ± 0.46pp | 8.76 ± 0.01 |
| C1 | matched_lora | 3/42,43,44 | 34.71% ± 0.22pp | 28.28% ± 1.75pp | 26.26% ± 1.75pp | 75.61% ± 0.30pp | 35.62 ± 10.50 |
| C2 | matched_lora | 3/42,43,44 | 34.71% ± 0.57pp | 28.28% ± 1.75pp | 27.27% ± 0.00pp | 75.79% ± 0.30pp | 39.29 ± 11.73 |


### 13.3 全部已完成端点逐seed大表



#### Meta-Llama-3.1-8B-Instruct：逐seed

| 方法 | seed | 全89↓ | F0↓ | retain↑ | MMLU↑ | ΔMMLU pp | PPL↓ | PPL/Orig | run_id |
|---|---|---|---|---|---|---|---|---|---|
| Original | 单次 | 66.67% | 78.79% | 84.85% | 69.12% | 0.00 | 6.428 | 1.00 | llama31_8b_Original |
| GA | 42 | 34.46% | 21.21% | 27.27% | 68.42% | -0.70 | 46.097 | 7.17 | llama31_8b_GA_matched_lora_seed42 |
| GA | 43 | 35.21% | 27.27% | 27.27% | 59.30% | -9.82 | 359.251 | 55.89 | llama31_8b_GA_matched_lora_seed43 |
| GA | 44 | 34.08% | 24.24% | 30.30% | 69.12% | 0.00 | 31.636 | 4.92 | llama31_8b_GA_matched_lora_seed44 |
| GradDiff | 42 | 55.06% | 30.30% | 75.76% | 69.30% | 0.18 | 6.747 | 1.05 | llama31_8b_GradDiff_matched_lora_seed42 |
| GradDiff | 43 | 52.06% | 33.33% | 93.94% | 69.12% | 0.00 | 6.926 | 1.08 | llama31_8b_GradDiff_matched_lora_seed43 |
| GradDiff | 44 | 50.19% | 30.30% | 81.82% | 68.07% | -1.05 | 8.774 | 1.36 | llama31_8b_GradDiff_matched_lora_seed44 |
| NPO | 42 | 58.05% | 39.39% | 81.82% | 69.30% | 0.18 | 6.459 | 1.00 | llama31_8b_NPO_matched_lora_seed42 |
| NPO | 43 | 55.43% | 39.39% | 84.85% | 69.12% | 0.00 | 6.495 | 1.01 | llama31_8b_NPO_matched_lora_seed43 |
| NPO | 44 | 58.43% | 33.33% | 87.88% | 68.77% | -0.35 | 6.445 | 1.00 | llama31_8b_NPO_matched_lora_seed44 |
| SimNPO | 42 | 77.15% | 90.91% | 90.91% | 68.95% | -0.18 | 6.441 | 1.00 | llama31_8b_SimNPO_matched_lora_seed42 |
| SimNPO | 43 | 74.53% | 87.88% | 90.91% | 69.12% | 0.00 | 6.447 | 1.00 | llama31_8b_SimNPO_matched_lora_seed43 |
| SimNPO | 44 | 75.66% | 93.94% | 90.91% | 68.77% | -0.35 | 6.437 | 1.00 | llama31_8b_SimNPO_matched_lora_seed44 |
| RMU | 42 | 66.29% | 81.82% | 87.88% | 69.30% | 0.18 | 6.444 | 1.00 | llama31_8b_RMU_matched_lora_seed42 |
| RMU | 43 | 66.67% | 78.79% | 87.88% | 70.35% | 1.23 | 6.428 | 1.00 | llama31_8b_RMU_matched_lora_seed43 |
| RMU | 44 | 67.42% | 81.82% | 87.88% | 69.30% | 0.18 | 6.435 | 1.00 | llama31_8b_RMU_matched_lora_seed44 |
| C1 | 42 | 32.58% | 24.24% | 24.24% | 64.91% | -4.21 | 40.929 | 6.37 | llama31_8b_C1_matched_lora_seed42_audit |
| C1 | 43 | 36.33% | 27.27% | 27.27% | 61.93% | -7.19 | 211.926 | 32.97 | llama31_8b_C1_matched_lora_seed43_audit |
| C1 | 44 | 33.71% | 24.24% | 30.30% | 70.00% | 0.88 | 23.375 | 3.64 | llama31_8b_C1_matched_lora_seed44_audit |
| C2 | 42 | 32.96% | 24.24% | 24.24% | 65.44% | -3.68 | 53.887 | 8.38 | llama31_8b_C2_matched_lora_seed42_audit |
| C2 | 43 | 35.58% | 27.27% | 27.27% | 60.00% | -9.12 | 253.822 | 39.49 | llama31_8b_C2_matched_lora_seed43_audit |
| C2 | 44 | 33.71% | 24.24% | 30.30% | 69.82% | 0.70 | 27.690 | 4.31 | llama31_8b_C2_matched_lora_seed44_audit |


#### Qwen3-8B：逐seed

| 方法 | seed | 全89↓ | F0↓ | retain↑ | MMLU↑ | ΔMMLU pp | PPL↓ | PPL/Orig | run_id |
|---|---|---|---|---|---|---|---|---|---|
| Original | 单次 | 56.18% | 69.70% | 87.88% | 75.09% | 0.00 | 8.582 | 1.00 | qwen3_8b_Original |
| GA | 42 | 34.83% | 27.27% | 27.27% | 75.79% | 0.70 | 54.541 | 6.36 | qwen3_8b_GA_matched_lora_seed42 |
| GA | 43 | 34.08% | 27.27% | 24.24% | 76.32% | 1.23 | 32.682 | 3.81 | qwen3_8b_GA_matched_lora_seed43 |
| GA | 44 | 35.96% | 30.30% | 24.24% | 75.79% | 0.70 | 67.480 | 7.86 | qwen3_8b_GA_matched_lora_seed44 |
| GradDiff | 42 | 63.30% | 36.36% | 78.79% | 75.44% | 0.35 | 9.462 | 1.10 | qwen3_8b_GradDiff_matched_lora_seed42 |
| GradDiff | 43 | 53.93% | 33.33% | 75.76% | 75.44% | 0.35 | 9.338 | 1.09 | qwen3_8b_GradDiff_matched_lora_seed43 |
| GradDiff | 44 | 55.06% | 39.39% | 75.76% | 75.44% | 0.35 | 9.353 | 1.09 | qwen3_8b_GradDiff_matched_lora_seed44 |
| NPO | 42 | 58.43% | 33.33% | 81.82% | 75.44% | 0.35 | 8.793 | 1.02 | qwen3_8b_NPO_matched_lora_seed42 |
| NPO | 43 | 48.69% | 33.33% | 69.70% | 75.61% | 0.53 | 8.782 | 1.02 | qwen3_8b_NPO_matched_lora_seed43 |
| NPO | 44 | 49.44% | 27.27% | 72.73% | 75.09% | 0.00 | 8.765 | 1.02 | qwen3_8b_NPO_matched_lora_seed44 |
| SimNPO | 42 | 71.91% | 93.94% | 93.94% | 75.96% | 0.88 | 8.339 | 0.97 | qwen3_8b_SimNPO_matched_lora_seed42 |
| SimNPO | 43 | 71.91% | 90.91% | 90.91% | 75.79% | 0.70 | 8.345 | 0.97 | qwen3_8b_SimNPO_matched_lora_seed43 |
| SimNPO | 44 | 70.04% | 87.88% | 90.91% | 76.32% | 1.23 | 8.343 | 0.97 | qwen3_8b_SimNPO_matched_lora_seed44 |
| RMU | 42 | 56.93% | 72.73% | 84.85% | 75.09% | 0.00 | 8.774 | 1.02 | qwen3_8b_RMU_matched_lora_seed42 |
| RMU | 43 | 56.55% | 72.73% | 84.85% | 75.79% | 0.70 | 8.762 | 1.02 | qwen3_8b_RMU_matched_lora_seed43 |
| RMU | 44 | 56.93% | 72.73% | 84.85% | 75.96% | 0.88 | 8.745 | 1.02 | qwen3_8b_RMU_matched_lora_seed44 |
| C1 | 42 | 34.83% | 27.27% | 27.27% | 75.96% | 0.88 | 33.307 | 3.88 | qwen3_8b_C1_matched_lora_seed42_audit |
| C1 | 43 | 34.46% | 27.27% | 24.24% | 75.44% | 0.35 | 26.464 | 3.08 | qwen3_8b_C1_matched_lora_seed43_audit |
| C1 | 44 | 34.83% | 30.30% | 27.27% | 75.44% | 0.35 | 47.078 | 5.49 | qwen3_8b_C1_matched_lora_seed44_audit |
| C2 | 42 | 34.83% | 27.27% | 27.27% | 75.44% | 0.35 | 43.822 | 5.11 | qwen3_8b_C2_matched_lora_seed42_audit |
| C2 | 43 | 34.08% | 27.27% | 27.27% | 75.96% | 0.88 | 25.979 | 3.03 | qwen3_8b_C2_matched_lora_seed43_audit |
| C2 | 44 | 35.21% | 30.30% | 27.27% | 75.96% | 0.88 | 48.081 | 5.60 | qwen3_8b_C2_matched_lora_seed44_audit |


### 13.4 分指标描述性排序

只对同模型、matched-LoRA且三个seed已齐的方法按均值排名，Original、原生、缺种子和失败项不入榜；并列采用1/2/2/4竞争排名。不给跨指标总分，不把单列第一写成算法最优。


#### Meta-Llama-3.1-8B-Instruct：已齐方法的列内排名

| 方法 | 全89名次↓ | F0名次↓ | retain名次↑ | MMLU名次↑ | PPL名次↓ |
|---|---|---|---|---|---|
| C2 | 1 | 2 | 6 | 7 | 6 |
| C1 | 2 | 2 | 6 | 5 | 5 |
| GA | 3 | 1 | 5 | 5 | 7 |
| GradDiff | 4 | 4 | 4 | 4 | 4 |
| NPO | 5 | 5 | 3 | 2 | 3 |
| RMU | 6 | 6 | 2 | 1 | 1 |
| SimNPO | 7 | 7 | 1 | 3 | 2 |


#### Qwen3-8B：已齐方法的列内排名

| 方法 | 全89名次↓ | F0名次↓ | retain名次↑ | MMLU名次↑ | PPL名次↓ |
|---|---|---|---|---|---|
| C1 | 1 | 1 | 6 | 4 | 5 |
| C2 | 1 | 1 | 5 | 3 | 6 |
| GA | 3 | 1 | 7 | 2 | 7 |
| NPO | 4 | 4 | 4 | 7 | 3 |
| RMU | 5 | 6 | 2 | 4 | 2 |
| GradDiff | 6 | 5 | 3 | 6 | 4 |
| SimNPO | 7 | 7 | 1 | 1 | 1 |


排名随尚未完成方法加入可能变化；这只是指定配方和预算的均值次序。Llama高PPL方差与小多跳差距说明需要逐seed及配对统计，不以小数点多写几位增强证据。


<a id="s14"></a>
## 14. 配对差值、PPL和机制解释

### 14.1 同seed配对：C1/C2相对GA

以下差值定义为候选−参照，全部先在相同模型/seed匹配，百分比差单位pp。全89/F0/ΔNLL/PPL期望负，保持/MMLU期望正；没有统一综合分。

| 模型 | 配对 | seed | 全89 Δpp | F0 Δpp | retain Δpp | MMLU Δpp | PPL差 | meanNLL差 |
|---|---|---|---|---|---|---|---|---|
| Meta-Llama-3.1-8B-Instruct | C1−GA | 42 | -1.873 | 3.030 | -3.030 | -3.509 | -5.169 | -0.1189 |
| Meta-Llama-3.1-8B-Instruct | C1−GA | 43 | 1.124 | 0.000 | 0.000 | 2.632 | -147.325 | -0.5278 |
| Meta-Llama-3.1-8B-Instruct | C1−GA | 44 | -0.375 | 0.000 | 0.000 | 0.877 | -8.261 | -0.3026 |
| Meta-Llama-3.1-8B-Instruct | C2−GA | 42 | -1.498 | 3.030 | -3.030 | -2.982 | 7.790 | 0.1561 |
| Meta-Llama-3.1-8B-Instruct | C2−GA | 43 | 0.375 | 0.000 | 0.000 | 0.702 | -105.429 | -0.3474 |
| Meta-Llama-3.1-8B-Instruct | C2−GA | 44 | -0.375 | 0.000 | 0.000 | 0.702 | -3.946 | -0.1332 |
| Meta-Llama-3.1-8B-Instruct | C2−C1 | 42 | 0.375 | 0.000 | 0.000 | 0.526 | 12.958 | 0.2751 |
| Meta-Llama-3.1-8B-Instruct | C2−C1 | 43 | -0.749 | 0.000 | 0.000 | -1.930 | 41.896 | 0.1804 |
| Meta-Llama-3.1-8B-Instruct | C2−C1 | 44 | 0.000 | 0.000 | 0.000 | -0.175 | 4.316 | 0.1694 |
| Qwen3-8B | C1−GA | 42 | 0.000 | 0.000 | 0.000 | 0.175 | -21.234 | -0.4932 |
| Qwen3-8B | C1−GA | 43 | 0.375 | 0.000 | 0.000 | -0.877 | -6.217 | -0.2110 |
| Qwen3-8B | C1−GA | 44 | -1.124 | 0.000 | 3.030 | -0.351 | -20.402 | -0.3600 |
| Qwen3-8B | C2−GA | 42 | 0.000 | 0.000 | 0.000 | -0.351 | -10.719 | -0.2188 |
| Qwen3-8B | C2−GA | 43 | 0.000 | 0.000 | 3.030 | -0.351 | -6.703 | -0.2295 |
| Qwen3-8B | C2−GA | 44 | -0.749 | 0.000 | 3.030 | 0.175 | -19.399 | -0.3390 |
| Qwen3-8B | C2−C1 | 42 | 0.000 | 0.000 | 0.000 | -0.526 | 10.516 | 0.2744 |
| Qwen3-8B | C2−C1 | 43 | -0.375 | 0.000 | 3.030 | 0.526 | -0.485 | -0.0185 |
| Qwen3-8B | C2−C1 | 44 | 0.375 | 0.000 | 0.000 | 0.526 | 1.003 | 0.0211 |

### 14.2 配对差的均值与波动


| 模型 | 配对 | n | 全89差均值±SD pp | retain差均值pp | PPL差均值 | NLL差均值 | 全89严格更低种子数 |
|---|---|---|---|---|---|---|---|
| Meta-Llama-3.1-8B-Instruct | C1−GA | 3 | -0.375 ± 1.498 | -1.010 | -53.585 | -0.3165 | 2/3 |
| Meta-Llama-3.1-8B-Instruct | C2−GA | 3 | -0.499 ± 0.943 | -1.010 | -33.862 | -0.1082 | 2/3 |
| Meta-Llama-3.1-8B-Instruct | C2−C1 | 3 | -0.125 ± 0.572 | 0.000 | 19.723 | 0.2083 | 1/3 |
| Qwen3-8B | C1−GA | 3 | -0.250 ± 0.780 | 1.010 | -15.951 | -0.3547 | 1/3 |
| Qwen3-8B | C2−GA | 3 | -0.250 ± 0.432 | 2.020 | -12.274 | -0.2624 | 1/3 |
| Qwen3-8B | C2−C1 | 3 | -0.000 ± 0.375 | 1.010 | 3.678 | 0.0923 | 1/3 |

这里的种子数不是显著性检验；三次中2次更低不等于p<.05。完整统计需对共享主体/路径整簇配对，多个方法比较要登记校正；既不能只对267个相关改写独立bootstrap，也不能用两根CI是否重叠替代配对比较。

### 14.3 为什么不能说“方向空间无效”

C2相对C1增加共享方向过滤，但共同的F0护栏、反馈和修正半径会改变其最终贡献。若方向产生的候选经护栏大幅撤销，终点接近不等于方向信号本身不存在；若随机同rank子空间同样有效，则结构特异性解释不成立。C4/C2D以及逐步raw→route→feedback→final审计是必要区分，不靠最终一列推因果。

### 14.4 为什么PPL与MMLU看起来不一致

MMLU衡量固定多学科选项判断，PPL覆盖一般文章token预测。它们测不同能力投影；一个稳定、另一个恶化并不逻辑矛盾。Qwen的MMLU略升不能抵消retain从约88%降至约26%的观察，也不能将PPL上升全部归因于针对性多跳封堵。

Llama-C1三个seed的PPL差异大，说明均值易受单seed高值影响；报告中同时列出了逐seed、标准差和NLL差。若需几何均值或稳健中位数，应明确增加审计项，不事后替换原定主报告。

### 14.5 关联代价与无关损伤

允许某些共享支持知识承担遗忘代价，不要求所有retain完全不变。要建立“有针对性的代价”，需在看输出前分层：直接事实重合、共享主体/客体、共享支持边、路径邻近、无观测重合；匹配原始掌握/关系难度并报告交集规模。最后一类不等于语义绝对独立。

若各层同样大幅损失且通用PPL也恶化，则“只削弱必要桥接知识”不足以解释结果。若仅MCQA选项相对排序改变而目标答案LP未降，也不能宣称深层记忆被清除。

### 14.6 帕累托分析合同

同模型、同监督权限和明示预算下，点a在所有预定义坐标不劣且至少一项严格优于b时，a才支配b。必须包含F0有效性，不把Original或未遗忘的高utility点当成功。当前只可报告有限样本经验前沿；随机对照、调参和独立确认不足时，不宣传全局最优。

不同机制A2/A3/B2不应平滑连成曲线；同方法强度扫描只连实测点。F0F1与F0-only分别计算条件前沿，同时展示额外监督参照；不隐去支配GSU的点。


<a id="s15"></a>
## 15. 故障、复现缺口和资源管理

### 15.1 已观察故障表

| 实验 | 错误/异常 | 性质 | 处理和记录 |
|---|---|---|---|
| 3B早期G组 | PEFT重复适配器警告 | 基座复用风险 | 独立底座clean重跑，旧run留档 |
| exp004 COV16 | 57步Dykstra300循环上限 | 数值求解门失败 | 独立重跑提高固定上限，不能用失败run填结果 |
| exp004 CUM16 | 原诊断字段未减累计目标 | 记录字段错误 | 从原始日志重算；实际LP保护失败仍成立 |
| exp010局部续跑 | status仍running但进程已退出 | 状态陈旧 | 从新目录完整续跑，不把36步当114步 |
| exp012早期 | waiting_for_c2控制器退出、HF网络、Xet401、cwd校验 | 调度/网络/路径 | 保留每个失败批次，学术加速与禁用Xet在新批次记录 |
| exp012 smoke | torch.dtype不能JSON序列化 | 输出metadata缺陷 | JSON-safe修复，新batch06完成，不覆盖batch05 |
| exp015 native | YAML科学计数LR被解析为字符串，优化器TypeError | 配置类型错误 | 24smoke失败、24formal跳过，不能算算法退化 |
| exp016 KLUE native | attn_bias strideM/strideH需4对齐 | 流式保存恢复与fused注意力后向兼容 | 2smoke失败、6formal跳过；exp017新worker改math SDPA并回归 |
| 存储门 | checkpoint空间不足 | 资源失败 | 明确失败，不删证据、不改成LoRA后冒充原生 |

错误消息支持的是具体错误发生，不支持“方法本质无效”。toy数值通过也不会自动覆盖真实BF16、变长batch、mask布局和融合内核的所有情况。

### 15.2 当前源代码与冻结run

报告基于源码解释实现，但以下区别必须保持：

- 共享代码中已有学习率float转换，不表示旧exp015已经修复；旧run源码在其快照中。
- 本地新增的decisive_controls/development_evaluation和math SDPA修复属于exp017后续，不能回填exp016 KLUE成绩。
- 同名方法在新baselines/中采用作者来源配方、原文提示和独立早停，不自动替代历史LoRA行。
- 旧源码快照、失败栈和环境文件不可因为重复而删除；文档合并只改变说明，不改实验事实。

### 15.3 checkpoint保留

每模型/方法/seed/轨道保留一个成功最终checkpoint。native与LoRA分轨；smoke不当最终模型；失败权重可删但指标/配置/日志保留，活跃run不能删。

2026-10-24已执行checkpoint-only清理：保留42个最终权重文件，删除63个权重路径，其中35smoke、14旧run、13重复快照链接、1个有完成副本的孤立旧NPO。因多数是硬链接，路径字节和不等于实际回收；估计释放约1.14GiB。明确失败运行当时无权重可删。详见exp/sync/sync_20261024_checkpoint_retention清单。

基础模型未删；逐项校验路径、哈希、状态和活跃进程。原生权重使用lossless XOR/zlib差分或选中神经元原生切片，可恢复性需精确底座及文件哈希；不是量化或LoRA近似。恢复测试通过只认证被测格式，不免除新模型实模roundtrip。

### 15.4 资源边界

截至2026-10-25 12:40云端数据盘约剩7.5GB、系统盘约15GB。每seed最终adapter、原生delta、逐题/学科输出都会继续增长；本报告不因记录空间风险自动删除产物或更改预算。真实模型原生CPU权重、激活、梯度、FP32动量受120GiB容器RAM限制，不能凭宿主机free-h的一TiB可见值判断容器无限内存。

<a id="s16"></a>
## 16. exp017修复、随机对照及调参计划

来源：[exp017实验卡](../exp/runs/exp017_decisive_validation/card.md)。截至云端12:40观察，仅等待exp016依赖；以下是预注册计划，不是已完成证据。

### 16.1 固定补充组

| 组 | 精确定义 | 作用 |
|---|---|---|
| KLUE修复 | math SDPA，关闭fused/flash/memory-efficient/cuDNN相关分支；归因/筛样/参数域不变 | 修复stride故障，实模smoke必过 |
| GA-F0F1 | 38训练事实，114步；评测F0仍33 | 直接扩展监督参照 |
| Cg | 相同反馈/护栏/优化器计划，全组件权重1，无span | 一般护栏/反馈解释 |
| C1R | C1分数按模块类型内部置换，seed+7919，直方图精确保持 | 组件对应关系特异性 |
| C2D | 同组件权重、每张量同rank与None fallback，按名称和seed固定随机子空间 | 关联方向特异性 |

两模型×三seed×5组=30正式训练，smoke单独。C控制8步smoke/114正式，扩展GA2步smoke/114正式，不改变C1/C2旧结果或测试题。

### 16.2 开发集与选择规则

未分配101簇先与官方角色按直接三元组去重，按固定hash排序，取24开发forget、24开发retain，其余53不消费。这个划分是索引/直接事实隔离，不是实体或图完全独立。

方法GA、GradDiff、NPO、SimNPO、RMU；lr网格2.5e-5/5e-5/1e-4，seed42、114步，其余matched配置固定：5×3×2模型=30开发试验。不跑主forget、多跳测试、完整能力或test464来选超参；不保存这些开发试验为最终模型成绩。

~~~python
for model in models:
    for method in fixed_five_baselines:
        trials = run_dev_only(method, learning_rates=[2.5e-5, 5e-5, 1e-4])
        feasible = [t for t in trials
                    if t.dev_direct <= 1/3 and t.dev_retain_drop_pp <= 10]
        if not feasible:
            record_no_selected_config()  # 不事后放宽门槛
            continue
        selected = min(feasible, key=(dev_hop2_macro, dev_direct, learning_rate))
        freeze(selected.learning_rate)
        confirm_from_fresh_base_on_official_F0(seeds=[42,43,44])
    strongest = freeze_best_dev_method_before_official_results()
    confirm_expanded_F0F1(strongest, seeds=[42,43,44])
~~~

最多30个被选配置确认端点，另加每模型最强开发赢家的F0F1三seed最多6个；加固定30组，总计最多66正式端点和30开发试验。无可行解、失败和被跳过不计完成。

### 16.3 统计边界

这轮只搜索LR，不是穷尽论文所有beta/保持系数；开发发生在exp016探索之后，不能重命名成从未见过任何测试结果的纯净确认。所有选择规则和失败分支在卡中固定，多seed报告保持模型/seed/subject配对，测试输出不反哺选择。


<a id="s17"></a>
## 17. RQ证据判定与论文可写范围

### 17.1 RQ—证据—下一门

| RQ | 已有直接观察 | 仍需排除的解释 | 判定与下一证据 |
|---|---|---|---|
| RQ1 原始遗忘与关联回忆如何分离 | G组与各baseline的直接/多跳残留不同 | 底模不会答、选项评分伪影、广泛退化 | 现象有探索证据；同checkpoint评分/生成审计 |
| RQ2 GSU是否改善多跳—能力权衡 | C1/C2与GA均值接近，PPL相对GA低、仍高于Original | 单强度点/坏baseline配方/额外计算 | 最优未确立；开发冻结强baseline、同F0/等计算曲线 |
| RQ3 组件、方向、护栏谁贡献 | 本地C0/Cg/C1/C2/C4及历史随机先验 | C3R不匹配C1；一般缩放足以解释 | 缺8B匹配对照；exp017 Cg/C1R/C2D |
| RQ4 定位与实际更新是否对应 | 位移、锚点LP、回退与历史组件撤销日志 | 相关分数不等于因果电路；反馈被撤销 | 需同模型组件预算干预及预测—实际LP配对 |
| RQ5 跨seed/模型是否稳定 | 两8B C1/C2的3seed完成 | 两模型换架构，F1历史来源共享，Llama高方差 | 有重复观察，无规模因果；独立数据/模型重探测 |
| RQ6 关联代价与通用退化如何分布 | retain、MMLU和PPL给出不同投影 | “代价必要”与“无关误伤”都未识别 | 先冻结关联分层、独立test与生成式行为 |

### 17.2 可以严谨写入方法的局部性质

~~~text
(1) Orthogonal projection: ||P d|| <= ||d||, P = U U^T, U^T U = I.
(2) For fixed 0<=w<=1, ||w*(d+P d)/2|| <= ||d||.
(3) Trust correction by construction: ||d_route-d0|| <= rho*||d0||.
(4) If raw progress r>0 and replay is deterministic, alpha=0 is feasible
    for the F0 guard ell_final <= ell_before - kappa*r + tolerance, kappa<=1.
~~~

性质(2)是原始几何路由的范数界，加入反馈后的总位移不自动同样收缩；最终共同预算只约束与d0的偏离。性质(4)不保证r>0、不保证累计F0、不保证retain。

对关联查询LP若梯度z局部L光滑：

~~~text
ell(theta+d) - ell(theta) <= z^T d + (L/2)||d||^2
z^T d = (Pz)^T d + ((I-P)z)^T d
       <= -gamma + delta*||(I-P)d||   if (Pz)^T d <= -gamma and ||(I-P)z||<=delta
~~~

这个条件分析说明需要测量对齐符号、投影外项和曲率，不是从组件cosine直接推出gamma>0，更不是所有多跳恢复消失的证明。框架有数值性质不代表代理识别了真实记忆连接。

### 17.3 论文主张的证据门

| 拟写主张 | 当前可采用的表述 | 禁止跳跃 |
|---|---|---|
| 关联指导可执行 | 有实现、日志和多模型训练结果 | 已证明定位真实电路 |
| 多跳残留减少 | 指定模型/配方下相对Original数值下降 | 所有恢复途径均消失 |
| 优于GA | 报告同seed差和能力代价 | 小均值差即显著或整体支配 |
| 优于文献方法 | 逐配方报告、充分调参和测试后再判断 | 单个弱配置、smoke失败当算法失败 |
| 帕累托优势 | 经预定义坐标/预算及配对统计的有限样本结论 | 全球最优、跨模型共同前沿 |
| 选择性代价 | 完成预冻结关联分层后判断 | 所有retain下降都是合理代价 |

### 17.4 WWW稿件如何使用本报告

本报告是技术附件/证据底稿，不是把全部内容塞进正文。正文围绕权限定义→更新指导→匹配baseline→机制对照→独立确认与代价；核心表只选同协议结果，历史M/A/B恢复细节进入附录。

可先写问题、方法、局部性质和评测合同；摘要优势句、最终方法选择和显著性结论等exp017及独立确认后冻结。论文相关工作中要区分已有图式恢复现象和本项目“转成更新指导”的待验证区别。本文没有为WWW接收率或会议信息作外部调研，不据此承诺投稿结果。

<a id="s18"></a>
## 18. 复现操作、检查表和项目职责

### 18.1 只读检查顺序

~~~text
1. 读取实验卡，确认目标、主指标、seeds和预算。
2. 读取run/config、source/hash、数据/模型manifest，确认不是只看当前共享源码。
3. 检查进程/最近日志/status是否一致；running字段可能陈旧。
4. 只汇总completed且正式非smoke端点，独立记录failed和skipped。
5. 由逐题记录重算回忆，按总NLL和token数复算PPL，核对MMLU subject/sample数。
6. 同model/method/seed/track联结，不把重复端点当独立seed。
7. 同时报告原始值、相对Original变化、成本及信息访问。
8. 更新报告快照并保存采集时间与文件SHA256；不改旧run。
~~~

### 18.2 本地验证命令示例

以下仅为复现说明，本次编写报告没有执行训练、安装包或重启云端队列。

~~~powershell
# 项目指定本地Python；不要用未确认的系统python替换环境
& D:/Anaconda/envs/cuda126_env/python.exe -B -m pytest exp/train/tests/test_cloud_gsu_c2.py -q
& D:/Anaconda/envs/cuda126_env/python.exe -B -m pytest exp/train/tests/test_complete_reproduction.py -q
& D:/Anaconda/envs/cuda126_env/python.exe -B -m pytest exp/train/tests/test_8b_multiseed.py -q

# 新独立baseline来源验证（不代表正式实验性能通过）
& D:/Anaconda/envs/cuda126_env/python.exe -B -c "from baselines.bootstrap import verify; print(len(verify()))"

# 新独立轨先看计划/资源门，正式运行另用唯一run目录
& D:/Anaconda/envs/cuda126_env/python.exe -B -m baselines.train --method RMU --plan
~~~

云端使用冻结venv及快照内命令；学术加速和HF_HUB_DISABLE_XET等环境变量须记录。密码不写报告、脚本、配置或命令历史；本报告不收录SSH凭据。

### 18.3 工程验收清单

| 环节 | 必须留下的证据 |
|---|---|
| 原始数据 | 许可、revision/commit、完整文件哈希、split顺序、去重/重合审计 |
| 模型 | 仓库ID、版本/文件哈希、tokenizer、dtype、是否量化、加载器 |
| 方法 | 损失符号/归约、alpha/beta、目标模块、原生/LoRA、优化器、warmup、裁剪 |
| 随机性 | 初始化、forget顺序、retain/alternate/随机组件/随机方向各流的seed |
| 停机 | 固定步、UA阈值、开发选择和超时规则；测试不调参 |
| 日志 | 每步原始/最终位移、optimizer step count、锚点前后LP、guard alpha、跳过样本 |
| 评测 | 原问、改写、全/覆盖/非覆盖路径、关联分层、独立保持、能力、生成行为 |
| 原生掩码 | 未选权重与动量不移动，完整保存/恢复及实际logit对照 |
| 结果 | 每seed、n、均值/样本SD、原始输出、采集哈希、失败与缺失 |
| 资源 | 底模加载、定位、训练、反馈、评测各时间/显存，CPU内存/磁盘限制 |

### 18.4 文件职责与编号

- docs/下四份精简正文用于日常阅读；本报告是用户要求的单文件大技术快照，允许集中重复展示表格，但不成为第二套实时手工成绩库。
- exp/runs的card/run_log/配置/逐题输出/源码快照不可重写；exp/sync保留采集与清理回执。
- baselines/vendor及source_lock锁定来源，不为“去重”改文件。
- exp016_8b_multiseed和exp016_local_baselines历史重号，引用必须用完整目录名+run_id。
- 本报告的38/42完成数、磁盘和等待状态都是带时间的观察，不自动刷新；更新应重采证据再生成。

<a id="s19"></a>
## 19. 历史报告全部表格汇编

本节逐表摘录七份历史结果报告中的Markdown表格，供离线单文件阅读。保留当时表头、模型、分母及口径；它们不是本报告新增实测，也不与8B当前结果混排。原表若有“最强/成功”等历史标注，不自动升级为本报告结论，前文证据审计优先。个别原表源于日志汇总而非完整逐题JSON，不能补造统计。

表格按来源报告和附近标题标记，原始run引用在源文件；完整源文件哈希见第20节。嵌套卡片、文献默认参数、相对LP标尺和历史MCQA不得直接拼成同一总分。


### 19.1 阶段1预实验报告

来源：[exp/results/阶段1预实验报告.md](../exp/results/阶段1预实验报告.md)；1张原表。

#### 表19.1.1 结果

| 指标 | 数值 | 门槛 | 判定 |
|---|---|---|---|
| 单跳一致率均值 | 0.669 | ≥ 0.30 | 通过 大幅通过 |
| τ 扫描掌握率（0.5–0.9） | 0.67 → 0.63 | — | 通过 平坦，结论对 τ 不敏感 |
| hop2 整体正确率 | 0.675 | — | 通过 |
| **latent multi-hop（核心）** | **0.703 ± 0.056**（n=256 子项，87 簇） | ≫ 0.333 地板 | 通过 **现象存在** |


### 19.2 研究内容一实验结果报告

来源：[exp/results/研究内容一实验结果报告.md](../exp/results/研究内容一实验结果报告.md)；6张原表。

#### 表19.2.1 研究内容一 · 实验完整结果报告

| 项 | 值 |
|---|---|
| 日期 | 2026-10-17 |
| 模型 | **Qwen2.5-1.5B-Instruct**（本地模型目录由 `QWEN_1_5B_PATH` 配置，bfloat16） |
| 环境 | cuda126_env（PyTorch 2.6.0+cu126，transformers 5.14.1），RTX 4060 Ti 16GB |
| 数据 | FaithUn（200 实体 / 664 簇；split：forget 33 / retaining 66 / test 465），sha256 校验通过 |
| 训练 | 自实现 GA-LoRA（r=8, α=16, lr=5e-5, 3 epochs, KL 保持 α=10, bf16） |
| 代码 | `exp/probe/`（阶段 1–4）、`exp/train/`（阶段 5–6），全部可复现 |

#### 表19.2.2 二、流水线各阶段结果

| 阶段 | 产出 | 数值 |
|---|---|---|
| 1 预实验（go/no-go） | latent multi-hop | **0.703 ± 0.056**（n=256 子项；地板 1/3）→ **GO** |
| 2 探测过滤 | 外部图 → 内生图 | 61 候选 → **41 掌握（67%）**，过滤 33% |
| 3 活性核验 | 活跃路径 | **56 / 89**（63%） |
| 4 贪心选集（B=5） | **F₁ = F₀ + 5 个扩展事实** | 覆盖活跃路径 25/56；全 cover 需 31 个 |
| 5 训练 | 5 个 LoRA adapter | `exp/adapters/1p5B_v2_{G1..G5}/` |
| 6 评测 | 三口径指标 | 见第三节 |

#### 表19.2.3 二、流水线各阶段结果

| # | 事实 | c | 覆盖活跃路径 |
|---|---|---|---|
| 1 | (USA, capital, Washington, D.C.) | 1.00 | 7 |
| 2 | (USA, continent, North America) | 1.00 | 7 |
| 3 | (USA, currency, United States dollar) | 1.00 | 7 |
| 4 | (Los Angeles, country, USA) | 1.00 | 2 |
| 5 | (Los Angeles, continent, North America) | 1.00 | 2 |

#### 表19.2.4 三、五组对比（核心结果）

| 组 | 遗忘集（规模） | hop1 UA ↓ | hop1 改写 ↓ | **hop2 恢复率 ↓** | same_object ↑ | retaining ↑ |
|---|---|---|---|---|---|---|
| orig | — | 0.64 | 0.65 | 0.64 | 0.76 | 0.76 |
| G1 | 只忘 F₀（33） | 0.18 | 0.20 | 0.36 | 0.32 | 0.39 |
| **G2** | **F₀ + 贪心 F\*（38）** | **0.18** | 0.19 | **0.25** | 0.26 | 0.39 |
| G3 | F₀ + 全候选（94） | 0.36 | 0.29 | 0.22 | 0.30 | 0.61 |
| G4 | F₀ + 随机 5（38） | 0.18 | 0.20 | 0.31 | 0.31 | 0.36 |
| G5 | 只忘 F\*（5，无 F₀） | 0.61 | 0.64 | 0.56 | 0.75 | 0.70 |

#### 表19.2.5 四点五、内生知识图谱解读（图 12）

| 颜色 | 含义 | 数量 | 来源 |
|---|---|---|---|
| 红 | $F_0$ 待遗忘事实（名人的 hop1 事实） | 33 | 数据集 forget 集 |
| 绿 | 探测掌握但未被 F1 选中的候选支持事实（内生图保留） | 36 | 阶段 2 探测过滤（c ≥ τ）；另有 5 条以蓝色 F1 显示 |
| 灰虚线 | 被过滤事实（外部图有、模型未掌握） | 20 | 阶段 2 探测过滤 |
| 蓝粗 | $F_1$ 贪心选中的扩展事实 | 5 | 阶段 4（B=5） |

#### 表19.2.6 六、产物清单

| 类型 | 路径 |
|---|---|
| 真实图 4 张 | `研究内容一-示意图/fig8_real_pilot.png`、`fig9_real_groups.png`、`fig10_real_efficiency.png`、`fig11_real_greedy.png`（均已标注模型） |
| 结果数据 | `exp/results/`：eval_results_1p5B{,_v2}.json、F1.json、candidates.json、active_paths.json、greedy_curve.json、pre_experiment_*.{json,jsonl} |
| 代码 | `exp/probe/pre_experiment.py`、`build_f1.py`、`plot_real.py`；`exp/train/train_eval.py`、`plot_real_results.py`（支持断点续跑）；`exp/viz/kg_viz.py`（图谱可视化模块，含 CLI 与模块 API） |
| 模型权重 | `exp/adapters/1p5B_v2_{G1..G5}/`（LoRA adapter，可直接加载复评） |


### 19.3 3B_extension_report

来源：[exp/results/3B_extension_report.md](../exp/results/3B_extension_report.md)；1张原表。

#### 表19.3.1 五组结果（采用 clean run）

| 组 | hop1_unlearn ↓ | hop1_test ↓ | hop2 ↓ | same_object ↑ | retain ↑ |
|---|---:|---:|---:|---:|---:|
| orig | 0.697 | 0.707 | 0.633 | 0.854 | 0.818 |
| G1 只忘 F0 | 0.121 | 0.141 | 0.255 | 0.229 | 0.364 |
| G2 F0 + 3B 贪心 F1 | 0.152 | 0.131 | **0.236** | 0.188 | 0.333 |
| G3 F0 + 全候选 | 0.212 | 0.242 | **0.157** | 0.222 | 0.545 |
| G4 F0 + 随机 5 | 0.182 | 0.182 | **0.236** | 0.215 | 0.364 |
| G5 只忘 F1 | 0.667 | 0.616 | 0.461 | 0.632 | 0.636 |


### 19.4 研究内容二实验报告

来源：[exp/results/研究内容二实验报告.md](../exp/results/研究内容二实验报告.md)；8张原表。

#### 表19.4.1 2. 实验设置

| 项    | 配置                                                                                             |                     |         |
| ---- | ---------------------------------------------------------------------------------------------- | ------------------- | ------- |
| 双模型对 | M（Qwen2.5-1.5B-Instruct 冻结）vs M′（GA-LoRA adapter：G1 只忘 $F_0$ / G2 $F_0$+$F_1$ 贪心5），G1/G2 分开跑不混 |                     |         |
| 评测口径 | 与内容一冻结协议一致：MCQA 三选一、长度归一化续写对数概率、无 chat template                                                |                     |         |
| 损伤定义 | \$d(f) = \log p\_M(a                                                                           | q) - \log p\_{M'}(a | q)\$，逐题 |
| 分组   | D 受损组 = same\_object/hop2 按 d\_G2 各 top-24；K 对照组 = 同来源 d\_G2 最低 24+24（源内匹配，排除来源混淆）             |                     |         |

#### 表19.4.2 4. 阶段 1：损伤打分（图 D1）

| 来源           | n   | mean d（G1） | mean d（G2） | 解读                   |
| ------------ | --- | ---------- | ---------- | -------------------- |
| $F_0$ hop1   | 33  | 5.31       | 6.32       | 遗忘生效                 |
| hop1 测试      | 99  | 4.54       | 5.57       | 遗忘生效                 |
| hop2         | 267 | 2.61       | 3.73       | 扩展集封堵对象              |
| same\_object | 144 | 3.75       | 4.87       | **共享答案 token 的连带损伤** |
| retain       | 33  | 3.38       | 4.28       | **KL 锚未守住的保护集**      |

#### 表19.4.3 7. 锚强度对照组 G6：分布级 KL 锚存在结构性盲区（否定性结果，价值重大）

| 组      | hop1\_unlearn | hop1\_test | hop2      | same\_object | retain    |
| ------ | ------------- | ---------- | --------- | ------------ | --------- |
| orig   | 0.636         | 0.646      | 0.644     | 0.757        | 0.758     |
| G1     | 0.182         | 0.202      | 0.363     | 0.319        | 0.394     |
| G2     | 0.182         | 0.192      | 0.255     | 0.257        | 0.394     |
| **G6** | **0.121**     | **0.121**  | **0.255** | **0.160**    | **0.333** |

#### 表19.4.4 8.1 核心数字（恢复率 = 占 orig−G2 可恢复空间的百分比）

| 配置              | D 恢复%    | $F_0$ 回弹% | $F_1$ 复活% | K 漂移% |
| --------------- | -------- | --------- | --------- | ----- |
| 层 27 MLP        | 9.6      | 8.3       | 7.7       | 5.2   |
| **层 22–27 MLP** | **34.4** | 30.3      | 31.9      | 22.1  |
| 随机 6 层 MLP      | 13.8     | 12.4      | 11.3      | 16.2  |
| top-12 神经元      | 3.8      | 3.1       | 2.7       | 4.5   |
| 随机 12 神经元       | 3.6      | 3.0       | 2.5       | 4.3   |

#### 表19.4.5 8.3 模块分解实验：核心原因的直接定位（受控实验）

| 撤销配置            | D 恢复%    | $F_0$ 回弹% | 结论                             |
| --------------- | -------- | --------- | ------------------------------ |
| 层 22–27 MLP     | 34.3     | 30.3      | 原方案                            |
| 层 22–27 注意力     | 9.0      | 7.8       | 注意力承载少                         |
| 层 22–27 两者      | 43.6     | 39.4      | 叠加近似可加                         |
| **全部 28 层注意力**  | 17.4     | 16.4      | 独立撤销收益占可恢复空间约 17%，**无层特异性**（6 层≈全层一半） |
| **全部 28 层 MLP** | **88.7** | 87.2      | 独立撤销收益占可恢复空间约 89%            |
| 全部两者（=orig）     | 100.0    | 100.0     | 校验通过                           |

#### 表19.4.6 8.4 共享 token 机制分析（解释 same\_object 全面崩溃与低层分离）

| 组               | 答案恰为遗忘答案    | 说明                                              |
| --------------- | ----------- | ----------------------------------------------- |
| D\_same\_object | 24/24（100%） | same\_object 答案与 $F_0$ **完全共享**（题目即"同一对象的其他问法"） |
| K\_same\_object | 24/24（100%） | 共享但损伤轻 → 崩溃与否取决于压得多狠，不是是否共享                     |
| D\_hop2         | 21/24（88%）  | 被压的 hop2 子项答案就是 $F_0$/$F_1$ 答案本身                |
| K\_hop2         | 0/24（0%）    | 未受压子项与遗忘集无共享                                    |

#### 表19.4.7 10. 阶段 4：可检验预测——内容二的存在证明（图 D4）

| 显影得分来源               | Spearman ρ | p        | 判定   |
| -------------------- | ---------- | -------- | ---- |
| G2 自身 FFN 漂移         | 0.824      | 7.0×10⁻⁸ | 通过    |
| **G1 FFN 漂移（跨遗忘模型对留出）** | **0.830**  | 4.8×10⁻⁸ | 通过 留出 |
| G2 隐状态漂移             | 0.842      | 2.0×10⁻⁸ | 通过    |
| 随机基线（\|ρ\| 均值）       | 0.152      | —        | 对照   |

#### 表19.4.8 11. 总对照：从规划到收官，猜想 vs 实测

| #  | 猜想/预期（出处）                          | 实测结果                        | 判定                 |
| -- | ---------------------------------- | --------------------------- | ------------------ |
| 1  | 扩展遗忘集封堵多跳（内容一 H2）                  | G2 0.25 < G4 0.31 < G1 0.36 | 通过 成立               |
| 2  | 受损事实在中高层漂移更大（规划 §3）                | 分离带在低层（G2 层1–4），高层为背景       | 未通过 修正：低层指纹          |
| 3  | 头归因产出 top-k 头（规划 §3）               | 跨头均匀 null                   | 未通过 否定（有划界价值）        |
| 4  | FFN top-k 神经元可作恢复目标（规划 §3）         | 神经元手术无效，层内分布式               | 未通过 修正：层/模块级         |
| 5  | Fisher 分与 d(f) 对齐 ρ≥0.5（规划 §2 验收）  | D/K 梯度剖面无差异                 | 未通过 否定（Fisher 路线不适用） |
| 6  | KL 锚失效是因为 α 太小（隐含假设）               | G6 强 12× 更糟；机制：覆盖域盲区        | 未通过 否定（结构性）          |
| 7  | 损伤集中在少数层（隐含假设）                     | MLP 承载 89% 但层间近均匀           | 未通过 修正：模块集中、层稀疏假设证伪  |
| 8  | 损伤主要在 MLP（内容二故事线猜想）                | 模块分解：MLP 88.7% vs 注意力 17.4% | 通过 成立               |
| 9  | same\_object 崩溃源于共享答案 token（§6 猜想） | 100% 答案共享 + hop2 受损组 88%    | 通过 成立               |
| 10 | 层定位有因果效度（阶段 2 工作假设）                | 定向 2.5× 随机；单层扫描 ρ=0.82      | 通过 成立               |
| 11 | **显影得分可预测留出撤销收益（成功判据）**            | ρ=0.83，p≈1e-8，跨遗忘模型对留出         | 通过 **成立（收官）**       |


### 19.5 研究内容二阶段2归因结果

来源：[exp/results/研究内容二阶段2归因结果.md](../exp/results/研究内容二阶段2归因结果.md)；1张原表。

#### 表19.5.1 5. 对规划的三处修正（已按实测更新）

| 规划预期 | 实测 | 影响 |
|---|---|---|
| 受损事实在中高层漂移更大 | 分离带在低中层（G2: 1–4 层；G1: 5–13 层），高层 K 反超 | 图 D2 故事改为"低层指纹、高层背景" |
| 头归因产出 top-k 头 | 跨头均匀，null | 阶段 3 候选组件中删除头，只保留 FFN 神经元与（待测）层撤销 |
| 梯度与 Fisher 交叉验证 | D/K 梯度剖面无差异 | 阶段 1.2 改为"验证 d(f) 与 Fisher 无关/弱相关"这一否定性对齐 |


### 19.6 研究内容三实验结果报告

来源：[exp/results/研究内容三实验结果报告.md](../exp/results/研究内容三实验结果报告.md)；8张原表。

#### 表19.6.1 0. 结论速览

| 判定项 | 目标 | 实测最优 | 是否达成 |
|---|---|---|---|
| 受损集恢复率 | ≥ 30% | 82.0%（R1 无约束） | 通过 大幅超出 |
| F0 回弹率 | < 10% | 16.3%（R5_surgery，§11） | 未通过 未达成（旧方案最优 44.5%） |
| F1 复活率 | ≤ 5% | 15.3%（R5_full）/ 21.3%（R5_surgery） | 未通过 未达成 |
| K 集漂移 | ≤ 5% | **4.8%（R5_surgery，lp 口径）** | 通过 首次达成（§11） |

#### 表19.6.2 1. R0：恢复基础设施

| 产物 | 规模 | 说明 |
|---|---|---|
| `supervision.json` | 57 题（hop2 49 + retain 8） | 恢复训练监督集，来自内容二受损探测 |
| `anchors.json` | 142 条 | F0 锚点（hop1_unlearn/hop1_test），约束回弹用 |
| `energy_space.npz` | 每层 gate/down，η ∈ {80, 90, 95}% | 能量空间投影矩阵（η = 锚点能量覆盖分位） |
| `layer_lr.json` | 层 22–27 权重 2.1–4.4 | 内容二显影得分转层学习率（选址器用法） |

#### 表19.6.3 2. 总表（13 配置 × 5 指标，相对标尺 %）

| 配置 | 恢复% | hop1回弹% | F1复活% | K漂移% | retain恢复% |
|---|---|---|---|---|---|
| **R1_rec**（无约束） | **82.0** | 75.5 | 74.0 | 80.2 | 88.9 |
| B_kl（KL 锚，旧范式对照） | 66.4 | 56.6 | 64.4 | 79.4 | 65.8 |
| R3 λ=0.1 η=80/90/95 | 79.8 / 79.8 / 79.9 | 73.1 / 73.2 / 73.4 | 72.3 / 72.2 / 72.8 | 78.0 / 77.6 / 77.7 | 86.5 / 86.7 / 86.8 |
| R3 λ=1 η=80/90/95 | 77.0 / 74.5 / 73.7 | 65.8 / 64.4 / 64.9 | 70.1 / 67.8 / 66.9 | 75.0 / 73.9 / 73.5 | 82.7 / 80.8 / 80.3 |
| R3 λ=10 η=80/90/95 | 69.7 / **62.0** / 64.7 | **46.2** / **44.5** / 50.0 | 66.2 / **57.2** / 62.8 | 71.7 / 71.0 / 69.3 | 66.5 / 64.2 / 66.7 |
| **R2_ans**（β=10 hinge） | −33.5 | −55.6 | −41.1 | −136.0 | −66.4 |
| **R4_combo**（hinge+λ=1） | −16.1 | −29.4 | −18.3 | −72.1 | −39.7 |

#### 表19.6.4 7. 旧范式对照：能量投影优于 KL 锚

| | 恢复% | 回弹% | 判读 |
|---|---|---|---|
| B_kl（rec + KL 锚） | 66.4 | 56.6 | 旧范式最好水平 |
| R3 λ=10 η=80 | 69.7 | 46.2 | **两轴同时占优** |

#### 表19.6.5 8. 图索引

| 图 | 文件 | 内容 |
|---|---|---|
| 图 R1 | `研究内容一-示意图/fig20_r1_scatter.png` | 恢复-回弹全景散点（含内容二组件撤销参照点 34.4/30.3 ★） |
| 图 R2 | `研究内容一-示意图/fig21_r2_tradeoff.png` | λ-恢复-回弹三方关系（9 点网格 + R1 无约束色带） |
| 图 R3 | `研究内容一-示意图/fig22_r3_bars.png` | 六评测台 acc + lp 柱状对比（orig/G2 基线补齐） |

#### 表19.6.6 证据 B：同一更新的机械证明（GPU 实测，R1 adapter，层 22–27）

| 层 | cos(D激活, F0激活) | cos(ΔWx_D, ΔWx_F0) | F0/D 更新幅度比 |
|---|---|---|---|
| 22–27 | +0.86 ~ +0.94 | **+0.996 ~ +1.000** | 1.11–1.31 |

#### 表19.6.7 证据 C：约束本身是"盲"的（为什么 λ 只能当阻尼）

| 配置 | η=80 | η=90 | η=95 |
|---|---|---|---|
| R1_rec（无约束） | 0.025 | 0.049 | 0.080 |
| R3 λ=10 | 0.003 | 0.008 | 0.041 |

#### 表19.6.8 结果

| 配置 | 恢复% | 回弹% | F1复活% | K漂移% | 回弹/恢复比 |
|---|---|---|---|---|---|
| R5_surgery（纯手术） | **31.2** | **16.3** | 21.3 | **4.8** | **0.52** |
| R5_full（手术+软化hinge） | 11.8 | **−6.7**（过冲） | 15.3 | 4.1 | — |
| 参照：R3 λ=10（阻尼最优） | 62.0 | 44.5 | 57.2 | 71.0 | 0.72 |
| 参照：组件撤销（内容二 ★） | 34.4 | 30.3 | 32.0 | — | 0.88 |


### 19.7 新遗忘方法GSU对比报告

来源：[exp/results/新遗忘方法GSU对比报告.md](../exp/results/新遗忘方法GSU对比报告.md)；14张原表。

#### 表19.7.1 1. 方法设计：每个组件都来自前三个研究内容的证据

| 组件 | 设计 | 证据来源 |
|---|---|---|
| 遗忘集 | F0 + F1 贪心5（G2 同款） | 内容一：拓展遗忘集是多跳封堵的关键（hop2 0.36→0.25），5 个事实达 FaithUn 全图扩展（G3）约 90% 效果 |
| 覆盖方式 | 全层全模块 LoRA（G-recipe 同款） | **M1 实验的教训**：只打层 22–27 MLP 时遗忘不彻底（见 §4 负面结果）——损伤定位 ≠ 遗忘可用定位 |
| 保护方式 | 每步在各可训练参数张量上从遗忘梯度减去保持集锚点梯度方向的投影（4 锚点/步） | 内容三 R5 的镜像（恢复侧手术 → 遗忘侧手术）；G6 已证 KL 锚加强 12× 反而更差（retain 0.333<0.394，same_object 0.160<0.257）——分布级阻尼有结构性盲区，必须做梯度级手术 |

#### 表19.7.2 2. 五集对比（内容一口径）

| 组 | hop1_unlearn(F0) | hop1_test | hop2(多跳) | same_object | retain |
|---|---|---|---|---|---|
| orig | 0.636 | 0.646 | 0.644 | 0.757 | 0.758 |
| G1 只忘F0 | 0.182 | 0.202 | 0.363 | 0.319 | 0.394 |
| G2 拓展集+KL | 0.182 | 0.192 | 0.255 | 0.257 | 0.394 |
| G6 拓展集+强KL | 0.121 | 0.121 | 0.255 | 0.160 | 0.333 |
| **M3 GSU（本文）** | **0.152** | 0.242 | **0.270** | 0.201 | **0.485** |
| M2 全层+半手术 | 0.333 | 0.394 | 0.311 | 0.292 | 0.424 |
| M1 定位+手术 | 0.424 | 0.465 | 0.596 | 0.667 | 0.727 |

#### 表19.7.3 3. 六集对比（内容三口径，D=多跳受损集，K=无关对照集）

| 组 | F0 | D_so | D_hop2 | K | retain | F1_hop2 |
|---|---|---|---|---|---|---|
| orig | 0.636 | 0.542 | 0.833 | 0.583 | 0.758 | 0.907 |
| G1 | 0.182 | 0.083 | 0.250 | 0.562 | 0.394 | 0.413 |
| G2 | 0.182 | 0.083 | 0.083 | 0.562 | 0.394 | 0.253 |
| G6 | 0.121 | 0.000 | 0.000 | 0.479 | 0.333 | 0.267 |
| **M3 GSU** | **0.152** | 0.083 | **0.000** | **0.604** | **0.485** | 0.333 |
| M1 | 0.424 | 0.417 | 0.667 | 0.583 | 0.727 | 0.760 |

#### 表19.7.4 5. M3 更新方向补强（预注册单 seed 筛选已完成）

| 变体 | 更新规则 | 要回答的问题 |
|---|---|---|
| M3C | 沿用 M3 的逐参数投影，只在该参数张量的遗忘梯度会降低锚点目标时投影 | 无冲突步是否因无条件投影而损失了遗忘效率？ |
| M3U | 先执行 AdamW，再计算实际参数位移与采样锚点聚合梯度的内积；只有一阶预测会降低锚点目标时才投影该次位移 | AdamW 的实际更新是否偏离梯度层保护方向？ |

#### 表19.7.5 冻结主指标与关键次指标

| 变体 | `D_hop2_no_F1_overlap`（12） | `D_hop2_F1_overlap`（12） | F1_hop2（75） | retain（33） | K（48） | same_object（24） |
|---|---:|---:|---:|---:|---:|---:|
| M3 复评 | 0/12 | 0/12 | 25/75 | 16/33 | 29/48 | 2/24 |
| M3C | 0/12（+0） | 0/12 | 24/75 | 17/33（+1） | 29/48 | 2/24 |
| M3U | 1/12（+1） | 0/12 | 9/75 | 13/33（−3） | 28/48 | 0/24 |

#### 表19.7.6 M3U 逐步诊断与 M3A 验证

| 变体 | `D_hop2_no_F1_overlap`（12） | `D_hop2_F1_overlap`（12） | F1_hop2（75） | retain（33） | K（48） | same_object（24） |
|---|---:|---:|---:|---:|---:|---:|
| M3 | 0/12 | 0/12 | 25/75 | 16/33 | 29/48 | 2/24 |
| M3U | 1/12 | 0/12 | 9/75 | 13/33 | 28/48 | 0/24 |
| M3A | 3/12 | 2/12 | 9/75 | 11/33 | 21/48 | 6/24 |

#### 表19.7.7 6. Retain 锚点覆盖与累计保护诊断

| 变体 | 锚点策略 | `D_hop2_no_F1_overlap`（12） | `D_hop2_F1_overlap`（12） | F1_hop2（75） | retain（33） | K（48） | 实际残差 sanity |
|---|---|---:|---:|---:|---:|---:|---|
| M3A | 4 条/步，独立采样 | 3/12 | 2/12 | 9/75 | 11/33 | 21/48 | 通过 `-1e-5` 现场门，但未通过原始 `-1e-6` 诊断门 |
| COV16 | 16 条/步，新抽样；全程覆盖 33/33 | 9/12 | 5/12 | 29/75 | 20/33 | 11/48 | 未通过：9/1824 低于 `-1e-5` |
| CUM16 | 固定 16/16，线性累计下限 | 9/12 | 6/12 | 41/75 | 23/33 | 22/48 | 未通过：4/1824 低于 `-1e-5` |

#### 表19.7.8 实际 logprob 反馈补强：FB16

| 变体 | 主指标 `D_hop2_no_F1_overlap`（12） | F1_hop2（75） | retain（33） | K（48） | 反馈后直接 logprob 下降 | 反馈后均值 |
|---|---:|---:|---:|---:|---:|---:|
| COV16 | 9/12 | 29/75 | 20/33 | 11/48 | 866/1824 | +0.026811 |
| FB16 | 3/12 | 22/75 | 19/33 | 9/48 | 838/1824 | -0.053329 |

#### 表19.7.9 五集总表（全部方法）

| 方法 | F0 | hop2 | same_object | retain | 判定 |
|---|---|---|---|---|---|
| orig | 0.636 | 0.644 | 0.757 | 0.758 | — |
| G1 只忘F0+KL | 0.182 | 0.363 | 0.319 | 0.394 | 多跳漏封 |
| G2 拓展+KL | 0.182 | 0.255 | 0.257 | 0.394 | GSU 在 F0/retain 上改善，hop2/same_object 存在交换 |
| G6 拓展+强KL | 0.121 | 0.255 | 0.160 | 0.333 | KL 盲区 |
| **M3 GSU（历史点估计）** | **0.152** | **0.270** | 0.201 | **0.485** | D_hop2 含 12/24 个 F1 重叠项 |
| GA（纯，无保护） | 0.394 | 0.300 | 0.236 | 0.333 | 无保护崩坏 |
| CEU | 0.242 | 0.262 | 0.382 | 0.394 | 强对手 |
| PDU | 0.242 | **0.225** | **0.271** | 0.424 | 最强 baseline |
| GradDiff | 0.364 | 0.640 | 0.632 | 0.939 | 保护-遗忘对冲 |
| NPO | 0.424 | 0.592 | 0.653 | 0.848 | 同上 |
| DPO | 0.697 | 0.719 | 0.806 | 0.909 | 弱遗忘 |
| SimNPO* | 0.818 | 0.768 | 0.819 | 0.939 | 未有效遗忘 |
| RMU* | 0.818 | 0.760 | 0.819 | 0.939 | 未有效遗忘 |
| WGA* | 0.758 | 0.775 | 0.833 | 0.939 | 未有效遗忘 |
| SatImp* | 0.848 | 0.764 | 0.826 | 0.939 | 未有效遗忘 |
| UNDIAL* | 0.818 | 0.742 | 0.847 | 0.939 | 未有效遗忘 |

#### 表19.7.10 六集（关键列）

| 方法 | D_so | D_hop2 | K | retain | F1_hop2 |
|---|---|---|---|---|---|
| orig | 0.542 | 0.833 | 0.583 | 0.758 | 0.907 |
| G2 | 0.083 | 0.083 | 0.562 | 0.394 | 0.253 |
| **M3 GSU** | 0.083 | **0.000** | **0.604** | **0.485** | 0.333 |
| GA(纯) | 0.125 | 0.125 | 0.500 | 0.333 | 0.040 |
| CEU | 0.167 | 0.167 | 0.542 | 0.394 | 0.347 |
| PDU | 0.083 | 0.083 | 0.542 | 0.424 | **0.000** |
| GradDiff | 0.500 | 0.750 | 0.542 | 0.939 | 0.760 |
| NPO | 0.417 | 0.708 | 0.542 | 0.848 | 0.760 |

#### 表19.7.11 8. 产物清单

| 产物 | 路径 |
|---|---|
| GSU 方法代码 | `exp/train/train_gsu.py`（M1/M2/M3 历史变体；新增 M3C/M3U，epoch 断点续训，双口径评测） |
| 文献 baseline 移植 | `exp/train/train_baselines.py`（GradDiff/NPO/SimNPO/RMU，损失逐行核对自 open-unlearning） |
| 六集补评工具 | `exp/train/eval_sixset.py`（G1/G6 六集数据由它产出） |
| 对比图 | `研究内容一-示意图/fig23_gsu_compare.png` |
| adapter | `exp/adapters/1p5B_v2_GSUM1/GSUM2/GSUM3、…_BLGradDiff/BLNPO/BLSimNPO/BLRMU` |
| 结果 | `exp/results/eval_results_gsu.jsonl`（五集）、`gsu_sixset.jsonl`（六集） |

#### 表19.7.12 指标定义与方向（主表统一口径）

| 指标 | 含义 | 方向 | 如何解读 |
|---|---|---|---|
| F0直接正确率 | 原始遗忘事实上的选择题答对比例，F0是本次要求遗忘的直接事实集合 | 越低越好 | 越低表示目标事实越难被直接回忆；不能单独证明没有多跳恢复 |
| F0改写正确率 | 同一F0事实换成改写问法后的答对比例 | 越低越好 | 检查遗忘是否只针对原句，还是对不同问法也有效 |
| 覆盖多跳正确率 | 历史F1覆盖切片上的多跳问题答对比例，共75题 | 越低越好 | 检查已知关联支持事实能否继续恢复遗忘事实；只代表覆盖切片，不代表全部89条路径 |
| 全89路径宏平均回忆率 | 全部hop2路径先按路径求平均，再对路径求宏平均 | 越低越好 | 当前多跳回忆主指标；比单纯汇总题目准确率更不受路径长度和题目数量影响 |
| 非锚点保持率 | heldout retain/非锚点审计集上的正确率 | 越高越好 | 越高表示遗忘更新对未指定保持对象的影响较小；非锚点不等于完全无关联知识 |
| 训练保护池正确率 | 训练过程中用于保护的retain样本在评测时的正确率 | 越高越好 | 检查保护目标是否被保住，但可能受保护集过拟合影响 |
| F1直接正确率 | 扩展遗忘事实或关联支持事实上的直接正确率 | 越低越好 | 用于判断扩展监督是否真的被遗忘；不能和F0-only方法混成同一监督预算结论 |
| answer logprob | 模型对目标答案完整序列的平均对数概率 | 遗忘集越低越好 | 越负表示模型给目标答案的概率越低；必须在同一模型、同一tokenizer和同一题集内比较 |
| MCQA margin | 正确选项分数减去最强错误选项分数 | 遗忘集越低越好；保持集越高越好 | 小于0通常表示模型更偏向错误选项；与answer logprob不是同一个量 |
| Delta answer logprob | 训练后相对Original的目标答案对数概率变化 | 遗忘集越负越好；保持集越接近0越好 | 反映更新造成的概率位移，不能直接等同于答题准确率变化 |
| MMLU-10/科目准确率 | 57个科目各抽10题、5-shot的通用知识子集准确率 | 越高越好 | 是通用能力审计，不是完整MMLU，也不等于全部推理能力 |
| MMLU Delta百分点 | 相对同模型Original的MMLU准确率变化 | 越高越好 | 0表示无变化，负值表示能力下降；只在同一模型内比较 |
| WikiText-2 token PPL | 固定WikiText-2测试文本上的token级困惑度 | 越低越好 | 越低表示下一个token预测更好；不同tokenizer或不同模型之间不比较绝对值 |
| PPL比值 | 方法PPL除以同模型Original PPL | 越低越好 | 1表示无变化，大于1表示困惑度上升；比绝对PPL更适合模型内比较 |
| 墙钟时间 | 从启动到完成的实际训练或评测耗时 | 越低越好 | 代表计算成本，不代表遗忘质量 |
| 峰值显存 | 运行过程中的最大GPU显存占用 | 越低越好 | 代表资源需求，不代表模型效果 |
| 可训练参数量 | 实际更新的参数数量 | 越低通常越省资源 | 只能作为成本/参数效率信息，不能单独作为方法优劣依据 |

#### 表19.7.13 2026-10-23：本地 C 组已完成结果（部分批次）

| 方法 | 全89路径宏回忆 ↓ | 覆盖多跳 /75 ↓ | F0 /33 ↓ | F1 /5 | 留出retain /33 ↑ | 护栏回退步数 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 44.94% | 52 | 8 | 5 | 15 | 0 |
| Cg | 32.58% | 9 | 12 | 1 | 10 | 98 |
| C1 | 27.72% | 0 | 12 | 0 | 12 | 103 |
| C2 | 29.96% | 7 | 12 | 0 | 10 | 109 |
| C3 | 30.71% | 7 | 13 | 0 | 10 | 108 |

#### 表19.7.14 2026-10-23 最终完成：本地 C 组八变体筛选

| 排名 | 方法 | 全89路径宏回忆 ↓ | 覆盖多跳 /75 ↓ | F0残留 /33 ↓ | F1 /5 | 留出retain /33 ↑ | 护栏回退 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | C1 | 27.72% | 0 | 12 | 0 | 12 | 103 |
| 2 | C4 | 29.59% | 7 | 12 | 0 | 10 | 107 |
| 3并列 | C3R | 29.96% | 7 | 12 | 0 | 10 | 109 |
| 3并列 | C2 | 29.96% | 7 | 12 | 0 | 10 | 109 |
| 5 | C3 | 30.71% | 7 | 13 | 0 | 10 | 108 |
| 6 | C3D | 31.46% | 9 | 13 | 0 | 10 | 105 |
| 7 | Cg | 32.58% | 9 | 12 | 1 | 10 | 98 |
| 8 | C0 | 44.94% | 52 | 8 | 5 | 15 | 0 |


<a id="s20"></a>
## 20. 来源文件、哈希与快照说明

### 20.1 历史云端成本补表

下表来自16端点旧云端比较JSON，不是8B成本。墙钟包含其加载、定位/训练和既定评测，不能当作纯训练时间；RMU参数范围不同，不能只按秒数排名算法。8B逐端点成本本次未完整回收，不从旧值估算。

| 模型 | 方法 | 总墙钟s↓ | 峰值GiB↓ | 可训练参数↓ |
|---|---|---|---|---|
| Qwen3.5-4B | Original | 103.81 | 8.568 | 0 |
| Llama-3.2-3B-Instruct | Original | 38.03 | 6.027 | 0 |
| Qwen3.5-4B | GA | 195.90 | 9.185 | 16232448 |
| Llama-3.2-3B-Instruct | GA | 83.90 | 6.255 | 12156928 |
| Qwen3.5-4B | GradDiff | 222.40 | 9.583 | 16232448 |
| Llama-3.2-3B-Instruct | GradDiff | 96.89 | 6.333 | 12156928 |
| Qwen3.5-4B | NPO | 228.23 | 9.596 | 16232448 |
| Llama-3.2-3B-Instruct | NPO | 104.97 | 6.344 | 12156928 |
| Qwen3.5-4B | SimNPO | 213.21 | 9.596 | 16232448 |
| Llama-3.2-3B-Instruct | SimNPO | 96.29 | 6.336 | 12156928 |
| Qwen3.5-4B | GA [F0+F1] | 181.18 | 9.185 | 16232448 |
| Llama-3.2-3B-Instruct | GA [F0+F1] | 82.47 | 6.255 | 12156928 |
| Qwen3.5-4B | RMU | 129.08 | 8.852 | 282624 |
| Llama-3.2-3B-Instruct | RMU | 43.46 | 6.044 | 270336 |
| Llama-3.2-3B-Instruct | GSU-C2 | 1550.45 | 8.377 | 12156928 |
| Qwen3.5-4B | GSU-C2 | 3131.66 | 12.025 | 16232448 |

来源：[exp/analysis/cloud_comparison_20261022/report_v3_klue_pending/comparison.json](../exp/analysis/cloud_comparison_20261022/report_v3_klue_pending/comparison.json)。

### 20.2 证据文件清单

下列SHA256针对本报告编写时实际读取的本地文件。云端exp015/016汇总是本次回收副本；每个row另带远程endpoint、样本/PPL或metrics哈希。当前共享源码哈希仅认证本次读到的版本，不替代旧run冻结source哈希。

| 序号 | 来源文件 | 字节 | SHA256 |
|---|---|---|---|
| 1 | [exp/results/研究内容三实验结果报告.md](../exp/results/研究内容三实验结果报告.md) | 13688 | 32edc4ba2fc349d963268d3812383f05bbcd64a1f37960046b2153d1ba296835 |
| 2 | [exp/results/eval_results_1p5B_v2.json](../exp/results/eval_results_1p5B_v2.json) | 1134 | 8ec3413d039bf783721cd83a90539bd325efa62f4f14ab3f92cc7c2fb78b8775 |
| 3 | [exp/results/eval_results_3B_v2_clean.json](../exp/results/eval_results_3B_v2_clean.json) | 1153 | d3d313a6de25a888992dad56c3b0dd31c2caf84bf50ad613f518eee2e790bf4b |
| 4 | [exp/results/pre_experiment_summary.json](../exp/results/pre_experiment_summary.json) | 617 | fa13c19c6ae0ce8479c370c1077cc4e2379741fd43ac7a75473fee49a6c446eb |
| 5 | [exp/results/pre_experiment_summary_3B.json](../exp/results/pre_experiment_summary_3B.json) | 623 | 409b2eb7b8b38adc3787ed770d7c1f1a3c4b98f01b1bfac21f889daebcce6fb5 |
| 6 | [exp/results/eval_results_gsu.jsonl](../exp/results/eval_results_gsu.jsonl) | 2573 | 9e9f63150737d26d4af7e94a31f41ffedfc65ab112483489cefeadc5a10f4d50 |
| 7 | [exp/results/damage_summary.json](../exp/results/damage_summary.json) | 635 | bf583853a07d31fea1c47dfdb4fe2194a722cd4c62c3017a0b9100c71f185538 |
| 8 | [exp/results/restore_summary.json](../exp/results/restore_summary.json) | 2440 | 54722d68d306b6325586e686b95c49336b8dfebd94d19dfead9baed9fab4599a |
| 9 | [exp/runs/exp006_gsu_memory_guided/runs/run_batch_20261021_132023_431bf0_A0/metrics.json](../exp/runs/exp006_gsu_memory_guided/runs/run_batch_20261021_132023_431bf0_A0/metrics.json) | 395633 | d385001b75dde958bf627ff2eca2365384c2825795cd1893593a11f003d555a5 |
| 10 | [exp/runs/exp006_gsu_memory_guided/runs/run_batch_20261021_132023_431bf0_A1/metrics.json](../exp/runs/exp006_gsu_memory_guided/runs/run_batch_20261021_132023_431bf0_A1/metrics.json) | 395117 | 60d4ded83cb2e4be53c0dceb7e6fdfcc0fb0a883f1d8552bb15f4275506eea75 |
| 11 | [exp/runs/exp006_gsu_memory_guided/runs/run_batch_20261021_132023_431bf0_A2/metrics.json](../exp/runs/exp006_gsu_memory_guided/runs/run_batch_20261021_132023_431bf0_A2/metrics.json) | 395557 | c1d99f3d1feb468b36b22b2daf03e0bffe62a45b6ef27e475994ff9fe7a8a7e1 |
| 12 | [exp/runs/exp006_gsu_memory_guided/runs/run_batch_20261021_132023_431bf0_A2R/metrics.json](../exp/runs/exp006_gsu_memory_guided/runs/run_batch_20261021_132023_431bf0_A2R/metrics.json) | 395330 | 3469b2bed605e07335f86b24d4c55f77318faed954c5e524ad02868c0b6d048e |
| 13 | [exp/runs/exp006_gsu_memory_guided/runs/run_batch_20261021_132023_431bf0_A3/metrics.json](../exp/runs/exp006_gsu_memory_guided/runs/run_batch_20261021_132023_431bf0_A3/metrics.json) | 395509 | 0dd39d0d6ef1ca64c2f2afe81a1d00f0e7b2ad287890d965f6c67e4c518ab831 |
| 14 | [exp/runs/exp006_gsu_memory_guided/runs/run_batch_20261021_132023_431bf0_A3R/metrics.json](../exp/runs/exp006_gsu_memory_guided/runs/run_batch_20261021_132023_431bf0_A3R/metrics.json) | 395590 | 95c7923c82c87b1fe2243e2a160d61a0b116b7929eab4f87b73b7994cc4e3a4e |
| 15 | [exp/runs/exp006_gsu_memory_guided/runs/run_batch_20261021_132023_431bf0_A4/metrics.json](../exp/runs/exp006_gsu_memory_guided/runs/run_batch_20261021_132023_431bf0_A4/metrics.json) | 394992 | 9697203377ab38046ca97189332095776859b5b11cddb06d96b32bf5b95dc8c3 |
| 16 | [exp/runs/exp007_gsu_integrated_guidance/runs/run_batch_20261022_121504_28dccc_B0/metrics.json](../exp/runs/exp007_gsu_integrated_guidance/runs/run_batch_20261022_121504_28dccc_B0/metrics.json) | 399510 | 611a39f277f3b1196599763f2d2a06a9655bc79155e6e59ce45ee525761bb5c4 |
| 17 | [exp/runs/exp007_gsu_integrated_guidance/runs/run_batch_20261022_121504_28dccc_B1/metrics.json](../exp/runs/exp007_gsu_integrated_guidance/runs/run_batch_20261022_121504_28dccc_B1/metrics.json) | 398994 | 3e6c8c792ec48ffdb6b916137401a54ddb575a17c4aaa5910cf7f6a5aee5f807 |
| 18 | [exp/runs/exp007_gsu_integrated_guidance/runs/run_batch_20261022_121504_28dccc_B2/metrics.json](../exp/runs/exp007_gsu_integrated_guidance/runs/run_batch_20261022_121504_28dccc_B2/metrics.json) | 399245 | cd1db3361f190ab7a89c93cc97dc59177eaf982a1e6eb71a8393cffcf930d471 |
| 19 | [exp/runs/exp007_gsu_integrated_guidance/runs/run_batch_20261022_121504_28dccc_B2R/metrics.json](../exp/runs/exp007_gsu_integrated_guidance/runs/run_batch_20261022_121504_28dccc_B2R/metrics.json) | 399185 | 1997c4c0fda2017492ce14adcfd86ee502f97d2632130b885921e1977dd1666e |
| 20 | [exp/runs/exp007_gsu_integrated_guidance/runs/run_batch_20261022_121504_28dccc_B3/metrics.json](../exp/runs/exp007_gsu_integrated_guidance/runs/run_batch_20261022_121504_28dccc_B3/metrics.json) | 399508 | 07d1265d2a32da2598520cfa0228ee14797cda1daf8a7f984ed65eb1fc6ef8b5 |
| 21 | [exp/runs/exp007_gsu_integrated_guidance/runs/run_batch_20261022_121504_28dccc_B3R/metrics.json](../exp/runs/exp007_gsu_integrated_guidance/runs/run_batch_20261022_121504_28dccc_B3R/metrics.json) | 399441 | 13378516ab67aa7e34b42c2a14da1fb12cc83e6313e927b4939fe9c0b108ef1d |
| 22 | [exp/runs/exp010_gsu_step_routing/final_comparison_20261023.md](../exp/runs/exp010_gsu_step_routing/final_comparison_20261023.md) | 1093 | c247a3e8438f34121080e286ec3d2b0951408e4c07fa0e16087532520cf1631d |
| 23 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C0/status.json](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C0/status.json) | 167 | 2250a00e58a9da4ecb4daa03f4a6d84ab450df7a3c5dc92c2bca3ac606567604 |
| 24 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C0/metrics.json](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C0/metrics.json) | 427203 | 6a162cc705a92c77d63bb26b605854817d4844988b567482529d6c928d8ee158 |
| 25 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C0/diagnostics.jsonl](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C0/diagnostics.jsonl) | 789263 | ed3b6bb2120296df8e6a52e29a01d202d742085f595f96bb4162c35dc70f40a0 |
| 26 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/Cg/status.json](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/Cg/status.json) | 167 | d840c118f90568c16ba8cea33ce1e2f9b6795d23aa13f29422b146c5a1e81bb6 |
| 27 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/Cg/metrics.json](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/Cg/metrics.json) | 425877 | 0b715252585576439bfa29453faf5b0c689582cc029c1879803337dfa0e9b563 |
| 28 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/Cg/diagnostics.jsonl](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/Cg/diagnostics.jsonl) | 582742 | 2ede763ad59122351f1b1316dfa271a697c6fc9a6fc8f6481685e112b8e59954 |
| 29 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C1/status.json](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C1/status.json) | 167 | 732bdd349c27849255e17663ddfb597691204d32947c7fa70bbf45dc7b82b11b |
| 30 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C1/metrics.json](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C1/metrics.json) | 426492 | dcd08b23c5a30931a36765419917198c2a8761b2ce31f1641de4192cf24da9a0 |
| 31 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C1/diagnostics.jsonl](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C1/diagnostics.jsonl) | 642109 | 77ac009a4f526f488e1045c0944cd5e533ddb9d0acb846e3b338e4893162a24f |
| 32 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C2/status.json](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C2/status.json) | 166 | 6eb79c1962cd734cd3c6b906d20b3cf4fdf3402b15b68b7acb3446c75a8f652f |
| 33 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C2/metrics.json](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C2/metrics.json) | 425991 | ae55dba0397ea8179f69c2d5b11c30bfd3c6fedcb09bc6127d0a6b82f2fb8e7c |
| 34 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C2/diagnostics.jsonl](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C2/diagnostics.jsonl) | 639198 | ea9b964cb7e199ea6465ac67f189a7937ff984d810093c0a1b6781215de9bd10 |
| 35 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C3/status.json](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C3/status.json) | 166 | 12d27c4c251866dbeb545f4a9a4662e920819161cb6a87da64268d5b2a9f5509 |
| 36 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C3/metrics.json](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C3/metrics.json) | 425567 | a31622f2d7c1bc5531514b0d5eaf85c6cd2f554cd81dfe20a821c706e0e2d3be |
| 37 | [exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C3/diagnostics.jsonl](../exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1/C3/diagnostics.jsonl) | 635674 | 3044398c44a889fc4c83fc41ee8c13011d280216ebda5dba2a3dce68da775425 |
| 38 | [exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C3R/status.json](../exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C3R/status.json) | 166 | dd02257befdb5de899391c9379f2582df30336b768d2ddda27330390ca37b5c6 |
| 39 | [exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C3R/metrics.json](../exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C3R/metrics.json) | 425697 | 7b4492bdb874e8aa469a7fafbb1eda75de8cb6eb8a6e74356449639064237f3a |
| 40 | [exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C3R/diagnostics.jsonl](../exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C3R/diagnostics.jsonl) | 640960 | 55661b3f70cb9338f30950e49176e3fa3b5d20257ac9a17eb0d63e62356c25f2 |
| 41 | [exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C3D/status.json](../exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C3D/status.json) | 166 | b78123063fa2bbf1c60aac3c8245fe78400797d0e9187d5173d86b8db3a2c4d7 |
| 42 | [exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C3D/metrics.json](../exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C3D/metrics.json) | 425731 | 86d7fa66cc0672499c8c834b2a52ecfd715b1d8ca54b3f66b102e059d96689f0 |
| 43 | [exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C3D/diagnostics.jsonl](../exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C3D/diagnostics.jsonl) | 640384 | c396e1ee759ec9eec280c66459954cff092773d9d6c9532daadfb16b702bd0ad |
| 44 | [exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C4/status.json](../exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C4/status.json) | 167 | d8c641a1ac759473bc2b504467368d08d1707bffa72d86cebf3334f5ee3d6c16 |
| 45 | [exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C4/metrics.json](../exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C4/metrics.json) | 425863 | 7b1c363718ae89d10db0abfac0dc47af647769e70e6d14ee28f0db0c06e1fa92 |
| 46 | [exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C4/diagnostics.jsonl](../exp/runs/exp010_gsu_step_routing/runs/local_continue_20261023_205325/C4/diagnostics.jsonl) | 641207 | c81c77ea7761f95230e6e693583fd0422ce365756eaa5c73ca283161c323570b |
| 47 | [exp/runs/exp016_local_baselines/runs/continue_20261024_064547/comparison.json](../exp/runs/exp016_local_baselines/runs/continue_20261024_064547/comparison.json) | 6946 | 77621e1d6aa73be521407a5b56699c4ed4153d4f13dc50b5cae79c23133a4bc8 |
| 48 | [baselines/runs/formal_lora_20261024_120150/queue_status.json](../baselines/runs/formal_lora_20261024_120150/queue_status.json) | 49359 | b7f910ab668bca93b8fea4b8a39aa8b18ce6f4ca5646102ae0ee6ef45c021558 |
| 49 | [baselines/runs/formal_lora_20261024_120150/comparison.md](../baselines/runs/formal_lora_20261024_120150/comparison.md) | 785 | c65cbe0bb5f31c2464040fe2ea7c7f2101799ee3678dfbde83449f6eb82a142a |
| 50 | [exp/analysis/cloud_comparison_20261022/evidence/exp012_capability/comparison.json](../exp/analysis/cloud_comparison_20261022/evidence/exp012_capability/comparison.json) | 11844 | 4f15f2da99f13842ec379ce0262f771d6a5188e40d8fd35b22460a2afa458446 |
| 51 | [exp/runs/exp014_cloud_c1/runs/batch_20261023_01/llama32_3b_C1_audit/behavior.json](../exp/runs/exp014_cloud_c1/runs/batch_20261023_01/llama32_3b_C1_audit/behavior.json) | 407509 | b561d7b6d2ab8604ce120fa28a32893bb3296782dccb0b796444bcc83f48a644 |
| 52 | [exp/runs/exp014_cloud_c1/runs/batch_20261023_01/llama32_3b_C1_audit/mmlu.json](../exp/runs/exp014_cloud_c1/runs/batch_20261023_01/llama32_3b_C1_audit/mmlu.json) | 151628 | 31b2053e9ff31ba4f1f47d692572972283862990db4e960a5b8242c2447c18ef |
| 53 | [exp/runs/exp014_cloud_c1/runs/batch_20261023_01/llama32_3b_C1_audit/ppl.json](../exp/runs/exp014_cloud_c1/runs/batch_20261023_01/llama32_3b_C1_audit/ppl.json) | 566 | d9a53d5e6c23652c080a44507210bacda055c383f0d65a1c15c2e3b8ebb48c88 |
| 54 | [exp/runs/exp014_cloud_c1/runs/batch_20261023_01/qwen35_4b_C1_audit/behavior.json](../exp/runs/exp014_cloud_c1/runs/batch_20261023_01/qwen35_4b_C1_audit/behavior.json) | 406347 | 2ad33941fe81914c8179dc6ddb7ce36abd3871519c9514df4fc81d71bbfac863 |
| 55 | [exp/runs/exp014_cloud_c1/runs/batch_20261023_01/qwen35_4b_C1_audit/mmlu.json](../exp/runs/exp014_cloud_c1/runs/batch_20261023_01/qwen35_4b_C1_audit/mmlu.json) | 151579 | 0474edc8bb34afdd8fd8b6c8846831ef1141f19353b4519ca9c0beec42151094 |
| 56 | [exp/runs/exp014_cloud_c1/runs/batch_20261023_01/qwen35_4b_C1_audit/ppl.json](../exp/runs/exp014_cloud_c1/runs/batch_20261023_01/qwen35_4b_C1_audit/ppl.json) | 566 | 4bda48c8c1546172fa1b8cb0ec73ffea2a6661020cd59827c414a4e7dcbcfbbd |
| 57 | [exp/analysis/technical_report_20261025/evidence/exp016/comparison.json](../exp/analysis/technical_report_20261025/evidence/exp016/comparison.json) | 45957 | 8432ce34b09d902082a3eb07da55e32e0cc175473c7c2e374931198e31bede28 |
| 58 | [exp/analysis/technical_report_20261025/evidence/exp016/status.json](../exp/analysis/technical_report_20261025/evidence/exp016/status.json) | 4065 | bbd4fefef724d6884c26b1d9dc44bde9fc9a93b35db57901bc9585ba97be125c |
| 59 | [exp/analysis/technical_report_20261025/evidence/exp016/manifest.json](../exp/analysis/technical_report_20261025/evidence/exp016/manifest.json) | 223755 | 7d718036f9583358f3c8d4bb29cab42f16de950d83092d7d4b420f4cba691491 |
| 60 | [exp/results/阶段1预实验报告.md](../exp/results/阶段1预实验报告.md) | 2895 | 15190e15bc6fd39b8acaf73df78e093baad7b930ed9bb663b4371c54ff27715a |
| 61 | [exp/results/研究内容一实验结果报告.md](../exp/results/研究内容一实验结果报告.md) | 11546 | 61994f7dd5de3453d633f955870273c5aafa89a3a0d02beaadd05c1bfdaeb845 |
| 62 | [exp/results/3B_extension_report.md](../exp/results/3B_extension_report.md) | 2135 | c8935ee9ccebb2af9923acc1231f0657edc6ad5024a8608fea19bb1d6cde9d0c |
| 63 | [exp/results/研究内容二实验报告.md](../exp/results/研究内容二实验报告.md) | 21167 | 3e1092b055b2aa55b42029c3037997a3555fcde016dffd9128d238aa8183e800 |
| 64 | [exp/results/研究内容二阶段2归因结果.md](../exp/results/研究内容二阶段2归因结果.md) | 4633 | 7e8493e8cbf83dd852200b5bedba49cfb8b38e2f6fc79b65437225893568b16f |
| 65 | [exp/results/新遗忘方法GSU对比报告.md](../exp/results/新遗忘方法GSU对比报告.md) | 26097 | 6d40fbf58b5d91908627c60f0004bb309ed0fbf929897dbf8391e6754cb4dd68 |
| 66 | [exp/runs/exp017_decisive_validation/card.md](../exp/runs/exp017_decisive_validation/card.md) | 4935 | 7de049edd923ba6d0e969aea88012b986737881366998b4ba963eb78de59efe4 |
| 67 | [exp/runs/exp016_8b_multiseed/card.md](../exp/runs/exp016_8b_multiseed/card.md) | 4570 | 64c729348e1b38ec9c3e7f2c24505b5f325724aea69f23b4aff9e96523f0c99d |
| 68 | [exp/runs/exp015_complete_baselines/card.md](../exp/runs/exp015_complete_baselines/card.md) | 6440 | d22a4689415ba54ee762bcfd56a8e9588ed2732e6055c76ae53bff56dfb23790 |
| 69 | [exp/train/run_cloud_gsu_c2.py](../exp/train/run_cloud_gsu_c2.py) | 20340 | b455d1ae20df83cae21e4d52930fbeb446f9bea7ef83199c1f324a405fe59441 |
| 70 | [exp/train/run_local_gsu_c_group.py](../exp/train/run_local_gsu_c_group.py) | 25693 | 1d58db388c474f38c9d5cb4ec88d9473116f57800339ec9c0c45f3062a20d6c7 |
| 71 | [exp/train/diagnose_gsu_memory_guided.py](../exp/train/diagnose_gsu_memory_guided.py) | 71143 | 17116a82490b944531fbcf12d69818441b2670f799ed3af7642c07b282e98e6a |
| 72 | [exp/train/diagnose_gsu_integrated_guidance.py](../exp/train/diagnose_gsu_integrated_guidance.py) | 17926 | 783cdee92e80c8ef19711bfb60a1c4312ba351347a50fc2e69844240402fbd3d |
| 73 | [exp/train/diagnose_gsu_actual_feedback.py](../exp/train/diagnose_gsu_actual_feedback.py) | 24686 | c0803af7544037dbeda9c613087a3628727a0a02c3903cfa9318bf3ebd0a9bf2 |
| 74 | [exp/train/reproduction_losses.py](../exp/train/reproduction_losses.py) | 14397 | 1a5a0cf460b46189f484f45da1e139cf7f615913079167c6b0acb5b1fcc09c52 |
| 75 | [exp/train/reproduce_all_baselines.py](../exp/train/reproduce_all_baselines.py) | 34301 | a608587466c2b0466011754157343c7759b79eb978e7a6e81a25ad7c0a7fb28a |
| 76 | [exp/train/evaluate_cloud_capabilities.py](../exp/train/evaluate_cloud_capabilities.py) | 23274 | 4361b0088845e12ffa240c88165fe15280e14bf56a59b439d5f4b795b77f9795 |
| 77 | [exp/train/native_cpu_streaming.py](../exp/train/native_cpu_streaming.py) | 4065 | 3073b427428d91ad16e6670a6cd7932f5dd269c1292f91ab423cdab5d2b2e465 |
| 78 | [exp/train/decisive_controls.py](../exp/train/decisive_controls.py) | 3024 | 7d301ea84fd817429098b92b88ac4f395a9e7f32034ce9dfae4caaec394356f2 |
| 79 | [exp/train/development_evaluation.py](../exp/train/development_evaluation.py) | 1712 | 5b01f2b817cf02e1c248106b4d5f2a4953562e7278dc1c085bebb31574a548e7 |
| 80 | [baselines/README.md](../baselines/README.md) | 7365 | 342d11dc795195791f59adde5b7511728f5b75f9207c3597b171d9bc1f29f81b |
| 81 | [baselines/recipes.py](../baselines/recipes.py) | 6679 | 8caef9a0832cda287082cbbf9d0566ac3ab21fa6dc546450560926569950e281 |
| 82 | [baselines/objectives.py](../baselines/objectives.py) | 3506 | 0f43eb646378009870f82ada699635720d2e42143b25aff705bc5629af060a21 |
| 83 | [baselines/protocol.py](../baselines/protocol.py) | 2988 | 6140a815895bc3d5cf6c06b91c4f681885a263a1e703242f9274467e471bdddf |
| 84 | [baselines/source_lock.json](../baselines/source_lock.json) | 11496 | f8346e7898442ab8e463b1e5613e02f8da87740b6ee63969f5e38ec1a7a7e22c |
| 85 | [docs/plans/GSU最终候选与帕累托验证设计.md](../docs/plans/GSU最终候选与帕累托验证设计.md) | 9699 | 26d409de26c515d8242bb0dbd0c2a3ba4908f5d6f0ef68eee4ac021ce600ce21 |
| 86 | [docs/audits/本地baseline复现审计_20261024.md](../docs/audits/本地baseline复现审计_20261024.md) | 6986 | ddad64dc2b5943297cda389a4fd5b9a5f4f82681d8903bebb38b3a6e2e7b1b48 |
| 87 | [docs/reports/云端GSU-C2与指标说明.md](../docs/reports/云端GSU-C2与指标说明.md) | 5238 | b3d888ea24845b06e0987d52659c54668d836492439e2cf0a24a28190914508a |
| 88 | [exp/runs/exp004_gsu_retain_protection/posthoc_audit.md](../exp/runs/exp004_gsu_retain_protection/posthoc_audit.md) | 1904 | 9651fef0a77adbaad27c710f88efa2b845bc7efab742ed0644f7e62637c8ff3d |
| 89 | [exp/runs/exp005_gsu_actual_feedback/runs/run_20261020_1714_fb16/metrics.json](../exp/runs/exp005_gsu_actual_feedback/runs/run_20261020_1714_fb16/metrics.json) | 4436 | fac7618b2bbf0007795cf869bb5a9efa68c6de83c5e3fea7bc2a3cb7c8c7c83e |
| 90 | [exp/analysis/cloud_comparison_20261022/report_v3_klue_pending/comparison.json](../exp/analysis/cloud_comparison_20261022/report_v3_klue_pending/comparison.json) | 4453128 | 35c7f599fe6e5cc4057b2c6b9225cbea377a06f85e115a59a878d8cf7cdee303 |

### 20.3 数值来源规则

- 主结果表从指定JSON/JSONL转写，百分比×100一次；相对恢复摘要已经是百分比，不重复转换。每张表的分母、方向、模型、seed和监督合同在邻近文字注明。
- 8B均值/样本SD和配对差从逐seed记录重算；Original不复制三次，缺失或失败不填0。
- 第19节为历史Markdown小表汇编，不据其旧“最佳/成功”字样作新结论；当前主表和原始run优先。原始报告中的累计数、rho或归因机制解释若未本次重算，明确作为历史来源观察而非新增验证。
- 未回收的对齐版小模型C1完整指标、8B端点逐步日志/成本、exp017待运行结果和独立baseline完整能力表不伪造。
- 文献方法只按已存来源说明，不将本技术汇编当作本次独立外部文献核查。

### 20.4 报告与精简文档的维护关系

这是一份用户要求的可单文件阅读、包含大表和技术细节的研究快照；与四份日常精简文档分工不同。日常状态与新增成绩仍更新对应run及主表；技术总报告更新时重采证据并注明截止时间，不靠手动改几行维持“实时”。

本次没有启动训练、变更算法或回写历史结果。采集仅包含汇总、状态和源清单，没有传输模型权重；报告不含SSH密码。


