# exp006：多记忆组件引导并排验证

date: 2026-10-21
assumption_id: H-GSU-MM-1
method_version: GSU-v6-memory-guided
status: frozen; formal training authorized on 2026-10-21; runtime state in batch/run status.json

## hypothesis

仅在原始 F0 上施加遗忘损失，训练前利用多条关联记忆定位共享组件，可以减少关联多跳回忆，并比直接遗忘 F0+F1 更好地保留非目标知识。这是待验证假设，不是已经得到的实验结论。

区分两个问题：F1 支持事实参与一次性定位，不等于对 F1 施加遗忘损失；F0-only 更新也可能隐式损伤 F1，因此不能仅以训练标签宣称“没有遗忘 F1”。

## methods

| ID | 名称 | 遗忘损失输入 | 冻结组件引导 |
|---|---|---|---|
| A0 | F0-FB16 | F0 | 无 |
| A1 | F1-FB16 | F0+既有 5 条 F1 | 无，当前扩展基线 |
| A2 | F0-MM-Soft | F0 | 共享分数软权重 |
| A3 | F0-MM-Hard | F0 | 共享分数中位数硬掩码 |
| A4 | F1-MM-Soft | F0+既有 5 条 F1 | 与 A2 相同的软权重 |
| A2R | F0-MM-Soft-Random | F0 | 随机置换 A2 分数 |
| A3R | F0-MM-Hard-Random | F0 | 同一随机置换后取中位数掩码 |

所有组保留同一个 FB16 控制器。A5 PathQ 不实现：当前没有独立的 train-only 多跳查询集，不能把 hop2 评测题改作训练数据。

## expected_pattern

1. A1 相对 A0 降低 F1_hop2，检验直接扩展遗忘在同一反馈控制器下是否仍有作用。
2. A2/A3 相对 A0 降低 F1_hop2，同时主指标和 retain 不低于 A1。
3. 结构化引导优于对应随机引导，才有支持共享结构解释的方向性证据；随机对照持平时不能将收益归因于多记忆定位。
4. F0-only 候选应尽量保持 F1 的直接事实，而不是靠隐式遗忘公共支持事实换取 hop2 降低。
5. A4 检验引导与扩展监督是否互补，不预设它支配 A1。

## frozen_primary_metric

唯一主指标：`D_hop2_no_F1_overlap`，12 题的 MCQA 命中数/正确率，越高越好。它是与 F1 链不重叠的历史损伤诊断子集，不等同于独立 retain 集，也不能代替多跳遗忘指标。

关键指标：retain 33、F1_hop2 75（越低越好）、D_hop2_F1_overlap 12、D_hop2 24、hop1_unlearn 33、K 48、D_so 24、五集 same_object 与整体 hop2。

辅助审计：F0、直接 F1、关联记忆、各评测切片的答案 token 平均 logprob、旧 MCQA 正确候选分数、正确候选相对最强干扰项的 margin、逐题初始/最终变化、raw/final 锚点风险、每条锚点相对训练初始的累计损伤。

旧代码的 MCQA helper 实际计算整个 `q + answer` 序列的平均 logprob；本轮继续使用它计算行为指标和 margin，不静默更改历史口径。答案 mask 的条件 logprob 单独记录，不能与前者混报。

## frozen_primary_baselines

主要比较对象为本轮等预算重跑的 A0、A1，以及机制对照 A2R/A3R；不得以历史 G1 的 99 步结果代替 A0 的 114 步结果。

历史参考，不替代本轮基线：

| 方法 | 主指标 | retain | F1_hop2 | 来源 |
|---|---|---|---|---|
| M3 | 0/12 | 16/33 | 25/75 | exp002 / run_20261019_1631_m3_reval |
| M3A | 3/12 | 11/33 | 9/75 | exp003 / run_20261020_1332_m3a_train_diagnostic |
| COV16 | 9/12 | 20/33 | 29/75 | exp004 / run_20261020_cov16_rerun |
| CUM16 | 9/12 | 23/33 | 41/75 | exp004 / run_20261020_cum16 |
| FB16 | 3/12 | 19/33 | 22/75 | exp005 / run_20261020_1714_fb16 |

历史 FB16 raw→final 锚点下降为 1780/1824→838/1824；相对 COV16 的 866/1824 只减少 28 次。这只支持当步反馈修正，不支持累计保持已经解决。

## frozen component score

Qwen2.5-1.5B 的本地配置为 28 层，每层 7 个目标矩阵，共 **196 个组件**；224 对应 32 层模型，不能套用到当前模型。每个组件合并对应 LoRA A/B 参数的梯度，所有分数在初始 LoRA 状态计算一次：

```text
g_ij = grad log p(F0_i answer | F0_i question) on component j
mu_j = mean_i(g_ij)
A_j = norm(sum_i g_ij) / sum_i norm(g_ij)
P_j = sum_m path_count_m * cosine(mu_j, grad(F1_m,j)) / sum_m path_count_m
S_j = A_j * max(P_j, 0)
soft weight_j = S_j / max_j(S_j)
hard weight_j = 1[S_j >= median(S)]
```

零范数的一致性/cosine 记为 0；全体 S=0 时软权重为全 1。中位数采用通常定义，偶数时取中间两数均值；相等分数全部保留，不强制 top-k。重加权后恢复原始整体梯度范数。如果非零原始梯度被冻结权重完全消除，运行失败，不悄悄改回无引导算法。

组件平均梯度强度仅记录，不额外乘入 S。关联记忆严格等于既有 F1 的 5 条直接事实，其活跃路径数固定为 7、7、7、2、2。P 使用这些固定路径数加权的梯度一致性，不声称已经找到因果电路或实现逐路径梯度匹配。

A3 仅掩掉遗忘梯度，不冻结整个参数：AdamW weight decay 和后续 retain 反馈仍可能修改掩掉的组件。A2 的固定正权重也可能被 AdamW 自适应归一化抵消；这必须作为可能的零结果解释，而不是预先认定定位有效。

## null_control

- A2R/A3R 用 seed=7961 随机置换同一分数向量，保留分数/软权重分布与硬掩码数量；不随机重新筛选 F1。
- A0/A1 的权重恒为 1，不使用定位分数更新；它们也执行相同定位预计算，以保持初始化和可审计开销一致。
- 无实际 retain 风险时 FB16 不修正；无引导时梯度必须逐元素保持相同；全零分数时退化为恒等权重。

## budget

- 模型：本地 Qwen2.5-1.5B-Instruct；不在本卡内自动换成 3B。
- 数据：FaithUn 固定原始文件及 forget/retaining/test split，F0=33，F1=5；不重新选择 F1，不使用全部 61 个候选训练。
- seed=42；LoRA r=8、alpha=16、dropout=0；全层 q/k/v/o/gate/up/down。
- AdamW：lr=5e-5、betas=(0.9,0.999)、eps=1e-8、weight_decay=0.01；沿用历史默认值并显式归档。
- 每组恰好 114 次更新，7 组共 798 次；不做参数搜索或自动追加训练。
- F0-only：3 次完整遍历的 99 步，再取第 4 次固定置换的前 15 条；每条 F0 出现 3 或 4 次。F0+F1：3 次完整遍历，每条出现 3 次。
- 排序沿用旧 FB16 的原地连续 shuffle，block k 使用 seed=42+k；所有组预先保存 schedule 和重复计数。这里“3 epochs”仅为 F0+F1 的参考预算，不能把 F0-only 的 114 步称为严格 3 epochs。
- 每步从前 33 条 retain 中抽取 16 条，保存整个 114 步的相同锚点计划；覆盖必须为 33/33。
- FB16 固定实际下降阈值 1e-4、最多 2 轮、总修正范数不超过 raw AdamW 位移的 0.30，审计余量 1e-6；不改 AdamW 动量。
- 全 66 条 retaining 在初始/最终额外审计，并分别报告训练锚点 33 条与未作保护的后 33 条。全 66 条不是独立留出集。
- 单卡串行；GPU 时间、峰值内存和存储开销待运行实际记录，当前不编造耗时预算。

## screening gate and kill rules

为避免事后解释，本轮方向性筛选使用离散的“至少少命中 1/75”而非未经定义的“明显降低”：候选 F1_hop2 必须低于 A0，主指标不低于 A1，retain 不低于 A1。

对原设计中“不可接受的隐式 F1 遗忘”采用保守的零额外损伤门：F0-only 候选的直接 F1 MCQA 命中数、平均答案 logprob 均不得低于 A0。仍报告每条事实和最坏损伤，不以均值隐藏损伤；未过此门不能称为“保持支持事实的多跳遗忘”。这是事前筛选约定，不是已有实验规律。

对应随机对照如果也能达到相同方向性门，本轮不能确认共享结构机制；单次随机置换和单 seed 无法证明统计显著性。即使方向性门通过，也只能提出独立多 seed 确认，不能写为已验证优越算法。

非有限值、空答案 mask、无效 split/F1、组件数量异常、solver 不收敛、预算超限、锚点缺失、更新不足或评测计数不一致均立即失败并保留现场；不覆盖旧 run，不在本卡内搜索阈值救援。若全体分数为零或硬掩码全部保留，报告定位退化，不包装成非平凡机制成功。

## observed

尚无完整训练结果。自检和预检仅证明实现的部分契约，不填入遗忘/保持效果结论；正式数字必须引用新 run_id 的 metrics.json 和 diagnostics.jsonl。

## sanity_checks

- 原始数据与 split 的 SHA-256、源代码/方法/实验卡快照及 dirty patch 在运行前归档。
- 所有 F0-only 组使用同一 114 步 F0 计划；A1/A4 使用同一 F0+F1 计划；锚点计划跨所有组完全相同。
- 定位前后 LoRA 参数哈希不变；跨组初始 LoRA 和原始组件分数哈希一致。
- F1_hop2 是既有 F1 覆盖链的诊断，不是独立泛化集。既有 F1 的选择使用了原始多跳探测元数据，不能称为全新的 train-only 多跳划分。
- hop2 评测题不进入梯度目标；A0/A2/A3/A2R/A3R 遗忘更新日志中只能出现 F0 来源。
- raw 概率在 AdamW 之后、反馈之前实际测量；不能读取旧 helper 的修正后字段冒充 raw。
- 累计损伤直接比较当前 logprob 与初始 logprob，不用采样步的局部变化求和代替；38/76/114 步复测全部 33 条锚点。
- 五集与六集保持旧 helper 的行为，新增 answer-only 审计单独命名。

## interpretation

待结果。任何新增的事后分析标为 exploratory；不因结果改变本卡的主指标、对照或门槛。

## verdict

untested

## cause_guess

unknown

## next_action

continue：实现自检、离线预检与 30 项回归测试已通过；按用户 2026-10-21 的启动授权，用 start_gsu_memory_guided.ps1 串行运行七组并自动汇总。保持本卡的算法、预算、指标和筛选门不变，失败保留现场并停止，不自动追加实验。

## assets_salvaged

固定 F1/活跃路径、FB16 控制器、原五集/六集评测、同预算数据与锚点计划、隐式 F1 审计、逐步 raw/final 与累计 logprob 记录。
