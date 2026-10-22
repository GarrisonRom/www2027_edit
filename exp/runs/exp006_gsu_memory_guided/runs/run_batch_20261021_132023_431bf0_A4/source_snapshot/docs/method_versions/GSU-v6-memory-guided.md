# Method Version: GSU-v6-memory-guided

date: 2026-10-21
parent: GSU-v5-actual-feedback
status: implementation; untested on full model
experiment: exp006_gsu_memory_guided

## what_changed

增加一条与旧实现隔离的诊断入口：`exp/train/diagnose_gsu_memory_guided.py`。不修改 M3/M3U/M3A、COV16/CUM16、FB16 的算法文件、adapter 或历史结果。

本轮只有一个算法干预：在 AdamW 前，用训练开始时冻结的组件权重重加权当前遗忘梯度，再恢复整体梯度范数。F0-only 与 F0+F1 两种监督、软/硬引导及随机分数置换形成 A0–A4、A2R/A3R 七组对照。FB16 的实际 logprob 反馈保持阈值 1e-4、最多两轮、位移修正比例 0.30。

## why

exp004 的 wider coverage 与 exp005 的实际反馈均有局部机制证据，但没有解决最终累计保持问题。exp005 的 `run_20261020_1714_fb16` raw/final 锚点下降数为 1780/1824 和 838/1824，retain=19/33；相对 COV16 的下降率只改善 1.54 个百分点，未通过原推进门。

新的可证伪假设 H-GSU-MM-1 是：第一阶段定位共享记忆所使用的组件，然后仅在原始 F0 上遗忘，可能减少关联多跳回忆而不必直接遗忘 F1 支持事实。这不是继续调整反馈阈值，也不是已经证明定位有效。

## pipeline

1. 固定 FaithUn 数据、split、已有 5 条 F1 和其活跃路径元数据；不重新筛选。
2. 在相同初始 LoRA 上逐例计算 F0 和 F1 直接事实的答案梯度，不执行 optimizer step；检查定位前后参数哈希相同。
3. 每个组件合并 LoRA A/B 梯度，计算 F0 梯度内部一致性 A、F0 均值与 F1 直接事实梯度的路径数加权 cosine P，冻结 S=A*max(P,0)。
4. 软引导使用 S/max(S)，硬引导使用 S>=median(S)。偶数中位数取中间两值均值，ties 全部保留。全零分数使软引导退化为恒等权重。
5. 根据组别仅计算 F0 或 F0+F1 的当前遗忘梯度，应用冻结权重并保持整体梯度范数；随后执行一次 AdamW。
6. 保存 raw 状态下 16 条锚点的真实答案 logprob，再调用原 FB16 反馈；分别记录反馈后的 logprob 与预算。
7. 每条锚点记录相对训练初始的累计概率变化，38/76/114 步重新审计整个 33 条保护池；最后评测固定五集/六集和全 66 条 retain。

## components and boundaries

本地 Qwen2.5-1.5B 的 config.json 是 28 层，共 28*7=196 个目标组件、392 个 LoRA A/B 参数张量。224 是 32 层模型的组件数，不适用于当前模型。保留所有层的 q/k/v/o/gate/up/down，不重新使用仅高层 MLP 的定位假设。

P 使用 F1 支持事实梯度及既有活跃路径数，不能解读为已恢复精确多跳因果连接。F1 与 associated_memory 审计目前是同一组 5 条事实的两个语义标签，不是两个独立验证集。初始化时 LoRA B 为零，一次性分数只反映这一初始参数化下的梯度；这也是需要随机对照的原因。

A3 约束的是遗忘梯度，非选中组件仍可能被 weight decay 和 retain 反馈修改。软权重的固定正缩放可能被 AdamW 自适应尺度部分抵消，本轮保留这个可检验的零结果，不改成额外的更新缩放机制。

既有 F1 的选择使用过多跳探测元数据；本轮只保证评测题文本不作为新训练目标，不声称 F1 选择与评测完全独立。A5 PathQ 继续暂缓，须先有独立 train-only 查询。

## reproducibility

- 卡片：`exp/runs/exp006_gsu_memory_guided/card.md`，含完整基线、主指标、预算、方向性门及 kill rules。
- seed=42；Qwen2.5-1.5B；LoRA r=8/alpha=16/dropout=0；AdamW lr=5e-5；每组 114 步。
- F0-only 为 33*3+15 步，F0+F1 为 38*3 步；使用预生成的连续原地 shuffle 计划，同类型跨组计划一致。
- 各组使用相同的 16/33 锚点抽样计划；随机引导 seed=7961，保持分数分布和掩码数量。
- 运行前归档 config.json、源代码快照、输入 SHA-256、Git commit/dirty patch、依赖/GPU、命令和数据/锚点计划；未提交的新脚本通过 source_snapshot 保留。
- 定位后另存 resolved_config.json、component_scores.json、原始分数及初始 LoRA 哈希；不覆盖运行前配置。
- 所有 run 必须使用新目录；失败/中断写入 status.json 并保留日志，不复用旧 run。
- 对模型权重记录本地路径、文件大小和 mtime，配置/tokenizer 文件记录 SHA-256；没有把权重的 size/mtime 冒充完整内容哈希。

## evaluation

主指标保持 `D_hop2_no_F1_overlap`，12 题，越高越好。多跳遗忘的关键判据仍是 F1_hop2 与整体 hop2，不能把主指标高误说成遗忘更强。

保留旧 MCQA helper 计算行为正确率和正确候选 margin；它使用整个 q+answer 的长度归一化分数。新增 answer_logprob 为 mask 后答案 token 均值，correct_answer_logprob 为旧 MCQA 正确候选分数，两者分开。

retain 的前 33 条是保护池，后 33 条才是相对本轮保护的留出样本；全 66 条汇总并非独立 held-out 成绩。F0-only 的直接 F1 命中和平均答案 logprob 必须同时审计，不能只看 F1_hop2。

## notes

尚未启动完整训练，也没有算法提升结论。实现修复包括：使用真正的数学中位数；恢复正确的 raw/final 采样时点；识别实际 196 组件；保留与旧 FB16 一致的原地 shuffle；补齐预运行档案与真实累计审计。它们不构成新的效果证据。
