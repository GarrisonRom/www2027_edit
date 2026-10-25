# exp007：历史损伤归因与梯度/路径引导整合

date: 2026-10-22
assumption_id: H-GSU-INT-1
method_version: GSU-v7-integrated-damage-gradient-path
status: implementation frozen; not trained

## hypothesis

历史 G1/G2 损伤归因可以作为训练前组件路由的先验。将它与当前 F0 内部梯度一致性、F0--F1 直接事实梯度方向和活跃路径权重结合，可能比只使用当前梯度/路径分数更集中地把遗忘更新写入已经有损伤证据的共享组件，从而降低 F1 多跳回忆。

这是一个可证伪的机制假设，不是“组件损伤等于因果电路”的声明。历史归因只有 FFN 层级的直接证据，不能伪装成 q/k/v/o/gate/up/down 的完整矩阵级 prior。

## methods

| ID | 训练对象 | 引导信号 |
|---|---|---|
| B0 | F0 | 无引导，等预算基线 |
| B1 | F0+固定 5 条 F1 | 无引导，扩展基线 |
| B2 | F0 | 当前梯度/路径分数 × 历史 G1/G2 FFN 层先验，软权重 |
| B3 | F0 | 同 B2 的综合分数，中位数硬掩码 |
| B2R | F0 | 随机置换 B2 综合分数 |
| B3R | F0 | 随机置换 B3 综合分数 |

B2/B3 的历史先验只作用于 MLP LoRA 组件；attention LoRA 组件使用中性先验 1.0。B2R/B3R 保持综合分数分布和硬掩码数量不变，只随机打乱组件映射。

## expected_pattern

1. B1 应复现直接加入 F1 对多跳回忆的主要降低。
2. 若整合信号有效，B2/B3 的 F1_hop2 应低于 exp006 的 A0，并且至少有一个候选相对 exp006 的同类 F0-only 方法改善。
3. B2/B3 的收益应超过 B2R/B3R；若随机对照相同，历史归因与梯度/路径整合不能被解释为结构性定位。
4. B2/B3 的直接 F1 审计和目标相关 retain 损伤应与 F1_hop2 一起报告，不能只看多跳下降。
5. 若 B2/B3 只改变保留能力而 F1_hop2 不下降，说明历史损伤先验和训练前共享梯度之间没有形成有效更新路由。

## frozen_primary_metric

唯一主指标：`D_hop2_no_F1_overlap`，12 题，越高越好。关键遗忘指标为 `F1_hop2`（75 题，越低越好），并报告 D_hop2_F1_overlap、hop1_unlearn、K、same_object、retain、直接 F1 和全 66 条 retaining 审计。

本实验不把 retain 总量当作唯一成功标准：retain 必须按目标相关性进一步分层。现阶段先保留 exp006 的 33 条锚点池和后 33 条非锚点审计；新的关系分层审计另建数据分析卡。

## frozen_primary_baselines

- 新鲜等预算 B0/B1：都使用 114 次更新、同一 FB16 反馈、同一锚点计划。
- exp006 同类参考：A0 F1_hop2=52/75、主指标=9/12；A2=57/75；A3=47/75；A2R=64/75；A3R=73/75；A1=22/75。
- 历史归因证据：G2 FFN per-layer damage profile、G1/G2 shared top-20 FFN neuron 与层级/模块撤销结果。历史数字只作为冻结先验来源，不混入新 run 的训练目标。

## frozen_score

当前梯度/路径分数由 exp006 固定为：

```text
gradient_path_S(j) = A_j * max(P_j, 0)
```

历史先验由 G1/G2 的 `attribution/ffn_G1.json` 与 `ffn_G2.json` 的 `per_layer.D` 构成：

```text
damage_prior(l) = sqrt(
    D_G1(l) / max_l D_G1(l)
    * D_G2(l) / max_l D_G2(l)
)
```

最终综合分数：

```text
integrated_S(j) = gradient_path_S(j) * damage_prior(layer(j))   # MLP
integrated_S(j) = gradient_path_S(j)                           # attention, neutral prior
```

不搜索温度、不引入 top-k、不使用最终评测结果调分数。每个 run 保存原始 gradient_path_S、历史 prior、integrated_S、组件类型和来源哈希。

## null_control

- B2R/B3R：随机置换综合分数到组件，保留分布和硬掩码数量。
- B0/B1：无引导，验证整合分数不是普通 FB16 或训练预算效应。
- attention 中性 prior：明确是证据缺口的保守处理，不把 attention 归因结果伪造出来。
- 定位前后 LoRA 参数哈希必须一致；历史归因文件只读，不能被新 run 改写。

## budget

- Qwen2.5-1.5B-Instruct，seed=42；LoRA r=8、alpha=16、dropout=0。
- 全层 q/k/v/o/gate/up/down；AdamW lr=5e-5；114 次更新；16 条 retain 锚点；FB16 阈值、反馈轮数和 0.30 修正预算不变。
- B0/B2/B3/B2R/B3R 使用 F0-only 固定计划；B1 使用 F0+F1 三轮计划。
- 单 seed 机制筛选，不自动启动多 seed，不修改 exp006 历史结果。

## kill rules

- 历史 FFN 文件缺失、哈希变化、profile 非法、组件数量异常、非有限值、训练步数不足、反馈 solver 失败或评测计数异常，run 失败并保留现场。
- B2/B3 与随机对照无差异时，拒绝“整合定位优于随机”机制解释。
- B2/B3 不低于 A0 的 F1_hop2 时，不支持“损伤 prior 强化多跳封堵”。
- 即使 B2/B3 通过单 seed 方向性门，也只能进入新的多 seed 卡，不能直接升级为方法 claim。

## observed

待运行。本卡不预填结果。

## verdict

untested

## next_action

continue：先完成代码自检和单 seed 机制筛选；结果后依据随机对照和目标相关 retain 分层决定是否继续。

## assets_salvaged

exp006 FB16 trainer、F0/F1 固定计划、历史 G1/G2 FFN damage profiles、shared top-20 FFN 证据、层/模块撤销结果、全量审计和并排汇总工具。
