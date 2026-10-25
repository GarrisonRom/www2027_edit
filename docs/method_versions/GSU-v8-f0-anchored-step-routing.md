# GSU-v8: F0-anchored shared-subspace actual-step routing

date: 2026-10-22
parent: GSU-v6 / GSU-v7
status: proposed, not implemented or trained
experiment: exp010_gsu_step_routing (draft)

## Motivation and evidence

exp006 A3 has lower overall recall than B2, while exp007 B2 improves only the
F1-covered slice versus B0 and loses F0 forgetting strength. B2 correct-answer
LP is higher despite lower MCQA accuracy. Neither result justifies a best-method claim.

Fixed gradient scaling may be attenuated by Adam normalization. This theoretical
failure mode requires direct measured-gradient-versus-displacement verification.
Component location does not determine direction inside a component.

## Proposed change

Preserve F0-only forget loss; reuse A/B risk scores as separate ablation branches.
Soft floor w=(1+normalized_score)/2. Filter unmodified AdamW displacement through
per-LoRA-tensor F1 gradient spans, refreshed before updates1/39/77. Main proposal
v_j=w_j*(d0_j+P_j*d0_j)/2. No added F1 gradient or F1 forget loss.

Total routing+feedback correction <=.30||d0||. Keep FB16 unchanged, then validate
actual F0 decrease >=90% of positive raw proposal decrease; fixed interpolation
backtracking alpha1,.5,.25,0 and raw fallback. Reaudit retain after safeguard.
Frozen zero-gradient tensor fallback is identity, not a permanent hard mask.

This accesses F1 labels for guidance and backward probes; it is not F0-only data
access. Full mathematical contract and eight controls are in
../GSU最终候选与帕累托验证设计.md.

## Reproducibility and boundaries

No run exists. Constants are a proposed single-shot probe, not tuned/validated optima.
Risk scores are association/damage proxies, not proven causal memory circuits.
New architecture needs new localization; do not transfer Qwen2.5 layer priors.
Do not mutate historical v6/v7 source or reinterpret their frozen success gates.


## 2026-10-23：本地 C 组已完成结果（部分批次）

来源：`exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1`，Qwen2.5-1.5B，seed42，各114步。C0八步回放最大差0，五项正式计数与历史B0一致，复现门通过。

| 方法 | 全89路径宏回忆 ↓ | 覆盖多跳 /75 ↓ | F0 /33 ↓ | F1 /5 | 留出retain /33 ↑ | 护栏回退步数 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 44.94% | 52 | 8 | 5 | 15 | 0 |
| Cg | 32.58% | 9 | 12 | 1 | 10 | 98 |
| C1 | 27.72% | 0 | 12 | 0 | 12 | 103 |
| C2 | 29.96% | 7 | 12 | 0 | 10 | 109 |
| C3 | 30.71% | 7 | 13 | 0 | 10 | 108 |

观察：C1在已完成Cg/C1/C2/C3中，全路径回忆最低；相对Cg/C2，F0残留相同且留出retain更高。C1未支配C0：直接遗忘和retain均较差。C1的F1直接事实为0/5，不能解释为保持支持事实而独立切断推理。方向过滤C2及历史先验C3在本次观察中未超过C1，不证明梯度信息无效；C1本身通过F0/F1梯度构造组件分数。

状态核验：C3R文件停于2026-10-23 14:46:41、第36/114步；检查时未发现对应Python进程。队列疑似中断，不能按旧status=running宣称正常运行。C3R无最终指标，C3D/C4未完成；不得填零或参加排名。本次不重启、不覆盖原始run，也不改云端主候选。C3仍为本地事前主候选，C1只为事后探索性领先候选；随机对照、多seed与云端独立确认待完成。

详细逐题证据见各变体metrics.json及diagnostics.jsonl，最终选择不用38/76步最优checkpoint。


## 2026-10-23 最终完成：本地 C 组八变体筛选

批次分为原批次与续跑批次，均为 Qwen2.5-1.5B、seed42、114步/变体。C3R在原运行第36步中断后，从相同初始化在独立续跑目录完整重跑；C3D、C4也在续跑目录完成。C0八步回放与历史B0数值最大差0，五项正式计数复现门通过。八组均有最终 metrics.json；续跑总状态为 completed。

主指标按全89路径宏回忆升序排列：

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

C1在主指标排名第一，C4第二；C2与C3R在全89路径指标并列第三。C1相对C4降低全路径宏回忆1.87个百分点，覆盖多跳为0/75而C4为7/75；两者F0残留均为12/33，C1留出retain为12/33、C4为10/33。C1 F1直接题为0/5。单seed结果仅构成探索性筛选：不声称统计或全局最优；C3仍是事前指定主候选，C1是事后观察到的经验领先候选。因C1的F1直接题全部答错，不能声称其在保留支持事实时独立切断多跳推理。五项门槛和逐题原始证据见两批的 metrics.json、diagnostics.jsonl，以及 exp010/results_20261023.md。
