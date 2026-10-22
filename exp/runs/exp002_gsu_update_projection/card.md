# exp002：GSU 更新方向补强

**状态：** 单 seed 预注册筛选和 M3U 逐步机制诊断已完成；M3C/M3U 均未通过进入多 seed 的冻结门槛，原 M3U 暂不修改。
**假设：** H-GSU-UPD-1。
**方法版本：** GSU-v2 候选，见 [`GSU-v2.md`](../../../docs/method_versions/GSU-v2.md)。

## 目的与假设

检验旧 M3 的无条件梯度投影是否在无冲突时也移除了有效遗忘方向，以及梯度投影是否能约束 AdamW 的实际参数位移。M3C 或 M3U 预期在保留 F1 设计内遗忘效果的同时，提高 D_hop2 非 F1 重叠子集的正确率；M3C/M3U 也可能无收益，不能预设成功。

预期模式：M3C 的条件投影触发率低于总参数张量检查数，且不削弱 F0/F1 遗忘；M3U 只在实际位移的一阶锚点变化为负时修正，修正步比例及保留指标可用于判断 AdamW 偏离是否实际存在。单 seed 结果只筛选后续候选。

## 冻结比较

- **模型与数据：** Qwen2.5-1.5B-Instruct；固定 FaithUn 数据、split、G2 遗忘例和 33 条 retain 锚点；不改数据或锚点。
- **条件：** 历史 M3、M3C、M3U；同一 seed（`train_eval.SEED=42`）、LoRA r=8/alpha=16、lr=5e-5、3 epochs、每步 4 个锚点。M3 使用现有 adapter，仅以新结果标签重评；M3C/M3U 各训练一次。
- **null control：** M3 是冻结基线；M3C 在不冲突参数张量上不执行投影，M3U 在实际更新不降低锚点目标时不执行修正。这两种 no-op 条件必须保持恒等行为。
- **唯一主指标：** `D_hop2_no_F1_overlap` 的三选一准确率（12 题，越高越好），主比较为 M3C/M3U 对 M3。
- **关键次指标：** `D_hop2_F1_overlap`（12 题）、全体 `D_hop2`（24 题）、`F1_hop2`（75 题，越低表示遗忘目标越彻底）、`hop1_unlearn`（33 题，越低越好）、retain（33 题，越高越好）、K（48 题，越高越好）及 same_object（24 题）。同时记录 M3C 冲突投影计数和 M3U 实际更新修正计数。
- **冻结基线：** M3；不根据结果替换主要比较对象。

## 预算与停止条件

- 每个新变体 1 个 seed、38 个遗忘例、3 epochs（约 114 次更新），每次更新抽 4 个 retain 锚点；总计 2 次训练。M3 只做一次评测，不重训。
- 不在此实验内搜索 lr、alpha、锚点数量或投影阈值；如需调参，另建实验卡并使用独立验证依据。
- 单变体出现非有限 loss/指标、空评测集或断点/适配器不一致时立即停止并修复，不将该运行计为有效结果。
- 候选只有在主指标至少多答对 2/12 题，且 F1_hop2 恶化不超过 2/75、retain 下降不超过 2/33 时才进入多 seed 确认；否则记录为未通过本轮筛选，不临时改门槛。
- 无论结果如何，每个变体只跑一个 seed；单 seed 不能作为显著性或普遍有效性的证据。

## 评测与产物

`D_hop2` 24 题中有 12 题与 F1 设计内遗忘链重叠、12 题不重叠。后者只是“未与 F1 链重叠”的诊断子集，不自动等同于完整 retain 集或受保护知识集。F1_hop2 是设计内目标指标，不是独立遗忘集外泛化指标。

复评 M3 并训练候选时使用新前缀 `gsu_projection`，结果写入 `exp/results/eval_results_gsu_projection.jsonl` 与 `exp/results/gsu_projection_sixset.jsonl`；不追加旧结果文件。新适配器和断点分别使用 `1p5B_v2_GSUM3C`、`1p5B_v2_GSUM3U` 名称。实际运行时记录代码提交 SHA、模型路径/标识、依赖、GPU、开始时间、命令、结果文件和异常。

建议命令：

```powershell
python exp/train/train_gsu.py --variant M3 --eval_only --result_tag gsu_projection
python exp/train/train_gsu.py --variant M3C --result_tag gsu_projection
python exp/train/train_gsu.py --variant M3U --result_tag gsu_projection
```

## 既有观测

- 历史 M3：D_hop2=0/24、retain=16/33、F1_hop2=25/75，来源为 `exp/results/gsu_sixset.jsonl`；对应结果无独立 run_id，不能补造。
- 历史 D_hop2 只有汇总预测；当前没有独立保存的 12/12 子集逐题结果。交集规模来自 `exp/results/attribution/items.json`、`exp/results/F1.json` 和固定 forget split。
- 本轮 M3 复评：`D_hop2_no_F1_overlap=0/12`、`D_hop2_F1_overlap=0/12`、`F1_hop2=25/75`、retain=`16/33`，来源 [`run_20261019_1631_m3_reval`](runs/run_20261019_1631_m3_reval/metrics.json)。
- 本轮 M3C：主指标=`0/12`（相对 M3 +0），`F1_hop2=24/75`、retain=`17/33`；三 epoch 投影触发数分别为 `12981/14896`、`13143/14896`、`11449/14896`，来源 [`run_20261019_1631_m3c_train_eval`](runs/run_20261019_1631_m3c_train_eval/metrics.json)。
- 本轮 M3U：主指标=`1/12`（相对 M3 +1），`F1_hop2=9/75`、retain=`13/33`；三 epoch 实际更新修正为 `38/38`、`37/38`、`37/38`，来源 [`run_20261019_1631_m3u_train_eval`](runs/run_20261019_1631_m3u_train_eval/metrics.json)。
- M3C/M3U 的主指标均未达到至少多答对 `2/12`；M3U 的 retain 下降 `3/33`，超过允许的 `2/33`。两个候选都不进入多 seed 确认。
- M3U 逐步诊断：同一批采样锚点的直接答案对数概率在修正后相对原始 AdamW 有 `437/456` 个变好，平均配对增益 `+0.656008`；负的直接变化由 `427/456` 降至 `261/456`。来源 [`run_20261020_1115_m3u_diagnostic_rerun`](runs/run_20261020_1115_m3u_diagnostic_rerun/metrics.json) 与 `diagnostics.jsonl`。
- 该诊断的最终指标与历史 M3U 完全一致：主指标=`1/12`、retain=`13/33`、F1_hop2=`9/75`；逐锚点实际位移冲突仍为 `257/456`。局部锚点概率改善没有转化为累计 retain 改善。

## 解释与判定

- **解释：** 旧 D_hop2 汇总同时包含 F1 设计内遗忘目标和未重叠题，无法单独解释为保留效用或目标遗忘；该切分也不自动构成受保护 retain 集。
- **observed：** 以上数值分别来自三个 run 的 `metrics.json`；12+12 子集在三组记录中均完整，M3/M3C/M3U 主指标依次为 `0/12`、`0/12`、`1/12`。M3U 的 retain=`13/33`，相对 M3=`16/33` 下降 3 题。
- **interpretation：** M3C 没有改善唯一主指标；M3U 多答对 1 题，但未达到预注册的 +2 门槛，且 retain 超出允许损失。M3U 的 F1_hop2 降至 `9/75`，显示设计内遗忘更深，但不足以抵消主指标和 retain 门槛未通过。单 seed 结果仅用于筛选。
- **verdict：** reject（仅指本轮预注册候选筛选未通过，不代表对 GSU 方向作普遍否定）。
- **cause_guess：** mechanism（诊断后部分支持）：聚合锚点修正能改善同批次平均直接概率，但仍允许个别锚点受损，且局部收益没有传递到累计 retain 评测。
- **next_action：** strengthen；不追加 M3C/M3U 多 seed，也不修改原 M3U。若补强预算获批，按 [`probe01`](probes/probe01.md) 另建实验卡验证 M3A 逐锚点约束；当前不启动新训练。
- **assets_salvaged：** 历史 M3 adapter/结果、F1 与 D_hop2 子集构造、quick 扫描、双口径评测、投影触发计数。

## Sanity checks

- 确认 12+12 子集互斥且完整覆盖 D_hop2；核对 F1 链位置映射和固定 forget split。
- 确认 M3C/M3U 的 seed、遗忘例顺序、锚点抽样序列、LoRA 配置和训练预算与 M3 一致；训练入口新增 `torch.manual_seed(42)`，补齐 LoRA 初始化 RNG 控制，差异记录在各 run 的 `patch.diff`。
- M3U 仅在 `g_r · delta < 0` 且 `||g_r||^2 > 1e-12` 时修正；投影只针对采样锚点聚合梯度的一阶局部变化，逐步记录并审查修正次数，最终准确率另行评估。
- 检查输出新旧文件隔离、无 adapter/断点混用；报告逐题命中数，避免 12 题小子集只写百分比。
- 三组正式比较和 M3U 诊断均完成五集与六集评测；未发现异常退出、非有限指标或 CUDA OOM，所有评测集计数符合冻结样本量。

## 后续动作

正式比较和 M3U 诊断均已归档。原 M3U 保持冻结；若结果支持新候选，只进入新实验卡定义的验证，不依据同一评测集临时增加方法组件。
