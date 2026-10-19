# exp002：GSU 更新方向补强

**状态：** 预注册完成，尚未运行。
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

## 解释与判定

- **解释：** 旧 D_hop2 汇总同时包含 F1 设计内遗忘目标和未重叠题，无法单独解释为保留效用或目标遗忘；该切分也不自动构成受保护 retain 集。
- **verdict：** pending。M3C/M3U 尚未运行，不更新效果 claim。
- **cause_guess：** 待实验判定；当前仅有待检验的优化机制假设（梯度条件与 AdamW 实际位移）及已确认的指标重叠问题。
- **next_action：** 按冻结协议运行一次 seed；只有通过预设筛选门才进入多 seed 确认。
- **assets_salvaged：** 历史 M3 adapter/结果、F1 与 D_hop2 子集构造、quick 扫描、双口径评测、投影触发计数。

## Sanity checks

- 确认 12+12 子集互斥且完整覆盖 D_hop2；核对 F1 链位置映射和固定 forget split。
- 确认 M3C/M3U 的 seed、遗忘例顺序、锚点抽样序列、LoRA 配置和训练预算与 M3 一致。
- M3U 仅在 `g_r · delta < 0` 且 `||g_r||^2 > 1e-12` 时修正；投影只针对采样锚点聚合梯度的一阶局部变化，逐步记录并审查修正次数，最终准确率另行评估。
- 检查输出新旧文件隔离、无 adapter/断点混用；报告逐题命中数，避免 12 题小子集只写百分比。

## 后续动作

运行后分别填写 `observed`（事实、run_id、原始产物）与 `interpretation`，再更新 `docs/研究证据总账.md`。若结果支持候选，只进入预先定义的多 seed 确认；若未支持，保留负结果，不依据同一评测集临时增加方法组件。
