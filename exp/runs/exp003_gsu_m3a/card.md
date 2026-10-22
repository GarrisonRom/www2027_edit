# exp003：GSU-v3 M3A 逐锚点更新约束

**状态：** 固定 seed 的单候选验证已完成；M3A 未通过 retain 门槛，不进入多 seed。
**假设：** H-GSU-M3A-1。
**方法版本：** GSU-v3 M3A，见 [`GSU-v3.md`](../../../docs/method_versions/GSU-v3.md)。

## hypothesis

M3U 的聚合锚点约束改善了采样锚点的平均局部对数概率，但仍允许单个锚点受损。将 AdamW 实际位移投影到四个独立锚点半空间的交集，能够减少逐锚点冲突，并在保持 M3 遗忘效果的同时恢复 retain 和非 F1 重叠 D_hop2 表现。

## expected_pattern

相同的四个采样锚点下，M3A 修正后的每个 `g_i · delta` 应在数值容差内非负；直接锚点答案对数概率下降数应少于 M3U；如果断链定位正确，retain 应至少达到 `14/33`，主指标应达到 `2/12` 或更高，且 F1_hop2 不超过 M3 的 `27/75`。

## frozen comparison

- **模型与数据：** Qwen2.5-1.5B-Instruct；FaithUn 数据、固定 split、G2 遗忘集和 33 条 retain 样本不变。
- **训练：** seed=42、LoRA r=8/alpha=16、dropout=0、lr=5e-5、AdamW、3 epochs、38 个遗忘例、每步 4 个锚点。
- **M3 baseline：** `D_hop2_no_F1_overlap=0/12`、retain=`16/33`、F1_hop2=`25/75`。
- **M3U baseline：** `D_hop2_no_F1_overlap=1/12`、retain=`13/33`、F1_hop2=`9/75`；来源 `run_20261019_1631_m3u_train_eval`。
- **唯一主指标：** `D_hop2_no_F1_overlap`（12 题）。
- **关键次指标：** `D_hop2_F1_overlap`、F1_hop2、retain、K、same_object 及逐锚点直接答案对数概率。

## null_control

当 AdamW 原始位移已经满足所有 `g_i · delta >= 0` 时，M3A 必须保持恒等更新；诊断记录修正前后位移与对数概率，确认无冲突步没有被额外修改。原始 M3U 和 M3 的训练规则不改写。

## budget and gate

- 单 seed、单候选、114 个更新；不搜索学习率、锚点数量、投影阈值或修正强度。
- M3A 只有在主指标至少比 M3 多 `2/12`、retain 不低于 `14/33`、F1_hop2 不高于 `27/75` 时才进入多 seed。
- 任意非有限值、投影求解失败、评测集数量异常或适配器不一致都使本 run 失效；不在本 run 内调参救援。
- 主动集求解使用 4 个约束的 KKT 枚举，不新增第三方依赖；AdamW 动量状态不因事后修正改变。

## observed

- Source: [`run_20261020_1332_m3a_train_diagnostic`](runs/run_20261020_1332_m3a_train_diagnostic/metrics.json).
- Primary `D_hop2_no_F1_overlap`: `3/12`; M3 baseline `0/12`.
- retain: `11/33`; M3 baseline `16/33`; M3U baseline `13/33`.
- F1_hop2: `9/75`; K: `21/48`; same_object: `6/24`.
- Direct anchor log-probability improved after M3A versus raw AdamW on `439/456` paired observations; mean paired improvement `+1.716301`.
- The mathematical active-set residual minimum was `-2.44e-15`; actual writeback residual minimum was `-9.35e-6`, with `38/456` below `-1e-6` and none below `-1e-5`.

## interpretation

M3A supports the local mechanism diagnosis: individual half-space correction improves the sampled anchors and raises the primary subset to `3/12`. It does not support the full method hypothesis because retain falls to `11/33`, below the frozen gate and below M3U. The local first-order condition does not transfer to cumulative retain performance in this setting.

## verdict

reject for the frozen single-seed advancement gate.

## next_action

harvest the result and redirect future work toward retain-anchor coverage or cumulative protection. Do not start multi-seed confirmation or add another projection variant under this card.

## sanity_checks

- Verify 114/114 update records and 4 anchors per step.
- Verify all final individual constraint residuals are nonnegative within `1e-6`.
- Verify raw feasible steps receive zero correction.
- Verify final five-set and six-set counts match the frozen sample sizes.
- Compare direct log-probability changes before AdamW, after raw AdamW, and after M3A correction.
