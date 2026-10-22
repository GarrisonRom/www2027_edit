# exp005：GSU 实际 logprob 反馈保护

**状态：** 固定 seed 机制筛选已完成；运行 sanity 通过，但候选未通过推进门。
**假设：** H-GSU-FB-1。
**方法版本：** GSU-v5-actual-feedback，见 [`GSU-v5-actual-feedback.md`](../../../docs/method_versions/GSU-v5-actual-feedback.md)。

## hypothesis

exp004 的主要断点是线性约束与真实非线性 logprob 之间的反馈缺失。保持 COV16 的 16 锚点覆盖，若在 AdamW proposal 后根据实际锚点 logprob 下降进行有限反馈修正，可以降低真实锚点下降并在不牺牲遗忘轴的情况下维持或提高最终 retain。

## expected_pattern

- FB16 每步抽取 16 条新锚点，跨 114 步覆盖 `33/33` 条 retain 身份。
- 相对 COV16 的 `866/1824` 个最终直接 logprob 下降观察，FB16 的下降数应减少至少 10 个百分点；反馈后的风险数应随轮次减少。
- 每步总修正位移比例由预算硬限制在 `0.30` 以内，最多进行 2 轮反馈；无风险时应为 no-op。
- 最终 retain 至少达到 COV16 的 `20/33`，主指标至少达到 `3/12`，且 F1_hop2 不超过 COV16 的 `29/75`。同时观察遗忘相关六集指标，不能只看 retain。

## null_control

如果 AdamW proposal 后 16 个锚点的实际 logprob 变化都不低于 `-1e-4`，反馈控制器必须不修改参数；日志记录风险数、反馈轮数和修正范数，可验证该 no-op 条件。合成 Gram 矩阵测试验证最小范数补偿和 trust-region 缩放满足约束及预算。

## frozen comparison

- **模型与数据：** Qwen2.5-1.5B-Instruct；FaithUn 固定数据、split、G2 遗忘集和 33 条 retain 样本。
- **训练：** seed=`42`、LoRA `r=8/alpha=16/dropout=0`、AdamW、lr=`5e-5`、3 epochs、38 个遗忘例、114 个更新。
- **候选：** FB16 每步新抽 16 条 retain 锚点；反馈阈值 `1e-4`，最多 2 轮，修正预算为 raw AdamW 位移的 `0.30` 倍。
- **唯一主指标：** `D_hop2_no_F1_overlap`（12 题，越高越好）。
- **关键次指标：** retain（33 题）、`D_hop2_F1_overlap`、F1_hop2、K、same_object、逐步实际锚点 logprob、反馈风险数、实际残差、覆盖数和修正位移比例。
- **冻结 baselines：** M3=`0/12` 主指标、retain=`16/33`；M3A=`3/12`、`11/33`；COV16=`9/12`、`20/33`、F1_hop2=`29/75`；CUM16=`9/12`、`23/33`、F1_hop2=`41/75`。

## budget and gate

- 1 个固定 seed、114 个更新；不做超参数搜索，不修改 M3/M3U/M3A/COV16/CUM16 历史结果。
- 非有限值、反馈 solver 不收敛、实际更新数不足、评测样本数异常或任一步修正比例超过 `0.30 + 1e-6`，当前 run 记为 failed。
- 进入后续多 seed 的门：主指标至少 `3/12`、retain 至少 `20/33`、F1_hop2 不高于 `29/75`，且直接 logprob 下降率比 COV16 至少降低 10 个百分点；否则只保留为单 seed 机制证据。

## observed

来源 [`run_20261020_1714_fb16`](runs/run_20261020_1714_fb16/metrics.json) 及其 `diagnostics.jsonl` 独立审计：

- 114/114 更新完成，16 条新锚点跨步覆盖 `33/33` 个 retain 身份；反馈 solver 失败 `0`，非有限值 `0`。
- 主指标 `D_hop2_no_F1_overlap=3/12`，retain `19/33`，F1_hop2 `22/75`，K `9/48`，D_so `3/24`，五集 same_object 均值 `0.256944`。
- raw 实际锚点 logprob 下降 `1780/1824`，反馈后下降 `838/1824`；`1750/1824` 个反馈后观察优于 raw AdamW。raw 均值为 `-1.155235`，反馈后均值为 `-0.053329`。
- 相对 COV16 的 `866/1824` 个最终下降，FB16 只减少 `28/1824`，即 `1.54` 个百分点；COV16 的反馈后均值为 `+0.026811`。
- 反馈轮次共 `227`，其中 113/114 步使用两轮；平均修正比例 `0.215843`，最大 `0.300000102`，未超过 `0.300001` 审计余量。

## sanity_checks

- 通过：更新数 `114/114`、锚点覆盖 `33/33`、反馈轮次不超过 2、修正比例不超过 `0.30+1e-6`、Dykstra 失败 `0`、非有限值 `0`。
- 通过：五集和六集样本数符合冻结协议，adapter 与 ep1–ep3 checkpoint 均存在。
- 通过：无风险时控制器不进入反馈；全 run 仅 1 个步骤只使用一轮，其他步骤按风险进入反馈。
- 未通过推进机制门：最终直接下降率相对 COV16 仅改善 `1.54` 个百分点，低于事前冻结的 `10` 个百分点。
- 未通过最终保持门：retain 为 `19/33`，低于事前冻结的 `20/33`。主指标和 F1_hop2 分别满足 `3/12` 与 `22/75` 门槛，但不能抵消保持门失败。

## verdict

reject（针对本候选的推进门，不否定“实际反馈可修正 raw proposal”的局部观察）。实际反馈显著改善了相对 raw AdamW 的当步 logprob，但没有超过 COV16 的最终轨迹和 retain 点估计；单 seed 不支持继续多 seed。

## next_action

harvest/redirect：保留 FB16 的风险反馈诊断和预算证据；停止在本卡内继续调阈值、轮数或预算。若继续 GSU，应先解释第三 epoch 的累积下降和当前锚点覆盖到最终 retain 的断点，再另建实验卡。

## assets_salvaged

exp004 的 COV16/CUM16 逐步日志、Dykstra solver、覆盖统计、直接/实际 logprob 诊断和固定评测协议。
