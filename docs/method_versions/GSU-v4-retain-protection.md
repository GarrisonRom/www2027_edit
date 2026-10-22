# Method Version: GSU-v4-retain-protection

date: 2026-10-20
parent: GSU-v3
status: diagnostic candidates; single-seed screen completed and not advanced

## what_changed

- **COV16：** 将 M3A 的每步 4 个独立 retain 锚点扩大为 16 个新抽样锚点；把 AdamW 实际位移投影到这 16 个逐锚点半空间的交集。
- **CUM16：** 训练开始固定 16 个 retain 锚点；每个更新为其当前答案对数概率相对训练初始值建立线性累计下限，并把实际位移投影到这些约束的交集。
- 使用系数空间 Dykstra 求解半空间交集，保留 AdamW 动量状态不变；日志保存原始/修正后的位移、逐锚点直接答案对数概率和实际约束残差。

## why

GSU-v3 M3A 的局部逐锚点约束改善了 `439/456` 个采样锚点相对原始 AdamW 的直接答案对数概率，但最终 retain 只有 `11/33`。这留下两个可区分的机制解释：每步保护的锚点太少，或同一锚点在多步更新中反复累积损伤。exp004 分别用更宽覆盖和固定累计下限验证这两个解释。

## reproducibility

- experiment card: `exp/runs/exp004_gsu_retain_protection/card.md`
- code path: `exp/train/diagnose_gsu_retain_protection.py`
- fixed model/data/training settings: card and each run `config.json`
- first runs: `run_20261020_cov16`, `run_20261020_cum16`

## constraints and limits

这是诊断候选，不是已验证的最终 GSU 方法。COV16 不提供跨步累计下限；CUM16 只覆盖 16/33 条 retain 样本。若局部直接答案概率改善不能传递到冻结 retain 指标，不能把代理约束解释为保留能力保证。

## validation result

- COV16 (`run_20261020_cov16_rerun`) covered all `33/33` retain anchors across updates and reached primary `9/12`, retain `20/33`, and F1_hop2 `29/75`. Direct anchor log-probability improved over raw AdamW on `1788/1824` paired observations, but `9/1824` actual writeback residuals were below `-1e-5` and F1_hop2 missed the frozen `27/75` gate.
- CUM16 (`run_20261020_cum16`) kept a persistent `16/16` memory bank and reached primary `9/12`, retain `23/33`, and F1_hop2 `41/75`. The independently recomputed cumulative residual fell below `-1e-5` on `4/1824` observations, and the worst fixed-anchor actual log-probability gap relative to initialization was `-25.5761`; the linearized floor did not provide actual cumulative protection.
- Verdict: partial for the coverage diagnosis and reject for the linearized cumulative-protection mechanism. Neither candidate enters multi-seed confirmation.

The first COV16 run (`run_20261020_cov16`) is retained as a failed numerical-convergence field record: it stopped after 57 updates at the fixed 300-cycle limit. The repaired rerun raised only the fixed solver cycle budget to 3000. The CUM16 runtime violation field was audited post hoc because the original expression compared the dot to zero instead of subtracting the cumulative target; archived snapshots remain unchanged.
