# Qwen2.5-3B-Instruct 扩展验证记录

日期：2026-10-17。该记录沿用 FaithUn 的 MCQA 三选一协议和已有 GA-LoRA 训练脚本，使用本地 `Qwen2.5-3B-Instruct`，不覆盖 1.5B 结果。

## 探测与选集

- 预实验：133 个簇；hop1 一致率均值 `0.739`；hop2 整体正确率 `0.610`；hop1 正确条件下 latent multi-hop 为 `0.625`，高于随机地板 `0.333`。
- 候选事实：61 个；3B 掌握 45 个（`tau=0.75`）。
- 3B 活跃路径：50 条；预算 `B=5` 的贪心 F1 覆盖 19 条。
- F1：`USA/capital`、`USA/currency`、`USA/continent`、`Los Angeles/country`、`Marian Shields Robinson/country of citizenship`。

## 五组结果（采用 clean run）

| 组 | hop1_unlearn ↓ | hop1_test ↓ | hop2 ↓ | same_object ↑ | retain ↑ |
|---|---:|---:|---:|---:|---:|
| orig | 0.697 | 0.707 | 0.633 | 0.854 | 0.818 |
| G1 只忘 F0 | 0.121 | 0.141 | 0.255 | 0.229 | 0.364 |
| G2 F0 + 3B 贪心 F1 | 0.152 | 0.131 | **0.236** | 0.188 | 0.333 |
| G3 F0 + 全候选 | 0.212 | 0.242 | **0.157** | 0.222 | 0.545 |
| G4 F0 + 随机 5 | 0.182 | 0.182 | **0.236** | 0.215 | 0.364 |
| G5 只忘 F1 | 0.667 | 0.616 | 0.461 | 0.632 | 0.636 |

在本次小规模、单次 seed 对照中，G2 的 hop2 恢复率比 G1 低 1.9 个百分点，并与 G4 持平；G3 使用全部候选事实，hop2 最低但保留效用仍明显受损。因此当前 3B 结果支持“扩展有用”的方向性判断，但不能据此宣称探测驱动选择显著优于随机，仍需多 seed 与逐题配对检验。所有含 F0 的组 retaining 均低于 orig，保留效用仍是主要限制。

## 产物

- 预实验：`pre_experiment_summary_3B.json`、`pre_experiment_records_3B.jsonl`
- 选集：`3B_v1_candidates.json`、`3B_v1_active_paths.json`、`3B_v1_greedy_curve.json`、`3B_v1_F1.json`
- 适配器：`exp/adapters/3B_v2_clean_G1` 至 `exp/adapters/3B_v2_clean_G5`
- 评测：`eval_results_3B_v2_clean.json`、`eval_results_3B_v2_clean.jsonl`
- 旧现场：`3B_v1_*` 保留在 `exp/results/`，因 PEFT 重复适配器警告不作为最终数字
- 运行档案：`exp/runs/exp001_3B_extension/`
