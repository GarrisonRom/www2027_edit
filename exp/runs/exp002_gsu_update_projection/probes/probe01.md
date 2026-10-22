# Strengthen Probe: probe01

exp_id: exp002_gsu_update_projection
date: 2026-10-19; diagnostic executed 2026-10-20
cycle: probe 1; one fixed-seed validation authorized in this workspace
failed_runs: run_20261019_1631_m3c_train_eval, run_20261019_1631_m3u_train_eval
diagnostic_runs: run_20261020_1107_m3u_diagnostic (failed launch), run_20261020_1115_m3u_diagnostic_rerun

## 失败事实

- M3 baseline: `D_hop2_no_F1_overlap=0/12`, `F1_hop2=25/75`, retain=`16/33`.
- M3C: primary=`0/12`, `F1_hop2=24/75`, retain=`17/33`; conditional projection occurred 37,573/44,688 tensor checks.
- M3U: primary=`1/12`, `F1_hop2=9/75`, retain=`13/33`; actual-update correction occurred on 112/114 steps.
- Both candidates failed the frozen gate: primary gain was below +2/12; M3U also exceeded the allowed retain loss by one item.
- Sources: the three run `metrics.json` files and `exp/results/gsu_projection_sixset.jsonl`.

## 逐步诊断结果

- The corrected M3U diagnostic completed all 114 updates with 112 corrections; direct anchor log-probability improved for `437/456` paired anchor observations, with mean paired improvement `+0.656008`.
- Individual first-order conflicts decreased from `434/456` raw AdamW displacements to `257/456` after the aggregate correction, so the aggregate constraint is locally useful but not an individual-anchor guarantee.
- The final metrics were identical to the historical M3U run: primary=`1/12`, retain=`13/33`, F1_hop2=`9/75`, K=`28/48`. The local probability improvement did not transfer to cumulative retain accuracy.
- Source: `runs/run_20261020_1115_m3u_diagnostic_rerun/diagnostics.jsonl` and `metrics.json`.

## 断链定位

1. Claim target: increase accuracy on the 12 non-F1-overlap D_hop2 questions while keeping F1 forgetting and retain within the frozen bounds. M3C gained 0/12; M3U gained 1/12.
2. M3U mechanism: project the AdamW update against the sum of four sampled retain-anchor gradients. The aggregate constraint triggered on 112/114 steps, but final retain was 13/33.
3. The diagnostic now records each sampled anchor's individual `g_i dot delta` and direct answer log-probability before/after correction. Aggregate correction reduces but does not remove individual conflicts, while final retain remains unchanged.
broken_link: partially confirmed aggregate-to-individual transfer failure; cumulative/nonlinear and out-of-sample effects remain unresolved, so this is not proof that per-anchor projection will pass the final gate.

## 三路反推

### 实验反推
- 调节变量扫描（收益出现/消失的边界）: do not sweep hyperparameters in this probe; keep the frozen data, seed, optimizer, and training budget.
- 干预/剂量-反应设计: on each M3A step, separately record each of the four sampled anchors' `g_i dot delta` before and after the update correction, plus projected/raw update norm ratio.
- 极端化与最有利 setting: use the same four-anchor draw as M3U; this directly tests aggregate-vs-individual constraints without changing anchor count.
- 失败案例归因: per-item retain predictions were not saved in exp002; capture them in the next run, but treat them as exploratory diagnostics.

### 理论/逻辑反推
- 收益成立的充分/必要条件，当前违反哪条: `sum_i g_i dot delta >= 0` does not imply `g_i dot delta >= 0` for every anchor; M3U enforces only the former.
- 上界/下界推导，headroom 估計: the primary has 12/12 items of raw accuracy headroom from M3; no theoretical upper bound proving zero headroom is available.
- 失败反置 / 对偶机制: replace one aggregate half-space with the intersection of per-anchor half-spaces while retaining the same AdamW proposal.
- 最简模型重现: with `g1=(1,0)`, `g2=(-1,2)`, and `delta=(-1,0)`, the aggregate dot is zero while `g1 dot delta` is negative. This demonstrates why the aggregate check can mask an individual conflict.

### 类比迁移
- 相邻问题的已知解法: constrained optimization projects a proposal onto the feasible intersection of linear inequalities; this is the direct mathematical analogue, not a novelty claim.
- 迁移前提是否成立: gradients are local linear approximations and all four constraints are feasible at zero update; higher-order and cumulative effects remain unguaranteed.

## 候选方法

### candidate M3A (GSU-v3)
- derivation（为什么它修复 broken_link）: compute a separate retain gradient `g_i` for each of the four sampled anchors, apply AdamW to obtain `delta`, then find the closest update `z` to `delta` subject to `g_i dot z >= 0` for every sampled anchor. Leave AdamW moments unchanged. This directly enforces the individual first-order conditions that M3U's aggregate constraint omits.
- method_version: `docs/method_versions/GSU-v3.md` (implemented and validated once; frozen gate failed)
- expected_pattern: corrected sampled-anchor dots are all nonnegative up to numerical tolerance; retain is at least 14/33; primary gains at least +2/12 over M3; F1_hop2 worsens by no more than 2/75. The original exp002 gate remains unchanged.
- discriminating_test（M′ / 旧方法 / 保守解释 三种结果如何区分）: M3A vs frozen M3 and M3U, seed=42, same 38 forget examples, same four-anchor draws, same LoRA and optimizer settings. If M3A removes individual negative dots and improves retain while meeting the original primary gate, per-anchor cancellation is supported. If individual dots were not negative or metrics remain below gate, this mechanism is not supported. If retain recovers but primary does not meet +2/12, record the trade-off and do not advance to multi-seed.
- budget: one seed, one candidate, 3 epochs / 114 updates, no hyperparameter search; estimated wall time about 10 minutes on the same GPU based on exp002 runs. No multi-seed confirmation within this probe.
- kill_rule: do not proceed to multi-seed unless all original exp002 thresholds pass; stop the candidate for non-finite values, invalid/empty evaluation sets, or adapter mismatch. No threshold changes or rescue tuning within this probe.
- run_ids: `run_20261020_1332_m3a_train_diagnostic`; the M3U diagnostic is `run_20261020_1115_m3u_diagnostic_rerun`
- verdict: partial mechanism support, candidate rejected by frozen gate; primary improved to `3/12` but retain fell to `11/33`

## 本轮结论
- claim 是否仍有可争取空间: yes; the frozen primary is 0/12 for M3 and has 11/12 headroom, but this is only a finite test-set ceiling, not evidence of achievable gain.
- 下一轮候选或收敛理由: M3A completed its fixed one-seed card and failed the retain gate. Harvest this result and redirect to retain-anchor coverage or cumulative protection; do not run multi-seed or add another projection variant under probe01.

## gate 状态
- strengthen_budget 剩余: the allocated one fixed-seed M3A validation is consumed; multi-seed budget remains unallocated
- 是否允许进入 pivot（收缩/转向/harvest）: yes; M3A is harvested as a partial mechanism result and the projection branch is closed for now
