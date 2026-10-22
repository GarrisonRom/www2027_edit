# exp004 post-hoc audit

## Source runs

- Failed numerical field record: `run_20261020_cov16` — stopped at 57/114 updates because 300 Dykstra cycles did not reach the fixed coefficient convergence criterion; no final metrics used.
- COV16 repaired rerun: `run_20261020_cov16_rerun` — 114/114 updates, solver failures 0.
- CUM16: `run_20261020_cum16` — 114/114 updates, solver failures 0.

## Frozen final metrics

| run | primary `D_hop2_no_F1_overlap` | retain | F1_hop2 | K | D_hop2_F1_overlap |
|---|---:|---:|---:|---:|---:|
| COV16 | 9/12 | 20/33 | 29/75 | 11/48 | 5/12 |
| CUM16 | 9/12 | 23/33 | 41/75 | 22/48 | 6/12 |

## Mechanism checks

- COV16 sampled 16 anchors per update and observed all 33 retain identities. Direct anchor logprob improved relative to raw AdamW on 1788/1824 paired observations; mean step delta changed from -1.770595 to +0.026811.
- CUM16 reused one 16-anchor memory bank for all 114 updates. Direct anchor logprob improved relative to raw AdamW on 1731/1824 paired observations; mean step delta changed from -2.522097 to +0.001722.
- COV16 actual writeback constraint residual: minimum -3.62e-05; 9/1824 below -1e-5.
- CUM16 actual cumulative residual (`final_anchor_dot_delta - cumulative_target`): minimum -4.96e-05; 4/1824 below -1e-5.
- CUM16 fixed-anchor actual logprob gap relative to initialization: minimum -25.5761. The linearized floor did not preserve the nonlinear trajectory.

## Decision

COV16 provides partial evidence that anchor coverage contributes to the M3A retain failure, but it misses the frozen F1_hop2 and actual-residual gates. CUM16 rejects the interpretation that a first-order cumulative floor is sufficient for real retain protection. Neither candidate enters multi-seed confirmation. Future trajectory-aware protection, if authorized, requires a new experiment card and a direct actual-logprob criterion.
