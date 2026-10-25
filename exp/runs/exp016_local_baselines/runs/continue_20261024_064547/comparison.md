# Local C-group and baseline comparison

Qwen2.5-1.5B, seed42; provisional descriptive ranking, no significance claim.
KLUE is a separate native paper reimplementation track; its retain33 were trained.
C variants use F1 guidance; C0 omits the F0 guard. Data access and optimization differ.
MMLU/PPL pending. Missing/failed methods have no performance row.

| Method | All89 recall ↓ | Covered /75 ↓ | F0 /33 ↓ | F1 /5 ↓ | Retain /33 ↑ | Heldout |
|---|---:|---:|---:|---:|---:|---|
| PDU | 21.72% | 0 | 7 | 1 | 13 | True |
| GA | 22.85% | 22 | 6 | 3 | 13 | True |
| C1 | 27.72% | 0 | 12 | 0 | 12 | True |
| C4 | 29.59% | 7 | 12 | 0 | 10 | True |
| C3R | 29.96% | 7 | 12 | 0 | 10 | True |
| C2 | 29.96% | 7 | 12 | 0 | 10 | True |
| C3 | 30.71% | 7 | 13 | 0 | 10 | True |
| C3D | 31.46% | 9 | 13 | 0 | 10 | True |
| Cg | 32.58% | 9 | 12 | 1 | 10 | True |
| CEU | 35.21% | 37 | 9 | 5 | 18 | True |
| C0 | 44.94% | 52 | 8 | 5 | 15 | True |
| DPOmis | 59.18% | 58 | 11 | 5 | 22 | True |
| UNDIAL | 62.17% | 63 | 15 | 5 | 23 | True |
| RMU | 64.42% | 71 | 21 | 5 | 26 | True |
| NPO | 65.17% | 69 | 11 | 5 | 26 | True |
| KLUE | 65.54% | 73 | 8 | 5 | 28 | False |
| DPOrej | 67.04% | 70 | 14 | 5 | 24 | True |
| GradDiff | 74.91% | 74 | 13 | 5 | 24 | True |
| SimNPO | 75.66% | 75 | 28 | 5 | 24 | True |
| SatImp | 75.66% | 75 | 28 | 5 | 24 | True |
| WGA | 75.66% | 75 | 26 | 5 | 24 | True |

## Raw sources

- PDU: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp016_local_baselines\runs\continue_20261024_064547\PDU_formal\metrics.json
- GA: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp016_local_baselines\runs\local_20261023_233138\GA_formal\metrics.json
- C1: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp010_gsu_step_routing\runs\batch_20261023_092800_Cgroup_restart1\C1\metrics.json
- C4: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp010_gsu_step_routing\runs\local_continue_20261023_205325\C4\metrics.json
- C3R: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp010_gsu_step_routing\runs\local_continue_20261023_205325\C3R\metrics.json
- C2: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp010_gsu_step_routing\runs\batch_20261023_092800_Cgroup_restart1\C2\metrics.json
- C3: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp010_gsu_step_routing\runs\batch_20261023_092800_Cgroup_restart1\C3\metrics.json
- C3D: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp010_gsu_step_routing\runs\local_continue_20261023_205325\C3D\metrics.json
- Cg: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp010_gsu_step_routing\runs\batch_20261023_092800_Cgroup_restart1\Cg\metrics.json
- CEU: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp016_local_baselines\runs\continue_20261024_064547\CEU_formal\metrics.json
- C0: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp010_gsu_step_routing\runs\batch_20261023_092800_Cgroup_restart1\C0\metrics.json
- DPOmis: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp016_local_baselines\runs\continue_20261024_064547\DPOmis_formal\metrics.json
- UNDIAL: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp016_local_baselines\runs\continue_20261024_064547\UNDIAL_formal\metrics.json
- RMU: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp016_local_baselines\runs\local_20261023_233138\RMU_formal\metrics.json
- NPO: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp016_local_baselines\runs\local_20261023_233138\NPO_formal\metrics.json
- KLUE: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp016_local_baselines\runs\local_20261023_233138\KLUE_formal\metrics.json
- DPOrej: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp016_local_baselines\runs\continue_20261024_064547\DPOrej_formal\metrics.json
- GradDiff: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp016_local_baselines\runs\local_20261023_233138\GradDiff_formal\metrics.json
- SimNPO: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp016_local_baselines\runs\local_20261023_233138\SimNPO_formal\metrics.json
- SatImp: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp016_local_baselines\runs\continue_20261024_064547\SatImp_formal\metrics.json
- WGA: C:\Users\RoboSAMA\Documents\kimi\Workspaces\蓝恩鹏\exp\runs\exp016_local_baselines\runs\continue_20261024_064547\WGA_formal\metrics.json
