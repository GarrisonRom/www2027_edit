# Run Log: GSU-v4 CUM16 cumulative retain protection diagnostic

exp_id: exp004_gsu_retain_protection
assumption_id: H-GSU-RP-2
method_version: GSU-v4-retain-protection CUM16
operator: Codex
started: 2026-10-20 15:39:26 +08:00
ended: 2026-10-20 16:19:34 +08:00
status: failed_sanity_gate

## purpose

Test whether M3A's local improvement fails because the same retain knowledge is damaged across multiple updates. CUM16 fixes 16 retain anchors at initialization and projects every realized AdamW displacement against each anchor's linearized cumulative floor relative to its initial answer log probability. The original M3/M3U/M3A update implementations remain unchanged.

## preregistered expectation

- The same 16 anchor identities occur in all 114 update records.
- Every corrected cumulative first-order constraint is satisfied within the run card tolerance.
- Fixed-anchor actual answer log-probability drops relative to initialization and final retain improve relative to M3A if cumulative damage is the dominant missing mechanism.
- The frozen advancement gate is primary at least `3/12`, retain at least `14/33`, and F1_hop2 at most `27/75`.

## code

- repo / url: local workspace / https://github.com/GarrisonRom/www2027_edit
- git_commit: 6ec831cfa25e07fcdcc908459f76a678c45434fb
- working_tree: dirty; see `patch.diff`, `working_tree_status.txt`, and the exact script snapshot
- script: `exp/train/diagnose_gsu_retain_protection.py`
- script_sha256: `A342F1AABF78DFB3B0366CF98B3E61B912083BCAE0D601E0CB2DB04AE4DFE793`
- config snapshot: `config.json`

## environment

- language/runtime: Python 3.10.18 at `D:\Anaconda\envs\cuda126_env\python.exe`
- key dependencies: torch 2.6.0+cu126; transformers 5.14.1; peft 0.20.0; accelerate 1.14.0; safetensors 0.8.0; numpy 1.26.4
- CUDA/driver or platform: CUDA 12.6; native Windows
- hardware: NVIDIA GeForce RTX 4060 Ti 16GB

## launch

- command: see `config.json`
- expected updates: 114
- no hyperparameter search; no changes to original M3/M3U/M3A

## data

- dataset: FaithUn local dataset, `data/faithun/FaithUn_Dataset.json`
- dataset_sha256: `D570CC3E30C083357CA45AB4DD9DF807ABBD800F51A657D89629E4FE7B3ADBB0`
- forget split sha256: `315312DD50B69993F3E90796E7162A9ADE22B771F3AA9EC5F570CEAAAD815428`
- retain split sha256: `8680043FE5C1F4F0517D7377664677EE225CD9EEBCB72D542C7709B1F5B6E214`
- split rule: frozen G2 forget examples (38) and first 33 retain samples; one persistent 16-anchor memory bank

## reproducibility

- seed: `42` for Python and PyTorch
- forget order: `random.Random(42 + epoch).shuffle`
- memory bank: `random.Random(42).sample` once before training; reused each update
- deterministic flags: PyTorch deterministic algorithms not enabled; backend defaults unchanged
- correction: coefficient-space Dykstra projection; AdamW moments unchanged

## outputs

- raw step diagnostics: `diagnostics.jsonl`
- epoch summaries: `epoch_metrics.jsonl`
- final metrics: `metrics.json`
- checkpoints: `checkpoints/`
- final adapter: `adapter/`
- stdout: `stdout.log`
- stderr: `stderr.log`

## raw_metrics

- update steps: `114/114`; projection failures: `0`; Dykstra failures: `0`
- persistent memory bank: `16/16` anchor identities; all sampled identities: `16`
- primary `D_hop2_no_F1_overlap`: `9/12`; retain: `23/33`; F1_hop2: `41/75`; K: `22/48`; D_hop2_F1_overlap: `6/12`
- raw direct anchor log-probability decreases: `1778/1824`; final step decreases: `902/1824`; final better than raw AdamW: `1731/1824`
- mean direct per-step anchor log-probability delta: raw `-2.522097`; final `+0.001722`
- mean correction-to-raw-update norm ratio: `0.259820`; maximum `2.177987`
- solver residual minimum: `-1.00e-08`; actual cumulative residual minimum (`final_anchor_dot_delta - cumulative_target`): `-4.96e-05`
- actual cumulative residuals below `-1e-5`: `4/1824`; final actual answer log-probability gap relative to initialization: minimum `-25.5761099`

## anomalies

- The run completed all training and evaluation stages without solver failure, but the actual cumulative residual sanity gate failed: 4 post-writeback residuals were below `-1e-5`.
- The original run-time field `final_anchor_violation_tol` compared `final_anchor_dot_delta` with zero instead of subtracting the CUM16 target. The values above were independently recomputed from `diagnostics.jsonl`; the logging expression was corrected after the run for future experiments, without changing this run's snapshot or metrics.
- The linearized cumulative floor did not prevent nonlinear actual answer log-probability loss: the worst fixed memory anchor was `25.5761` below its initial log probability by the final evaluation trajectory.

## status

Training and evaluation completed, but the run failed the predeclared actual-cumulative-residual sanity gate. The run directory is immutable; any further rerun must use a new run id.
