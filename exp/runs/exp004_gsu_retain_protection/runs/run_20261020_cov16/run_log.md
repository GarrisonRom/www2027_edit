# Run Log: GSU-v4 COV16 retain coverage diagnostic

exp_id: exp004_gsu_retain_protection
assumption_id: H-GSU-RP-1
method_version: GSU-v4-retain-protection COV16
operator: Codex
started: 2026-10-20 14:47:03 +08:00
ended: 2026-10-20 15:06:11 +08:00
status: failed

## purpose

Test whether the four-anchor M3A result is limited by per-update retain-anchor coverage. COV16 samples 16 fresh retain anchors at every update and projects the realized AdamW displacement onto the intersection of their individual local half-spaces. The original M3/M3U/M3A update implementations remain unchanged.

## preregistered expectation

- Every update has 16 sampled anchors and the cross-step unique coverage is higher than M3A's four-anchor schedule.
- Corrected first-order anchor displacement constraints are satisfied within the run card tolerance.
- Direct anchor log-probability damage and final retain improve relative to M3A if coverage is the dominant missing mechanism.
- The frozen advancement gate is primary at least `3/12`, retain at least `14/33`, and F1_hop2 at most `27/75`.

## code

- repo / url: local workspace / https://github.com/GarrisonRom/www2027_edit
- git_commit: 6ec831cfa25e07fcdcc908459f76a678c45434fb
- working_tree: dirty; see `patch.diff`, `working_tree_status.txt`, and the exact script snapshot
- script: `exp/train/diagnose_gsu_retain_protection.py`
- script_sha256: `19B264DE88617B3AA04B07D8623E1350190679DC6A19C816648CC4BBB3CCB70`
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
- split rule: frozen G2 forget examples (38) and first 33 retain samples; 16 fresh anchors per update

## reproducibility

- seed: `42` for Python and PyTorch
- forget order: `random.Random(42 + epoch).shuffle`
- anchor order: `random.Random(42).sample`, 16 fresh anchors per update
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

pending; fill from `metrics.json` and `diagnostics.jsonl` after completion.

## anomalies

- The run stopped at epoch 2, step 20 after 300 Dykstra cycles; the minimum solver residual was `-3.5620357330401475e-08`, just outside the `1e-8` solver tolerance. This was a numerical convergence failure, not a CUDA OOM, non-finite loss, or data error.
- The run produced 57 complete update records and one epoch checkpoint, but no final adapter or final metrics. These partial artifacts are retained for diagnosis and are not used as experiment results.
- A new run id is required after the solver repair; this run is not resumed or overwritten.

## status

Prepared before launch. The run directory is immutable after completion; any rerun must use a new run id.
