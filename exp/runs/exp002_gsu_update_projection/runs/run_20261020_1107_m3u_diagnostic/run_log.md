# Run Log: original M3U diagnostic replay

exp_id: exp002_gsu_update_projection
assumption_id: diagnostic-only M3U anchor-probability audit
method_version: original M3U AdamW actual-update correction; no algorithm change
operator: Codex
started: 2026-10-20 11:07:11 +08:00
ended: 2026-10-20 11:11:24 +08:00
status: failed

## purpose

Replay the original M3U update path and directly measure whether its post-hoc correction improves the actual log probability of the same sampled anchor answers. The run records the AdamW proposal, the correction, per-anchor displacement dots, and anchor log-probability changes before and after correction. Diagnostic passes do not change the update rule or optimizer state.

## code

- repo / url: local workspace / https://github.com/GarrisonRom/www2027_edit
- git_commit: 6ec831cfa25e07fcdcc908459f76a678c45434fb
- working tree: dirty; see `patch.diff` and `working_tree_status.txt`
- script: `exp/train/diagnose_gsu_m3u.py`
- script_sha256: `10C3B33104D34413F9B1C062D25791005264EB05A1210F929080B2CD5B63AE29`
- config snapshot: `config.json`

## environment

- language/runtime: Python 3.10.18 at `D:\Anaconda\envs\cuda126_env\python.exe`
- key dependencies: torch 2.6.0+cu126; transformers 5.14.1; peft 0.20.0; accelerate 1.14.0; safetensors 0.8.0; numpy 1.26.4
- CUDA/driver or platform: CUDA 12.6; native Windows
- hardware: NVIDIA GeForce RTX 4060 Ti 16GB

## launch

- command: see `config.json`
- config snapshot: `config.json`
- expected updates: 114

## data

- dataset: FaithUn local dataset, `data/faithun/FaithUn_Dataset.json`
- dataset_sha256: `D570CC3E30C083357CA45AB4DD9DF807ABBD800F51A657D89629E4FE7B3ADBB0`
- forget split sha256: `315312DD50B69993F3E90796E7162A9ADE22B771F3AA9EC5F570CEAAAD815428`
- retain split sha256: `8680043FE5C1F4F0517D7377664677EE225CD9EEBCB72D542C7709B1F5B6E214`
- split rule: frozen G2 forget examples (38) and first 33 retain indices; four anchors sampled per update

## reproducibility

- seed: `42` for Python and PyTorch
- forget order: `random.Random(42 + epoch).shuffle`
- anchor order: `random.Random(42).sample`, matching original M3U
- deterministic flags: PyTorch deterministic algorithms not enabled; backend defaults unchanged
- diagnostic passes: per-anchor gradients and post-update scoring do not call `opt.step()` or modify optimizer state; LoRA dropout is 0.0

## outputs

- raw step diagnostics: `diagnostics.jsonl`
- epoch summaries: `epoch_metrics.jsonl`
- final metrics: `metrics.json`
- checkpoints: `checkpoints/`
- final adapter: `adapter/`
- stdout: `stdout.log`
- stderr: `stderr.log`

## pre-registered decision rule for this diagnostic

- First compare `anchor_lp_after_correction` with `anchor_lp_after_raw_adamw` on the same sampled anchors.
- Treat the correction as locally supported only if it reduces the number and magnitude of negative direct log-probability changes without an implausibly large correction-to-raw-update norm ratio.
- Separately inspect whether aggregate and individual-anchor first-order dots agree with the direct log-probability changes.
- Do not modify M3U or implement a new projection rule from a proxy-only improvement; a change requires direct anchor-probability evidence and final metric evidence.

## raw_metrics

none; the process exited before model loading and before any update step.

## observations

none; see `stderr.log` for the launch traceback.

## anomalies

- The launch used the misspelled environment variable `QWEN_1P5B_PATH`; `train_eval.py` therefore fell back to the workspace model path, and Transformers rejected it as an invalid local/repository path.
- The process exited before model loading completed and before any update step; no metrics or diagnostics were produced.
- A corrected independent rerun was created as `run_20261020_1115_m3u_diagnostic_rerun`; raw files in this failed run are preserved.

## status

Run failed at launch. No algorithm change was made to the M3U update path.
