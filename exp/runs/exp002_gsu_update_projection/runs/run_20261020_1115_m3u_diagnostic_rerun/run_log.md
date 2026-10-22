# Run Log: original M3U diagnostic replay (corrected launch)

exp_id: exp002_gsu_update_projection
assumption_id: diagnostic-only M3U anchor-probability audit
method_version: original M3U AdamW actual-update correction; no algorithm change
operator: Codex
started: 2026-10-20 11:15:00 +08:00
ended: 2026-10-20 11:28:01 +08:00
status: completed
rerun_of: run_20261020_1107_m3u_diagnostic

## purpose

Replay the original M3U update path and directly measure whether its post-hoc correction improves the actual log probability of the same sampled anchor answers. The run records the AdamW proposal, the correction, per-anchor displacement dots, and anchor log-probability changes before and after correction. Diagnostic passes do not change the update rule or optimizer state.

## correction from prior failed run

The prior run used the misspelled environment variable `QWEN_1P5B_PATH`, so `train_eval.py` fell back to a nonexistent workspace model path and completed zero update steps. This rerun uses the required `QWEN_1_5B_PATH`; no source-code or algorithm change was made.

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

- update steps: 114/114
- correction events: 112/114; by epoch 38/38, 37/38, 37/38
- raw sampled-anchor displacement dots negative: 434/456
- corrected sampled-anchor displacement dots negative: 257/456
- raw direct anchor log-probability decreases: 427/456
- corrected direct anchor log-probability decreases: 261/456
- corrected direct log-probability better than raw AdamW: 437/456; worse: 11/456; equal: 8/456
- mean direct anchor log-probability delta: raw `-0.612951`, corrected `+0.043057`; mean paired improvement `+0.656008`
- paired improvement median: `+0.352780`
- correction-to-raw-update norm ratio: mean `0.168816`, median `0.184765`, maximum `0.347718`
- raw displacement-dot/direct-log-probability correlation: `0.962523`
- corrected displacement-dot/direct-log-probability correlation: `0.909642`
- final aggregate dot residual: maximum absolute value `1.9984`; negative residuals after correction were at most `-2.76e-6`, consistent with floating-point accumulation rather than a substantive remaining conflict
- five_set: hop1_unlearn `0.151515`, hop1_test `0.232323`, hop2 `0.254682`, same_object `0.138889`, retain `0.393939`
- six_set: D_so `0/24`, D_hop2 `1/24`, D_hop2_F1_overlap `0/12`, D_hop2_no_F1_overlap `1/12`, K `28/48`, retain `13/33`, F1_hop2 `9/75`

## observations

- The original aggregate M3U correction improves the same sampled anchors' direct answer log probabilities in the majority of cases and changes the mean paired delta from negative to slightly positive.
- The correction also reduces, but does not eliminate, individual-anchor first-order conflicts: 434/456 negative displacement dots become 257/456.
- The final five-set and six-set metrics are numerically identical to the historical M3U run: primary `1/12`, retain `13/33`, F1_hop2 `9/75`, K `28/48`.
- Therefore the post-hoc correction is locally real, but its aggregate sampled-anchor guarantee does not transfer to all individual anchors or to the cumulative retain evaluation.
- Decision: do not modify the original M3U implementation in place. A per-anchor constraint candidate may be evaluated only as a separately versioned, separately logged experiment; this diagnostic alone does not justify changing the production method.

## anomalies

- Previous launch failure is archived in `run_20261020_1107_m3u_diagnostic`; this run corrects only its environment-variable typo.
- No traceback, CUDA OOM, non-finite metric, or incomplete update was observed.
- `stderr.log` contains model-loading progress output but no error; structured JSON files are authoritative.
- The diagnostic added per-anchor gradient passes and post-update scoring passes. These did not call `opt.step()` or modify optimizer state; they increased runtime only.

## status

Run completed successfully. The raw step-level observations are in `diagnostics.jsonl`; epoch summaries are in `epoch_metrics.jsonl`; final metrics are in `metrics.json`. No original M3U algorithm change was made.
