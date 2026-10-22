# Run Log: run_20261019_1631_m3u_train_eval

exp_id: exp002_gsu_update_projection
assumption_id: H-GSU-UPD-1
method_version: GSU-v2 M3U AdamW update projection
operator: Codex
started: 2026-10-19 16:48:57 +08:00
ended: 2026-10-19 16:58:49 +08:00
status: completed

## code
- repo / url: local workspace / https://github.com/GarrisonRom/www2027_edit
- git_commit: 6ec831cfa25e07fcdcc908459f76a678c45434fb
- working tree: dirty; exact change in patch.diff

## environment
- language/runtime: Python 3.10.18 at D:\Anaconda\envs\cuda126_env\python.exe
- key dependencies / lock hash: torch 2.6.0+cu126; transformers 5.14.1; peft 0.20.0; accelerate 1.14.0; safetensors 0.8.0; numpy 1.26.4; tqdm 4.67.1; no lock file
- CUDA/driver or platform: CUDA 12.6; NVIDIA driver 610.88
- image/container: native Windows environment

## launch
- command: see config.json; model path supplied through QWEN_1_5B_PATH
- config snapshot: config.json

## data
- dataset & version/path: FaithUn local dataset, data/faithun/FaithUn_Dataset.json
- checksum: dataset D570CC3E30C083357CA45AB4DD9DF807ABBD800F51A657D89629E4FE7B3ADBB0; forget split 315312DD50B69993F3E90796E7162A9ADE22B771F3AA9EC5F570CEAAAD815428; retain split 8680043FE5C1F4F0517D7377664677EE225CD9EEBCB72D542C7709B1F5B6E214
- split rule: frozen G2 forget examples (38); first 33 retaining split indices used as anchor pool

## reproducibility
- seeds: Python random and PyTorch CPU/CUDA RNG set to 42 by entrypoint
- deterministic flags: deterministic algorithms not enabled; backend defaults unchanged
- other randomness (sampling/order/init): forget order uses seed+epoch; retain anchors sample 4 from seed-42 RNG; LoRA initialization uses seeded PyTorch RNG

## hardware
- device type & count: NVIDIA GeForce RTX 4060 Ti 16GB x1
- parallelism: one process; sequential GPU runs

## budget (planned / actual)
- wall-clock: 3 epochs and both evaluations; actual 9m52s
- accelerator-hours: approximately 0.164 GPU-hours
- other cost (storage/traffic/labeling): local; no model download

## outputs
- checkpoints: exp/adapters/1p5B_v2_GSUM3U_ckpt (one file per completed epoch)
- predictions/artifacts: exp/adapters/1p5B_v2_GSUM3U; exp/results/eval_results_gsu_projection.jsonl; exp/results/gsu_projection_sixset.jsonl; per-run metrics.json
- figures: none

## raw_metrics
- primary: D_hop2_no_F1_overlap = 1/12 vs M3 0/12; below the +2/12 screening gate
- secondary & CI over seeds: F1_hop2 9/75 vs M3 25/75; retain 13/33 vs M3 16/33 (decline of 3 exceeds gate); one seed only, no CI
- key subgroups: D_hop2_no_F1_overlap (n=12), D_hop2_F1_overlap (n=12)

## logs
- stdout: stdout.log (includes epoch loss and correction checks/events)
- stderr: stderr.log

## anomalies
- AdamW update corrections by epoch: 38/38, 37/38, 37/38 (112/114 total); raw values and metrics in metrics.json
- stderr contains model weight-loading progress only; no traceback or CUDA OOM. Native stdout includes Windows console encoding artifacts; metrics.json and result JSONL are authoritative.

## links
- experiment card: ../../card.md
- rerun_of / supersedes: none
- related runs: M3 re-evaluation and M3C candidate
