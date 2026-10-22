# Run Log: run_20261019_1631_m3_reval

exp_id: exp002_gsu_update_projection
assumption_id: H-GSU-UPD-1
method_version: historical GSUM3 adapter re-evaluation
operator: Codex
started: 2026-10-19 16:32:39 +08:00
ended: 2026-10-19 16:39:57 +08:00
status: completed

## code
- repo / url: local workspace / https://github.com/GarrisonRom/www2027_edit
- git_commit: 6ec831cfa25e07fcdcc908459f76a678c45434fb
- working tree: dirty; exact change in patch.diff (PyTorch seed fix; eval-only run)

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
- split rule: frozen exp002 G2 forget examples (38) and first 33 retain indices

## reproducibility
- seeds: train_eval.SEED=42; evaluation only, existing adapter fixed by SHA-256
- deterministic flags: PyTorch deterministic algorithms not enabled
- other randomness (sampling/order/init): no training or sampling in this run

## hardware
- device type & count: NVIDIA GeForce RTX 4060 Ti 16GB x1
- parallelism: one process; sequential GPU runs

## budget (planned / actual)
- wall-clock: evaluation only; actual 7m18s
- accelerator-hours: approximately 0.12 GPU-hours
- other cost (storage/traffic/labeling): local; no model download

## outputs
- checkpoints: existing exp/adapters/1p5B_v2_GSUM3; SHA-256 recorded in config.json
- predictions/artifacts: exp/results/eval_results_gsu_projection.jsonl; exp/results/gsu_projection_sixset.jsonl; per-run metrics.json
- figures: none

## raw_metrics
<!-- Primary and secondary metrics will be copied from this run's isolated result rows. -->
- primary: D_hop2_no_F1_overlap = 0/12; see metrics.json
- secondary & CI over seeds: D_hop2_F1_overlap 0/12; F1_hop2 25/75; retain 16/33; one seed only, no CI
- key subgroups: D_hop2_no_F1_overlap (n=12), D_hop2_F1_overlap (n=12)

## logs
- stdout: stdout.log
- stderr: stderr.log

## anomalies
- PyTorch RNG was not seeded in the original training entrypoint; one-line seed fix is captured in patch.diff. This re-evaluation loads a fixed historical adapter and is unaffected by adapter initialization.
- Native log text includes garbled Chinese characters due to Windows console encoding; structured metrics.json and result JSONL are valid UTF-8 and are the source of truth. stderr contains model weight-loading progress only; no traceback or CUDA OOM.

## links
- experiment card: ../../card.md
- rerun_of / supersedes: historical M3 summary only; prior run had no run_id
- related runs: M3C and M3U candidate runs
