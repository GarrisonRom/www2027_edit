# Run Log: original M3 diagnostic replay

exp_id: exp002_gsu_update_projection
assumption_id: diagnostic-only M3 geometry audit
method_version: original M3 with additional observations; no algorithm change
operator: Codex
started: 2026-10-20 10:27:00 +08:00 (prepared before launch)
ended: 2026-10-20 10:51:59 +08:00
status: completed

## purpose

Measure whether the original M3 projection is actually associated with gradient conflict, whether the per-tensor projection differs from a global projection, and whether the resulting AdamW displacement changes the sampled retain anchors. The M3 update rule is kept unchanged; extra per-anchor backward passes and post-update scoring are diagnostic only.

## code

- repo / url: local workspace / https://github.com/GarrisonRom/www2027_edit
- git_commit: 6ec831cfa25e07fcdcc908459f76a678c45434fb
- working tree: dirty; diagnostic script is untracked; script hash is frozen in config.json
- script: `exp/train/diagnose_gsu_m3.py`
- script_sha256: `3C23DBCAC4D61187FF742E3E540E94D44974253B50E040B7E62833A698650A07`

## environment

- language/runtime: Python 3.10.18 at `D:\Anaconda\envs\cuda126_env\python.exe`
- key dependencies: torch 2.6.0+cu126; transformers 5.14.1; peft 0.20.0; accelerate 1.14.0; safetensors 0.8.0; numpy 1.26.4
- CUDA/driver or platform: CUDA 12.6; NVIDIA driver 610.88; native Windows
- hardware: NVIDIA GeForce RTX 4060 Ti 16GB

## launch

- command: see `config.json`
- config snapshot: `config.json`

## data

- dataset: FaithUn local dataset, `data/faithun/FaithUn_Dataset.json`
- dataset_sha256: `D570CC3E30C083357CA45AB4DD9DF807ABBD800F51A657D89629E4FE7B3ADBB0`
- forget split sha256: `315312DD50B69993F3E90796E7162A9ADE22B771F3AA9EC5F570CEAAAD815428`
- retain split sha256: `8680043FE5C1F4F0517D7377664677EE225CD9EEBCB72D542C7709B1F5B6E214`
- split rule: frozen G2 forget examples (38) and first 33 retain indices; four anchors sampled per update

## reproducibility

- seed: `42` for Python and PyTorch
- forget order: `random.Random(42 + epoch).shuffle`
- anchor order: `random.Random(42).sample`, matching original M3
- deterministic flags: PyTorch deterministic algorithms not enabled; backend defaults unchanged
- diagnostic passes: do not update parameters or optimizer state; LoRA dropout is 0.0

## outputs

- raw step diagnostics: `diagnostics.jsonl`
- epoch summaries: `epoch_metrics.jsonl`
- final metrics: `metrics.json`
- checkpoints: `checkpoints/`
- final adapter: `adapter/`
- stdout: `stdout.log`
- stderr: `stderr.log`

## raw_metrics

- update steps: 114/114
- five_set: hop1_unlearn `0.1818`, hop1_test `0.2626`, hop2 `0.2772`, same_object `0.1806`, retain `0.4848`
- six_set: D_so `0.0833`, D_hop2 `0.0000`, D_hop2_F1_overlap `0.0000`, D_hop2_no_F1_overlap `0.0000`, K `0.5833`, retain `0.4848`, F1_hop2 `0.3200`
- global gradient conflict: `114/114`
- AdamW actual displacement conflict under aggregate retain gradient: `110/114`
- sampled-anchor actual displacement conflict: `413/456`
- direct sampled-anchor log-probability decreases: `377/456`
- mean per-tensor positive dot fraction: `0.8396`
- mean M3 per-tensor projected gradient norm ratio: `0.9337`
- mean corresponding global projection norm ratio: `0.9573`
- correlation between actual aggregate displacement dot and mean direct anchor log-probability change: `0.8783`
- correlation between anchor-level actual displacement dot and direct log-probability change: `0.8165`

## observations

1. The original M3 run had a positive aggregate forget/retain gradient dot on all 114 steps. This is direct evidence that the retain-direction conflict proxy was active in this run; it is not evidence that unconditional per-tensor projection is optimal.
2. The original M3 projection made the aggregate gradient dot with the retain gradient approximately zero, but AdamW produced a negative actual displacement dot on 110/114 steps. This directly confirms that the gradient projection did not constrain the optimizer's realized update.
3. The per-anchor view is heterogeneous: 428/456 raw anchor gradient dots were positive, while 247/456 dots remained positive after the per-tensor M3 projection. After AdamW, 413/456 anchor displacement dots were negative and 377/456 anchor log-probability changes were negative. The aggregate and individual anchor signals therefore do not behave identically.
4. The direct log-probability change tracks the first-order displacement proxy, but not perfectly; the correlations are `0.8165` at anchor level and `0.8783` at step level. This supports using the proxy as a diagnostic, not as a hard guarantee.
5. The diagnostic replay's final accuracy is close to the historical M3 re-evaluation but is not an exact paired comparison because the historical M3 adapter was trained before its initialization state was recorded.

## anomalies

- No traceback, non-finite metric, CUDA OOM, or incomplete update was observed.
- Native console output contains the same Windows encoding artifacts as earlier runs; structured JSON files are authoritative.
- The diagnostic added four per-anchor gradient passes and post-update scoring passes. These passes did not call `opt.step()` or modify optimizer state; they increased runtime only.

## status

The run completed successfully. The raw step-level observations are in `diagnostics.jsonl`; final metrics are in `metrics.json`. No algorithm change was made to the M3 update path.
