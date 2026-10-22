# Run Log: GSU-v3 M3A per-anchor update projection

exp_id: exp003_gsu_m3a
assumption_id: H-GSU-M3A-1
method_version: GSU-v3 M3A per-anchor AdamW update projection
operator: Codex
started: 2026-10-20 13:32:00 +08:00
ended: 2026-10-20 13:50:10 +08:00
status: completed

## purpose

Test whether replacing M3U's aggregate retain-gradient correction with the nearest update satisfying four individual sampled-anchor first-order constraints improves direct anchor log probabilities and the frozen retain/forget metrics.

## preregistered expectation

- Each corrected sampled-anchor displacement dot should be nonnegative within `1e-6`.
- Individual negative direct anchor log-probability changes should decrease relative to raw AdamW and M3U.
- The original exp002 gate remains frozen: primary at least `2/12` against M3, retain at least `14/33`, F1_hop2 at most `27/75`.

## code

- repo / url: local workspace / https://github.com/GarrisonRom/www2027_edit
- git_commit: 6ec831cfa25e07fcdcc908459f76a678c45434fb
- working_tree: dirty; see `patch.diff` and `working_tree_status.txt`
- script: `exp/train/diagnose_gsu_m3a.py`
- script_sha256: `B2804BE5C94AE1D68F3606150064767C67DF0BC7ED1771B92836EA9E97E75086`
- config snapshot: `config.json`

## environment

- language/runtime: Python 3.10.18 at `D:\Anaconda\envs\cuda126_env\python.exe`
- key dependencies: torch 2.6.0+cu126; transformers 5.14.1; peft 0.20.0; accelerate 1.14.0; safetensors 0.8.0; numpy 1.26.4
- CUDA/driver or platform: CUDA 12.6; native Windows
- hardware: NVIDIA GeForce RTX 4060 Ti 16GB

## launch

- command: see `config.json`
- expected updates: 114
- no hyperparameter search; no modification to the original M3/M3U implementation

## data

- dataset: FaithUn local dataset, `data/faithun/FaithUn_Dataset.json`
- dataset_sha256: `D570CC3E30C083357CA45AB4DD9DF807ABBD800F51A657D89629E4FE7B3ADBB0`
- forget split sha256: `315312DD50B69993F3E90796E7162A9ADE22B771F3AA9EC5F570CEAAAD815428`
- retain split sha256: `8680043FE5C1F4F0517D7377664677EE225CD9EEBCB72D542C7709B1F5B6E214`
- split rule: frozen G2 forget examples (38) and first 33 retain samples; four anchors sampled per update

## reproducibility

- seed: `42` for Python and PyTorch
- forget order: `random.Random(42 + epoch).shuffle`
- anchor order: `random.Random(42).sample`, matching M3/M3U
- deterministic flags: PyTorch deterministic algorithms not enabled; backend defaults unchanged
- correction: nearest Euclidean update projection onto individual anchor half-spaces; AdamW moments unchanged

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
- projection failures: 0
- active-set corrections: 114/114; mean active constraints `3.263`
- raw sampled-anchor displacement dots negative: `446/456`
- final sampled-anchor displacement dots negative under float32 diagnostic accumulation: `191/456`
- final dots below `-1e-6`: `38/456`; none below `-1e-5`
- active-set solver constraint minimum: `-2.44e-15`; actual parameter-writeback minimum: `-9.35e-6`
- raw direct anchor log-probability decreases: `441/456`
- final direct anchor log-probability decreases: `232/456`
- final direct log-probability better than raw AdamW: `439/456`; worse: `17/456`
- mean direct anchor log-probability delta: raw `-1.723526`, final `-0.007225`; mean paired improvement `+1.716301`
- correction-to-raw-update norm ratio: mean `0.165821`, median `0.142980`, maximum `0.358783`
- five_set: hop1_unlearn `0.303030`, hop1_test `0.282828`, hop2 `0.269663`, same_object `0.250000`, retain `0.333333`
- six_set: D_so `4/24`, D_hop2 `5/24`, D_hop2_F1_overlap `2/12`, D_hop2_no_F1_overlap `3/12`, K `21/48`, retain `11/33`, F1_hop2 `9/75`

## observations

- M3A raises the frozen primary from M3 `0/12` to `3/12`, meeting the primary improvement threshold.
- M3A retain is `11/33`, below the frozen minimum `14/33`, and below both M3 `16/33` and M3U `13/33`.
- The per-anchor KKT solution satisfies all four theoretical half-space constraints to numerical precision, and direct anchor log-probability changes improve for `439/456` paired observations.
- The final metrics show that reducing local anchor conflict does not preserve cumulative retain accuracy under this protocol.
- Verdict: reject M3A for the frozen single-seed advancement gate; do not enter multi-seed confirmation.

## anomalies

- The active-set solution was computed from float32 dot products and had minimum solver residual `-2.44e-15`. After writing the correction into model parameters, the measured minimum residual was `-9.35e-6`; no residual was below `-1e-5`. This is a small numerical writeback/accumulation discrepancy, but it means the predeclared `1e-6` hard-residual check is not met on 38/456 observations.
- No traceback, CUDA OOM, projection failure, non-finite metric, or incomplete update was observed.
- The numerical residual did not prevent direct log-probability and final metric evaluation; it is retained as an implementation limitation rather than silently changing the gate.

## status

Run completed successfully. Raw step diagnostics are in `diagnostics.jsonl`; epoch summaries are in `epoch_metrics.jsonl`; final metrics are in `metrics.json`. The original M3/M3U implementations were not modified.
