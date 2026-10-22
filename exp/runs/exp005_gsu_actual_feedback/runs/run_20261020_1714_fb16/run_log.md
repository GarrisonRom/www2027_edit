# Run Log: GSU-v5 FB16 actual-logprob feedback

exp_id: exp005_gsu_actual_feedback
assumption_id: H-GSU-FB-1
method_version: GSU-v5-actual-feedback
run_id: run_20261020_1714_fb16
operator: Codex
started: 2026-10-20 17:16:23 +08:00
ended: 2026-10-20 18:35:04 +08:00
status: completed_failed_advancement_gate

## purpose

Test whether measuring the actual retain-anchor answer logprob after the raw AdamW proposal and feeding that error into a bounded correction repairs the nonlinear transfer failure seen in COV16/CUM16.

## preregistered expectation

- 16 fresh anchors per update and `33/33` unique coverage over 114 updates.
- Final direct anchor logprob drop rate at least 10 percentage points lower than COV16's `866/1824` final decreases.
- Each update uses at most two feedback rounds and total correction-to-raw norm ratio is at most `0.30`.
- Final retain at least `20/33`, primary at least `3/12`, and F1_hop2 at most `29/75`.

## code and provenance

- repository: local workspace / https://github.com/GarrisonRom/www2027_edit
- commit: `6ec831cfa25e07fcdcc908459f76a678c45434fb`
- working tree: dirty; see `working_tree_status.txt`
- exact script snapshot: `diagnose_gsu_actual_feedback.py.snapshot`
- script SHA256: `758290236B733E90CC7C8A7683B33D88ABA7372387B274DE278ADD1D9C482613`
- configuration snapshot: `config.json`

## environment

- interpreter: `D:\Anaconda\envs\cuda126_env\python.exe`
- Python 3.10.18; torch 2.6.0+cu126; CUDA 12.6
- transformers 5.14.1; peft 0.20.0; accelerate 1.14.0
- GPU: NVIDIA GeForce RTX 4060 Ti 16GB

## launch

- fixed seed: 42
- expected updates: 114
- no hyperparameter search
- actual-drop tolerance: `1e-4`
- maximum feedback rounds: 2
- maximum total correction ratio: `0.30`

## data

- dataset: FaithUn local dataset, `data/faithun/FaithUn_Dataset.json`
- dataset SHA256: `D570CC3E30C083357AD45AB4DD9DF807ABBD800F51A657D89629E4FE7B3ADBB0`
- forget split SHA256: `315312DD50B69993F3E90796E7162A9ADE22B771F3AA9EC5F570CEAAAD815428`
- retain split SHA256: `8680043FE5C1F4F0517D7377664677EE225CD9EEBCB72D542C7709B1F5B6E214`

## outputs

Training will write `diagnostics.jsonl`, `epoch_metrics.jsonl`, `metrics.json`, epoch checkpoints and the final adapter under this run directory. The launch wrapper will capture stdout and stderr separately.

## raw_metrics

All values below are from this run's `metrics.json` and an independent audit of `diagnostics.jsonl`.

- update steps: `114/114`; projection failures: `0`; Dykstra failures: `0`
- unique sampled retain anchors: `33/33`; anchors per update: `16`
- primary `D_hop2_no_F1_overlap`: `3/12`; retain: `19/33`; F1_hop2: `22/75`; K: `9/48`; `D_so`: `3/24`; five-set same_object mean: `0.256944`
- raw direct anchor log-probability decreases: `1780/1824`; final decreases: `838/1824`; final better than raw AdamW: `1750/1824`
- mean direct anchor log-probability delta: raw `-1.155235`; final `-0.053329`
- mean correction-to-raw-update norm ratio: `0.215843`; median `0.210068`; maximum `0.300000102`; no value exceeded the frozen `0.300001` audit allowance
- feedback rounds: `227` across 114 steps; `113` steps used two rounds and one used one round
- actual risk observations: `3000`; final actual deltas below `-1e-5`: `838/1824`; minimum final delta `-17.209110`
- runtime stderr contains only model-loading progress output; no traceback or non-finite error was recorded

## anomalies

- The candidate reduced actual anchor damage relative to its own raw AdamW proposal, but the improvement over the stronger COV16 reference was small: `838/1824` versus `866/1824` final decreases, and the final mean was `-0.053329` versus COV16 `+0.026811`.
- The final retain score was `19/33`, one item below the frozen advancement gate of `20/33`; the direct-drop reduction was `1.54` percentage points, below the preregistered 10-point mechanism gate.

## status

Training and evaluation completed. Runtime sanity passed, but the preregistered advancement gate failed; this run is retained as a single-seed diagnostic and does not authorize multi-seed confirmation.
