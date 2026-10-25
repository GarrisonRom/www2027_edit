# exp008 remote baselines

Started: 2026-10-22 (Asia/Shanghai). Exploratory single-seed screen.
Protocol: [card.md](card.md). Runner: ../../train/run_remote_baselines.py.

## Active remote batch

- Snapshot: `/root/autodl-tmp/www2027_baselines_20261022_v2`
- Batch ID: `batch_20261022_01`
- Controller PID at dispatch: 5158 (a PID alone is not evidence of progress).
- Run root: snapshot + `/exp/runs/exp008_remote_baselines/runs/batch_20261022_01`
- Status: run root + `/status.json`
- Incremental comparison: run root + `/comparison.json`, generated as jobs finish.
- Per-job files: config.json, modules.json, schedule.json (trained methods),
  diagnostics.jsonl, metrics.json, status.json and adapter/.
- Per-job stdout/stderr are in the batch root; controller logs are at snapshot root.
- Source/data/model provenance: batch_manifest.json, environment.txt,
  snapshot baseline_provenance.json and baseline_dirty.patch.

At dispatch verification: qwen35_4b_Original_F0 was in EVALUATION_STARTED;
GPU memory 9270 MiB and utilization 40%. No final baseline metrics existed yet.
This document is a dispatch record, not a live status report.

## Queue

For each method in Original, GA, GradDiff, NPO, SimNPO, evaluate/train Qwen3.5-4B
and then Llama-3.2-3B-Instruct. All trained methods here use F0-only.
Finally run GA F0+F1 for each model as expanded-supervision references.
Total: 2 original evaluations + 10 training runs, 114 steps per training run.

## Verification before dispatch

- Actual-model loss/backward smoke passed GA, GradDiff, NPO and SimNPO for both
  models, with finite losses/gradients, one-item MCQA/answer-LP audit and LoRA save.
- Compatibility archive: `/root/autodl-tmp/www2027_baselines_20261022/exp/runs/exp008_remote_baselines/runs/smoke_20261022_01`.
- Final snapshot only adds logging, incremental summaries, model-interleaved order
  and offline target-selection tests; loss and model-loading paths are unchanged.
- Target-selection unit tests: 2 passed locally and 2 passed remotely.
- Local frozen exp007 source/configuration was not modified.

Do not compare this transfer screen with historical GSU as if architecture,
prompt, environment and trainable-parameter budgets were all matched. These
baselines establish per-model references for later adapted GSU experiments.
