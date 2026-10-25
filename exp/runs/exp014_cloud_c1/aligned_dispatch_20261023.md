# Local-worker aligned cloud C1 rerun

Status at dispatch: running Llama 8-update smoke; not a completed training result.
Batch: batch_20261023_03_local_aligned.
Remote snapshot: /root/autodl-tmp/www2027_sync_20261023_c1_v3_local_aligned.
Rerun of: batch_20261023_01; batch02 interrupted during smoke for architecture parser correction.

The queue directly executes run_local_gsu_c_group.worker, variant C1,
with --cloud_architecture. Local defaults remain unchanged.
Shared: train mode, CAUSAL_LM LoRA wrapper, seed42, 114 updates, optimizer,
component scoring, F0/anchor schedule, no span projection, feedback, guard,
and checkpoint evaluations at updates38/76/114. Smoke uses8 updates.
Cloud exceptions: model loader and full text-component enumeration support
Llama3.2 and Qwen3.5 linear_attn. Model, tokenizer, hardware and packages differ
from local1.5B; this is not a same-model numerical replay.

Sequence per model: smoke, training, frozen all89/MMLU-10-subject/WikiText2 audit.
Both Llama3.2-3B and Qwen3.5-4B are queued. Worker writes config, source/data/model
hashes, environment, schedule, diagnostics, checkpoints and immutable run outputs.
Tests: cloud C1 alignment and C2 tests,7 passed (Windows cuda126_env).
No claim that alignment improves outcomes before new results are available.
