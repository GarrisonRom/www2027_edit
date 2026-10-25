# Checkpoint retention 2026-10-24

The user authorized checkpoint-only cleanup and confirmed one final checkpoint per model/method/seed.
Because exp015 contains separate native and matched-LoRA tracks, retention keys include track so those results are not conflated.

Remote inventory covered all directories matching /root/autodl-tmp/www*/exp/runs. The cleanup tool removed only model-weight checkpoint files and did not delete base models, source, configs, statuses, logs, metrics, or evaluation JSON.

Final result and exact per-file SHA256/delete reasons are in checkpoint_retention_manifest.json.
Kept final files: 42. Removed weight files: 63 (35 smoke artifacts, 14 superseded runs, 13 duplicate snapshot links, 1 stale-running orphan with a completed matching checkpoint). There were zero weight files from runs explicitly marked failed/interrupted; the old incomplete Llama NPO orphan was removed only after confirming a completed same-seed/model/method/track counterpart and no active process using it.

The 63 file entries sum to approximately 8.71 GiB of paths, mostly hardlinks. Estimated unique blocks freed: 1.22 GB. /root/autodl-tmp available space increased from about 11 GB to 12 GB. The 8B model base files were untouched.

The active 8B C1 seed42 formal run remains running; it loads the base model and will write its own final adapter at completion. Future runs already write one terminal adapter/checkpoint per run; checkpoint_eval files are metrics JSON, not model weights.

