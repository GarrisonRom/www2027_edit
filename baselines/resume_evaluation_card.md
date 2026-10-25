# Evaluation-only repair and continuation

Hypothesis: separating generation/capability stages and resuming completed items prevents
monolithic6h timeouts from discarding work. Expected: imported rows unchanged, only missing
items computed, partial scores explicitly provisional. Control: same checkpoint/input/prompt.
Frozen metrics: original9640QA records, both prompt types and all probability reductions;
greedy64token generation EM/F1; full57subject5shot MMLU; pinned WikiText2 PPL.
No training, LR selection or method-score claim change. Original plus11saved checkpoints.
Budget:72h queue; score/generation/MMLU2h slices, PPL1h; scheduler grants extra30min for
loading/download/setup. Completed work resumes per item, per PPL window, per MMLU subject.
Existing results imported only after source/model/input/row validation. Tail corruption is
retained as an anomaly and only the interrupted final JSONL record is recomputed.
Capability jobs use cached data only during continuation to avoid repeated network hangs.
Missing caches are explicit failures, not zero values or successful complete evaluations.
Sanity: same scoring source hash as old batch, duplicate/conflict detection, finite scores,
strict count and complete flags, checkpoint hashes; old outputs and weights never overwritten.
Observed/verdict: pending tests and launch receipt. Next: tests, actual resumability smoke,
launch isolated continuation, confirm imported counts and actual missing-item progress.
Assets salvaged: all11trained adapters, Original6027 and GA interrupted per-item rows.
