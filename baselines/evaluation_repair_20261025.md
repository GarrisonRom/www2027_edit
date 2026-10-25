# Evaluation repair and restart receipt

Final restart2026-10-25 11:06:52 Asia/Shanghai; queue PID145320.
Active root:baselines/runs/eval_resume_20261025_110652.
Old batch:baselines/runs/formal_lora_20261024_120150.

Scope: evaluation only, all11trained adapters and Original preserved. Old monolithic
queue178748 and GA worker186396 explicitly stopped. Their status files are historical
and may remain running; see new queue rather than interpreting stale status as live.

New stages: scoring, generation, PPL windows, MMLU subjects. Each stage has its own
status/metrics and budget, not a shared6h kill. Score/generation/MMLU slices2h and PPL1h,
with30min setup grace and72h batch ceiling. Incomplete work resumes in later slices.
Probability/generation resume per-item, PPL per-token-window, MMLU per-standard subject.
Partial metrics explicitly provisional. Stage completion does not certify other stages.

Imported source scoring hashes are identical to the old snapshot. Base/checkpoint/input
identities checked. Old items validated against canonical metadata and score formulas.
Available old work:Original6027items andGA951items, probability ANDgeneration records.
Original old files untouched. A trailing interrupted JSONL fragment is retained as anomaly
and recomputed, never mistaken for a valid result. Conflicts/duplicates stop reuse.

Verification:55tests passed, including loss/LoRA regression and new resume/import cases,
PPL exact target count and no repeat forwards, MMLU per-subject reuse, bounded Windows
atomic-write retry. Actual1.5B two-item score smoke completed; repeat invocation left
items.jsonl SHA256identical and avoided reloading/computing the completed items.
Harness registers many mmlu-prefixed variants; standard selection verified57task YAMLs,
excluding macro groups. New small preflight runs110052/110500were stopped before useful
formal scoring to fix this filter and transient Windows file-lock handling; kept as evidence.

Capabilities: use cache-only in this continuation to prevent repeated network stalls.
Missing MMLU/WikiText data remain explicit failed phases and require a separate data
preparation/rerun; not replaced with zero scores and not called completed evaluations.
No promise of a complete capability table until those datasets and stages succeed.

Progress: active-root/queue_status.json, comparison.md/json and Method/phase/status.json.
Training weights and prior results are not overwritten or retrained. Original remaining
scoring3613items;GA remaining8689items before any new continuation work.
