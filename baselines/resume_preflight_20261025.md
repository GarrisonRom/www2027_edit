# Resumable-evaluation preflight record

Old queue178748 and active GA evaluation186396 stopped under user-authorized repair.
All checkpoints and old per-item outputs retained unchanged. Old status files may remain
running and are not evidence of a live process.
Preliminary continuation188248 stopped before useful new scoring while final preflight
found an overbroad MMLU task-name filter. The harness registers2006mmlu-prefixed variants;
standard subject selection now checks the pinned mmlu/default YAML directory, exactly57.
Preliminary artifacts at runs/eval_resume_20261025_110052 are retained, not formal results.
Old scoring source hashes still match current protocol/evaluate implementations.
Initial regression52passed; further PPL-window/MMLU-subject checks added before relaunch.
