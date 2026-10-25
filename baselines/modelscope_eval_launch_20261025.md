# ModelScope local MMLU/PPL evaluation launch

Started2026-10-25 12:51:29 Asia/Shanghai. Queue PID108460.
Active root:baselines/runs/modelscope_eval_20261025_125129.
Training source:formal_lora_20261024_120150;11saved LoRA checkpoints unchanged.
Prior evaluation source:eval_resume_20261025_110652. Queue145320 and worker161084 stopped
to avoid competing GPU jobs. Imported completedOriginal9640score records and GA7350score
records after checkpoint/input/item validation; old files unchanged. Old status may remain
running and is historical; use the new queue_status.json for actual active jobs.

Execution: full PPL for Original+11methods, full MMLU57subjects/14042test items5shot for
same12endpoints, then remaining QA score/generation phases. Per-stage time slicing/resume
retained. No training, no hyperparameter selection and no change in probability QA protocol.
Source: verified ModelScope local Parquet underD:/GSU_datasets/modelscope_eval_20261025_02.
Manifest SHA25640a222a50dc7eeb6e581562cadff9f813b56e14154b5e5e6c9198cdfc7ea19bd.
All file hashes verified before queue snapshot. Network disabled for workers; local reads
explicitly passed to the evaluator. Cloud pinnedHF revision byte-equivalence not certified;
record these as ModelScope-source evaluation results, not silently the old HF revision.

Smoke validation (not full dataset performance):
-Original PPL2048input tokens/2047prediction targets, PPL7.323453521763826, finite.
-Original MMLU abstract_algebra2test questions5shot, accuracy.5,8option-likelihood requests;
 task prompts/options/dev/test/first_n sampler preserved, only dataset path replaced.
-62regression tests passed, including local task override preserving source prompt/splits.
Sources: runs/modelscope_ppl_smoke_20261025_01 andmodelscope_mmlu_smoke_20261025_01.

Budget:72h total; PPL1h,MMLU2h,scoring/generation2h slices with30min setup grace.
Progress/metrics are retained per PPL window, MMLU subject and QA item. Incomplete results
remain provisional. Final ranking waits for completed metrics, no test-driven method choice.
