# ModelScope local capability execution and scoring continuation

Hypothesis: verified local data eliminates network download failures without changing
model checkpoints or MMLU prompt/5shot/seed/test order. Expected: real-model finite PPL,
MMLU question likelihoods and subject results; no predetermined performance direction.
Control: Original plus11saved LoRA endpoints, identical source/task protocols.
Frozen metrics: full WikiText2-raw-v1 test PPL, full57subject/14042question MMLU5shot.
Source: D:/GSU_datasets/modelscope_eval_20261025_02 with verified174Parquet files and
manifest hash. Source is ModelScope; cloud HF revision byte-equivalence not certified.
Independent smoke: Original2048tokens PPL and2abstract-algebra questions, explicitly
NOT full capability scores. Formal queue prioritizes PPL then MMLU, then resumes QA
scoring and generation. Existing Original9640/GA7350score records imported after validation.
No training or hyperparameter change. Old stage files retained; stopped queue145320/
worker161084 relinquish GPU for a single unified replacement queue.
Budget:72h overall, PPL1h and MMLU2h per resumable slice, scoring/generation2h. Window,
subject and item progress retained. Missing/failed metrics never marked completed.
Sanity: all57configs retain exact subject descriptions/prompt/first_n dev5shot; only
dataset path changes to verified localParquet. Stable options/order and answer mapping.
Observed/verdict: pending smoke/formal logs. Next: dependency/unit checks, two real smoke
jobs sequentially, isolated formal launch; report actual progress rather than promises.
Assets:11trained checkpoints, downloaded corpus, all previous score/generation records.
