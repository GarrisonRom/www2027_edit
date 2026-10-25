# Verification record

2026-10-24. Baselines are source-specific algorithm transfers, not certified author-exact
results on the original models. Historical pilot weights/metrics remain unchanged.

-Final contract suite:26passed, including source-specific parameters, same-prompt scoring,
  training/evaluation option-order identity, token/sequence reductions, DPO retain direction,
  no test-metric hyperparameter selection, failed-capability status and full-MMLU no-sampling.
-Actual Qwen2.5-1.5B Original smoke01 caught that new candidate-context bug before formal
  baseline comparisons. It is fixed; smoke01 failure and traceback remain in runs/.
-Smoke02 completed the new stratified behavior scores (11groups,9640 full evaluation items
  registered) then stalled on HuggingFace WikiText HEAD connectivity. Process177896 was
  stopped deliberately to avoid unbounded network retries. Its status file is stale and must
  not be interpreted as a successful/full run. Raw partial outputs remain preserved.
-lm_eval0.4.11 installed in baselines/.deps (isolated from shared CUDA environment), text
  HFLM import verified. This is dependency validation, not an MMLU performance result.
-Subsequent offline smoke is used to validate missing-dataset failure reporting. No
  unsupported capability values will be filled with0 or included in formal ranking.
-Native1.5B full Adam requires estimated~24GB CPU working state plus buffers. Current
  machine has32GB physical RAM and substantially less free RAM; a strict resource gate
  prevents an accidental LoRA fallback or a claimed successful native reproduction.

Final validation additions:

-All9640evaluation records checked on the actual Qwen tokenizer, all3candidates under
  raw/paper contexts: no invalid option sets or tokenizer-boundary mismatches.
-RMU native smoke02 passed2updates with41,287,680actual trainable parameters confined
  to layers5/6/7down-proj; c20,full QA non-padding activation MSE. Losses5531.58/6059.22
  are finite and much larger than answer-only adaptation; this is not a performance claim.
  Scope matches the author's full-sequence objective except an explicit padding exclusion.
-Original smoke04 completed real-model stratified scoring and greedy generation. It correctly
  ended evaluation_incomplete when MMLU/WikiText data were unavailable offline. Dependency
  import errors from smoke03 were repaired separately; final remaining errors are missing data.
-MMLU task registration succeeds with no dependency errors. Both cloud-style sampled570
  questions and full57-subject5shot modes exist; neither has a new performance result yet.
-GA resource preflight:1,543,714,304params; estimated26,846,912,512CPUbytes required,
  available9,982,205,952bytes at inspection. Full native GA did NOT start or fall back toLoRA.
-SimNPO author GitHub config blob9aeeac3c preserved as API JSON, hash-locked with references;
  default beta2.5/gamma.1375/seed1001/weight_decay.01 and1epochwarmup replace old YAML defaults.
-Original training-retain calibration: pooled token NLL3.8012419243653617 on66items/192tokens.
  PDU epsilon remains explicitly unset until a calibrated trial is registered; no paper constant
  from TOFU is silently copied into a different dataset's constraint.
-Paper3.5 UA/UA-extended/TA/MA formulas implemented; both same-answer pooling scopes and
  resulting score variants are retained because exact author-code SA aggregation is unavailable.

New formal trials, hyperparameter search, full MMLU/PPL and multi-seed comparisons have
not completed in this implementation task. See experiment_card.md and per-run manifests.

Portability:91dependency/reference/input files hash-verified. Lock keys normalized to POSIX
with Windows legacy-path compatibility; copying baselines/ to Linux does not require the
old exp016 directory when vendor/runtime is present. Source-only cloud checkouts without
.git record unavailable Git identity and retain exact source hashes instead of failing.
