# exp017: repaired KLUE and decisive guidance controls

Date2026-10-25. Independent follow-up; never edits active exp016 snapshots or historical outcomes.
Hypothesis: true association guidance must outperform matched null guidance beyond raw GA/guard effects;
source-aligned KLUE must pass real-model numerical gates before comparison.
Expected pattern: no assumed winner. If C1R or C2D matches C1/C2, guidance-specific interpretation is unsupported.
Null: Cg (guard/feedback only), C1R (within-module-type weight permutation), C2D (same per-tensor rank random subspace).
Primary: official forget all89 path-macro recall lower; F0 direct residual and independent retain reported alongside.
Secondary: fixed MMLU57x10, full WikiText2 PPL/NLL, F1/covered75, costs and source hashes.

## Fixed formal runs

Models: Qwen3-8B, Meta-Llama3.1-8B-Instruct, exact downloaded files from exp016.
Seeds42/43/44. KLUE native repair +GA-F0F1 +Cg +C1R +C2D =30 trained endpoints.
Each model/method has its own smoke. C controls use8-update smoke and114 formal updates;
GA expanded uses2-update smoke,114 formal updates over38 training facts (F0 eval remains33).
C1R permutes the C1 weights within module type, seed+7919, score histogram exactly preserved.
C2D preserves actual component weights; randomized basis matches each tensor rank and None fallback,
fixed by tensor name and training seed. Refresh timing and total correction radius match C2.
Cg shares feedback/guard/optimizer/update schedule with C1/C2 and sets component weights1, no span.
Comparators: exp016 C1/C2/GA same model, seed and114-update protocol; no mixed native/LoRA rank.

## KLUE repair

Observed exp016 failure: fused/memory-efficient attention backward requires aligned mask strides;
CPU saved-tensor offload restores a mask layout that violates that kernel contract.
Use math SDPA only, explicitly disable flash/memory-efficient/cuDNN SDPA in this new worker process.
No attribution/logprob/selection/optimizer-domain substitution. Actual8B smoke mandatory.
Tests reproduce a padded batch4 length18, two forwards before backward, BF16, streamed native
weights on both Llama and Qwen3 decoder classes. Old failed run retained and rerun_of references exp016.
Also normalize YAML exponent learning rates to float, with all method/track profile type regression.

## Development-only tuning and confirmation

Unused FaithUn101 clusters are separate from official forget33/retain66/test464.
Hash-order them, deduplicate direct triples and exclude any direct triple in official roles.
Choose first24 development forget, next24 development retain. Remaining53 are not consumed.
This is index/direct-fact disjoint, NOT a claim of entity/graph independence.
Train development forget only, protection development retain only; do not use official test scores,
F1 guidance, official retain, capability scores or test464 for hyperparameter selection.
Methods fixed before new results: GA, GradDiff, NPO, SimNPO, RMU.
LR grid2.5e-5,5e-5,1e-4; seed42 only;114updates; all other matched-LoRA profile settings fixed.
30 dev trials total (5methods x3LR x2models). Original dev retain measured before training each trial.
Feasible: dev direct MCQA <=1/3 and dev retain drop <=10 percentage points.
Choose minimum dev hop2 path-macro among feasible; ties by lower direct then smaller LR.
No feasible candidate -> explicitly no selected config, never relax gates after seeing results.
No full capability/test464/main forget evaluation and no saved model checkpoint for dev trials.
Freeze each selected LR then train official F0 seed42/43/44 from fresh base; per-method own smoke.
At most30 such confirmations. Independently freeze strongest dev winner per model and run its
F0F1 expanded version3seeds, with own smoke: up to6 additional confirmations.
Thus at most66 formal endpoints plus30 dev trials; failures/not-selected are not completed outcomes.
The search is intentionally bounded to LR; it is not a claim of exhaustive tuning of all paper hyperparameters.
The new development follows exploratory exp016, not a retroactive pristine confirmatory protocol.

## Execution and gates

After exp016 terminal state, numerical regression on released GPU, then KLUE smoke first.
One GPU, no competing trainer or modification of prior frozen sources. Queue21days max,
worker72h, watchdog74h. Native resource/space failures remain visible; no quantization/LoRA fallback.
Preserve each model/method/seed final checkpoint, metrics, configs, logs, raw samples and hashes.
Selected confirmation and control results append to existing comparison main tables, not just new notes.
Observed: implementation/tests complete, actual cloud outcomes recorded by separate run logs.
Verdict: untested. Next action: numerical gate, real8B smoke, formal/development trials, frozen selection and confirmation.
Assets: isolated development split/selection, matched controls, SDPA regression and independent queue.

