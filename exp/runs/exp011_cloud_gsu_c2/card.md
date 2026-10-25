# exp011: selected GSU-C2 cloud comparison

Date: 2026-10-22. Authorized by user request to select and run one cloud method.
Status: both model runs completed; exp012 capability audit dispatched; exploratory only.
Parent: proposed GSU-v8 public cloud C2, not historical layer-prior C3.

## hypothesis

Current-model component risk and F1 gradient spans can route F0 AdamW steps to
reduce associated recall with a better observed direct/retain tradeoff than the
fixed exp008 GradDiff reference. Extra feedback computation is not cost-matched.

## expected_pattern / primary

Primary compatibility endpoint: existing F1-covered hop2 accuracy75, lower better.
Report directF0/33, paraphrases/99, F1/5, train-retain33, heldout33, answer LP and
MCQA margin. Also measure full89-path/267 question and noncovered64-path recall
for C2; existing baselines lack those fields, so no all-hop2 superiority claim.
New full formal study primary remains all-hop2 as proposed in manuscript; this is
only a pilot against existing archived baselines, not substitution for that study.

## frozen baselines / null

All exp008/00914 outcomes on the identical models/evaluation items: Original,
GA, GradDiff, NPO, SimNPO, RMU-LoRA and expandedGA. Primary balanced reference
GradDiff; GA shows stronger forgetting endpoint and must not be hidden.
No structural null trained in this one-method request; cannot establish mechanism
or RQ3 without random/guard-only follow-ups. Do not label no-null evidence causal.
KLUE is additionally REQUIRED as pending localization comparator; not yet executed.
FaithUn is the benchmark, not an executed algorithm baseline named FaithUn.

## frozen method and budget

Two models: Llama3.2-3B-Instruct first, then Qwen3.5-4B; seed42;114 updates each.
LoRA targets exactly baseline.attach_lora, FP32 adapters, r8 alpha16 dropout0.
AdamWlr5e-5/betas.9,.999/eps1e-8/wd.01. Frozen F0 schedule and16-anchor plan.
Current gradient/path S only; weight(.5+.5*S/max); no old FFN prior. F1 gradients
guide component scores and per-tensor rank<=5 spans, refresh1/39/77, cutoff1e-6.
Route actual AdamW displacement(.5raw+.5projection)*weight; total routing+FB16
correction <=.30raw norm; feedback<=2 rounds and actual tolerance1e-4 inherited.
Final F0 decrease >=.9positive raw decrease +1e-6 numerical tolerance; try alpha
1,.5,.25,0. No extra F1 forget loss or test-query acceptance. Optimizer advances
exactly once. Model eval mode ensures deterministic forward during train/guard;
dropout0 and no gradient checkpointing, trainable loss still backpropagates.

Smoke both actual models: full initial scores, basis refresh,2actual optimizer
updates with16anchors, guard, finite gradients, optimizer count, LP audit, adapter
save/reload. Smoke adapters never initialize formal jobs. Failed smoke stops batch.
Formal budget228 updates plus4smoke updates; per-model wall limit6hours, max12hours
formal plussmoke; no automatic retries, hyperparameter search or hidden ablations.
Weights/adapter outputs retained. Timeout/failure stops subsequent model.

## provenance / sanity

Snapshot code, card, dirty patch, actual pip freeze, data/model file hashes and
commands before runs. Data and evaluation exactly existing fixed FaithUn loader.
CurrentF1 remains historical5fact transfer, not new model-specific path discovery.
Scoring must not mutate adapter. Per-module targets/count/init hash match original
baseline per model; source checks before each job. New directories only.

## observed / verdict / next_action

Remote runs: `llama32_3b_C2` and `qwen35_4b_C2`, 114 updates each.
Llama: covered hop2 1/75, F0 8/33, heldout retain 7/33, all89-path macro recall
30.34%. Qwen: covered 12/75, F0 10/33, heldout retain 8/33, all89-path macro
recall 38.58%. In the existing screen GA has the same covered/F0/heldout counts
for both models. This pilot shows no observed C2 gain on those measures; it is
single seed, has no random structural control and uses extra guidance compute.
Baseline all89-path evaluation plus MMLU/PPL is dispatched in exp012. Do not
infer mechanism or broad ability before those outputs. Original runs unchanged.

Chinese reader-facing explanation of GA vs C2, metric meanings/directions, complete
currently comparable count table, and interpretation limits:
docs/云端GSU-C2与指标说明.md. In brief, C2 matches GA-F0 on the three directly
comparable headline counts for each model; this is no observed gain, not evidence
that C2 is worse. C2 has additional guidance/feedback compute. Full89 baseline,
capability audit, cost matching, controls, and multi-seed evidence are still missing.

## screening boundaries

Exploratory GradDiff rectangle: QwenF0<=12/33,covered<=45/75,heldout>=16/33;
LlamaF0<=14/33,covered<=62/75,heldout>=22/33. Not a success guarantee or claim of
dominating all methods. Failures are evidence; do not revise method from these
test outcomes and call same-run confirmation. Related/unrelated audit missing.
