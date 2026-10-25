# Source-specific baseline reconstruction and evaluation

Hypothesis: historical differences may partly reflect recipe/evaluation adaptation.
Expected pattern: source-specific recipes and conditional MCQA may change relative results;
no promised GSU improvement or post-result primary endpoint change.
Null/control: Original with identical evaluation and fixed shuffled options.
Primary: Figure8 prompt answer-conditional sequence sum MCQA; forget paraphrase/multihop
lower better, independent test464 direct/paraphrase/multihop and same-answer higher better.
Covered/noncovered path slices, conditional mean and legacy-score sensitivity reported.
Capability: MMLU57subjects5shot seed42 sampled10/subject (not full MMLU), pinned WikiText2
test token-weighted PPL; optional greedy-generation EM/tokenF1.
Baselines: FaithUn GA,GradDiff,DPOmis,DPOrej,NPO,RMU,KLUE; author/source-transfer extended
SimNPO,CEU,PDU,WGA,SatImp,UNDIAL. No silent LoRA fallback.
Budget: recipe-specific; FaithUn batch4,max150epochs,UA<=.33 each epoch, registered source-range
LR plans; each trial6h. PDU explicit calibrated epsilon and >5warmup epochs.
Verification: Original stratified1/group real-model smoke, not a formal comparison row.
Sanity: independent preference direction, token-weighted accumulation, source/input locks,
tokenizer boundaries, finite gradients; failed capabilities never silently count as complete.
Observed/verdict: pending real-model validation; different model means no author-exact claim.
Cause guess: recipe/evaluation mismatch under repair.
Next action: tests, Original smoke, resource/dependency checks, then registered trials;
choose using training-UA only, never final test multihop scores.
Assets: pinned source losses/documents and historical pilot evidence retained.
