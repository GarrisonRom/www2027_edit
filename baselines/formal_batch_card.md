# New local formal LoRA batch

Hypothesis: source-specific corrected objectives produce measurable forgetting/retention
under explicit LoRA constraints. Expected pattern: source-specific tradeoffs, not assumed
GSU superiority. Null control: unmodified Original under the same common evaluation.
Frozen metrics: Figure8 conditional-answer MCQA UA/UAextended/TA/SA/MA, covered/noncovered
path macro, raw-scoring sensitivity, greedy generation EM/F1, full MMLU57subject5shot,
WikiText2 token PPL. Independent retain is test464, not training retain66.
Methods: GA,GradDiff,NPO,SimNPO,RMU,DPOmis,DPOrej,CEU,WGA,SatImp,UNDIAL.
PDU deferred until constraint calibrated; KLUE excluded per user LoRA-only instruction.
Budget: one source-default config/method, seed per source recipe, 2update smoke then formal
source epochs/UA stop, LoRA r8alpha16dropout0, per-child6h, batch72h. No full LR grid in
this batch; resulting checkpoints are NOT validated lowest-LR or tuned-method optima.
Schedule: train and save11checkpoints first, then Original and each checkpoint evaluation.
Failure: independent method smoke/training failure does not block unrelated methods;
capability failures/incomplete output explicit. No native/old114step table merge.
Sanity: immutable source snapshot and input hashes, finite gradients, strict LoRA targets,
unchanged objective batching; all source/dependency/config/model hashes recorded before jobs.
Observed/verdict: pending. Next action: launch and confirm real formal training update.
Cause guess: prior recipe/evaluation adaptation differences; no causal conclusion yet.
Assets: immutable baseline package, verified objective tests, old pilot results retained.
