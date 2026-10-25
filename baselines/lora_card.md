# Explicit LoRA-adapted comparison

User instruction: use LoRA when the method supports it, omit methods without a verified
adaptation. This authorizes a distinct LoRA track, not silent native-to-LoRA fallback.
Hypothesis: loss-based baseline objectives remain executable through low-rank factors.
Expected pattern: finite gradients and adapter updates with exactly unchanged backbone;
no prediction that source-native and LoRA performance are equivalent.
Controls: frozen backbone, Original common evaluation; prior native runs separate.
Included: GA,GradDiff,NPO,SimNPO,RMU,DPOmis,DPOrej,CEU,PDU,WGA,SatImp,UNDIAL.
Excluded: KLUE; native neuron localization/update mask has no validated equivalent here.
Primary metrics: common source MCQA/test464/multihop slices plus MMLU/PPL; no switch.
Budget: preserve each source recipe's batch,epochs,LR/stop rules; LoRA r8,alpha16,dropout0.
RMU restricts adapters to layers5/6/7down-proj; other methods use decoder linear targets.
PDU retains calibrated epsilon and >5warmup requirement; CEU short-answer skips disclosed.
Verification: tiny actual-transformer tests across12objectives; two-step real-model GA/RMU
smokes with unique run directories. No formal performance comparison produced by smoke.
Environment: D:/Anaconda/envs/cuda126_env/python.exe; frozen sources and run manifests.
Observed/verdict: pending verification outputs; tests are executability, not paper parity.
Cause guess: native optimizer RAM can be avoided with explicit changed parameter domain.
Next action: verify finite gradients, frozen backbone and real-model smoke, generate configs_lora.
Assets: all existing native recipes/results preserved; source loss and evaluation code reused.
