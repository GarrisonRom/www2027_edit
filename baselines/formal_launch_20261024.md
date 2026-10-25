# Local formal LoRA launch receipt

Launch2026-10-24 12:01:50 Asia/Shanghai. Queue PID178748.
Batch: baselines/runs/formal_lora_20261024_120150.
Entry: D:/Anaconda/envs/cuda126_env/python.exe -B -m baselines.launch_formal.
Model: E:/12models/modelscope/Qwen2.5-1.5B-Instruct.
Scope:11source-specific single-config LoRA trials, each own2update smoke then formal
training/checkpoint; later Original and all checkpoints common full QA/generation/full
MMLU/WikiText evaluation. This is not a completed LR search or final tuned comparison.
Methods:GA,GradDiff,NPO,SimNPO,RMU,DPOmis,DPOrej,CEU,WGA,SatImp,UNDIAL.
PDU deferred uncalibrated constraint; KLUE excluded unvalidated LoRA neuron mask.
Budget:6h per child,72h queue. Hidden background process, no recurring monitoring.
Evidence:43regression tests passed;91locked dependency/input/reference hashes verified.
Queue creates independent snapshot and fixed method config files before training; model
files fully hashed. Capability failure remains evaluation_incomplete, never completed.
Status/childPIDs: batch/queue_status.json. Per-method logs and run status preserved.
Source/checkpoint manifests and environment retained; old exp016 results unchanged.
