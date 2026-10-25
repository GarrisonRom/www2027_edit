# LoRA适配验收（2026-10-24）

结论：12种目标可通过LoRA因子训练，KLUE不进入此比较。

已通过41项测试，其中12种方法逐一使用真实Llama transformer结构的小模型和PEFT LoRA：
验证loss/gradient有限、adapter实际更新、冻结底座每个tensor严格不变。另检查源目标参数
和预算未被LoRA开关覆盖、RMU组件范围保留、KLUE显式拒绝、原生/LoRA选择结果不混组。
这是可执行性与冻结语义验证，不是证明全参数与LoRA的优化过程或性能相同。

新适配卡：lora_card.md；生成配置：configs_lora；入口：run.ps1 -LoraAdapted。
真实1.5B验证目录：runs/ga_lora_smoke_20261024_01、runs/rmu_lora_smoke_20261024_01。
最终状态以各自status.json为准；两步smoke不是正式对比结果。原生结果/配置保留。
两项实际均为smoke_passed、2updates：GA可训练9,232,384参数，RMU可训练251,904参数。
GA主存工作状态估算684,589,056bytes、检查时可用9,866,059,776bytes；该估算不包含GPU
上的冻结模型/激活，不把它误称为训练总内存。两项不再触发原生全参数Adam的主存门检。

KLUE排除理由：本项目原实现通过原生FFN神经元行/列mask限制更新；直接给低秩因子套相同
mask没有已验证的等价性。此处未证明所有可能的LoRA改写都不可能，只是不纳入未经验证的实现。
PDU需单独校准保持约束，CEU的首token排除照旧披露，不因LoRA就自动宣称复现成功。
