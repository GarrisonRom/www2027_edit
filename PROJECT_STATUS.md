# 项目状态

## 当前阶段

- 项目：探测驱动的扩展遗忘集、损伤显影与定向恢复
- 当前里程碑：M5 三项研究内容实证结果归档
- 当前任务：统一证据口径，补做多 seed、R 禁区和恢复硬指标的稳健性实验
- 状态：研究内容一至三的单模型主线实验已完成；3B clean 扩展验证已完成；关键硬指标仍有未通过项
- 更新时间：2026-10-19

## 已完成

- 核验 FaithUn 上游仓库可访问。
- 固定上游提交：`6a912b7d71b3259740aab92071ae3c4a7eab2b00`。
- 下载并保存 10 MB 的 `FaithUn_Dataset.json`。
- 保存官方 `forget`、`retaining`、`test` 三个 split 文件。
- 完成实体数、事实簇数、字段结构和 split 索引的本地核验。
- 保存上游 README、许可证和 SHA-256 校验记录。
- 下载 `open-unlearning` 到 `code/open-unlearning/`，固定主分支提交 `17cbbc87192e6934deb92875c359c91bbd837fb4`。
- 确认本地模型 `Qwen2.5-1.5B-Instruct` 和 `Qwen2.5-3B-Instruct` 的配置、tokenizer 均可离线读取。
- 完成 Qwen2.5-1.5B-Instruct 的一次 GPU 前向测试。
- 使用当前 Python 3.10 环境完成 `open-unlearning/src` 的源码语法检查。
- 完成 Qwen2.5-3B-Instruct 独立预实验：hop1 一致率均值 0.739，latent multi-hop 0.625。
- 完成 3B 候选探测与选集：61 个候选中掌握 45 个，50 条活跃路径，B=5 覆盖 19 条。
- 完成 3B G1-G5 训练与评测，最终适配器保存在 `exp/adapters/3B_v2_clean_G1` 至 `3B_v2_clean_G5`。
- 保存 3B 结果、运行日志和配置快照到 `exp/results/3B_extension_report.md` 与 `exp/runs/exp001_3B_extension/`。
- 发现并修正训练组复用 PEFT 基座导致的重复适配器警告；以每组独立基座重跑 `3B_v2_clean`，最终数字以该版本为准。
- 完成研究内容二阶段 0–4：逐题损伤表、双模型归因、组件撤销、模块分解和留出预测验证；最终显影得分与单层撤销收益的 Spearman 相关约为 0.83。
- 完成研究内容三 R0–R5：能量空间投影、答案级约束消融和逐步梯度正交化；R5_surgery 的恢复率为 31.2%、F0 回弹为 16.3%、F1 复活为 21.3%、K 漂移为 4.8%。
- 完成 GSU 梯度手术遗忘及 11 个移植 baseline 的同口径对照；GSU 在 D_hop2 与 retain 的联合坐标上是当前自研方法的最佳工作点，但不对所有指标和 baseline 构成严格支配。

## 当前数据结论

- 数据包含 200 个实体、664 个事实簇。
- 每个事实簇实际包含五类字段：`hop1_unlearn`、`hop1_test`、`hop2_test`、`hop3_test`、`same_object`。
- 三个官方 split 共覆盖 563 个事实簇，互不重叠；101 个事实簇未被分配（其中 test split 实际为 464 簇）。
- 未分配事实簇暂不自动使用，避免改变上游实验口径。

## 当前计算环境

- Python：Conda 环境 `cuda126_env`（具体安装位置由各运行环境配置）
- Python 版本：3.10.18
- PyTorch：2.6.0+cu126
- CUDA：12.6
- GPU：NVIDIA GeForce RTX 4060 Ti
- 已有依赖：`torch`、`transformers`、`peft`、`datasets`、`accelerate`、`safetensors`
- 已补齐依赖：`hydra-core==1.3.2`、`hydra-colorlog==1.2.0`、`bitsandbytes==0.49.2`、`sentencepiece==0.2.1`、`rouge-score==0.1.2`、`tensorboard==2.18.0`、`scikit-learn==1.5.2`、`wandb==0.21.4`、`deepspeed==0.16.5`

`open-unlearning` 当前仓库的 `setup.py` 声明 Python >=3.11，但这不是本项目立即切换环境的依据。先在现有 Python 3.10 环境中直接运行源码和最小验证；依赖安装采用逐项补齐，暂不整体安装仓库 requirements，避免覆盖现有 CUDA/PyTorch 组合。

仓库自带的 Qwen2.5 配置默认使用 Hugging Face 模型名、`flash_attention_2` 和 `bfloat16`。本项目后续运行时需要在项目配置中覆盖为本地模型路径，并先尝试 `sdpa` 或 `eager` 注意力实现；不直接修改上游配置文件。`deepspeed` 在 Windows 导入时提示 `aio.lib` 和 `cufile.lib` 链接文件缺失，但模块导入成功；单卡首轮不启用相关异步 I/O 和 ZeRO 功能。

全局环境的 `pip check` 仍报告若干与本项目无关的已有包冲突，主要来自 `lerobot`、`pyriemann`、`robosuite` 和 `umap-learn`；本次没有为消除这些冲突而升级或删除其他包。

## 下一步

1. 对 1.5B 主线和 3B clean 结果补做逐题配对显著性、置信区间和多 seed；3B 当前只能作为单 seed 扩展验证。
2. 将公共知识加入 `R` 禁区，重选 F1 并重训 G2'，验证撤回权限约束是否能减少公共知识误伤。
3. 扩大 retaining 与通用保持集，比较 GSU、NPO、PDU 等方法在相同预算下的保持效用。
4. 延伸 R5：降低恢复/回弹耦合并提升 MCQA 恢复，直到回弹与 F1 复活硬指标有明确结论；同时补做 3B 恢复复现。
5. 整合论文正文、图表和 WWW 匿名投稿材料；不得把单 seed、单模型结果写成普遍结论。

## 当前判据与证据缺口

| 研究内容 | 已确认结果 | 尚未通过或尚需补强 |
|---|---|---|
| 内容一 | 1.5B 上 G2 hop2=0.255，低于 G1=0.363 和 G4=0.311；3B clean 上 G2=0.236，低于 G1=0.255、与 G4=0.236 持平 | R 禁区、扩大 retaining、多 seed 与配对统计；3B 不宣称显著优于随机 |
| 内容二 | ρ=0.824（G2 内部）、0.830（G1→G2 留出）；层/模块干预与机制分析完成 | 多 seed、独立事实留出和正式 Fisher 对照仍需补强 |
| 内容三 | R5_surgery 将回弹/恢复比降至 0.52，K 漂移达到 4.8% | F0 回弹 16.3% > 10%，F1 复活 21.3% > 5%，MCQA 恢复仍低于 logprob 恢复 |
| GSU | D_hop2=0.000、retain=0.485、K=0.604 | 与 PDU 在 same_object、F1_hop2 等轴上各有优劣，需多 seed 后再定稿结论 |

## 管理规则

- `data/faithun/` 只保存上游原始数据及其说明、校验信息。
- 冻结评测协议保存在 `docs/评测协议-v1.md`；协议变化必须升级版本并新建对应 run。
- 派生数据、探测结果、图结构和分析结果统一保存在 `exp/results/`；运行配置与日志保存在 `exp/runs/`，LoRA 适配器保存在 `exp/adapters/`。
- 不覆盖原始数据，不在没有协议版本的情况下修改数据筛选口径。
- 每个实验记录数据版本、模型版本、代码提交、随机种子和输出路径。
