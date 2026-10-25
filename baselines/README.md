# Baselines：原文算法与共同评测

## 当前活动评测（2026-10-25 12:51）

runs/modelscope_eval_20261025_125129为活动队列，使用已验证的魔搭本地MMLU/PPL，优先能力
评测再续跑QA。上一个eval_resume_20261025_110652已停止，Original9640/GA7350评分记录已
登记导入；不要将旧running标记解释为活动进程。启动验证见modelscope_eval_launch_20261025.md。
离线Original2048token PPL及2题5shot MMLU实测通过，正式完整指标以新队列实际产物为准。

## 已下载的魔搭评测数据

MMLU/WikiText2已校验下载至D:/GSU_datasets/modelscope_eval_20261025_02。
MMLU57科test14042/dev285/validation1531；WikiText-2-raw-v1 test4358。
本地读取接口local_capability_data.py无需联网，路径登记capability_data.json。
来源、旧脚本首题丢失修复、哈希和验证见modelscope_data_20261025.md。
下载不自动改变正在运行的冻结评测快照，也不等于已产出MMLU/PPL分数。

## 2026-10-25 评测修复

正式训练权重仍在formal_lora_20261024_120150；旧一体化评测队列已停止。
新活动目录：runs/eval_resume_20261025_110652，四类评测分别保存、分别续跑。
状态以新目录queue_status.json和各Method/phase/status.json为准，旧running标记可能陈旧。
修复/验证记录：evaluation_repair_20261025.md。旧Original6027题和GA951题结果复用，不重训。
网络不可用时能力阶段明确失败，不将其算作完整成功；已算逐题结果继续保留。

## 当前本地比较采用显式LoRA适配

按用户要求，支持LoRA的算法可进入独立适配轨道。configs_lora保存Original加12种方法：
GA、GradDiff、NPO、SimNPO、RMU、DPOmis、DPOrej、CEU、PDU、WGA、SatImp、UNDIAL。
KLUE没有经过验证的等价LoRA神经元掩码实现，排除出此轨道，不用近似版本凑数。
此前默认拒绝FaithUn方法使用LoRA是防止静默改变原生复现的保护，并非这些目标无法适配。

适配只改变参数域：r8/alpha16/dropout0；保留各方法已核实的loss、批量、LR、预热、早停。
RMU仅为层5/6/7down-proj添加LoRA；其余方法使用语言decoder线性层。PDU仍需校准epsilon，
CEU仍披露短答案跳过；使用LoRA不解决这两项实验合同。原生轨道和适配轨道分开选择/报告。

    ./baselines/run.ps1 -Method GA -LoraAdapted -Smoke
    ./baselines/run.ps1 -Method RMU -LoraAdapted -Smoke
    # 正式运行移除-Smoke，完整MMLU添加-MmluFull
    & D:/Anaconda/envs/cuda126_env/python.exe -B -m baselines.preflight --method GA --lora-adapted --model E:/12models/modelscope/Qwen2.5-1.5B-Instruct

LoRA对照是原目标在受限参数域中的适配，不声称与全参数优化轨迹/性能等价。也不自动宣称
与C1预算完全一致：源方法学习率/批量/早停和RMU可训练组件数继续显式报告。

历史exp016是固定114步LoRA先导实验，不是此目录的正式结果。本目录独立维护新实现，
source_lock.json锁定依赖源码和输入副本，不从变化中的exp/train导入。
references保存FaithUn原文/MCQA模板和open-unlearning作者社区说明。
迁移到Qwen/FaithUn不等于原论文原模型逐位复现；recipes.py登记来源、参数和实现约定。

## 已落实的修正

- FaithUn轨道原生训练，batch4、最多150epochs，训练UA<=.33早停，不统一114步。
- DPO在forget偏好替代答案，retain偏好真实答案；原文拒答文本，beta搜索范围。
- RMU c20、retain权重20/50/100/150/200/300、原WMDP层5/6/7 down-proj范围。
- KLUE概率归因、失配正则、5%神经元、未遗忘筛选；FP32归因，显式gated映射。
- PDU显式校准epsilon，保证dual预热后训练，不默认填未知阈值。
- SatImp按作者说明gamma1/alpha.1；UNDIAL原文LoRA r8/alpha16、lr1e-4、有效batch32。
- SimNPO按作者仓库forget.yaml默认值beta2.5/gamma.1375、seed1001、batch4、weight_decay.01，
  一轮预热后线性衰减；保留原始GitHub blob9aeeac3c配置，明确不是Qwen上的已验证最优参数。
- 微批量按各目标token/sequence归约加权，避免答案长度不同时改变目标。

## 共同评测

Figure8指令与Question/Options/Your Answer，固定选项顺序。原图未规定列表分隔符，显式
使用JSON列表。原作者训练提示、优化器未公布的细节仍列为约定，不冒充已确认。
记录逐题分数：F0、改写、多跳、同答案、retain66、独立test464及其扩展、F1、覆盖多跳。
主评测为paper-prompt答案条件序列概率，附答案均值/历史整段评分敏感性。
可选贪心生成EM/tokenF1；MMLU57科5shot、每科10题固定抽样；WikiText2 PPL。
后两项复用云端窗口/token计数和sample-map代码，失败明确记录，不填0或宣称完整成功。
retain66参与训练时不称heldout，独立保持用test464；Original用相同入口。
--mmlu-full或PowerShell的-MmluFull可运行完整57科5shot；默认10题/科是明确标注的抽样。
论文3.5指标UA、UA扩展、TA、MA及不同SA汇总范围的Score分别记录，不复制历史脚本的
自定义AA命名冒充论文指标。SA原始代码汇总范围不可得时，保留两种明确命名的版本。

## 项目根目录命令

    & D:/Anaconda/envs/cuda126_env/python.exe -B -m baselines.bootstrap
    & D:/Anaconda/envs/cuda126_env/python.exe -B -m pytest baselines/tests -q
    & D:/Anaconda/envs/cuda126_env/python.exe -B -m baselines.train --method RMU --plan
    & D:/Anaconda/envs/cuda126_env/python.exe -B -m baselines.train --method Original --model E:/12models/modelscope/Qwen2.5-1.5B-Instruct --run-dir baselines/runs/original_smoke_unique --smoke --behavior-only

完整Original评测移除--smoke和--behavior-only，可加--generation。
原生训练先用--method DPOrej等在独立目录执行--smoke，通过后新目录正式运行。
--plan只登记搜索，不启动数百次训练；所有trial需留档，按事前规则选择最低可达LR。
没有完成更低LR候选不能称最低可达LR。网格离散化不是原论文公开的精确列表。
PDU需--retain-eps并满足初始retain NLL校准。能力依赖通过requirements-eval.txt安装到
专用环境或本目录.deps；本地import已验证，数据下载/实际MMLU仍须真实验证。

原生FP32 master/Adam需要较大主存，32GB机器当前余量可能不足。资源不足明确失败，
不能自动换LoRA参与原生比较。测试/冒烟不是完整论文结果；正式结论仍需开发集调参、
完整能力评测与多seed。其他模型、数据和原文未规定的约定始终随配置披露。

详细验收事实见verification.md；机器资源预检见preflight_GA_20261024.json；全9640条数据
检查见preflight_data_20261024.json。正式训练前可执行：

    & D:/Anaconda/envs/cuda126_env/python.exe -B -m baselines.preflight --method RMU --model E:/12models/modelscope/Qwen2.5-1.5B-Instruct --data

当前实现转移不等于完整作者结果复现；不要将smoke、旧exp016表或能力缺失端点标为新正式结果。
同步到Linux时保留vendor/runtime、references和source_lock.json；.deps中的Windows轮子不
同步，按requirements-eval.txt在目标机重新安装。无.git的代码快照仍可运行并记录源码哈希。
