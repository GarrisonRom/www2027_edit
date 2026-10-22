# exp006 运行入口

本轮比较 A0–A4 与随机对照 A2R/A3R，检验“多条关联记忆只用于一次性组件定位，随后仅遗忘原始 F0”是否能替代直接扩展遗忘监督。

- 预注册：[card.md](card.md)
- 方法说明：[GSU-v6-memory-guided.md](../../../docs/method_versions/GSU-v6-memory-guided.md)
- 实现：[diagnose_gsu_memory_guided.py](../../train/diagnose_gsu_memory_guided.py)
- 离线回归测试：[test_gsu_memory_guided.py](../../train/tests/test_gsu_memory_guided.py)

所有命令在项目根目录执行。固定使用 `D:\Anaconda\envs\cuda126_env\python.exe`，无需切换环境或升级依赖。当前 1.5B 实验不是 3B 扩展；脚本会拒绝不符合冻结模型配置的输入。

## 不训练的检查

基础自检：

```powershell
& 'D:\Anaconda\envs\cuda126_env\python.exe' -B exp/train/diagnose_gsu_memory_guided.py --self_test
```

完整回归测试使用随机初始化的小型 CPU Qwen，不加载 1.5B 权重。临时产物仅用于验证实现，不计入 FaithUn 方法效果：

```powershell
& 'D:\Anaconda\envs\cuda126_env\python.exe' -B -m unittest discover -s exp/train/tests -p test_gsu_memory_guided.py -v
```

离线预检核对模型 config、原始数据校验和、split、固定 F1、旧评测切片和预算；不加载模型权重，不开始遗忘训练：

```powershell
& 'D:\Anaconda\envs\cuda126_env\python.exe' -B exp/train/diagnose_gsu_memory_guided.py --preflight --model 'E:\12models\modelscope\Qwen2.5-1.5B-Instruct'
```

预期：196 个组件；F0=33、F1=5；保护池=33；额外 retain 审计=66，含后 33 条保护留出；F0-only 与 F0+F1 都为 114 步。

## 正式训练

一键启动脚本会先检查 CUDA/bf16、自检和数据，然后在后台按 A0、A1、A2、A3、A4、A2R、A3R 串行训练；每组 114 步，全部完成后自动并排汇总。任何一组失败即停止，不自动调参或重试。

```powershell
& './exp/train/start_gsu_memory_guided.ps1'
```

只查看启动计划，不创建批次或开始训练：

```powershell
& './exp/train/start_gsu_memory_guided.ps1' -DryRun
```

脚本返回后台控制进程 PID 和批次目录。`batches/<batch_id>/status.json` 记录当前阶段、正在运行的组及完成列表；该目录同时保存启动快照、冻结文件哈希和各阶段日志。关闭当前对话不会要求控制进程退出，但电脑需保持开机、避免休眠。

训练期间不要改动冻结的训练代码、测试、数据、方法说明、实验卡和本说明。组间会检查哈希；检测到变化将停止，后续修改应使用新批次。脚本不会终止其他应用或已存在的训练进程。

以下为手动入口。每组从原始模型和相同 seed 重新初始化，不共享已经训练的 adapter。

单组示例：

```powershell
& 'D:\Anaconda\envs\cuda126_env\python.exe' -B exp/train/diagnose_gsu_memory_guided.py --variant A0 --model 'E:\12models\modelscope\Qwen2.5-1.5B-Instruct' --run_dir exp/runs/exp006_gsu_memory_guided/runs/run_YYYYMMDD_HHMMSS_A0
```

批量入口示例，替换日期标记后再执行。这里的循环只串行启动，不同时占用多份模型显存；任一组失败立即停止：

```powershell
$experimentStamp = 'YYYYMMDD_HHMMSS'
$experimentVariants = @('A0', 'A1', 'A2', 'A3', 'A4', 'A2R', 'A3R')
foreach ($experimentVariant in $experimentVariants) {
    $experimentRun = "exp/runs/exp006_gsu_memory_guided/runs/run_${experimentStamp}_${experimentVariant}"
    & 'D:\Anaconda\envs\cuda126_env\python.exe' -B exp/train/diagnose_gsu_memory_guided.py --variant $experimentVariant --model 'E:\12models\modelscope\Qwen2.5-1.5B-Instruct' --run_dir $experimentRun
    if ($LASTEXITCODE -ne 0) { throw "Stopped at $experimentVariant; preserve the failed run." }
}
```

不要将同一目录作为重跑目标。脚本拒绝已经存在的 run 目录，包括空目录；失败后应使用新 run_id，并在结果解读中注明重跑关系。本入口不提供在旧 run 中续写/覆盖的选项。

## 并排汇总

各组完成后，使用全新汇总目录。汇总仅读取现有结果，不启动训练，不修改来源 run：

```powershell
$experimentStamp = 'YYYYMMDD_HHMMSS'
$experimentVariants = @('A0', 'A1', 'A2', 'A3', 'A4', 'A2R', 'A3R')
$experimentRuns = $experimentVariants | ForEach-Object { "exp/runs/exp006_gsu_memory_guided/runs/run_${experimentStamp}_$_" }
& 'D:\Anaconda\envs\cuda126_env\python.exe' -B exp/train/diagnose_gsu_memory_guided.py --compare $experimentRuns --comparison_dir "exp/runs/exp006_gsu_memory_guided/comparisons/screen_${experimentStamp}"
```

输出 `comparison.md` 和 `comparison.json`，按 A0/A1 基线报告主指标、retain、F1_hop2、直接 F1 损伤、留出 retain 和方向性筛选门。缺少组别会明确标为 incomplete；混用初始状态、组件分数、代码/数据、锚点计划或同类训练计划时拒绝合并。

即使筛选门通过，也不是统计显著性结论。随机引导达到相同门时，不支持“共享记忆结构导致提升”的解释。

## 每个 run 的产物

| 产物 | 用途 |
|---|---|
| config.json | 开始训练前冻结的模型、环境、输入/源代码哈希和预算 |
| source_snapshot/、patch.diff、working_tree_status.txt | 保留实际执行代码，包含未提交的新文件 |
| resolved_config.json | 定位后分数签名、初始 LoRA 签名和实际组件配置 |
| component_scores.json | 每个组件 A/P/S、软权重、硬掩码及随机映射 |
| schedule.json、anchor_schedule.json | 每个样本的重复次数、顺序和逐步锚点索引 |
| initial_audits.json、initial_set_audits.json | 初始 F0/F1、关联记忆、retain 及评测切片的审计 |
| diagnostics.jsonl | 逐步 raw/final 概率、风险、线性方向、修正比例及累计变化 |
| cumulative_audit.jsonl | 38/76/114 步对全部保护池的实际累计 logprob 复测 |
| block_metrics.json、metrics.json | 原始五集/六集指标与逐题审计 |
| checkpoints/、adapter/ | 各 38 步段末的 LoRA 权重和最终 adapter |
| stdout.log、stderr.log、status.json、run_log.md | 运行状态、异常、耗时与显存峰值 |

模型权重本身不复制进 run；模型元数据有 SHA-256，权重文件仅记录大小和修改时间，不能把它当成完整权重内容哈希。

在正式结果产生前，不要将基础自检、离线预检或小模型实现测试作为算法效果证据。实际批次进度以 status.json、各组 diagnostics.jsonl 和最终 metrics.json 为准。
