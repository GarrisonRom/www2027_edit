# exp007 运行入口

本实验把已有的历史损伤归因和 exp006 的 F0/F1 梯度-路径分数整合起来。它不修改 exp006，也不自动启动训练。

## 后台批量启动

按 B0、B1、B2、B3、B2R、B3R 顺序串行运行六组；先完成 CUDA/bf16、自检、离线回归测试和数据预检，全部组结束后自动生成比较报告。任一阶段失败立即停止，保留现场，不自动重试或覆盖。

```powershell
& './exp/train/start_gsu_integrated_guidance.ps1'
```

仅显示批次计划（不训练、不创建批次）：

```powershell
& './exp/train/start_gsu_integrated_guidance.ps1' -DryRun
```

仅做真实数据/模型预检（不训练、不创建批次）：

```powershell
& './exp/train/start_gsu_integrated_guidance.ps1' -PreflightOnly
```

后台返回 batch ID、控制进程 PID 和状态文件位置。各批次快照与日志保存在 `exp/runs/exp007_gsu_integrated_guidance/batches/<batch_id>/`。训练期间电脑需保持运行，冻结清单中的代码、数据和历史归因文件不能改动。

## 一键启动

启动脚本会检查固定输入哈希，运行 CUDA/bf16 检查、自检和 exp007 离线回归测试、数据预检，然后在后台串行运行六组并自动汇总。任一步失败会停止；不自动重试或覆盖 run。启动命令：

```powershell
& './exp/train/start_gsu_integrated_guidance.ps1'
```

只显示计划，不启动训练：

```powershell
& './exp/train/start_gsu_integrated_guidance.ps1' -DryRun
```

只做数据/模型预检，不创建 batch、不训练：

```powershell
& './exp/train/start_gsu_integrated_guidance.ps1' -PreflightOnly
```

批次状态、启动清单和阶段日志保存在 `exp/runs/exp007_gsu_integrated_guidance/batches/<batch_id>/`。训练期间电脑需保持运行；不要修改启动快照清单中的代码、实验卡、数据或历史归因文件。

## 自检

```powershell
& 'D:\Anaconda\envs\cuda126_env\python.exe' -B exp/train/diagnose_gsu_integrated_guidance.py --self_test
```

## 预检

```powershell
& 'D:\Anaconda\envs\cuda126_env\python.exe' -B exp/train/diagnose_gsu_integrated_guidance.py --preflight --model 'E:\12models\modelscope\Qwen2.5-1.5B-Instruct'
```

## 单组运行

```powershell
& 'D:\Anaconda\envs\cuda126_env\python.exe' -B exp/train/diagnose_gsu_integrated_guidance.py --variant B2 --model 'E:\12models\modelscope\Qwen2.5-1.5B-Instruct' --run_dir exp/runs/exp007_gsu_integrated_guidance/runs/run_YYYYMMDD_HHMMSS_B2
```

可用组别：`B0`、`B1`、`B2`、`B3`、`B2R`、`B3R`。先运行 B0/B1，再运行 B2/B3 和随机对照；每组新建目录，不覆盖旧 run。

## 结果汇总

```powershell
& 'D:\Anaconda\envs\cuda126_env\python.exe' -B exp/train/diagnose_gsu_integrated_guidance.py --compare <run_B0> <run_B1> <run_B2> <run_B3> <run_B2R> <run_B3R> --comparison_dir exp/runs/exp007_gsu_integrated_guidance/comparisons/screen_YYYYMMDD_HHMMSS
```

每个 run 都会保存历史 prior、gradient_path_S、integrated_S、输入哈希、代码快照、逐步日志、全量审计和原始五集/六集指标。
