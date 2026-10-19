# 探测驱动的知识遗忘与恢复

本仓库维护 WWW 2027 投稿相关的研究代码、实验方案、评测结果和论文图表。研究主线包括：探测驱动的遗忘范围扩展、遗忘损伤定位，以及对合法保留知识的定向恢复。当前主实验覆盖 Qwen2.5-1.5B，并有 Qwen2.5-3B 扩展实验。

## 当前状态

- 项目状态和最近进展：[`PROJECT_STATUS.md`](PROJECT_STATUS.md)
- 冻结评测口径：[`docs/评测协议-v1.md`](docs/评测协议-v1.md)
- 实验规划：[`研究内容一实验规划.md`](研究内容一实验规划.md)、[`研究内容二实验规划.md`](研究内容二实验规划.md)、[`研究内容三实验规划.md`](研究内容三实验规划.md)
- 证据与结果报告：[`docs/研究证据总账.md`](docs/研究证据总账.md)、[`exp/results/`](exp/results/)

## 仓库结构

```text
.
├── code/open-unlearning/   # 固定上游提交的 Git 子模块
├── data/faithun/           # FaithUn 数据、官方 splits、许可证与校验和
├── docs/                   # 协议和研究记录
├── exp/                    # 探测、训练、评测、分析代码与结果
├── WWW投稿模板/            # 匿名 LaTeX 起始文件和投稿说明
└── 研究内容一-示意图/       # 研究图表
```

模型权重、LoRA 适配器、训练断点、激活转储、私人开题材料和机器日志不纳入版本控制。FaithUn 数据依其上游 MIT 许可证分发；其他数据或第三方材料须遵守各自许可证。

## 获取代码

```powershell
git clone --recurse-submodules https://github.com/GarrisonRom/www2027_edit.git
cd www2027_edit
```

已克隆但未初始化子模块时，运行 `git submodule update --init --recursive`。

## 环境准备

建议使用 Python 3.10 或兼容版本。先按本机 CUDA 版本安装 PyTorch，再安装项目直接依赖：

```powershell
python -m pip install -r requirements.txt
```

1.5B 脚本从 `QWEN_1_5B_PATH` 环境变量读取本地模型目录；默认目录为仓库下的 `models/Qwen2.5-1.5B-Instruct`。运行 3B 实验时通过脚本的 `--model` 参数指定模型目录。模型权重需由每位开发者自行获取，并遵循模型发布方的许可证。

Windows PowerShell 示例：

```powershell
$env:QWEN_1_5B_PATH = 'E:\models\Qwen2.5-1.5B-Instruct'
python exp/probe/pre_experiment.py --max_test 100
```

实验输出默认写入 `exp/results/`、`exp/runs/` 和 `exp/adapters/`。适配器和大体积中间产物保存在本机，不要提交到 Git；请将运行参数、seed、数据校验和、模型标识及代码提交号记录在对应 run 记录中。

## 协作方式

请从 `main` 拉取最新版本，为每项工作建立短期分支；提交前检查 `git status`，确认没有模型权重、个人材料、密钥或本机路径配置。提交信息采用简洁的动词描述，例如 `Add GSU update-direction ablation`。研究协议变化、新旧实验结果和派生数据均应保留版本记录，不覆盖原始数据或既有 run。

详细约定见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
