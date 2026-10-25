# 探测驱动的知识遗忘与恢复

本仓库维护 WWW 2027 投稿相关的研究代码、实验方案、评测结果和论文图表。当前研究重点是关联知识指导的 GSU 遗忘更新及 baseline 对齐验证；早期范围扩展、损伤定位与恢复实验作为历史依据保留。覆盖本地 Qwen2.5-1.5B/3B，以及云端 Llama/Qwen 的小模型和8B实验；未完成实验不等于已确认结论。

## 当前状态

入口整理日期：2026-10-25。本文负责导航；运行进度以对应 run 的 status、日志和原始指标为准。

| 用途 | 唯一默认入口 | 边界 |
|---|---|---|
| 项目状态 | [PROJECT_STATUS.md](PROJECT_STATUS.md) | 最新工作与历史里程碑分开 |
| Markdown文档导航 | [docs/README.md](docs/README.md) | 方案、协议、审计、报告和文献分类；旧路径仅跳转 |
| 全实验技术详报 | [GSU实验技术总报告](docs/GSU实验技术总报告.md) | 约1900行、公式/伪代码、大小表和来源哈希；带时间的快照 |
| 研究与方法方案 | [合并后的统一方案](docs/plans/GSU最终候选与帕累托验证设计.md) | 大纲、假设、数据与验证要求只维护一份 |
| 云端结果总览 | [云端比较入口](exp/analysis/cloud_comparison_20261022/README.md) | 默认主表 report_v3_klue_pending；本地副本可能落后云端 |
| 新的独立 baseline 实现 | [baselines/README.md](baselines/README.md) | 原生/LoRA适配分轨，vendor与来源锁不手工去重 |
| 8B三种子运行 | [exp016实验卡](exp/runs/exp016_8b_multiseed/card.md) / [队列脚本](exp/train/run_8b_multiseed.py) | seed42/43/44；不修改正在运行的冻结快照 |
| C1/C2共同实现 | [C组worker](exp/train/run_local_gsu_c_group.py) | 与旧 diagnose 模块仍有依赖，不能按文件名删除 |
| 中文论文工作稿 | [main.tex](WWW投稿模板/WWW2027/GSU中文草稿/main.tex) | 不是投稿定稿；表格/文字状态需随证据更新 |

### 文档与归档约定

根目录三份“研究内容实验规划”保留原路径，标为历史规划；docs按plans、audits、reports、literature和method_versions分类，协议、证据总账和决策保留在顶层。旧同名路径只有兼容跳转，不重复维护正文。exp/results中的旧报告是历史分析，统一从其README阅读，不是新的云端排行榜。原始run中的配置、失败日志、逐题指标、源码快照与哈希都是复现材料，不能按重复文件批量删除。

实验编号存在历史重号：exp016_8b_multiseed 与 exp016_local_baselines。引用必须使用完整目录名和run_id；暂不改名，避免破坏路径、来源锁及已启动任务。

- 项目状态和最近进展：[`PROJECT_STATUS.md`](PROJECT_STATUS.md)
- 冻结评测口径：[`docs/评测协议-v1.md`](docs/评测协议-v1.md)
- 实验规划：[`研究内容一实验规划.md`](研究内容一实验规划.md)、[`研究内容二实验规划.md`](研究内容二实验规划.md)、[`研究内容三实验规划.md`](研究内容三实验规划.md)
- 证据与结果报告：[`docs/研究证据总账.md`](docs/研究证据总账.md)、[`exp/results/`](exp/results/)

## 仓库结构

```text
.
├── code/open-unlearning/   # 固定上游提交的 Git 子模块
├── data/faithun/           # FaithUn 数据、官方 splits、许可证与校验和
├── docs/                   # 协议、方法版本、决策与研究记录
├── baselines/              # 独立baseline实现、固定配方、锁定来源；runs和.deps不提交
├── exp/                    # 探测、训练、评测、分析代码与结果
├── WWW投稿模板/            # 匿名 LaTeX 起始文件和投稿说明
└── 研究内容一-示意图/       # 研究图表
```

模型权重、LoRA 适配器、训练断点、激活转储、私人开题材料和机器日志不纳入版本控制。FaithUn 数据依其上游 MIT 许可证分发；其他数据或第三方材料须遵守各自许可证。

当前代码职责：exp/train为云端/历史实验运行实现，baselines为独立对齐实现，tools为下载/同步/存储维护，exp/analysis为派生分析。两套训练实现尚未证明可互换；本次入口整理不合并算法或改配方。

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
