# 研究文档：只维护四份正文

需要单文件读全部细节时，使用[GSU实验技术总报告](GSU实验技术总报告.md)：包含方法演变、公式/伪代码、完整大小表、逐seed差值、故障与来源哈希。这是2026-10-25证据快照，不替代下面四份日常文档或实时主表。

2026-10-25内容合并。方案、大纲、假设、数据筛选已合并，KLUE与baseline审计合并；重复成绩表、旧“尚未启动”状态和故障流水账从日常说明移出。旧入口仅跳转，不再更新。

| 要读什么 | 唯一正文 | 不在这里重复维护 |
|---|---|---|
| 研究主线、C1/C2、RQ、缺失对照、数据权限、统计与写作 | [研究与验证方案](plans/GSU最终候选与帕累托验证设计.md) | 实时状态、逐轮成绩 |
| 方法配方差异、KLUE、实现风险、验收门 | [Baseline统一审计](audits/本地baseline复现审计_20261024.md) | 执行队列的完成数 |
| 每个指标的含义、方向、MMLU/PPL与计分边界 | [指标说明](reports/云端GSU-C2与指标说明.md) | 主表、C组排名 |
| DUMIC的核心方法与本项目关系 | [精简文献笔记](literature/论文精读-DUMIC深度遗忘.md) | 重复待办和跨论文成绩排名 |

## 事实各归一处

- 当前任务：[PROJECT_STATUS](../PROJECT_STATUS.md)；成绩：[云端主表导航](../exp/analysis/cloud_comparison_20261022/README.md)。本地表可能落后云端，本文不复制其数字。
- 本地C组：[exp010最终八组表](../exp/runs/exp010_gsu_step_routing/final_comparison_20261023.md)；其他旧结果：[历史结果索引](../exp/results/README.md)。
- 可执行配方与来源：[baselines/README.md](../baselines/README.md)；论文：[main.tex](../WWW投稿模板/WWW2027/GSU中文草稿/main.tex)。
- 历史协议、主张和决定：[评测v1](评测协议-v1.md)、[证据总账](研究证据总账.md)、[决策记录](06_decisions.md)、[方法版本](method_versions/)。它们是版本历史，不通过删改旧决定消除表面矛盾。

## 归档与兼容

[合并前原文ZIP](archive/markdown_before_content_merge_20261025.zip)含10份旧正文和旧导航，可恢复被压缩的细节；[归档清单](archive/content_merge_20261025.json)记录原路径、哈希和合并去向。仅对这份约52KB的文档归档添加Git忽略例外，防止协作时丢失原文；没有放开模型或其他ZIP的忽略规则，本次不自动提交。

根目录三份研究规划、exp/results历史报告、run内card/run_log/源码快照、第三方README、baselines/vendor及来源锁不动。旧同名路径只是兼容页；日常只读四份正文，新增观察更新对应职责文件，不再新增一份“最新总结”。
