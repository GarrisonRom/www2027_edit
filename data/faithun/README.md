# FaithUn 数据集

## 当前状态

- 状态：已下载并冻结
- 冻结日期：2026-10-17
- 上游仓库：https://github.com/yny0506/faithun
- 上游提交：`6a912b7d71b3259740aab92071ae3c4a7eab2b00`
- 原始数据：`FaithUn_Dataset.json`

## 文件结构

```text
data/faithun/
├── FaithUn_Dataset.json
├── splits/
│   ├── forget_set_split.txt
│   ├── retaining_set_split.txt
│   └── test_set_split.txt
├── README.md
├── UPSTREAM_README.md
├── UPSTREAM_LICENSE
└── checksums.sha256
```

`FaithUn_Dataset.json` 和 `splits/` 下的三个文件是上游原始文件。实验脚本不得直接改写这些文件；后续清洗、索引和模型探测结果统一放入单独的派生目录。

## 已核验结构

- 实体数量：200
- 事实簇数量：664
- 每个事实簇均包含：`hop1_unlearn`、`hop1_test`、`hop2_test`、`hop3_test`、`same_object`
- 事实图字段：`graphs`
- Wikidata ID 字段：`id_graphs`
- 每个问题集同时提供问题、正确答案和干扰答案选项

数据文件的顶层结构为：

```text
{
  "data": {
    "Q<id>": [cluster, ...]
  },
  "description": { ... }
}
```

`hop2_test` 和 `hop3_test` 的问题按事实簇保存为多跳问题组；`id_graphs` 保存对应的实体和关系 ID，可直接用于后续支持路径提取。

## 官方 split 核验

split 文件使用按数据文件顺序编号的全局事实簇索引，当前核验结果如下：

| split | 簇数量 | 索引范围 |
|---|---:|---:|
| forget | 33 | 23-661 |
| retaining | 66 | 0-658 |
| test | 464 | 1-663 |
| 合计已分配 | 563 | 无重叠 |

还有 101 个事实簇没有出现在这三个 split 文件中。该现象来自上游文件的实际内容，不是本地下载或复制错误。本项目暂不把未分配簇自动归入任何 split；后续实验若使用它们，必须在协议文件中单独声明用途。

## 项目使用约定

1. `forget_set_split.txt` 是选择初始遗忘簇 `F0` 的候选范围。
2. 研究内容一的第一条种子实验应先固定一个簇，再扩展到多个簇。
3. `hop1_unlearn` 用于初始遗忘对象，`hop1_test`、`hop2_test`、`hop3_test` 和 `same_object` 用于评测或路径分析。
4. 主评测口径暂定为 FaithUn 的三选一 MCQA；生成式概率仅作为辅助指标。
5. 数据文件版本、split 版本、提示模板、随机种子和模型版本必须一起写入 `protocols/v1/` 后再开始正式实验。

完整上游说明和许可证分别保存在 `UPSTREAM_README.md` 与 `UPSTREAM_LICENSE`。
