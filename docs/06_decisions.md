# Decision Log

| date | trigger | evidence_summary (exp/run/probe) | failure_layer | strengthen_gate (probes / 跳过理由) | decision | old_claim | new_claim_if_any | frozen_assets | rerun_required |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-19 | H-GSU-UPD-1 single-seed screen failed its frozen advancement gate | exp002: M3C `0/12` primary; M3U `1/12`, retain `13/33`, F1_hop2 `9/75`; see run_20261019_1631_m3c_train_eval and run_20261019_1631_m3u_train_eval | mechanism (provisional; aggregate local constraint did not transfer to the preregistered primary/retain combination) | probe01 proposes per-anchor diagnostics and M3A intersection projection; not run. Project-level strengthen budget is unknown; no claim narrowing or multi-seed authorization. | strengthen | Preserve the current GSU claim scope and report single-seed limits. | none | M3 adapter, M3C/M3U adapters, FaithUn data/splits, exp002 thresholds, raw JSONL and run logs | yes, only after a new card and budget approval |
| 2026-10-20 | M3U diagnostic tested whether the post-hoc correction improves actual sampled-anchor answer log probabilities | run_20261020_1115_m3u_diagnostic_rerun: `437/456` paired anchor log-probability improvements, mean `+0.656008`; final metrics unchanged at primary `1/12`, retain `13/33`, F1_hop2 `9/75`; individual displacement conflicts remain `257/456` | mechanism boundary (aggregate local correction works, but transfer to individual anchors and cumulative retain is unsupported) | Direct-probability diagnostic completed. Do not alter M3U in place; M3A per-anchor intersection remains a separately proposed probe. Budget and new card still required. | strengthen | M3U's aggregate first-order correction is not sufficient evidence of final retain protection. | Local correction is supported; final benefit is not. | Diagnostic JSONL/metrics, historical M3U metrics, probe01, exp002 thresholds | yes, M3A only after budget approval and a separate preregistered card |
| 2026-10-20 | M3A tested the per-anchor intersection projection derived from probe01 | run_20261020_1332_m3a_train_diagnostic: primary `3/12`, retain `11/33`, F1_hop2 `9/75`; direct anchor log-probability improved on `439/456` paired observations | mechanism partially supported, final retain transfer failed | One fixed-seed M3A run completed. The primary improved, but retain missed the `14/33` gate; no multi-seed confirmation or further projection variant. | harvest/redirect | Per-anchor constraints may fix the aggregate-to-individual gap. | They improve local anchor behavior and primary accuracy, but do not protect cumulative retain in this setting. | M3A diagnostics, exp003 card/run, GSU-v3 method record, exp002 baselines | no for M3A; new retain-coverage experiment would require a new card |
| 2026-10-20 | exp005 tested actual post-proposal logprob feedback with a fixed correction budget | `run_20261020_1714_fb16`: 114/114 steps, coverage `33/33`; raw decreases `1780/1824`, final decreases `838/1824`; relative to COV16 `866/1824` only `1.54` percentage points better; primary `3/12`, retain `19/33`, F1_hop2 `22/75`; maximum correction ratio within `0.30+1e-6` | mechanism boundary; local raw-to-final correction works, but the bounded controller does not transfer to a better final retain/trajectory point | Runtime sanity passed and `1750/1824` paired observations improved over raw AdamW, but final mean delta `-0.053329` was below COV16 `+0.026811` and retain missed the `20/33` gate. The candidate was not searched further under this card. | harvest/redirect | Actual logprob feedback is sufficient to reduce part of the raw proposal damage under a trust budget. | It is not sufficient to improve COV16's final retain and forgetting trade-off in this setting; no multi-seed claim. | exp005 card, FB16 metrics/diagnostics, GSU-v5 method record, COV16 reference | no; a new trajectory candidate requires a new card |

decision: harvest/redirect

## 2026-10-22：记录讨论中的开放解释

本次讨论没有改变现有 claim 或实验 verdict。新增开放假设 H12–H16，集中在：

1. retaining 与 forget 是否只在事实簇索引上互斥，而在实体、答案和多跳路径上存在结构关联；
2. retain 下降中有多少是目标相关的必要代价，多少是无关知识误伤；
3. 3B 的更强事实掌握/实体泛化是否会导致更强的共享路径依赖；
4. 多记忆组件定位是否有机会替代 F1 扩展监督。

当前证据只支持“这些解释值得测试”，不支持任何一个解释已成立。详细预测和验证设计见 `docs/待验证假设与验证矩阵.md`。后续若实施，必须新建实验卡并冻结分层规则、共同题目、seed 和 kill rules。

## 2026-10-22：建立 GSU-v7 整合候选

依据 exp006 的 A2/A3 与随机对照结果，建立新实验 `exp007_gsu_integrated_guidance`，不修改 exp006。新候选把历史 G1/G2 FFN 层级损伤 prior 与当前 F0/F1 梯度-活跃路径分数相乘；由于现有证据没有完整 attention 矩阵级归因，attention 使用中性 prior，并在方法文档中明确记录证据缺口。

当前决策是 `strengthen`，不是 `accept`：先做代码自检和单 seed 筛选，比较 B0/B1/B2/B3/B2R/B3R；只有结构化综合分数超过随机对照，且 F1_hop2 有方向性改善，才考虑另建多 seed 卡。

## 2026-10-22：A/B复查后的帕累托审计与有界补强草案

不更改既有claim、实验卡或历史成功门。A/B已完成：B2覆盖切片44/75优于B0的52/75，
但F0为13/33对8/33、全部hop2为49.44%对44.94%；A3全部hop2为32.21%。
现有证据不支持把B2命名为综合最优。覆盖切片/留出retain坐标下B2被扩展监督B1支配，
在F0-only有限候选中非支配；该描述既非全局最优也非统计确认。

新增设计 `docs/GSU最终候选与帕累托验证设计.md`、GSU-v8方法草案和exp010八组
机制筛选草案，决策为strengthen/untested。拟在AdamW实际位移上隔离组件路由、
F1共享方向与F0进展护栏；F1仅作指导的信息访问必须披露。 proposed预算912步/16GPU小时，
尚未审批/实施/启动，不自动追加训练。在全部hop2/留出retain坐标下A3支配B2，
不能以B2的覆盖切片单独定义最优。远程不同模型baseline不能与本地GSU合算前沿；
正式画图仍需风格确认、同模型GSU及完整89路径评测，retain关联分层须另冻结。

## 2026-10-22：论文工作稿统一到云端验证

按用户要求修订GSU中文草稿main.tex：最终有效性、消融、解释性和成本实验全部以
云端两模型同协议为准；本地A/B只作设计来源。详细写入F0-only信息权限、实际位移
路由、进展护栏、条件推导和六个RQ，不把待实现方法写成已成功。
旧exp010草案及历史运行保持不变；新云端设计在没有GSU结果前指定无旧模型先验的
C2为公共主候选，理由为跨架构适用性，旧先验分支须重新校准和登记。
本次为论文/实验设计修订，不启动训练、不改变已有方法结果或将候选升级为accept。

## 2026-10-22：云端baseline全指标排序与源码同步

按用户要求采集exp008/009全部14个实测端点的100个证据文件，保留逐题输出、
配置、训练日志与哈希，分析归档于exp/analysis/cloud_comparison_20261022。
每模型分别对F0/改写/覆盖多跳/保护池/非锚点的正确率、答案LP、margin及LP变化
排序，Original与扩展监督作为独立参照，未运行GSU无成绩与排名。F1及旧损伤切片
不强设一致效用方向。C2仅给出相对GradDiff的探索目标区域，不作为预测或成功证据。
现有云端七项核心源码与本地一致；开发源码同步到新云端快照，历史run和代码不覆盖。
本次只同步与分析，不启动训练，不改变claim或旧实验门，仍待完整89路径与GSU实测。

## 2026-10-22：授权单候选云端比较与补充KLUE

用户要求选一个方法在云端比较，选择公共主候选GSU-C2，新增exp011独立卡及实现；
只跑两模型seed42各114步，不自动启动整组消融。先通过实模2步更新/保存重载检查。
用户进一步要求加入FaithUn/KLUE：FaithUn是已有基准，KLUE尚无实测端点，
列为必须补齐的定位baseline；审查原算法和官方代码后另卡适配，未完成不伪造排名。

exp011现已完成：Llama3.2-3B和Qwen3.5-4B各114步seed42。覆盖多跳/F0/非锚点retain
计数与GA相同；C2全部89路径宏回忆分别0.3034/0.3858，baseline全89路径尚未统一复评，
因此目前没有观测到C2相对GA的优势。exp012已派发独立云端评测快照，不训练模型，
补做16端点的全89路径、MMLU-10/科目与WikiText-2 PPL。新结果将回填工作稿，未完成前留空。

## 2026-10-22：新增数据与能力评测资源筛选

用户提供RWKU、llm2graph、Unlearning Mirage、MQuAKE及MMLU/PPL线索。核查后
建议优先补通用能力审计MMLU和WikiText tokenPPL；MQuAKE-v2原始链作为外部多跳
验证的优先改造资源，RWKU另建实体级遗忘轨。llm2graph当前不跑，Mirage纳入直接
相关工作而不必全套复跑，避免把图式恢复现象作为独有贡献。KLUE仍为必补baseline。
本地OpenUnlearning已有MMLU任务入口，未发现专用PPL配置；harness的word/bytePPL
不能冒充tokenPPL。详细建议与来源见docs/新增数据集与通用能力评测筛选.md。
本轮只审查整理，不安装依赖、不下载大数据、不启动训练/评测、不改当前C2环境。

## 2026-10-23：补充云端 GA/C2 方法与指标解释

用户反馈云端比较只报少量数字、缺少方法和指标说明。新增 docs/云端GSU-C2与指标说明.md，
解释 GA-F0 与 GSU-C2 的目标函数/更新路由区别、F0/F1/多跳/retain/log-prob/margin/
MMLU/PPL/成本等口径，列出 exp008/009 与 exp011 当前可比计数，并明确全89 baseline、
MMLU/PPL、算力匹配、消融和多 seed 的缺口。exp011卡和研究证据总账已链接该说明。
结论限定为：两模型上 C2 与 GA-F0 的主要已测 F0、覆盖多跳和 heldout retain 计数相同；
这是无可见增益，不是 C2 已证明更差或更优。exp012 的 batch_20261023_05 完成 PPL
短 smoke 与全89行为审计，但在 MMLU 输出 JSON 序列化 torch.dtype 时失败，正式端点未开始。
失败现场保留，修复后须用新 batch；状态详见 exp012 卡及 docs/云端GSU-C2与指标说明.md。


## 2026-10-23：本地 C 组已完成结果（部分批次）

来源：`exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1`，Qwen2.5-1.5B，seed42，各114步。C0八步回放最大差0，五项正式计数与历史B0一致，复现门通过。

| 方法 | 全89路径宏回忆 ↓ | 覆盖多跳 /75 ↓ | F0 /33 ↓ | F1 /5 | 留出retain /33 ↑ | 护栏回退步数 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 44.94% | 52 | 8 | 5 | 15 | 0 |
| Cg | 32.58% | 9 | 12 | 1 | 10 | 98 |
| C1 | 27.72% | 0 | 12 | 0 | 12 | 103 |
| C2 | 29.96% | 7 | 12 | 0 | 10 | 109 |
| C3 | 30.71% | 7 | 13 | 0 | 10 | 108 |

观察：C1在已完成Cg/C1/C2/C3中，全路径回忆最低；相对Cg/C2，F0残留相同且留出retain更高。C1未支配C0：直接遗忘和retain均较差。C1的F1直接事实为0/5，不能解释为保持支持事实而独立切断推理。方向过滤C2及历史先验C3在本次观察中未超过C1，不证明梯度信息无效；C1本身通过F0/F1梯度构造组件分数。

状态核验：C3R文件停于2026-10-23 14:46:41、第36/114步；检查时未发现对应Python进程。队列疑似中断，不能按旧status=running宣称正常运行。C3R无最终指标，C3D/C4未完成；不得填零或参加排名。本次不重启、不覆盖原始run，也不改云端主候选。C3仍为本地事前主候选，C1只为事后探索性领先候选；随机对照、多seed与云端独立确认待完成。

详细逐题证据见各变体metrics.json及diagnostics.jsonl，最终选择不用38/76步最优checkpoint。


## 2026-10-23 最终完成：本地 C 组八变体筛选

批次分为原批次与续跑批次，均为 Qwen2.5-1.5B、seed42、114步/变体。C3R在原运行第36步中断后，从相同初始化在独立续跑目录完整重跑；C3D、C4也在续跑目录完成。C0八步回放与历史B0数值最大差0，五项正式计数复现门通过。八组均有最终 metrics.json；续跑总状态为 completed。

主指标按全89路径宏回忆升序排列：

| 排名 | 方法 | 全89路径宏回忆 ↓ | 覆盖多跳 /75 ↓ | F0残留 /33 ↓ | F1 /5 | 留出retain /33 ↑ | 护栏回退 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | C1 | 27.72% | 0 | 12 | 0 | 12 | 103 |
| 2 | C4 | 29.59% | 7 | 12 | 0 | 10 | 107 |
| 3并列 | C3R | 29.96% | 7 | 12 | 0 | 10 | 109 |
| 3并列 | C2 | 29.96% | 7 | 12 | 0 | 10 | 109 |
| 5 | C3 | 30.71% | 7 | 13 | 0 | 10 | 108 |
| 6 | C3D | 31.46% | 9 | 13 | 0 | 10 | 105 |
| 7 | Cg | 32.58% | 9 | 12 | 1 | 10 | 98 |
| 8 | C0 | 44.94% | 52 | 8 | 5 | 15 | 0 |

C1在主指标排名第一，C4第二；C2与C3R在全89路径指标并列第三。C1相对C4降低全路径宏回忆1.87个百分点，覆盖多跳为0/75而C4为7/75；两者F0残留均为12/33，C1留出retain为12/33、C4为10/33。C1 F1直接题为0/5。单seed结果仅构成探索性筛选：不声称统计或全局最优；C3仍是事前指定主候选，C1是事后观察到的经验领先候选。因C1的F1直接题全部答错，不能声称其在保留支持事实时独立切断多跳推理。五项门槛和逐题原始证据见两批的 metrics.json、diagnostics.jsonl，以及 exp010/results_20261023.md。
