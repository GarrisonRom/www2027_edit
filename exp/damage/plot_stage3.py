# -*- coding: utf-8 -*-
"""研究内容二 · 阶段3 绘图：图 D3 组件撤销收益曲线（G2）"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WS = Path(__file__).resolve().parents[2]
OUT = WS / "exp" / "results"
ATT = OUT / "attribution"
FIGDIR = WS / "研究内容一-示意图"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = {  # 基准 mean_lp（来自 per_item_G2 / per_item_orig）
    "D_so": (-11.828, -4.720), "D_hop2": (-9.579, -3.423),
    "hop1_unlearn": (-9.899, -3.576), "F1_hop2": (-7.521, -3.156),
    "K": (-5.761, -3.761),
}

recs = {json.loads(l)["config"]: json.loads(l)["sets"]
        for l in open(ATT / "ablation_G2.jsonl", encoding="utf-8")}

def frac(cfg, s):
    """恢复可恢复空间的比例"""
    g2, orig = BASE[s]
    return (recs[cfg][s]["mean_lp"] - g2) / (orig - g2)

# ---------- 图 D3 ----------
fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

# Panel A: 关键配置的恢复率 vs 回弹率
ax = axes[0]
cfgs = ["undo_mlp_27", "undo_mlp_22_27", "undo_mlp_rand6",
        "undo_neurons_top12", "undo_neurons_rand12"]
labels = ["层27 MLP", "层22–27 MLP", "随机6层 MLP", "top12 神经元", "随机12神经元"]
x = np.arange(len(cfgs))
w = 0.36
d_frac = [100 * np.mean([frac(c, "D_so"), frac(c, "D_hop2")]) for c in cfgs]
ax.bar(x - w/2, d_frac, w, color="#1f77b4", label="受损集 D 恢复率（占可恢复空间 %）")
rebound = [100 * frac(c, "hop1_unlearn") for c in cfgs]
ax.bar(x + w/2, rebound, w, color="#d62728", label="$F_0$ 遗忘回弹率（hop1，%）")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=12)
ax.set_ylabel("%")
ax.set_title("定向 vs 随机撤销：恢复收益与遗忘回弹\n（模型：Qwen2.5-1.5B + G2 adapter）")
ax.legend(fontsize=9)

# Panel B: 累积收益曲线（层 27→22 逐层加回；神经元 top-k）
ax = axes[1]
xs = list(range(0, 7))
d_curve = [0.0] + [100 * np.mean([frac(f"curve_mlp_27down_{i}", "D_so"),
                                  frac(f"curve_mlp_27down_{i}", "D_hop2")]) for i in range(1, 7)]
rb_curve = [0.0] + [100 * frac(f"curve_mlp_27down_{i}", "hop1_unlearn") for i in range(1, 7)]
f1_curve = [0.0] + [100 * frac(f"curve_mlp_27down_{i}", "F1_hop2") for i in range(1, 7)]
ax.plot(xs, d_curve, "o-", color="#1f77b4", lw=1.8, label="D 恢复率")
ax.plot(xs, rb_curve, "s--", color="#d62728", lw=1.8, label="$F_0$ 回弹率")
ax.plot(xs, f1_curve, "^:", color="#ff7f0e", lw=1.8, label="$F_1$ 复活率")
ax.set_xlabel("撤销 MLP 层数（自层 27 向下累加）")
ax.set_ylabel("%")
ax.set_title("层撤销累积收益曲线：恢复-回弹剪刀差")
ax.legend(fontsize=9)
ax.text(0.98, 0.05, "神经元 top-k 曲线（k=1,2,4,8,12）恢复率均 < 2%，\n与随机神经元无差别——层内损伤是分布式的",
        transform=ax.transAxes, ha="right", fontsize=8.5, color="#555",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ccc"))

# Panel C: 恢复-回弹权衡散点（每配置一个点）
ax = axes[2]
for c, lab, col, mk in [("undo_mlp_27", "层27", "#1f77b4", "o"),
                        ("curve_mlp_27down_2", "层27-26", "#4a9bd4", "o"),
                        ("curve_mlp_27down_3", "层27-25", "#6fb3de", "o"),
                        ("curve_mlp_27down_4", "层27-24", "#93c9e8", "o"),
                        ("curve_mlp_27down_5", "层27-23", "#b8ddf1", "o"),
                        ("undo_mlp_22_27", "层22-27", "#0b5394", "o"),
                        ("undo_mlp_rand6", "随机6层", "#7f7f7f", "x"),
                        ("undo_neurons_top12", "top12神经元", "#2ca02c", "^"),
                        ("undo_neurons_rand12", "随机12神经元", "#a9a9a9", "^")]:
    ax.scatter(100 * frac(c, "hop1_unlearn"), 100 * np.mean([frac(c, "D_so"), frac(c, "D_hop2")]),
               color=col, marker=mk, s=70, zorder=3)
    ax.annotate(lab, (100 * frac(c, "hop1_unlearn"), 100 * np.mean([frac(c, "D_so"), frac(c, "D_hop2")])),
                textcoords="offset points", xytext=(6, 4), fontsize=8.5)
ax.set_xlabel("$F_0$ 遗忘回弹率（%）→ 越小越好")
ax.set_ylabel("受损集 D 恢复率（%）→ 越大越好")
ax.set_title("撤销策略权衡图：左上 = 理想区\n（恢复多、回弹小）")
ax.grid(alpha=0.3)

fig.suptitle("图 D3 · 组件撤销因果实验：把 M′ 的组件换回 M，损伤真的修复吗？（Qwen2.5-1.5B，G2）",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
p = FIGDIR / "fig18_d3_ablation.png"
fig.savefig(p, dpi=160, bbox_inches="tight")
print("saved:", p)

# 汇总表
print(f"{'配置':<24}{'D恢复%':>8}{'hop1回弹%':>10}{'F1复活%':>9}{'K漂移%':>8}")
for c in cfgs:
    print(f"{c:<24}{100*np.mean([frac(c,'D_so'),frac(c,'D_hop2')]):>8.1f}"
          f"{100*frac(c,'hop1_unlearn'):>10.1f}{100*frac(c,'F1_hop2'):>9.1f}{100*frac(c,'K'):>8.1f}")
