# -*- coding: utf-8 -*-
"""研究内容一 · 阶段6 真实结果图表（Qwen2.5-1.5B, v2 超参）"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

WS = Path(__file__).resolve().parents[2]
RES = WS / "exp" / "results"
FIG = WS / "研究内容一-示意图"

v2 = json.load(open(RES / "eval_results_1p5B_v2.json", encoding="utf-8"))
cands = json.load(open(RES / "candidates.json", encoding="utf-8"))["facts"]
greedy = json.load(open(RES / "greedy_curve.json", encoding="utf-8"))

GROUPS = ["orig", "G1", "G2", "G3", "G4", "G5"]
LABELS = {"orig": "orig", "G1": "G1 只忘$F_0$", "G2": "G2 $F_0$+贪心$F^*$(5)",
          "G3": "G3 $F_0$+全候选(61)", "G4": "G4 $F_0$+随机5", "G5": "G5 只忘$F^*$(5)"}
COLORS = {"orig": "#8C8C8C", "G1": "#DD8452", "G2": "#4C72B0", "G3": "#55A868", "G4": "#C49BD9", "G5": "#C44E52"}

# ---------- 图 A：三指标 × 六组 ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
metrics = [("hop1_unlearn", "直接遗忘 UA（hop1）↓"), ("hop2", "多跳恢复率（hop2）↓"), ("retain", "保留效用（retaining set）↑")]
for ax, (mk, title) in zip(axes, metrics):
    vals = [v2[g][mk] for g in GROUPS]
    bars = ax.bar([LABELS[g] for g in GROUPS], vals,
                  color=[COLORS[g] for g in GROUPS])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    if mk in ("hop1_unlearn", "hop2"):
        ax.axhline(1 / 3, color="black", ls="--", lw=1)
        ax.text(5.4, 0.35, "地板 1/3", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 0.9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=0.3, axis="y")
fig.suptitle("五组遗忘对比（Qwen2.5-1.5B-Instruct，GA-LoRA v2：lr5e-5, 3ep, KL α=10，真实数据）",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / "fig9_real_groups.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ---------- 图 B：扩展效率（hop2 恢复率 vs 遗忘集规模）----------
fig, ax = plt.subplots(figsize=(6.5, 4.3))
sizes = {"G1": 33, "G2": 38, "G3": 94, "G4": 38, "G5": 5}
for g in ["G1", "G2", "G3", "G4", "G5"]:
    ax.scatter(sizes[g], v2[g]["hop2"], s=140, color=COLORS[g], zorder=5,
               label=f"{g} ({sizes[g]} facts)")
    ax.annotate(g, (sizes[g], v2[g]["hop2"]), textcoords="offset points",
                xytext=(6, 6), fontsize=10)
ax.axhline(1 / 3, color="black", ls="--", lw=1, label="随机地板 1/3")
ax.axhline(v2["orig"]["hop2"], color="#8C8C8C", ls=":", lw=1, label=f"orig={v2['orig']['hop2']:.2f}")
ax.set_xlabel("遗忘集规模（事实数）")
ax.set_ylabel("多跳恢复率（越低越好）")
ax.set_title("扩展效率：G2 以 5 个扩展事实逼近 G3 全候选（Qwen2.5-1.5B，真实数据）", fontsize=11)
ax.set_ylim(0.15, 0.75)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig10_real_efficiency.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ---------- 图 C：探测过滤 + 贪心覆盖（真实）----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
fig.suptitle("探测过滤与贪心覆盖（Qwen2.5-1.5B-Instruct，真实数据）", fontsize=12, fontweight="bold")
ax = axes[0]
m = sum(1 for f in cands if f["mastered"])
bars = ax.bar(["外部候选图", "内生图（探测后）"], [len(cands), m],
              color=["#8C8C8C", "#4C72B0"])
for b, v in zip(bars, [len(cands), m]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v), ha="center", fontsize=11)
ax.set_ylabel("候选支持事实数")
ax.set_title(f"(a) 外部图→内生图：过滤 {len(cands)-m} 条 ({(len(cands)-m)/len(cands)*100:.0f}%)", fontsize=11)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
xs = [c[0] for c in greedy["curve"]]
ys = [c[1] for c in greedy["curve"]]
ax.plot(xs, ys, "o-", color="#4C72B0", lw=2, ms=4)
ax.axvline(5, color="#C44E52", ls="--", lw=1.2, label="B=5（本文 $F_1$）")
ax.axhline(greedy["P_total"], color="gray", ls=":", lw=1, label=f"活跃路径总数={greedy['P_total']}")
ax.set_xlabel("已选扩展事实数")
ax.set_ylabel("累计覆盖活跃路径数")
ax.set_title("(b) 贪心覆盖曲线（真实，$F_1$ 取 B=5）", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig11_real_greedy.png", bbox_inches="tight", dpi=150)
plt.close(fig)

print("saved fig9 / fig10 / fig11")
