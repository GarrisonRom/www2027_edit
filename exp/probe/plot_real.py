# -*- coding: utf-8 -*-
"""阶段1 预实验真实数据图表"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

WS = Path(__file__).resolve().parents[2]
OUT = WS / "exp" / "results"
FIG = WS / "研究内容一-示意图"

recs = [json.loads(l) for l in open(OUT / "pre_experiment_records.jsonl", encoding="utf-8")]
consist = [r["hop1_consistency"] for r in recs]
h2_all = [it["acc"] for r in recs for it in r["hop2_items"]]
h2_g_h1 = [it["acc"] for r in recs if r["hop1_correct"] for it in r["hop2_items"]]

taus = [x / 20 for x in range(10, 19)]
curve = [sum(1 for c in consist if c >= t) / len(consist) for t in taus]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax = axes[0]
ax.plot(taus, curve, "o-", color="#4C72B0", lw=2)
ax.axvspan(0.6, 0.8, color="gray", alpha=0.12)
ax.text(0.7, 0.45, "协议扫描区间", ha="center", fontsize=9, color="#555555")
ax.set_xlabel(r"掌握判定阈值 $\tau$")
ax.set_ylabel("模型掌握的事实比例")
ax.set_title("(a) τ 敏感性曲线（真实数据）", fontsize=11)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

ax = axes[1]
bins = [0, 0.125, 0.375, 0.625, 0.875, 1.01]
labels = ["0/4", "1/4", "2/4", "3/4", "4/4"]
cnt = [sum(1 for c in consist if bins[i] <= c < bins[i + 1]) for i in range(5)]
bars = ax.bar(labels, cnt, color=["#C44E52", "#DD8452", "#DD8452", "#55A868", "#4C72B0"])
for b, n in zip(bars, cnt):
    ax.text(b.get_x() + b.get_width() / 2, n + 1, str(n), ha="center", fontsize=10)
ax.set_xlabel("4 问法一致通过数")
ax.set_ylabel("事实数")
ax.set_title(f"(b) hop1 一致率分布（n={len(consist)}）", fontsize=11)
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
import math
n = len(h2_g_h1)
p = sum(h2_g_h1) / n
ci = 1.96 * math.sqrt(p * (1 - p) / n)
ax.bar(["整体", "hop1 正确\n前提下"], [sum(h2_all) / len(h2_all), p],
       color=["#8C8C8C", "#4C72B0"], yerr=[0, ci], capsize=5)
ax.axhline(1 / 3, color="#C44E52", ls="--", lw=1.5, label="随机地板 1/3")
ax.text(1, p + ci + 0.03, f"{p:.2f}±{ci:.2f}", ha="center", fontsize=11, color="#4C72B0")
ax.set_ylabel("hop2 多跳正确率")
ax.set_title(f"(c) latent multi-hop（n={n} 子项）", fontsize=11)
ax.set_ylim(0, 1)
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("阶段1 预实验（Qwen2.5-1.5B-Instruct，真实数据）", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG / "fig8_real_pilot.png", bbox_inches="tight", dpi=150)
print("saved:", FIG / "fig8_real_pilot.png")
print(f"latent={p:.3f} ci={ci:.3f}")
