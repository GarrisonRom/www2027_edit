# -*- coding: utf-8 -*-
"""GSU 遗忘方法对比图（fig23）：左=多跳封堵×保持效用帕累托，右=六评测台柱状"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WS = Path(__file__).resolve().parents[2]
RES = WS / "exp" / "results"
FIGDIR = WS / "研究内容一-示意图"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

v2 = json.load(open(RES / "eval_results_1p5B_v2.json", encoding="utf-8"))
g6 = json.load(open(RES / "g6_anchor_summary.json", encoding="utf-8"))["metrics"]
five = {k: v2[k] for k in ("orig", "G1", "G2")}
five["G6"] = g6["G6"]
for l in open(RES / "eval_results_gsu.jsonl", encoding="utf-8"):
    r = json.loads(l)
    g = r["group"].replace("GSU", "")
    five[g] = {k: v for k, v in r.items() if k != "group"}
six = {"orig": None, "G2": None}
six["orig"] = {k: v["mean_correct"] for k, v in
               json.load(open(RES / "restore_baseline_orig_g2.json", encoding="utf-8"))["orig"].items()}
six["G2"] = {k: v["mean_correct"] for k, v in
             json.load(open(RES / "restore_baseline_orig_g2.json", encoding="utf-8"))["G2"].items()}
for l in open(RES / "gsu_sixset.jsonl", encoding="utf-8"):
    r = json.loads(l)
    g = r["group"].replace("GSU", "")
    six[g] = {k: v["mean_correct"] for k, v in r["sets"].items()}

# ---------- 左：帕累托（retain × 多跳封堵率）----------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
style = {"orig": ("o", "#7f7f7f"), "G1": ("s", "#9467bd"), "G2": ("s", "#1f77b4"),
         "G6": ("^", "#8c564b"), "M1": ("D", "#2ca02c"), "M2": ("P", "#ff7f0e"),
         "M3": ("*", "#d62728"),
         "BLGradDiff": ("v", "#17becf"), "BLNPO": ("v", "#bcbd22"),
         "BLSimNPO": ("x", "#e377c2"), "BLRMU": ("x", "#7f7f7f"),
         "BLGA": ("v", "#17becf"), "BLCEU": ("v", "#0e9594"),
         "BLPDU": ("v", "#065a82"), "BLDPO": ("x", "#636363"),
         "BLWGA": ("x", "#969696"), "BLSatImp": ("x", "#c7c7c7"),
         "BLUNDIAL": ("x", "#525252")}
label = {"orig": "orig", "G1": "G1 只忘F0", "G2": "G2 拓展集+KL", "G6": "G6 拓展集+强KL",
         "M1": "M1 定位+手术", "M2": "M2 全层+半手术", "M3": "M3(GSU) 全层+手术",
         "BLGradDiff": "GradDiff", "BLNPO": "NPO",
         "BLSimNPO": "SimNPO*", "BLRMU": "RMU*",
         "BLGA": "GA(纯)", "BLCEU": "CEU", "BLPDU": "PDU", "BLDPO": "DPO*",
         "BLWGA": "WGA*", "BLSatImp": "SatImp*", "BLUNDIAL": "UNDIAL*"}
off = {"M2": (8, -14), "BLDPO": (6, -13), "BLWGA": (-42, 8), "BLSatImp": (-56, -3),
       "BLUNDIAL": (6, -13), "BLSimNPO": (-10, 10), "BLRMU": (-16, -16)}
for g, (mk, col) in style.items():
    if g not in five:
        continue
    block = 1 - five[g]["hop2"]          # 多跳封堵率
    ax.scatter(five[g]["retain"], block * 100, marker=mk, s=280 if g == "M3" else 100,
               c=col, zorder=5 if g == "M3" else 3,
               edgecolors="black" if g == "M3" else "none")
    ax.annotate(label[g], (five[g]["retain"], block * 100), textcoords="offset points",
                xytext=off.get(g, (7, 6)), fontsize=9, fontweight="bold" if g == "M3" else "normal")
ax.set_xlabel("保持集 retain 准确率 → 越大越好（效用）")
ax.set_ylabel("多跳封堵率 = 1 − hop2 准确率 → 越大越好（遗忘质量）")
ax.set_title("遗忘质量-效用帕累托（Qwen2.5-1.5B-Instruct，同遗忘集/同预算）\n* = 该默认配置下未发生有效遗忘")
ax.grid(alpha=0.3)
ax.text(0.02, 0.03, "右上 = 理想区", transform=ax.transAxes, fontsize=10, color="green")

# ---------- 右：六评测台柱状 ----------
ax = axes[1]
sets6 = ["hop1_unlearn", "D_so", "D_hop2", "K", "retain", "F1_hop2"]
glabels6 = ["hop1(F0)", "D_so", "D_hop2", "K", "retain", "F1_hop2"]
groups6 = ["orig", "G2", "BLGA", "BLCEU", "BLPDU", "M3"]
lab6 = {"orig": "orig", "G2": "G2", "BLGA": "GA(纯)", "BLCEU": "CEU",
        "BLPDU": "PDU", "M3": "M3 GSU(本文)"}
x = np.arange(len(sets6))
w = 0.8 / len(groups6)
for gi, g in enumerate(groups6):
    ax.bar(x + gi * w - 0.4 + w / 2, [six[g][k] for k in sets6], w, label=lab6[g])
ax.set_xticks(x)
ax.set_xticklabels(glabels6, fontsize=9)
ax.set_ylabel("MCQA 准确率")
ax.set_title("六评测台对比（Qwen2.5-1.5B-Instruct）")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("图 23 · GSU 梯度手术遗忘 vs 全 baseline 对比（同遗忘集 F0+F1贪心5 / 同 LoRA / 同预算；GA~UNDIAL 移植自 open-unlearning）",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIGDIR / "fig23_gsu_compare.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("saved fig23_gsu_compare.png")
for g in ("orig", "G1", "G2", "G6", "M1", "M2", "M3"):
    if g in five:
        print(g, {k: round(v, 3) for k, v in five[g].items()})
