# -*- coding: utf-8 -*-
"""研究内容一 · 数学形式示意图（非实验结果）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot

setup_plot()

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

OUT = Path(__file__).parent / "研究内容一-示意图"
OUT.mkdir(exist_ok=True)

BLUE, GREEN, RED, GRAY, ORANGE = "#4C72B0", "#55A868", "#C44E52", "#8C8C8C", "#DD8452"
EPS = 1 / 3

fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))

# ================= 左面板：复合条件概率的图式计算 =================
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8.6)
ax.axis("off")
ax.set_title("(a) 复合条件概率：间接恢复的图式计算", fontsize=12.5, fontweight="bold")
ax.text(5, 8.15, r"$p_M(o\mid s,P)=\sum_{x}\,p_M(x\mid s,r_1)\cdot p_M(o\mid x,r_2)$",
        ha="center", fontsize=13, color="#222222")

def node(x, y, label, r=0.52, fc="#EAF0F8", ec=BLUE):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=1.8, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=10.5, zorder=4)

def edge(x1, y1, x2, y2, w, color, label=None, lx=0, ly=0, ls="-", fs=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, color=color, lw=max(0.8, 6 * w),
                                 linestyle=ls, zorder=2, shrinkA=14, shrinkB=14))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center",
                fontsize=fs, color=color, zorder=5)

# 层 1：s 与桥接实体
node(1.4, 3.4, "s\n奥巴马")
xs = [(4.4, 4.8, "火奴鲁鲁", 0.85), (4.4, 3.4, "芝加哥", 0.06), (4.4, 2.0, "纽约", 0.03)]
for x, y, lab, w in xs:
    node(x, y, lab, r=0.55)
    edge(1.4, 3.4, x, y, w, BLUE, f"{w:.2f}", lx=-0.15)

# 层 2：x 到 o
node(8.6, 3.4, "o\n夏威夷州")
for x, y, lab, w in xs:
    w2 = {"火奴鲁鲁": 1.0, "芝加哥": 0.0, "纽约": 0.0}[lab]
    edge(x, y, 8.6, 3.4, max(w2, 0.04), GREEN if w2 > 0 else GRAY,
         f"{w2:.1f}" if w2 > 0 else "0", lx=0.25)

# 直接通道（已遗忘，虚线，上方绕行）
ax.add_patch(FancyArrowPatch((1.4, 3.95), (8.6, 3.95), arrowstyle="-|>", mutation_scale=14,
                             color=RED, lw=1.3, ls="--", zorder=1,
                             connectionstyle="arc3,rad=-0.5"))
ax.text(5, 6.75, r"直接通道 $p_{M'}(o\mid s,r)\leq\varepsilon$（已遗忘）",
        ha="center", fontsize=9.5, color=RED)

# 求和结果
ax.text(5, 0.95, r"$0.85\times1.0+0.06\times0+0.03\times0=0.85>\varepsilon=\frac{1}{3}$",
        ha="center", fontsize=12, color=GREEN, fontweight="bold")
ax.text(5, 0.38, "直接问答不出，绕弯问仍答得出 → 间接恢复成立（定义 4）",
        ha="center", fontsize=10, color="#444444")

# ================= 右面板：预算约束最大覆盖 =================
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("(b) 预算约束最大覆盖：扩展遗忘集的选择", fontsize=12.5, fontweight="bold")
ax.text(5, 7.55, r"$F^*=\arg\max_{F\subseteq E\setminus R,\,|F|\leq B}\;\left|\bigcup_{e\in F}S(e)\right|$",
        ha="center", fontsize=12.5, color="#222222")

facts = [("e_1", 6.0, ""), ("e_2", 5.0, "①"), ("e_3", 4.0, ""), ("e_4", 3.0, "禁区"), ("e_5", 2.0, "②"), ("e_6", 1.0, "③")]
paths = [("P_1", 6.0), ("P_2", 5.0), ("P_3", 4.0), ("P_4", 3.0), ("P_5", 2.0), ("P_6", 1.0)]
S = {"e_1": {"P_1"}, "e_2": {"P_1", "P_2", "P_4"}, "e_3": {"P_2"},
     "e_4": {"P_3"}, "e_5": {"P_3", "P_5"}, "e_6": {"P_5", "P_6"}}

fy = {n: y for n, y, _ in facts}
py = {n: y for n, y in paths}
for n, y, tag in facts:
    inR = tag == "禁区"
    ax.add_patch(Circle((1.7, y), 0.42, fc="#FBEAEA" if inR else "#EAF0F8",
                        ec=RED if inR else BLUE, lw=1.8, zorder=3))
    ax.text(1.7, y, f"${n}$", ha="center", va="center", fontsize=10.5, zorder=4,
            color=RED if inR else BLUE)
    if tag == "禁区":
        ax.text(0.72, y, tag, ha="center", va="center", fontsize=8.5, color=RED)
    elif tag:
        ax.text(0.72, y, tag, ha="center", va="center", fontsize=11, color=ORANGE, fontweight="bold")
for n, y in paths:
    ax.add_patch(Circle((8.3, y), 0.42, fc="#E8F3EC", ec=GREEN, lw=1.8, zorder=3))
    ax.text(8.3, y, f"${n}$", ha="center", va="center", fontsize=10.5, color=GREEN, zorder=4)

for e, covers in S.items():
    for p in covers:
        ax.plot([2.15, 7.85], [fy[e], py[p]], color=GRAY, lw=1.1, alpha=0.75, zorder=1)

ax.text(5, 0.42, r"贪心顺序：$e_2$(增益3) → $e_5$(增益2) → $e_6$(增益1)；$B=3$，$g(F)=6=|P_{act}|$",
        ha="center", fontsize=10.5, color="#222222")
ax.text(5, 0.02, r"$e_4$ 属撤回禁区 $R$，从候选域剔除（硬约束 $F\subseteq E\setminus R$）",
        ha="center", fontsize=9.5, color=RED)

fig.suptitle("研究内容一 · 数学形式示意图", fontsize=14.5, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "fig7_math_schematic.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("saved:", OUT / "fig7_math_schematic.png")
