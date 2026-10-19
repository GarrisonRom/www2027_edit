# -*- coding: utf-8 -*-
"""研究内容一 · 方法示意图（非实验结果）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot

setup_plot()

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent / "研究内容一-示意图"
OUT.mkdir(exist_ok=True)

BLUE, GREEN, RED, GRAY, ORANGE = "#4C72B0", "#55A868", "#C44E52", "#8C8C8C", "#DD8452"


def box(ax, x, y, w, h, title, lines, fc="#EAF0F8", ec=BLUE, title_color=BLUE, fs=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=1.6))
    ax.text(x, y + h / 2 - 0.16, title, ha="center", va="top",
            fontsize=fs + 1.5, fontweight="bold", color=title_color)
    ax.text(x, y + h / 2 - 0.42, lines, ha="center", va="top", fontsize=fs, color="#333333")


def arrow(ax, x1, y1, x2, y2, color=GRAY, style="-|>", lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, color=color, lw=lw, linestyle=ls))


# ================= 图 0：方法总体流程 =================
fig, ax = plt.subplots(figsize=(13.5, 6.2))
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 6.2)
ax.axis("off")
ax.set_title("研究内容一方法流程：探测驱动的扩展遗忘集（示意图）", fontsize=14, fontweight="bold", pad=12)

W, H = 2.35, 1.30
y1, y2 = 4.55, 1.55
xs = [1.45, 3.95, 6.45, 8.95, 11.45]

# 第一行（左→右）
box(ax, xs[0], y1, W, H, "① 定种子", "待遗忘事实集 $F_0$\n（FaithUn 实体簇 hop1）")
box(ax, xs[1], y1, W, H, "② 候选池", "hop2 链式三元组\n第一跳事实（外部图）", fc="#F3F3F3", ec=GRAY, title_color=GRAY)
box(ax, xs[2], y1, W, H, "③ 前向探测 ★", "多模板打分 $c(f),\\bar p(f)$\n模型实际掌握过滤", fc="#E8F3EC", ec=GREEN, title_color=GREEN)
box(ax, xs[3], y1, W, H, "④ 内生图", "$G_\\tau=(V,K_\\tau,w)$\n条件概率加权图", fc="#E8F3EC", ec=GREEN, title_color=GREEN)
box(ax, xs[4], y1, W, H, "⑤ 活性核验 ★", "原始模型直答 hop2\n活跃路径集 $P_{act}$", fc="#E8F3EC", ec=GREEN, title_color=GREEN)
for i in range(4):
    arrow(ax, xs[i] + W / 2 + 0.05, y1, xs[i + 1] - W / 2 - 0.05, y1)

# 下行（右→左蛇形）
box(ax, xs[4], y2, W, H, "⑥ 贪心选集 ★", "$\\max g(F)$ s.t. $F\\subseteq E\\setminus R$\n$|F|\\leq B$，$(1-1/e)$ 保证", fc="#E8F3EC", ec=GREEN, title_color=GREEN)
box(ax, xs[2], y2, W, H, "⑦ 遗忘训练", "GA-LoRA / NPO\n$F_1=F_0\\cup F^*$ → $M'$", fc="#FDF1E7", ec=ORANGE, title_color=ORANGE)
box(ax, xs[0], y2, W + 0.3, H, "⑧ 三口径评测", "直问 $p(o|s,r)$ / 绕弯 $p(o|s,P)$ / 乱问 $R_{eval}$", fc="#FDF1E7", ec=ORANGE, title_color=ORANGE)
arrow(ax, xs[4], y1 - H / 2 - 0.06, xs[4], y2 + H / 2 + 0.06)
arrow(ax, xs[4] - W / 2 - 0.05, y2, xs[2] + W / 2 + 0.05, y2)
arrow(ax, xs[2] - W / 2 - 0.05, y2, xs[0] + (W + 0.3) / 2 + 0.05, y2)

# 撤回权限禁区（约束注入 ⑥）
box(ax, xs[4], 0.52, W, 0.78, "撤回权限禁区 $R$", "合法保留事实（硬约束）", fc="#FBEAEA", ec=RED, title_color=RED, fs=8)
arrow(ax, xs[4], 0.94, xs[4], y2 - H / 2 - 0.06, color=RED, ls="--")

# 协议冻结说明
ax.text(6.75, 0.35, "★ = 自研模块（约 300 行）　|　③⑤⑧ 使用冻结协议：模板、种子、评测口径实验前存档，全程不得改动",
        ha="center", fontsize=9, color="#555555")
fig.savefig(OUT / "fig0_pipeline.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ================= 图 6：内生图与路径选择示例 =================
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.6)
ax.axis("off")
ax.set_title("内生图与扩展遗忘集选择示例（示意图）", fontsize=14, fontweight="bold", pad=10)


def node(ax, x, y, label, r=0.42, fc="#EAF0F8"):
    ax.add_patch(plt.Circle((x, y), r, fc=fc, ec=BLUE, lw=1.5, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=10, zorder=4)


def edge(ax, x1, y1, x2, y2, label, color=GRAY, lw=1.6, ls="-", lx=None, ly=None, fs=8.5):
    arrow(ax, x1, y1, x2, y2, color=color, lw=lw, ls=ls)
    ax.text(lx if lx else (x1 + x2) / 2, (ly if ly else (y1 + y2) / 2) + 0.13, label,
            ha="center", fontsize=fs, color=color, zorder=5)


# 节点
node(ax, 1.6, 3.9, "奥巴马")
node(ax, 5.4, 3.9, "火奴鲁鲁")
node(ax, 9.2, 3.9, "夏威夷州")
node(ax, 1.6, 1.3, "普纳荷学校")
node(ax, 5.4, 1.3, "米歇尔")
node(ax, 9.2, 1.3, "芝加哥")

# 边
# F0：待遗忘（红粗）
edge(ax, 2.05, 3.9, 4.95, 3.9, "出生于（$F_0$，待遗忘）", color=RED, lw=3.2)
# 活跃支持路径（绿）：学校 --位于--> 火奴鲁鲁
ax.add_patch(FancyArrowPatch((2.05, 1.72), (5.05, 3.55), arrowstyle="-|>",
                             mutation_scale=16, color=GREEN, lw=2.2, zorder=2))
ax.text(3.0, 2.25, "位于", fontsize=8.5, color=GREEN)
# 链条挂接：学校 --曾就读于--> 奥巴马
ax.add_patch(FancyArrowPatch((1.6, 1.72), (1.6, 3.48), arrowstyle="-|>",
                             mutation_scale=16, color=GREEN, lw=2.2, zorder=2))
ax.text(1.05, 2.6, "曾就读于", fontsize=8.5, color=GREEN)
edge(ax, 5.85, 3.9, 8.75, 3.9, "是…的首府", color=GREEN, lw=2.2)
# R 禁区（蓝色虚线框提示）
ax.add_patch(FancyBboxPatch((6.0, 3.42), 2.6, 0.95, boxstyle="round,pad=0.05",
                            fc="none", ec=BLUE, lw=1.6, ls="--", zorder=6))
ax.text(7.3, 4.55, "$R$：公共知识，禁止撤回", ha="center", fontsize=9, color=BLUE)
# 非活跃候选（灰）
edge(ax, 5.85, 1.3, 8.75, 1.3, "出生于", color=GRAY, lw=1.4)
ax.add_patch(FancyArrowPatch((2.05, 1.6), (4.98, 1.35), arrowstyle="-|>", mutation_scale=14,
                             color=GRAY, lw=1.4, ls=":", zorder=2))
ax.text(3.3, 1.75, "配偶", fontsize=8.5, color=GRAY)
ax.text(7.3, 0.72, "路径不指向 o：非候选", ha="center", fontsize=8.5, color=GRAY)

# 底部：选集结果
ax.text(5.5, 0.18, "探测过滤：灰色事实若模型未稳定掌握则剔除　→　活性核验：绿色路径模型现场可推导　→　贪心选集：优先覆盖多条活跃路径的事实进入 $F^*$",
        ha="center", fontsize=9, color="#444444")
fig.savefig(OUT / "fig6_method_graph.png", bbox_inches="tight", dpi=150)
plt.close(fig)

print("saved:")
for name in ["fig0_pipeline.png", "fig6_method_graph.png"]:
    print(" ", OUT / name)
