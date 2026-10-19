# -*- coding: utf-8 -*-
"""研究内容一 · 初步示意线型图（模拟数据，非实验结果）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot

setup_plot()

import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "研究内容一-示意图"
OUT.mkdir(exist_ok=True)

EPS = 1 / 3
TAG = "（示意数据，非实验结果）"

# ---------- 图 1：τ 敏感性曲线 ----------
tau = np.linspace(0.5, 0.9, 9)
# 模拟：掌握率随 τ 下降，7B 高于 2B
keep_2b = 0.92 / (1 + np.exp((tau - 0.62) * 9))
keep_7b = 0.95 / (1 + np.exp((tau - 0.78) * 9))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(tau, keep_2b, "o-", color="#4C72B0", label="Gemma-2-2B")
ax.plot(tau, keep_7b, "s-", color="#DD8452", label="Llama-3.1-8B")
ax.axvspan(0.6, 0.8, color="gray", alpha=0.12, label="扫描区间 [0.6, 0.8]")
ax.set_xlabel(r"掌握判定阈值 $\tau$")
ax.set_ylabel("模型稳定掌握的事实比例")
ax.set_title("图 1  探测阈值 τ 敏感性曲线 " + TAG)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
ax.legend()
fig.savefig(OUT / "fig1_tau_sensitivity.png", bbox_inches="tight")
plt.close(fig)

# ---------- 图 2：贪心选集边际收益曲线 ----------
k = np.arange(0, 41)
# 模拟次模饱和曲线：g(k) = P_total * (1 - exp(-k/8))
P_total = 60.0
g = P_total * (1 - np.exp(-k / 8.0))
B = 15
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(k, g, "-", color="#4C72B0", lw=2)
ax.fill_between(k, 0, g, alpha=0.08, color="#4C72B0")
ax.axvline(B, color="#C44E52", ls="--", label=f"预算 $B={B}$")
ax.axhline(P_total * (1 - 1 / np.e), color="gray", ls=":", label=r"$(1-1/e)\approx 63.2\%$ 上限参考")
ax.plot(B, P_total * (1 - np.exp(-B / 8.0)), "o", color="#C44E52")
ax.annotate("边际收益递减\n（次模性）", xy=(25, g[25]), xytext=(28, 12),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)
ax.set_xlabel("已选入扩展集的事实数 $|F|$")
ax.set_ylabel(r"累计覆盖活跃路径数 $g(F)$")
ax.set_title("图 2  贪心选集的边际覆盖收益 " + TAG)
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.savefig(OUT / "fig2_greedy_coverage.png", bbox_inches="tight")
plt.close(fig)

# ---------- 图 3：多跳恢复率 vs 预算 ----------
bgt = np.linspace(0, 40, 41)
# G1 只忘 F0：不随预算变化，恢复率高
g1 = np.full_like(bgt, 0.75)
# G2 探测驱动：快速下降后压到地板
g2 = EPS + (0.78 - EPS) * np.exp(-bgt / 6.0)
# G3 全邻域：也下降但单位预算效率低、误伤大（恢复率略高）
g3 = EPS + (0.78 - EPS) * np.exp(-bgt / 14.0)
# G4 随机等规模：缓慢下降
g4 = EPS + (0.78 - EPS) * np.exp(-bgt / 30.0)
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.axhline(EPS, color="black", ls="--", lw=1, label=r"随机地板 $\varepsilon=1/3$")
ax.plot(bgt, g1, "-", color="#8C8C8C", label="G1 只忘 $F_0$（基线）")
ax.plot(bgt, g3, "-", color="#DD8452", label="G3 全邻域扩展（不筛选）")
ax.plot(bgt, g4, "-", color="#55A868", label="G4 等规模随机扩展")
ax.plot(bgt, g2, "-", color="#4C72B0", lw=2.5, label="G2 探测驱动 $F_1$（本文）")
ax.axvline(B, color="#C44E52", ls=":", lw=1)
ax.annotate("$B$", xy=(B, 0.9), color="#C44E52", fontsize=11)
ax.set_xlabel("扩展遗忘集规模（事实数）")
ax.set_ylabel("多跳恢复率（越低越好）")
ax.set_title("图 3  多跳恢复率–预算曲线（四组对照） " + TAG)
ax.set_ylim(0.25, 1.0)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
fig.savefig(OUT / "fig3_recovery_vs_budget.png", bbox_inches="tight")
plt.close(fig)

# ---------- 图 4：扩展效率帕累托 ----------
# x = 遗忘集规模, y = 多跳恢复率; 每条方法一条"规模-恢复"轨迹
size = np.linspace(2, 40, 39)
r2 = EPS + (0.78 - EPS) * np.exp(-size / 6.0)      # G2
r3 = EPS + (0.78 - EPS) * np.exp(-size / 14.0)     # G3
r4 = EPS + (0.78 - EPS) * np.exp(-size / 30.0)     # G4
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.axhline(EPS, color="black", ls="--", lw=1, label=r"随机地板 $\varepsilon=1/3$")
ax.plot(size, r3, "-", color="#DD8452", label="G3 全邻域扩展")
ax.plot(size, r4, "-", color="#55A868", label="G4 随机扩展")
ax.plot(size, r2, "-", color="#4C72B0", lw=2.5, label="G2 探测驱动（本文）")
ax.plot(8, EPS + (0.78 - EPS) * np.exp(-8 / 6.0), "*", color="#C44E52", ms=16, zorder=5)
ax.annotate("G2 工作点：约 8 个事实\n即达地板", xy=(8, 0.36), xytext=(14, 0.48),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)
ax.plot(20, 0.75, "^", color="#8C8C8C", ms=10, zorder=5)
ax.annotate("G1 只忘 $F_0$：\n恢复率恒定≈0.75", xy=(20, 0.75), xytext=(26, 0.68),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)
ax.set_xlabel("遗忘集规模（事实数，越少误伤越小）")
ax.set_ylabel("多跳恢复率（越低越好）")
ax.set_title("图 4  扩展效率帕累托图（主结果示意） " + TAG)
ax.set_ylim(0.25, 1.0)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
fig.savefig(OUT / "fig4_pareto.png", bbox_inches="tight")
plt.close(fig)

# ---------- 图 5：训练过程监控 ----------
steps = np.arange(0, 501, 10)
forget_loss = 2.4 - 1.7 / (1 + np.exp(-(steps - 180) / 60)) + np.random.RandomState(0).normal(0, 0.03, len(steps))
retain_loss = 0.25 + 0.05 / (1 + np.exp(-(steps - 250) / 90)) + np.random.RandomState(1).normal(0, 0.008, len(steps))
ua = 0.95 - 0.63 / (1 + np.exp(-(steps - 200) / 55)) + np.random.RandomState(2).normal(0, 0.01, len(steps))
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
axes[0].plot(steps, forget_loss, color="#C44E52", label="forget loss")
axes[0].plot(steps, retain_loss, color="#4C72B0", label="retain loss")
axes[0].set_xlabel("训练步")
axes[0].set_ylabel("损失")
axes[0].set_title("损失曲线 " + TAG, fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=9)
axes[1].plot(steps, ua, color="#4C72B0")
axes[1].axhline(EPS, color="black", ls="--", lw=1, label=r"地板 $1/3$")
axes[1].set_xlabel("训练步")
axes[1].set_ylabel("遗忘集 UA")
axes[1].set_title("遗忘达标过程 " + TAG, fontsize=10)
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig5_training_monitor.png", bbox_inches="tight")
plt.close(fig)

print("saved:")
for p in sorted(OUT.glob("*.png")):
    print(" ", p)
