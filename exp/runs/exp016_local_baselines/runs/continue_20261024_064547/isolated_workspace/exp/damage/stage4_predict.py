# -*- coding: utf-8 -*-
"""研究内容二 · 阶段4：显影得分 → 撤销收益 的预测验证（Spearman，图 D4）
预测对象：全层单层撤销扫描（scan_mlp_G2.jsonl，28 点，阶段3未做过）
显影得分：ffn per_layer D 漂移（G2 与 G1 双源，G1→G2 为跨模型留出检验）
随机基线：1000 次随机打分的 |ρ| 分布
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

WS = Path(__file__).resolve().parents[2]
OUT = WS / "exp" / "results"
ATT = OUT / "attribution"
FIGDIR = WS / "研究内容一-示意图"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

G2_BASE = (-11.828 + -9.579) / 2  # per_item_G2 的 D 集基准

scan = {json.loads(l)["layer"]: json.loads(l)["mean_lp"]
        for l in open(ATT / "scan_mlp_G2.jsonl", encoding="utf-8")}
layers = sorted(scan)
recovery = np.array([scan[l] - G2_BASE for l in layers])

ffn_g2 = json.load(open(ATT / "ffn_G2.json", encoding="utf-8"))["per_layer"]
ffn_g1 = json.load(open(ATT / "ffn_G1.json", encoding="utf-8"))["per_layer"]
score_g2 = np.array([ffn_g2[l]["D"] for l in layers])
score_g1 = np.array([ffn_g1[l]["D"] for l in layers])
# 备选得分：隐状态漂移（G2，D 组）
z = np.load(ATT / "drift_items_G2.npz")
score_drift = z["D"].mean(0)[1:]  # 28 层（去掉嵌入层 0）
score_drift = np.array([score_drift[l] for l in layers])

rng = np.random.default_rng(42)
rand_rho = []
for _ in range(1000):
    perm = rng.permutation(len(recovery))
    rand_rho.append(stats.spearmanr(score_g2[perm], recovery).statistic)
rand_abs = np.mean(np.abs(rand_rho))

res = {}
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
for ax, (name, sc, src) in zip(axes, [
        ("得分来源：G2 自身 FFN 漂移", score_g2, "G2"),
        ("得分来源：G1 FFN 漂移（跨模型留出）", score_g1, "G1→G2"),
        ("得分来源：G2 隐状态漂移", score_drift, "drift")]):
    rho, p = stats.spearmanr(sc, recovery)
    res[name] = {"rho": float(rho), "p": float(p)}
    ax.scatter(sc, recovery, c=layers, cmap="viridis", s=55, zorder=3)
    m, b = np.polyfit(sc, recovery, 1)
    xs = np.linspace(sc.min(), sc.max(), 50)
    ax.plot(xs, m * xs + b, "--", color="#d62728", lw=1.5)
    for l in [18, 27]:
        ax.annotate(f"L{l}", (sc[l], recovery[l]), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    cb = plt.colorbar(ax.collections[0], ax=ax, shrink=0.85)
    cb.set_label("层号")
    ax.set_xlabel("显影得分（受损组 FFN 漂移 / 隐状态漂移）")
    ax.set_ylabel("单层撤销的实测恢复（对数概率提升）")
    ax.set_title(f"{name}\nSpearman ρ = {rho:.3f}，p = {p:.4f}", fontsize=11)
    ax.grid(alpha=0.3)
res["random_baseline_mean_abs_rho"] = float(rand_abs)
fig.suptitle("图 D4 · 显影得分预测组件撤销收益：28 层留出验证（Qwen2.5-1.5B，G2）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIGDIR / "fig19_d4_predict.png", dpi=160, bbox_inches="tight")
print(json.dumps(res, ensure_ascii=False, indent=1))
print("saved:", FIGDIR / "fig19_d4_predict.png")
