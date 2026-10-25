# -*- coding: utf-8 -*-
"""
研究内容二 · 阶段2 绘图：图 D2（逐层漂移）/ 头归因热力图 / FFN 神经元 / 梯度对照
对齐检验：受损组 D vs 对照组 K 的逐层配对 t 检验（scipy），结果存 attribution/stats_stage2.json
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

stat_out = {}

# ---------- 图 D2：逐层漂移曲线 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
for ax, group, color in [(axes[0], "G2", "#d62728"), (axes[1], "G1", "#7f7f7f")]:
    z = np.load(ATT / f"drift_items_{group}.npz")
    D, K = z["D"], z["K"]  # (48, 29)
    md, mk = D.mean(0), K.mean(0)
    sd = D.std(0) / np.sqrt(len(D))
    sk = K.std(0) / np.sqrt(len(K))
    xs = np.arange(29)
    ax.plot(xs, md, color=color, lw=1.8, label="D 受损组（same_object/hop2 top-24+24）")
    ax.fill_between(xs, md - sd, md + sd, color=color, alpha=0.2)
    ax.plot(xs, mk, color="#2ca02c", lw=1.8, ls="--", label="K 对照组（源内 d 最低 48 题）")
    ax.fill_between(xs, mk - sk, mk + sk, color="#2ca02c", alpha=0.15)
    t, p = stats.ttest_rel(D, K)
    sig = np.where(p < 0.01)[0]
    ax.scatter(sig, md[sig] + 0.004, marker="*", s=90, color="black", zorder=5,
               label="p<0.01（逐层配对 t）")
    ax.set_xlabel("层（0=嵌入，1–28=Transformer 层）")
    ax.set_title(f"模型对：Qwen2.5-1.5B vs {group}（LoRA 遗忘适配器）")
    ax.legend(fontsize=9, loc="upper left")
    stat_out[f"drift_{group}"] = {
        "sig_layers_p_lt_0.01": [int(i) for i in sig],
        "max_t_layer": int(np.argmax(np.abs(t))),
        "max_t": float(np.max(np.abs(t))),
        "mean_diff_last5": float((md - mk)[-5:].mean()),
    }
axes[0].set_ylabel("末 token 隐状态 M vs M′ 余弦漂移 $1-\\cos$")
fig.suptitle("图 D2 · 逐层激活漂移：受损事实与对照事实在哪些层分离（Qwen2.5-1.5B）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIGDIR / "fig14_drift_curves.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------- 头归因热力图 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
for ax, group in [(axes[0], "G2"), (axes[1], "G1")]:
    diff = np.load(ATT / f"head_diff_{group}.npy")
    vmax = np.percentile(np.abs(diff), 99)
    im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("注意力头")
    ax.set_ylabel("层")
    ax.set_title(f"{group}：受损−对照 末token注意力差（Qwen2.5-1.5B）")
    plt.colorbar(im, ax=ax, shrink=0.85)
    l, h = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    ax.add_patch(plt.Rectangle((h - .5, l - .5), 1, 1, fill=False, edgecolor="k", lw=2))
    ax.text(0.5, -0.06, f"null 结果：跨头均匀（mean|diff|={np.abs(diff).mean():.4f}，"
                        f"max={np.abs(diff).max():.4f}），无局部头",
            transform=ax.transAxes, ha="center", fontsize=10, color="#8b0000")
    stat_out[f"heads_{group}"] = {"max_abs_layer": int(l), "max_abs_head": int(h),
                                  "max_abs": float(diff[l, h]),
                                  "mean_abs": float(np.abs(diff).mean())}
fig.suptitle("图 D2b · 注意力头归因：28 层 × 12 头热力图（黑框=|差值|最大头）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIGDIR / "fig15_head_heatmap.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------- FFN 神经元 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
ffn_data = {}
for ax, group, color in [(axes[0], "G2", "#d62728"), (axes[1], "G1", "#7f7f7f")]:
    d = json.load(open(ATT / f"ffn_{group}.json", encoding="utf-8"))
    ffn_data[group] = d
    pl = d["per_layer"]
    xs = [r["layer"] for r in pl]
    ax.plot(xs, [r["D"] for r in pl], color=color, lw=1.8, label="D 受损组")
    ax.plot(xs, [r["K"] for r in pl], color="#2ca02c", lw=1.8, ls="--", label="K 对照组")
    diff = np.array([r["D"] for r in pl]) - np.array([r["K"] for r in pl])
    ax.set_xlabel("层")
    ax.set_title(f"{group}：FFN 中间神经元平均激活漂移（Qwen2.5-1.5B）")
    ax.legend(fontsize=9)
    stat_out[f"ffn_{group}"] = {"peak_layer_D": int(np.argmax([r["D"] for r in pl]))}
axes[0].set_ylabel("$|\\mathrm{act}_{M'} - \\mathrm{act}_M|$（逐神经元平均）")
fig.suptitle("图 D2c · FFN 神经元归因：漂移峰值在中高层，top-k 组件冻结给阶段3", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIGDIR / "fig16_ffn_curves.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# 两组共享 top 神经元（全局排序，组件清单给阶段3）
top_g2 = {(r["layer"], r["neuron"]) for r in sorted(ffn_data["G2"]["topk"], key=lambda r: -r["diff"])[:20]}
top_g1 = {(r["layer"], r["neuron"]) for r in sorted(ffn_data["G1"]["topk"], key=lambda r: -r["diff"])[:20]}
shared = sorted(top_g2 & top_g1)
stat_out["ffn_shared_top20"] = shared
stat_out["ffn_G2_global_top5"] = sorted(ffn_data["G2"]["topk"], key=lambda r: -r["diff"])[:5]
print("G1/G2 全局 top-20 神经元交集:", shared)

# ---------- 梯度对照 ----------
fig, ax = plt.subplots(figsize=(8, 4.6))
for group, color in [("G2", "#d62728"), ("G1", "#7f7f7f")]:
    g = json.load(open(ATT / f"grad_{group}.json", encoding="utf-8"))
    xs = np.arange(1, 29)
    ax.plot(xs, g["D"], color=color, lw=1.8, label=f"{group}·D 受损组")
    ax.plot(xs, g["K"], color=color, lw=1.8, ls="--", alpha=0.6, label=f"{group}·K 对照组")
    stat_out[f"grad_{group}"] = {"D_peak_layer": int(np.argmax(g["D"])) + 1,
                                 "K_peak_layer": int(np.argmax(g["K"])) + 1}
ax.set_yscale("log")
ax.set_xlabel("层")
ax.set_ylabel("答案 NLL 逐层梯度平方和（Fisher 代理）")
ax.set_title("图 D2d · 梯度归因对照（模型：Qwen2.5-1.5B，冻结原模型）")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIGDIR / "fig17_grad_curves.png", dpi=160, bbox_inches="tight")
plt.close(fig)

json.dump(stat_out, open(ATT / "stats_stage2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(stat_out, ensure_ascii=False, indent=1)[:1200])
print("figures saved.")
