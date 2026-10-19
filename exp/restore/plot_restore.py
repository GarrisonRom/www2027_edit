# -*- coding: utf-8 -*-
"""研究内容三 · 结果汇总 + 图 R1/R2/R3"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WS = Path(__file__).resolve().parents[2]
RES = WS / "exp" / "results"
FIGDIR = WS / "研究内容一-示意图"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# orig/G2 基线：与全部恢复配置同一次 build_sets() 口径（eval_baseline.py 产出）
BASELINE = json.load(open(RES / "restore_baseline_orig_g2.json", encoding="utf-8"))
BASE = {k: (BASELINE["G2"][k]["mean_lp"], BASELINE["orig"][k]["mean_lp"])
        for k in ("D_so", "D_hop2", "hop1_unlearn", "F1_hop2", "K", "retain")}
BASE_ACC = {k: (BASELINE["G2"][k]["mean_correct"], BASELINE["orig"][k]["mean_correct"])
            for k in BASE}

rows = {}
for l in open(RES / "restore_results.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if not r["quick"]:
        rows[r["config"]] = r["sets"]

ORDER = ["R1_rec", "B_kl", "R3_l0.1_e80", "R3_l0.1_e90", "R3_l0.1_e95",
         "R3_l1_e80", "R3_l1_e90", "R3_l1_e95", "R3_l10_e80", "R3_l10_e90",
         "R3_l10_e95", "R5_full", "R5_surgery", "R2_ans", "R4_combo"]


def dfrac(s):
    g2 = (BASE["D_so"][0] + BASE["D_hop2"][0]) / 2
    orig = (BASE["D_so"][1] + BASE["D_hop2"][1]) / 2
    v = (s["D_so"]["mean_lp"] + s["D_hop2"]["mean_lp"]) / 2
    return 100 * (v - g2) / (orig - g2)


def frac(s, k):
    return 100 * (s[k]["mean_lp"] - BASE[k][0]) / (BASE[k][1] - BASE[k][0])


print(f"{'配置':<14}{'恢复%':>7}{'hop1回弹%':>10}{'F1复活%':>9}{'K漂移%':>8}{'retain恢复%':>10}")
table = {}
for c in ORDER:
    s = rows[c]
    t = dict(rec=dfrac(s), rb=frac(s, "hop1_unlearn"), f1=frac(s, "F1_hop2"),
             k=frac(s, "K"), ret=frac(s, "retain"))
    table[c] = t
    print(f"{c:<14}{t['rec']:>7.1f}{t['rb']:>10.1f}{t['f1']:>9.1f}{t['k']:>8.1f}{t['ret']:>10.1f}")
json.dump(table, open(RES / "restore_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------- 图 R1：恢复-回弹散点 ----------
fig, ax = plt.subplots(figsize=(8.5, 6))
for c in ORDER:
    t = table[c]
    col = "#7f7f7f" if c.startswith("B_") else ("#d62728" if t["rb"] < 10 and t["rec"] > 30 else "#1f77b4")
    mk = "s" if c.startswith("R3") else ("^" if c.startswith("B_") else "o")
    ax.scatter(t["rb"], t["rec"], c=col, marker=mk, s=90, zorder=3)
    ax.annotate(c, (t["rb"], t["rec"]), textcoords="offset points", xytext=(7, 5), fontsize=9)
ax.scatter(30.3, 34.4, c="black", marker="*", s=260, zorder=4)
ax.annotate("组件撤销\n(内容二)", (30.3, 34.4), textcoords="offset points", xytext=(8, -22), fontsize=9)
ax.axhspan(30, 100, xmin=0, xmax=1/3, color="#2ca02c", alpha=0.08)
ax.text(3, 90, "目标区：恢复≥30%\n且回弹<10%", fontsize=10, color="#2ca02c")
ax.set_xlabel("$F_0$ 遗忘回弹率（%）→ 越小越好")
ax.set_ylabel("受损集恢复率（%）→ 越大越好")
ax.set_title("图 R1 · 定向恢复：恢复-回弹全景（Qwen2.5-1.5B，G2→恢复 adapter）")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGDIR / "fig20_r1_scatter.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------- 图 R2：投影强度 trade-off ----------
fig, ax = plt.subplots(figsize=(8.5, 5.6))
lams = [0.1, 1, 10]
for eta, mk in [(80, "o"), (90, "s"), (95, "^")]:
    rec = [table[f"R3_l{l}_e{eta}"]["rec"] for l in lams]
    rb = [table[f"R3_l{l}_e{eta}"]["rb"] for l in lams]
    f1 = [table[f"R3_l{l}_e{eta}"]["f1"] for l in lams]
    ax.plot(lams, rec, mk + "-", color="#1f77b4", label=f"恢复率 η={eta}")
    ax.plot(lams, rb, mk + "--", color="#d62728", label=f"F0 回弹 η={eta}")
    ax.plot(lams, f1, mk + ":", color="#ff7f0e", label=f"F1 复活 η={eta}")
ax.plot(lams, [table["R1_rec"]["rec"]] * 3, color="#1f77b4", alpha=0.25, lw=6)
ax.plot(lams, [table["R1_rec"]["rb"]] * 3, color="#d62728", alpha=0.25, lw=6)
ax.text(0.1, table["R1_rec"]["rec"] + 1, "R1 无约束", fontsize=9, color="#1f77b4")
ax.set_xscale("log")
ax.set_xlabel("投影强度 λ（能量空间惩罚系数）")
ax.set_ylabel("%")
ax.set_title("图 R2 · 恢复-投影强度-回弹三方关系（提案点名实验）\n色带 = 无约束基线（Qwen2.5-1.5B）")
ax.legend(fontsize=8.5, ncol=3)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGDIR / "fig21_r2_tradeoff.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------- 图 R3：六组柱状对比 ----------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
groups = ["orig", "G2", "R1_rec", "R3_l1_e90", "R3_l10_e90", "B_kl"]
glabels = ["orig", "G2(遗忘后)", "R1 无约束", "R3 λ=1", "R3 λ=10", "B KL锚"]
sets_l = ["hop1_unlearn", "D_so", "D_hop2", "K", "F1_hop2", "retain"]
ax = axes[0]
x = np.arange(len(sets_l))
w = 0.8 / len(groups)
for gi, (g, gl) in enumerate(zip(groups, glabels)):
    if g in ("orig", "G2"):
        vals = [BASE_ACC[k][0 if g == "G2" else 1] for k in sets_l]
    else:
        vals = [rows[g][k]["mean_correct"] for k in sets_l]
    ax.bar(x + gi * w - 0.4 + w / 2, vals, w, label=gl)
ax.set_xticks(x)
ax.set_xticklabels(["hop1(F0)", "D_so", "D_hop2", "K", "F1_hop2", "retain"], fontsize=9)
ax.set_ylabel("MCQA 准确率")
ax.set_title("恢复后各评测集准确率（Qwen2.5-1.5B）")
ax.legend(fontsize=9)
ax = axes[1]
for gi, (g, gl) in enumerate(zip(groups, glabels)):
    if g in ("orig", "G2"):
        vals = [BASE[k][0 if g == "G2" else 1] for k in sets_l]
    else:
        vals = [rows[g][k]["mean_lp"] for k in sets_l]
    ax.bar(x + gi * w - 0.4 + w / 2, vals, w, label=gl)
ax.axhline(-3.576, color="gray", ls=":", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(["hop1(F0)", "D_so", "D_hop2", "K", "F1_hop2", "retain"], fontsize=9)
ax.set_ylabel("平均答案对数概率")
ax.set_title("恢复后各评测集答案对数概率（越接近 orig 越好）")
ax.legend(fontsize=9)
fig.suptitle("图 R3 · 定向恢复验收：六评测台对比", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIGDIR / "fig22_r3_bars.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("figures saved: fig20/21/22")
