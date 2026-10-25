# -*- coding: utf-8 -*-
"""
研究内容二 · 阶段1：逐题损伤打分 + 图 D1
损伤定义（论文式）:
  对遗忘后模型 M' 与冻结原模型 M，在事实 f 的标准问句 q 上，
  逐题损伤 d(f) = logp_M(a|q) - logp_M'(a|q)      (a 为答案选项)
即原模型与遗忘后模型对同一答案的长度归一化对数概率之差。
d(f) > 0 表示遗忘使模型对该答案的置信度下降（遗忘生效）；
d(f) < 0 表示遗忘反而抬高了该答案（意外增益 / 保护集受损现场）。

输入: exp/results/per_item_{orig,G1,G2}.jsonl （阶段0 产物）
输出: exp/results/damage_scores.jsonl   （逐题损伤表，后续阶段2-4 的输入）
      研究内容一-示意图/fig13_damage_rank.png （图 D1：损伤排序 + 分源统计）
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

WS = Path(__file__).resolve().parents[2]
OUT = WS / "exp" / "results"
FIGDIR = WS / "研究内容一-示意图"

SRC_ORDER = ["hop1_unlearn", "hop1_test", "hop2", "same_object", "retain"]
SRC_LABEL = {
    "hop1_unlearn": "$F_0$ 遗忘集(hop1)",
    "hop1_test": "$F_0$ 遗忘集(hop1测试)",
    "hop2": "多跳 hop2",
    "same_object": "同对象 same_object",
    "retain": "保持集 retain",
}
SRC_COLOR = {
    "hop1_unlearn": "#d62728",
    "hop1_test": "#ff9896",
    "hop2": "#1f77b4",
    "same_object": "#ff7f0e",
    "retain": "#2ca02c",
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def load(name):
    rec = {}
    for line in open(OUT / f"per_item_{name}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rec[r["item_key"]] = r
    return rec


def main():
    orig, g1, g2 = load("orig"), load("G1"), load("G2")
    keys = sorted(orig.keys())
    rows = []
    for k in keys:
        o, a, b = orig[k], g1[k], g2[k]
        rows.append({
            "item_key": k, "source": o["source"], "cluster_pos": o["cluster_pos"],
            "qid": o["qid"], "question": o["question"], "answer": o["answer"],
            "d_G1": o["lp_answer"] - a["lp_answer"],
            "d_G2": o["lp_answer"] - b["lp_answer"],
            "margin_orig": o["margin"], "margin_G1": a["margin"], "margin_G2": b["margin"],
        })
    with open(OUT / "damage_scores.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 汇总统计 ----
    print(f"{'source':<15}{'n':>5}{'mean d_G1':>12}{'mean d_G2':>12}{'G2<0占比':>10}")
    stats = {}
    for s in SRC_ORDER:
        rs = [r for r in rows if r["source"] == s]
        m1 = sum(r["d_G1"] for r in rs) / len(rs)
        m2 = sum(r["d_G2"] for r in rs) / len(rs)
        neg = sum(1 for r in rs if r["d_G2"] < 0) / len(rs)
        stats[s] = (m1, m2, neg, len(rs))
        print(f"{s:<15}{len(rs):>5}{m1:>12.4f}{m2:>12.4f}{neg:>10.2%}")
    json.dump({s: {"mean_d_G1": v[0], "mean_d_G2": v[1], "neg_frac_G2": v[2], "n": v[3]}
               for s, v in stats.items()},
              open(OUT / "damage_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 图 D1 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    for s in SRC_ORDER:
        rs = sorted([r for r in rows if r["source"] == s], key=lambda r: -r["d_G2"])
        ys = [r["d_G2"] for r in rs]
        ax.plot(range(len(ys)), ys, color=SRC_COLOR[s], lw=1.6, label=SRC_LABEL[s])
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("题目（按 G2 损伤降序）")
    ax.set_ylabel("逐题损伤 $d(f)$")
    ax.set_title("损伤排序曲线（模型：Qwen2.5-1.5B-Instruct + G2 遗忘适配器）")
    ax.legend(fontsize=9)

    ax = axes[1]
    x = range(len(SRC_ORDER))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [stats[s][0] for s in SRC_ORDER], w,
           label="G1（只忘 $F_0$）", color="#7f7f7f")
    ax.bar([i + w / 2 for i in x], [stats[s][1] for s in SRC_ORDER], w,
           label="G2（$F_0$+$F_1$ 贪心5）", color="#d62728")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([SRC_LABEL[s] for s in SRC_ORDER], fontsize=9)
    ax.set_ylabel("平均逐题损伤 $d(f)$")
    ax.set_title("分源平均损伤对比（Qwen2.5-1.5B）")
    ax.legend(fontsize=9)

    fig.suptitle("图 D1 · 逐题损伤 $d(f)=\\log p_M(a|q)-\\log p_{M'}(a|q)$：扩展遗忘集是否造成非目标损伤", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = FIGDIR / "fig13_damage_rank.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    print("saved:", p)


if __name__ == "__main__":
    main()
