# -*- coding: utf-8 -*-
"""内生知识图谱可视化：F0 + 候选支持事实 + F1 选中"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

WS = Path(__file__).resolve().parents[2]
RES = WS / "exp" / "results"
DATA = WS / "data" / "faithun" / "FaithUn_Dataset.json"
SPLITS = WS / "data" / "faithun" / "splits"
FIG = WS / "研究内容一-示意图"

cands = json.load(open(RES / "candidates.json", encoding="utf-8"))["facts"]
f1_set = {(f["subject"], f["relation"], f["object"])
          for f in json.load(open(RES / "F1.json", encoding="utf-8"))["F_star"]}

data = json.load(open(DATA, encoding="utf-8"))["data"]
clusters = [(qid, ci, c) for qid, clist in data.items() for ci, c in enumerate(clist)]
forget_idx = [int(x) for x in (SPLITS / "forget_set_split.txt").read_text().split()]
f0_facts = []
for idx in forget_idx:
    c = clusters[idx][2]
    g = c["hop1_unlearn"]["graphs"]
    f0_facts.append((g[0], g[1], g[2]))

G = nx.Graph()
edge_kind = {}
for s, r, o in f0_facts:
    G.add_edge(s, o, relation=r)
    edge_kind[(s, o)] = "F0"
for f in cands:
    s, r, o = f["subject"], f["relation"], f["object"]
    G.add_edge(s, o, relation=r)
    key = (s, o)
    if (s, r, o) in f1_set:
        edge_kind[key] = "F1"
    elif f["mastered"]:
        edge_kind[key] = "mastered"
    else:
        edge_kind[key] = "filtered"

# 共享邻居去重（同对实体多条事实取最重要类别）
def rank(k):
    return {"F1": 3, "F0": 2, "mastered": 1, "filtered": 0}[k]

best = {}
for (s, o), k in edge_kind.items():
    if (s, o) not in best or rank(k) > rank(best[(s, o)]):
        best[(s, o)] = k

pos = nx.spring_layout(G, seed=7, k=0.6, iterations=80)

fig, ax = plt.subplots(figsize=(15, 11))
ax.axis("off")
ax.set_title("FaithUn Forget 集内生知识图谱（Qwen2.5-1.5B-Instruct 探测）\n"
             r"红=$F_0$ 待遗忘 (33) | 绿=探测掌握 (41) | 灰虚线=被过滤 (20) | 蓝粗=$F_1$ 选中 (5)",
             fontsize=13, fontweight="bold")

style = {
    "F0":       dict(edge_color="#C44E52", width=2.2, style="-",  alpha=0.9),
    "F1":       dict(edge_color="#4C72B0", width=6.0, style="-",  alpha=0.85),
    "mastered": dict(edge_color="#55A868", width=1.4, style="-",  alpha=0.6),
    "filtered": dict(edge_color="#999999", width=1.0, style="--", alpha=0.45),
}
for k, st in style.items():
    es = [(s, o) for (s, o), kk in best.items() if kk == k]
    if es:
        nx.draw_networkx_edges(G, pos, edgelist=es, ax=ax, **st)

deg = dict(G.degree())
sizes = [80 + 60 * deg[n] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                       node_color="#EAF0F8", edgecolors="#4C72B0", linewidths=1.0, alpha=0.95)

# 只标注高度数节点 + F1 相关节点，避免混乱
important = {n for n in G.nodes if deg[n] >= 3}
for s, r, o in f1_set:
    important |= {s, o}
labels = {n: (n if len(n) <= 22 else n[:20] + "…") for n in important}
nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)

# 图例
from matplotlib.lines import Line2D
legend = [
    Line2D([0], [0], color="#C44E52", lw=2.2, label="$F_0$ 待遗忘 (33)"),
    Line2D([0], [0], color="#4C72B0", lw=5, label="$F_1$ 选中的扩展事实 (5)"),
    Line2D([0], [0], color="#55A868", lw=1.4, label="探测掌握：内生图保留 (41)"),
    Line2D([0], [0], color="#999999", lw=1, ls="--", label="被过滤：模型未掌握 (20)"),
]
ax.legend(handles=legend, loc="lower left", fontsize=10, framealpha=0.9)

fig.tight_layout()
out = FIG / "fig12_endogenous_kg.png"
fig.savefig(out, bbox_inches="tight", dpi=160)
print("saved:", out, "| nodes:", G.number_of_nodes(), "| edges:", len(best))
