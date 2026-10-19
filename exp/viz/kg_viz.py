# -*- coding: utf-8 -*-
"""
kg_viz.py — 内生知识图谱可视化模块（可复用）

用法（命令行）:
    python kg_viz.py --candidates candidates.json --f1 F1.json --out fig.png

用法（模块）:
    from kg_viz import load_endogenous_graph, draw_kg
    G, meta = load_endogenous_graph(candidates, f1_facts, f0_facts)
    draw_kg(G, meta, out_path="fig.png", title="...")
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 边类别样式（rank 越大越重要，同一对实体只画最重要的一类）
EDGE_STYLE = {
    "F1":       dict(edge_color="#4C72B0", width=6.0, style="-",  alpha=0.85, rank=3),
    "F0":       dict(edge_color="#C44E52", width=2.2, style="-",  alpha=0.90, rank=2),
    "mastered": dict(edge_color="#55A868", width=1.4, style="-",  alpha=0.60, rank=1),
    "filtered": dict(edge_color="#999999", width=1.0, style="--", alpha=0.45, rank=0),
}
CATEGORY_LABELS = {
    "F0": "$F_0$ 待遗忘",
    "F1": "$F_1$ 选中的扩展事实",
    "mastered": "探测掌握（内生图保留）",
    "filtered": "被过滤（模型未掌握）",
}
KIND_ZH = {"F0": "待遗忘", "F1": "F₁选中", "mastered": "掌握", "filtered": "过滤"}


def load_endogenous_graph(candidates, f1_facts, f0_facts):
    """构建内生图。

    参数:
        candidates: list[dict]  探测过的候选支持事实, 需含 subject/relation/object/mastered
        f1_facts:   set[(s,r,o)] 选中的扩展事实
        f0_facts:   list[(s,r,o)] 种子遗忘事实
    返回:
        G:    networkx.Graph, 边属性 kind / relation
        meta: dict, 统计信息
    """
    G = nx.Graph()
    edge_kind = {}
    for s, r, o in f0_facts:
        G.add_edge(s, o, relation=r)
        edge_kind[(s, o)] = "F0"
    n_cand = n_m = n_f = 0
    for f in candidates:
        s, r, o = f["subject"], f["relation"], f["object"]
        G.add_edge(s, o, relation=r)
        n_cand += 1
        if (s, r, o) in f1_facts:
            edge_kind[(s, o)] = "F1"
        elif f["mastered"]:
            edge_kind[(s, o)] = "mastered"
            n_m += 1
        else:
            edge_kind[(s, o)] = "filtered"
            n_f += 1
    # 同一对实体多条事实时保留最重要类别
    best = {}
    for (s, o), k in edge_kind.items():
        if (s, o) not in best or EDGE_STYLE[k]["rank"] > EDGE_STYLE[best[(s, o)]]["rank"]:
            best[(s, o)] = k
    nx.set_edge_attributes(G, {(s, o): k for (s, o), k in best.items()}, "kind")
    meta = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": len(best),
        "n_f0": len(f0_facts),
        "n_candidates": n_cand,
        "n_mastered": n_m,
        "n_filtered": n_f,
        "n_f1": len(f1_facts),
    }
    return G, meta


def draw_kg(G, meta, out_path, title=None, figsize=(15, 11), seed=7,
            label_min_degree=3, label_f1_nodes=True, label_max_len=22, dpi=160):
    """绘制并保存知识图谱 PNG。"""
    pos = nx.spring_layout(G, seed=seed, k=0.6, iterations=80)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    if title is None:
        title = (f"内生知识图谱\n红=$F_0$({meta['n_f0']}) | 绿=掌握({meta['n_mastered']}) | "
                 f"灰虚线=过滤({meta['n_filtered']}) | 蓝粗=$F_1$({meta['n_f1']})")
    ax.set_title(title, fontsize=13, fontweight="bold")

    kinds = nx.get_edge_attributes(G, "kind")
    for k, st in EDGE_STYLE.items():
        es = [(s, o) for (s, o), kk in kinds.items() if kk == k]
        if es:
            nx.draw_networkx_edges(G, pos, edgelist=es, ax=ax,
                                   edge_color=st["edge_color"], width=st["width"],
                                   style=st["style"], alpha=st["alpha"])
    deg = dict(G.degree())
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=[80 + 60 * deg[n] for n in G.nodes()],
                           node_color="#EAF0F8", edgecolors="#4C72B0",
                           linewidths=1.0, alpha=0.95)
    important = {n for n in G.nodes if deg[n] >= label_min_degree}
    if label_f1_nodes:
        for s, o, d in G.edges(data=True):
            if d["kind"] == "F1":
                important |= {s, o}
    labels = {n: (n if len(n) <= label_max_len else n[:label_max_len - 1] + "…") for n in important}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)

    handles = [Line2D([0], [0], color=EDGE_STYLE[k]["edge_color"],
                      lw=max(1, EDGE_STYLE[k]["width"] / 1.5),
                      ls=EDGE_STYLE[k]["style"],
                      label=f"{CATEGORY_LABELS[k]} ({meta['n_' + ('f0' if k=='F0' else 'f1' if k=='F1' else k)]})")
               for k in ["F0", "F1", "mastered", "filtered"]]
    ax.legend(handles=handles, loc="lower left", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return meta


def _default_inputs(ws):
    res = ws / "exp" / "results"
    data = json.load(open(ws / "data" / "faithun" / "FaithUn_Dataset.json", encoding="utf-8"))["data"]
    clusters = [(qid, ci, c) for qid, clist in data.items() for ci, c in enumerate(clist)]
    splits = ws / "data" / "faithun" / "splits"
    forget_idx = [int(x) for x in (splits / "forget_set_split.txt").read_text().split()]
    f0 = []
    for idx in forget_idx:
        g = clusters[idx][2]["hop1_unlearn"]["graphs"]
        f0.append((g[0], g[1], g[2]))
    cands = json.load(open(res / "candidates.json", encoding="utf-8"))["facts"]
    f1 = {(f["subject"], f["relation"], f["object"])
          for f in json.load(open(res / "F1.json", encoding="utf-8"))["F_star"]}
    return cands, f1, f0


def main():
    ap = argparse.ArgumentParser(description="内生知识图谱可视化")
    ap.add_argument("--ws", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--candidates", default=None, help="候选事实 JSON（默认读 exp/results/candidates.json）")
    ap.add_argument("--f1", default=None, help="F1 JSON（默认读 exp/results/F1.json）")
    ap.add_argument("--out", default=None, help="输出 PNG 路径")
    ap.add_argument("--title", default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ws = Path(args.ws)
    cands, f1, f0 = _default_inputs(ws)
    if args.candidates:
        cands = json.load(open(args.candidates, encoding="utf-8"))["facts"]
    if args.f1:
        f1 = {(f["subject"], f["relation"], f["object"]) for f in json.load(open(args.f1, encoding="utf-8"))["F_star"]}
    out = args.out or str(ws / "研究内容一-示意图" / "fig12_endogenous_kg.png")

    G, meta = load_endogenous_graph(cands, f1, f0)
    title = args.title or ("FaithUn Forget 集内生知识图谱（Qwen2.5-1.5B-Instruct 探测）\n"
                           f"红=$F_0$({meta['n_f0']}) | 绿=掌握({meta['n_mastered']}) | "
                           f"灰虚线=过滤({meta['n_filtered']}) | 蓝粗=$F_1$({meta['n_f1']})")
    draw_kg(G, meta, out, title=title, seed=args.seed)
    print("saved:", out, "|", meta)


if __name__ == "__main__":
    main()
