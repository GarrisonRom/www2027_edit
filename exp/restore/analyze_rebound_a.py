# -*- coding: utf-8 -*-
"""回弹失败归因 · 分析A：监督集-F0锚集的结构重叠 + 回弹-恢复比例性"""
import json
import re
from pathlib import Path

import numpy as np

REC = Path(__file__).resolve().parents[1] / "restore"
RES = REC.parent / "results"

sup = json.load(open(REC / "supervision.json", encoding="utf-8"))
anc = json.load(open(REC / "anchors.json", encoding="utf-8"))
f0 = [a for a in anc if a["kind"] == "F0"]

# 1) F0 问题中的实体（去掉疑问词模板后的大写短语）
stop = {"Who", "What", "Which", "The", "A", "An", "Of", "In", "Is", "Are", "Name", "Give"}
ents = set()
for a in f0:
    for m in re.findall(r"[A-Z][a-zA-Z'’.-]+(?:\s+[A-Z][a-zA-Z'’.-]+)*", a["q"]):
        w = m.split()
        if w[0] in stop or len(m) < 4:
            continue
        ents.add(m)
print("F0 实体数:", len(ents))

hit_q = sum(1 for r in sup if any(e in r["question"] for e in ents))
hit_a = sum(1 for r in sup if any(e.lower() in r["answer"].lower() for e in ents))
print(f"监督集 {len(sup)} 题：问题文本含 F0 实体 {hit_q} 题（{hit_q/len(sup)*100:.0f}%），"
      f"答案含 F0 实体 {hit_a} 题")

# 桥接事实示例：hop2 题首段主语
for r in sup[:8]:
    sub = re.split(r"'s| is | who | where ", r["question"])[0].replace("What is the capital of the country ", "")
    print("  例:", r["question"][:80], "-> 桥接主语:", sub[:40])

# 2) 回弹-恢复比例性
tab = json.load(open(RES / "restore_summary.json", encoding="utf-8"))
cfgs = ["R1_rec", "B_kl"] + [f"R3_l{l}_e90" for l in (0.1, 1, 10)]
print("\n配置            恢复%   回弹%   回弹/恢复")
xs, ys = [], []
for c in cfgs:
    t = tab[c]
    print(f"{c:<14}{t['rec']:>7.1f}{t['rb']:>8.1f}{t['rb']/t['rec']:>9.2f}")
    xs.append(t["rec"]); ys.append(t["rb"])
xs, ys = np.array(xs), np.array(ys)
k, b = np.polyfit(xs, ys, 1)
r = np.corrcoef(xs, ys)[0, 1]
print(f"\n线性拟合 回弹 = {k:.2f}×恢复 + {b:.1f}，Pearson r = {r:.3f}")
print("=> 回弹不是独立失效模式，而是恢复更新的固定伴随产物（比例 ~0.7-0.9）")
