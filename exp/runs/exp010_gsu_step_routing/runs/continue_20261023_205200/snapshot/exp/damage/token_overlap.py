# -*- coding: utf-8 -*-
"""分析：D/K 组答案与遗忘集答案的 token 重叠（解释低层分离 + 共享token机制）"""
import json
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
ATT = WS / "exp" / "results" / "attribution"

items = json.load(open(ATT / "items.json", encoding="utf-8"))
qa = json.load(open(ATT / "items_qa.json", encoding="utf-8"))

# 遗忘集答案：F0 33 簇 hop1 + F1 5 个扩展事实
f1 = json.load(open(WS / "exp" / "results" / "F1.json", encoding="utf-8"))["F_star"]
forget_ans = {f["object"].lower() for f in f1}
for l in open(WS / "exp" / "results" / "per_item_orig.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["source"] == "hop1_unlearn":
        forget_ans.add(r["answer"].lower())

def overlap(keys):
    n_shared = n_sub = 0
    for k in keys:
        a = qa[k]["a"].lower()
        if a in forget_ans:
            n_shared += 1
        elif any(w in a for w in forget_ans if len(w) > 4) or any(a in w for w in forget_ans if len(a) > 4):
            n_sub += 1
    return n_shared, n_sub, len(keys)

for name in ["D_same_object", "K_same_object", "D_hop2", "K_hop2"]:
    s, sub, n = overlap(items[name])
    print(f"{name:<16} 答案恰为遗忘答案 {s}/{n}（{s/n:.0%}），子串重叠 {sub}/{n}（{sub/n:.0%}）")
