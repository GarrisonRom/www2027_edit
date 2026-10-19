# -*- coding: utf-8 -*-
"""
研究内容一 · 阶段2-4：候选池探测过滤（内生图）→ 活性核验 → 贪心选集 F1
模型: Qwen2.5-1.5B-Instruct
协议 v1 续:
  - 候选支持事实 = forget 簇 hop2 链的扩展三元组 (chain[1:])
  - 探测: 每事实 2 个手工模板 + 同类关系 distractor 2 个, MCQA 打分
    c(f) = 模板一致率; 掌握阈值 τ=0.75 (预实验证实双峰, τ 不敏感)
  - 活性路径: 阶段1 记录中 hop2 子项 acc>=2/3
  - 覆盖 S(e): e 所在簇的活跃 hop2 链
  - 贪心: 预算 B 内最大覆盖
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WS = Path(__file__).resolve().parents[2]
DATA = WS / "data" / "faithun" / "FaithUn_Dataset.json"
SPLITS = WS / "data" / "faithun" / "splits"
OUT = WS / "exp" / "results"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("QWEN_1_5B_PATH", str(WS / "models" / "Qwen2.5-1.5B-Instruct"))
TAU = 0.75
B_MAIN = 5
SEED = 42

TEMPLATES = {
    "country of citizenship": ["What country does {s} hold citizenship in?", "Which country is {s} a citizen of?"],
    "place of birth": ["Where was {s} born?", "What is the place of birth of {s}?"],
    "continent": ["Which continent is {s} associated with?", "On which continent is {s} located?"],
    "occupation": ["What is the occupation of {s}?", "What does {s} do for a living?"],
    "capital": ["What is the capital of {s}?", "Which city serves as the capital of {s}?"],
    "currency": ["What is the currency of {s}?", "Which currency is used in {s}?"],
    "highest point": ["What is the highest point of {s}?", "What is the highest elevation in {s}?"],
    "country": ["In which country is {s} located?", "What country is {s} in?"],
    "founded by": ["Who founded {s}?", "{s} was founded by whom?"],
    "religion": ["What is the religion of {s}?", "Which religion does {s} belong to?"],
    "location": ["Where is {s} located?"],
    "native language": ["What is the native language of {s}?"],
    "sport": ["Which sport does {s} play?"],
    "headquarters location": ["Where is the headquarters of {s} located?"],
}
FALLBACK = ["What is the {r} of {s}?"]


def build_clusters(data):
    return [(qid, ci, c) for qid, clist in data.items() for ci, c in enumerate(clist)]


@torch.no_grad()
def score_choices(model, tok, question, choices):
    results = []
    for c in choices:
        enc = tok(question + " " + c, return_tensors="pt").to(model.device)
        n = enc["input_ids"].shape[1]
        lp = torch.log_softmax(model(**enc).logits.float(), -1)
        tgt = enc["input_ids"][0, 1:]
        v = lp[0, :n - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        results.append(float(v.sum()) / len(v))
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tag", default="1p5B")
    ap.add_argument("--records", default=str(OUT / "pre_experiment_records.jsonl"))
    args = ap.parse_args()
    random.seed(SEED)
    data = json.load(open(DATA, encoding="utf-8"))["data"]
    clusters = build_clusters(data)
    forget_idx = [int(x) for x in (SPLITS / "forget_set_split.txt").read_text().split()]

    # ---- 阶段1记录: 活性路径 ----
    recs = {json.loads(l)["global_idx"]: json.loads(l)
            for l in open(args.records, encoding="utf-8")}

    # ---- 候选池: 扩展事实 (cluster, sub_idx, triple) ----
    candidates = []          # {fact, cluster, sub_idx}
    rel_objects = defaultdict(set)
    for idx in forget_idx:
        c = clusters[idx][2]
        hop2 = c.get("hop2_test", {})
        for si in range(len(hop2.get("answers", []))):
            chain = hop2["graphs"][si] if si < len(hop2.get("graphs", [])) else None
            if not chain or len(chain) < 2:
                continue
            for tri in chain[1:]:
                s, r, o = tri
                candidates.append({"subject": s, "relation": r, "object": o,
                                   "cluster": idx, "sub_idx": si})
                rel_objects[r].add(o)
    # 去重 (同事实可能服务多条链)
    uniq = {}
    for cd in candidates:
        key = (cd["subject"], cd["relation"], cd["object"])
        uniq.setdefault(key, {"subject": cd["subject"], "relation": cd["relation"],
                              "object": cd["object"], "covers": []})
        uniq[key]["covers"].append((cd["cluster"], cd["sub_idx"]))
    facts = list(uniq.values())
    print(f"候选支持事实: {len(candidates)} 次出现, 去重后 {len(facts)} 个")

    # ---- 探测 ----
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    for f in facts:
        r = f["relation"]
        tpls = TEMPLATES.get(r, FALLBACK)
        pool = list(rel_objects.get(r, set()) - {f["object"]})
        dis = random.sample(pool, min(2, len(pool))) if len(pool) >= 1 else ["unknown", "none"]
        while len(dis) < 2:
            dis.append("unknown" + str(len(dis)))
        per_t = []
        for t in tpls:
            q = t.format(s=f["subject"], r=r)
            sc = score_choices(model, tok, q, [f["object"]] + dis)
            per_t.append(int(max(range(3), key=lambda k: sc[k]) == 0))
        f["consistency"] = sum(per_t) / len(per_t)
        f["mastered"] = bool(f["consistency"] >= TAU)
        f["distractors"] = dis

    mastered = [f for f in facts if f["mastered"]]
    print(f"探测过滤: 掌握 {len(mastered)}/{len(facts)} = {len(mastered)/len(facts):.2f} (外部图→内生图)")

    # ---- 活性路径 ----
    active = {}
    for idx in forget_idx:
        r = recs.get(idx)
        if not r:
            continue
        for it in r["hop2_items"]:
            if it["correct"]:
                active[(idx, it["sub_idx"])] = True
    P_act = list(active.keys())
    print(f"活跃路径: {len(P_act)} / {sum(len(recs[i]['hop2_items']) for i in forget_idx if i in recs)}")

    # ---- 覆盖 + 贪心 ----
    E = [f for f in mastered]
    for f in E:
        f["S"] = [c for c in f["covers"] if c in active]
    E = [f for f in E if f["S"]]
    uncovered = set(P_act)
    greedy_order = []
    curve = [(0, 0)]
    F_sel = []
    while uncovered and len(greedy_order) < len(E):
        best = max(E, key=lambda f: len(set(f["S"]) & uncovered) if f not in greedy_order else -1,
                   )
        gain = len(set(best["S"]) & uncovered)
        if gain == 0:
            break
        greedy_order.append(best)
        F_sel.append(best)
        uncovered -= set(best["S"])
        curve.append((len(greedy_order), len(P_act) - len(uncovered)))
    print(f"可覆盖活跃路径: {len(P_act) - len(uncovered)}/{len(P_act)}; 全选需 {len(F_sel)} 事实")
    for b in range(1, min(10, len(curve))):
        pass

    F1 = F_sel[:B_MAIN]
    print(f"\nF1 = F0(33簇hop1) + {len(F1)} 个扩展事实 (B={B_MAIN}):")
    for f in F1:
        print(f"  - ({f['subject']}, {f['relation']}, {f['object']}) c={f['consistency']:.2f} 覆盖{len(f['S'])}条活跃路径")

    # ---- 保存 ----
    json.dump({"facts": facts, "tau": TAU}, open(OUT / f"{args.tag}_candidates.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({"P_act": [list(p) for p in P_act]}, open(OUT / f"{args.tag}_active_paths.json", "w", encoding="utf-8"), indent=1)
    json.dump({"curve": curve, "P_total": len(P_act)}, open(OUT / f"{args.tag}_greedy_curve.json", "w", encoding="utf-8"), indent=1)
    json.dump({"B": B_MAIN,
               "F_star": [{"subject": f["subject"], "relation": f["relation"], "object": f["object"],
                           "consistency": f["consistency"], "S": [list(c) for c in f["S"]]} for f in F1]},
              open(OUT / f"{args.tag}_F1.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nsaved: candidates.json / active_paths.json / greedy_curve.json / F1.json")


if __name__ == "__main__":
    main()
