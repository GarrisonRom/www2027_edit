# -*- coding: utf-8 -*-
"""
研究内容二 · 阶段0：逐题损伤落盘（per-item evaluation）
对 orig / G1 / G2 三个模型逐题打分，记录答案选项与最强干扰项的对数概率，
作为阶段1 损伤打分 d(f) = logp_orig - logp_forget 的输入。

口径与研究内容一 evaluate() 完全一致：MCQA 三选一、长度归一化对数概率、无 chat template。
增量写 exp/results/per_item_{model}.jsonl，已存在的 (model,item) 跳过，可断点续跑。
"""
import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

WS = Path(__file__).resolve().parents[2]
DATA = WS / "data" / "faithun" / "FaithUn_Dataset.json"
SPLITS = WS / "data" / "faithun" / "splits"
OUT = WS / "exp" / "results"
ADAPTERS = WS / "exp" / "adapters"
MODEL = os.environ.get("QWEN_1_5B_PATH", str(WS / "models" / "Qwen2.5-1.5B-Instruct"))
TAG = "1p5B_v2"  # 与训练超参一致的 v2 系列


def build_clusters(data):
    return [(qid, ci, c) for qid, clist in data.items() for ci, c in enumerate(clist)]


@torch.no_grad()
def score_choices(model, tok, question, choices):
    res = []
    for c in choices:
        enc = tok(question + " " + c, return_tensors="pt").to(model.device)
        n = enc["input_ids"].shape[1]
        lp = torch.log_softmax(model(**enc).logits.float(), -1)
        tgt = enc["input_ids"][0, 1:]
        v = lp[0, :n - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        res.append(float(v.sum()) / len(v))
    return res


def build_items(clusters, forget_idx, retain_idx):
    """逐题列表，每题: (key, source, cluster_pos, qid, question, answer, falses)"""
    items = []
    for pos, idx in enumerate(forget_idx):
        qid, _, c = clusters[idx]
        h1 = c["hop1_unlearn"]
        items.append((f"hop1_unlearn|{pos}", "hop1_unlearn", pos, qid,
                      h1["questions"][0], h1["answers"], h1["false_answer_options"]))
        t = c["hop1_test"]
        for qi, q in enumerate(t["questions"]):
            items.append((f"hop1_test|{pos}|{qi}", "hop1_test", pos, qid,
                          q, t["answers"], t["false_answer_options"]))
        hop2 = c.get("hop2_test", {})
        n_sub = len(hop2.get("answers", []))
        for si in range(n_sub):
            qs = hop2["questions"][si] if si < len(hop2["questions"]) else []
            a = hop2["answers"][si] if si < len(hop2["answers"]) else ""
            fo = hop2["false_answer_options"][si] if si < len(hop2.get("false_answer_options", [])) else []
            for qi, q in enumerate(qs):
                items.append((f"hop2|{pos}|{si}|{qi}", "hop2", pos, qid, q, a, fo))
        so = c.get("same_object", {})
        n_so = 0
        for si, (qs, a, fo) in enumerate(zip(so.get("questions", []), so.get("answers", []), so.get("false_answer_options", []))):
            if qs and a and fo:
                for qi, q in enumerate(qs[:2]):  # 与 evaluate() 同口径：每组前 2 问
                    items.append((f"same_object|{pos}|{si}|{qi}", "same_object", pos, qid,
                                  q, a, fo))
                    n_so += 1
    for pos, idx in enumerate(retain_idx):
        qid, _, c = clusters[idx]
        h1 = c["hop1_unlearn"]
        items.append((f"retain|{pos}", "retain", pos, qid,
                      h1["questions"][0], h1["answers"], h1["false_answer_options"]))
    return items


def load_done(path):
    done = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done.add(r["item_key"])
    return done


def eval_model(name, items, tok):
    if name == "orig":
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
        label = "orig"
    else:
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
        m = PeftModel.from_pretrained(base, str(ADAPTERS / f"{TAG}_{name}")).eval()
        label = name
    out_path = OUT / f"per_item_{label}.jsonl"
    done = load_done(out_path)
    todo = [it for it in items if it[0] not in done]
    print(f"[{label}] 总题数 {len(items)}，已完成 {len(done)}，待跑 {len(todo)}")
    f = open(out_path, "a", encoding="utf-8")
    for i, (key, source, pos, qid, q, a, fo) in enumerate(todo):
        sc = score_choices(m, tok, q, [a] + list(fo))
        lp_answer = sc[0]
        lp_best_wrong = max(sc[1:])
        rec = {"model": label, "item_key": key, "source": source,
               "cluster_pos": pos, "qid": qid, "question": q, "answer": a,
               "lp_answer": lp_answer, "lp_best_wrong": lp_best_wrong,
               "correct": int(max(range(len(sc)), key=lambda k: sc[k]) == 0),
               "margin": lp_answer - lp_best_wrong}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 50 == 0:
            f.flush()
            print(f"[{label}] {i+1}/{len(todo)}")
    f.close()
    del m
    torch.cuda.empty_cache()
    print(f"[{label}] 完成，写入 {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["orig", "G1", "G2"])
    args = ap.parse_args()

    data = json.load(open(DATA, encoding="utf-8"))["data"]
    clusters = build_clusters(data)
    forget_idx = [int(x) for x in (SPLITS / "forget_set_split.txt").read_text().split()]
    retain_idx = [int(x) for x in (SPLITS / "retaining_set_split.txt").read_text().split()][:33]
    items = build_items(clusters, forget_idx, retain_idx)
    print(f"逐题清单共 {len(items)} 题")

    tok = AutoTokenizer.from_pretrained(MODEL)
    for name in args.models:
        eval_model(name, items, tok)


if __name__ == "__main__":
    main()
