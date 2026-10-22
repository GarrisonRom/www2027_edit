# -*- coding: utf-8 -*-
"""
研究内容一 · 阶段5-6：五组 GA-LoRA 遗忘训练 + 三口径评测
自实现 GA-LoRA (避免外部仓库依赖):
  forget loss = 答案 token 平均对数概率 (梯度下降 = 对数概率上升的反操作, 即遗忘)
  retain loss = KL(M' || M) on 保持集, M 为冻结原模型
组定义:
  orig 原模型 | G1 只忘F0 | G2 F0+F1(贪心5) | G3 F0+全候选(61,不筛选)
  G4 F0+随机5 | G5 只忘F1的5个扩展事实(无F0)
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

WS = Path(__file__).resolve().parents[2]
DATA = WS / "data" / "faithun" / "FaithUn_Dataset.json"
SPLITS = WS / "data" / "faithun" / "splits"
OUT = WS / "exp" / "results"
ADAPTERS = WS / "exp" / "adapters"
MODEL = os.environ.get("QWEN_1_5B_PATH", str(WS / "models" / "Qwen2.5-1.5B-Instruct"))
SEED = 42

TPL1 = {
    "country of citizenship": "What country does {s} hold citizenship in?",
    "place of birth": "Where was {s} born?",
    "continent": "Which continent is {s} associated with?",
    "occupation": "What is the occupation of {s}?",
    "capital": "What is the capital of {s}?",
    "currency": "What is the currency of {s}?",
    "highest point": "What is the highest point of {s}?",
    "country": "In which country is {s} located?",
    "founded by": "Who founded {s}?",
    "religion": "What is the religion of {s}?",
    "location": "Where is {s} located?",
    "native language": "What is the native language of {s}?",
    "sport": "Which sport does {s} play?",
    "headquarters location": "Where is the headquarters of {s} located?",
}


def build_clusters(data):
    return [(qid, ci, c) for qid, clist in data.items() for ci, c in enumerate(clist)]


def enc_qa(tok, q, a):
    qi = tok(q, add_special_tokens=True)["input_ids"]
    full = tok(q + " " + a, add_special_tokens=True)["input_ids"]
    return full, len(qi)


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


@torch.no_grad()
def evaluate(model, tok, clusters, forget_idx, retain_idx):
    res = {"hop1_unlearn": [], "hop1_test": [], "hop2": [], "same_object": [], "retain": []}
    def mcqa(q, a, f):
        sc = score_choices(model, tok, q, [a] + list(f))
        return int(max(range(len(sc)), key=lambda k: sc[k]) == 0)
    for idx in forget_idx:
        c = clusters[idx][2]
        res["hop1_unlearn"].append(mcqa(c["hop1_unlearn"]["questions"][0], c["hop1_unlearn"]["answers"], c["hop1_unlearn"]["false_answer_options"]))
        res["hop1_test"] += [mcqa(q, c["hop1_test"]["answers"], c["hop1_test"]["false_answer_options"]) for q in c["hop1_test"]["questions"]]
        hop2 = c.get("hop2_test", {})
        for si in range(len(hop2.get("answers", []))):
            qs = hop2["questions"][si] if si < len(hop2["questions"]) else []
            a = hop2["answers"][si] if si < len(hop2["answers"]) else ""
            fo = hop2["false_answer_options"][si] if si < len(hop2.get("false_answer_options", [])) else []
            if qs and a and fo:
                res["hop2"].append(sum(mcqa(q, a, fo) for q in qs) / len(qs))
        so = c.get("same_object", {})
        for qs, a, fo in zip(so.get("questions", []), so.get("answers", []), so.get("false_answer_options", [])):
            if qs and a and fo:
                res["same_object"].append(sum(mcqa(q, a, fo) for q in qs[:2]) / min(2, len(qs)))
    for idx in retain_idx:
        c = clusters[idx][2]
        res["retain"].append(mcqa(c["hop1_unlearn"]["questions"][0], c["hop1_unlearn"]["answers"], c["hop1_unlearn"]["false_answer_options"]))
    return {k: sum(v) / len(v) for k, v in res.items()}


def make_group_examples(group, clusters, forget_idx, f1, cands):
    ex = []
    f0 = []
    for idx in forget_idx:
        c = clusters[idx][2]
        f0.append((c["hop1_unlearn"]["questions"][0], c["hop1_unlearn"]["answers"]))
    ext_all = [(TPL1.get(f["relation"], "What is the {r} of {s}?").format(s=f["subject"], r=f["relation"]), f["object"])
               for f in cands]
    ext_f1 = [(TPL1.get(f["relation"], "What is the {r} of {s}?").format(s=f["subject"], r=f["relation"]), f["object"])
              for f in f1]
    rnd = random.Random(SEED)
    if group == "G1":
        ex = f0
    elif group == "G2":
        ex = f0 + ext_f1
    elif group == "G3":
        ex = f0 + ext_all
    elif group == "G4":
        ex = f0 + rnd.sample(ext_all, min(5, len(ext_all)))
    elif group == "G5":
        ex = ext_f1
    return ex


def train_group(model_name, group, forget_ex, retain_ex, tok, base, lr=5e-5, epochs=3, alpha=10.0):
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    EPOCHS = epochs
    for ep in range(EPOCHS):
        random.Random(SEED + ep).shuffle(forget_ex)
        tot = 0.0
        for q, a in forget_ex:
            full, ctx_len = enc_qa(tok, q, a)
            ids = torch.tensor([full]).to(model.device)
            labels = ids.clone()
            labels[:, :ctx_len] = -100
            out = model(input_ids=ids)
            lp = torch.log_softmax(out.logits.float(), -1)
            tgt = ids[:, 1:]
            mask = (labels[:, 1:] != -100).float()
            if mask.sum() == 0:
                continue
            tok_lp = (lp[:, :ids.shape[1] - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask).sum() / mask.sum()
            loss_f = tok_lp  # 下降 = 遗忘
            # KL on a random retain example
            rq, ra = random.Random(hash((q, ep)) & 0xffff).choice(retain_ex)
            rfull, rctx = enc_qa(tok, rq, ra)
            rids = torch.tensor([rfull]).to(model.device)
            # The base weights are already frozen; disabling LoRA gives the
            # reference logits without loading a second full model.
            with model.disable_adapter(), torch.no_grad():
                lf = model(input_ids=rids).logits.float()
            lo = model(input_ids=rids).logits.float()
            kl = (F.softmax(lf, -1) * (F.log_softmax(lf, -1) - F.log_softmax(lo, -1))).sum(-1).mean()
            loss = loss_f + alpha * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"  [{group}] epoch {ep+1}/{EPOCHS} loss={tot/len(forget_ex):.4f}")
    torch.cuda.empty_cache()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["G1", "G2", "G3", "G4", "G5"])
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tag", default="1p5B")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=10.0)
    args = ap.parse_args()

    random.seed(SEED)
    data = json.load(open(DATA, encoding="utf-8"))["data"]
    clusters = build_clusters(data)
    forget_idx = [int(x) for x in (SPLITS / "forget_set_split.txt").read_text().split()]
    retain_idx = [int(x) for x in (SPLITS / "retaining_set_split.txt").read_text().split()][:33]

    prefix = args.tag if (OUT / f"{args.tag}_candidates.json").exists() else args.tag.split("_")[0]
    cand_path = OUT / f"{prefix}_candidates.json"
    f1_path = OUT / f"{prefix}_F1.json"
    if not cand_path.exists():
        cand_path, f1_path = OUT / "candidates.json", OUT / "F1.json"
    f1 = json.load(open(f1_path, encoding="utf-8"))["F_star"]
    cands = json.load(open(cand_path, encoding="utf-8"))["facts"]

    tok = AutoTokenizer.from_pretrained(args.model)
    retain_ex = [(clusters[i][2]["hop1_unlearn"]["questions"][0], clusters[i][2]["hop1_unlearn"]["answers"])
                 for i in retain_idx]

    results = {}
    if not args.skip_train:
        for g in args.groups:
            path = ADAPTERS / f"{args.tag}_{g}"
            if path.exists():
                print(f"[{g}] adapter 已存在，跳过训练")
                continue
            # Load a fresh frozen base for every group so PEFT adapters cannot
            # leak state across the ablation groups.
            base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
            for p in base.parameters():
                p.requires_grad = False
            ex = make_group_examples(g, clusters, forget_idx, f1, cands)
            print(f"[{g}] forget examples: {len(ex)}")
            m = train_group(args.model, g, ex, retain_ex, tok, base,
                            lr=args.lr, epochs=args.epochs, alpha=args.alpha)
            path = ADAPTERS / f"{args.tag}_{g}"
            if path.exists():
                shutil.rmtree(path)
            m.save_pretrained(path)
            del m, base
            torch.cuda.empty_cache()

    # ---- 评测（增量落盘，可断点续跑）----
    jsonl_path = OUT / f"eval_results_{args.tag}.jsonl"
    done = {}
    if jsonl_path.exists():
        for line in open(jsonl_path, encoding="utf-8"):
            r = json.loads(line)
            done[r["group"]] = r
    if "orig" not in done:
        r = evaluate(AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval(), tok, clusters, forget_idx, retain_idx)
        done["orig"] = {"group": "orig", **r}
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(done["orig"], ensure_ascii=False) + "\n")
        print("orig:", {k: round(v, 3) for k, v in r.items()})
    for g in args.groups:
        if g in done:
            print(f"{g}: (已有) ", {k: round(v, 3) for k, v in done[g].items() if k != "group"})
            continue
        path = ADAPTERS / f"{args.tag}_{g}"
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
        from peft import PeftModel
        m = PeftModel.from_pretrained(base, str(path))
        m.eval()
        r = evaluate(m, tok, clusters, forget_idx, retain_idx)
        rec = {"group": g, **r}
        done[g] = rec
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{g}:", {k: round(v, 3) for k, v in r.items()})
        del m, base
        torch.cuda.empty_cache()

    allr = {r["group"]: {k: v for k, v in r.items() if k != "group"} for r in done.values()}
    json.dump(allr, open(OUT / f"eval_results_{args.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved:", OUT / f"eval_results_{args.tag}.json")


if __name__ == "__main__":
    main()
