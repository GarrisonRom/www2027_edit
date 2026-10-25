# -*- coding: utf-8 -*-
"""
G6 对照组：G2 同款遗忘集（F0 + F1 贪心5），KL 锚加强（每步 4 条保持样本、alpha=30）
回答：锚加强后 retain 能否守住，且 hop2 封堵不回弹？
复用 train_eval.py 的数据管线与 make_group_examples，训练循环为本地改写（多锚 KL）。
产物：exp/adapters/1p5B_v2_G6/
"""
import json
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import train_eval as te  # 同目录，数据管线复用

WS = Path(__file__).resolve().parents[2]
OUT = WS / "exp" / "results"
ADAPTERS = WS / "exp" / "adapters"
TAG = "1p5B_v2"
GROUP = "G6"
LR, EPOCHS, ALPHA, K_ANCHOR = 5e-5, 3, 30.0, 4


def train_strong_anchor(forget_ex, retain_ex, tok, base):
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(EPOCHS):
        random.Random(te.SEED + ep).shuffle(forget_ex)
        tot = 0.0
        for q, a in forget_ex:
            full, ctx_len = te.enc_qa(tok, q, a)
            ids = torch.tensor([full]).to(model.device)
            labels = ids.clone()
            labels[:, :ctx_len] = -100
            out = model(input_ids=ids)
            lp = torch.log_softmax(out.logits.float(), -1)
            tgt = ids[:, 1:]
            mask = (labels[:, 1:] != -100).float()
            loss_f = (lp[:, :ids.shape[1] - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask).sum() / mask.sum()
            # ---- 加强锚：K_ANCHOR 条保持样本上的平均 KL(M || M') ----
            kls = []
            for rq, ra in random.Random(hash((q, ep)) & 0xffff).sample(retain_ex, K_ANCHOR):
                rfull, _ = te.enc_qa(tok, rq, ra)
                rids = torch.tensor([rfull]).to(model.device)
                with model.disable_adapter(), torch.no_grad():
                    lf = model(input_ids=rids).logits.float()
                lo = model(input_ids=rids).logits.float()
                kls.append((F.softmax(lf, -1) * (F.log_softmax(lf, -1) - F.log_softmax(lo, -1))).sum(-1).mean())
            kl = torch.stack(kls).mean()
            loss = loss_f + ALPHA * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"  [{GROUP}] epoch {ep+1}/{EPOCHS} loss={tot/len(forget_ex):.4f} "
              f"(forget={float(loss_f):.3f} kl={float(kl):.5f})")
    torch.cuda.empty_cache()
    return model


def main():
    random.seed(te.SEED)
    data = json.load(open(te.DATA, encoding="utf-8"))["data"]
    clusters = te.build_clusters(data)
    forget_idx = [int(x) for x in (te.SPLITS / "forget_set_split.txt").read_text().split()]
    retain_idx = [int(x) for x in (te.SPLITS / "retaining_set_split.txt").read_text().split()][:33]
    f1 = json.load(open(OUT / "F1.json", encoding="utf-8"))["F_star"]
    cands = json.load(open(OUT / "candidates.json", encoding="utf-8"))["facts"]
    tok = AutoTokenizer.from_pretrained(te.MODEL)
    retain_ex = [(clusters[i][2]["hop1_unlearn"]["questions"][0], clusters[i][2]["hop1_unlearn"]["answers"])
                 for i in retain_idx]
    ex = te.make_group_examples("G2", clusters, forget_idx, f1, cands)  # 与 G2 同款遗忘集
    print(f"[{GROUP}] forget examples: {len(ex)}（=G2），KL 锚 {K_ANCHOR} 样本 × alpha={ALPHA}")

    path = ADAPTERS / f"{TAG}_{GROUP}"
    if path.exists():
        print("adapter 已存在，跳过训练")
        return
    base = AutoModelForCausalLM.from_pretrained(te.MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in base.parameters():
        p.requires_grad = False
    m = train_strong_anchor(ex, retain_ex, tok, base)
    m.save_pretrained(path)
    print("saved:", path)


if __name__ == "__main__":
    main()
