# -*- coding: utf-8 -*-
"""R5：每步三向梯度正交化修复（证据D的方案）
g = g_rec − proj_{g_F0}(g_rec) − proj_{g_F1}(g_rec)，配软化 hinge（β=1, margin=0.5）
断点：每个 epoch 存 checkpoint，重跑自动续。
用法: python train_r5.py [--config R5_full|R5_surgery] [--eval_only]"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "damage"))
import component_ablation as ca
import eval_per_item as ep
from train_restore import (ADAPTERS, CONFIGS, EPOCHS, LR0, MODEL, REC, RES,
                           SEED, TAG, answer_lp_t, enc_qa, load_g2_merged, rec_loss)

SPECS = {
    "R5_full":    dict(surgery=True,  beta=1.0, margin=0.5),
    "R5_surgery": dict(surgery=True,  beta=0.0, margin=0.5),
}
ANCH_PER_STEP = 4  # F0/F1 各采样条数，组 batch 一次反传


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="R5_full", choices=list(SPECS))
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()
    spec = SPECS[args.config]
    out_adapter = ADAPTERS / f"{TAG}_{args.config}"
    ckpt = out_adapter.parent / (out_adapter.name + "_ckpt")
    done_epochs = 0
    if ckpt.exists():
        done_epochs = len(list(ckpt.glob("ep*.pt")))
    if out_adapter.exists():
        done_epochs = EPOCHS
    tok = AutoTokenizer.from_pretrained(MODEL)
    sup = json.load(open(REC / "supervision.json", encoding="utf-8"))
    anchors = json.load(open(REC / "anchors.json", encoding="utf-8"))
    anc_f0 = [a for a in anchors if a["kind"] == "F0"]
    anc_f1 = [a for a in anchors if a["kind"] == "F1"]
    w = json.load(open(REC / "layer_lr.json", encoding="utf-8"))

    if done_epochs < EPOCHS and not args.eval_only:
        g2 = load_g2_merged()
        model = get_peft_model(g2, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["gate_proj", "up_proj", "down_proj"]))
        model.train()
        if done_epochs > 0:
            sd = torch.load(ckpt / f"ep{done_epochs}.pt", map_location="cuda")
            model.load_state_dict({**model.state_dict(), **sd}, strict=False)
            print(f"[{args.config}] 恢复 epoch {done_epochs}")
        params = [p for n, p in model.named_parameters() if p.requires_grad]
        groups = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                m = re.search(r"layers\.(\d+)\.", n)
                groups.setdefault(int(m.group(1)) if m else 0, []).append(p)
        opt = torch.optim.AdamW([{"params": ps, "lr": LR0 * w[str(l)]} for l, ps in groups.items()])
        rng = random.Random(SEED)
        print(f"[{args.config}] 监督 {len(sup)} 题 × {EPOCHS} ep, surgery={spec['surgery']} "
              f"beta={spec['beta']} margin={spec['margin']}")
        for ep_i in range(done_epochs, EPOCHS):
            rng.shuffle(sup)
            tot = 0.0
            for ex in sup:
                # ---- 反传1：恢复损失 + 软化 hinge ----
                opt.zero_grad()
                loss = rec_loss(model, tok, ex["question"], ex["answer"])
                if spec["beta"] > 0:
                    for an in rng.sample(anchors, ANCH_PER_STEP):
                        lp = answer_lp_t(model, tok, an["q"], an["a"])
                        loss = loss + spec["beta"] * torch.clamp(lp - an["lp_G2"] - spec["margin"], min=0)
                loss.backward()
                g_rec = {p: p.grad.clone() for p in params}
                # ---- 反传2：锚点上升方向（F0/F1 分开）----
                opt.zero_grad()
                samp_f0 = rng.sample(anc_f0, ANCH_PER_STEP)
                samp_f1 = rng.sample(anc_f1, min(ANCH_PER_STEP, len(anc_f1)))
                lp0 = sum(answer_lp_t(model, tok, a["q"], a["a"]) for a in samp_f0)
                lp1 = sum(answer_lp_t(model, tok, a["q"], a["a"]) for a in samp_f1)
                lp0.backward(retain_graph=True)
                g_f0 = {p: p.grad.clone() for p in params}
                opt.zero_grad()
                lp1.backward()
                g_f1 = {p: p.grad.clone() for p in params}
                # ---- 三向正交化：g_rec 减去 F0、F1 方向投影 ----
                for p in params:
                    g = g_rec[p]
                    for ga in (g_f0[p], g_f1[p]):
                        denom = float((ga * ga).sum()) + 1e-12
                        g = g - float((g * ga).sum()) / denom * ga
                    p.grad = g
                opt.step()
                tot += float(loss)
            print(f"  [{args.config}] ep{ep_i+1}/{EPOCHS} loss={tot/len(sup):.4f}", flush=True)
            ckpt.mkdir(exist_ok=True)
            torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if "lora" in n},
                       ckpt / f"ep{ep_i+1}.pt")
        model.save_pretrained(out_adapter)
        del model
        torch.cuda.empty_cache()
        print("saved:", out_adapter)

    # ---- 评测（同 stage_eval 口径，append 到 restore_results.jsonl）----
    out_path = RES / "restore_results.jsonl"
    done = set()
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["config"], r["quick"]))
    if (args.config, False) in done:
        print(f"[{args.config}] 已有完整评测，跳过")
        return
    from peft import PeftModel
    g2 = load_g2_merged()
    rec = PeftModel.from_pretrained(g2, str(out_adapter)).eval()
    sets, qa = ca.build_sets()
    res = ca.eval_sets(rec, tok, sets, qa)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"config": args.config, "quick": False, "sets": res},
                           ensure_ascii=False) + "\n")
    d = np.mean([res["D_so"]["mean_lp"], res["D_hop2"]["mean_lp"]])
    print(f"[{args.config}] quick=False D mean_lp={d:.3f}")


if __name__ == "__main__":
    main()
