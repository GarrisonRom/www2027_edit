# -*- coding: utf-8 -*-
"""
研究内容三 · R0–R4 一体化实验脚本
  python train_restore.py --stage r0                 构建监督集/锚集/能量空间/层权重
  python train_restore.py --stage train --config R1_rec
  python train_restore.py --stage eval  --config R1_rec [--quick]

设计（见 研究内容三实验规划.md）：
  恢复 = G2-merge + 新 LoRA(只 MLP, r8a16)，逐层 lr 按显影得分加权
  L = L_rec + beta*L_ans + lambda*L_proj
  L_rec : 监督集答案 NLL（梯度下降 = 答案对数概率回升）
  L_ans : F0/F1 答案 hinge（禁止答案概率回升，封堵共享token盲区）
  L_proj: sum_l ||A_l U_l||^2_F （能量空间投影：禁止从受保护输入子空间读取）
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "damage"))
import component_ablation as ca
import eval_per_item as ep

WS = Path(__file__).resolve().parents[2]
REC = WS / "exp" / "restore"
RES = WS / "exp" / "results"
ATT = RES / "attribution"
ADAPTERS = WS / "exp" / "adapters"
MODEL = ep.MODEL
G2_ADAPTER = ADAPTERS / "1p5B_v2_G2"
TAG = "restore"
SEED = 42
LR0, EPOCHS = 5e-5, 3
D_THRESH = 2.0
HINGE_MARGIN = 0.0

CONFIGS = {
    "R1_rec":      dict(beta=0.0, lam=0.0, eta=0.90, kl=False),
    "R2_ans":      dict(beta=10.0, lam=0.0, eta=0.90, kl=False),
    "R3_l0.1_e80": dict(beta=0.0, lam=0.1, eta=0.80, kl=False),
    "R3_l0.1_e90": dict(beta=0.0, lam=0.1, eta=0.90, kl=False),
    "R3_l0.1_e95": dict(beta=0.0, lam=0.1, eta=0.95, kl=False),
    "R3_l1_e80":   dict(beta=0.0, lam=1.0, eta=0.80, kl=False),
    "R3_l1_e90":   dict(beta=0.0, lam=1.0, eta=0.90, kl=False),
    "R3_l1_e95":   dict(beta=0.0, lam=1.0, eta=0.95, kl=False),
    "R3_l10_e80":  dict(beta=0.0, lam=10.0, eta=0.80, kl=False),
    "R3_l10_e90":  dict(beta=0.0, lam=10.0, eta=0.90, kl=False),
    "R3_l10_e95":  dict(beta=0.0, lam=10.0, eta=0.95, kl=False),
    "R4_combo":    dict(beta=10.0, lam=1.0, eta=0.90, kl=False),
    "B_kl":        dict(beta=0.0, lam=0.0, eta=0.90, kl=True),  # 旧范式对照 rec+KL
}


def load_g2_merged():
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    return PeftModel.from_pretrained(base, str(G2_ADAPTER)).merge_and_unload().eval()


@torch.no_grad()
def answer_lp(model, tok, q, a):
    enc = tok(q + " " + a, return_tensors="pt").to(model.device)
    n = enc["input_ids"].shape[1]
    lp = torch.log_softmax(model(**enc).logits.float(), -1)
    tgt = enc["input_ids"][0, 1:]
    v = lp[0, :n - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return float(v.sum()) / len(v)


def rec_loss(model, tok, q, a):
    """监督损失：答案 NLL（下降 = 答案概率回升）"""
    qi = tok(q, add_special_tokens=True)["input_ids"]
    full = tok(q + " " + a, add_special_tokens=True)["input_ids"]
    ids = torch.tensor([full]).to(model.device)
    labels = ids.clone()
    labels[:, :len(qi)] = -100
    out = model(input_ids=ids)
    lp = torch.log_softmax(out.logits.float(), -1)
    tgt = ids[:, 1:]
    mask = (labels[:, 1:] != -100).float()
    return -(lp[:, :ids.shape[1] - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask).sum() / mask.sum()


# ================= R0 =================
def stage_r0():
    REC.mkdir(parents=True, exist_ok=True)
    # ---- 1. 恢复监督集 ----
    rows = [json.loads(l) for l in open(RES / "damage_scores.jsonl", encoding="utf-8")]
    f1 = json.load(open(RES / "F1.json", encoding="utf-8"))["F_star"]
    forget_ans = {f["object"].lower() for f in f1}
    for l in open(RES / "per_item_orig.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["source"] == "hop1_unlearn":
            forget_ans.add(r["answer"].lower())

    def shared(a):
        a = a.lower()
        return any(a == f or (len(f) > 4 and f in a) or (len(a) > 4 and a in f) for f in forget_ans)

    sup = [r for r in rows if r["margin_orig"] > 0 and r["d_G2"] >= D_THRESH and not shared(r["answer"])]
    by = {}
    for r in sup:
        by[r["source"]] = by.get(r["source"], 0) + 1
    print(f"监督集 {len(sup)} 题: {by}")
    json.dump([{k: r[k] for k in ["item_key", "source", "question", "answer", "d_G2"]} for r in sup],
              open(REC / "supervision.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 2. 锚集：F0 全部问法 + F1 模板问法 ----
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probe"))
    from build_f1 import TEMPLATES, FALLBACK
    data = json.load(open(ep.DATA, encoding="utf-8"))["data"]
    clusters = ep.build_clusters(data)
    forget_idx = [int(x) for x in (ep.SPLITS / "forget_set_split.txt").read_text().split()]
    anchors = []
    for idx in forget_idx:
        c = clusters[idx][2]
        ans = c["hop1_unlearn"]["answers"]
        for q in [c["hop1_unlearn"]["questions"][0]] + list(c["hop1_test"]["questions"]):
            anchors.append({"q": q, "a": ans, "kind": "F0"})
    for f in f1:
        tpls = TEMPLATES.get(f["relation"], FALLBACK)
        for t in tpls:
            anchors.append({"q": t.format(s=f["subject"], r=f["relation"]),
                            "a": f["object"], "kind": "F1"})

    tok = AutoTokenizer.from_pretrained(MODEL)
    g2 = load_g2_merged()
    for an in anchors:
        an["lp_G2"] = answer_lp(g2, tok, an["q"], an["a"])
    print(f"锚集 {len(anchors)} 条（F0 {sum(1 for a in anchors if a['kind']=='F0')} + F1 {sum(1 for a in anchors if a['kind']=='F1')}）")
    json.dump(anchors, open(REC / "anchors.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 3. 能量空间：锚集激活协方差的 top-m 主方向 ----
    NL = g2.config.num_hidden_layers
    cov_g = [None] * NL   # gate/up 输入（post-LN, 1536）
    cov_d = [None] * NL   # down 输入（中间激活, 8960）

    def mk(which, slot):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            c = x.T @ x
            slot[0] = c if slot[0] is None else slot[0] + c
        return hook

    hooks = []
    slots = {}
    for l in range(NL):
        slots[("g", l)] = [None]
        slots[("d", l)] = [None]
        hooks.append(g2.model.layers[l].mlp.gate_proj.register_forward_hook(mk("g", slots[("g", l)])))
        hooks.append(g2.model.layers[l].mlp.down_proj.register_forward_hook(mk("d", slots[("d", l)])))
    with torch.no_grad():
        for an in anchors:
            enc = tok(an["q"] + " " + an["a"], return_tensors="pt").to("cuda")
            g2(**enc)
    for h in hooks:
        h.remove()
    n_tok = sum(len(tok(an["q"] + " " + an["a"], add_special_tokens=True)["input_ids"]) for an in anchors)

    U = {}
    for l in range(NL):
        for key, ddim in [("g", 1536), ("d", 8960)]:
            C = slots[(key, l)][0] / n_tok
            evals, evecs = torch.linalg.eigh(C)
            ev = evals.flip(0).cpu().numpy()
            cum = np.cumsum(ev) / ev.sum()
            row = {}
            for eta in [0.80, 0.90, 0.95]:
                m = max(1, int(np.searchsorted(cum, eta)) + 1)
                row[f"U{int(eta*100)}"] = evecs.flip(1)[:, :m].half().cpu().numpy()
            U[f"{key}{l}"] = row
    np.savez(REC / "energy_space.npz", **{k2: v2 for k1, row in U.items() for k2, v2 in
                                          [(f"{k1}_{tag}", mat) for tag, mat in row.items()]})
    print("能量空间已存 energy_space.npz（每层 gate/down 三档 eta）")
    del g2
    torch.cuda.empty_cache()

    # ---- 4. 逐层学习率权重 ----
    ffn = json.load(open(ATT / "ffn_G2.json", encoding="utf-8"))["per_layer"]
    s = np.array([r["D"] for r in ffn])
    w = s / s.mean()
    json.dump({str(l): float(w[l]) for l in range(28)},
              open(REC / "layer_lr.json", "w", encoding="utf-8"), indent=1)
    print("层权重: 22-27 =", [round(float(w[l]), 2) for l in range(22, 28)],
          " 0-4 =", [round(float(w[l]), 2) for l in range(5)])


# ================= train =================
def answer_lp_t(model, tok, q, a):
    """带梯度的答案长度归一化对数概率"""
    enc = tok(q + " " + a, return_tensors="pt").to(model.device)
    n = enc["input_ids"].shape[1]
    lp = torch.log_softmax(model(**enc).logits.float(), -1)
    tgt = enc["input_ids"][0, 1:]
    v = lp[0, :n - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return v.sum() / len(v)


def proj_penalty(model, Uz, eta_key):
    """能量空间投影惩罚：sum_l ||A_l U_l||^2_F（禁止从受保护输入子空间读取）"""
    total = None
    mods = dict(model.named_modules())
    for l in range(28):
        Ug = torch.tensor(Uz[f"g{l}_{eta_key}"]).to("cuda").float()
        Ud = torch.tensor(Uz[f"d{l}_{eta_key}"]).to("cuda").float()
        for name, Um in [("gate_proj", Ug), ("up_proj", Ug), ("down_proj", Ud)]:
            pat = re.compile(rf"layers\.{l}\.mlp\.{name}\.lora_A\.default$")
            key = next(k for k in mods if pat.search(k))
            A = mods[key].weight
            term = ((A @ Um) ** 2).sum()
            total = term if total is None else total + term
    return total


def enc_qa(tok, q, a):
    qi = tok(q, add_special_tokens=True)["input_ids"]
    full = tok(q + " " + a, add_special_tokens=True)["input_ids"]
    return full, len(qi)


def stage_train(config):
    spec = CONFIGS[config]
    out_adapter = ADAPTERS / f"{TAG}_{config}"
    if out_adapter.exists():
        print(f"[{config}] adapter 已存在，跳过训练")
        return
    sup = json.load(open(REC / "supervision.json", encoding="utf-8"))
    anchors = json.load(open(REC / "anchors.json", encoding="utf-8"))
    w = json.load(open(REC / "layer_lr.json", encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    g2 = load_g2_merged()
    model = get_peft_model(g2, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["gate_proj", "up_proj", "down_proj"]))
    model.train()
    groups = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            m = re.search(r"layers\.(\d+)\.", n)
            groups.setdefault(int(m.group(1)) if m else 0, []).append(p)
    opt = torch.optim.AdamW([{"params": ps, "lr": LR0 * w[str(l)]} for l, ps in groups.items()])
    Uz = np.load(REC / "energy_space.npz") if spec["lam"] > 0 else None
    eta_key = f"U{int(spec['eta'] * 100)}"
    retain_ex = None
    if spec["kl"]:
        clusters = ep.build_clusters(json.load(open(ep.DATA, encoding="utf-8"))["data"])
        retain_idx = [int(x) for x in (ep.SPLITS / "retaining_set_split.txt").read_text().split()][:33]
        retain_ex = [(clusters[i][2]["hop1_unlearn"]["questions"][0], clusters[i][2]["hop1_unlearn"]["answers"])
                     for i in retain_idx]

    rng = random.Random(SEED)
    print(f"[{config}] 监督 {len(sup)} 题 × {EPOCHS} ep, beta={spec['beta']} lam={spec['lam']} kl={spec['kl']}")
    for ep_i in range(EPOCHS):
        rng.shuffle(sup)
        tot = 0.0
        for ex in sup:
            loss = rec_loss(model, tok, ex["question"], ex["answer"])
            if spec["beta"] > 0:
                for an in rng.sample(anchors, 4):
                    lp = answer_lp_t(model, tok, an["q"], an["a"])
                    loss = loss + spec["beta"] * torch.clamp(lp - an["lp_G2"] - HINGE_MARGIN, min=0)
            if spec["lam"] > 0:
                loss = loss + spec["lam"] * proj_penalty(model, Uz, eta_key)
            if spec["kl"]:
                rq, ra = rng.choice(retain_ex)
                rfull, _ = enc_qa(tok, rq, ra)
                rids = torch.tensor([rfull]).to(model.device)
                with model.disable_adapter():
                    lf = model(input_ids=rids).logits.float()
                lo = model(input_ids=rids).logits.float()
                kl = (F.softmax(lf, -1) * (F.log_softmax(lf, -1) - F.log_softmax(lo, -1))).sum(-1).mean()
                loss = loss + 10.0 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"  [{config}] ep{ep_i+1}/{EPOCHS} loss={tot/len(sup):.4f}", flush=True)
    model.save_pretrained(out_adapter)
    del model
    torch.cuda.empty_cache()
    print("saved:", out_adapter)


# ================= eval =================
def stage_eval(config, quick):
    out_path = RES / "restore_results.jsonl"
    done = set()
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["config"], r["quick"]))
    if (config, quick) in done:
        print(f"[{config}] quick={quick} 已有结果，跳过")
        return
    tok = AutoTokenizer.from_pretrained(MODEL)
    g2 = load_g2_merged()
    rec = PeftModel.from_pretrained(g2, str(ADAPTERS / f"{TAG}_{config}")).eval()
    sets, qa = ca.build_sets()
    if quick:
        sets = {k: v for k, v in sets.items() if k in ("D_so", "D_hop2")}
    res = ca.eval_sets(rec, tok, sets, qa)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"config": config, "quick": quick, "sets": res},
                           ensure_ascii=False) + "\n")
    d = np.mean([res["D_so"]["mean_lp"], res["D_hop2"]["mean_lp"]])
    print(f"[{config}] quick={quick} D mean_lp={d:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["r0", "train", "eval"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.stage == "r0":
        stage_r0()
    elif args.stage == "train":
        assert args.config in CONFIGS
        stage_train(args.config)
    else:
        assert args.config in CONFIGS
        stage_eval(args.config, args.quick)
