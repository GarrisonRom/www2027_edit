# -*- coding: utf-8 -*-
"""
研究内容二 · 阶段2：激活/梯度归因
定位"伤在网络的哪一层、哪些头、哪些 FFN 神经元"，并与阶段1 损伤分对齐。

四部分（均可断点续跑，产物在 exp/results/attribution/）：
  drift  逐层 hidden states 的 M vs M′ 余弦漂移（受损组 vs 对照组曲线 → 图 D2 数据）
  heads  逐头注意力（末 token 对前文的平均注意力）M′−M 差值，受损−对照 → 28×12 热力图
  ffn    FFN 中间神经元激活 |M−M′| 漂移，受损−对照 → top-k 候选组件（阶段3 输入）
  grad   答案 NLL 对原模型 M 的逐层梯度范数（Fisher 代理，受损 vs 对照）

设计纪律（对齐研究内容二实验规划 §3）：
  组间配对：D 受损组 = same_object top-24 + hop2 top-24（按 d_G2）
            K 对照组 = 同来源 d_G2 最低 24+24（源内匹配，排除源混淆）
  双模型对 G1 / G2 分开跑，不混。
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

WS = Path(__file__).resolve().parents[2]
OUT = WS / "exp" / "results"
ATT = OUT / "attribution"
ADAPTERS = WS / "exp" / "adapters"
MODEL = os.environ.get("QWEN_1_5B_PATH", str(WS / "models" / "Qwen2.5-1.5B-Instruct"))
TAG = "1p5B_v2"
N_TOP = 24


def select_items():
    """受损组 D 与源内对照组 K，保存到 items.json 供所有 part 复用。"""
    p = ATT / "items.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))
    rows = [json.loads(l) for l in open(OUT / "damage_scores.jsonl", encoding="utf-8")]
    sel = {}
    for src in ["same_object", "hop2"]:
        rs = sorted([r for r in rows if r["source"] == src], key=lambda r: -r["d_G2"])
        d = rs[:N_TOP]
        k = rs[-N_TOP:]
        sel[f"D_{src}"] = [r["item_key"] for r in d]
        sel[f"K_{src}"] = [r["item_key"] for r in k]
    sel["D"] = sel["D_same_object"] + sel["D_hop2"]
    sel["K"] = sel["K_same_object"] + sel["K_hop2"]
    ATT.mkdir(parents=True, exist_ok=True)
    json.dump(sel, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 顺带存题目文本，阶段3 复用
    qa = {r["item_key"]: {"q": r["question"], "a": r["answer"], "d_G2": r["d_G2"]}
          for r in rows if r["item_key"] in set(sel["D"] + sel["K"])}
    json.dump(qa, open(ATT / "items_qa.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return sel


def load_pair(group, tok, eager=False):
    kw = {"attn_implementation": "eager"} if eager else {}
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", **kw).eval()
    if group == "orig":
        return m, None
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", **kw).eval()
    mp = PeftModel.from_pretrained(base, str(ADAPTERS / f"{TAG}_{group}")).eval()
    return m, mp


def qa_ids(tok, q, a):
    return tok(q + " " + a, return_tensors="pt").to("cuda")


# ---------- drift ----------
def part_drift(group, items, qa):
    m, mp = load_pair(group, None)
    tok = AutoTokenizer.from_pretrained(MODEL)
    D = {k: [] for k in range(29)}
    K = {k: [] for k in range(29)}
    for label, keys in [("D", items["D"]), ("K", items["K"])]:
        for k in keys:
            q, a = qa[k]["q"], qa[k]["a"]
            enc = qa_ids(tok, q, a)
            h0 = m(**enc, output_hidden_states=True).hidden_states
            h1 = mp(**enc, output_hidden_states=True).hidden_states
            for l in range(29):
                v0 = h0[l][0, -1].float()
                v1 = h1[l][0, -1].float()
                cos = torch.dot(v0, v1) / (v0.norm() * v1.norm() + 1e-8)
                (D if label == "D" else K)[l].append(1 - float(cos))
    res = {"D": [float(np.mean(D[l])) for l in range(29)],
           "K": [float(np.mean(K[l])) for l in range(29)]}
    json.dump(res, open(ATT / f"drift_{group}.json", "w", encoding="utf-8"), indent=1)
    np.savez(ATT / f"drift_items_{group}.npz",
             D=np.array([D[l] for l in range(29)]).T,   # (n_item, 29)
             K=np.array([K[l] for l in range(29)]).T)
    print(f"[drift {group}] D 末层漂移 {res['D'][-1]:.4f}，K {res['K'][-1]:.4f}")


# ---------- heads ----------
def part_heads(group, items, qa):
    m, mp = load_pair(group, None, eager=True)  # eager 才能拿到注意力矩阵
    tok = AutoTokenizer.from_pretrained(MODEL)
    NL = m.config.num_hidden_layers
    NH = m.config.num_attention_heads
    dsum = np.zeros((NL, NH)); ksum = np.zeros((NL, NH))
    for label, keys in [("D", items["D"]), ("K", items["K"])]:
        for k in keys:
            enc = qa_ids(tok, qa[k]["q"], qa[k]["a"])
            with torch.no_grad():
                a1 = mp(**enc, output_attentions=True).attentions  # (L,)[1,H,s,s]
            for l in range(NL):
                att = a1[l][0, :, -1, :].float().mean(-1)  # 末 token 对前文，逐头平均
                v = att.cpu().numpy()
                if label == "D":
                    dsum[l] += v
                else:
                    ksum[l] += v
    n = len(items["D"])
    diff = dsum / n - ksum / n
    np.save(ATT / f"head_diff_{group}.npy", diff)
    json.dump({"NL": NL, "NH": NH}, open(ATT / f"head_meta_{group}.json", "w"))
    top = np.dstack(np.unravel_index(np.argsort(-np.abs(diff), axis=None), diff.shape))[0][:15]
    print(f"[heads {group}] |diff| 最大的 5 个头 (layer,head):",
          [(int(l), int(h), round(float(diff[l, h]), 4)) for l, h in top[:5]])


# ---------- ffn ----------
def part_ffn_v2(group, items, qa):
    """FFN 中间激活（down_proj 输入，即 act(gate)*up），M vs M′ 绝对漂移。"""
    m, mp = load_pair(group, None)
    tok = AutoTokenizer.from_pretrained(MODEL)
    NL = m.config.num_hidden_layers
    dsum = np.zeros(NL, dtype=object)
    ksum = np.zeros(NL, dtype=object)
    dsum[:] = None; ksum[:] = None

    def run(model, enc):
        acts = [None] * NL
        def mk(l):
            def hook(mod, inp, out):
                acts[l] = inp[0][0, -1].float().detach()
            return hook
        core = model.model
        if not hasattr(core, "layers"):  # PeftModel 多套一层
            core = core.model
        hs = [core.layers[l].mlp.down_proj.register_forward_hook(mk(l)) for l in range(NL)]
        try:
            with torch.no_grad():
                model(**enc)
        finally:
            for h in hs:
                h.remove()
        return acts

    for label, keys in [("D", items["D"]), ("K", items["K"])]:
        acc = dsum if label == "D" else ksum
        for k in keys:
            enc = qa_ids(tok, qa[k]["q"], qa[k]["a"])
            a1 = run(mp, enc)
            a0 = run(m, enc)
            for l in range(NL):
                d = (a1[l] - a0[l]).abs().cpu().numpy()
                acc[l] = d if acc[l] is None else acc[l] + d
    n = len(items["D"])
    per_layer_mean = []
    topk = []
    for l in range(NL):
        dmean = dsum[l] / n
        kmean = ksum[l] / n
        per_layer_mean.append({"layer": l,
                               "D": float(dmean.mean()), "K": float(kmean.mean())})
        diff = dmean - kmean
        idx = np.argsort(-diff)[:20]
        for i in idx:
            topk.append({"layer": l, "neuron": int(i), "diff": float(diff[i]),
                         "D": float(dmean[i]), "K": float(kmean[i])})
    json.dump({"per_layer": per_layer_mean, "topk": topk},
              open(ATT / f"ffn_{group}.json", "w", encoding="utf-8"), indent=1)
    best = sorted(topk, key=lambda r: -r["diff"])[:5]
    print(f"[ffn {group}] top-5 神经元:", [(r["layer"], r["neuron"], round(r["diff"], 3)) for r in best])


# ---------- grad（原模型 M 上的 Fisher 代理）----------
def part_grad(group, items, qa):
    m, _ = load_pair("orig", None)
    tok = AutoTokenizer.from_pretrained(MODEL)
    NL = m.config.num_hidden_layers
    for l in range(NL):
        for p in m.model.layers[l].parameters():
            p.requires_grad_(True)
    dnorm = np.zeros(NL); knorm = np.zeros(NL)
    for label, keys in [("D", items["D"]), ("K", items["K"])]:
        for k in keys:
            enc = qa_ids(tok, qa[k]["q"], qa[k]["a"])
            ids = enc["input_ids"]
            n = ids.shape[1]
            qi = tok(qa[k]["q"], add_special_tokens=True)["input_ids"]
            ctx = len(qi)
            out = m(input_ids=ids)
            lp = torch.log_softmax(out.logits.float(), -1)
            tgt = ids[0, 1:]
            v = lp[0, :n - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            mask = torch.arange(n - 1, device="cuda") >= ctx - 1
            loss = -(v * mask.float()).sum() / mask.float().sum()
            m.zero_grad(set_to_none=True)
            loss.backward()
            for l in range(NL):
                g2 = sum(float((p.grad ** 2).sum()) for p in m.model.layers[l].parameters()
                         if p.grad is not None)
                (dnorm if label == "D" else knorm)[l] += g2
    n = len(items["D"])
    res = {"D": (dnorm / n).tolist(), "K": (knorm / n).tolist()}
    json.dump(res, open(ATT / f"grad_{group}.json", "w", encoding="utf-8"), indent=1)
    print(f"[grad {group}] D 梯度范数峰值层 {int(np.argmax(dnorm))}，K 峰值层 {int(np.argmax(knorm))}")


PARTS = {"drift": part_drift, "heads": part_heads, "ffn": part_ffn_v2, "grad": part_grad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=list(PARTS))
    ap.add_argument("--models", nargs="+", default=["G2", "G1"])
    args = ap.parse_args()

    items = select_items()
    qa = json.load(open(ATT / "items_qa.json", encoding="utf-8"))
    ext = {"drift": "json", "heads": "npy", "ffn": "json", "grad": "json"}[args.part]
    for g in args.models:
        done = list(ATT.glob(f"{args.part}_{g}*.{ext}"))
        if done:
            print(f"[{args.part} {g}] 已有产物，跳过")
            continue
        print(f"=== {args.part} / {g} ===")
        PARTS[args.part](g, items, qa)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
