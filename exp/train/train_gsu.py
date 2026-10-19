# -*- coding: utf-8 -*-
"""LGSU：定位梯度手术遗忘（Localized Gradient-Surgery Unlearning）
基于内容一/二/三全部证据设计的新遗忘方法，与 G1/G2/G6 同口径对比。

设计（每个组件都有前序实验证据）：
1. 遗忘集 = G2 同款（F0 + F1 贪心5）——内容一：拓展遗忘集是多跳封堵的关键（hop2 0.36→0.25）
2. 定位：LoRA 只打在层 22–27 的 MLP（gate/up/down）——内容二：FFN 损伤带+显影得分定位
3. 手术保护：每步从遗忘梯度中减去保持集锚点梯度方向的投影 g_f ← g_f − proj_{g_r}(g_f)
   ——内容三 R5 的镜像（恢复侧手术换成遗忘侧手术）；G6 已证 KL 锚加强 12× 反而更差
   （retain 0.333<0.394, same_object 0.160<0.257），分布级阻尼有结构性盲区

变体：
  M1 LGSU       = 定位(22–27 MLP) + 手术保护          ← 主方法
  M3 GSU-full   = 全层全模块(G-recipe同款) + 手术保护   ← 消融：定位的贡献
  M3C           = M3 的条件梯度投影
  M3U           = 按 AdamW 实际参数位移做条件投影
断点：逐 epoch 存 checkpoint，重跑续训。
M3C/M3U 默认写入独立的 gsu_projection 结果文件。
用法: python train_gsu.py --variant M1|M2|M3|M3C|M3U [--eval_only]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

TE = Path(__file__).resolve().parent
sys.path.insert(0, str(TE))
sys.path.insert(0, str(TE.parent / "damage"))
import train_eval as te
import component_ablation as ca

WS = Path(__file__).resolve().parents[2]
OUT = WS / "exp" / "results"
ADAPTERS = WS / "exp" / "adapters"
TAG = "1p5B_v2"
LR, EPOCHS, N_ANCHOR = 5e-5, 3, 4

VARIANTS = {
    "M1": dict(layers=list(range(22, 28)), modules=["gate_proj", "up_proj", "down_proj"],
               alpha=1.0, projection="always"),
    "M2": dict(layers=None, modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
               alpha=0.5, projection="always"),
    "M3": dict(layers=None, modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
               alpha=1.0, projection="always"),
    "M3C": dict(layers=None, modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                alpha=1.0, projection="conditional"),
    "M3U": dict(layers=None, modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                alpha=1.0, projection="adamw_update"),
}


def lp_masked(model, tok, q, a):
    """与 G-recipe 同口径：答案 token 平均对数概率（带梯度）"""
    full, ctx = te.enc_qa(tok, q, a)
    ids = torch.tensor([full]).to(model.device)
    labels = ids.clone()
    labels[:, :ctx] = -100
    out = model(input_ids=ids)
    lp = torch.log_softmax(out.logits.float(), -1)
    tgt = ids[:, 1:]
    mask = (labels[:, 1:] != -100).float()
    if mask.sum() == 0:
        return None
    return (lp[:, :ids.shape[1] - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask).sum() / mask.sum()


def adamw_step_with_anchor_projection(opt, params, anchor_grads):
    """Project AdamW's actual parameter update when its first-order anchor change is negative."""
    before = [p.detach().clone() for p in params]
    opt.step()
    corrected = False
    with torch.no_grad():
        dot = torch.zeros((), device=params[0].device, dtype=torch.float32)
        denom = torch.zeros_like(dot)
        for p, old in zip(params, before):
            ga = anchor_grads[p].float()
            delta = p.detach().float() - old.float()
            dot += torch.sum(ga * delta)
            denom += torch.sum(ga * ga)
        if float(dot) < 0.0 and float(denom) > 1e-12:
            coeff = float(dot / denom)
            for p in params:
                p.add_(anchor_grads[p], alpha=-coeff)
            corrected = True
    return corrected


def project_gradient_if_conflicting(grad, anchor_grad):
    dot = float((grad * anchor_grad).sum())
    if dot <= 0.0:
        return grad, False
    denom = float((anchor_grad * anchor_grad).sum()) + 1e-12
    return grad - dot / denom * anchor_grad, True


def train(variant, forget_ex, retain_ex, tok, base, start_ep=0):
    spec = VARIANTS[variant]
    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=spec["modules"],
                      layers_to_transform=spec["layers"])
    model = get_peft_model(base, lcfg)
    if start_ep > 0:
        ck = ADAPTERS / f"{TAG}_GSU{variant}_ckpt"
        sd = torch.load(ck / f"ep{start_ep}.pt", map_location="cuda")
        model.load_state_dict({**model.state_dict(), **sd}, strict=False)
        print(f"  [{variant}] 恢复 epoch {start_ep}")
    model.train()
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    rng = random.Random(te.SEED)
    for ep in range(start_ep, EPOCHS):
        random.Random(te.SEED + ep).shuffle(forget_ex)
        tot = 0.0
        projection_events = projection_checks = 0
        mode = spec["projection"]
        for q, a in forget_ex:
            # 反传1：遗忘损失（答案 lp 下降 = 遗忘）
            opt.zero_grad()
            loss_f = lp_masked(model, tok, q, a)
            if loss_f is None:
                continue
            loss_f.backward()
            g_f = {p: p.grad.clone() for p in params}
            # 反传2：保持集锚点上升方向 g_r（4 条）
            opt.zero_grad()
            anchors = rng.sample(retain_ex, N_ANCHOR)
            lps = []
            for rq, ra in anchors:
                lp = lp_masked(model, tok, rq, ra)
                if lp is not None:
                    lps.append(lp)
            torch.stack(lps).sum().backward()
            g_r = {p: p.grad.clone() for p in params}
            if mode != "adamw_update":
                alpha = spec["alpha"]
                for p in params:
                    g = g_f[p]
                    ga = g_r[p]
                    if mode == "conditional":
                        projection_checks += 1
                        g, projected = project_gradient_if_conflicting(g, ga)
                        projection_events += int(projected)
                    else:
                        dot = float((g * ga).sum())
                        denom = float((ga * ga).sum()) + 1e-12
                        g = g - alpha * dot / denom * ga
                    p.grad = g
            else:
                for p in params:
                    p.grad = g_f[p]

            if mode == "adamw_update":
                projection_checks += 1
                projection_events += int(adamw_step_with_anchor_projection(opt, params, g_r))
            else:
                opt.step()
            tot += float(loss_f)
        projection_log = (f" projection={projection_events}/{projection_checks}"
                          if mode in ("conditional", "adamw_update") else "")
        print(f"  [{variant}] ep{ep+1}/{EPOCHS} forget_loss={tot/len(forget_ex):.4f}"
              f"{projection_log}", flush=True)
        ck = ADAPTERS / f"{TAG}_GSU{variant}_ckpt"
        ck.mkdir(exist_ok=True)
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if "lora" in n},
                   ck / f"ep{ep+1}.pt")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--eval_stage", default="both", choices=["5", "6", "none", "both"])
    ap.add_argument("--result_tag", default=None,
                    help="结果文件名前缀；使用新标签可隔离新评测与历史 JSONL")
    args = ap.parse_args()
    args.result_tag = args.result_tag or (
        "gsu_projection" if args.variant in ("M3C", "M3U") else "gsu"
    )
    name = f"GSU{args.variant}"
    out_adapter = ADAPTERS / f"{TAG}_{name}"
    ckpt = ADAPTERS / f"{TAG}_{name}_ckpt"
    done_epochs = len(list(ckpt.glob("ep*.pt"))) if ckpt.exists() else 0
    if out_adapter.exists():
        done_epochs = EPOCHS

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
    forget_ex = te.make_group_examples("G2", clusters, forget_idx, f1, cands)  # = G2 遗忘集

    if done_epochs < EPOCHS and not args.eval_only:
        base = AutoModelForCausalLM.from_pretrained(te.MODEL, dtype=torch.bfloat16,
                                                    device_map="cuda").eval()
        for p in base.parameters():
            p.requires_grad = False
        m = train(args.variant, forget_ex, retain_ex, tok, base, start_ep=done_epochs)
        m.save_pretrained(out_adapter)
        del m, base
        torch.cuda.empty_cache()
        print("saved:", out_adapter)

    # ---- 评测1：内容一五集（同 te.evaluate 口径，增量落盘）----
    j5 = OUT / f"eval_results_{args.result_tag}.jsonl"
    done5 = set()
    if j5.exists():
        for l in open(j5, encoding="utf-8"):
            done5.add(json.loads(l)["group"])
    if name not in done5 and args.eval_stage in ("5", "both"):
        base = AutoModelForCausalLM.from_pretrained(te.MODEL, dtype=torch.bfloat16,
                                                    device_map="cuda").eval()
        m = PeftModel.from_pretrained(base, str(out_adapter)).eval()
        r = te.evaluate(m, tok, clusters, forget_idx, retain_idx)
        with open(j5, "a", encoding="utf-8") as f:
            f.write(json.dumps({"group": name, **r}, ensure_ascii=False) + "\n")
        print(f"{name} 5集:", {k: round(v, 3) for k, v in r.items()})
        del m, base
        torch.cuda.empty_cache()

    # ---- 评测2：内容三六集（同 restore 口径）----
    j6 = OUT / f"{args.result_tag}_sixset.jsonl"
    done6 = set()
    if j6.exists():
        for l in open(j6, encoding="utf-8"):
            done6.add(json.loads(l)["group"])
    if name not in done6 and args.eval_stage in ("6", "both"):
        base = AutoModelForCausalLM.from_pretrained(te.MODEL, dtype=torch.bfloat16,
                                                    device_map="cuda").eval()
        m = PeftModel.from_pretrained(base, str(out_adapter)).eval()
        sets, qa = ca.build_sets()
        res = ca.eval_sets(m, tok, sets, qa)
        with open(j6, "a", encoding="utf-8") as f:
            f.write(json.dumps({"group": name, "sets": res}, ensure_ascii=False) + "\n")
        print(f"{name} 6集:", {k: round(v["mean_correct"], 3) for k, v in res.items()})
        del m, base
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
