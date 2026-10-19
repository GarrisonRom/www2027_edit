# -*- coding: utf-8 -*-
"""文献 baseline 移植：GradDiff / NPO / SimNPO / RMU
损失逐行核对自 code/open-unlearning/src/trainer/unlearn/*.py，
移植到本工作区 LoRA 训练循环（train_gsu.py 同款协议），保证同口径对比。
同遗忘集（F0+F1贪心5）、同 LoRA(r8a16)/lr/epochs、同双评测（内容一五集+内容三六集）。
ref 模型用 disable_adapter 技巧（= 冻结原模型，与仓库 _prepare_ref_model 等价）。
用法: python train_baselines.py --method GradDiff|NPO|SimNPO|RMU [--eval_only] [--eval_stage 5|6|none|both]
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

TE = Path(__file__).resolve().parent
sys.path.insert(0, str(TE))
sys.path.insert(0, str(TE.parent / "damage"))
import train_eval as te
import component_ablation as ca
from train_gsu import lp_masked

WS = Path(__file__).resolve().parents[2]
OUT = WS / "exp" / "results"
ADAPTERS = WS / "exp" / "adapters"
TAG = "1p5B_v2"
LR, EPOCHS = 5e-5, 3
GAMMA, ALPHA, BETA, DELTA, STEER = 1.0, 1.0, 1.0, 0.0, 20.0

CFG = {
    # 全方法统一 LoRA（RMU 除外，它按仓库定义只训 down_proj 5-7 层）
    "GradDiff": dict(lora=dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"])),
    "GA":       dict(lora=dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"])),
    "NPO":      dict(lora=dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"])),
    "SimNPO":   dict(lora=dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"])),
    "RMU":      dict(lora=dict(target_modules=["down_proj"], layers_to_transform=[5, 6, 7]),
                     module_layer=7),
    "CEU":      dict(lora=dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"])),
    "DPO":      dict(lora=dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"])),
    "PDU":      dict(lora=dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"])),
    "WGA":      dict(lora=dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"])),
    "SatImp":   dict(lora=dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"])),
    "UNDIAL":   dict(lora=dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"])),
}


def nll_sum(model, tok, q, a):
    """答案 token NLL 之和（仓库 compute_batch_nll 的等价，单序列版）"""
    full, ctx = te.enc_qa(tok, q, a)
    ids = torch.tensor([full]).to(model.device)
    labels = ids.clone()
    labels[:, :ctx] = -100
    out = model(input_ids=ids)
    logits = out.logits[..., :-1, :].float()
    tgt = ids[:, 1:]
    mask = (labels[:, 1:] != -100).float()
    nll = F.cross_entropy(logits.transpose(-1, -2), tgt, reduction="none")
    return (nll * mask).sum(), mask.sum()


def retain_nll(model, tok, rq, ra):
    """仓库 compute_retain_loss(retain_loss_type='NLL')：retain 的 CE（升 = 伤保持）"""
    full, ctx = te.enc_qa(tok, rq, ra)
    ids = torch.tensor([full]).to(model.device)
    labels = ids.clone()
    labels[:, :ctx] = -100
    out = model(input_ids=ids)
    logits = out.logits[..., :-1, :].float()
    tgt = ids[:, 1:]
    mask = (labels[:, 1:] != -100).float()
    nll = F.cross_entropy(logits.transpose(-1, -2), tgt, reduction="none")
    return (nll * mask).sum() / mask.sum()


def per_token_ce(model, tok, q, a):
    """逐 token CE（带梯度），返回 (ce, mask, outputs, shifted_logits, shifted_tgt, ids)"""
    full, ctx = te.enc_qa(tok, q, a)
    ids = torch.tensor([full]).to(model.device)
    labels = ids.clone()
    labels[:, :ctx] = -100
    out = model(input_ids=ids)
    logits = out.logits[..., :-1, :].float()
    tgt = ids[:, 1:]
    mask = (labels[:, 1:] != -100)
    ce = F.cross_entropy(logits.transpose(-1, -2), tgt, reduction="none")
    return ce, mask, out, logits, tgt, ids


def compute_loss(method, model, tok, q, a, rq, ra, state):
    if method in ("GradDiff", "NPO", "SimNPO", "RMU", "DPO", "PDU", "WGA", "SatImp", "UNDIAL"):
        ret = retain_nll(model, tok, rq, ra)
    if method == "GA":
        # 仓库 grad_ascent.py: loss = -CE(forget)（HF Trainer 最小化）= 本框架 +lp 下降
        return lp_masked(model, tok, q, a)
    if method == "GradDiff":
        # 仓库 grad_diff.py: loss = gamma*(-CE(forget)) + alpha*CE(retain)
        # 本框架 lp = -CE（G-recipe 遗忘项 = +lp 下降），等价改写：gamma*lp_masked + alpha*retain_NLL
        return GAMMA * lp_masked(model, tok, q, a) + ALPHA * ret
    if method == "NPO":
        # 仓库 npo.py + utils.compute_dpo_loss(win=None, lose=forget):
        #   lose_log_ratio = -(nll_M - nll_ref);  loss_forget = -2/β·logσ(β·(0 - lose_log_ratio))
        nll_m, n = nll_sum(model, tok, q, a)
        with model.disable_adapter(), torch.no_grad():
            nll_ref, _ = nll_sum(model, tok, q, a)
        lose_log_ratio = -(nll_m - nll_ref)
        forget_loss = -2 / BETA * F.logsigmoid(-BETA * lose_log_ratio)
        return GAMMA * forget_loss + ALPHA * ret
    if method == "SimNPO":
        # 仓库 simnpo.py: forget = -2/β·logσ(β·(nll_M/n_tok - δ))
        nll_m, n = nll_sum(model, tok, q, a)
        forget_loss = -2 / BETA * F.logsigmoid(BETA * (nll_m / n - DELTA))
        return GAMMA * forget_loss + ALPHA * ret
    if method == "RMU":
        # 仓库 rmu.py: forget = 掩码MSE(层7输出, 随机单位向量×steering_coeff)
        cache = []
        def hook(mod, inp, out):
            cache.append(out[0] if isinstance(out, tuple) else out)
        full, ctx = te.enc_qa(tok, q, a)
        ids = torch.tensor([full]).to(model.device)
        layers = [m for n_, m in model.named_modules()
                  if re.fullmatch(rf".*layers\.{CFG['RMU']['module_layer']}$", n_)]
        h = layers[0].register_forward_hook(hook)
        out = model(input_ids=ids)
        h.remove()
        act = cache[0]                                   # [1, s, d]
        if state.get("ctrl") is None:
            g = torch.Generator().manual_seed(te.SEED)
            rv = torch.rand(1, 1, act.shape[-1], generator=g)
            state["ctrl"] = (rv / rv.norm() * STEER).to(act.dtype).to(act.device)
        ctrl = state["ctrl"].expand_as(act)
        labels = ids.clone()
        labels[:, :ctx] = -100
        mask = (labels != -100).unsqueeze(-1).float()    # [1, s, 1]
        mse = ((act - ctrl) ** 2 * mask).mean(dim=2).sum(dim=1) / mask.sum(dim=(1, 2))
        return GAMMA * mse.mean() + ALPHA * ret
    if method == "CEU":
        # 仓库 ceu.py: 目标分布 = softmax(答案 token 位置 logit 置 -inf)，CE 向目标看齐
        # footnote 技巧：忽略前 1 个答案 token
        ce, mask, out, logits, tgt, ids = per_token_ce(model, tok, q, a)
        valid = mask.clone()
        if int(mask.sum()) >= 2:  # 单 token 答案跳过 footnote 技巧，避免空集 NaN
            v = valid[0]
            for i in range(len(v)):
                if v[i]:
                    v[i] = False
                    break
        vl = logits[0][valid[0]]                      # [n, V]
        vt = tgt[0][valid[0]]                         # [n]
        tgt_logits = vl.detach().clone()
        tgt_logits.scatter_(1, vt.unsqueeze(-1), float("-inf"))
        tgt_prob = F.softmax(tgt_logits, dim=-1)
        return F.cross_entropy(vl, tgt_prob)
    if method == "DPO":
        # 仓库 dpo.py: win=alternate 问法, lose=原问法；log_ratio = -(nll_M - nll_ref)
        q_alt, a_alt = state["alt_fn"](q, a)
        nll_lose, _ = nll_sum(model, tok, q, a)
        nll_win, _ = nll_sum(model, tok, q_alt, a_alt)
        with model.disable_adapter(), torch.no_grad():
            nll_lose_ref, _ = nll_sum(model, tok, q, a)
            nll_win_ref, _ = nll_sum(model, tok, q_alt, a_alt)
        win_ratio = -(nll_win - nll_win_ref)
        lose_ratio = -(nll_lose - nll_lose_ref)
        return GAMMA * (-2 / BETA * F.logsigmoid(BETA * (win_ratio - lose_ratio))) + ALPHA * ret
    if method == "PDU":
        # 仓库 pdu.py（primal_dual=False）: forget = 掩码均值 (maxLogit - meanLogit)^2
        ce, mask, out, logits, tgt, ids = per_token_ce(model, tok, q, a)
        lg = logits[0]
        fl = ((lg.max(dim=-1).values - lg.mean(dim=-1)) ** 2)
        return GAMMA * (fl * mask[0]).sum() / mask[0].sum() + ALPHA * ret
    if method == "WGA":
        # 仓库 utils.compute_wga_loss: weight = exp(-ce)^β，遗忘加权上升
        ce, mask, out, logits, tgt, ids = per_token_ce(model, tok, q, a)
        w = ((-ce).exp().detach()) ** BETA
        return GAMMA * (-(w * ce)[mask].mean()) + ALPHA * ret
    if method == "SatImp":
        # 仓库 utils.compute_satimp_loss: β1=5 饱和权 × β2=1 重要权，γ=1 α=0.1
        ce, mask, out, logits, tgt, ids = per_token_ce(model, tok, q, a)
        w_sat = ((-ce).exp().detach()) ** 5.0
        w_imp = (1 - (-ce).exp().detach()) ** 1.0
        return 1.0 * (-((w_sat * w_imp) * ce)[mask].mean()) + 0.1 * ret
    if method == "UNDIAL":
        # 仓库 utils.compute_undial_loss: soft_label = softmax(ref_logits − β·正确token掩码)
        ce, mask, out, logits, tgt, ids = per_token_ce(model, tok, q, a)
        with model.disable_adapter(), torch.no_grad():
            ref_logits = model(input_ids=ids).logits[..., :-1, :].float()
        m = torch.zeros_like(ref_logits)
        bidx = torch.arange(m.shape[0]).view(-1, 1, 1)
        sidx = torch.arange(m.shape[1]).view(1, -1, 1)
        m[bidx.to(ref_logits.device), sidx.to(ref_logits.device), tgt.unsqueeze(-1)] = 1.0
        soft = F.softmax(ref_logits - m * BETA, dim=-1)
        return GAMMA * F.cross_entropy(
            logits.transpose(-1, -2), soft.transpose(-1, -2), reduction="mean") + ALPHA * ret
    raise ValueError(method)


def train(method, forget_ex, retain_ex, tok, base, start_ep=0, alt_map=None):
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        **CFG[method]["lora"]))
    if start_ep > 0:
        sd = torch.load(ADAPTERS / f"{TAG}_BL{method}_ckpt" / f"ep{start_ep}.pt", map_location="cuda")
        model.load_state_dict({**model.state_dict(), **sd}, strict=False)
        print(f"  [{method}] 恢复 epoch {start_ep}")
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    rng = random.Random(te.SEED)
    state = {}
    if alt_map is not None:
        state["alt_fn"] = lambda q, a: alt_map.get((q, a), (q, a))
    for ep in range(start_ep, EPOCHS):
        random.Random(te.SEED + ep).shuffle(forget_ex)
        tot = 0.0
        for q, a in forget_ex:
            rq, ra = rng.choice(retain_ex)
            opt.zero_grad()
            loss = compute_loss(method, model, tok, q, a, rq, ra, state)
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"  [{method}] ep{ep+1}/{EPOCHS} loss={tot/len(forget_ex):.4f}", flush=True)
        ck = ADAPTERS / f"{TAG}_BL{method}_ckpt"
        ck.mkdir(exist_ok=True)
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if "lora" in n},
                   ck / f"ep{ep+1}.pt")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(CFG))
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--eval_stage", default="both", choices=["5", "6", "none", "both"])
    args = ap.parse_args()
    method = args.method
    out_adapter = ADAPTERS / f"{TAG}_BL{method}"
    ckpt = ADAPTERS / f"{TAG}_BL{method}_ckpt"
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
    forget_ex = te.make_group_examples("G2", clusters, forget_idx, f1, cands)

    alt_map = None
    if method == "DPO":
        # alternate 问法：F0 用同簇 hop1_test 改写问题（同答案），F1 用模板第二句式
        sys.path.insert(0, str(TE.parent / "probe"))
        from build_f1 import TEMPLATES, FALLBACK
        alt_map = {}
        for idx in forget_idx:
            c = clusters[idx][2]
            q0 = c["hop1_unlearn"]["questions"][0]
            a = c["hop1_unlearn"]["answers"]
            qt = c["hop1_test"]["questions"]
            if qt:
                alt_map[(q0, a)] = (qt[0], a)
        for f in f1:
            tpls = TEMPLATES.get(f["relation"], FALLBACK)
            lose_q = te.TPL1.get(f["relation"], "What is the {r} of {s}?").format(
                s=f["subject"], r=f["relation"])
            alt_q = tpls[1].format(s=f["subject"], r=f["relation"]) if len(tpls) > 1 else lose_q
            alt_map[(lose_q, f["object"])] = (alt_q, f["object"])
        print(f"[DPO] alternate 映射 {len(alt_map)} 条")

    if done_epochs < EPOCHS and not args.eval_only:
        base = AutoModelForCausalLM.from_pretrained(te.MODEL, dtype=torch.bfloat16,
                                                    device_map="cuda").eval()
        for p in base.parameters():
            p.requires_grad = False
        m = train(method, forget_ex, retain_ex, tok, base, start_ep=done_epochs, alt_map=alt_map)
        m.save_pretrained(out_adapter)
        del m, base
        torch.cuda.empty_cache()
        print("saved:", out_adapter)

    name = f"BL{method}"
    j5 = OUT / "eval_results_gsu.jsonl"
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
    j6 = OUT / "gsu_sixset.jsonl"
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
