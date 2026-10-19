# -*- coding: utf-8 -*-
"""回弹失败归因 · 分析B：R1 adapter 的更新效果对齐（GPU，单次加载）"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "damage"))
import component_ablation as ca
import eval_per_item as ep
from train_restore import load_g2_merged

REC = Path(__file__).resolve().parents[1] / "restore"
RES = REC.parent / "results"

sup = json.load(open(REC / "supervision.json", encoding="utf-8"))
anc = json.load(open(REC / "anchors.json", encoding="utf-8"))
f0 = [a for a in anc if a["kind"] == "F0"]
sup_h2 = [r for r in sup if r["source"] == "hop2"][:12]
f0s = f0[:12]

from peft import PeftModel
tok = AutoTokenizer.from_pretrained(ep.MODEL)
g2 = load_g2_merged()
model = PeftModel.from_pretrained(g2, str(RES.parent / "adapters" / "restore_R1_rec")).eval()

acts = {}
def mk(name):
    def hook(mod, inp, out):
        acts.setdefault(name, []).append(inp[0].detach().float().mean(1))  # [1, d]
    return hook

hooks = []
modsd = dict(model.named_modules())
layers_mod = None
for k, v in modsd.items():
    if k.endswith("model.layers") or k == "model.model.layers":
        layers_mod = v
        break
for l in range(22, 28):
    hooks.append(layers_mod[l].mlp.gate_proj.register_forward_hook(mk(f"raw{l}")))
with torch.no_grad():
    for r in sup_h2:
        for l in range(22, 28):
            acts.pop(f"raw{l}", None)  # 只清本次的钩子键
        enc = tok(r["question"] + " " + r["answer"], return_tensors="pt").to("cuda")
        model(**enc)
        for l in range(22, 28):
            acts.setdefault(f"raw{l}_D", []).append(acts.pop(f"raw{l}")[0])
    for a in f0s:
        for l in range(22, 28):
            acts.pop(f"raw{l}", None)
        enc = tok(a["q"] + " " + a["a"], return_tensors="pt").to("cuda")
        model(**enc)
        for l in range(22, 28):
            acts.setdefault(f"raw{l}_F", []).append(acts.pop(f"raw{l}")[0])

print("层   cos(原始激活_D, 原始激活_F0)   共享子空间证据")
cos_raw = []
for l in range(22, 28):
    D = torch.stack(acts[f"raw{l}_D"]).mean(0).flatten()
    F = torch.stack(acts[f"raw{l}_F"]).mean(0).flatten()
    c = torch.nn.functional.cosine_similarity(D, F, dim=0).item()
    cos_raw.append(c)
    print(f"{l}   {c:+.3f}")

# ΔW = B@A 作用效果：对激活 x，ΔW x = B(Ax)。逐层算 D 样本与 F0 样本的 cos(ΔWx_D, ΔWx_F)
print("\n层   cos(ΔWx_D, ΔWx_F0)  （ΔW = LoRA B@A，gate_proj）")
mods = dict(model.named_modules())
with torch.no_grad():
    for l in range(22, 28):
        patA = [k for k in mods if k.endswith(f"layers.{l}.mlp.gate_proj.lora_A.default")]
        patB = [k for k in mods if k.endswith(f"layers.{l}.mlp.gate_proj.lora_B.default")]
        A = mods[patA[0]].weight.float()
        B = mods[patB[0]].weight.float()
        ds, fs = [], []
        for x in acts[f"raw{l}_D"]:
            d = B @ (A @ x.flatten())
            ds.append(d)
        for x in acts[f"raw{l}_F"]:
            f = B @ (A @ x.flatten())
            fs.append(f)
        dM, fM = torch.stack(ds).mean(0), torch.stack(fs).mean(0)
        c = torch.nn.functional.cosine_similarity(dM, fM, dim=0).item()
        nd = dM.norm().item(); nf = fM.norm().item()
        print(f"{l}   {c:+.3f}   (|ΔWx_D|={nd:.2f}, |ΔWx_F0|={nf:.2f}, 比值 F0/D={nf/nd:.2f})")

json.dump({"layers": list(range(22, 28)), "cos_raw": cos_raw},
          open(RES / "rebound_analysis_B.json", "w", encoding="utf-8"), indent=1)
