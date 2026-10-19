# -*- coding: utf-8 -*-
"""回弹失败归因 · 分析D：梯度冲突探测——恢复梯度与F0梯度是否可分离（决定R5正交化是否可行）"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "damage"))
import eval_per_item as ep
from train_restore import load_g2_merged, answer_lp_t

REC = Path(__file__).resolve().parents[1] / "restore"
RES = REC.parent / "results"

sup = json.load(open(REC / "supervision.json", encoding="utf-8"))
anc = json.load(open(REC / "anchors.json", encoding="utf-8"))
f0 = [a for a in anc if a["kind"] == "F0"]
sup_h2 = [r for r in sup if r["source"] == "hop2"][:8]
f0s = f0[:8]

from peft import PeftModel, LoraConfig, get_peft_model
tok = AutoTokenizer.from_pretrained(ep.MODEL)
g2 = load_g2_merged()
lcfg = LoraConfig(r=8, lora_alpha=16, target_modules=["gate_proj", "up_proj", "down_proj"],
                  layers_to_transform=list(range(22, 28)))
model = get_peft_model(g2, lcfg).eval()
params = [p for n, p in model.named_parameters() if "lora" in n]

def grad_of(items, sign):
    model.zero_grad(set_to_none=True)
    tot = 0
    for it in items:
        q, a = (it["question"], it["answer"]) if "question" in it else (it["q"], it["a"])
        lp = answer_lp_t(model, tok, q, a)
        tot = tot + sign * lp
    tot.backward()
    return torch.cat([p.grad.flatten().float() for p in params])

gD = grad_of(sup_h2, +1)     # 恢复：提升 D 答案 lp
gF = grad_of(f0s, +1)        # 回弹：提升 F0 答案 lp（与"不许回弹"反向）
cos = torch.nn.functional.cosine_similarity(gD, gF, dim=0).item()
print(f"cos(恢复梯度, F0梯度) = {cos:+.3f}  （全部LoRA参数, 层22-27）")
print(f"|gD|={gD.norm():.2f}  |gF|={gF.norm():.2f}")
frac = cos * gF.norm() / gD.norm()
print(f"若做梯度手术（从 gD 中减去 gF 方向）：剩余恢复信号 ≈ sqrt(1-cos²)×|gD| = "
      f"{(1-cos**2)**0.5 * 100:.0f}%")
json.dump({"cos_grad": cos}, open(RES / "rebound_analysis_D.json", "w", encoding="utf-8"), indent=1)
