# -*- coding: utf-8 -*-
"""回弹失败归因 · 分析C：能量空间约束饱和度（纯 CPU）
度量各 R3 adapter 的 lora_A 有多少能量落在能量空间 U 内：sat = ||A U||^2_F / ||A||^2_F"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

WS = Path(__file__).resolve().parents[2]
ADAPTERS = WS / "exp" / "adapters"
Uz = np.load(WS / "exp" / "restore" / "energy_space.npz")

configs = ["R1_rec", "R3_l0.1_e90", "R3_l1_e90", "R3_l10_e90", "R3_l10_e80", "R3_l10_e95"]
print(f"{'配置':<14}{'饱和度η=80':>10}{'饱和度η=90':>10}{'饱和度η=95':>10}")
res = {}
for c in configs:
    sd = load_file(str(ADAPTERS / f"restore_{c}" / "adapter_model.safetensors"))
    sats = {}
    for eta in (80, 90, 95):
        num, den = 0.0, 0.0
        for l in range(28):
            for nm in ("gate_proj", "up_proj", "down_proj"):
                A = sd[f"base_model.model.model.layers.{l}.mlp.{nm}.lora_A.weight"].float()
                U = torch.tensor(Uz[f"{'g' if nm != 'down_proj' else 'd'}{l}_U{eta}"]).float()
                num += ((A @ U) ** 2).sum().item()
                den += (A ** 2).sum().item()
        sats[eta] = num / den
    res[c] = sats
    print(f"{c:<14}{sats[80]:>10.3f}{sats[90]:>10.3f}{sats[95]:>10.3f}")

json.dump(res, open(WS / "exp" / "results" / "rebound_analysis_C.json", "w", encoding="utf-8"), indent=1)
print("\n解读：饱和度→0 说明 λ 继续增大只会进一步压扁 A（恢复率下降），约束已饱和，无新信息可用")
