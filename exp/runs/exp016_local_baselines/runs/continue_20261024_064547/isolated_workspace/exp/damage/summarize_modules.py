# -*- coding: utf-8 -*-
"""模块分解汇总：损伤存在哪类矩阵里"""
import json
from pathlib import Path

BASE = {"D_so": (-11.828, -4.720), "D_hop2": (-9.579, -3.423),
        "hop1_unlearn": (-9.899, -3.576), "F1_hop2": (-7.521, -3.156), "K": (-5.761, -3.761)}
recs = {json.loads(l)["config"]: json.loads(l)["sets"]
        for l in open(Path(__file__).resolve().parents[2] / "exp" / "results" / "attribution" / "ablation_G2.jsonl",
                      encoding="utf-8")}


def dfrac(s):
    g2 = (BASE["D_so"][0] + BASE["D_hop2"][0]) / 2
    orig = (BASE["D_so"][1] + BASE["D_hop2"][1]) / 2
    v = (s["D_so"]["mean_lp"] + s["D_hop2"]["mean_lp"]) / 2
    return 100 * (v - g2) / (orig - g2)


def frac(s, k):
    return 100 * (s[k]["mean_lp"] - BASE[k][0]) / (BASE[k][1] - BASE[k][0])


print(f"{'配置':<26}{'D恢复%':>8}{'hop1回弹%':>10}{'F1复活%':>9}")
for c in ["undo_mlp_22_27", "undo_attn_22_27", "undo_attn_mlp_22_27",
          "undo_attn_all", "undo_mlp_all", "undo_all",
          "undo_neurons_top12", "undo_mlp_rand6"]:
    if c not in recs:
        print(c, "缺失")
        continue
    s = recs[c]
    print(f"{c:<26}{dfrac(s):>8.1f}{frac(s, 'hop1_unlearn'):>10.1f}{frac(s, 'F1_hop2'):>9.1f}")

s = recs["undo_all"]
print("\n一致性校验 undo_all vs orig 基准：")
for k, bk in [("D_so", "D_so"), ("hop1_unlearn", "hop1"), ("F1_hop2", "F1_hop2")]:
    print(f"  {k}: undo_all={s[k]['mean_lp']:.3f}  orig={BASE[bk][1]:.3f}  差={s[k]['mean_lp'] - BASE[bk][1]:+.4f}")
