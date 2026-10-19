# -*- coding: utf-8 -*-
"""评测 orig 与 G2(遗忘后) 在内容三六评测台上的基线（供图 R3 对照）"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "damage"))
import component_ablation as ca
import eval_per_item as ep

WS = Path(__file__).resolve().parents[2]
RES = WS / "exp" / "results"
OUT = RES / "restore_baseline_orig_g2.json"

def main():
    tok = AutoTokenizer.from_pretrained(ep.MODEL)
    sets, qa = ca.build_sets()
    out = {}
    base = AutoModelForCausalLM.from_pretrained(ep.MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    out["orig"] = ca.eval_sets(base, tok, sets, qa)
    del base
    torch.cuda.empty_cache()
    from train_restore import load_g2_merged
    g2 = load_g2_merged()
    out["G2"] = ca.eval_sets(g2, tok, sets, qa)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for m, s in out.items():
        d = (s["D_so"]["mean_lp"] + s["D_hop2"]["mean_lp"]) / 2
        print(m, "D mean_lp=%.3f" % d, {k: round(s[k]["mean_correct"], 3) for k in s})

if __name__ == "__main__":
    main()
