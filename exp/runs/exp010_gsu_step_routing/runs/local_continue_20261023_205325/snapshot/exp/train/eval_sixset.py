# -*- coding: utf-8 -*-
"""对任意既有 adapter 跑内容三六集评测（增量落盘 gsu_sixset.jsonl）
用法: python eval_sixset.py --group G1 --adapter 1p5B_v2_G1"""
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "damage"))
import component_ablation as ca
import train_eval as te

OUT = Path(__file__).resolve().parents[2] / "exp" / "results"
ADAPTERS = OUT.parent / "adapters"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--adapter", required=True)
    args = ap.parse_args()
    j6 = OUT / "gsu_sixset.jsonl"
    done = set()
    if j6.exists():
        for l in open(j6, encoding="utf-8"):
            done.add(json.loads(l)["group"])
    if args.group in done:
        print(f"{args.group}: 已有六集结果，跳过")
        return
    tok = AutoTokenizer.from_pretrained(te.MODEL)
    base = AutoModelForCausalLM.from_pretrained(te.MODEL, dtype=torch.bfloat16,
                                                device_map="cuda").eval()
    m = PeftModel.from_pretrained(base, str(ADAPTERS / args.adapter)).eval()
    sets, qa = ca.build_sets()
    res = ca.eval_sets(m, tok, sets, qa)
    with open(j6, "a", encoding="utf-8") as f:
        f.write(json.dumps({"group": args.group, "sets": res}, ensure_ascii=False) + "\n")
    print(f"{args.group} 6集:", {k: round(v["mean_correct"], 3) for k, v in res.items()})


if __name__ == "__main__":
    main()
