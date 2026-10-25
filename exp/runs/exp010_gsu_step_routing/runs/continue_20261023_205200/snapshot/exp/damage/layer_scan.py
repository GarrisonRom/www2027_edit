# -*- coding: utf-8 -*-
"""
研究内容二 · 阶段4：全层单层撤销扫描（模型只加载一次，层权重 CPU 缓存往返）
每配置：undo 层 l 的 MLP → 评 D_so/D_hop2 → 从缓存恢复层 l → 下一层。
增量写 scan_mlp_G2.jsonl，可断点续跑。
"""
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import eval_per_item as ep

WS = Path(__file__).resolve().parents[2]
OUT = WS / "exp" / "results"
ATT = OUT / "attribution"
SPLITS = WS / "data" / "faithun" / "splits"
ADAPTERS = WS / "exp" / "adapters"
MODEL = ep.MODEL
TAG = "1p5B_v2"


@torch.no_grad()
def lp_mcqa(model, tok, q, a, falses):
    lp_a, best = None, None
    for i, c in enumerate([a] + list(falses)):
        enc = tok(q + " " + c, return_tensors="pt").to(model.device)
        n = enc["input_ids"].shape[1]
        lp = torch.log_softmax(model(**enc).logits.float(), -1)
        tgt = enc["input_ids"][0, 1:]
        v = lp[0, :n - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        s = float(v.sum()) / len(v)
        if i == 0:
            lp_a = s
        elif best is None or s > best:
            best = s
    return lp_a


def main():
    items = json.load(open(ATT / "items.json", encoding="utf-8"))
    data = json.load(open(ep.DATA, encoding="utf-8"))["data"]
    clusters = ep.build_clusters(data)
    forget_idx = [int(x) for x in (SPLITS / "forget_set_split.txt").read_text().split()]
    retain_idx = [int(x) for x in (SPLITS / "retaining_set_split.txt").read_text().split()][:33]
    all_items = ep.build_items(clusters, forget_idx, retain_idx)
    qa = {it[0]: {"q": it[4], "a": it[5], "falses": list(it[6])} for it in all_items}
    keys = items["D_same_object"] + items["D_hop2"]

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    mp = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval(),
        str(ADAPTERS / f"{TAG}_G2"))
    model = mp.merge_and_unload().eval()

    # CPU 缓存：merged(G2) 与 base(M) 的各层 MLP 权重
    layers = model.model.layers
    g2_cache = {l: {n: getattr(layers[l].mlp, n).weight.detach().to("cpu").clone()
                    for n in ["gate_proj", "up_proj", "down_proj"]} for l in range(28)}
    base_cache = {l: {n: getattr(base.model.layers[l].mlp, n).weight.detach().to("cpu").clone()
                      for n in ["gate_proj", "up_proj", "down_proj"]} for l in range(28)}

    out_path = ATT / "scan_mlp_G2.jsonl"
    done = {json.loads(l)["layer"] for l in open(out_path, encoding="utf-8")} if out_path.exists() else set()
    f = open(out_path, "a", encoding="utf-8")
    for l in range(28):
        if l in done:
            print(f"层 {l} 已有，跳过")
            continue
        # undo 层 l：换回 base
        for n in ["gate_proj", "up_proj", "down_proj"]:
            getattr(layers[l].mlp, n).weight.copy_(base_cache[l][n].to("cuda"))
        lps = []
        for k in keys:
            lps.append(lp_mcqa(model, tok, qa[k]["q"], qa[k]["a"], qa[k]["falses"]))
        f.write(json.dumps({"layer": l, "n": len(keys),
                            "mean_lp": sum(lps) / len(lps)}, ensure_ascii=False) + "\n")
        f.flush()
        # 恢复层 l：回到 merged(G2)
        for n in ["gate_proj", "up_proj", "down_proj"]:
            getattr(layers[l].mlp, n).weight.copy_(g2_cache[l][n].to("cuda"))
        print(f"层 {l:>2} 完成 mean_lp={sum(lps)/len(lps):.3f}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
