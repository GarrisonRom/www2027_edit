# -*- coding: utf-8 -*-
"""
研究内容二 · 阶段3：组件撤销因果实验（图 D3 数据源）
把 G2 中被归因定位的组件权重替换回冻结原模型 M 的对应权重（merge 后手术替换），
评测受损事实集 D 的恢复率，并以随机撤销做对照。

撤销操作（在 G2 merge 后的全权重模型上）：
  mlp_layers  指定层的 gate/up/down_proj 整体换回 M（层撤销）
  neurons     指定 (layer, idx) 的 down_proj 整行换回 M（神经元撤销；
              该中间神经元的输出贡献 = down_proj 第 idx 行 × 激活）

评测集：
  D_so / D_hop2   受损组 same_object 24 / hop2 24（恢复目标）；D_hop2 另按 F1 重叠拆分
  K               对照组 48（不应有明显变化）
  hop1_unlearn    33（遗忘判据：撤销的是损伤组件，F0 遗忘不应回弹）
  retain          33（保持集顺带观察）
  F1_hop2         覆盖 F1 事实的 hop2 子项（F1 复活判据：不得恢复）
配置逐个增量写 ablation_{group}.jsonl，已跑过的跳过（可断点续跑）。
"""
import argparse
import json
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import eval_per_item as ep  # 同目录：复用 build_items 保证口径一致

WS = Path(__file__).resolve().parents[2]
OUT = WS / "exp" / "results"
ATT = OUT / "attribution"
SPLITS = WS / "data" / "faithun" / "splits"
ADAPTERS = WS / "exp" / "adapters"
MODEL = ep.MODEL
TAG = "1p5B_v2"


@torch.no_grad()
def lp_mcqa(model, tok, q, a, falses):
    """返回答案长度归一化对数概率 + 是否 argmax 正确"""
    lp_a, best_wrong = None, None
    for i, c in enumerate([a] + list(falses)):
        enc = tok(q + " " + c, return_tensors="pt").to(model.device)
        n = enc["input_ids"].shape[1]
        lp = torch.log_softmax(model(**enc).logits.float(), -1)
        tgt = enc["input_ids"][0, 1:]
        v = lp[0, :n - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        s = float(v.sum()) / len(v)
        if i == 0:
            lp_a = s
        elif best_wrong is None or s > best_wrong:
            best_wrong = s
    return lp_a, int(lp_a > best_wrong)


def build_sets():
    """评测集 = item_key 列表；题面/干扰项从数据集按 build_items 同口径重建"""
    items = json.load(open(ATT / "items.json", encoding="utf-8"))
    data = json.load(open(ep.DATA, encoding="utf-8"))["data"]
    clusters = ep.build_clusters(data)
    forget_idx = [int(x) for x in (SPLITS / "forget_set_split.txt").read_text().split()]
    retain_idx = [int(x) for x in (SPLITS / "retaining_set_split.txt").read_text().split()][:33]
    all_items = ep.build_items(clusters, forget_idx, retain_idx)
    qa = {it[0]: {"q": it[4], "a": it[5], "falses": list(it[6])} for it in all_items}

    per = {json.loads(l)["item_key"]: json.loads(l)
           for l in open(OUT / "per_item_G2.jsonl", encoding="utf-8")}
    d_hop2 = items["D_hop2"]
    sets = {"D_so": items["D_same_object"], "D_hop2": d_hop2, "K": items["K"],
            "hop1_unlearn": [k for k, r in per.items() if r["source"] == "hop1_unlearn"],
            "retain": [k for k, r in per.items() if r["source"] == "retain"]}
    # F1 覆盖的 hop2 子项
    pos2global = {pos: g for pos, g in enumerate(forget_idx)}
    f1 = json.load(open(OUT / "F1.json", encoding="utf-8"))["F_star"]
    f1_chains = {(int(g), int(s)) for f in f1 for g, s in f["S"]}
    sets["F1_hop2"] = [k for k in per if k.split("|")[0] == "hop2"
                       and (pos2global[int(k.split("|")[1])], int(k.split("|")[2])) in f1_chains]
    f1_hop2_keys = set(sets["F1_hop2"])
    sets["D_hop2_F1_overlap"] = [k for k in d_hop2 if k in f1_hop2_keys]
    sets["D_hop2_no_F1_overlap"] = [k for k in d_hop2 if k not in f1_hop2_keys]
    for name, keys in sets.items():
        print(f"  评测集 {name}: {len(keys)} 题")
    return sets, qa


def undo_mlp_layers(model, base, layers):
    for l in layers:
        dst, src = model.model.layers[l].mlp, base.model.layers[l].mlp
        dst.gate_proj.weight.copy_(src.gate_proj.weight)
        dst.up_proj.weight.copy_(src.up_proj.weight)
        dst.down_proj.weight.copy_(src.down_proj.weight)


def undo_neurons(model, base, neurons):
    """恢复中间神经元 i 的整条通路：gate/up 第 i 行（产生该神经元）+
    down_proj 第 i 列（消费该神经元）"""
    for l, i in neurons:
        mlp, bmlp = model.model.layers[l].mlp, base.model.layers[l].mlp
        mlp.gate_proj.weight[i, :].copy_(bmlp.gate_proj.weight[i, :])
        mlp.up_proj.weight[i, :].copy_(bmlp.up_proj.weight[i, :])
        mlp.down_proj.weight[:, i].copy_(bmlp.down_proj.weight[:, i])


def undo_attn_layers(model, base, layers):
    """恢复指定层的注意力投影权重（q/k/v/o）"""
    for l in layers:
        dst, src = model.model.layers[l].self_attn, base.model.layers[l].self_attn
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            getattr(dst, name).weight.copy_(getattr(src, name).weight)


def eval_sets(model, tok, sets, qa):
    res = {}
    for name, keys in sets.items():
        lps, crr = [], []
        for k in keys:
            lp, ok = lp_mcqa(model, tok, qa[k]["q"], qa[k]["a"], qa[k]["falses"])
            lps.append(lp)
            crr.append(ok)
        res[name] = {"n": len(keys), "mean_lp": sum(lps) / len(lps),
                     "mean_correct": sum(crr) / len(crr)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="G2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sets, qa = build_sets()
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    shared = json.load(open(ATT / "stats_stage2.json", encoding="utf-8"))["ffn_shared_top20"]
    top12 = [(int(l), int(n)) for l, n in shared[:12]]
    rnd = random.Random(args.seed)
    rand12 = [(rnd.randrange(28), rnd.randrange(8960)) for _ in range(12)]
    rand_layers = rnd.sample(range(28), 6)

    configs = [
        ("undo_mlp_22_27", {"mlp_layers": list(range(22, 28))}),
        ("undo_mlp_27", {"mlp_layers": [27]}),
        ("undo_neurons_top12", {"neurons": top12}),
        ("undo_neurons_rand12", {"neurons": rand12}),
        ("undo_mlp_rand6", {"mlp_layers": rand_layers}),
        # ---- 模块分解：损伤到底存在哪类矩阵里 ----
        ("undo_attn_22_27", {"attn_layers": list(range(22, 28))}),
        ("undo_attn_mlp_22_27", {"attn_layers": list(range(22, 28)),
                                 "mlp_layers": list(range(22, 28))}),
        ("undo_attn_all", {"attn_layers": list(range(28))}),
        ("undo_mlp_all", {"mlp_layers": list(range(28))}),
        ("undo_all", {"attn_layers": list(range(28)), "mlp_layers": list(range(28))}),
    ]
    for m in [1, 2, 4, 8, 12]:
        configs.append((f"curve_neurons_top{m}", {"neurons": top12[:m]}))
    for i in range(1, 7):
        configs.append((f"curve_mlp_27down_{i}", {"mlp_layers": list(range(27, 27 - i, -1))}))
    # ---- 阶段4：全层单层撤销扫描（预测验证的对象，quick=只评 D 集加速）----
    for l in range(28):
        configs.append((f"scan_mlp_{l}", {"mlp_layers": [l], "quick": True}))

    out_path = ATT / f"ablation_{args.group}.jsonl"
    done = {json.loads(l)["config"] for l in open(out_path, encoding="utf-8")} if out_path.exists() else set()
    f = open(out_path, "a", encoding="utf-8")
    for name, spec in configs:
        if name in done:
            print(f"[{name}] 已有，跳过")
            continue
        mp = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval(),
            str(ADAPTERS / f"{TAG}_{args.group}"))
        model = mp.merge_and_unload().eval()
        if "mlp_layers" in spec:
            undo_mlp_layers(model, base, spec["mlp_layers"])
        if "attn_layers" in spec:
            undo_attn_layers(model, base, spec["attn_layers"])
        if "neurons" in spec:
            undo_neurons(model, base, spec["neurons"])
        quick_sets = ("D_so", "D_hop2", "D_hop2_F1_overlap", "D_hop2_no_F1_overlap")
        esets = {k: v for k, v in sets.items() if k in quick_sets} if spec.get("quick") else sets
        res = eval_sets(model, tok, esets, qa)
        f.write(json.dumps({"config": name, "group": args.group, "sets": res},
                           ensure_ascii=False) + "\n")
        f.flush()
        summary = " ".join(
            f"{key} lp={res[key]['mean_lp']:.3f}"
            for key in ("D_so", "D_hop2", "D_hop2_F1_overlap", "D_hop2_no_F1_overlap",
                        "hop1_unlearn", "F1_hop2")
            if key in res
        )
        print(f"[{name}] {summary}")
        del model, mp
        torch.cuda.empty_cache()
    f.close()


if __name__ == "__main__":
    main()
