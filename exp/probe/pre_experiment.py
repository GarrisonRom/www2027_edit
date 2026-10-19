# -*- coding: utf-8 -*-
"""
研究内容一 · 阶段1 go/no-go 预实验
模型: Qwen2.5-1.5B-Instruct (本地)
协议 v1（冻结）:
  - 打分方式: MCQA 三选一, 续写对数概率 (长度归一化), 无 chat template
    context = question, continuation = choice
  - hop1 掌握判定: 一致率 c(f) = 4 个问法 (hop1_unlearn 1 + hop1_test 3 改写) 中
    argmax 选项 == 正确答案的比例
  - hop2 组合率: 每个 hop2 子项 3 个改写, 子项正确率 = 改写正确比例
    latent multi-hop = P(hop2 子项正确 | 簇 hop1 正确)
用法:
  python pre_experiment.py --max_test 100
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WS = Path(__file__).resolve().parents[2]
DATA = WS / "data" / "faithun" / "FaithUn_Dataset.json"
SPLITS = WS / "data" / "faithun" / "splits"
OUT = WS / "exp" / "results"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("QWEN_1_5B_PATH", str(WS / "models" / "Qwen2.5-1.5B-Instruct"))


def build_clusters(data):
    """按字典序拼接簇, 与 splits 的全局簇索引对齐 (README: concatenated sequentially)"""
    clusters = []
    for qid, clist in data.items():
        for ci, c in enumerate(clist):
            clusters.append((qid, ci, c))
    return clusters


@torch.no_grad()
def score_choices(model, tok, question, choices, batch_size=8):
    """返回每个 choice 的长度归一化对数概率"""
    results = []
    ctx_ids = [tok(q, add_special_tokens=True)["input_ids"] for q in [question] * len(choices)]
    for i in range(0, len(choices), batch_size):
        c_batch = choices[i:i + batch_size]
        ctx_batch = ctx_ids[i:i + batch_size]
        texts = [q + " " + c for q, c in zip([question] * len(c_batch), c_batch)]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
        logits = model(**enc).logits
        # 右 padding: 每个序列的预测位置 t -> token t+1
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        for j in range(len(c_batch)):
            ids = enc["input_ids"][j]
            attn = enc["attention_mask"][j]
            ctx_len = len(ctx_batch[j])
            n = int(attn.sum())
            tgt = ids[1:n]
            lp = logprobs[j, :n - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            cont = lp[ctx_len - 1:] if ctx_len >= 1 else lp
            results.append(float(cont.sum()) / max(len(cont), 1))
    return results


def mcqa_correct(model, tok, question, answer, falses):
    choices = [answer] + list(falses)
    scores = score_choices(model, tok, question, choices)
    return int(max(range(len(scores)), key=lambda k: scores[k]) == 0), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_test", type=int, default=100)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tag", default="", help="为预实验结果增加独立后缀，避免覆盖默认结果")
    args = ap.parse_args()

    data = json.load(open(DATA, encoding="utf-8"))["data"]
    clusters = build_clusters(data)
    forget_idx = set(int(x) for x in (SPLITS / "forget_set_split.txt").read_text().split())
    test_idx = set(int(x) for x in (SPLITS / "test_set_split.txt").read_text().split())
    use_idx = sorted(forget_idx) + sorted(test_idx)[:args.max_test]
    print(f"clusters total={len(clusters)}, forget={len(forget_idx)}, test_used={min(len(test_idx), args.max_test)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    records = []
    for gi, idx in enumerate(use_idx):
        qid, ci, c = clusters[idx]
        # ---- hop1: 4 个问法的一致率 ----
        q_unlearn = c["hop1_unlearn"]["questions"][0]
        q_paras = c["hop1_test"]["questions"]
        ans = c["hop1_unlearn"]["answers"]
        falses = c["hop1_unlearn"]["false_answer_options"]
        per_q = []
        for q in [q_unlearn] + list(q_paras):
            ok, _ = mcqa_correct(model, tok, q, ans, falses)
            per_q.append(ok)
        consistency = sum(per_q) / len(per_q)
        hop1_correct = int(consistency >= 2 / 3)  # 4 问法中 ≥3 对

        # ---- hop2: 每个子项的组合率 ----
        hop2 = c.get("hop2_test", {})
        h2_items = []
        n_sub = len(hop2.get("answers", []))
        for si in range(n_sub):
            qs = hop2["questions"][si] if si < len(hop2["questions"]) else []
            a = hop2["answers"][si] if si < len(hop2["answers"]) else ""
            fo = hop2["false_answer_options"][si] if si < len(hop2.get("false_answer_options", [])) else []
            if not qs or not a or not fo:
                continue
            sub = []
            for q in qs:
                ok, _ = mcqa_correct(model, tok, q, a, fo)
                sub.append(ok)
            h2_items.append({"sub_idx": si, "acc": sum(sub) / len(sub),
                             "correct": sum(sub) / len(sub) >= 2 / 3, "n_paraphrases": len(sub)})
        records.append({
            "global_idx": idx, "qid": qid, "cluster": ci,
            "hop1_consistency": consistency, "hop1_correct": hop1_correct,
            "hop2_items": h2_items,
        })
        if (gi + 1) % 20 == 0:
            print(f"  {gi + 1}/{len(use_idx)} done")

    # ---- 汇总 ----
    consist = [r["hop1_consistency"] for r in records]
    taus = [x / 20 for x in range(10, 19)]  # 0.50 - 0.90
    tau_curve = {t: sum(1 for c in consist if c >= t) / len(consist) for t in taus}

    h2_all = [it["acc"] for r in records for it in r["hop2_items"]]
    h2_given_h1 = [it["acc"] for r in records if r["hop1_correct"] for it in r["hop2_items"]]
    n_clusters_h1 = sum(r["hop1_correct"] for r in records)

    summary = {
        "model": args.model,
        "n_clusters": len(records),
        "n_hop1_facts": len(consist),
        "hop1_consistency_mean": sum(consist) / len(consist),
        "tau_curve": {str(k): v for k, v in tau_curve.items()},
        "n_hop2_items": len(h2_all),
        "hop2_acc_overall": sum(h2_all) / max(len(h2_all), 1),
        "hop2_acc_given_hop1": sum(h2_given_h1) / max(len(h2_given_h1), 1),
        "n_clusters_hop1_correct": n_clusters_h1,
        "random_floor": 1 / 3,
    }
    suffix = f"_{args.tag}" if args.tag else ""
    summary_path = OUT / f"pre_experiment_summary{suffix}.json"
    records_path = OUT / f"pre_experiment_records{suffix}.jsonl"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(records_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n===== go/no-go 结果 =====")
    print(f"单跳一致率均值: {summary['hop1_consistency_mean']:.3f}")
    print("τ 扫描掌握率: " + ", ".join(f"τ={t:.2f}:{v:.2f}" for t, v in tau_curve.items()))
    print(f"hop2 子项总数: {len(h2_all)}, 整体正确率: {summary['hop2_acc_overall']:.3f}")
    print(f"latent multi-hop (hop1对的前提下): {summary['hop2_acc_given_hop1']:.3f} "
          f"(地板 0.333, n_items={len(h2_given_h1)}, n_clusters_h1={n_clusters_h1})")
    print(f"saved: {summary_path} / {records_path}")


if __name__ == "__main__":
    main()
