# -*- coding: utf-8 -*-
"""Diagnostic-only replay of the original GSU M3 update rule.

The optimizer and M3 projection are kept identical to train_gsu.py.  This
script adds measurements for the aggregate retain gradient, each sampled
anchor gradient, the per-tensor M3 projection, the actual AdamW displacement,
and the direct retain log-probability change after each update.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

TE = Path(__file__).resolve().parent
sys.path.insert(0, str(TE))
sys.path.insert(0, str(TE.parent / "damage"))
import train_eval as te
import component_ablation as ca
from train_gsu import lp_masked

WS = Path(__file__).resolve().parents[2]
OUT = WS / "exp" / "results"
LR, EPOCHS, N_ANCHOR = 5e-5, 3, 4
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def scalar_map_dot(params, left, right):
    value = torch.zeros((), device=params[0].device, dtype=torch.float32)
    for p in params:
        value = value + torch.sum(left[p].float() * right[p].float())
    return float(value)


def scalar_map_norm2(params, values):
    value = torch.zeros((), device=params[0].device, dtype=torch.float32)
    for p in params:
        value = value + torch.sum(values[p].float() * values[p].float())
    return float(value)


def clone_param_grads(params):
    return {
        p: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for p in params
    }


def clone_grad_tuple(params, grads):
    return {
        p: (g.detach().clone() if g is not None else torch.zeros_like(p))
        for p, g in zip(params, grads)
    }


def sha1_text(value):
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def train_and_diagnose(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = run_dir / "diagnostics.jsonl"
    epoch_path = run_dir / "epoch_metrics.jsonl"
    metrics_path = run_dir / "metrics.json"

    random.seed(te.SEED)
    torch.manual_seed(te.SEED)

    data = json.load(open(te.DATA, encoding="utf-8"))["data"]
    clusters = te.build_clusters(data)
    forget_idx = [int(x) for x in (te.SPLITS / "forget_set_split.txt").read_text().split()]
    retain_idx = [int(x) for x in (te.SPLITS / "retaining_set_split.txt").read_text().split()][:33]
    f1 = json.load(open(OUT / "F1.json", encoding="utf-8"))["F_star"]
    cands = json.load(open(OUT / "candidates.json", encoding="utf-8"))["facts"]

    tok = AutoTokenizer.from_pretrained(te.MODEL)
    retain_ex = [
        (clusters[i][2]["hop1_unlearn"]["questions"][0],
         clusters[i][2]["hop1_unlearn"]["answers"])
        for i in retain_idx
    ]
    forget_ex = te.make_group_examples("G2", clusters, forget_idx, f1, cands)

    base = AutoModelForCausalLM.from_pretrained(
        te.MODEL, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    for p in base.parameters():
        p.requires_grad = False
    model = get_peft_model(
        base,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    rng = random.Random(te.SEED)
    epoch_rows = []
    total_rows = 0

    with open(diagnostics_path, "w", encoding="utf-8") as diag:
        for ep in range(EPOCHS):
            random.Random(te.SEED + ep).shuffle(forget_ex)
            epoch_rows_raw = []
            total_forget = 0.0
            for step, (q, a) in enumerate(forget_ex):
                opt.zero_grad()
                loss_f = lp_masked(model, tok, q, a)
                if loss_f is None:
                    continue
                loss_f.backward()
                g_f = clone_param_grads(params)

                opt.zero_grad()
                anchors = rng.sample(retain_ex, N_ANCHOR)
                lps = []
                for rq, ra in anchors:
                    lp = lp_masked(model, tok, rq, ra)
                    if lp is not None:
                        lps.append(lp)
                # This is the original M3 retain-gradient computation.
                torch.stack(lps).sum().backward()
                g_r = clone_param_grads(params)
                lp_before = [float(lp.detach()) for lp in lps]

                # Extra diagnostic passes only. They do not alter parameters
                # or optimizer state and are not used to form the M3 update.
                anchor_grads = []
                for rq, ra in anchors:
                    opt.zero_grad()
                    lp_i = lp_masked(model, tok, rq, ra)
                    grads_i = torch.autograd.grad(
                        lp_i, params, allow_unused=True, retain_graph=False
                    )
                    anchor_grads.append(clone_grad_tuple(params, grads_i))

                g_proj = {}
                tensor_dots = []
                raw_norm2 = 0.0
                projected_norm2 = 0.0
                for p in params:
                    g = g_f[p]
                    ga = g_r[p]
                    dot = float((g * ga).sum())
                    denom = float((ga * ga).sum()) + 1e-12
                    # Exact original M3 rule: unconditional, per-tensor
                    # projection with alpha=1.
                    gp = g - dot / denom * ga
                    g_proj[p] = gp
                    tensor_dots.append(dot)
                    raw_norm2 += float(torch.sum(g.float() * g.float()))
                    projected_norm2 += float(torch.sum(gp.float() * gp.float()))
                    p.grad = gp

                global_dot = scalar_map_dot(params, g_f, g_r)
                global_gf_norm2 = scalar_map_norm2(params, g_f)
                global_gr_norm2 = scalar_map_norm2(params, g_r)
                global_projected_norm2 = (
                    global_gf_norm2
                    - global_dot * global_dot / max(global_gr_norm2, 1e-12)
                )

                before = [p.detach().clone() for p in params]
                opt.step()
                delta = {
                    p: p.detach().float() - old.float()
                    for p, old in zip(params, before)
                }

                actual_delta_dot = scalar_map_dot(params, g_r, delta)
                delta_norm2 = scalar_map_norm2(params, delta)
                anchor_dot_raw = [
                    scalar_map_dot(params, ga, g_f) for ga in anchor_grads
                ]
                anchor_dot_projected = [
                    scalar_map_dot(params, ga, g_proj) for ga in anchor_grads
                ]
                anchor_dot_delta = [
                    scalar_map_dot(params, ga, delta) for ga in anchor_grads
                ]

                with torch.no_grad():
                    lp_after = [
                        float(lp_masked(model, tok, rq, ra))
                        for rq, ra in anchors
                    ]

                row = {
                    "epoch": ep + 1,
                    "step": step + 1,
                    "forget_question_sha1": sha1_text(q),
                    "forget_lp": float(loss_f.detach()),
                    "anchor_lp_before": lp_before,
                    "anchor_lp_after": lp_after,
                    "anchor_lp_delta": [b - a for a, b in zip(lp_before, lp_after)],
                    "global_gf_gr_dot": global_dot,
                    "global_gf_norm": global_gf_norm2 ** 0.5,
                    "global_gr_norm": global_gr_norm2 ** 0.5,
                    "global_gf_gr_cos": global_dot / max(
                        (global_gf_norm2 * global_gr_norm2) ** 0.5, 1e-12
                    ),
                    "global_conflict": global_dot > 0.0,
                    "tensor_count": len(params),
                    "tensor_positive_dot_fraction": sum(d > 0.0 for d in tensor_dots) / len(tensor_dots),
                    "tensor_dot_min": min(tensor_dots),
                    "tensor_dot_max": max(tensor_dots),
                    "m3_projected_norm_ratio": (projected_norm2 / max(raw_norm2, 1e-12)) ** 0.5,
                    "global_projected_norm_ratio": (max(global_projected_norm2, 0.0) / max(global_gf_norm2, 1e-12)) ** 0.5,
                    "m3_projected_global_dot": scalar_map_dot(params, g_proj, g_r),
                    "adamw_delta_norm": delta_norm2 ** 0.5,
                    "adamw_delta_gr_dot": actual_delta_dot,
                    "adamw_delta_retain_conflict": actual_delta_dot < 0.0,
                    "anchor_dot_raw_gradient": anchor_dot_raw,
                    "anchor_dot_projected_gradient": anchor_dot_projected,
                    "anchor_dot_actual_delta": anchor_dot_delta,
                    "anchor_actual_conflict": [d < 0.0 for d in anchor_dot_delta],
                }
                diag.write(json.dumps(row, ensure_ascii=False) + "\n")
                diag.flush()
                epoch_rows_raw.append(row)
                total_forget += float(loss_f.detach())
                total_rows += 1

            def mean(key):
                vals = [r[key] for r in epoch_rows_raw]
                return sum(vals) / len(vals) if vals else None

            epoch_summary = {
                "epoch": ep + 1,
                "steps": len(epoch_rows_raw),
                "forget_lp_mean": total_forget / max(len(epoch_rows_raw), 1),
                "global_conflict_rate": mean("global_conflict"),
                "tensor_positive_dot_fraction_mean": mean("tensor_positive_dot_fraction"),
                "m3_projected_norm_ratio_mean": mean("m3_projected_norm_ratio"),
                "global_projected_norm_ratio_mean": mean("global_projected_norm_ratio"),
                "adamw_actual_conflict_rate": mean("adamw_delta_retain_conflict"),
                "anchor_actual_conflict_rate": sum(
                    sum(row["anchor_actual_conflict"])
                    for row in epoch_rows_raw
                ) / max(4 * len(epoch_rows_raw), 1),
                "mean_anchor_lp_delta": sum(
                    sum(row["anchor_lp_delta"]) / 4 for row in epoch_rows_raw
                ) / max(len(epoch_rows_raw), 1),
            }
            epoch_rows.append(epoch_summary)
            with open(epoch_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(epoch_summary, ensure_ascii=False) + "\n")
            print(json.dumps(epoch_summary, ensure_ascii=False), flush=True)

            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(
                {n: p.detach().cpu() for n, p in model.named_parameters() if "lora" in n},
                ckpt_dir / f"ep{ep + 1}.pt",
            )

    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir)

    model.eval()
    five = te.evaluate(model, tok, clusters, forget_idx, retain_idx)
    sets, qa = ca.build_sets()
    six = ca.eval_sets(model, tok, sets, qa)
    six_serialized = {
        k: {"n": int(v["n"]), "mean_lp": float(v["mean_lp"]), "mean_correct": float(v["mean_correct"])}
        for k, v in six.items()
    }
    metrics = {
        "five_set": {"group": "GSUM3_DIAG", **five},
        "six_set": {"group": "GSUM3_DIAG", "sets": six_serialized},
        "diagnostics": {
            "steps": total_rows,
            "epochs": epoch_rows,
            "raw_jsonl": str(diagnostics_path),
            "algorithm": "original M3 unconditional per-tensor gradient projection",
            "extra_passes": "per-anchor gradients and post-update log-probabilities only",
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps({"saved": str(metrics_path), "steps": total_rows}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    train_and_diagnose(Path(args.run_dir).resolve())


if __name__ == "__main__":
    main()
