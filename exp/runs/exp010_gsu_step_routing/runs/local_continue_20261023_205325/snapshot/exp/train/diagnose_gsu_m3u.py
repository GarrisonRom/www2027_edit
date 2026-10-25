# -*- coding: utf-8 -*-
"""Diagnostic replay of the original M3U AdamW update correction.

The M3U update is preserved exactly.  The script records the AdamW proposal
before correction, the corrected displacement, and direct anchor log-probability
changes at both points.
"""
import argparse
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
from diagnose_gsu_m3 import (
    EPOCHS,
    LR,
    N_ANCHOR,
    OUT,
    TARGET_MODULES,
    clone_grad_tuple,
    clone_param_grads,
    scalar_map_dot,
    scalar_map_norm2,
    sha1_text,
)


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
    total_corrections = 0

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
                # This is the original M3U retain-gradient computation.
                torch.stack(lps).sum().backward()
                g_r = clone_param_grads(params)
                lp_before = [float(lp.detach()) for lp in lps]

                # Extra diagnostic passes. They do not alter model or
                # optimizer state and are not used to form the M3U proposal.
                anchor_grads = []
                for rq, ra in anchors:
                    opt.zero_grad()
                    lp_i = lp_masked(model, tok, rq, ra)
                    grads_i = torch.autograd.grad(
                        lp_i, params, allow_unused=True, retain_graph=False
                    )
                    anchor_grads.append(clone_grad_tuple(params, grads_i))

                # Exact original M3U path: raw forget gradient -> AdamW step.
                for p in params:
                    p.grad = g_f[p]
                before = [p.detach().clone() for p in params]
                opt.step()
                delta_raw = {
                    p: p.detach().float() - old.float()
                    for p, old in zip(params, before)
                }
                raw_dot = scalar_map_dot(params, g_r, delta_raw)
                denom = scalar_map_norm2(params, g_r)
                correction_coeff = raw_dot / max(denom, 1e-12)
                raw_anchor_dot = [
                    scalar_map_dot(params, ga, delta_raw) for ga in anchor_grads
                ]
                with torch.no_grad():
                    lp_after_raw = [
                        float(lp_masked(model, tok, rq, ra))
                        for rq, ra in anchors
                    ]

                corrected = raw_dot < 0.0 and denom > 1e-12
                if corrected:
                    # Exact correction used by adamw_step_with_anchor_projection.
                    with torch.no_grad():
                        for p in params:
                            p.add_(g_r[p], alpha=-correction_coeff)
                    total_corrections += 1

                delta_final = {
                    p: p.detach().float() - old.float()
                    for p, old in zip(params, before)
                }
                corrected_dot = scalar_map_dot(params, g_r, delta_final)
                corrected_anchor_dot = [
                    scalar_map_dot(params, ga, delta_final) for ga in anchor_grads
                ]
                with torch.no_grad():
                    lp_after_final = [
                        float(lp_masked(model, tok, rq, ra))
                        for rq, ra in anchors
                    ]

                raw_delta_norm2 = scalar_map_norm2(params, delta_raw)
                final_delta_norm2 = scalar_map_norm2(params, delta_final)
                correction_norm = abs(correction_coeff) * max(denom, 0.0) ** 0.5
                row = {
                    "epoch": ep + 1,
                    "step": step + 1,
                    "forget_question_sha1": sha1_text(q),
                    "forget_lp": float(loss_f.detach()),
                    "anchor_lp_before": lp_before,
                    "anchor_lp_after_raw_adamw": lp_after_raw,
                    "anchor_lp_after_correction": lp_after_final,
                    "anchor_lp_delta_raw": [b - a for a, b in zip(lp_before, lp_after_raw)],
                    "anchor_lp_delta_final": [b - a for a, b in zip(lp_before, lp_after_final)],
                    "g_r_delta_raw_dot": raw_dot,
                    "g_r_delta_final_dot": corrected_dot,
                    "adamw_raw_conflict": raw_dot < 0.0,
                    "correction_applied": corrected,
                    "correction_coeff": correction_coeff,
                    "g_r_norm": max(denom, 0.0) ** 0.5,
                    "adamw_raw_delta_norm": max(raw_delta_norm2, 0.0) ** 0.5,
                    "adamw_final_delta_norm": max(final_delta_norm2, 0.0) ** 0.5,
                    "correction_norm": correction_norm,
                    "correction_to_raw_norm_ratio": correction_norm / max(raw_delta_norm2, 1e-12) ** 0.5,
                    "anchor_dot_raw_delta": raw_anchor_dot,
                    "anchor_dot_final_delta": corrected_anchor_dot,
                    "anchor_raw_conflict": [d < 0.0 for d in raw_anchor_dot],
                    "anchor_final_conflict": [d < 0.0 for d in corrected_anchor_dot],
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
                "raw_conflict_rate": mean("adamw_raw_conflict"),
                "correction_rate": mean("correction_applied"),
                "final_aggregate_conflict_rate": sum(
                    r["g_r_delta_final_dot"] < 0.0 for r in epoch_rows_raw
                ) / max(len(epoch_rows_raw), 1),
                "raw_anchor_conflict_rate": sum(
                    sum(r["anchor_raw_conflict"])
                    for r in epoch_rows_raw
                ) / max(4 * len(epoch_rows_raw), 1),
                "final_anchor_conflict_rate": sum(
                    sum(r["anchor_final_conflict"])
                    for r in epoch_rows_raw
                ) / max(4 * len(epoch_rows_raw), 1),
                "raw_mean_anchor_lp_delta": sum(
                    sum(r["anchor_lp_delta_raw"]) / 4 for r in epoch_rows_raw
                ) / max(len(epoch_rows_raw), 1),
                "final_mean_anchor_lp_delta": sum(
                    sum(r["anchor_lp_delta_final"]) / 4 for r in epoch_rows_raw
                ) / max(len(epoch_rows_raw), 1),
                "mean_correction_to_raw_norm_ratio": mean("correction_to_raw_norm_ratio"),
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
        "five_set": {"group": "GSUM3U_DIAG", **five},
        "six_set": {"group": "GSUM3U_DIAG", "sets": six_serialized},
        "diagnostics": {
            "steps": total_rows,
            "corrections": total_corrections,
            "epochs": epoch_rows,
            "raw_jsonl": str(diagnostics_path),
            "algorithm": "original M3U AdamW proposal followed by aggregate retain-gradient displacement correction",
            "extra_passes": "per-anchor gradients and log-probabilities before AdamW, after raw AdamW, and after correction",
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps({"saved": str(metrics_path), "steps": total_rows, "corrections": total_corrections}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    train_and_diagnose(Path(args.run_dir).resolve())


if __name__ == "__main__":
    main()
