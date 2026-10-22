# -*- coding: utf-8 -*-
"""Diagnostic replay of GSU-v3 M3A with per-anchor update constraints.

M3A keeps the original forget-gradient -> AdamW proposal.  After AdamW has
updated the parameters, it computes the closest update z to the realized
displacement delta subject to g_i dot z >= 0 for every sampled retain anchor.
The AdamW moment state is left unchanged by the correction.
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


def _gram_and_rhs(params, anchor_grads, delta):
    """Return G_ij=<g_i,g_j> and b_i=<g_i,delta> in float64 CPU tensors."""
    n = len(anchor_grads)
    gram = [[0.0 for _ in range(n)] for _ in range(n)]
    rhs = []
    for i in range(n):
        rhs.append(scalar_map_dot(params, anchor_grads[i], delta))
        for j in range(i, n):
            value = scalar_map_dot(params, anchor_grads[i], anchor_grads[j])
            gram[i][j] = value
            gram[j][i] = value
    return (
        torch.tensor(gram, dtype=torch.float64),
        torch.tensor(rhs, dtype=torch.float64),
    )


def project_to_anchor_halfspaces(params, delta, anchor_grads):
    """Project delta onto {z: <g_i,z> >= 0} with an exact 4-constraint solve.

    The projection dual has one nonnegative multiplier per active constraint.
    Since M3A uses four anchors, enumerating all active sets is small and gives
    a deterministic solution without adding a QP dependency.
    """
    count = len(anchor_grads)
    if count == 0:
        return [], [], [], {"active_set": [], "solver": "empty"}

    gram, rhs = _gram_and_rhs(params, anchor_grads, delta)
    tolerance = 1e-8
    dual_tolerance = 1e-9
    best = None

    for mask in range(1 << count):
        active = [i for i in range(count) if mask & (1 << i)]
        multipliers = torch.zeros(count, dtype=torch.float64)
        if active:
            sub = gram[active][:, active]
            sub_rhs = -rhs[active]
            try:
                solution = torch.linalg.solve(sub, sub_rhs)
            except RuntimeError:
                solution = torch.linalg.pinv(sub) @ sub_rhs
            if not torch.isfinite(solution).all():
                continue
            multipliers[active] = solution

        constraint_values = rhs + gram @ multipliers
        if not torch.isfinite(constraint_values).all():
            continue
        if active and bool((multipliers[active] < -dual_tolerance).any()):
            continue
        if bool((constraint_values < -tolerance).any()):
            continue

        correction_norm2 = float(multipliers @ gram @ multipliers)
        if correction_norm2 < -1e-7:
            continue
        candidate = (max(correction_norm2, 0.0), active, multipliers, constraint_values)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        raise RuntimeError(
            "M3A active-set projection found no feasible KKT solution; "
            f"rhs={rhs.tolist()} gram={gram.tolist()}"
        )

    _, active, multipliers, constraint_values = best
    with torch.no_grad():
        for anchor_index, multiplier in enumerate(multipliers.tolist()):
            if abs(multiplier) <= dual_tolerance:
                continue
            for p in params:
                p.add_(anchor_grads[anchor_index][p], alpha=float(multiplier))

    return (
        multipliers.tolist(),
        rhs.tolist(),
        constraint_values.tolist(),
        {
            "active_set": active,
            "solver": "enumerated_kkt_active_sets",
            "raw_constraint_min": float(rhs.min()),
            "final_constraint_min": float(constraint_values.min()),
        },
    )


def clone_grads_for_anchors(params, model, tok, anchors):
    grads = []
    for rq, ra in anchors:
        opt_loss = lp_masked(model, tok, rq, ra)
        values = torch.autograd.grad(
            opt_loss, params, allow_unused=True, retain_graph=False
        )
        grads.append(clone_grad_tuple(params, values))
    return grads


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
        (
            clusters[i][2]["hop1_unlearn"]["questions"][0],
            clusters[i][2]["hop1_unlearn"]["answers"],
        )
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
    projection_failures = 0

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
                valid_anchors = []
                for rq, ra in anchors:
                    lp = lp_masked(model, tok, rq, ra)
                    if lp is not None:
                        lps.append(lp)
                        valid_anchors.append((rq, ra))
                if not lps:
                    continue
                torch.stack(lps).sum().backward()
                g_r = clone_param_grads(params)
                lp_before = [float(lp.detach()) for lp in lps]

                # The aggregate gradient is recorded for comparison, but the
                # M3A correction uses each anchor gradient separately.
                anchor_grads = clone_grads_for_anchors(
                    params, model, tok, valid_anchors
                )

                for p in params:
                    p.grad = g_f[p]
                before = [p.detach().clone() for p in params]
                opt.step()
                delta_raw = {
                    p: p.detach().float() - old.float()
                    for p, old in zip(params, before)
                }
                raw_anchor_dot = [
                    scalar_map_dot(params, ga, delta_raw) for ga in anchor_grads
                ]
                with torch.no_grad():
                    lp_after_raw = [
                        float(lp_masked(model, tok, rq, ra))
                        for rq, ra in valid_anchors
                    ]

                try:
                    multipliers, raw_constraints, final_constraints, solver_info = (
                        project_to_anchor_halfspaces(params, delta_raw, anchor_grads)
                    )
                    corrected = any(abs(v) > 1e-9 for v in multipliers)
                except RuntimeError:
                    projection_failures += 1
                    raise

                if corrected:
                    total_corrections += 1
                delta_final = {
                    p: p.detach().float() - old.float()
                    for p, old in zip(params, before)
                }
                final_anchor_dot = [
                    scalar_map_dot(params, ga, delta_final) for ga in anchor_grads
                ]
                with torch.no_grad():
                    lp_after_final = [
                        float(lp_masked(model, tok, rq, ra))
                        for rq, ra in valid_anchors
                    ]

                raw_delta_norm2 = scalar_map_norm2(params, delta_raw)
                final_delta_norm2 = scalar_map_norm2(params, delta_final)
                correction = {
                    p: delta_final[p] - delta_raw[p]
                    for p in params
                }
                correction_norm2 = scalar_map_norm2(params, correction)
                row = {
                    "epoch": ep + 1,
                    "step": step + 1,
                    "forget_question_sha1": sha1_text(q),
                    "forget_lp": float(loss_f.detach()),
                    "anchor_count": len(valid_anchors),
                    "anchor_lp_before": lp_before,
                    "anchor_lp_after_raw_adamw": lp_after_raw,
                    "anchor_lp_after_correction": lp_after_final,
                    "anchor_lp_delta_raw": [
                        b - a for a, b in zip(lp_before, lp_after_raw)
                    ],
                    "anchor_lp_delta_final": [
                        b - a for a, b in zip(lp_before, lp_after_final)
                    ],
                    "aggregate_g_r_delta_raw_dot": scalar_map_dot(
                        params, g_r, delta_raw
                    ),
                    "aggregate_g_r_delta_final_dot": scalar_map_dot(
                        params, g_r, delta_final
                    ),
                    "raw_anchor_dot_delta": raw_anchor_dot,
                    "final_anchor_dot_delta": final_anchor_dot,
                    "raw_anchor_conflict": [d < 0.0 for d in raw_anchor_dot],
                    "final_anchor_conflict": [d < 0.0 for d in final_anchor_dot],
                    "final_anchor_violation_tol": [d < -1e-6 for d in final_anchor_dot],
                    "correction_applied": corrected,
                    "active_set": solver_info["active_set"],
                    "multipliers": multipliers,
                    "raw_constraints": raw_constraints,
                    "solver_constraints": final_constraints,
                    "raw_delta_norm": max(raw_delta_norm2, 0.0) ** 0.5,
                    "final_delta_norm": max(final_delta_norm2, 0.0) ** 0.5,
                    "correction_norm": max(correction_norm2, 0.0) ** 0.5,
                    "correction_to_raw_norm_ratio": max(correction_norm2, 0.0) ** 0.5
                    / max(raw_delta_norm2, 1e-12) ** 0.5,
                    "solver": solver_info["solver"],
                }
                diag.write(json.dumps(row, ensure_ascii=False) + "\n")
                diag.flush()
                epoch_rows_raw.append(row)
                total_forget += float(loss_f.detach())
                total_rows += 1

            def mean(key):
                vals = [r[key] for r in epoch_rows_raw]
                return sum(vals) / len(vals) if vals else None

            anchor_total = sum(r["anchor_count"] for r in epoch_rows_raw)
            epoch_summary = {
                "epoch": ep + 1,
                "steps": len(epoch_rows_raw),
                "forget_lp_mean": total_forget / max(len(epoch_rows_raw), 1),
                "raw_aggregate_conflict_rate": sum(
                    r["aggregate_g_r_delta_raw_dot"] < 0.0
                    for r in epoch_rows_raw
                ) / max(len(epoch_rows_raw), 1),
                "correction_rate": mean("correction_applied"),
                "raw_anchor_conflict_rate": sum(
                    sum(r["raw_anchor_conflict"])
                    for r in epoch_rows_raw
                ) / max(anchor_total, 1),
                "final_anchor_conflict_rate": sum(
                    sum(r["final_anchor_conflict"])
                    for r in epoch_rows_raw
                ) / max(anchor_total, 1),
                "final_anchor_violation_rate_tol": sum(
                    sum(r["final_anchor_violation_tol"])
                    for r in epoch_rows_raw
                ) / max(anchor_total, 1),
                "raw_mean_anchor_lp_delta": sum(
                    sum(r["anchor_lp_delta_raw"]) / r["anchor_count"]
                    for r in epoch_rows_raw
                ) / max(len(epoch_rows_raw), 1),
                "final_mean_anchor_lp_delta": sum(
                    sum(r["anchor_lp_delta_final"]) / r["anchor_count"]
                    for r in epoch_rows_raw
                ) / max(len(epoch_rows_raw), 1),
                "mean_correction_to_raw_norm_ratio": mean(
                    "correction_to_raw_norm_ratio"
                ),
                "mean_active_constraints": sum(
                    len(r["active_set"]) for r in epoch_rows_raw
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
        k: {
            "n": int(v["n"]),
            "mean_lp": float(v["mean_lp"]),
            "mean_correct": float(v["mean_correct"]),
        }
        for k, v in six.items()
    }
    metrics = {
        "five_set": {"group": "GSUM3A_DIAG", **five},
        "six_set": {"group": "GSUM3A_DIAG", "sets": six_serialized},
        "diagnostics": {
            "steps": total_rows,
            "corrections": total_corrections,
            "projection_failures": projection_failures,
            "epochs": epoch_rows,
            "raw_jsonl": str(diagnostics_path),
            "algorithm": "M3A AdamW proposal followed by nearest update projection onto four individual anchor half-spaces",
            "solver": "enumerated KKT active sets over four constraints",
            "optimizer_moments": "unchanged by post-hoc correction",
            "extra_passes": "aggregate anchor gradient, per-anchor gradients, and log-probabilities before/raw/final update",
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "saved": str(metrics_path),
                "steps": total_rows,
                "corrections": total_corrections,
                "projection_failures": projection_failures,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    train_and_diagnose(Path(args.run_dir).resolve())


if __name__ == "__main__":
    main()
