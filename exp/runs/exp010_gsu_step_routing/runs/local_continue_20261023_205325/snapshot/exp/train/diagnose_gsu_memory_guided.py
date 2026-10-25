# -*- coding: utf-8 -*-
"""GSU-v6: frozen multi-memory component guidance screen.

This script implements the preregistered A0-A4 comparison and the paired
random controls A2R/A3R. It keeps the FB16 post-AdamW feedback controller
unchanged. The only method change is whether the forget gradient is
reweighted by a component score computed once before training.

The score is computed at LoRA-component granularity:

    A_j = ||sum_i grad(F0_i, j)|| / sum_i ||grad(F0_i, j)||
    P_j = path-weighted mean cosine(mean_F0_j, grad(F1_fact, j))
    S_j = A_j * max(P_j, 0)

F1 facts are used as direct supporting-fact memories. Their active-path
metadata supplies fixed weights; hop2 evaluation question text is never used
as a training example or as a gradient target.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "damage"))

import component_ablation as ca
import train_eval as te
from diagnose_gsu_actual_feedback import (
    ACTUAL_DROP_TOL,
    MAX_FEEDBACK_ROUNDS,
    MAX_CORRECTION_RATIO,
    anchor_gradients,
    anchor_logprobs,
    feedback_update,
)
from diagnose_gsu_m3 import (
    TARGET_MODULES,
    clone_param_grads,
    scalar_map_dot,
    scalar_map_norm2,
    sha1_text,
)
from train_gsu import lp_masked


WS = Path(__file__).resolve().parents[2]
OUT = WS / "exp" / "results"
SEED = te.SEED
LR = 5e-5
EPOCHS = 3
TARGET_UPDATES = 114
N_ANCHOR = 16
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
RATIO_EPS = 1e-6
RANDOM_GUIDE_SEED = SEED + 7919

VARIANTS = ("A0", "A1", "A2", "A3", "A4", "A2R", "A3R")
F0_ONLY = ("A0", "A2", "A3", "A2R", "A3R")
EXPERIMENT = "exp006_gsu_memory_guided"
METHOD = "GSU-v6-memory-guided"
EXPECTED_SET_COUNTS = {
    "D_so": 24, "D_hop2": 24, "K": 48, "hop1_unlearn": 33,
    "retain": 33, "F1_hop2": 75, "D_hop2_F1_overlap": 12,
    "D_hop2_no_F1_overlap": 12,
}
FROZEN_DATA_SHA256 = "d570cc3e30c083357ca45ab4dd9df807abbd800f51a657d89629e4fe7b3adbb0"
F1_TEMPLATES = te.TPL1
COMPONENT_RE = re.compile(
    r"\.layers\.(\d+)\.(self_attn|mlp)\.([^.]+)\.lora_[AB]\."
)


def write_json(path, value, overwrite=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w" if overwrite else "x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def stable_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def finite_values(values, label):
    if not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError(f"Non-finite {label}")


def finite_gradients(gradients, label):
    if any(not bool(torch.isfinite(value).all()) for value in gradients.values()):
        raise RuntimeError(f"Non-finite {label}")


def component_key(parameter_name):
    match = COMPONENT_RE.search(parameter_name)
    if not match:
        return None
    layer, block, module = match.groups()
    return f"L{int(layer):02d}.{block}.{module}"


def collect_component_groups(model):
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    groups = defaultdict(list)
    parameter_to_component = {}
    parameter_names = {}
    for name, parameter in named:
        key = component_key(name)
        if key is None:
            raise RuntimeError(f"Unparsed trainable parameter: {name}")
        groups[key].append(parameter)
        parameter_to_component[parameter] = key
        parameter_names[parameter] = name

    if not groups:
        raise RuntimeError("No LoRA target components were found")
    if any(len(parameters) != 2 for parameters in groups.values()):
        raise RuntimeError("Each component must contain one LoRA A/B pair")
    return (
        [parameter for _, parameter in named
         if parameter in parameter_to_component],
        dict(sorted(groups.items())),
        parameter_to_component,
        parameter_names,
    )


def make_component_cpu_sums(groups):
    return {
        key: {
            parameter: torch.zeros(parameter.shape, dtype=torch.float32)
            for parameter in parameters
        }
        for key, parameters in groups.items()
    }


def cpu_gradient_snapshot(model, tok, params, question, answer):
    model.zero_grad(set_to_none=True)
    value = lp_masked(model, tok, question, answer)
    if value is None:
        raise RuntimeError("Empty answer mask during frozen component scoring")
    finite_values([value.detach()], "component-scoring logprob")
    gradients = torch.autograd.grad(
        value, params, allow_unused=True, retain_graph=False
    )
    snapshot = {}
    for parameter, gradient in zip(params, gradients):
        if gradient is None:
            snapshot[parameter] = torch.zeros(
                parameter.shape, dtype=torch.float32
            )
        else:
            snapshot[parameter] = gradient.detach().float().cpu()
    finite_gradients(snapshot, "component-scoring gradient")
    return snapshot, float(value.detach())


def component_norm(parameters, gradients):
    norm2 = 0.0
    for parameter in parameters:
        value = gradients[parameter].float()
        norm2 += float(torch.sum(value * value))
    return math.sqrt(max(norm2, 0.0))


def accumulate_component_snapshot(sums, norm_sums, groups, snapshot):
    for key, parameters in groups.items():
        norm_sums[key] += component_norm(parameters, snapshot)
        for parameter in parameters:
            sums[key][parameter].add_(snapshot[parameter])


def cosine_from_mean(mean_grad, parameters, memory_grad):
    dot = 0.0
    mean_norm2 = 0.0
    memory_norm2 = 0.0
    for parameter in parameters:
        left = mean_grad[parameter].float()
        right = memory_grad[parameter].float()
        dot += float(torch.sum(left * right))
        mean_norm2 += float(torch.sum(left * left))
        memory_norm2 += float(torch.sum(right * right))
    denominator = math.sqrt(max(mean_norm2 * memory_norm2, 0.0))
    if denominator <= 1e-20:
        return 0.0
    return max(-1.0, min(1.0, dot / denominator))


def compute_component_scores(model, tok, f0_items, memory_items, params, groups):
    """Compute frozen A/P/S values without changing model parameters."""
    f0_sums = make_component_cpu_sums(groups)
    f0_norm_sums = {key: 0.0 for key in groups}
    f0_count = 0
    f0_skipped = []

    for item in f0_items:
        result = cpu_gradient_snapshot(
            model, tok, params, item["question"], item["answer"]
        )
        if result is None:
            f0_skipped.append(item["id"])
            continue
        snapshot, _ = result
        accumulate_component_snapshot(
            f0_sums, f0_norm_sums, groups, snapshot
        )
        f0_count += 1

    if f0_count == 0:
        raise RuntimeError("F0 component scoring produced no valid gradients")

    f0_means = {
        key: {
            parameter: f0_sums[key][parameter] / float(f0_count)
            for parameter in parameters
        }
        for key, parameters in groups.items()
    }

    alignment_sums = {key: 0.0 for key in groups}
    alignment_weights = {key: 0.0 for key in groups}
    memory_count = 0
    memory_skipped = []
    for item in memory_items:
        result = cpu_gradient_snapshot(
            model, tok, params, item["question"], item["answer"]
        )
        if result is None:
            memory_skipped.append(item["id"])
            continue
        snapshot, _ = result
        path_weight = float(item["path_count"])
        if path_weight <= 0.0:
            raise RuntimeError("A selected memory must have at least one active path")
        for key, parameters in groups.items():
            alignment_sums[key] += path_weight * cosine_from_mean(
                f0_means[key], parameters, snapshot
            )
            alignment_weights[key] += path_weight
        memory_count += 1

    rows = []
    for key, parameters in groups.items():
        f0_mean_norm = component_norm(parameters, f0_means[key])
        denominator = f0_norm_sums[key]
        consistency = (
            f0_mean_norm * f0_count / denominator
            if denominator > 1e-20 else 0.0
        )
        consistency = max(0.0, min(1.0, consistency))
        path_consistency = (
            alignment_sums[key] / alignment_weights[key]
            if alignment_weights[key] > 0.0 else 0.0
        )
        score = consistency * max(path_consistency, 0.0)
        rows.append({
            "component": key,
            "parameter_count": sum(
                int(parameter.numel()) for parameter in parameters
            ),
            "f0_gradient_count": f0_count,
            "memory_gradient_count": memory_count,
            "path_weight_total": alignment_weights[key],
            "f0_internal_consistency_A": consistency,
            "f0_memory_path_consistency_P": path_consistency,
            "f0_average_gradient_strength": (
                f0_norm_sums[key] / float(f0_count)
            ),
            "shared_score_S": score,
        })

    return rows, {
        "f0_count": f0_count,
        "f0_skipped": f0_skipped,
        "memory_count": memory_count,
        "memory_skipped": memory_skipped,
        "formula": {
            "A": "||sum_i grad(F0_i,j)|| / sum_i ||grad(F0_i,j)||",
            "P": "path-weighted mean cosine(mean_F0_j, grad(F1_fact,j))",
            "S": "A * max(P, 0)",
        },
    }


def build_component_guide(score_rows, variant):
    if variant not in VARIANTS or not score_rows:
        raise ValueError("Unknown variant or empty component scores")
    score_by_key = {
        row["component"]: float(row["shared_score_S"])
        for row in score_rows
    }
    keys = sorted(score_by_key)
    values = [score_by_key[key] for key in keys]
    finite_values(values, "component scores")
    if len(score_by_key) != len(score_rows) or min(values) < 0.0:
        raise ValueError("Component scores must be unique and nonnegative")
    maximum = max(values) if values else 0.0

    def soft_weights(score_map):
        maximum_value = max(score_map.values()) if score_map else 0.0
        if maximum_value <= 0.0:
            return {key: 1.0 for key in score_map}, True
        return {
            key: max(0.0, float(value)) / maximum_value
            for key, value in score_map.items()
        }, False

    random_values = list(values)
    random.Random(RANDOM_GUIDE_SEED).shuffle(random_values)
    random_score_by_key = dict(zip(keys, random_values))
    structured_soft, structured_zero_fallback = soft_weights(score_by_key)
    random_soft, random_zero_fallback = soft_weights(random_score_by_key)
    median_score = statistics.median(values)
    random_median = statistics.median(random_values)
    structured_hard = {
        key: 1.0 if score_by_key[key] >= median_score else 0.0
        for key in keys
    }
    random_hard = {
        key: 1.0 if random_score_by_key[key] >= random_median else 0.0
        for key in keys
    }

    if variant in ("A2", "A4"):
        weights = structured_soft
        mode = "structured_soft"
        applied = True
    elif variant == "A3":
        weights = structured_hard
        mode = "structured_hard_median"
        applied = True
    elif variant == "A2R":
        weights = random_soft
        mode = "random_soft_permutation"
        applied = True
    elif variant == "A3R":
        weights = random_hard
        mode = "random_hard_permutation_median"
        applied = True
    else:
        weights = {key: 1.0 for key in keys}
        mode = "none"
        applied = False

    for row in score_rows:
        key = row["component"]
        row["structured_soft_weight"] = structured_soft[key]
        row["random_soft_weight"] = random_soft[key]
        row["structured_hard_keep"] = bool(structured_hard[key])
        row["random_hard_keep"] = bool(random_hard[key])
        row["applied_weight"] = weights[key]

    return {
        "variant": variant,
        "mode": mode,
        "applied": applied,
        "weights": weights,
        "structured_median_S": median_score,
        "random_median_S": random_median,
        "structured_active_components": sum(
            value > 0.0 for value in structured_soft.values()
        ),
        "structured_hard_kept_components": sum(
            value > 0.0 for value in structured_hard.values()
        ),
        "random_hard_kept_components": sum(
            value > 0.0 for value in random_hard.values()
        ),
        "maximum_S": maximum,
        "random_seed": RANDOM_GUIDE_SEED,
        "random_mapping": {
            key: random_score_by_key[key] for key in keys
        },
        "zero_score_fallback": {
            "structured_soft": structured_zero_fallback,
            "random_soft": random_zero_fallback,
        },
        "score_definition": (
            "All component weights are frozen before update 1. "
            "Soft weights are S/max(S); hard weights keep S >= median(S). "
            "If every score is zero, the soft guide is the A0 identity."
        ),
    }


def apply_component_guidance(
    params, raw_gradients, parameter_to_component, weights, apply
):
    finite_gradients(raw_gradients, "raw forget gradient")
    raw_norm2 = scalar_map_norm2(params, raw_gradients)
    if not apply or all(float(weights[parameter_to_component[p]]) == 1.0 for p in params):
        return (
            {
                parameter: raw_gradients[parameter].clone()
                for parameter in params
            },
            {
                "raw_norm": math.sqrt(max(raw_norm2, 0.0)),
                "guided_norm": math.sqrt(max(raw_norm2, 0.0)),
                "renorm_scale": 1.0,
                "fallback_to_raw": False,
            },
        )

    weighted = {}
    weighted_norm2 = 0.0
    for parameter in params:
        weight = float(weights[parameter_to_component[parameter]])
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("Guidance weights must be finite and nonnegative")
        value = raw_gradients[parameter] * weight
        weighted[parameter] = value
        weighted_norm2 += float(torch.sum(value.float() * value.float()))

    if weighted_norm2 <= 1e-24:
        if raw_norm2 > 1e-24:
            raise RuntimeError("The frozen guide annihilated a nonzero gradient")
        return (
            {
                parameter: raw_gradients[parameter].clone()
                for parameter in params
            },
            {
                "raw_norm": math.sqrt(max(raw_norm2, 0.0)),
                "guided_norm": math.sqrt(max(raw_norm2, 0.0)),
                "renorm_scale": 1.0,
                "fallback_to_raw": False,
            },
        )

    scale = math.sqrt(max(raw_norm2, 0.0) / weighted_norm2)
    guided = {
        parameter: weighted[parameter] * scale
        for parameter in params
    }
    guided_norm2 = scalar_map_norm2(params, guided)
    finite_values([raw_norm2, guided_norm2, scale], "guided gradient norm")
    return guided, {
        "raw_norm": math.sqrt(max(raw_norm2, 0.0)),
        "guided_norm": math.sqrt(max(guided_norm2, 0.0)),
        "renorm_scale": scale,
        "fallback_to_raw": False,
    }


def fact_question(fact):
    template = F1_TEMPLATES.get(
        fact["relation"], "What is the {r} of {s}?"
    )
    return template.format(
        s=fact["subject"], r=fact["relation"]
    )


def build_f0_items(clusters, forget_idx):
    items = []
    for position, index in enumerate(forget_idx):
        cluster = clusters[index][2]
        items.append({
            "id": f"F0:{position}:{index}",
            "source": "F0",
            "cluster_index": int(index),
            "question": cluster["hop1_unlearn"]["questions"][0],
            "answer": cluster["hop1_unlearn"]["answers"],
            "false_options": list(
                cluster["hop1_unlearn"]["false_answer_options"]
            ),
            "path_count": 0,
        })
    return items


def build_f1_items(f1_facts, candidates, active_paths):
    candidate_by_key = {
        (
            fact["subject"],
            fact["relation"],
            fact["object"],
        ): fact
        for fact in candidates
    }
    active = {
        (int(path[0]), int(path[1])) for path in active_paths
    }
    items = []
    for position, fact in enumerate(f1_facts):
        key = (fact["subject"], fact["relation"], fact["object"])
        if key not in candidate_by_key:
            raise RuntimeError(f"Frozen F1 fact is missing from candidates: {key}")
        candidate = candidate_by_key[key]
        raw_paths = [
            (int(path[0]), int(path[1])) for path in fact.get("S", [])
        ]
        if not raw_paths or len(set(raw_paths)) != len(raw_paths):
            raise RuntimeError(f"Empty or duplicated frozen F1 paths: {key}")
        if not set(raw_paths).issubset(active):
            raise RuntimeError(f"Frozen F1 path is not in active_paths.json: {key}")
        path_ids = raw_paths
        false_options = list(candidate.get("distractors", []))
        if len(false_options) != 2 or fact["object"] in false_options:
            raise RuntimeError(f"Invalid frozen F1 distractors: {key}")
        items.append({
            "id": (
                f"F1:{position}:{fact['subject']}|"
                f"{fact['relation']}|{fact['object']}"
            ),
            "source": "F1",
            "fact_key": list(key),
            "question": fact_question(fact),
            "answer": fact["object"],
            "false_options": false_options[:2],
            "path_ids": [list(path) for path in path_ids],
            "path_count": len(path_ids),
        })
    return items


def build_retain_items(clusters, retain_idx):
    items = []
    for position, index in enumerate(retain_idx):
        cluster = clusters[index][2]
        items.append({
            "id": f"retain:{position}:{index}",
            "source": "retain",
            "cluster_index": int(index),
            "question": cluster["hop1_unlearn"]["questions"][0],
            "answer": cluster["hop1_unlearn"]["answers"],
            "false_options": list(
                cluster["hop1_unlearn"]["false_answer_options"]
            ),
            "path_count": 0,
        })
    return items


def load_inputs():
    if sha256_file(te.DATA) != FROZEN_DATA_SHA256:
        raise RuntimeError("Dataset checksum differs from the frozen FaithUn protocol")
    data = read_json(te.DATA)["data"]
    clusters = te.build_clusters(data)
    indices = {
        name: [int(value) for value in (te.SPLITS / filename).read_text().split()]
        for name, filename in (
            ("forget", "forget_set_split.txt"),
            ("retain", "retaining_set_split.txt"),
            ("test", "test_set_split.txt"),
        )
    }
    for name, count in (("forget", 33), ("retain", 66), ("test", 464)):
        values = indices[name]
        if len(values) != count or len(set(values)) != count:
            raise RuntimeError(f"Unexpected {name} split size or duplicates")
        if not all(0 <= value < len(clusters) for value in values):
            raise RuntimeError(f"Invalid {name} split index")
    for left, right in (("forget", "retain"), ("forget", "test"), ("retain", "test")):
        if set(indices[left]) & set(indices[right]):
            raise RuntimeError(f"Overlapping {left}/{right} split")
    facts = read_json(OUT / "F1.json")["F_star"]
    if len(facts) != 5:
        raise RuntimeError("The frozen F1 must contain exactly five facts")
    candidates = read_json(OUT / "candidates.json")["facts"]
    active = read_json(OUT / "active_paths.json")["P_act"]
    if any(int(path[0]) not in indices["forget"] for path in active):
        raise RuntimeError("An active path is outside the fixed forget split")
    f0 = build_f0_items(clusters, indices["forget"])
    f1 = build_f1_items(facts, candidates, active)
    if len({tuple(item["fact_key"]) for item in f1}) != 5:
        raise RuntimeError("Duplicate F1 facts")
    for legacy_group, items in (("G1", f0), ("G2", f0 + f1)):
        legacy = te.make_group_examples(
            legacy_group, clusters, indices["forget"], facts, candidates
        )
        if legacy != [(item["question"], item["answer"]) for item in items]:
            raise RuntimeError(f"Training QA differs from the frozen {legacy_group}")
    with contextlib.redirect_stdout(sys.stderr):
        sets, qa = ca.build_sets()
    if {key: len(value) for key, value in sets.items()} != EXPECTED_SET_COUNTS:
        raise RuntimeError("Frozen evaluation set counts changed")
    overlap = set(sets["D_hop2_F1_overlap"])
    no_overlap = set(sets["D_hop2_no_F1_overlap"])
    if overlap & no_overlap or overlap | no_overlap != set(sets["D_hop2"]):
        raise RuntimeError("The primary/overlap evaluation partition is invalid")
    if any(key not in qa for values in sets.values() for key in values):
        raise RuntimeError("An evaluation item has no frozen question/answer")
    return {
        "clusters": clusters, "indices": indices, "F0": f0, "F1": f1,
        "retain_all": build_retain_items(clusters, indices["retain"]),
        "sets": sets, "qa": qa,
    }


def preflight(model_path):
    model_dir = Path(model_path).resolve()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Local model is missing; supply --model: {model_dir}")
    model_config = read_json(model_dir / "config.json")
    if (model_config.get("model_type"), model_config.get("num_hidden_layers"),
            model_config.get("hidden_size")) != ("qwen2", 28, 1536):
        raise RuntimeError("This frozen screen requires the local Qwen2.5-1.5B model")
    inputs = load_inputs()
    plans = {}
    for name, items in (("F0", inputs["F0"]), ("F0+F1", inputs["F0"] + inputs["F1"])):
        plans[name] = schedule_manifest(
            make_update_schedule(items, TARGET_UPDATES, SEED), items, name
        )
    return inputs, {
        "model": str(model_dir), "model_config": model_config,
        "component_count": int(model_config["num_hidden_layers"]) * len(TARGET_MODULES),
        "F0_count": len(inputs["F0"]), "F1_count": len(inputs["F1"]),
        "retain_anchor_pool": 33, "retain_audit_count": 66,
        "retain_heldout_count": 33, "eval_counts": EXPECTED_SET_COUNTS,
        "memory_path_counts": [item["path_count"] for item in inputs["F1"]],
        "schedules": plans,
    }


def make_update_schedule(items, target_updates, seed):
    if not items:
        raise ValueError("Cannot build a schedule from an empty example list")
    schedule = []
    block = 0
    order = list(range(len(items)))
    while len(schedule) < target_updates:
        # Match legacy FB16: shuffle the previous permutation in place.
        random.Random(seed + block).shuffle(order)
        remaining = target_updates - len(schedule)
        for item_index in order[:remaining]:
            item = items[item_index]
            schedule.append({
                "update": len(schedule) + 1,
                "schedule_block": block + 1,
                "item_index": item_index,
                "item_id": item["id"],
                "source": item.get("source", "unknown"),
                "question": item["question"],
                "answer": item["answer"],
                "question_sha1": sha1_text(item["question"]),
            })
        block += 1
    return schedule


def make_anchor_schedule(pool_size=33, updates=TARGET_UPDATES):
    if pool_size < N_ANCHOR:
        raise ValueError("Anchor pool is smaller than the frozen per-update count")
    rng = random.Random(SEED)
    return [rng.sample(range(pool_size), N_ANCHOR) for _ in range(updates)]


def schedule_manifest(schedule, items, source):
    counts = defaultdict(int)
    for row in schedule:
        counts[row["item_id"]] += 1
    return {
        "source": source,
        "seed": SEED,
        "ordering": "in-place shuffle of previous permutation with seed+block",
        "example_count": len(items),
        "target_updates": len(schedule),
        "schedule_blocks": max(row["schedule_block"] for row in schedule),
        "item_counts": dict(sorted(counts.items())),
        "plan": [
            {
                "update": row["update"],
                "schedule_block": row["schedule_block"],
                "item_id": row["item_id"],
                "source": row["source"],
                "question_sha1": row["question_sha1"],
            }
            for row in schedule
        ],
    }


def answer_logprob(model, tok, item):
    value = lp_masked(model, tok, item["question"], item["answer"])
    if value is None:
        raise RuntimeError(f"Empty answer mask in audit: {item.get('id')}")
    result = float(value.detach())
    finite_values([result], "answer-only audit logprob")
    return result


def choice_scores(model, tok, item):
    choices = [item["answer"]] + list(item["false_options"])
    scores = te.score_choices(model, tok, item["question"], choices)
    finite_values(scores, "legacy MCQA scores")
    return scores


def collect_item_audit(model, tok, items):
    was_training = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for item in items:
            scores = choice_scores(model, tok, item)
            wrong = max(scores[1:]) if len(scores) > 1 else None
            rows.append({
                "id": item["id"],
                "source": item["source"],
                "answer_logprob": answer_logprob(model, tok, item),
                "correct_answer_logprob": scores[0],
                "mcqa_scores": scores,
                "mcqa_scoring": "legacy_full_sequence_mean_logprob",
                "mcqa_margin": (
                    scores[0] - wrong if wrong is not None else None
                ),
                "mcqa_correct": (
                    int(scores[0] > wrong) if wrong is not None else None
                ),
                "path_count": int(item.get("path_count", 0)),
            })
    if was_training:
        model.train()
    return rows


def attach_initial_deltas(initial_rows, final_rows):
    initial_by_id = {row["id"]: row for row in initial_rows}
    for row in final_rows:
        initial = initial_by_id[row["id"]]
        row["initial_answer_logprob"] = initial["answer_logprob"]
        row["answer_logprob_delta"] = (
            row["answer_logprob"] - initial["answer_logprob"]
        )
        row["initial_correct_answer_logprob"] = initial["correct_answer_logprob"]
        row["correct_answer_logprob_delta"] = (
            row["correct_answer_logprob"] - initial["correct_answer_logprob"]
        )
        if row["mcqa_margin"] is not None:
            row["initial_mcqa_margin"] = initial["mcqa_margin"]
            row["mcqa_margin_delta"] = (
                row["mcqa_margin"] - initial["mcqa_margin"]
            )
    return final_rows


def summarize_audit(rows):
    def mean(key):
        values = [
            float(row[key]) for row in rows
            if row.get(key) is not None
        ]
        return sum(values) / len(values) if values else None

    deltas = [
        float(row["answer_logprob_delta"])
        for row in rows if "answer_logprob_delta" in row
    ]
    correct = [
        int(row["mcqa_correct"]) for row in rows
        if row.get("mcqa_correct") is not None
    ]
    return {
        "n": len(rows),
        "mean_answer_logprob": mean("answer_logprob"),
        "mean_correct_answer_logprob": mean("correct_answer_logprob"),
        "mean_initial_answer_logprob": mean("initial_answer_logprob"),
        "mean_answer_logprob_delta": (
            sum(deltas) / len(deltas) if deltas else None
        ),
        "min_answer_logprob_delta": min(deltas) if deltas else None,
        "max_answer_logprob_delta": max(deltas) if deltas else None,
        "mean_mcqa_margin": mean("mcqa_margin"),
        "mean_initial_mcqa_margin": mean("initial_mcqa_margin"),
        "mean_mcqa_margin_delta": mean("mcqa_margin_delta"),
        "mean_correct_answer_logprob_delta": mean("correct_answer_logprob_delta"),
        "mcqa_accuracy": (
            sum(correct) / len(correct) if correct else None
        ),
    }


def collect_group_audits(model, tok, groups):
    result = {}
    for name, items in groups.items():
        rows = collect_item_audit(model, tok, items)
        result[name] = {
            "summary": summarize_audit(rows),
            "items": rows,
        }
    return result


def collect_set_audit(model, tok, sets, qa):
    was_training = model.training
    model.eval()
    result = {}
    with torch.no_grad():
        for name, keys in sets.items():
            rows = []
            for key in keys:
                item = {
                    "id": key,
                    "source": name,
                    "question": qa[key]["q"],
                    "answer": qa[key]["a"],
                    "false_options": list(qa[key]["falses"]),
                    "path_count": 0,
                }
                scores = choice_scores(model, tok, item)
                wrong = max(scores[1:])
                rows.append({
                    "id": key,
                    "answer_logprob": answer_logprob(model, tok, item),
                    "correct_answer_logprob": scores[0],
                    "mcqa_scores": scores,
                    "mcqa_scoring": "legacy_full_sequence_mean_logprob",
                    "mcqa_margin": scores[0] - wrong,
                    "mcqa_correct": int(scores[0] > wrong),
                })
            result[name] = {
                "summary": {
                    "n": len(rows),
                    "mean_answer_logprob": (
                        sum(row["answer_logprob"] for row in rows)
                        / len(rows) if rows else None
                    ),
                    "mean_mcqa_margin": (
                        sum(row["mcqa_margin"] for row in rows)
                        / len(rows) if rows else None
                    ),
                    "mean_correct_answer_logprob": (
                        sum(row["correct_answer_logprob"] for row in rows)
                        / len(rows) if rows else None
                    ),
                    "mcqa_accuracy": (
                        sum(row["mcqa_correct"] for row in rows)
                        / len(rows) if rows else None
                    ),
                },
                "items": rows,
            }
    if was_training:
        model.train()
    return result


def summarize_step_rows(rows):
    anchor_count = sum(row["anchor_count"] for row in rows)
    raw_deltas = [
        delta for row in rows for delta in row["anchor_lp_delta_raw"]
    ]
    final_deltas = [
        delta for row in rows for delta in row["anchor_lp_delta_final"]
    ]
    return {
        "steps": len(rows),
        "anchor_observations": anchor_count,
        "raw_actual_drop_count": sum(delta < 0.0 for delta in raw_deltas),
        "final_actual_drop_count": sum(delta < 0.0 for delta in final_deltas),
        "raw_actual_risk_count": sum(
            delta < -ACTUAL_DROP_TOL for delta in raw_deltas
        ),
        "final_actual_risk_count": sum(
            delta < -ACTUAL_DROP_TOL for delta in final_deltas
        ),
        "raw_mean_anchor_lp_delta": (
            sum(raw_deltas) / len(raw_deltas) if raw_deltas else None
        ),
        "final_mean_anchor_lp_delta": (
            sum(final_deltas) / len(final_deltas) if final_deltas else None
        ),
        "feedback_rounds": sum(
            row["feedback_round_count"] for row in rows
        ),
        "mean_correction_to_raw_norm_ratio": (
            sum(row["correction_to_raw_norm_ratio"] for row in rows)
            / len(rows) if rows else None
        ),
        "max_correction_to_raw_norm_ratio": (
            max(row["correction_to_raw_norm_ratio"] for row in rows)
            if rows else None
        ),
    }


def forget_update(model, tok, params, opt, example, anchors, guide, mapping):
    """One forget-only AdamW proposal followed by the unchanged FB16 controller."""
    opt.zero_grad(set_to_none=True)
    loss = lp_masked(model, tok, example["question"], example["answer"])
    if loss is None:
        raise RuntimeError("An update has no answer tokens; the 114-step plan cannot skip it")
    finite_values([loss.detach()], "forget logprob")
    loss.backward()
    raw_gradients = clone_param_grads(params)
    guided, guide_step = apply_component_guidance(
        params, raw_gradients, mapping, guide["weights"], guide["applied"]
    )
    finite_values([guide_step["raw_norm"], guide_step["guided_norm"]], "gradient norms")
    lp_before = anchor_logprobs(model, tok, anchors)
    finite_values(lp_before, "pre-update anchor logprobs")
    anchor_grads = anchor_gradients(params, model, tok, anchors)
    for gradient in anchor_grads:
        finite_gradients(gradient, "retain-anchor gradient")
    for parameter in params:
        parameter.grad = guided[parameter]
    before = [parameter.detach().clone() for parameter in params]
    opt.step()
    finite_gradients({parameter: parameter.detach() for parameter in params}, "AdamW parameters")
    delta_raw = {
        parameter: parameter.detach().float() - old.float()
        for parameter, old in zip(params, before)
    }
    # The legacy helper's lp_after_raw return is measured after correction.
    # Capture the actual raw state here without changing that historical helper.
    lp_after_raw = anchor_logprobs(model, tok, anchors)
    feedback = feedback_update(
        params, model, tok, anchors, lp_before, before, delta_raw, anchor_grads
    )
    final_delta = {
        parameter: parameter.detach().float() - old.float()
        for parameter, old in zip(params, before)
    }
    lp_final = feedback["lp_after_final"]
    raw_deltas = [after - initial for initial, after in zip(lp_before, lp_after_raw)]
    final_deltas = [after - initial for initial, after in zip(lp_before, lp_final)]
    finite_values(raw_deltas + final_deltas + [feedback["final_ratio"]], "anchor diagnostics")
    if feedback["final_ratio"] > MAX_CORRECTION_RATIO + RATIO_EPS:
        raise RuntimeError("Feedback trust budget exceeded")
    if feedback["feedback_round_count"] > MAX_FEEDBACK_ROUNDS:
        raise RuntimeError("Feedback round budget exceeded")
    raw_dots = [scalar_map_dot(params, gradient, delta_raw) for gradient in anchor_grads]
    final_dots = [scalar_map_dot(params, gradient, final_delta) for gradient in anchor_grads]
    return {
        "forget_lp": float(loss.detach()), "anchor_count": len(anchors),
        "anchor_lp_before": lp_before,
        "anchor_lp_after_raw_adamw": lp_after_raw,
        "anchor_lp_after_feedback": lp_final,
        "anchor_lp_delta_raw": raw_deltas, "anchor_lp_delta_final": final_deltas,
        "raw_actual_risk": [value < -ACTUAL_DROP_TOL for value in raw_deltas],
        "final_actual_risk": [value < -ACTUAL_DROP_TOL for value in final_deltas],
        "raw_anchor_dot_delta": raw_dots, "final_anchor_dot_delta": final_dots,
        "aggregate_retain_delta_raw_dot": sum(raw_dots),
        "aggregate_retain_delta_final_dot": sum(final_dots),
        "raw_delta_norm": math.sqrt(max(scalar_map_norm2(params, delta_raw), 0.0)),
        "final_delta_norm": math.sqrt(max(scalar_map_norm2(params, final_delta), 0.0)),
        "correction_to_raw_norm_ratio": feedback["final_ratio"],
        "feedback_round_count": feedback["feedback_round_count"],
        "feedback_rounds": feedback["feedback_rounds"],
        "total_risk_seen": feedback["total_risk_seen"],
        "total_solver_cycles": feedback["total_solver_cycles"],
        "raw_forget_gradient_norm": guide_step["raw_norm"],
        "guided_forget_gradient_norm": guide_step["guided_norm"],
        "guidance_renorm_scale": guide_step["renorm_scale"],
        "guidance_fallback_to_raw": guide_step["fallback_to_raw"],
    }


def cumulative_snapshot(model, tok, items, initial, update):
    rows = []
    with torch.no_grad():
        for item in items:
            value = answer_logprob(model, tok, item)
            rows.append({
                "id": item["id"], "answer_logprob": value,
                "initial_answer_logprob": initial[item["id"]],
                "delta_from_initial": value - initial[item["id"]],
            })
    return {"update": update, "n": len(rows), "items": rows,
            "minimum_delta_from_initial": min(row["delta_from_initial"] for row in rows)}


def save_checkpoint(model, run_dir, update):
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(
        {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if "lora_" in name
        },
        checkpoint_dir / f"update_{update:03d}.pt",
    )


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def input_manifest():
    paths = [
        te.DATA,
        te.SPLITS / "forget_set_split.txt",
        te.SPLITS / "retaining_set_split.txt",
        te.SPLITS / "test_set_split.txt",
        OUT / "F1.json", OUT / "candidates.json", OUT / "active_paths.json",
        OUT / "attribution" / "items.json", OUT / "per_item_G2.jsonl",
    ]
    return {str(path.relative_to(WS)): sha256_file(path) for path in paths}


def capture_source(run_dir):
    paths = [
        Path(__file__), HERE / "diagnose_gsu_actual_feedback.py",
        HERE / "diagnose_gsu_retain_protection.py", HERE / "diagnose_gsu_m3.py",
        HERE / "train_gsu.py", HERE / "train_eval.py",
        HERE.parent / "damage" / "component_ablation.py",
        HERE.parent / "damage" / "eval_per_item.py",
        HERE / "tests" / "test_gsu_memory_guided.py",
        WS / "docs" / "method_versions" / "GSU-v6-memory-guided.md",
        WS / "exp" / "runs" / EXPERIMENT / "card.md",
    ]
    manifest = {}
    for source in paths:
        relative = source.resolve().relative_to(WS)
        destination = run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest[str(relative)] = sha256_file(destination)
    return manifest


def git_output(*arguments):
    result = subprocess.run(
        ["git", *arguments], cwd=WS, capture_output=True, check=True,
        encoding="utf-8", errors="replace", timeout=30,
    )
    return result.stdout


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.log.write(value)
        self.log.flush()
        return self.stream.write(value)

    def flush(self):
        self.log.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def run_logged(args):
    started_at = time.monotonic()
    run_dir = Path(args.run_dir).resolve()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)
    status = {"run_id": run_dir.name, "status": "running", "started": timestamp()}
    write_json(run_dir / "status.json", status)
    with open(run_dir / "stdout.log", "x", encoding="utf-8") as stdout_log, \
            open(run_dir / "stderr.log", "x", encoding="utf-8") as stderr_log, \
            contextlib.redirect_stdout(Tee(sys.stdout, stdout_log)), \
            contextlib.redirect_stderr(Tee(sys.stderr, stderr_log)):
        try:
            commit = git_output("rev-parse", "HEAD").strip()
            tree = git_output("status", "--short", "--untracked-files=all")
            (run_dir / "working_tree_status.txt").write_text(tree, encoding="utf-8")
            (run_dir / "patch.diff").write_text(
                git_output("diff", "--binary", "--no-ext-diff", "HEAD"), encoding="utf-8"
            )
            sources = capture_source(run_dir)
            inputs, plan = preflight(args.model or te.MODEL)
            plan.update({
                "experiment": EXPERIMENT, "method_version": METHOD,
                "run_id": run_dir.name, "variant": args.variant,
                "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
                "git_commit": commit, "working_tree": "dirty" if tree.strip() else "clean",
                "source_sha256": sources, "input_sha256": input_manifest(),
                "seed": SEED, "random_guide_seed": RANDOM_GUIDE_SEED,
                "lr": LR, "updates": TARGET_UPDATES, "reference_epochs": EPOCHS,
                "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                         "target_modules": TARGET_MODULES},
                "optimizer": {"name": "AdamW", "betas": [0.9, 0.999],
                              "eps": 1e-8, "weight_decay": 0.01},
                "feedback": {"anchors": N_ANCHOR, "actual_drop_tolerance": ACTUAL_DROP_TOL,
                             "max_rounds": MAX_FEEDBACK_ROUNDS,
                             "correction_ratio": MAX_CORRECTION_RATIO},
                "environment": {
                    "python": sys.version, "executable": sys.executable,
                    "platform": platform.platform(), "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "peft": __import__("peft").__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                },
                "model_files": {
                    path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                    for path in Path(plan["model"]).iterdir() if path.is_file()
                },
                "model_weight_hashing": "file sizes/mtimes only; config/tokenizer hashes stored separately",
                "model_metadata_sha256": {
                    path.name: sha256_file(path)
                    for path in Path(plan["model"]).iterdir()
                    if path.is_file() and path.suffix in (".json", ".txt", ".jinja")
                },
            })
            write_json(run_dir / "config.json", plan)
            (run_dir / "run_log.md").write_text(
                f"# Run Log: {run_dir.name}\n\nexp_id: {EXPERIMENT}\n"
                f"method_version: {METHOD}\nstarted: {status['started']}\n"
                "status: see status.json (includes failed/interrupted runs)\n\n"
                f"git_commit: {commit}\nworking_tree: {plan['working_tree']}\n\n"
                "## Launch\n\n" + plan["command"] + "\n\n"
                "## Reproducibility\n\nComplete pre-training config: config.json. "
                "Tracked dirty diff: patch.diff. Untracked executable sources and documents: "
                "source_snapshot/. Input and source SHA-256 values are frozen in config.json. "
                "Model weight payloads are identified by size/mtime, not full content hashes.\n\n"
                "## Outputs\n\nstdout.log, stderr.log, status.json, component_scores.json, "
                "schedule.json, anchor_schedule.json, diagnostics.jsonl, cumulative_audit.jsonl, "
                "initial_audits.json, initial_set_audits.json, metrics.json, checkpoints/, adapter/.\n\n"
                "No results were available at launch. Failed runs keep their traceback.\n",
                encoding="utf-8",
            )
            train_model(args, inputs, plan)
            status.update(status="completed", ended=timestamp())
        except BaseException as error:
            traceback.print_exc()
            status.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                ended=timestamp(), error=f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            status["wall_seconds"] = time.monotonic() - started_at
            if torch.cuda.is_initialized():
                status["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                status["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
            write_json(run_dir / "status.json", status, overwrite=True)


def self_test():
    fake_f0 = [
        {
            "id": f"f0:{i}",
            "question": f"q{i}",
            "answer": "a",
            "false_options": ["x", "y"],
            "source": "F0",
            "path_count": 0,
        }
        for i in range(33)
    ]
    fake_f1 = [
        {
            "id": f"f1:{i}",
            "question": f"m{i}",
            "answer": "a",
            "false_options": ["x", "y"],
            "source": "F1",
            "path_count": 1,
        }
        for i in range(5)
    ]
    schedule_f0 = make_update_schedule(fake_f0, TARGET_UPDATES, SEED)
    schedule_f1 = make_update_schedule(
        fake_f0 + fake_f1, TARGET_UPDATES, SEED
    )
    assert len(schedule_f0) == TARGET_UPDATES
    assert len(schedule_f1) == TARGET_UPDATES
    counts_f0 = defaultdict(int)
    counts_f1 = defaultdict(int)
    for row in schedule_f0:
        counts_f0[row["item_id"]] += 1
    for row in schedule_f1:
        counts_f1[row["item_id"]] += 1
    assert min(counts_f0.values()) == 3
    assert max(counts_f0.values()) == 4
    assert set(counts_f1.values()) == {3}

    fake_rows = [
        {
            "component": "L00.mlp.up_proj",
            "shared_score_S": 1.0,
        },
        {
            "component": "L00.mlp.down_proj",
            "shared_score_S": 0.5,
        },
        {
            "component": "L00.self_attn.q_proj",
            "shared_score_S": 0.0,
        },
        {
            "component": "L01.mlp.up_proj",
            "shared_score_S": 0.0,
        },
    ]
    guide = build_component_guide(fake_rows, "A3")
    assert guide["structured_hard_kept_components"] == 2
    parameter_a = torch.nn.Parameter(torch.zeros(2))
    parameter_b = torch.nn.Parameter(torch.zeros(2))
    parameters = [parameter_a, parameter_b]
    raw = {
        parameter_a: torch.tensor([3.0, 4.0]),
        parameter_b: torch.tensor([0.0, 4.0]),
    }
    mapping = {
        parameter_a: "L00.mlp.up_proj",
        parameter_b: "L00.mlp.down_proj",
    }
    weighted, info = apply_component_guidance(
        parameters,
        raw,
        mapping,
        {
            "L00.mlp.up_proj": 1.0,
            "L00.mlp.down_proj": 0.5,
        },
        True,
    )
    assert abs(info["guided_norm"] - info["raw_norm"]) < 1e-5
    assert torch.allclose(weighted[parameter_a], raw[parameter_a] * info["renorm_scale"])
    print(json.dumps({
        "self_test": "passed",
        "f0_schedule_updates": len(schedule_f0),
        "f1_schedule_updates": len(schedule_f1),
        "hard_kept": guide["structured_hard_kept_components"],
    }, ensure_ascii=False))


def train_model(args, inputs, plan):
    run_dir = Path(args.run_dir).resolve()

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()

    model_path = plan["model"]

    data_path = te.DATA
    split_dir = te.SPLITS
    clusters = inputs["clusters"]
    forget_idx = inputs["indices"]["forget"]
    all_retain_idx = inputs["indices"]["retain"]
    retain_idx = all_retain_idx[:33]

    f1_path = OUT / "F1.json"
    candidate_path = OUT / "candidates.json"
    active_path = OUT / "active_paths.json"
    f0_items, f1_items = inputs["F0"], inputs["F1"]
    retain_items_all = inputs["retain_all"]
    if args.variant in F0_ONLY:
        forget_items = f0_items
        schedule_source = "F0_fixed_repeat_plan"
    else:
        forget_items = f0_items + f1_items
        schedule_source = "F0_plus_F1_three_blocks"
    schedule = make_update_schedule(
        forget_items, TARGET_UPDATES, SEED
    )
    schedule_info = schedule_manifest(
        schedule, forget_items, schedule_source
    )

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    ).eval()
    for parameter in base.parameters():
        parameter.requires_grad = False
    model = get_peft_model(
        base,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )
    model.train()
    params, groups, parameter_to_component, parameter_names = (
        collect_component_groups(model)
    )
    del parameter_names
    if len(groups) != plan["component_count"]:
        raise RuntimeError(
            f"Expected {plan['component_count']} LoRA components, "
            f"got {len(groups)}"
        )

    initial_parameter_hash = parameter_digest(model)
    score_rows, score_meta = compute_component_scores(
        model, tok, f0_items, f1_items, params, groups
    )
    if parameter_digest(model) != initial_parameter_hash:
        raise RuntimeError("Frozen component scoring mutated model parameters")
    base_scores_hash = stable_hash(score_rows)
    guide = build_component_guide(score_rows, args.variant)

    config = {
        "experiment": "exp006_gsu_memory_guided",
        "variant": args.variant,
        "parent_method": "GSU-v5-actual-feedback",
        "model": model_path,
        "initial_lora_sha256": initial_parameter_hash,
        "base_scores_sha256": base_scores_hash,
        "data": {
            "dataset": str(data_path),
            "dataset_sha256": sha256_file(data_path),
            "forget_split": str(split_dir / "forget_set_split.txt"),
            "retain_split": str(split_dir / "retaining_set_split.txt"),
            "f1_path": str(f1_path),
            "candidate_path": str(candidate_path),
            "active_path": str(active_path),
        },
        "seed": SEED,
        "training": {
            "lr": LR,
            "epochs_reference": EPOCHS,
            "target_updates": TARGET_UPDATES,
            "forget_examples": len(forget_items),
            "f0_examples": len(f0_items),
            "f1_examples": len(f1_items),
            "retain_anchor_pool": len(retain_idx),
            "retain_audit_count": len(retain_items_all),
            "lora": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_modules": TARGET_MODULES,
                "component_count": len(groups),
            },
        },
        "feedback": {
            "anchors_per_update": N_ANCHOR,
            "actual_drop_tolerance": ACTUAL_DROP_TOL,
            "max_feedback_rounds": MAX_FEEDBACK_ROUNDS,
            "max_correction_to_raw_norm_ratio": MAX_CORRECTION_RATIO,
            "optimizer_moments": "unchanged by post-hoc correction",
        },
        "component_guidance": guide,
        "score_meta": score_meta,
        "schedule": schedule_info,
        "training_boundary": (
            "F0-only variants use the fixed 114-update schedule with 99 "
            "updates from three complete F0 passes and 15 entries from a "
            "fourth seeded permutation. F0+F1 variants use 3 complete "
            "38-example passes."
        ),
        "memory_boundary": (
            "F1 direct supporting-fact QA and active-path IDs are used for "
            "frozen component scoring. Hop2 evaluation question/answer text "
            "is not used as a training target."
        ),
        "code": {
            "script": str(Path(__file__).resolve()),
            "feedback_helper": str(
                (HERE / "diagnose_gsu_actual_feedback.py").resolve()
            ),
        },
    }
    write_json(run_dir / "resolved_config.json", config)
    write_json(run_dir / "component_scores.json", {
        "variant": args.variant,
        "score_meta": score_meta,
        "guide": guide,
        "components": score_rows,
    })
    write_json(run_dir / "schedule.json", schedule_info)

    audit_groups = {
        "F0": f0_items,
        "F1": f1_items,
        "associated_memory": f1_items,
        "retain_all": retain_items_all,
        "retain_train": retain_items_all[:33],
        "retain_heldout": retain_items_all[33:],
    }
    initial_audits = collect_group_audits(model, tok, audit_groups)
    write_json(run_dir / "initial_audits.json", initial_audits)
    initial_set_audits = collect_set_audit(model, tok, inputs["sets"], inputs["qa"])
    write_json(run_dir / "initial_set_audits.json", initial_set_audits)

    opt = torch.optim.AdamW(params, lr=LR)
    anchor_items = retain_items_all[:33]
    retain_ex = [
        (item["question"], item["answer"])
        for item in anchor_items
    ]
    anchor_schedule = make_anchor_schedule(len(retain_ex))
    write_json(run_dir / "anchor_schedule.json", {
        "seed": SEED, "anchor_pool_ids": [item["id"] for item in anchor_items],
        "pool_indices_by_update": anchor_schedule,
    })
    anchor_initial = {
        row["id"]: row["answer_logprob"]
        for row in initial_audits["retain_all"]["items"][:33]
    }
    diagnostics_path = run_dir / "diagnostics.jsonl"
    block_rows = defaultdict(list)
    all_anchor_ids = set()
    total_updates = 0

    cumulative_path = run_dir / "cumulative_audit.jsonl"
    with open(diagnostics_path, "x", encoding="utf-8") as diagnostics, \
            open(cumulative_path, "x", encoding="utf-8") as cumulative_log:
        for planned in schedule:
            pool_indices = anchor_schedule[planned["update"] - 1]
            anchors = [retain_ex[index] for index in pool_indices]
            anchor_ids = [anchor_items[index]["id"] for index in pool_indices]
            all_anchor_ids.update(anchor_ids)
            update_row = forget_update(
                model, tok, params, opt, planned, anchors, guide, parameter_to_component
            )
            row = {
                **update_row,
                "variant": args.variant,
                "update": planned["update"],
                "schedule_block": planned["schedule_block"],
                "forget_example_id": planned["item_id"],
                "forget_source": planned["source"],
                "forget_question_sha1": planned["question_sha1"],
                "anchor_ids": anchor_ids,
                "anchor_question_sha1": [sha1_text(question) for question, _ in anchors],
                "anchor_initial_lp": [anchor_initial[key] for key in anchor_ids],
                "anchor_cumulative_lp_delta_final": [
                    value - anchor_initial[key]
                    for key, value in zip(anchor_ids, update_row["anchor_lp_after_feedback"])
                ],
            }
            diagnostics.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            diagnostics.flush()
            block_rows[planned["schedule_block"]].append(row)
            total_updates += 1
            if total_updates in (38, 76, TARGET_UPDATES):
                save_checkpoint(model, run_dir, total_updates)
                snapshot = cumulative_snapshot(model, tok, anchor_items, anchor_initial, total_updates)
                cumulative_log.write(json.dumps(snapshot, ensure_ascii=False, allow_nan=False) + "\n")
                cumulative_log.flush()
                print(json.dumps({
                    "variant": args.variant,
                    "update": total_updates,
                    "block": planned["schedule_block"],
                    "raw_risk_count": sum(
                        sum(item["raw_actual_risk"])
                        for item in block_rows[planned["schedule_block"]]
                    ),
                    "final_risk_count": sum(
                        sum(item["final_actual_risk"])
                        for item in block_rows[planned["schedule_block"]]
                    ),
                }, ensure_ascii=False), flush=True)

    if total_updates != TARGET_UPDATES:
        raise RuntimeError(
            f"Training stopped at {total_updates}/{TARGET_UPDATES} updates"
        )
    if len(all_anchor_ids) != 33:
        raise RuntimeError("The frozen anchor schedule did not cover all 33 anchors")

    block_metrics = {
        str(block): summarize_step_rows(rows)
        for block, rows in sorted(block_rows.items())
    }
    write_json(run_dir / "block_metrics.json", block_metrics)
    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir)

    model.eval()
    five = te.evaluate(model, tok, clusters, forget_idx, retain_idx)
    sets, qa = inputs["sets"], inputs["qa"]
    six = ca.eval_sets(model, tok, sets, qa)
    six_serialized = {
        key: {
            "n": int(value["n"]),
            "mean_lp": float(value["mean_lp"]),
            "mean_correct": float(value["mean_correct"]),
        }
        for key, value in six.items()
    }
    final_audits = collect_group_audits(model, tok, audit_groups)
    for name in final_audits:
        final_audits[name]["items"] = attach_initial_deltas(
            initial_audits[name]["items"],
            final_audits[name]["items"],
        )
        final_audits[name]["summary"] = summarize_audit(
            final_audits[name]["items"]
        )
    set_audits = collect_set_audit(model, tok, sets, qa)
    for name, audit in set_audits.items():
        attach_initial_deltas(initial_set_audits[name]["items"], audit["items"])
        if len(audit["items"]) != EXPECTED_SET_COUNTS[name]:
            raise RuntimeError(f"Final audit count mismatch: {name}")
    metrics = {
        "five_set": {"group": f"GSU_{args.variant}_DIAG", **five},
        "six_set": {
            "group": f"GSU_{args.variant}_DIAG",
            "sets": six_serialized,
        },
        "logprob_audits": final_audits,
        "set_logprob_audits": set_audits,
        "diagnostics": {
            "variant": args.variant,
            "updates": total_updates,
            "unique_anchor_count": len(all_anchor_ids),
            "anchor_pool_count": len(retain_idx),
            "anchor_observations": total_updates * N_ANCHOR,
            "blocks": block_metrics,
            "raw_jsonl": str(diagnostics_path),
            "component_scores": str(run_dir / "component_scores.json"),
            "initial_audits": str(run_dir / "initial_audits.json"),
            "cumulative_audit": str(cumulative_path),
            "base_scores_sha256": base_scores_hash,
            "initial_lora_sha256": initial_parameter_hash,
            "anchor_schedule_sha256": stable_hash(anchor_schedule),
            "cumulative_retain": {
                "worst_final_delta": min(
                    row["answer_logprob_delta"]
                    for row in final_audits["retain_train"]["items"]
                ),
                "per_anchor": final_audits["retain_train"]["items"],
            },
            "algorithm": (
                "F0-only or F0+F1 forget gradient; optional frozen "
                "LoRA-component guidance; 16 fresh anchors per update; "
                "unchanged FB16 actual logprob feedback."
            ),
            "feedback_constants": {
                "actual_drop_tolerance": ACTUAL_DROP_TOL,
                "max_feedback_rounds": MAX_FEEDBACK_ROUNDS,
                "max_correction_to_raw_norm_ratio": MAX_CORRECTION_RATIO,
            },
            "implicit_f1_audit": (
                "F0-only variants are audited on direct F1 answer logprob, "
                "MCQA margin and F1_hop2; these values are not training "
                "targets for A0/A2/A3/A2R/A3R."
            ),
        },
    }
    write_json(run_dir / "metrics.json", metrics)
    print(json.dumps({
        "saved": str(run_dir / "metrics.json"),
        "variant": args.variant,
        "updates": total_updates,
        "unique_anchor_count": len(all_anchor_ids),
        "primary": six_serialized["D_hop2_no_F1_overlap"],
        "retain": five.get("retain"),
    }, ensure_ascii=False), flush=True)


def compare_runs(run_paths, comparison_dir):
    """Read completed runs only; never start training or rewrite source runs."""
    records = {}
    common_signatures = set()
    score_signatures, lora_signatures, anchor_signatures = set(), set(), set()
    schedule_signatures = defaultdict(set)
    sources = []
    for value in run_paths:
        path = Path(value).resolve()
        if read_json(path / "status.json")["status"] != "completed":
            raise RuntimeError(f"Cannot compare an incomplete run: {path}")
        config = read_json(path / "config.json")
        resolved = read_json(path / "resolved_config.json")
        metrics = read_json(path / "metrics.json")
        variant = config["variant"]
        if variant not in VARIANTS or variant in records:
            raise ValueError(f"Unknown or duplicate variant: {variant}")
        common_signatures.add(stable_hash({
            key: config[key] for key in (
                "model", "seed", "lr", "updates", "lora", "optimizer", "feedback",
                "input_sha256", "source_sha256", "model_metadata_sha256", "model_files",
            )
        }))
        if metrics["diagnostics"]["updates"] != TARGET_UPDATES:
            raise RuntimeError(f"Invalid update budget: {path}")
        if metrics["diagnostics"]["unique_anchor_count"] != 33:
            raise RuntimeError(f"Incomplete anchor coverage: {path}")
        sets = metrics["six_set"]["sets"]
        if {key: row["n"] for key, row in sets.items()} != EXPECTED_SET_COUNTS:
            raise RuntimeError(f"Invalid final evaluation counts: {path}")
        score_signatures.add(resolved["base_scores_sha256"])
        lora_signatures.add(resolved["initial_lora_sha256"])
        anchor_signatures.add(metrics["diagnostics"]["anchor_schedule_sha256"])
        schedule_signatures["F0" if variant in F0_ONLY else "F0+F1"].add(
            stable_hash(read_json(path / "schedule.json"))
        )
        records[variant] = metrics
        sources.append({"variant": variant, "run_dir": str(path),
                        "metrics_sha256": sha256_file(path / "metrics.json")})
    if any(len(values) != 1 for values in (
        common_signatures, score_signatures, lora_signatures, anchor_signatures,
        *schedule_signatures.values(),
    )):
        raise RuntimeError("Runs do not share frozen inputs, initialization, scores or schedules")
    if not {"A0", "A1"}.issubset(records):
        raise RuntimeError("Both fresh equal-budget A0 and A1 baselines are required")

    rows = []
    for variant in VARIANTS:
        if variant not in records:
            continue
        metrics = records[variant]
        sets = metrics["six_set"]["sets"]
        f1 = metrics["logprob_audits"]["F1"]["summary"]
        rows.append({
            "variant": variant,
            "primary_correct": round(sets["D_hop2_no_F1_overlap"]["mean_correct"] * 12),
            "retain_correct": round(sets["retain"]["mean_correct"] * 33),
            "F1_hop2_correct": round(sets["F1_hop2"]["mean_correct"] * 75),
            "F1_direct_correct": round(f1["mcqa_accuracy"] * 5),
            "F1_mean_answer_logprob": f1["mean_answer_logprob"],
            "F1_mean_answer_logprob_delta": f1["mean_answer_logprob_delta"],
            "F1_worst_answer_logprob_delta": f1["min_answer_logprob_delta"],
            "F1_mean_mcqa_margin": f1["mean_mcqa_margin"],
            "hop1_unlearn": sets["hop1_unlearn"]["mean_correct"],
            "K": sets["K"]["mean_correct"], "same_object": metrics["five_set"]["same_object"],
            "hop2": metrics["five_set"]["hop2"],
            "retain_heldout": metrics["logprob_audits"]["retain_heldout"]["summary"],
            "worst_final_anchor_lp_delta": metrics["diagnostics"]["cumulative_retain"]["worst_final_delta"],
        })
    indexed = {row["variant"]: row for row in rows}

    def directional_gate(row):
        return {
            "F1_hop2_below_A0": row["F1_hop2_correct"] < indexed["A0"]["F1_hop2_correct"],
            "primary_not_below_A1": row["primary_correct"] >= indexed["A1"]["primary_correct"],
            "retain_not_below_A1": row["retain_correct"] >= indexed["A1"]["retain_correct"],
            "F1_direct_correct_not_below_A0": row["F1_direct_correct"] >= indexed["A0"]["F1_direct_correct"],
            "F1_direct_logprob_not_below_A0": row["F1_mean_answer_logprob"] >= indexed["A0"]["F1_mean_answer_logprob"],
        }

    for variant in ("A2", "A3", "A2R", "A3R"):
        if variant in indexed:
            gate = directional_gate(indexed[variant])
            indexed[variant]["directional_gate"] = gate
            indexed[variant]["directional_gate_passed"] = all(gate.values())
    for candidate, control in (("A2", "A2R"), ("A3", "A3R")):
        if candidate in indexed:
            indexed[candidate]["random_control_reaches_same_gate"] = (
                indexed[control]["directional_gate_passed"] if control in indexed else None
            )
    output = {
        "experiment": EXPERIMENT, "created": timestamp(), "sources": sources,
        "rows": rows, "missing_variants": [key for key in VARIANTS if key not in indexed],
        "complete_screen": set(records) == set(VARIANTS),
        "interpretation": "Single-seed directional screen only; no significance or automatic advancement claim",
    }
    comparison_dir = comparison_dir.resolve()
    comparison_dir.parent.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(exist_ok=False)
    write_json(comparison_dir / "comparison.json", output)
    table = [
        "# exp006 Side-by-Side Screen", "",
        "Single-seed observations only. Missing variants: " + ", ".join(output["missing_variants"]), "",
        "| Method | Primary /12 | Retain /33 | F1 hop2 /75 | F1 direct /5 | F1 answer LP delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['variant']} | {row['primary_correct']} | {row['retain_correct']} | "
            f"{row['F1_hop2_correct']} | {row['F1_direct_correct']} | "
            f"{row['F1_mean_answer_logprob_delta']:.6f} |"
        )
    table.extend(["", "Raw per-run numbers, hashes, held-out retain and directional gates: comparison.json.",
                  "A random control passing the same gate does not support a shared-memory mechanism."])
    (comparison_dir / "comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(comparison_dir), "complete_screen": output["complete_screen"]}))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--run_dir")
    parser.add_argument("--model", default=None)
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--preflight", action="store_true",
                        help="Validate local data, model metadata and schedules without loading weights")
    parser.add_argument("--compare", nargs="+", metavar="RUN_DIR")
    parser.add_argument("--comparison_dir", help="New directory for an immutable side-by-side report")
    args = parser.parse_args()
    if sum((args.self_test, args.preflight, bool(args.compare))) > 1:
        parser.error("Choose only one of --self_test, --preflight or --compare")
    if args.self_test:
        self_test()
        return
    if args.preflight:
        _, plan = preflight(args.model or te.MODEL)
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.compare:
        if not args.comparison_dir:
            parser.error("--compare requires a new --comparison_dir")
        compare_runs(args.compare, Path(args.comparison_dir))
        return
    if args.variant is None or args.run_dir is None:
        parser.error(
            "--variant and --run_dir are required unless --self_test is used"
        )
    run_logged(args)


if __name__ == "__main__":
    main()
