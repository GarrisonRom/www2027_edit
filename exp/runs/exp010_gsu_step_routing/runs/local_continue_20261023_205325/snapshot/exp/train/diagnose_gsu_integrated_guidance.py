# -*- coding: utf-8 -*-
"""GSU-v7: historical damage prior x current F0/F1 gradient/path guidance.

This is an isolated exp007 entry point. It reuses the audited exp006 trainer
and FB16 controller, but replaces the component score with:

    integrated_S(j) = current_gradient_path_S(j) * historical_ffn_prior(j)

`historical_ffn_prior` is derived only from the frozen G1/G2 per-layer FFN
damage profiles. It is applied to MLP LoRA components. Attention components
receive a neutral prior of one because the existing attribution artifacts do
not provide a defensible q/k/v/o component-level damage estimate.

No historical result is used as a training target. The historical files only
provide a frozen routing prior and are copied into each run manifest.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.path.insert(0, str(HERE))
# Isolated runtime: configuring B variants must not mutate the imported A trainer.
_spec = importlib.util.spec_from_file_location(
    "gsu_exp007_runtime", HERE / "diagnose_gsu_memory_guided.py"
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

_ORIGINAL_COMPUTE_COMPONENT_SCORES = base.compute_component_scores
_ORIGINAL_BUILD_COMPONENT_GUIDE = base.build_component_guide
_ORIGINAL_WRITE_JSON = base.write_json
_ORIGINAL_INPUT_MANIFEST = base.input_manifest
_LAST_DAMAGE_META = None


EXPERIMENT = "exp007_gsu_integrated_guidance"
METHOD = "GSU-v7-integrated-damage-gradient-path"
HISTORICAL_FILES = (
    WS / "exp" / "results" / "attribution" / "ffn_G1.json",
    WS / "exp" / "results" / "attribution" / "ffn_G2.json",
    WS / "exp" / "results" / "attribution" / "stats_stage2.json",
    WS / "exp" / "results" / "attribution" / "ablation_G2.jsonl",
)
VARIANTS = ("B0", "B1", "B2", "B3", "B2R", "B3R")
F0_ONLY = ("B0", "B2", "B3", "B2R", "B3R")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_historical_damage_prior():
    """Return a frozen layer prior from G1/G2 FFN damage profiles.

    The geometric mean rewards layers that are high in both historical G1 and
    G2 damage profiles. The normalization is within each frozen profile and
    has no learned or searched temperature.
    """
    profiles = {}
    for name, path in (("G1", HISTORICAL_FILES[0]), ("G2", HISTORICAL_FILES[1])):
        rows = read_json(path)["per_layer"]
        values = {int(row["layer"]): float(row["D"]) for row in rows}
        if (len(rows) != 28 or set(values) != set(range(28)) or
                any(not math.isfinite(v) or v < 0.0 for v in values.values())):
            raise RuntimeError(f"Invalid historical FFN profile: {path}")
        maximum = max(values.values())
        if maximum <= 0.0:
            raise RuntimeError(f"Historical FFN profile is all zero: {path}")
        profiles[name] = {layer: value / maximum for layer, value in values.items()}
    prior = {
        layer: math.sqrt(profiles["G1"][layer] * profiles["G2"][layer])
        for layer in range(28)
    }
    return prior, {
        "formula": "sqrt(normalized_ffn_D_G1_layer * normalized_ffn_D_G2_layer)",
        "profiles": profiles,
        "prior": prior,
        "mlp_only": True,
        "attention_prior": 1.0,
        "source_files": {str(path.relative_to(WS)): sha256_file(path) for path in HISTORICAL_FILES},
    }


def component_integrated_scores(model, tok, f0_items, memory_items, params, groups):
    global _LAST_DAMAGE_META
    rows, meta = _ORIGINAL_COMPUTE_COMPONENT_SCORES(
        model, tok, f0_items, memory_items, params, groups
    )
    layer_prior, damage_meta = load_historical_damage_prior()
    for row in rows:
        component = row["component"]
        layer = int(component[1:3])
        is_mlp = ".mlp." in component
        damage_prior = layer_prior[layer] if is_mlp else 1.0
        gradient_path_score = float(row["shared_score_S"])
        row["historical_damage_prior"] = damage_prior
        row["historical_damage_scope"] = "ffn_layer" if is_mlp else "neutral_attention"
        row["gradient_path_score_S"] = gradient_path_score
        row["integrated_score_S"] = gradient_path_score * damage_prior
        row["shared_score_S"] = row["integrated_score_S"]
    meta = dict(meta)
    meta["integrated_score"] = damage_meta
    _LAST_DAMAGE_META = damage_meta
    meta["formula"] = {
        "gradient_path": "A_j * max(P_j, 0)",
        "historical_damage": damage_meta["formula"],
        "integrated": "gradient_path_S_j * historical_ffn_prior_layer(j) for MLP; gradient_path_S_j for attention",
    }
    return rows, meta


def integrated_component_guide(score_rows, variant):
    mapping = {
        "B0": "A0", "B1": "A1", "B2": "A2", "B3": "A3",
        "B2R": "A2R", "B3R": "A3R",
    }
    if variant not in mapping:
        raise ValueError(f"Unknown exp007 variant: {variant}")
    previous_variants = base.VARIANTS
    try:
        base.VARIANTS = ("A0", "A1", "A2", "A3", "A4", "A2R", "A3R")
        guide = _ORIGINAL_BUILD_COMPONENT_GUIDE(score_rows, mapping[variant])
    finally:
        base.VARIANTS = previous_variants
    guide = dict(guide)
    guide["variant"] = variant
    guide["score_source"] = "historical_ffn_damage_x_current_gradient_path"
    guide["historical_damage_prior_scope"] = "MLP layer prior; attention neutral"
    return guide


def input_manifest():
    manifest = _ORIGINAL_INPUT_MANIFEST()
    manifest.update({
        str(path.relative_to(WS)): sha256_file(path)
        for path in HISTORICAL_FILES
    })
    return manifest


def capture_source(run_dir):
    paths = [
        Path(__file__), HERE / "diagnose_gsu_memory_guided.py",
        HERE / "start_gsu_integrated_guidance.ps1",
        HERE / "tests" / "test_gsu_integrated_guidance.py",
        HERE / "diagnose_gsu_actual_feedback.py",
        HERE / "diagnose_gsu_retain_protection.py",
        HERE / "diagnose_gsu_m3.py", HERE / "train_gsu.py",
        HERE / "train_eval.py", HERE.parent / "damage" / "component_ablation.py",
        HERE.parent / "damage" / "eval_per_item.py",
        WS / "docs" / "method_versions" / "GSU-v7-integrated-damage-gradient-path.md",
        WS / "exp" / "runs" / EXPERIMENT / "card.md",
        WS / "exp" / "runs" / EXPERIMENT / "README.md",
        *HISTORICAL_FILES,
    ]
    manifest = {}
    for source in paths:
        relative = source.resolve().relative_to(WS)
        destination = run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest[str(relative)] = sha256_file(destination)
    return manifest


def patched_write_json(path, value, overwrite=False):
    if Path(path).name == "metrics.json" and _LAST_DAMAGE_META is not None:
        value = copy.deepcopy(value)
        value.setdefault("diagnostics", {})["historical_damage_prior"] = _LAST_DAMAGE_META
    if Path(path).name == "resolved_config.json":
        value = copy.deepcopy(value)
        value["experiment"] = EXPERIMENT
        value["method_version"] = METHOD
        value["parent_method"] = "GSU-v6-memory-guided"
        value["integrated_guidance"] = {
            "historical_sources": [str(path.relative_to(WS)) for path in HISTORICAL_FILES],
            "historical_scope": "G1/G2 FFN per-layer damage prior; neutral attention prior",
            "formula": "integrated S = gradient-path S x historical FFN layer prior",
        }
    return _ORIGINAL_WRITE_JSON(path, value, overwrite=overwrite)


def configure_base():
    base.VARIANTS = VARIANTS
    base.F0_ONLY = F0_ONLY
    base.EXPERIMENT = EXPERIMENT
    base.METHOD = METHOD
    base.compute_component_scores = component_integrated_scores
    base.build_component_guide = integrated_component_guide
    base.input_manifest = input_manifest
    base.capture_source = capture_source
    base.write_json = patched_write_json


def self_test():
    prior, meta = load_historical_damage_prior()
    assert len(prior) == 28
    assert max(prior.values()) == 1.0
    rows = [
        {"component": "L00.mlp.up_proj", "shared_score_S": 1.0},
        {"component": "L27.mlp.down_proj", "shared_score_S": 1.0},
        {"component": "L27.self_attn.q_proj", "shared_score_S": 1.0},
    ]
    integrated, _ = component_integrated_score_rows_for_test(rows, prior)
    assert integrated[0]["integrated_score_S"] < integrated[1]["integrated_score_S"]
    assert integrated[2]["historical_damage_prior"] == 1.0
    for variant in VARIANTS:
        guide = integrated_component_guide(integrated, variant)
        assert guide["variant"] == variant
    print(json.dumps({
        "self_test": "passed", "variants": VARIANTS,
        "historical_prior_min": min(prior.values()),
        "historical_prior_max": max(prior.values()),
        "historical_sources": meta["source_files"],
    }, ensure_ascii=False))


def component_integrated_score_rows_for_test(rows, prior):
    result = []
    for source in rows:
        row = dict(source)
        layer = int(row["component"][1:3])
        is_mlp = ".mlp." in row["component"]
        damage_prior = prior[layer] if is_mlp else 1.0
        row["historical_damage_prior"] = damage_prior
        row["gradient_path_score_S"] = row["shared_score_S"]
        row["integrated_score_S"] = row["shared_score_S"] * damage_prior
        row["shared_score_S"] = row["integrated_score_S"]
        result.append(row)
    return result, {}


def compare_runs(run_paths, comparison_dir):
    """Compare only complete exp007 runs; do not mutate any source run."""
    records = {}
    common_signatures = set()
    score_signatures, lora_signatures, anchor_signatures = set(), set(), set()
    schedule_signatures = {}
    sources = []
    for value in run_paths:
        path = Path(value).resolve()
        if base.read_json(path / "status.json")["status"] != "completed":
            raise RuntimeError(f"Cannot compare an incomplete run: {path}")
        config = base.read_json(path / "config.json")
        resolved = base.read_json(path / "resolved_config.json")
        metrics = base.read_json(path / "metrics.json")
        variant = config["variant"]
        if variant not in VARIANTS or variant in records:
            raise ValueError(f"Unknown or duplicate exp007 variant: {variant}")
        common_signatures.add(base.stable_hash({
            key: config[key] for key in (
                "model", "seed", "lr", "updates", "lora", "optimizer", "feedback",
                "input_sha256", "source_sha256", "model_metadata_sha256", "model_files",
            )
        }))
        if metrics["diagnostics"]["updates"] != base.TARGET_UPDATES:
            raise RuntimeError(f"Invalid update budget: {path}")
        if metrics["diagnostics"]["unique_anchor_count"] != 33:
            raise RuntimeError(f"Incomplete anchor coverage: {path}")
        sets = metrics["six_set"]["sets"]
        if {key: row["n"] for key, row in sets.items()} != base.EXPECTED_SET_COUNTS:
            raise RuntimeError(f"Invalid evaluation counts: {path}")
        score_signatures.add(resolved["base_scores_sha256"])
        lora_signatures.add(resolved["initial_lora_sha256"])
        anchor_signatures.add(metrics["diagnostics"]["anchor_schedule_sha256"])
        family = "F0" if variant in F0_ONLY else "F0+F1"
        schedule_signatures.setdefault(family, set()).add(
            base.stable_hash(base.read_json(path / "schedule.json"))
        )
        records[variant] = metrics
        sources.append({
            "variant": variant, "run_dir": str(path),
            "metrics_sha256": sha256_file(path / "metrics.json"),
            "config_sha256": sha256_file(path / "resolved_config.json"),
        })
    signatures = [common_signatures, score_signatures, lora_signatures, anchor_signatures]
    signatures.extend(schedule_signatures.values())
    if any(len(signature) != 1 for signature in signatures):
        raise RuntimeError("Runs differ in frozen code/data/init/score/schedule inputs")
    if not {"B0", "B1"}.issubset(records):
        raise RuntimeError("B0 and B1 fresh equal-budget baselines are required")

    rows = []
    for variant in VARIANTS:
        if variant not in records:
            continue
        metrics = records[variant]
        six = metrics["six_set"]["sets"]
        f1 = metrics["logprob_audits"]["F1"]["summary"]
        rows.append({
            "variant": variant,
            "primary_correct": round(six["D_hop2_no_F1_overlap"]["mean_correct"] * 12),
            "retain_correct": round(six["retain"]["mean_correct"] * 33),
            "F1_hop2_correct": round(six["F1_hop2"]["mean_correct"] * 75),
            "F1_direct_correct": round(f1["mcqa_accuracy"] * 5),
            "F1_mean_answer_logprob": f1["mean_answer_logprob"],
            "F1_mean_answer_logprob_delta": f1["mean_answer_logprob_delta"],
            "F1_worst_answer_logprob_delta": f1["min_answer_logprob_delta"],
            "F1_mean_mcqa_margin": f1["mean_mcqa_margin"],
            "hop1_unlearn_correct_rate": six["hop1_unlearn"]["mean_correct"],
            "K_correct_rate": six["K"]["mean_correct"],
            "same_object_correct_rate": metrics["five_set"]["same_object"],
            "hop2_correct_rate": metrics["five_set"]["hop2"],
            "retain_heldout": metrics["logprob_audits"]["retain_heldout"]["summary"],
            "worst_final_anchor_lp_delta": metrics["diagnostics"]["cumulative_retain"]["worst_final_delta"],
            "historical_damage_prior": metrics["diagnostics"].get("historical_damage_prior"),
        })
    indexed = {row["variant"]: row for row in rows}

    def gate(row):
        return {
            "F1_hop2_below_B0": row["F1_hop2_correct"] < indexed["B0"]["F1_hop2_correct"],
            "primary_not_below_B1": row["primary_correct"] >= indexed["B1"]["primary_correct"],
            "F1_direct_correct_not_below_B0": row["F1_direct_correct"] >= indexed["B0"]["F1_direct_correct"],
            "F1_answer_logprob_not_below_B0": row["F1_mean_answer_logprob"] >= indexed["B0"]["F1_mean_answer_logprob"],
        }

    for variant in ("B2", "B3", "B2R", "B3R"):
        if variant in indexed:
            checks = gate(indexed[variant])
            indexed[variant]["directional_gate"] = checks
            indexed[variant]["directional_gate_passed"] = all(checks.values())
    for candidate, control in (("B2", "B2R"), ("B3", "B3R")):
        if candidate in indexed:
            indexed[candidate]["random_control_reaches_same_gate"] = (
                indexed[control]["directional_gate_passed"] if control in indexed else None
            )

    result = {
        "experiment": EXPERIMENT, "method_version": METHOD,
        "created": base.timestamp(), "sources": sources, "rows": rows,
        "missing_variants": [variant for variant in VARIANTS if variant not in indexed],
        "complete_screen": set(indexed) == set(VARIANTS),
        "interpretation": "Single-seed mechanism screen only; do not infer significance or causal component identity",
    }
    output = Path(comparison_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    base.write_json(output / "comparison.json", result)
    lines = [
        "# exp007 Integrated Guidance Screen", "",
        "Single-seed observations only. Missing variants: " + ", ".join(result["missing_variants"]), "",
        "| Method | Primary /12 | Retain /33 | F1 hop2 /75 | F1 direct /5 | F1 answer LP delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['primary_correct']} | {row['retain_correct']} | "
            f"{row['F1_hop2_correct']} | {row['F1_direct_correct']} | "
            f"{row['F1_mean_answer_logprob_delta']:.6f} |"
        )
    lines.extend(["", "Full hashes, held-out retain, per-item audits and gates: comparison.json."])
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    configure_base()
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--run_dir")
    parser.add_argument("--model", default=None)
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--compare", nargs="+", metavar="RUN_DIR")
    parser.add_argument("--comparison_dir")
    args = parser.parse_args()
    if sum((args.self_test, args.preflight, bool(args.compare))) > 1:
        parser.error("Choose only one of --self_test, --preflight or --compare")
    if args.self_test:
        self_test()
        return
    if args.preflight:
        _, plan = base.preflight(args.model or base.te.MODEL)
        plan["integrated_guidance"] = {
            "historical_sources": [str(path.relative_to(WS)) for path in HISTORICAL_FILES],
            "historical_prior": load_historical_damage_prior()[1],
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.compare:
        if not args.comparison_dir:
            parser.error("--compare requires --comparison_dir")
        print(json.dumps(
            compare_runs(args.compare, Path(args.comparison_dir)),
            ensure_ascii=False,
        ))
        return
    if not args.variant or not args.run_dir:
        parser.error("--variant and --run_dir are required unless --self_test or --preflight is used")
    base.run_logged(args)


if __name__ == "__main__":
    main()
