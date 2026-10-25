"""Local exp010 C-group screen: C0/Cg/C1/C2/C3/C3R/C3D/C4."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_cloud_gsu_c2 as routed
import diagnose_gsu_integrated_guidance as integrated
import decisive_controls as controls

protocol = routed.protocol
baseline = routed.baseline
WS = Path(__file__).resolve().parents[2]
MODEL_DEFAULT = r"E:\12models\modelscope\Qwen2.5-1.5B-Instruct"
LEGACY_B0 = WS / "exp/runs/exp007_gsu_integrated_guidance/runs/run_batch_20261022_121504_28dccc_B0"
VARIANTS = ("C0", "Cg", "C1", "C2", "C3", "C3R", "C3D", "C4", "C1R", "C2D")
REPLAY_TOLERANCE = 1e-4
CONTRACT = {"seed": 42, "updates": 114, "lr": 5e-5, "weight_decay": .01,
    "betas": [.9, .999], "eps": 1e-8, "lora_r": 8, "lora_alpha": 16,
    "lora_dropout": 0., "route_floor": .5, "span_mix": .5,
    "svd_relative_cutoff": 1e-6, "refresh_before": [1, 39, 77],
    "correction_ratio": .30, "anchor_count": 16, "feedback_rounds": 2,
    "progress_fraction": .9, "progress_tolerance": 1e-6,
    "backtracking": [1., .5, .25, 0.], "batch_budget_seconds": 16*3600,
    "primary": "all89_path_macro_recall"}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def routing_weights(model, tokenizer, inputs, params, groups, variant):
    if variant == "C0":
        rows, meta = integrated.component_integrated_scores(
            model, tokenizer, inputs["F0"], inputs["F1"], params, groups)
        guide = integrated.integrated_component_guide(rows, "B0")
        return guide["weights"], {"score_source": "historical B0 compatibility probe",
            "rows": rows, "meta": meta, "guide": guide}
    if variant in ("Cg", "C4"):
        return {key: 1. for key in groups}, {"score_source": "none", "weights": {key: 1. for key in groups}}
    if variant in ("C3", "C3R", "C3D"):
        rows, meta_b = integrated.component_integrated_scores(
            model, tokenizer, inputs["F0"], inputs["F1"], params, groups)
        meta_a = meta_b
        source = "B"
    else:
        rows, meta_a = protocol.compute_component_scores(
            model, tokenizer, inputs["F0"], inputs["F1"], params, groups)
        meta_b = None
        source = "A"
    scores = {row["component"]: float(row["shared_score_S"]) for row in rows}
    if variant == "C3R":
        values = list(scores.values())
        random.Random(42+7919).shuffle(values)
        scores = dict(zip(sorted(scores), values))
    maximum = max(scores.values(), default=0.)
    if maximum <= 0.:
        weights = {key: 1. for key in groups}
    else:
        weights = {key: .5+.5*max(0., val)/maximum for key, val in scores.items()}
    return weights, {"score_source": source, "scores": scores,
                     "weights": weights, "A_meta": meta_a, "B_meta": meta_b}


def basis_for_variant(model, tokenizer, inputs, params, names, variant, refresh_seed):
    bases, metadata = routed.refresh_basis(model, tokenizer, inputs, params, names)
    if variant != "C3D":
        return bases, metadata
    random_bases = {}
    random_metadata = {}
    for param in params:
        basis = bases[param]
        if basis is None:
            random_bases[param] = None
            random_metadata[names[param]] = {"rank": None, "identity_fallback": True}
            continue
        rank = basis.shape[1]
        seed = int.from_bytes(hashlib.sha256(names[param].encode("utf-8")).digest()[:4], "big")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        matrix = torch.randn((param.numel(), 5), generator=generator)
        q, _ = torch.linalg.qr(matrix, mode="reduced")
        random_bases[param] = q[:, :rank].to(param.device)
        random_metadata[names[param]] = {"rank": int(rank), "identity_fallback": False,
                                         "fixed_basis_seed": seed}
    return random_bases, random_metadata


def path_audit(model, tokenizer, inputs):
    return routed.full_hop2_audit(model, tokenizer, inputs)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def legacy_replay_gate(smoke_dir):
    current = read_jsonl(Path(smoke_dir) / "diagnostics.jsonl")
    legacy = read_jsonl(LEGACY_B0 / "diagnostics.jsonl")
    if len(current) < 8 or len(legacy) < 8:
        return {"passed": False, "reason": "Need at least eight paired updates."}
    maximum = 0.0
    compared = []
    for index in range(8):
        now, ref = current[index], legacy[index]
        if now.get("item_id") != ref.get("forget_example_id"):
            return {"passed": False, "reason": f"F0 schedule mismatch at update {index+1}.",
                    "update": index+1}
        pairs = [(now["forget_lp"], ref["forget_lp"], "forget_lp"),
                 (now["correction_to_raw_norm_ratio"], ref["correction_to_raw_norm_ratio"], "feedback_ratio")]
        for key in ("anchor_lp_delta_raw", "anchor_lp_delta_final"):
            pairs.append((now[key], ref[key], key))
        now_rounds, ref_rounds = now["feedback_rounds"], ref["feedback_rounds"]
        if len(now_rounds) != len(ref_rounds):
            return {"passed": False, "reason": f"Feedback-round count mismatch at update {index+1}.",
                    "update": index+1}
        for nround, rround in zip(now_rounds, ref_rounds):
            pairs.append((nround["actual_deltas_before"], rround["actual_deltas_before"], "feedback_anchor_deltas"))
        for left, right, name in pairs:
            left_values = left if isinstance(left, list) else [left]
            right_values = right if isinstance(right, list) else [right]
            if len(left_values) != len(right_values):
                return {"passed": False, "reason": f"{name} shape mismatch at update {index+1}.",
                        "update": index+1}
            difference = max((abs(float(a)-float(b)) for a, b in zip(left_values, right_values)), default=0.)
            maximum = max(maximum, difference)
        compared.append(index+1)
    return {"passed": maximum <= REPLAY_TOLERANCE, "updates": compared,
            "max_abs_numeric_difference": maximum, "tolerance": REPLAY_TOLERANCE,
            "reference_run": str(LEGACY_B0)}


def legacy_metric_gate(metrics):
    old = json.loads((LEGACY_B0 / "metrics.json").read_text(encoding="utf-8"))
    observed = {
        "F0_direct_33": round(metrics["groups"]["F0"]["summary"]["mcqa_accuracy"] * 33),
        "F1_direct_5": round(metrics["groups"]["F1"]["summary"]["mcqa_accuracy"] * 5),
        "covered_hop2_75": round(metrics["sets"]["F1_hop2"]["summary"]["mcqa_accuracy"] * 75),
        "noncovered_hop2_12": round(metrics["sets"]["D_hop2_no_F1_overlap"]["summary"]["mcqa_accuracy"] * 12),
        "retain_heldout_33": round(metrics["groups"]["retain_heldout"]["summary"]["mcqa_accuracy"] * 33),
    }
    reference = {
        "F0_direct_33": round(old["logprob_audits"]["F0"]["summary"]["mcqa_accuracy"] * 33),
        "F1_direct_5": round(old["logprob_audits"]["F1"]["summary"]["mcqa_accuracy"] * 5),
        "covered_hop2_75": round(old["six_set"]["sets"]["F1_hop2"]["mean_correct"] * 75),
        "noncovered_hop2_12": round(old["six_set"]["sets"]["D_hop2_no_F1_overlap"]["mean_correct"] * 12),
        "retain_heldout_33": round(old["logprob_audits"]["retain_heldout"]["summary"]["mcqa_accuracy"] * 33),
    }
    return {"passed": observed == reference, "observed": observed,
            "reference": reference, "reference_run": str(LEGACY_B0)}


def worker(args):
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    seed=getattr(args,'seed',42)
    write(run/"config.json", {**vars(args), **CONTRACT, 'seed':seed})
    write(run/"status.json", {"status": "running", "phase": "loading"})
    try:
        cloud = getattr(args, "cloud_architecture", False)
        if cloud:
            routed.save_manifest(run, args)
            manifest = json.loads((run/"manifest.json").read_text(encoding="utf-8"))
            manifest["contract"] = {**CONTRACT,'seed':seed}
            manifest["worker"] = "run_local_gsu_c_group.worker"
            actual_manifest=Path(args.model)/'download_manifest_8b.json'
            if actual_manifest.exists():manifest['actual_model_manifest']=json.loads(actual_manifest.read_text(encoding='utf-8'))
            write(run/"manifest.json", manifest)
            (run/"run_log.md").write_text(
                "# Local-worker aligned C1 rerun\n\n"
                "rerun_of: exp014/batch_20261023_01\n"
                "Shared worker, routing, optimizer, train mode, CAUSAL_LM wrapper, "
                "and checkpoint evaluations match local C1. Cloud model loader and "
                "text-module enumeration are architecture-specific.\n", encoding="utf-8")
        random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        inputs = protocol.load_inputs()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        if cloud:
            base_model = baseline.load_model(args.model).eval()
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
        for parameter in base_model.parameters():
            parameter.requires_grad = False
        targets = baseline.target_modules(base_model) if cloud else integrated.base.TARGET_MODULES
        model = get_peft_model(base_model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=targets))
        model.train()
        if getattr(args,'gradient_checkpointing',False):
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        write(run/'execution.json',{'seed':seed,'gradient_checkpointing':getattr(args,'gradient_checkpointing',False),
             'model_train_mode':model.training,'lora_task_type':'CAUSAL_LM','worker':'local C-group'})
        params, groups, mapping, names = (
            routed.components(model) if cloud else protocol.collect_component_groups(model))
        if not cloud and len(groups) != 196:
            raise RuntimeError(f"Expected 196 local LoRA components, got {len(groups)}")
        initial = protocol.parameter_digest(model)
        weights, score_meta = routing_weights(model, tokenizer, inputs, params, groups, args.variant)
        if args.variant=='C1R':
            weights=controls.permute_within_type(weights,seed)
            score_meta['weights']=weights
            score_meta['control']='C1 weights permuted within module type; same score histogram'
        if protocol.parameter_digest(model) != initial:
            raise RuntimeError("Component scoring mutated initialized LoRA parameters")
        write(run/"modules.json", {"targets": targets, "component_count": len(groups),
              "trainable_parameters": sum(p.numel() for p in params), "initial_lora_sha256": initial})
        write(run/"component_scores.json", score_meta or {"score_source": "none", "weights": weights})
        updates = 8 if args.smoke else CONTRACT["updates"]
        schedule = protocol.make_update_schedule(inputs["F0"], updates, seed)
        if seed==42:
            anchor_schedule = protocol.make_anchor_schedule(updates=updates)
        else:
            anchor_rng=random.Random(seed)
            anchor_schedule=[anchor_rng.sample(range(33),16) for _ in range(updates)]
        write(run/"schedule.json", {"forget": schedule, "anchors": anchor_schedule})
        optimizer = torch.optim.AdamW(params, lr=CONTRACT["lr"], weight_decay=CONTRACT["weight_decay"],
                                      betas=tuple(CONTRACT["betas"]), eps=CONTRACT["eps"])
        bases = None
        guard_backtracks = 0
        zero_route = 0
        checkpoint_metrics = {}
        legacy_mapping = {parameter: protocol.component_key(names[parameter]) for parameter in params}
        with (run/"diagnostics.jsonl").open("x", encoding="utf-8") as log:
            for item, anchor_ids in zip(schedule, anchor_schedule):
                update = item["update"]
                needs_span = args.variant in ("C2", "C3", "C3R", "C3D", "C4", "C2D")
                if needs_span and (update in CONTRACT["refresh_before"] or (args.smoke and update == 2)):
                    bases, basis_meta = basis_for_variant(model, tokenizer, inputs, params, names,
                                                          args.variant, update)
                    if args.variant=='C2D':
                        bases,basis_meta=controls.random_subspaces(bases,names,seed)
                    write(run/f"basis_metadata_{update}.json", basis_meta)
                anchors = [(inputs["retain_all"][i]["question"], inputs["retain_all"][i]["answer"])
                           for i in anchor_ids]
                component_route = args.variant in ("C1", "C2", "C3", "C3R", "C3D", "C1R", "C2D")
                span_route = args.variant in ("C2", "C3", "C3R", "C3D", "C4", "C2D")
                w = weights if component_route else {key: 1. for key in groups}
                p = bases if span_route else {param: None for param in params}
                # C0 retains FB16 but omits the separately tested F0 progress guard.
                if args.variant == "C0":
                    row = integrated.base.forget_update(model, tokenizer, params, optimizer,
                                                        item, anchors, score_meta["guide"], legacy_mapping)
                    steps = [int(s["step"]) for s in optimizer.state.values() if "step" in s]
                    if len(set(steps)) != 1:
                        raise RuntimeError("Legacy C0 optimizer step count mismatch")
                    row.update({"optimizer_step_count": steps[0], "route_correction_ratio": 0.,
                                "accepted_alpha": None, "guard_skipped": True,
                                "final_correction_ratio": row["correction_to_raw_norm_ratio"]})
                else:
                    row = routed.train_step(model, tokenizer, params, optimizer,
                                            item, anchors, mapping, w, p)
                    guard_backtracks += int(row["accepted_alpha"] < 1.)
                zero_route += int(row.get("route_correction_ratio", 0.) <= 1e-12)
                if row["optimizer_step_count"] != update:
                    raise RuntimeError("Optimizer step count mismatch")
                row.update({"variant": args.variant, "update": update,
                            "item_id": item["item_id"], "elapsed_seconds": time.time()-started})
                log.write(json.dumps(row, allow_nan=False)+"\n"); log.flush()
                write(run/"status.json", {"status": "running", "phase": "training",
                      "update": update, "target_updates": updates, "elapsed_seconds": time.time()-started})
                if update % 10 == 0 or update == updates:
                    print("UPDATE", args.variant, update, "alpha", row.get("accepted_alpha"), flush=True)
                if not args.smoke and update in (38, 76, 114):
                    model.eval()
                    checkpoint = baseline.evaluate(model, tokenizer, inputs)
                    checkpoint["full_hop2"] = path_audit(model, tokenizer, inputs)
                    checkpoint_metrics[update] = checkpoint
                    write(run/f"checkpoint_eval_{update}.json", checkpoint)
                    model.train()
        model.save_pretrained(run/"adapter")
        if args.smoke:
            model.eval()
            expected=protocol.collect_item_audit(model,tokenizer,inputs['F0'][:1])[0]['answer_logprob']
            del optimizer, params, groups, mapping, names, bases
            del model,base_model,legacy_mapping,weights,score_meta
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            fresh = baseline.load_model(args.model)
            loaded = PeftModel.from_pretrained(fresh, str(run/"adapter")).eval()
            metrics = {"smoke": "passed", "updates": updates,
                       "roundtrip": protocol.collect_item_audit(loaded, tokenizer, inputs["F0"][:1])[0]["answer_logprob"]}
            metrics['roundtrip_error']=abs(metrics['roundtrip']-expected)
            if metrics['roundtrip_error']>1e-4:raise RuntimeError('Adapter roundtrip mismatch')
        else:
            model.eval()
            metrics = checkpoint_metrics[updates]
            metrics["diagnostics"] = {"updates": updates, "guard_backtracks": guard_backtracks,
                "route_zero_correction_steps": zero_route, "route_zero_correction_fraction": zero_route/updates}
        write(run/"metrics.json", metrics)
        write(run/"status.json", {"status": "completed", "ended": protocol.timestamp(),
              "updates": updates, "seconds": time.time()-started,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()})
    except BaseException:
        write(run/"status.json", {"status": "failed", "ended": protocol.timestamp(),
              "seconds": time.time()-started, "traceback": traceback.format_exc()})
        raise


def summarize(root):
    rows=[]
    for variant in VARIANTS:
        folder=root/variant
        if not (folder/"metrics.json").is_file(): continue
        m=json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
        rows.append({"variant":variant,
          "all89_path_macro_recall":m["full_hop2"]["all_hop2"]["summary"]["path_macro_accuracy"],
          "covered25_accuracy":m["sets"]["F1_hop2"]["summary"]["mcqa_accuracy"],
          "F0_direct":m["groups"]["F0"]["summary"]["mcqa_accuracy"],
          "F1_direct":m["groups"]["F1"]["summary"]["mcqa_accuracy"],
          "retain_train":m["groups"]["retain_train_pool"]["summary"]["mcqa_accuracy"],
          "retain_heldout":m["groups"]["retain_heldout"]["summary"]["mcqa_accuracy"],
          "guard_backtracks":m["diagnostics"]["guard_backtracks"],"run_dir":str(folder)})
    result={"experiment":"exp010_gsu_step_routing","status":"completed" if len(rows)==8 else "partial",
      "rows":rows,"missing_variants":[v for v in VARIANTS if v not in {x["variant"] for x in rows}],
      "interpretation":"Single seed 42; exploratory local screen, not an optimality or significance claim."}
    write(root/"comparison.json",result)
    lines=["# exp010 C-group screen","","Single seed 42; Qwen2.5-1.5B; exploratory only.","",
      "| Variant | All-89 path macro ↓ | Covered-hop2 ↓ | F0 direct ↓ | F1 direct | Retain train ↑ | Retain heldout ↑ | Guard backtracks |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
      lines.append(f"| {r['variant']} | {r['all89_path_macro_recall']:.4f} | {r['covered25_accuracy']:.4f} | {r['F0_direct']:.4f} | {r['F1_direct']:.4f} | {r['retain_train']:.4f} | {r['retain_heldout']:.4f} | {r['guard_backtracks']} |")
    (root/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def batch(args):
    root=Path(args.run_dir).resolve(); root.mkdir(parents=True,exist_ok=False); start=time.time()
    _, preflight=protocol.preflight(args.model)
    if (preflight["component_count"],preflight["F0_count"],preflight["F1_count"]) != (196,33,5):
        raise RuntimeError("Frozen Qwen2.5-1.5B exp010 preflight counts do not match")
    source_paths=[Path(__file__),Path(routed.__file__),Path(integrated.__file__),
      WS/"exp/train/run_remote_baselines.py",WS/"exp/train/train_baselines.py",
      WS/"exp/train/diagnose_gsu_memory_guided.py",WS/"exp/train/diagnose_gsu_actual_feedback.py",
      WS/"exp/train/diagnose_gsu_retain_protection.py",WS/"exp/train/diagnose_gsu_m3.py",
      WS/"exp/train/train_gsu.py",WS/"exp/damage/component_ablation.py",
      WS/"exp/damage/eval_per_item.py",WS/"exp/runs/exp010_gsu_step_routing/card.md"]
    frozen={str(p.resolve()):digest(p) for p in source_paths}
    reference_paths=[LEGACY_B0/"diagnostics.jsonl",LEGACY_B0/"metrics.json",
                     LEGACY_B0/"schedule.json",LEGACY_B0/"anchor_schedule.json",
                     LEGACY_B0/"resolved_config.json"]
    reference_hashes={str(p.resolve()):digest(p) for p in reference_paths}
    frozen_inputs=protocol.input_manifest()
    write(root/"batch_manifest.json",{"contract":CONTRACT,"model":args.model,"variants":VARIANTS,
      "source_sha256":frozen,"legacy_B0_reference_sha256":reference_hashes,
      "input_manifest":frozen_inputs,"torch":torch.__version__,
      "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"command":sys.argv})
    (root/"environment.txt").write_bytes(subprocess.check_output([sys.executable,"-m","pip","freeze"]))
    completed=[]
    jobs=[("smoke_C0","C0",True)]+[(v,v,False) for v in VARIANTS]
    try:
      for label,variant,smoke in jobs:
        if time.time()-start>=CONTRACT["batch_budget_seconds"]: raise TimeoutError("16h budget exhausted")
        if any(digest(Path(p))!=h for p,h in frozen.items()): raise RuntimeError("Frozen source changed")
        if any(digest(Path(p))!=h for p,h in reference_hashes.items()): raise RuntimeError("Legacy B0 reference changed")
        if protocol.input_manifest()!=frozen_inputs: raise RuntimeError("Frozen input data changed")
        run=root/label; command=[sys.executable,str(Path(__file__).resolve()),"--model",args.model,
          "--variant",variant,"--run_dir",str(run)]+(["--smoke"] if smoke else [])
        write(root/"status.json",{"status":"running","current":label,"completed":completed,"command":command})
        remaining=max(1,int(CONTRACT["batch_budget_seconds"]-(time.time()-start)))
        with (root/(label+".stdout.log")).open("x") as out,(root/(label+".stderr.log")).open("x") as err:
          try:
            subprocess.run(command,stdout=out,stderr=err,check=True,timeout=remaining)
          except subprocess.TimeoutExpired:
            status_path=run/"status.json"
            if status_path.is_file():
              state=json.loads(status_path.read_text(encoding="utf-8"))
              state.update({"status":"interrupted_budget","ended":protocol.timestamp(),
                            "message":"Batch budget reached; partial artifacts preserved"})
              write(status_path,state)
            raise
        completed.append(label)
        if smoke:
          replay=legacy_replay_gate(run)
          write(root/"c0_replay_gate.json", replay)
          if not replay["passed"]:
            raise RuntimeError("C0 eight-step replay did not match historical B0 within frozen tolerance")
        elif variant == "C0":
          metrics=json.loads((run/"metrics.json").read_text(encoding="utf-8"))
          metric_gate=legacy_metric_gate(metrics)
          write(root/"c0_metric_gate.json", metric_gate)
          if not metric_gate["passed"]:
            summarize(root)
            raise RuntimeError("Formal C0 count-valued metrics did not reproduce historical B0")
          summarize(root); write(root/"partial_results.json",{"completed":completed})
        if not smoke and variant != "C0":
          summarize(root); write(root/"partial_results.json",{"completed":completed})
      summarize(root)
      write(root/"status.json",{"status":"completed","completed":completed,"ended":protocol.timestamp(),"seconds":time.time()-start})
    except BaseException:
      summarize(root)
      write(root/"status.json",{"status":"failed","completed":completed,"ended":protocol.timestamp(),
        "seconds":time.time()-start,"traceback":traceback.format_exc()})
      raise


def main():
    p=argparse.ArgumentParser();p.add_argument("--batch",action="store_true");p.add_argument("--smoke",action="store_true")
    p.add_argument("--self_test",action="store_true")
    p.add_argument("--cloud_architecture", action="store_true")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--gradient_checkpointing",action="store_true")
    p.add_argument("--model",default=MODEL_DEFAULT);p.add_argument("--variant",choices=VARIANTS);p.add_argument("--run_dir",required=True)
    a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    if a.self_test:
        param=torch.nn.Parameter(torch.tensor([1.,2.]))
        raw={param:torch.tensor([1.,2.])}
        basis=torch.tensor([[1.],[0.]])
        routed_step,meta=routed.route_step([param],raw,{param:"x"},{"x":.5},{param:basis})
        assert torch.isfinite(routed_step[param]).all()
        assert meta["route_correction_ratio"] <= .3+1e-7
        assert torch.allclose(routed.project(torch.tensor([3.,4.]),basis),torch.tensor([3.,0.]))
        identity,_=routed.route_step([param],raw,{param:"x"},{"x":1.},{param:None})
        assert torch.equal(identity[param],raw[param])
        print(json.dumps({"self_test":"passed","route_ratio":meta["route_correction_ratio"],
                          "zero_span_identity":"passed"}))
        return
    if a.batch: batch(a)
    elif a.variant: worker(a)
    else: p.error("worker mode requires --variant")


if __name__=="__main__": main()
