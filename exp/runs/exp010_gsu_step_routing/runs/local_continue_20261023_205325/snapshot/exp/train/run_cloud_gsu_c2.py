"""GSU-C2 cloud pilot: component/span routing of actual F0 AdamW proposals.

Independent entry point; never patches frozen exp008/009 trainers or artifacts.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import torch
from peft import PeftModel
from transformers import AutoTokenizer

import run_remote_baselines as baseline
import diagnose_gsu_actual_feedback as feedback
from diagnose_gsu_m3 import scalar_map_norm2, scalar_map_dot

protocol = baseline.protocol
WS = Path(__file__).resolve().parents[2]
EXPERIMENT = 'exp011_cloud_gsu_c2'
MODELS = [('llama32_3b', '/root/autodl-tmp/models/Llama-3.2-3B-Instruct'),
          ('qwen35_4b', '/root/autodl-tmp/models/Qwen3.5-4B')]
CONTRACT = dict(seed=42, updates=114, lr=5e-5, weight_decay=.01, betas=[.9,.999],
    eps=1e-8, lora_r=8, lora_alpha=16, lora_dropout=0., route_floor=.5,
    span_mix=.5, svd_relative_cutoff=1e-6, refresh_before=[1,39,77],
    correction_ratio=.30, anchor_count=16, feedback_rounds=2,
    progress_fraction=.9, progress_tolerance=1e-6, backtracking=[1.,.5,.25,0.],
    per_model_timeout_seconds=21600, primary='historical_F1_hop2_accuracy',
    loss='F0 answer mean logprob only; F1 gradients used for guidance')


def components(model):
    groups = defaultdict(list)
    names = {}
    mapping = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if '.lora_A.' in name:
            key = name.split('.lora_A.')[0]
        elif '.lora_B.' in name:
            key = name.split('.lora_B.')[0]
        else:
            raise RuntimeError(f'Unexpected trainable parameter: {name}')
        groups[key].append(parameter)
        names[parameter], mapping[parameter] = name, key
    if not groups or any(len(values) != 2 for values in groups.values()):
        raise RuntimeError('Each language component must have exactly two LoRA tensors')
    params = list(names)
    if any(parameter.dtype != torch.float32 for parameter in params):
        raise RuntimeError('C2 trust-budget arithmetic requires FP32 LoRA parameters')
    return params, dict(sorted(groups.items())), mapping, names


def orthonormal_basis(matrix, relative_cutoff=1e-6):
    if matrix.ndim != 2 or not bool(torch.isfinite(matrix).all()):
        raise ValueError('Expected finite column matrix')
    if matrix.numel() == 0 or float(matrix.norm()) <= 1e-20:
        return None
    left, singular, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
    keep = singular > singular.max() * relative_cutoff
    basis = left[:, keep].contiguous()
    if not basis.shape[1]:
        return None
    gram = basis.T @ basis
    if not torch.allclose(gram, torch.eye(basis.shape[1], device=gram.device), atol=1e-4, rtol=1e-4):
        raise RuntimeError('Span basis is not orthonormal')
    return basis


def refresh_basis(model, tokenizer, inputs, params, names):
    snapshots = []
    for item in inputs['F1']:
        snapshot, _ = protocol.cpu_gradient_snapshot(model, tokenizer, params,
                                                      item['question'], item['answer'])
        snapshots.append(snapshot)
    bases, metadata = {}, {}
    for parameter in params:
        matrix = torch.stack([snapshot[parameter].flatten() for snapshot in snapshots], dim=1)
        basis = orthonormal_basis(matrix)
        bases[parameter] = None if basis is None else basis.to(parameter.device)
        metadata[names[parameter]] = {'rank': None if basis is None else basis.shape[1],
                                    'zero_gradient_identity_fallback': basis is None}
    return bases, metadata


def project(value, basis):
    if basis is None:
        return value.clone()
    flat = value.flatten().float()
    return (basis @ (basis.T @ flat)).view_as(value)


def route_step(params, raw, mapping, weights, bases):
    routed = {}
    for parameter in params:
        value = raw[parameter].float()
        routed[parameter] = weights[mapping[parameter]] * (.5 * value + .5 * project(value, bases[parameter]))
    raw_norm = math.sqrt(max(scalar_map_norm2(params, raw), 0.))
    change = {parameter: routed[parameter] - raw[parameter] for parameter in params}
    correction_norm = math.sqrt(max(scalar_map_norm2(params, change), 0.))
    scale = min(1., .3 * raw_norm / max(correction_norm, 1e-30))
    result = {parameter: raw[parameter] + scale * change[parameter] for parameter in params}
    return result, {'raw_norm': raw_norm, 'unclipped_route_correction_norm': correction_norm,
                    'route_trust_scale': scale,
                    'route_correction_ratio': scale * correction_norm / max(raw_norm, 1e-30)}


def set_displacement(params, before, delta):
    with torch.no_grad():
        for parameter, old in zip(params, before):
            parameter.copy_(old + delta[parameter])


def choose_guard(params, before, raw, candidate, evaluate, lp_before, lp_raw):
    progress = lp_before - lp_raw
    tests = []
    if progress <= 0:
        set_displacement(params, before, raw)
        return 0., lp_raw, [{'alpha': 0., 'reason': 'raw_proposal_not_descent', 'lp': lp_raw}], False
    bound = lp_before - .9 * progress + 1e-6
    for alpha in (1., .5, .25, 0.):
        delta = {parameter: raw[parameter] + alpha * (candidate[parameter] - raw[parameter])
                 for parameter in params}
        set_displacement(params, before, delta)
        lp = float(evaluate())
        if not math.isfinite(lp):
            raise RuntimeError('Nonfinite guarded F0 logprob')
        tests.append({'alpha': alpha, 'lp': lp, 'bound': bound})
        if lp <= bound:
            return alpha, lp, tests, True
    raise RuntimeError('Raw proposal unexpectedly failed deterministic F0 replay')


def lp_value(model, tokenizer, item):
    with torch.no_grad():
        value = float(baseline.losses.lp_masked(model, tokenizer, item['question'], item['answer']))
    if not math.isfinite(value):
        raise RuntimeError('Nonfinite logprob')
    return value


def full_hop2_audit(model, tokenizer, inputs):
    keys = [key for key in inputs['qa'] if key.startswith('hop2|')]
    covered = set(inputs['sets']['F1_hop2'])
    if len(keys) != 267:
        raise RuntimeError('Expected frozen267 full-hop2 paraphrases')
    groups = {'all_hop2': keys, 'noncovered_hop2': [key for key in keys if key not in covered]}
    result = protocol.collect_set_audit(model, tokenizer, groups, inputs['qa'])
    for audit in result.values():
        paths = defaultdict(list)
        for row in audit['items']:
            paths['|'.join(row['id'].split('|')[:3])].append(row['mcqa_correct'])
        audit['summary']['path_count'] = len(paths)
        audit['summary']['path_macro_accuracy'] = sum(sum(x)/len(x) for x in paths.values())/len(paths)
    return result


def train_step(model, tokenizer, params, optimizer, item, anchors, mapping, weights, bases):
    before_lp = lp_value(model, tokenizer, item)
    anchor_before = feedback.anchor_logprobs(model, tokenizer, anchors)
    before = [parameter.detach().clone() for parameter in params]
    optimizer.zero_grad(set_to_none=True)
    loss = baseline.losses.lp_masked(model, tokenizer, item['question'], item['answer'])
    loss.backward()
    if not math.isfinite(float(loss.detach())) or any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in params):
        raise RuntimeError('Nonfinite F0 loss/gradient')
    optimizer.step()
    raw = feedback.current_delta(params, before)
    raw_lp = lp_value(model, tokenizer, item)
    raw_anchor_lp = feedback.anchor_logprobs(model, tokenizer, anchors)
    routed, diagnostics = route_step(params, raw, mapping, weights, bases)
    set_displacement(params, before, routed)
    routed_lp = lp_value(model, tokenizer, item)
    fb = feedback.feedback_update(params, model, tokenizer, anchors, anchor_before,
                                  before, raw, None)
    candidate = feedback.current_delta(params, before)
    feedback_lp = lp_value(model, tokenizer, item)
    alpha, final_lp, trials, raw_descent = choose_guard(
        params, before, raw, candidate, lambda: lp_value(model, tokenizer, item), before_lp, raw_lp)
    final_anchor_lp = feedback.anchor_logprobs(model, tokenizer, anchors)
    final = feedback.current_delta(params, before)
    correction = {parameter: final[parameter] - raw[parameter] for parameter in params}
    final_ratio = math.sqrt(max(scalar_map_norm2(params, correction),0.))/max(diagnostics['raw_norm'],1e-30)
    if final_ratio > .3 + 1e-6:
        raise RuntimeError('Final displacement exceeded shared trust budget')
    step_counts = [int(state['step']) for state in optimizer.state.values() if 'step' in state]
    if len(set(step_counts)) != 1:
        raise RuntimeError('Optimizer tensors advanced inconsistently')
    diagnostics.update({'F0_lp_before': before_lp,'F0_lp_raw':raw_lp,'F0_lp_routed':routed_lp,
        'F0_lp_feedback':feedback_lp,'F0_lp_final':final_lp,'raw_descent':raw_descent,
        'accepted_alpha':alpha,'guard_trials':trials,'final_correction_ratio':final_ratio,
        'raw_final_cosine': scalar_map_dot(params,raw,final)/max(
            diagnostics['raw_norm']*math.sqrt(max(scalar_map_norm2(params,final),0.)),1e-30),
        'anchor_lp_before':anchor_before,'anchor_lp_raw':raw_anchor_lp,
        'anchor_lp_feedback':fb['lp_after_final'],'anchor_lp_final':final_anchor_lp,
        'feedback_rounds':fb['feedback_rounds'],'optimizer_step_count':step_counts[0]})
    return diagnostics


def save_manifest(run, args):
    sources = {str(path.relative_to(WS)):protocol.sha256_file(path)
               for folder in ('train','damage') for path in (WS/'exp'/folder).rglob('*.py')}
    baseline.write(run/'manifest.json', {'created':protocol.timestamp(),'command':sys.argv,
        'contract':CONTRACT,'args':vars(args),'source':sources,'inputs':protocol.input_manifest(),
        'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'model_manifest':json.loads(Path('/root/autodl-tmp/models/download_manifest.json').read_text())})
    (run/'environment.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))


def worker(args):
    run = Path(args.run_dir)
    run.mkdir(parents=True,exist_ok=False)
    baseline.write(run/'status.json',{'status':'running','phase':'preflight','started':protocol.timestamp()})
    baseline.write(run/'config.json',vars(args)|CONTRACT)
    (run/'run_log.md').write_text('# GSU-C2 cloud pilot\n\nImmutable sources: manifest.json and snapshot sync_manifest.json.\n'
        'Predeclared experiment: exp/runs/exp011_cloud_gsu_c2/card.md.\n'
        'No historical layer prior; F1 labels accessed only for guidance.\n',encoding='utf-8')
    started=time.time()
    try:
        save_manifest(run,args)
        inputs=protocol.load_inputs()
        random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        print('LOADING_MODEL',args.model,flush=True)
        model=baseline.load_model(args.model)
        model,targets=baseline.attach_lora(model)
        model.eval()
        params,groups,mapping,names=components(model)
        initial_hash=protocol.parameter_digest(model)
        baseline.write(run/'modules.json',{'targets':targets,'component_count':len(groups),
            'trainable_parameters':sum(parameter.numel() for parameter in params),
            'initial_lora_sha256':initial_hash})
        print('SCORING_COMPONENTS',len(groups),flush=True)
        baseline.write(run/'status.json',{'status':'running','phase':'component_scoring','started':protocol.timestamp()})
        rows,meta=protocol.compute_component_scores(model,tokenizer,inputs['F0'],inputs['F1'],params,groups)
        if initial_hash!=protocol.parameter_digest(model):raise RuntimeError('Scoring mutated adapter')
        maximum=max(row['shared_score_S'] for row in rows)
        weights={row['component']:(.5+.5*row['shared_score_S']/maximum if maximum>0 else 1.) for row in rows}
        baseline.write(run/'component_scores.json',{'components':rows,'meta':meta,'weights':weights,
                                                    'zero_score_identity_fallback':maximum<=0})
        steps=2 if args.smoke else 114
        schedule=protocol.make_update_schedule(inputs['F0'],steps,42)
        anchors=protocol.make_anchor_schedule(updates=steps)
        baseline.write(run/'schedule.json',{'forget':schedule,'anchors':anchors})
        optimizer=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01,betas=(.9,.999),eps=1e-8)
        bases=None
        with (run/'diagnostics.jsonl').open('x',encoding='utf-8') as log:
            for item,anchor_indices in zip(schedule,anchors):
                if time.time()-started>CONTRACT['per_model_timeout_seconds']:raise TimeoutError('Compute budget exhausted')
                update=item['update']
                if update in (1,39,77) or (args.smoke and update==2):
                    print('REFRESH_BASIS',update,flush=True)
                    bases,basis_meta=refresh_basis(model,tokenizer,inputs,params,names)
                    baseline.write(run/f'basis_metadata_{update}.json',basis_meta)
                batch_anchors=[(inputs['retain_all'][index]['question'],inputs['retain_all'][index]['answer']) for index in anchor_indices]
                diagnostics=train_step(model,tokenizer,params,optimizer,item,batch_anchors,mapping,weights,bases)
                if diagnostics['optimizer_step_count']!=update:raise RuntimeError('Optimizer advanced more than once')
                diagnostics.update({'update':update,'item_id':item['item_id'],
                    'anchor_indices':anchor_indices,'elapsed_seconds':time.time()-started})
                log.write(json.dumps(diagnostics,allow_nan=False)+'\n');log.flush()
                print('UPDATE',update,'rawLP',diagnostics['F0_lp_raw'],'finalLP',diagnostics['F0_lp_final'],
                      'alpha',diagnostics['accepted_alpha'],flush=True)
                baseline.write(run/'status.json',{'status':'running','phase':'training','update':update,
                    'target_updates':steps,'elapsed_seconds':time.time()-started})
        model.save_pretrained(run/'adapter')
        if args.smoke:
            model.eval()
            audit=protocol.collect_item_audit(model,tokenizer,inputs['F0'][:1])
            expected=audit[0]['answer_logprob']
            del optimizer,params,groups,mapping,names,bases,model
            torch.cuda.empty_cache()
            fresh=baseline.load_model(args.model)
            loaded=PeftModel.from_pretrained(fresh,str(run/'adapter')).eval()
            actual=protocol.collect_item_audit(loaded,tokenizer,inputs['F0'][:1])[0]['answer_logprob']
            if abs(expected-actual)>1e-4:raise RuntimeError('Adapter roundtrip altered answer LP')
            metrics={'smoke':'passed','updates':steps,'roundtrip_lp_error':abs(expected-actual)}
        else:
            print('EVALUATION_STARTED',flush=True)
            baseline.write(run/'status.json',{'status':'running','phase':'evaluation','update':steps})
            metrics=baseline.evaluate(model,tokenizer,inputs)
            metrics['full_hop2']=full_hop2_audit(model,tokenizer,inputs)
        baseline.write(run/'metrics.json',metrics)
        baseline.write(run/'status.json',{'status':'completed','ended':protocol.timestamp(),
            'updates':steps,'seconds':time.time()-started,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
    except BaseException:
        baseline.write(run/'status.json',{'status':'failed','ended':protocol.timestamp(),
                                          'traceback':traceback.format_exc()})
        raise


def compare_completed(batch):
    evidence=WS/'exp/analysis/cloud_comparison_20261022/evidence'
    rows=[]
    for experiment in ('exp008','exp009'):
        status=json.loads((evidence/experiment/'status.json').read_text())
        for run_id in status['completed']:
            if run_id.endswith('_smoke'):continue
            folder=evidence/experiment/run_id
            config=json.loads((folder/'config.json').read_text())
            metrics=json.loads((folder/'metrics.json').read_text())
            rows.append({'model':Path(config['model']).name,'method':config['method'],
                         'supervision':config['supervision'],'run_id':run_id,
                         'metrics':metrics,'source_sha256':protocol.sha256_file(folder/'metrics.json')})
    for name,path in MODELS:
        folder=batch/(name+'_C2')
        if (folder/'metrics.json').exists():
            rows.append({'model':Path(path).name,'method':'GSU-C2','supervision':'F0','run_id':folder.name,
                'metrics':json.loads((folder/'metrics.json').read_text()),
                'source_sha256':protocol.sha256_file(folder/'metrics.json')})
    baseline.write(batch/'comparison.json',{'rows':rows,'interpretation':
        'Single-seed pilot; C2 has additional guidance/feedback compute; no cross-model ranking; baseline full-hop2 missing'})
    lines=['# GSU-C2 cloud pilot comparison','',
        'Actual completed outcomes only. All methods use114 updates; computation is not matched.','',
        '| Model | Method | Supervision | F0 recall down | Paraphrase down | Covered-hop2 down | Heldout retain up |',
        '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        metrics=row['metrics'];g=metrics['groups']
        values=[g['F0']['summary']['mcqa_accuracy'],g['F0_paraphrases']['summary']['mcqa_accuracy'],
                metrics['sets']['F1_hop2']['summary']['mcqa_accuracy'],g['retain_heldout']['summary']['mcqa_accuracy']]
        lines.append('| '+row['model']+' | '+row['method']+' | '+row['supervision']+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    (batch/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def batch(args):
    root=Path(args.run_dir);root.mkdir(parents=True,exist_ok=False)
    baseline.write(root/'config.json',{'contract':CONTRACT,'models':MODELS,'command':sys.argv})
    frozen={str(path):protocol.sha256_file(path) for folder in ('train','damage')
            for path in (WS/'exp'/folder).rglob('*.py')}
    baseline.write(root/'source_manifest.json',frozen)
    completed=[]
    try:
        for smoke in (True,False):
            for name,model in MODELS:
                if any(protocol.sha256_file(Path(path))!=digest for path,digest in frozen.items()):
                    raise RuntimeError('Frozen source changed')
                label=name+('_smoke' if smoke else'_C2')
                command=[sys.executable,str(Path(__file__).resolve()),'--model',model,
                         '--run_dir',str(root/label)]+(['--smoke'] if smoke else [])
                baseline.write(root/'status.json',{'status':'running','current':label,'completed':completed,'command':command})
                with (root/(label+'.stdout.log')).open('x') as out,(root/(label+'.stderr.log')).open('x') as err:
                    subprocess.run(command,stdout=out,stderr=err,timeout=21600,check=True)
                completed.append(label)
                if not smoke:compare_completed(root)
        baseline.write(root/'status.json',{'status':'completed','completed':completed,'ended':protocol.timestamp()})
    except BaseException:
        baseline.write(root/'status.json',{'status':'failed','completed':completed,'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--run_dir',required=True)
    args=parser.parse_args()
    batch(args) if args.batch else worker(args)
