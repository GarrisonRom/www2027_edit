"""Offline-after-C2 MMLU and WikiText-2 audit of matched cloud endpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer

import run_remote_baselines as baseline
import run_cloud_gsu_c2 as c2

WS = Path(__file__).resolve().parents[2]
MMLU_LIMIT_PER_SUBJECT = 10
MMLU_FEWSHOT = 5
PPL_MAX_LENGTH = 2048
PPL_STRIDE = 512
WIKITEXT_DATASET = 'Salesforce/wikitext'
WIKITEXT_CONFIG = 'wikitext-2-raw-v1'
MODELS = {'qwen35_4b': '/root/autodl-tmp/models/Qwen3.5-4B',
          'llama32_3b': '/root/autodl-tmp/models/Llama-3.2-3B-Instruct'}
BASELINE_ROOTS = {
    'exp008': '/root/autodl-tmp/www2027_baselines_20261022_v2/exp/runs/exp008_remote_baselines/runs/batch_20261022_01',
    'exp009': '/root/autodl-tmp/www2027_rmu_20261022/exp/runs/exp009_remote_rmu/runs/batch_20261022_01'}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def json_metadata(value):
    """Normalize harness metadata (including torch.dtype) without altering metrics."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str, allow_nan=False))


def token_windows(token_count, max_length=PPL_MAX_LENGTH, stride=PPL_STRIDE):
    if token_count < 2 or max_length < 2 or not 0 < stride < max_length:
        raise ValueError('Invalid token/window configuration')
    windows, previous_end = [], 0
    for begin in range(0, token_count, stride):
        end = min(begin + max_length, token_count)
        target_start = max(previous_end, begin + 1)
        if target_start < end:
            windows.append({'begin': begin, 'end': end,
                            'target_start': target_start, 'target_count': end - target_start})
        previous_end = end
        if end == token_count:
            break
    if sum(window['target_count'] for window in windows) != token_count - 1:
        raise RuntimeError('Sliding windows do not score every next-token target exactly once')
    return windows


def wiki_text_revision():
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(WIKITEXT_DATASET, revision='main')
    if not info.sha:
        raise RuntimeError('Unable to freeze WikiText dataset revision')
    return info.sha


def load_wikitext(revision):
    from datasets import load_dataset
    dataset = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split='test', revision=revision)
    documents = [row['text'] for row in dataset]
    corpus = '\n\n'.join(documents)
    return corpus, {'dataset': WIKITEXT_DATASET, 'config': WIKITEXT_CONFIG,
        'split': 'test', 'revision': revision, 'document_count': len(documents),
        'text_sha256': sha256_bytes(corpus.encode('utf-8'))}


@torch.no_grad()
def token_ppl(model, tokenizer, corpus, *, token_limit=None):
    encoded = tokenizer(corpus, return_tensors='pt', add_special_tokens=False).input_ids
    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], dtype=encoded.dtype)
        encoded = torch.cat((bos, encoded), dim=1)
    if token_limit is not None:
        encoded = encoded[:, :token_limit]
    ids = encoded[0]
    model_config = model.config
    max_context = getattr(model_config, 'max_position_embeddings', None)
    if max_context is None:
        max_context = getattr(getattr(model_config, 'text_config', None),
                              'max_position_embeddings', PPL_MAX_LENGTH)
    max_length = min(PPL_MAX_LENGTH, int(max_context or PPL_MAX_LENGTH))
    stride = min(PPL_STRIDE, max_length - 1)
    windows = token_windows(len(ids), max_length=max_length, stride=stride)
    total_nll, total_targets = 0.0, 0
    model.eval()
    for window in windows:
        begin, end = window['begin'], window['end']
        target_start, target_end = window['target_start'], window['end']
        input_ids = ids[begin:end].unsqueeze(0).to(model.device)
        labels = input_ids.clone()
        labels[:, :target_start - begin] = -100
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :].contiguous().float()
        shifted_labels = labels[:, 1:].contiguous()
        nll = F.cross_entropy(logits.view(-1, logits.shape[-1]), shifted_labels.view(-1),
                              ignore_index=-100, reduction='sum')
        total_nll += float(nll)
        total_targets += window['target_count']
    if total_targets != len(ids) - 1 or not total_targets:
        raise RuntimeError('PPL token accounting mismatch')
    mean_nll = total_nll / total_targets
    return {'token_ppl': math.exp(mean_nll), 'mean_nll_nats': mean_nll,
            'total_nll_nats': total_nll, 'scored_tokens': total_targets,
            'input_tokens_including_bos': len(ids), 'max_length': max_length,
            'stride': stride, 'window_count': len(windows)}


def mmlu_sample_plan(subject_lengths, limit):
    import random
    if not subject_lengths or any(count < limit for count in subject_lengths.values()):
        raise ValueError('Every MMLU subject must have at least the requested sample count')
    return {name: sorted(random.Random(f'mmlu-{name}-42').sample(range(count), limit))
            for name, count in sorted(subject_lengths.items())}


def mmlu_eval(model, tokenizer, *, limit, output_dir):
    from lm_eval import simple_evaluate
    from lm_eval.models.hf_vlms import HFLM
    from lm_eval.tasks import TaskManager, get_task_dict

    harness_model = HFLM(model, tokenizer=tokenizer, batch_size=1, max_batch_size=1,
                         device='cuda')
    task_manager=TaskManager()
    task_tree=get_task_dict(['mmlu'],task_manager=task_manager)
    leaves=[]
    def visit(tree):
        for key,value in tree.items():
            if isinstance(value,dict):
                visit(value)
            else:
                task_name=value.dump_config().get('task',str(key))
                leaves.append((task_name,value))
    visit(task_tree)
    if len(leaves)!=57:
        raise RuntimeError(f'Expected57 MMLU subjects, found{len(leaves)}')
    sample_map={}
    for task_name,task in leaves:
        documents=task.eval_docs() if callable(task.eval_docs) else task.eval_docs
        if not hasattr(documents,'__len__'):
            documents=list(documents)
        count=len(documents)
        if count<limit:
            raise RuntimeError(f'MMLU subject {task_name} has only{count} test examples')
        sample_map[task_name]=mmlu_sample_plan({task_name:count},limit)[task_name]
    result = simple_evaluate(model=harness_model, tasks=['mmlu'], num_fewshot=MMLU_FEWSHOT,
        samples=sample_map, batch_size=1, log_samples=True, write_out=True,
        apply_chat_template=False, random_seed=42, numpy_random_seed=42,
        torch_random_seed=42, fewshot_random_seed=42, bootstrap_iters=0)
    results = result.get('results', {})
    samples = result.get('samples', {})
    per_subject = {}
    selected_count = 0
    for task_name, task_samples in samples.items():
        if task_name == 'mmlu':
            continue
        metrics=results.get(task_name,{})
        accuracies = [float(value) for key, value in metrics.items()
                      if key.split(',')[0] in ('acc', 'acc_norm') and isinstance(value, (int,float))]
        if not accuracies:
            raise RuntimeError(f'Missing MMLU accuracy metric for {task_name}')
        per_subject[task_name] = {'accuracy': accuracies[0], 'sample_count': len(task_samples),
            'selected_doc_ids':sample_map.get(task_name,[])}
        selected_count += len(task_samples)
    if len(per_subject) != 57:
        raise RuntimeError(f'MMLU group expanded to only {len(per_subject)} subjects')
    if any(value['sample_count'] != limit for value in per_subject.values()):
        raise RuntimeError('Frozen MMLU per-subject sample plan was not honored')
    macro = sum(item['accuracy'] for item in per_subject.values()) / len(per_subject)
    result['samples'] = samples
    raw_path = output_dir / 'mmlu_samples.json'
    raw_path.write_text(json.dumps(samples, ensure_ascii=False, default=str), encoding='utf-8')
    return {'protocol': 'lm-evaluation-harness mmlu group; 5-shot; no chat template',
        'limit_per_subject': limit, 'subject_count': len(per_subject),
        'evaluated_samples': selected_count, 'macro_subject_accuracy': macro,
        'micro_accuracy':sum(float(v['accuracy'])*v['sample_count'] for v in per_subject.values())/selected_count,
        'sample_plan':sample_map,
        'per_subject': per_subject, 'samples_sha256': sha256_bytes(raw_path.read_bytes()),
        'raw_result': json_metadata({key:value for key,value in result.items() if key!='samples'})}


def endpoint_list():
    endpoints = []
    evidence_root=WS/'exp/analysis/cloud_comparison_20261022/evidence'
    for short, model_path in MODELS.items():
        for experiment, root in BASELINE_ROOTS.items():
            # The frozen local evidence inventory defines the completed baseline IDs.
            evidence_name = 'exp008' if experiment == 'exp008' else 'exp009'
            batch_manifest = json.loads((evidence_root/evidence_name/'status.json').read_text())
            for run_id in batch_manifest['completed']:
                if run_id.endswith('_smoke') or not run_id.startswith(short):
                    continue
                config = json.loads((evidence_root/evidence_name/run_id/'config.json').read_text())
                endpoints.append({'model_key':short,'model_path':model_path,'run_id':run_id,
                    'method':config['method'],'supervision':config['supervision'],
                    'adapter_path':None if config['method']=='Original' else root+'/'+run_id+'/adapter',
                    'origin':'exp008' if experiment=='exp008' else 'exp009'})
        c2_path='/root/autodl-tmp/www2027_sync_20261022_c2_pilot_v1/exp/runs/exp011_cloud_gsu_c2/runs/batch_20261022_01/'+short+'_C2'
        endpoints.append({'model_key':short,'model_path':model_path,'run_id':short+'_C2',
            'method':'GSU-C2','supervision':'F0','adapter_path':c2_path+'/adapter','origin':'exp011',
            'training_metrics_path':c2_path+'/metrics.json'})
    if len(endpoints)!=16:
        raise RuntimeError(f'Expected16 baseline+GSU endpoints, got{len(endpoints)}')
    return endpoints


def worker(args):
    run=Path(args.run_dir);run.mkdir(parents=True,exist_ok=False)
    started=time.time()
    endpoint=json.loads(Path(args.endpoint_json).read_text(encoding='utf-8'))
    config=endpoint|{'mmlu_limit_per_subject':MMLU_LIMIT_PER_SUBJECT,
        'mmlu_fewshot':MMLU_FEWSHOT,'wiki_dataset':WIKITEXT_DATASET,
        'wiki_config':WIKITEXT_CONFIG,'wiki_revision':args.wiki_revision,
        'ppl_max_length':PPL_MAX_LENGTH,'ppl_stride':PPL_STRIDE,
        'smoke':args.smoke,'lm_eval_version':__import__('lm_eval').__version__}
    baseline.write(run/'config.json',config)
    (run/'run_log.md').write_text(
        '# Cloud capability evaluation run\n\n'
        'Config, sampled MMLU item IDs, dataset revision/hash, environment, source and adapter metadata are archived beside this log.\n'
        'This is an evaluation-only run; no model parameters are updated.\n',encoding='utf-8')
    baseline.write(run/'status.json',{'status':'running','phase':'load','started':protocol_time()})
    try:
        adapter_files=sorted(Path(endpoint['adapter_path']).glob('*.safetensors')) if endpoint['adapter_path'] else []
        if endpoint['adapter_path'] and not adapter_files:
            raise RuntimeError('Adapter safetensors are missing')
        adapter_hashes={path.name:baseline.protocol.sha256_file(path) for path in adapter_files}
        artifact={'adapter_path':endpoint['adapter_path'],'adapter_sha256':adapter_hashes}
        if endpoint.get('training_metrics_path'):
            artifact['training_metrics_sha256']=baseline.protocol.sha256_file(Path(endpoint['training_metrics_path']))
        baseline.write(run/'artifact_manifest.json',artifact)
        tokenizer=AutoTokenizer.from_pretrained(endpoint['model_path'],local_files_only=True,
                                                trust_remote_code=True)
        model=baseline.load_model(endpoint['model_path'])
        if endpoint['adapter_path']:
            model=PeftModel.from_pretrained(model,endpoint['adapter_path'],is_trainable=False)
        model.eval();model.config.use_cache=False
        ppl_corpus,dataset_meta=load_wikitext(args.wiki_revision)
        if args.smoke:ppl_corpus=ppl_corpus[:24000]
        ppl=token_ppl(model,tokenizer,ppl_corpus,token_limit=10000 if args.smoke else None)
        baseline.write(run/'ppl.json',{'dataset':dataset_meta,'metric':ppl})
        if endpoint['method']=='GSU-C2':
            training_metrics=json.loads(Path(endpoint['training_metrics_path']).read_text())
            behavior={'groups':training_metrics['groups'],'sets':training_metrics['sets'],
                      'full_hop2':training_metrics['full_hop2'],
                      'behavior_run_id':endpoint['run_id']}
        else:
            protocol=baseline.protocol
            inputs=protocol.load_inputs()
            behavior=baseline.evaluate(model,tokenizer,inputs)
            behavior['full_hop2']=c2.full_hop2_audit(model,tokenizer,inputs)
            behavior['behavior_run_id']=endpoint['run_id']
        baseline.write(run/'behavior.json',behavior)
        baseline.write(run/'status.json',{'status':'running','phase':'mmlu',
            'ppl_scored_tokens':ppl['scored_tokens'],
            'full_hop2_path_macro':behavior['full_hop2']['all_hop2']['summary']['path_macro_accuracy']})
        mmlu=mmlu_eval(model,tokenizer,limit=1 if args.smoke else MMLU_LIMIT_PER_SUBJECT,
                       output_dir=run)
        baseline.write(run/'mmlu.json',mmlu)
        baseline.write(run/'status.json',{'status':'completed','ended':protocol_time(),
            'seconds':time.time()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
    except BaseException:
        baseline.write(run/'status.json',{'status':'failed','ended':protocol_time(),
            'traceback':traceback.format_exc(),'seconds':time.time()-started})
        raise


def protocol_time():
    import datetime
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')


def queue(args):
    os.chdir(WS)
    root=Path(args.run_dir);root.mkdir(parents=True,exist_ok=False)
    source={str(path.relative_to(WS)):baseline.protocol.sha256_file(path)
        for folder in ('train','damage') for path in (WS/'exp'/folder).rglob('*.py')}
    baseline.write(root/'batch_manifest.json',{'experiment':'exp012_cloud_capability_audit',
       'endpoints':16,'mmlu':'mmlu group, five-shot, first10 test items per subject; maximum570 per model endpoint',
       'ppl':'Salesforce/wikitext wikitext-2-raw-v1 full test; token PPL, maxlen2048,stride512',
       'after_c2_status':args.after_status,'seed':42,'no_training':True,'source':source,
       'models':json.loads(Path('/root/autodl-tmp/models/download_manifest.json').read_text())})
    dependency=Path(args.after_status)
    baseline.write(root/'status.json',{'status':'waiting_for_c2','dependency':str(dependency)})
    while True:
        status=json.loads(dependency.read_text())
        if status['status']=='completed':break
        if status['status'] in ('failed','interrupted'):
            baseline.write(root/'status.json',{'status':'dependency_failed','dependency_status':status});return
        time.sleep(30)
    batch_started=time.time()
    venv=Path(args.venv)
    if not venv.exists():
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(venv)],check=True)
    vpython=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    install=subprocess.run([str(vpython),'-m','pip','install','lm-eval==0.4.11','datasets==3.0.1'],
        text=True,capture_output=True)
    (root/'lm_eval_install.stdout.log').write_text(install.stdout,encoding='utf-8')
    (root/'lm_eval_install.stderr.log').write_text(install.stderr,encoding='utf-8')
    if install.returncode:baseline.write(root/'status.json',{'status':'dependency_failed','exit_code':install.returncode});return
    version=subprocess.check_output([str(vpython),'-c','import lm_eval;print(lm_eval.__version__)']).decode().strip()
    if version!='0.4.11':raise RuntimeError('Unexpected lm-evaluation-harness version '+version)
    baseline.write(root/'environment.txt',subprocess.check_output([str(vpython),'-m','pip','freeze']).decode())
    revision=wiki_text_revision()
    endpoints=endpoint_list()
    baseline.write(root/'endpoints.json',endpoints)
    completed=[]
    for smoke,endpoint in [(True,next(e for e in endpoints if e['method']=='Original' and e['model_key']=='llama32_3b'))]+[(False,e) for e in endpoints]:
        if time.time()-batch_started>24*60*60:
            baseline.write(root/'status.json',{'status':'failed_budget','completed':completed,
                                                'budget_hours':24})
            return
        label=endpoint['model_key']+'_'+endpoint['method']+'_'+endpoint['supervision']+('_smoke' if smoke else '')
        if smoke:label='smoke_'+label
        target=root/label
        command=[str(vpython),str(Path(__file__).resolve()),'--endpoint_json',str(root/(label+'_endpoint.json')),
                 '--run_dir',str(target),'--wiki_revision',revision]+(['--smoke'] if smoke else [])
        baseline.write(root/(label+'_endpoint.json'),endpoint)
        if any(baseline.protocol.sha256_file(Path(path))!=digest for path,digest in source.items()):
            raise RuntimeError('Evaluation source changed after batch freeze')
        baseline.write(root/'status.json',{'status':'running','current':label,'completed':completed,'command':command})
        with (root/(label+'.stdout.log')).open('x') as out,(root/(label+'.stderr.log')).open('x') as err:
            try:
                subprocess.run(command,stdout=out,stderr=err,check=True,timeout=90*60)
            except subprocess.TimeoutExpired:
                baseline.write(target/'status.json',{'status':'timeout','seconds':90*60,
                    'message':'Endpoint timed out; partial outputs preserved'})
                raise
        if smoke:
            smoke_metrics=json.loads((target/'mmlu.json').read_text())
            if smoke_metrics['subject_count']!=57:raise RuntimeError('MMLU smoke did not cover all subjects')
        else:
            completed.append(label)
            baseline.write(root/'partial_results.json',{'completed':completed})
    summarize(root)
    baseline.write(root/'status.json',{'status':'completed','completed':completed,'ended':protocol_time()})


def summarize(root):
    rows=[]
    for folder in root.iterdir():
        if not folder.is_dir() or not (folder/'config.json').exists():continue
        status=json.loads((folder/'status.json').read_text())
        if status['status']!='completed':continue
        config=json.loads((folder/'config.json').read_text())
        if config.get('smoke'):
            continue
        ppl=json.loads((folder/'ppl.json').read_text())
        mmlu=json.loads((folder/'mmlu.json').read_text())
        behavior=json.loads((folder/'behavior.json').read_text())
        rows.append({'model_key':config['model_key'],'model':Path(config['model_path']).name,
            'method':config['method'],'supervision':config['supervision'],'run_id':folder.name,
            'mmlu_macro_57_subjects':mmlu['macro_subject_accuracy'],
            'mmlu_subject_count':mmlu['subject_count'],'mmlu_items':mmlu['evaluated_samples'],
            'token_ppl':ppl['metric']['token_ppl'],'mean_nll_nats':ppl['metric']['mean_nll_nats'],
            'ppl_tokens':ppl['metric']['scored_tokens'],
            'all_hop2_path_macro':behavior['full_hop2']['all_hop2']['summary']['path_macro_accuracy'],
            'covered_hop2_accuracy':behavior['sets']['F1_hop2']['summary']['mcqa_accuracy'],
            'direct_f0_accuracy':behavior['groups']['F0']['summary']['mcqa_accuracy'],
            'retain_heldout_accuracy':behavior['groups']['retain_heldout']['summary']['mcqa_accuracy']})
    if len(rows)!=16:
        raise RuntimeError(f'Expected16 formal endpoints in summary, found{len(rows)}')
    for model in {r['model_key'] for r in rows}:
        orig=next(r for r in rows if r['model_key']==model and r['method']=='Original')
        for row in [r for r in rows if r['model_key']==model]:
            row['mmlu_delta_pp']=100*(row['mmlu_macro_57_subjects']-orig['mmlu_macro_57_subjects'])
            row['ppl_ratio_to_original']=row['token_ppl']/orig['token_ppl']
            row['ppl_delta_nll_nats']=row['mean_nll_nats']-orig['mean_nll_nats']
    baseline.write(root/'comparison.json',{'rows':rows,'mmlu_direction':'higher is better',
        'ppl_direction':'lower is better within tokenizer/model; cross-model absolute values are not ranked',
        'mmlu_protocol':'5-shot, at most10 official test examples per each of57 subjects; pilot subset',
        'ppl_protocol':'WikiText-2 raw test, per-model tokenizer, context<=2048, stride512',
        'status':'completed' if len(rows)==16 else 'partial','missing_endpoints':16-len(rows)})
    lines=['# Cloud capability and forgetting audit','',
        'MMLU-10/subject is a balanced subject-coverage pilot subset, not full MMLU.',
        'Token PPL is compared within each tokenizer/model; do not rank absolute Qwen vs Llama PPL.',
        'All-hop2 path-macro is recomputed on all89 paths for each baseline; C2 reuses its frozen training-run evaluation.','',
        '| Model | Method | All-hop2 path recall ↓ | Covered-hop2 accuracy ↓ | F0 accuracy ↓ | Heldout retain ↑ | MMLU macro ↑ | ΔMMLU pp ↑ | WikiText token PPL ↓ | PPL ratio ↓ | ΔNLL ↓ |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in sorted(rows,key=lambda x:(x['model_key'],x['method'])):
        lines.append(f"| {row['model']} | {row['method']} [{row['supervision']}] | {row['all_hop2_path_macro']:.4f} | {row['covered_hop2_accuracy']:.4f} | {row['direct_f0_accuracy']:.4f} | {row['retain_heldout_accuracy']:.4f} | {row['mmlu_macro_57_subjects']:.4f} | {row['mmlu_delta_pp']:.2f} | {row['token_ppl']:.4f} | {row['ppl_ratio_to_original']:.4f} | {row['ppl_delta_nll_nats']:.4f} |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--queue',action='store_true')
    parser.add_argument('--after_status')
    parser.add_argument('--venv')
    parser.add_argument('--endpoint_json')
    parser.add_argument('--run_dir')
    parser.add_argument('--wiki_revision')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.queue:
        if not args.after_status or not args.venv or not args.run_dir:parser.error('--queue requires --after_status, --venv and --run_dir')
        queue(args)
    else:
        if not args.endpoint_json or not args.run_dir or not args.wiki_revision:parser.error('worker requires endpoint, run dir and wiki revision')
        worker(args)
