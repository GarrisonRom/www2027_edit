"""Two8B models, seeds42/43/44, all13 baselines and local-worker C1/C2.
All source and recipes are frozen; download and old GPU batch gates are explicit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import reproduce_all_baselines as baseline
import reproduction_losses as losses

WS=Path(__file__).resolve().parents[2]
SEEDS=(42,43,44)
MODELS=[('llama31_8b','/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct'),
        ('qwen3_8b','/root/models8b/Qwen3-8B')]
VENV='/root/autodl-tmp/venvs/lm_eval_0411_capability_20261023_v2/bin/python'


def write(p,x):
    p=Path(p);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def jobs_for(root):
    jobs=[]
    def append(label,kind,model,key,method,track,seed,smoke,command,gate):
        jobs.append(dict(label=label,kind=kind,model=model,model_key=key,method=method,
                         track=track,seed=seed,smoke=smoke,command=command,gate=gate))
    for key,model in MODELS:
        label=key+'_Original'
        append(label,'evaluation',model,key,'Original','none',None,False,
               [VENV,str(WS/'exp/train/evaluate_cloud_capabilities.py'),'--endpoint_json',str(root/(label+'_endpoint.json')),
                '--run_dir',str(root/label),'--wiki_revision',baseline.WIKI_REVISION],key+'_Original')
    specs=[('C1','matched_lora'),('C2','matched_lora'),('KLUE','native')]
    specs += [(m,'matched_lora') for m in losses.METHODS if m!='KLUE']
    specs += [(m,'native') for m in losses.METHODS if m!='KLUE']
    for method,track in specs:
        for key,model in MODELS:
            signature=f'{key}_{method}_{track}'
            for seed,smoke in [(42,True)]+[(s,False) for s in SEEDS]:
                label=signature+(f'_seed{seed}')+('_smoke' if smoke else '')
                if method in ('C1','C2'):
                    command=[VENV,str(WS/'exp/train/run_local_gsu_c_group.py'),'--cloud_architecture',
                             '--variant',method,'--model',model,'--seed',str(seed),
                             '--gradient_checkpointing','--run_dir',str(root/label)]
                else:
                    command=[VENV,str(WS/'exp/train/reproduce_all_baselines.py'),
                             '--method',method,'--track',track,'--model',model,'--seed',str(seed),
                             '--timeout_hours','72','--run_dir',str(root/label)]
                    command += ['--native_streaming'] if track=='native' else ['--gradient_checkpointing','--aligned_schedule']
                if smoke:command+=['--smoke']
                append(label,'training',model,key,method,track,seed,smoke,command,signature)
                if method in ('C1','C2') and not smoke:
                    audit=label+'_audit'
                    append(audit,'audit',model,key,method,track,seed,False,
                         [VENV,str(WS/'exp/train/evaluate_cloud_capabilities.py'),
                          '--endpoint_json',str(root/(audit+'_endpoint.json')),
                          '--run_dir',str(root/audit),'--wiki_revision',baseline.WIKI_REVISION],signature)
    return jobs


def endpoint_for(root,job):
    label=job['label']
    training=label.removesuffix('_audit')
    return dict(model_key=job['model_key'],model_path=job['model'],run_id=training,
                method=job['method'],supervision='none' if job['method']=='Original' else 'F0',
                adapter_path=None if job['method']=='Original' else str(root/training/'adapter'),
                origin='exp016',seed=job['seed'],
                **({} if job['method']=='Original' else {'training_metrics_path':str(root/training/'metrics.json')}))


def read_metric(folder,job):
    if not (folder/'status.json').exists():return None
    state=json.loads((folder/'status.json').read_text(encoding='utf-8'))
    if state['status']!='completed' or job['smoke']:return None
    if job['method'] in ('C1','C2') and job['kind']=='training':return None
    if (folder/'mmlu.json').exists() and (folder/'ppl.json').exists() and (folder/'behavior.json').exists():
        b=json.loads((folder/'behavior.json').read_text(encoding='utf-8'))
        m=json.loads((folder/'mmlu.json').read_text(encoding='utf-8'))
        p=json.loads((folder/'ppl.json').read_text(encoding='utf-8'))
        return dict(model=Path(job['model']).name,method=job['method'],track=job['track'],seed=job['seed'],
                    all89=b['full_hop2']['all_hop2']['summary']['path_macro_accuracy'],
                    f0=b['groups']['F0']['summary']['mcqa_accuracy'],
                    retain=b['groups']['retain_heldout']['summary']['mcqa_accuracy'],
                    retain_is_heldout=job['track']!='native',mmlu=m['macro_subject_accuracy'],ppl=p['metric']['token_ppl'],
                    mean_nll=p['metric']['mean_nll_nats'],run_id=folder.name,
                    evidence=str(folder),mmlu_sha256=losses.digest(folder/'mmlu.json'),ppl_sha256=losses.digest(folder/'ppl.json'))
    return None


def summarize(root,jobs):
    rows=[]
    for job in jobs:
        result=read_metric(root/job['label'],job)
        if result:rows.append(result)
    aggregates=[]
    for identity in sorted({(r['model'],r['method'],r['track']) for r in rows}):
        selected=[r for r in rows if (r['model'],r['method'],r['track'])==identity]
        seeds=[r['seed'] for r in selected if r['seed'] is not None]
        summary=dict(model=identity[0],method=identity[1],track=identity[2],n=len(selected),seeds=seeds,
                     three_seeds_complete=set(seeds)==set(SEEDS))
        for metric in ('all89','f0','retain','mmlu','ppl','mean_nll'):
            values=[r[metric] for r in selected]
            summary[metric]={'mean':statistics.mean(values),'sample_std':statistics.stdev(values) if len(values)>1 else None}
        aggregates.append(summary)
    write(root/'comparison.json',{'records':rows,'aggregates':aggregates,'rank_rule':'no mixing native/LoRA tracks; Original single reference'})
    lines=['# 8B全方法三种子实验','',
           '模型与测试题固定；Original每模型一次，训练种子42/43/44。未满三种子的汇总不是三种子结论。',
           'native使用全部66条retain，其retain指标不是留出测试；LoRA/C1/C2使用前33条。','',
           '| 模型 | 方法 | 轨道 | seed | 全89宏回忆 ↓ | F0 ↓ | MMLU ↑ | WikiText2 PPL ↓ |',
           '|---|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['method']} | {r['track']} | {r['seed']} | {100*r['all89']:.2f}% | {100*r['f0']:.2f}% | {100*r['mmlu']:.2f}% | {r['ppl']:.4f} |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    matrix=['| 方法 | 计划训练数（两模型×三seed） | 已完成三项评测 |','|---|---:|---:|']
    for method in list(losses.METHODS)+['C1','C2']:
        expected=6 if method in ('KLUE','C1','C2') else 12
        complete=sum(r['method']==method for r in rows)
        matrix.append(f'| {method} | {expected} | {complete} |')
    for version in ('report_v2','report_v3_klue_pending'):
        path=WS/'exp/analysis/cloud_comparison_20261022'/version/'README.md'
        if not path.exists():continue
        marker='## 8B三种子完整比较（exp016）'
        text=path.read_text(encoding='utf-8')
        if marker in text:text=text[:text.index(marker)].rstrip()
        path.write_text(text+'\n\n'+marker+'\n\n'+ '\n'.join(matrix)+'\n\n'+ '\n'.join(lines[2:])+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run_dir',required=True)
    p.add_argument('--download_status',required=True);p.add_argument('--after',required=True)
    p.add_argument('--plan_only',action='store_true');p.add_argument('--max_days',type=float,default=21.)
    a=p.parse_args();root=Path(a.run_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    jobs=jobs_for(root)
    source=baseline.source_manifest()
    write(root/'manifest.json',dict(created=baseline.base.protocol.timestamp(),jobs=jobs,seeds=SEEDS,
        models=MODELS,source=source,download_status=a.download_status,after=a.after,max_days=a.max_days,
        training_endpoints=162,original_evaluations=2,
        evaluation='MMLU sample plan seed42 fixed for every training seed; raw QA; same WikiText revision',
        primary='all89 path macro recall',checkpointing='nonreentrant for LoRA/C1/C2; CPU-native saved-tensor offload',
        native_precision='BF16 parameter storage with FP32 CPU optimizer moments, no duplicate FP32 master',
        storage='no overwrite; disk exhaustion explicit, no automatic model/adapter deletion'))
    write(root/'status.json',{'status':'planned','completed':[],'failures':[]})
    summarize(root,jobs)
    if a.plan_only:
        print(json.dumps({'training':162,'original':2,'jobs_with_smokes_and_C_audits':len(jobs)}));return
    start=time.time();done=[];failures=[];gates=set();failed_train=set()
    for dependency,role in [(a.download_status,'models'),(a.after,'previous_GPU_queue')]:
        while True:
            path=Path(dependency)
            state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'status':'not_created'}
            if state['status'] in ('completed','completed_with_failures','budget_exhausted'):
                if role=='models' and state['status']!='completed':raise RuntimeError('Model download incomplete')
                break
            if state['status'] in ('failed','interrupted','dependency_failed'):
                write(root/'status.json',{'status':'dependency_failed','role':role,'dependency':state});return
            write(root/'status.json',{'status':'waiting_for_dependency','role':role,'dependency':dependency,'completed':done,'failures':failures})
            time.sleep(30)
    manifests={}
    for key,model in MODELS:
        manifests[key]=json.loads((Path(model)/'download_manifest_8b.json').read_text(encoding='utf-8'))
        for file in manifests[key]['files']:
            path=Path(model)/file['path']
            if path.stat().st_size!=file['bytes'] or losses.digest(path)!=file['sha256']:
                raise RuntimeError('Downloaded model verification failed: '+str(path))
    write(root/'model_manifests.json',manifests)
    write(root/'status.json',{'status':'running','current':'GPU_execution_parity_tests','completed':done,'failures':failures})
    with (root/'gpu_parity.stdout.log').open('x') as out,(root/'gpu_parity.stderr.log').open('x') as err:
        test=subprocess.run([VENV,'-m','pytest','exp/train/tests/test_8b_multiseed.py','-q',
                '--basetemp',str(root/'pytest_gpu_parity')],cwd=WS,stdout=out,stderr=err,timeout=600)
    if test.returncode:
        write(root/'status.json',{'status':'preflight_failed','reason':'GPU numerical parity tests failed',
                'completed':done,'failures':failures});return
    for job in jobs:
        if time.time()-start>a.max_days*86400:
            write(root/'status.json',{'status':'budget_exhausted','completed':done,'failures':failures});return
        if job['gate'] in gates or (job['kind']=='audit' and job['label'].removesuffix('_audit') in failed_train):
            failures.append({'label':job['label'],'reason':'prerequisite_smoke_or_training_failed'});continue
        for file,sha in source.items():
            if losses.digest(WS/file)!=sha:raise RuntimeError('Frozen source changed: '+file)
        if job['kind'] in ('audit','evaluation'):
            write(root/(job['label']+'_endpoint.json'),endpoint_for(root,job))
        write(root/'status.json',{'status':'running','current':job['label'],'completed':done,'failures':failures})
        with (root/(job['label']+'.stdout.log')).open('x') as out,(root/(job['label']+'.stderr.log')).open('x') as err:
            try:code=subprocess.run(job['command'],cwd=WS,stdout=out,stderr=err,timeout=74*3600).returncode
            except subprocess.TimeoutExpired:code=124
        if code:
            failures.append({'label':job['label'],'exit_code':code})
            if job['smoke']:gates.add(job['gate'])
            if job['kind']=='training':failed_train.add(job['label'])
        else:done.append(job['label'])
        summarize(root,jobs)
    write(root/'status.json',{'status':'completed' if not failures else 'completed_with_failures',
                            'completed':done,'failures':failures,'ended':baseline.base.protocol.timestamp()})

if __name__=='__main__':
    main()

