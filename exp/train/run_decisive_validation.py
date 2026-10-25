"""Independent exp017: repaired KLUE, expanded supervision, matched nulls, dev-only LR search."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import reproduce_all_baselines as b
import decisive_controls as controls
import run_8b_multiseed as models

WS=Path(__file__).resolve().parents[2]
METHODS=('GA','GradDiff','NPO','SimNPO','RMU')
GRID=(2.5e-5,5e-5,1e-4)
SEEDS=(42,43,44)
PY=models.VENV


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def baseline_command(run,model,method,seed,*,track='matched_lora',supervision='F0',lr=None,smoke=False,dev=None):
    command=[PY,str(WS/'exp/train/reproduce_all_baselines.py'),'--method',method,'--track',track,
             '--model',model,'--seed',str(seed),'--supervision',supervision,'--run_dir',str(run),'--timeout_hours','72']
    if track=='native':command+=['--native_streaming','--sdpa_math']
    else:command+=['--gradient_checkpointing','--aligned_schedule']
    if lr is not None:command+=['--learning_rate',str(lr)]
    if smoke:command+=['--smoke']
    if dev is not None:command+=['--development_plan',str(dev)]
    return command


def initial_jobs(root,plan):
    jobs=[]
    def append(label,model,key,method,seed,command,kind,smoke=False):
        jobs.append(dict(label=label,model=model,model_key=key,method=method,seed=seed,
                         command=command,kind=kind,smoke=smoke))
    for key,model in models.MODELS:
        for smoke,seed in [(True,42)]+[(False,s) for s in SEEDS]:
            label=f'{key}_KLUE_repaired_seed{seed}'+('_smoke' if smoke else '')
            append(label,model,key,'KLUE',seed,baseline_command(root/label,model,'KLUE',seed,track='native',smoke=smoke),'formal',smoke)
        for method in ('GA-F0F1','Cg','C1R','C2D'):
            for smoke,seed in [(True,42)]+[(False,s) for s in SEEDS]:
                label=f'{key}_{method}_seed{seed}'+('_smoke' if smoke else '')
                if method=='GA-F0F1':
                    command=baseline_command(root/label,model,'GA',seed,supervision='F0F1',smoke=smoke)
                else:
                    command=[PY,str(WS/'exp/train/run_local_gsu_c_group.py'),'--cloud_architecture',
                         '--variant',method,'--seed',str(seed),'--model',model,'--gradient_checkpointing',
                         '--run_dir',str(root/label)]+(['--smoke'] if smoke else [])
                append(label,model,key,method,seed,command,'formal',smoke)
        for method in METHODS:
            for lr in GRID:
                label=f'{key}_dev_{method}_lr{lr:g}'
                append(label,model,key,method,42,baseline_command(root/label,model,method,42,lr=lr,dev=plan),'development')
    return jobs


def measured(root,job):
    path=root/job['label']
    if job['smoke'] or job['kind']=='development':return None
    if job['method'] in ('Cg','C1R','C2D'):
        p=path/'audit'
    else:p=path
    state=p/'status.json'
    if not state.exists() or json.loads(state.read_text(encoding='utf-8')).get('status')!='completed':return None
    needed=[p/name for name in ('behavior.json','mmlu.json','ppl.json')]
    if not all(x.exists() for x in needed):return None
    behavior,mmlu,ppl=[json.loads(x.read_text(encoding='utf-8')) for x in needed]
    return dict(model=job['model_key'],method=job['method'],seed=job['seed'],run_id=job['label'],
        all89=behavior['full_hop2']['all_hop2']['summary']['path_macro_accuracy'],
        f0=behavior['groups']['F0']['summary']['mcqa_accuracy'],
        mmlu=mmlu['macro_subject_accuracy'],ppl=ppl['metric']['token_ppl'],
        evidence=str(p),sha256={x.name:b.rl.digest(x) for x in needed})


def summarize(root,jobs):
    rows=[r for job in jobs if (r:=measured(root,job)) is not None]
    write(root/'comparison.json',{'rows':rows,'notes':'development trials excluded; matched nulls and extended supervision separate'})
    lines=['# 决定性补充实验 exp017','',
      'KLUE修复、GA-F0+F1、Cg、C1R、C2D和开发集调参确认。仅列真实完成结果，开发试验不混入正式测试。',
      'C1R是C1权重在同模块类型内的置换；C2D保留真实组件评分，改为同秩随机方向。',
      '| 模型 | 方法 | seed | F0 ↓ | 全89路径宏回忆 ↓ | MMLU ↑ | PPL ↓ |',
      '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['method']} | {r['seed']} | {r['f0']:.4f} | {r['all89']:.4f} | {r['mmlu']:.4f} | {r['ppl']:.4f} |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    coverage=['| 项目 | 当前安排 | 已完成正式端点 |','|---|---|---:|']
    for method in ('KLUE','GA-F0F1','Cg','C1R','C2D'):
        coverage.append(f"| {method} | 两模型×seed42/43/44，各自smoke门检 | {sum(r['method']==method for r in rows)} |")
    coverage.append('| 强baseline开发集调参 | 5方法×3学习率×2模型，选参后另作三种子确认 | 不计作正式测试 |')
    for version in ('report_v2','report_v3_klue_pending'):
        path=WS/'exp/analysis/cloud_comparison_20261022'/version/'README.md'
        if not path.exists():continue
        text=path.read_text(encoding='utf-8');marker='## 决定性补充比较（exp017）'
        if marker in text:text=text[:text.index(marker)].rstrip()
        path.write_text(text+'\n\n'+marker+'\n\n'+'\n'.join(coverage)+'\n\n'+'\n'.join(lines[2:])+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run_dir',required=True);p.add_argument('--after',required=True)
    p.add_argument('--plan_only',action='store_true');p.add_argument('--max_days',type=float,default=21.)
    args=p.parse_args();root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    inputs=b.base.protocol.load_inputs()
    dev=controls.unused_development_split(inputs['clusters'],inputs['indices']);plan=root/'development_plan.json';write(plan,dev)
    jobs=initial_jobs(root,plan);sources=b.source_manifest()
    sources.update({str(f.relative_to(WS)):b.rl.digest(f) for f in (WS/'exp/train/tests').glob('*.py')})
    write(root/'manifest.json',dict(created=b.base.protocol.timestamp(),source=sources,jobs=jobs,after=args.after,
        models=models.MODELS,seeds=SEEDS,lr_grid=GRID,development_methods=METHODS,dev_plan=dev,
        selection_rule='UA<=1/3, retain drop<=10pp, min dev path recall; no feasible config -> no winner',
        tuning_seed=42,confirmation_seeds=SEEDS,pending_tuned_jobs='up to30 F0 confirmations plus up to6 strongest F0F1 confirmations',
        claim_scope='follow-up development after exploratory exp016; not untouched preregistered original confirmatory evidence'))
    write(root/'status.json',{'status':'planned','completed':[],'failures':[]});summarize(root,jobs)
    if args.plan_only:
        print(json.dumps({'initial_jobs':len(jobs),'development_trials':30,'fixed_formal_endpoints':30,'max_selected_confirmations':36}));return
    done=[];failures=[];failed_gates=set();start=time.time()
    while True:
        path=Path(args.after)
        state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'status':'not_created'}
        if state['status'] in ('completed','completed_with_failures','budget_exhausted'):break
        if state['status'] in ('failed','interrupted','preflight_failed','dependency_failed'):
            write(root/'status.json',{'status':'dependency_failed','dependency':state});return
        write(root/'status.json',{'status':'waiting_for_dependency','dependency':args.after,'completed':done,'failures':failures})
        time.sleep(30)
    def run(job):
        if time.time()-start>args.max_days*86400:raise TimeoutError('Queue budget exhausted')
        for file,sha in sources.items():
            if b.rl.digest(WS/file)!=sha:raise RuntimeError('Source changed: '+file)
        signature=(job['model_key'],job['method'])
        if not job['smoke'] and signature in failed_gates:
            failures.append({'label':job['label'],'reason':'own_smoke_failed'});return
        write(root/'status.json',{'status':'running','current':job['label'],'completed':done,'failures':failures})
        with (root/(job['label']+'.stdout.log')).open('x') as out,(root/(job['label']+'.stderr.log')).open('x') as err:
            try:code=subprocess.run(job['command'],cwd=WS,stdout=out,stderr=err,timeout=74*3600).returncode
            except subprocess.TimeoutExpired:code=124
        if code:
            failures.append({'label':job['label'],'exit_code':code})
            if job['smoke']:failed_gates.add(signature)
        else:
            done.append(job['label'])
            if job['method'] in ('Cg','C1R','C2D') and not job['smoke']:
                path=root/job['label'];ep=path/'endpoint.json'
                write(ep,dict(model_key=job['model_key'],model_path=job['model'],method=job['method'],
                    run_id=job['label'],supervision='F0',adapter_path=str(path/'adapter'),origin='exp017'))
                cmd=[PY,str(WS/'exp/train/evaluate_cloud_capabilities.py'),'--endpoint_json',str(ep),
                     '--run_dir',str(path/'audit'),'--wiki_revision',b.WIKI_REVISION]
                with (path/'audit.stdout.log').open('x') as out,(path/'audit.stderr.log').open('x') as err:
                    audit=subprocess.run(cmd,cwd=WS,stdout=out,stderr=err,timeout=6*3600)
                if audit.returncode:failures.append({'label':job['label']+'_audit','exit_code':audit.returncode})
        summarize(root,jobs)
    try:
        # Real BF16 padded math-SDPA regression on released GPU, then true8B KLUE smoke.
        test=[PY,'-m','pytest','exp/train/tests/test_decisive_controls.py','-q','--basetemp',str(root/'pytest_preflight')]
        with (root/'preflight.stdout.log').open('x') as out,(root/'preflight.stderr.log').open('x') as err:
            result=subprocess.run(test,cwd=WS,stdout=out,stderr=err,timeout=600)
        if result.returncode:raise RuntimeError('Decisive control numerical preflight failed')
        for job in list(jobs):run(job)
        selection={};new_jobs=[]
        for key,model in models.MODELS:
            winners=[]
            for method in METHODS:
                trials=[]
                for job in jobs:
                    if job['kind']=='development' and job['model_key']==key and job['method']==method:
                        path=root/job['label']/'development_metrics.json'
                        if path.exists():trials.append(json.loads(path.read_text(encoding='utf-8'))|{'trial':job['label']})
                pick=controls.select_trial(trials);selection[key+'|'+method]=pick
                if pick['selected'] is None:continue
                winner=pick['selected'];winners.append(winner)
                smoke_label=f'{key}_{method}_devselected_seed42_smoke'
                new_jobs.append(dict(label=smoke_label,model=model,model_key=key,method=method+'-devselected',seed=42,
                    command=baseline_command(root/smoke_label,model,method,42,lr=winner['learning_rate'],smoke=True),kind='confirmation',smoke=True))
                for seed in SEEDS:
                    label=f'{key}_{method}_devselected_seed{seed}'
                    new_jobs.append(dict(label=label,model=model,model_key=key,method=method+'-devselected',seed=seed,
                        command=baseline_command(root/label,model,method,seed,lr=winner['learning_rate']),kind='confirmation',smoke=False))
            if winners:
                strongest=min(winners,key=lambda r:(r['multihop_macro'],r['direct_accuracy'],r['method']))
                selection[key+'|strongest']=strongest
                method=strongest['method'];smoke_label=f'{key}_{method}_devselected_F0F1_seed42_smoke'
                new_jobs.append(dict(label=smoke_label,model=model,model_key=key,method=method+'-devselected-F0F1',seed=42,
                    command=baseline_command(root/smoke_label,model,method,42,lr=strongest['learning_rate'],supervision='F0F1',smoke=True),kind='confirmation',smoke=True))
                for seed in SEEDS:
                    method=strongest['method'];label=f'{key}_{method}_devselected_F0F1_seed{seed}'
                    new_jobs.append(dict(label=label,model=model,model_key=key,method=method+'-devselected-F0F1',seed=seed,
                        command=baseline_command(root/label,model,method,seed,lr=strongest['learning_rate'],supervision='F0F1'),kind='confirmation',smoke=False))
        write(root/'development_selection.json',selection)
        write(root/'selected_confirmation_jobs.json',new_jobs)
        jobs.extend(new_jobs)
        for job in new_jobs:run(job)
        write(root/'status.json',{'status':'completed' if not failures else 'completed_with_failures','completed':done,
                                 'failures':failures,'ended':b.base.protocol.timestamp()})
    except BaseException:
        write(root/'status.json',{'status':'failed','completed':done,'failures':failures,'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()

