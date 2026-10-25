"""Durable isolated local LoRA trial batch, followed by common evaluations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from .bootstrap import ROOT,sha,verify
from .recipes import recipe,lora_adapt,validate

METHODS=('GA','GradDiff','NPO','SimNPO','RMU','DPOmis','DPOrej','CEU','WGA','SatImp','UNDIAL')


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def run_child(root,snapshot,env,job,state,deadline):
    if time.time()>=deadline:raise TimeoutError('Batch72hour budget')
    for name,expected in state['snapshot_sha256'].items():
        if sha(snapshot/name)!=expected:raise RuntimeError('Isolated source changed: '+name)
    job.update(status='running',started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    state['jobs'].append(job);write(root/'queue_status.json',state)
    print('START',job['name'],flush=True)
    with (root/(job['name']+'.stdout.log')).open('x',encoding='utf-8') as out, (root/(job['name']+'.stderr.log')).open('x',encoding='utf-8') as err:
        process=subprocess.Popen(job['command'],cwd=snapshot,env=env,stdout=out,stderr=err)
        job['pid']=process.pid;write(root/'queue_status.json',state)
        try:code=process.wait(timeout=min(6*3600,max(1,deadline-time.time())))
        except subprocess.TimeoutExpired:
            process.kill();process.wait();code=-1;job['reason']='six-hour child or batch deadline'
    job.update(returncode=code,status='completed' if code==0 else 'failed',ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    path=Path(job['artifact'])
    if (path/'status.json').exists():
        job['artifact_status']=json.loads((path/'status.json').read_text(encoding='utf-8'))
    if (path/'metrics.json').exists():
        m=json.loads((path/'metrics.json').read_text(encoding='utf-8'))
        job['evaluation_complete']=m.get('evaluation_complete',False)
        if not job['evaluation_complete']:job['status']='evaluation_incomplete'
    write(root/'queue_status.json',state);summarize(root,state)
    print('END',job['name'],job['status'],flush=True)
    return code


def summarize(root,state):
    rows=[]
    for job in state['jobs']:
        if job['phase']=='training' and job['status']=='completed':
            folder=Path(job['artifact']);s=json.loads((folder/'training_summary.json').read_text(encoding='utf-8'))
            rows.append(dict(method=job['method'],phase='training',updates=s['updates'],
                final_training_ua=s['final_training_ua'],stop=s['stop'],source=str(folder)))
        elif job['phase']=='evaluation' and (Path(job['artifact'])/'metrics.json').exists():
            m=json.loads((Path(job['artifact'])/'metrics.json').read_text(encoding='utf-8'))
            rows.append(dict(method=job['method'],phase='evaluation',complete=m['evaluation_complete'],
                paper_metrics=m['paper_metrics'],capability_status=m['capability_status'],source=job['artifact']))
    write(root/'comparison.json',dict(scope='new LoRA single-config trials; not completed hyperparameter search',rows=rows))
    lines=['# New LoRA formal batch','','Single source-specific config per method; not a tuned-method ranking.',
        'PDU deferred for calibration; KLUE excluded. Native/old114step results are not merged.','',
        '| Method | Phase | Updates | Final training UA | Evaluation complete |',
        '|---|---|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['method']} | {row['phase']} | {row.get('updates','')} | {row.get('final_training_ua','')} | {row.get('complete','')} |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--run-dir',required=True)
    args=p.parse_args();verify();root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    model=Path(args.model).resolve()
    configs={m:lora_adapt(recipe(m)) for m in METHODS}
    for cfg in configs.values():validate(cfg)
    snapshot=root/'snapshot';snapshot.mkdir()
    shutil.copytree(ROOT,snapshot/'baselines',ignore=shutil.ignore_patterns('runs','.deps','__pycache__','.pytest_cache'))
    frozen={f.relative_to(snapshot).as_posix():sha(f) for f in snapshot.rglob('*') if f.is_file()}
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'.deps'),env.get('PYTHONPATH','')])
    env['PYTHONUNBUFFERED']='1';env['HF_HUB_ETAG_TIMEOUT']='10';env['HF_HUB_DOWNLOAD_TIMEOUT']='30'
    # Preserve the actual machine's online/offline setting, never fake capability success.
    write(root/'manifest.json',dict(model=str(model),model_files={f.name:sha(f) for f in model.iterdir() if f.is_file()},
        configs=configs,snapshot_sha256=frozen,python=sys.executable,command=sys.argv,pid=os.getpid(),
        deferred={'PDU':'retain epsilon not calibrated; no guessed value','KLUE':'no validated LoRA equivalent'},
        scope='11 method-specific single-configuration formal LoRA trials, not full LR search',
        batch_hours=72,child_hours=6,evaluation='9640 QA records; generation EM/F1; full57subject5shot MMLU; WikiText2 PPL'))
    (root/'environment.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    installed=[]
    for metadata in (ROOT/'.deps').glob('*.dist-info/METADATA'):
        installed.append(dict(path=str(metadata),sha256=sha(metadata)))
    write(root/'eval_dependencies.json',installed)
    (root/'run_log.md').write_text('Frozen isolated source/config/model manifests precede all jobs. Smoke-gated single-config formal training, then common evaluation. Old runs unchanged. PDU deferred, KLUE excluded. Missing capability scores remain incomplete. See formal_batch_card.md.',encoding='utf-8')
    state=dict(status='running',pid=os.getpid(),jobs=[],snapshot_sha256=frozen)
    write(root/'queue_status.json',state);deadline=time.time()+72*3600
    trained=[]
    try:
        for method in METHODS:
            config=root/(method+'.config.json');write(config,configs[method])
            for smoke in (True,False):
                name=method+('_smoke' if smoke else '_formal');folder=root/name
                command=[sys.executable,'-B','-m','baselines.train','--method',method,'--lora-adapted',
                    '--model',str(model),'--run-dir',str(folder),'--config',str(config),'--hours','6']
                command+=['--smoke'] if smoke else ['--train-only']
                job=dict(name=name,method=method,phase='smoke' if smoke else 'training',command=command,artifact=str(folder))
                code=run_child(root,snapshot,env,job,state,deadline)
                if code!=0:break
                if not smoke:trained.append((method,folder/'checkpoint'))
        original=root/'Original_formal'
        command=[sys.executable,'-B','-m','baselines.train','--method','Original','--model',str(model),
            '--run-dir',str(original),'--mmlu-full','--generation']
        run_child(root,snapshot,env,dict(name='Original_evaluation',method='Original',phase='evaluation',
            command=command,artifact=str(original/'evaluation')),state,deadline)
        for method,checkpoint in trained:
            output=root/(method+'_evaluation')
            command=[sys.executable,'-B','-m','baselines.eval_checkpoint','--base-model',str(model),
                '--checkpoint',str(checkpoint),'--output',str(output),'--mmlu-full','--generation']
            run_child(root,snapshot,env,dict(name=method+'_evaluation',method=method,phase='evaluation',
                command=command,artifact=str(output)),state,deadline)
        state['status']='finished_with_failures' if any(j['status'] in ('failed','evaluation_incomplete') for j in state['jobs']) else 'finished'
    except BaseException:
        state.update(status='interrupted',traceback=traceback.format_exc());raise
    finally:
        write(root/'queue_status.json',state);summarize(root,state)


if __name__=='__main__':main()
