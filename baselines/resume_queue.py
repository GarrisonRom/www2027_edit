"""Replace the stopped monolithic evaluation queue without touching trained weights."""
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
from .resume_evaluation import write,read,checkpoint_identity,prepare_items,item_summary
from .resume_evaluation import jsonlines,validate_item
from .runtime import load
from .evaluate import records
from .protocol import dataset_manifest

METHODS=('Original','GA','GradDiff','NPO','SimNPO','RMU','DPOmis','DPOrej','CEU','WGA','SatImp','UNDIAL')
PHASES=('scoring','ppl','mmlu','generation')
HOURS=dict(scoring=2,generation=2,ppl=1,mmlu=2)


def summarize(root,state):
    rows=[];lines=['# Resumable evaluation progress','','Existing11trained checkpoints unchanged. Original plus all endpoints evaluated independently.',
        'Partial metrics are provisional; missing capability data is failure, never zero accuracy.','',
        '| Method | Scoring | Generation | PPL | MMLU |','|---|---|---|---|---|']
    for method in METHODS:
        row=dict(method=method)
        for phase in PHASES:
            folder=root/method/phase;status=read(folder/'status.json') if (folder/'status.json').exists() else dict(status='pending')
            m=read(folder/'metrics.json') if (folder/'metrics.json').exists() else None
            row[phase]=dict(status=status,metrics=m)
        rows.append(row)
        def cell(phase):
            s=row[phase]['status'];return f"{s['status']} {s.get('done','')}/{s.get('total','')}"
        lines.append('| '+method+' | '+' | '.join(cell(p) for p in ('scoring','generation','ppl','mmlu'))+' |')
    write(root/'comparison.json',dict(rows=rows))
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--model',required=True);p.add_argument('--local-data');p.add_argument('--import-stages');a=p.parse_args()
    phases=('ppl','mmlu','scoring','generation') if a.local_data else PHASES
    old=Path(a.previous).resolve();root=Path(a.run_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    core=load();verify();inputs=core.base.protocol.load_inputs();items=records(inputs)
    # Validate old and current scoring code BEFORE importing any result. Runtime dependencies
    # are independently verified through their original source lock.
    for name in ('protocol.py','evaluate.py'):
        if sha(ROOT/name)!=sha(old/'snapshot/baselines'/name):raise ValueError('Scoring protocol changed: '+name)
    old_manifest=read(old/'manifest.json');identity_base=checkpoint_identity(a.model,None)['base']
    if identity_base!=old_manifest['model_files']:raise ValueError('Base model differs from old batch')
    endpoints={}
    for method in METHODS:
        checkpoint=None if method=='Original' else old/(method+'_formal')/'checkpoint'
        if checkpoint is not None:
            status=read(checkpoint.parent/'status.json')
            if status['status']!='training_completed':raise ValueError('Not a completed training endpoint: '+method)
        endpoints[method]=dict(checkpoint=None if checkpoint is None else str(checkpoint),
            identity=dict(base=identity_base,checkpoint=None if checkpoint is None else
                {p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()}))
        imported=old/('Original_formal/evaluation' if method=='Original' else method+'_evaluation')
        if (imported/'manifest.json').exists():
            # Every written old row remains untouched; copy and independently validate it.
            for phase in ('scoring','generation'):
                output=root/method/phase;output.mkdir(parents=True)
                done=prepare_items(output,items,phase,endpoints[method]['identity'],imported)
                write(output/'identity.json',endpoints[method]['identity'])
                write(output/'metrics.json',item_summary(done,len(items),phase))
                write(output/'status.json',dict(status='completed' if len(done)==len(items) else 'pending',done=len(done),total=len(items)))
    if a.import_stages:
        source=Path(a.import_stages).resolve();byid={r['id']:r for r in items}
        for method in METHODS:
            for phase in ('scoring','generation'):
                origin=source/method/phase
                if not (origin/'manifest.json').exists():continue
                prior=read(origin/'manifest.json')
                if prior['identity']!=endpoints[method]['identity']:raise ValueError('Stage import checkpoint mismatch')
                if prior['input_sha256']!=dataset_manifest(items):raise ValueError('Stage import input mismatch')
                folder=root/method/phase;folder.mkdir(parents=True,exist_ok=True)
                done=prepare_items(folder,items,phase,endpoints[method]['identity'])
                count=0
                with (folder/'items.jsonl').open('a',encoding='utf-8') as f:
                    for row in jsonlines(origin/'items.jsonl'):
                        if row.get('id') not in byid:raise ValueError('Unknown stage-import item')
                        validate_item(row,byid[row['id']],phase)
                        if row['id'] in done:
                            if row!=done[row['id']]:raise ValueError('Conflicting stage import')
                            continue
                        f.write(json.dumps(row,ensure_ascii=False)+'\n');done[row['id']]=row;count+=1
                write(folder/'stage_import_receipt.json',dict(path=str(origin),added=count,total=len(done),sha256=sha(origin/'items.jsonl')))
                write(folder/'identity.json',endpoints[method]['identity']);write(folder/'metrics.json',item_summary(done,len(items),phase))
                write(folder/'status.json',dict(status='completed' if len(done)==len(items) else 'pending',done=len(done),total=len(items)))
    if a.local_data:
        from .local_capability_data import manifest,checked_files
        data_root,data_manifest=manifest(a.local_data)
        checked_files('wikitext',root=data_root)
        for subject in data_manifest['mmlu']['counts']:checked_files('mmlu',subject,data_root)
    snapshot=root/'snapshot';snapshot.mkdir()
    shutil.copytree(ROOT,snapshot/'baselines',ignore=shutil.ignore_patterns('runs','.deps','__pycache__','.pytest_cache'))
    locked={p.relative_to(snapshot).as_posix():sha(p) for p in snapshot.rglob('*') if p.is_file()}
    write(root/'manifest.json',dict(previous=str(old),model=a.model,endpoints=endpoints,source=locked,
        phases=phases,slice_hours=HOURS,batch_hours=72,command=sys.argv,pid=os.getpid(),
        local_capability_data=None if not a.local_data else dict(root=str(data_root),manifest_sha256=sha(data_root/'manifest.json')),
        old_queue_stopped='PIDs178748/186396 stopped by user-authorized repair; old status files retained unchanged',
        no_training=True,original_protocol_preserved=True))
    (root/'environment.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    (root/'run_log.md').write_text('Resumable evaluation repair. Immutable trained models, old per-item result imports validated. New independent score/generation/PPL/MMLU outputs. Source/config/model hashes frozen. Interrupted slices resume; failures explicit.',encoding='utf-8')
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(ROOT/'.deps'),env.get('PYTHONPATH','')])
    env.update(PYTHONUNBUFFERED='1',HF_HUB_ETAG_TIMEOUT='10',HF_HUB_DOWNLOAD_TIMEOUT='20')
    # A missing remote dataset must fail promptly rather than spending the whole GPU slice retrying.
    env['HF_HUB_OFFLINE']='1';env['HF_DATASETS_OFFLINE']='1'
    state=dict(status='running',pid=os.getpid(),jobs=[],note='verified local ModelScope capability data' if a.local_data else 'capabilities use cached data only; missing cache fails explicitly')
    write(root/'queue_status.json',state);summarize(root,state)
    deadline=time.monotonic()+72*3600;blocked=set()
    try:
        while time.monotonic()<deadline:
            pending=0
            for phase in phases:
                for method in METHODS:
                    key=(method,phase);folder=root/method/phase
                    if key in blocked:continue
                    if (folder/'status.json').exists() and read(folder/'status.json')['status']=='completed':continue
                    pending+=1
                    if time.monotonic()>=deadline:break
                    for name,h in locked.items():
                        if sha(snapshot/name)!=h:raise ValueError('Isolated evaluation source changed: '+name)
                    command=[sys.executable,'-B','-m','baselines.resume_evaluation','--base-model',a.model,
                        '--output',str(folder),'--phase',phase,'--hours',str(HOURS[phase])]
                    checkpoint=endpoints[method]['checkpoint']
                    if checkpoint:command+=['--checkpoint',checkpoint]
                    if a.local_data and phase in ('mmlu','ppl'):command+=['--local-data',str(data_root)]
                    job=dict(method=method,phase=phase,command=command,status='running',started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                    state['jobs'].append(job);write(root/'queue_status.json',state)
                    index=len(state['jobs']);prefix=root/f'{index:03d}_{method}_{phase}'
                    print('START',method,phase,flush=True)
                    with prefix.with_suffix('.stdout.log').open('x',encoding='utf-8') as out,prefix.with_suffix('.stderr.log').open('x',encoding='utf-8') as err:
                        process=subprocess.Popen(command,cwd=snapshot,env=env,stdout=out,stderr=err)
                        job['pid']=process.pid;write(root/'queue_status.json',state)
                        try:code=process.wait(timeout=min((HOURS[phase]+.5)*3600,max(1,deadline-time.monotonic())))
                        except subprocess.TimeoutExpired:
                            process.kill();process.wait();code=-1
                            folder.mkdir(parents=True,exist_ok=True)
                            write(folder/'status.json',dict(status='interrupted_timeout',resume_safe=True))
                    job.update(returncode=code,ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                    status=read(folder/'status.json') if (folder/'status.json').exists() else dict(status='failed')
                    job['status']=status['status']
                    if code not in (0,-1):blocked.add(key)
                    write(root/'queue_status.json',state);summarize(root,state)
                    print('END',method,phase,job['status'],flush=True)
            if not pending:break
        statuses=[]
        for method in METHODS:
            for phase in PHASES:
                path=root/method/phase/'status.json'
                statuses.append(read(path)['status'] if path.exists() else 'pending')
        state['status']='completed' if all(s=='completed' for s in statuses) else 'incomplete'
        state['blocked']=[dict(method=m,phase=p) for m,p in sorted(blocked)]
    except BaseException:
        state.update(status='interrupted',traceback=traceback.format_exc());raise
    finally:write(root/'queue_status.json',state);summarize(root,state)


if __name__=='__main__':main()
