"""Resumable independent scoring/generation/PPL/MMLU stages; never trains weights."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time
import traceback
import torch
import torch.nn.functional as F
from .bootstrap import ROOT,sha,verify
from .evaluate import records,aggregate,paper_metrics,text_metrics
from .protocol import scores,classify,prompt,dataset_manifest
from .runtime import load


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    # Windows scanners/readers can briefly lock a destination during atomic replacement.
    for attempt in range(20):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def jsonlines(path,repair=False):
    if not path.exists():return []
    data=path.read_bytes();chunks=data.splitlines(keepends=True);rows=[];offset=0
    for index,chunk in enumerate(chunks):
        try:row=json.loads(chunk)
        except (ValueError,UnicodeDecodeError):
            if index!=len(chunks)-1 or chunk.endswith(b'\n'):raise ValueError('Corrupt non-tail JSONL: '+str(path))
            if repair:
                backup=path.with_name(path.name+f'.partial_tail_{time.time_ns()}')
                backup.write_bytes(chunk)
                with path.open('r+b') as f:f.truncate(offset)
            break
        rows.append(row);offset+=len(chunk)
    if repair and rows and path.stat().st_size and not path.read_bytes().endswith(b'\n'):
        with path.open('ab') as f:f.write(b'\n')
    return rows


def checkpoint_identity(base,checkpoint):
    return dict(base={p.name:sha(p) for p in Path(base).iterdir() if p.is_file()},
        checkpoint=None if checkpoint is None else
            {p.name:sha(p) for p in Path(checkpoint).iterdir() if p.is_file()})


def validate_item(row,expected,stage):
    for key in expected:
        if row.get(key)!=expected[key]:raise ValueError(f'Item metadata differs: {expected["id"]}/{key}')
    if stage=='scoring':
        keys={p+'/'+m for p in ('paper_prompt','raw_qa') for m in ('answer_sum','answer_mean','full_sequence_mean')}
        if set(row.get('scoring',{}))!=keys:raise ValueError('Incomplete scoring item')
        for v in row['scoring'].values():
            if len(v['values'])!=3 or not all(math.isfinite(x) for x in v['values']):raise ValueError('Invalid scores')
            if any(v[k]!=value for k,value in classify(v['values']).items()):raise ValueError('Invalid classification')
    else:
        generation=row.get('generation',{})
        if not isinstance(generation.get('text'),str):raise ValueError('Missing generation')
        if any(generation[k]!=v for k,v in text_metrics(generation['text'],expected['answer']).items()):
            raise ValueError('Generation metric mismatch')


def prepare_items(output,items,stage,identity,import_dir=None):
    manifest=dict(version=1,stage=stage,input_sha256=dataset_manifest(items),identity=identity,
        expected=len(items),protocol='same Figure8 and raw QA scoring as frozen formal batch',
        generation=dict(max_new_tokens=64,do_sample=False),source_hashes={
            name:sha(ROOT/name) for name in ('protocol.py','evaluate.py','resume_evaluation.py')})
    mp=output/'manifest.json'
    if mp.exists():
        old=read(mp)
        for k in ('stage','input_sha256','identity','generation','source_hashes'):
            if old[k]!=manifest[k]:raise ValueError('Resume contract mismatch: '+k)
    else:write(mp,manifest)
    byid={r['id']:r for r in items};done={};path=output/'items.jsonl'
    for row in jsonlines(path,repair=True):
        if row.get('id') not in byid:raise ValueError('Unknown resumed item')
        validate_item(row,byid[row['id']],stage)
        if row['id'] in done:raise ValueError('Duplicate resumed item')
        done[row['id']]=row
    if import_dir is not None:
        old=Path(import_dir);prior=read(old/'manifest.json')
        if prior['input_sha256']!=manifest['input_sha256'] or prior.get('smoke'):
            raise ValueError('Legacy input manifest mismatch')
        imported=0
        with path.open('a',encoding='utf-8') as f:
            for original in jsonlines(old/'items.jsonl'):
                if original.get('id') not in byid:raise ValueError('Unknown legacy item')
                row=dict(byid[original['id']])
                row[stage if stage=='scoring' else 'generation']=original[stage if stage=='scoring' else 'generation']
                validate_item(row,byid[row['id']],stage)
                if row['id'] in done:
                    if row!=done[row['id']]:raise ValueError('Conflicting legacy and resumed row')
                    continue
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                done[row['id']]=row;imported+=1
        write(output/'import_receipt.json',dict(path=str(old),sha256=sha(old/'items.jsonl'),
            imported=imported,total_done=len(done),original_unchanged=True))
    return done


def item_summary(done,total,stage):
    ordered=list(done.values())
    if stage=='scoring':
        result=aggregate(ordered) if ordered else {}
        required=('forget/hop1_unlearn','forget/hop1_test','test/hop1_unlearn',
            'forget/hop2_test','test/hop2_test','forget/same_object','test/same_object')
        if all(k in result for k in required):result['paper_metrics']=paper_metrics(result)
    else:
        grouped=defaultdict(list)
        for row in ordered:grouped[row['group']].append(row['generation'])
        result={g:dict(n=len(rows),**{k:sum(r[k] for r in rows)/len(rows) for k in ('exact_match','token_f1')}) for g,rows in grouped.items()}
    return dict(stage=stage,completed_items=len(done),expected_items=total,complete=len(done)==total,
        provisional=len(done)!=total,metrics=result)


def item_stage(model,tok,items,output,stage,done,seconds):
    start=time.monotonic();model.eval()
    with (output/'items.jsonl').open('a',encoding='utf-8') as log:
        for item in items:
            if item['id'] in done:continue
            if time.monotonic()-start>=seconds:
                write(output/'status.json',dict(status='paused_budget',done=len(done),total=len(items)));break
            row=dict(item)
            if stage=='scoring':
                row['scoring']={}
                for prefix,prompted in (('paper_prompt',True),('raw_qa',False)):
                    for mode,values in scores(model,tok,item,prompted).items():
                        row['scoring'][prefix+'/'+mode]=dict(classify(values),values=values)
            else:
                q=tok(prompt(item),return_tensors='pt').to(model.device)
                with torch.no_grad():prediction=model.generate(**q,max_new_tokens=64,do_sample=False,pad_token_id=tok.eos_token_id)
                text=tok.decode(prediction[0,q['input_ids'].shape[1]:],skip_special_tokens=True)
                row['generation']=dict(text=text,**text_metrics(text,item['answer']))
            validate_item(row,item,stage)
            log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush();done[item['id']]=row
            write(output/'status.json',dict(status='running',done=len(done),total=len(items),last_id=item['id']))
            if len(done)%25==0:
                write(output/'metrics.json',item_summary(done,len(items),stage))
                print(stage,len(done),'/',len(items),flush=True)
    result=item_summary(done,len(items),stage);write(output/'metrics.json',result)
    write(output/'status.json',dict(status='completed' if result['complete'] else 'paused_budget',done=len(done),total=len(items)))


def ppl_stage(core,model,tok,output,seconds,local_root=None,token_limit=None):
    if local_root:
        from .local_capability_data import load_wikitext
        corpus,meta=load_wikitext(local_root)
    else:corpus,meta=core.capability.load_wikitext(core.WIKI_REVISION)
    ids=tok(corpus,return_tensors='pt',add_special_tokens=False).input_ids
    if tok.bos_token_id is not None:ids=torch.cat((torch.tensor([[tok.bos_token_id]],dtype=ids.dtype),ids),dim=1)
    if token_limit is not None:ids=ids[:,:token_limit]
    ids=ids[0];context=getattr(model.config,'max_position_embeddings',2048) or 2048
    length=min(2048,int(context));stride=min(512,length-1)
    windows=core.capability.token_windows(len(ids),length,stride)
    contract=dict(dataset=meta,input_tokens=len(ids),max_length=length,stride=stride,smoke=token_limit is not None)
    if (output/'dataset.json').exists() and read(output/'dataset.json')!=contract:raise ValueError('PPL dataset contract changed')
    write(output/'dataset.json',contract)
    done={}
    for row in jsonlines(output/'windows.jsonl',repair=True):
        i=row['index']
        if i in done or row['window']!=windows[i] or not math.isfinite(row['nll']):raise ValueError('Bad PPL window')
        done[i]=row
    start=time.monotonic();model.eval()
    with (output/'windows.jsonl').open('a',encoding='utf-8') as log:
        for i,w in enumerate(windows):
            if i in done:continue
            if time.monotonic()-start>=seconds:break
            b=ids[w['begin']:w['end']].unsqueeze(0).to(model.device);labels=b.clone()
            labels[:,:w['target_start']-w['begin']]=-100
            with torch.no_grad():
                logits=model(input_ids=b,use_cache=False).logits[:,:-1].float()
                value=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum'))
            if not math.isfinite(value):raise ValueError('Nonfinite PPL')
            row=dict(index=i,window=w,nll=value);done[i]=row
            log.write(json.dumps(row)+'\n');log.flush()
            write(output/'status.json',dict(status='running',done=len(done),total=len(windows)))
    n=sum(windows[i]['target_count'] for i in done);nll=sum(r['nll'] for r in done.values());complete=len(done)==len(windows)
    if complete and n!=len(ids)-1:raise ValueError('PPL target count mismatch')
    mean=nll/n if n else None
    write(output/'metrics.json',dict(complete=complete,provisional=not complete,scored_tokens=n,
        total_nll=nll,mean_nll=mean,token_ppl=math.exp(mean) if mean is not None and mean<709 else None,
        smoke=token_limit is not None,source=meta))
    write(output/'status.json',dict(status='completed' if complete else 'paused_budget',done=len(done),total=len(windows)))


def mmlu_subjects(manager):
    names=[]
    for name,entry in manager.task_index.items():
        if entry.get('type')!='task':continue
        path=entry.get('yaml_path')
        if not isinstance(path,str):continue
        path=Path(path)
        if name.startswith('mmlu_') and path.parent.name=='default' and path.parent.parent.name=='mmlu':
            names.append(name)
    names.sort()
    if len(names)!=57:raise ValueError(f'Expected57 standard MMLU subjects,got{len(names)}')
    return names


def mmlu_stage(model,tok,output,seconds,local_root=None,smoke=False):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager,get_task_dict
    manager=TaskManager();names=mmlu_subjects(manager)
    if smoke:names=names[:1]
    source=None
    if local_root:
        from .local_capability_data import manifest,mmlu_task_config
        base,meta=manifest(local_root)
        source=dict(name='ModelScope',manifest_sha256=sha(base/'manifest.json'),root=str(base))
    contract=dict(subjects=names,num_fewshot=5,seed=42,chat_template=False,full=not smoke,source=source,smoke=smoke)
    if (output/'protocol.json').exists() and read(output/'protocol.json')!=contract:raise ValueError('MMLU contract changed')
    write(output/'protocol.json',contract)
    harness=HFLM(model,tokenizer=tok,batch_size=1,max_batch_size=1,device='cuda')
    start=time.monotonic();done={}
    for name in names:
        path=output/(name+'.json')
        if path.exists():done[name]=read(path);continue
        if time.monotonic()-start>=seconds:break
        if local_root:
            from lm_eval.api.task import ConfigurableTask
            config=mmlu_task_config(manager,name,local_root)
            task=ConfigurableTask(config=config);tasks=[config]
        else:
            tree=get_task_dict([name],task_manager=manager);task=tree[name];tasks=[name]
        fingerprints={k:getattr(ds,'_fingerprint',None) for k,ds in task.dataset.items()}
        extra=dict(limit=2) if smoke else {}
        r=simple_evaluate(model=harness,tasks=tasks,num_fewshot=5,batch_size=1,log_samples=True,
            apply_chat_template=False,random_seed=42,numpy_random_seed=42,torch_random_seed=42,fewshot_random_seed=42,bootstrap_iters=0,**extra)
        samples=r['samples'][name];acc=[float(v) for k,v in r['results'][name].items() if k.split(',')[0]=='acc']
        if len(acc)!=1:raise ValueError('Ambiguous MMLU accuracy')
        sample_path=output/(name+'.items.jsonl')
        with sample_path.open('w',encoding='utf-8') as f:
            for sample in samples:f.write(json.dumps(sample,default=str,ensure_ascii=False)+'\n')
        documents=task.test_docs();expected=min(2,len(documents)) if smoke else len(documents)
        if len(samples)!=expected:raise ValueError('MMLU incomplete subject sample count')
        value=dict(n=len(samples),accuracy=acc[0],dataset_fingerprints=fingerprints,samples_sha256=sha(sample_path),source=source)
        write(path,value);done[name]=value
        write(output/'status.json',dict(status='running',done=len(done),total=len(names),last_subject=name))
    total=sum(r['n'] for r in done.values())
    write(output/'metrics.json',dict(complete=len(done)==len(names),provisional=len(done)!=len(names),full=not smoke,smoke=smoke,source=source,
        subjects=done,evaluated_samples=total,macro_accuracy=sum(r['accuracy'] for r in done.values())/len(done) if done else None,
        micro_accuracy=sum(r['accuracy']*r['n'] for r in done.values())/total if total else None))
    write(output/'status.json',dict(status='completed' if len(done)==len(names) else 'paused_budget',done=len(done),total=len(names)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-model',required=True);p.add_argument('--checkpoint')
    p.add_argument('--output',required=True);p.add_argument('--phase',choices=['scoring','generation','ppl','mmlu'],required=True)
    p.add_argument('--import-dir');p.add_argument('--hours',type=float,default=4);p.add_argument('--smoke-limit',type=int)
    p.add_argument('--local-data');p.add_argument('--capability-smoke',action='store_true')
    a=p.parse_args();core=load();output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=True)
    try:
        identity=checkpoint_identity(a.base_model,a.checkpoint)
        if (output/'identity.json').exists() and read(output/'identity.json')!=identity:raise ValueError('Checkpoint changed')
        write(output/'identity.json',identity)
        tok=core.AutoTokenizer.from_pretrained(a.base_model,local_files_only=True)
        items=records(core.base.protocol.load_inputs())
        if a.smoke_limit is not None:items=items[:a.smoke_limit]
        done={}
        if a.phase in ('scoring','generation'):
            done=prepare_items(output,items,a.phase,identity,a.import_dir)
            write(output/'metrics.json',item_summary(done,len(items),a.phase))
            if len(done)==len(items):
                write(output/'status.json',dict(status='completed',done=len(done),total=len(items)));return
        write(output/'status.json',dict(status='running',phase=a.phase,step='loading_model',done=len(done)))
        model=core.base.load_model(a.base_model)
        if a.checkpoint:
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,a.checkpoint)
        else:model.config.use_cache=False  # Match the original formal worker's generation contract.
        if a.phase in ('scoring','generation'):item_stage(model,tok,items,output,a.phase,done,a.hours*3600)
        elif a.phase=='ppl':ppl_stage(core,model,tok,output,a.hours*3600,a.local_data,2048 if a.capability_smoke else None)
        else:mmlu_stage(model,tok,output,a.hours*3600,a.local_data,a.capability_smoke)
    except BaseException:
        write(output/'status.json',dict(status='failed',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
