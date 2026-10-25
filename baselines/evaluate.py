"""Common evaluation: FaithUn prompt, conditional-score sensitivity, MMLU and PPL."""
from collections import defaultdict
import json
import re
from pathlib import Path
import torch
from .protocol import prompt,scores,classify,dataset_manifest


def records(inputs):
    rows=[]
    for split,indices in inputs['indices'].items():
        if split not in ('forget','retain','test'):continue
        for ci in indices:
            c=inputs['clusters'][ci][2]
            for field in ('hop1_unlearn','hop1_test'):
                fact=c[field]
                for qi,q in enumerate(fact['questions']):
                    rows.append(dict(id=f'{split}:{ci}:{field}:{qi}',group=f'{split}/{field}',
                        question=q,answer=fact['answers'],false_options=fact['false_answer_options'],path=None))
            if split=='retain':continue
            for field in ('hop2_test','same_object'):
                fact=c.get(field,{})
                for pi,(qs,a,fo) in enumerate(zip(fact.get('questions',[]),fact.get('answers',[]),fact.get('false_answer_options',[]))):
                    for qi,q in enumerate(qs):
                        rows.append(dict(id=f'{split}:{ci}:{field}:{pi}:{qi}',group=f'{split}/{field}',
                            question=q,answer=a,false_options=fo,path=f'{split}:{ci}:{field}:{pi}'))
    for row in inputs['F1']:rows.append(dict(row,group='guidance/F1',path=None))
    coverage={tuple(p) for f in inputs['F1'] for p in f['path_ids']}
    for r in rows:
        fields=r['id'].split(':')
        r['covered_F1']=r['group']=='forget/hop2_test' and (int(fields[1]),int(fields[3])) in coverage
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate evaluation IDs')
    return rows


def normalize(text):return ' '.join(re.sub(r'[^\w\s]',' ',text.lower()).split())


def text_metrics(predicted,answer):
    from collections import Counter
    p=normalize(predicted).split();a=normalize(answer).split()
    common=sum((Counter(p)&Counter(a)).values())
    f1=2*common/(len(p)+len(a)) if p and a else float(p==a)
    return dict(exact_match=int(normalize(predicted)==normalize(answer)),token_f1=f1)


def aggregate(items):
    groups=defaultdict(list)
    for r in items:
        groups[r['group']].append(r)
        if r['covered_F1']:groups['forget/covered_hop2'].append(r)
        elif r['group']=='forget/hop2_test':groups['forget/noncovered_hop2'].append(r)
    result={}
    for name,rows in groups.items():
        metrics={}
        for key in rows[0]['scoring']:
            values=[r['scoring'][key] for r in rows];paths=defaultdict(list)
            for r,v in zip(rows,values):
                if r['path']:paths[r['path']].append(v['correct'])
            metrics[key]=dict(n=len(rows),accuracy=sum(v['correct'] for v in values)/len(values),
                ties=sum(v['tie'] for v in values),mean_margin=sum(v['margin'] for v in values)/len(values))
            if paths:metrics[key]['path_macro_accuracy']=sum(sum(v)/len(v) for v in paths.values())/len(paths)
        if 'generation' in rows[0]:
            metrics['generation']={k:sum(r['generation'][k] for r in rows)/len(rows)
                                   for k in ('exact_match','token_f1')}
        result[name]=metrics
    return result


def paper_metrics(summary):
    key='paper_prompt/answer_sum'
    def a(group):return summary[group][key]['accuracy']
    sf=summary['forget/same_object'][key];st=summary['test/same_object'][key]
    sa=(sf['accuracy']*sf['n']+st['accuracy']*st['n'])/(sf['n']+st['n'])
    ma=((1-a('forget/hop2_test'))+a('test/hop2_test'))/2
    ua_ext=a('forget/hop1_test');ta=a('test/hop1_unlearn')
    return dict(UA=a('forget/hop1_unlearn'),UA_extended=ua_ext,TA=ta,
        SA_pooled=sa,SA_forget_mapped=sf['accuracy'],SA_test_mapped=st['accuracy'],
        MA_forget=a('forget/hop2_test'),MA_test=a('test/hop2_test'),MA=ma,
        Score_pooledSA=((1-ua_ext)+ta+sa+ma)/4,
        Score_forgetSA=((1-ua_ext)+ta+sf['accuracy']+ma)/4,
        score_scope_note='Paper3.5 total formula; SA pooling scope explicit, author-code exact scope not certified')


def evaluate(model,tok,inputs,output,*,limit=None,generation=False,capabilities=True,mmlu_limit=10):
    from .runtime import load
    core=load();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=records(inputs);all_count=len(rows)
    if limit is not None:
        # Stratified smoke: never mislabel a first-N subset as a full evaluation.
        counts=defaultdict(int);chosen=[]
        for row in rows:
            if counts[row['group']]<limit:chosen.append(row);counts[row['group']]+=1
        rows=chosen
    manifest=dict(evaluation_items=len(rows),full_items=all_count,smoke=limit is not None,
        input_sha256=dataset_manifest(rows),primary='paper_prompt/answer_sum',
        prompt_source='FaithUn Figure8 instruction, question, options, Your Answer',
        conventions=['options serialized as JSON list; source figure leaves delimiter unspecified',
            'candidate answer text scored conditionally, exact ties incorrect',
            'raw QA sensitivity reported separately, no chat template'],
        retain_status='retain66 trained for methods using retain; test464 is independent',
        mmlu_protocol='57 subjects,5-shot,seed42; full' if mmlu_limit is None else
            f'57 subjects,5-shot,seed42; sampled{mmlu_limit}/subject,not full MMLU')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    model.eval();results=[]
    with (output/'items.jsonl').open('x',encoding='utf-8') as log:
        for row in rows:
            paper=scores(model,tok,row,True);raw=scores(model,tok,row,False)
            result=dict(row,scoring={})
            for prefix,values in [('paper_prompt',paper),('raw_qa',raw)]:
                for mode,sc in values.items():result['scoring'][prefix+'/'+mode]=dict(classify(sc),values=sc)
            if generation:
                q=tok(prompt(row),return_tensors='pt').to(model.device)
                with torch.no_grad():pred=model.generate(**q,max_new_tokens=64,do_sample=False,
                    pad_token_id=tok.eos_token_id)
                text=tok.decode(pred[0,q['input_ids'].shape[1]:],skip_special_tokens=True)
                result['generation']=dict(text=text,**text_metrics(text,row['answer']))
            results.append(result);log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush()
    summary=aggregate(results)
    summary['paper_metrics']=paper_metrics(summary)
    cap_status='not_requested'
    if capabilities:
        cap_status='completed'
        def wiki():
            corpus,meta=core.capability.load_wikitext(core.WIKI_REVISION)
            return dict(dataset=meta,metric=core.capability.token_ppl(model,tok,corpus,
                token_limit=1000 if limit is not None else None))
        def mmlu():
            if mmlu_limit is None and limit is None:
                from .capabilities import full_mmlu
                return full_mmlu(model,tok,output)
            return core.capability.mmlu_eval(model,tok,limit=1 if limit is not None else mmlu_limit,output_dir=output)
        tasks=(('ppl',wiki),('mmlu',mmlu))
        for key,fn in tasks:
            try:summary[key]=fn()
            except Exception as error:
                summary[key]=dict(status='failed',error=repr(error));cap_status='incomplete'
    summary['capability_status']=cap_status
    summary['evaluation_complete']=cap_status=='completed'
    summary['scope']='smoke' if limit is not None else 'formal'
    (output/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary
