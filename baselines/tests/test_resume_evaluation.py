import json
from types import SimpleNamespace
import pytest
import torch
from baselines import resume_evaluation as e
from baselines.protocol import classify,dataset_manifest


def item(i):return dict(id=str(i),group='forget/hop1_test',question='q',answer='a',
    false_options=['b','c'],path=None,covered_F1=False)


def row(i):
    values=[-1.,-2.,-3.]
    return dict(item(i),scoring={p+'/'+m:dict(classify(values),values=values)
        for p in ('paper_prompt','raw_qa') for m in ('answer_sum','answer_mean','full_sequence_mean')},
        generation=dict(text='a',exact_match=1,token_f1=1.))


def legacy(tmp_path,n=2):
    old=tmp_path/'old';old.mkdir()
    items=[item(i) for i in range(n)]
    e.write(old/'manifest.json',dict(input_sha256=dataset_manifest(items),smoke=False))
    (old/'items.jsonl').write_text(''.join(json.dumps(row(i))+'\n' for i in range(n)),encoding='utf-8')
    return old,items


@pytest.mark.parametrize('stage',['scoring','generation'])
def test_import_and_repeat_does_not_duplicate_or_touch_old(tmp_path,stage):
    old,items=legacy(tmp_path);before=(old/'items.jsonl').read_bytes();out=tmp_path/'new';out.mkdir()
    first=e.prepare_items(out,items,stage,{},old)
    again=e.prepare_items(out,items,stage,{},old)
    assert len(first)==len(again)==2
    assert len(e.jsonlines(out/'items.jsonl'))==2
    assert (old/'items.jsonl').read_bytes()==before


def test_resume_rejects_changed_input_model_and_protocol(tmp_path):
    out=tmp_path/'new';out.mkdir();items=[item(0)]
    e.prepare_items(out,items,'scoring',{'model':'a'})
    with pytest.raises(ValueError,match='identity'):e.prepare_items(out,items,'scoring',{'model':'b'})
    with pytest.raises(ValueError,match='input_sha256'):e.prepare_items(out,[dict(item(0),answer='other')],'scoring',{'model':'a'})


def test_interrupted_tail_preserved_and_repaired_only_new_output(tmp_path):
    path=tmp_path/'items.jsonl';valid=json.dumps(row(0)).encode()+b'\n'
    path.write_bytes(valid+b'{"id":')
    assert len(e.jsonlines(path))==1 and path.read_bytes()!=valid
    assert len(e.jsonlines(path,repair=True))==1 and path.read_bytes()==valid
    assert list(tmp_path.glob('*.partial_tail_*'))[0].read_bytes()==b'{"id":'


def test_invalid_middle_record_never_silently_ignored(tmp_path):
    path=tmp_path/'items.jsonl';path.write_bytes(b'{bad}\n'+json.dumps(row(0)).encode()+b'\n')
    with pytest.raises(ValueError):e.jsonlines(path,repair=True)


def test_complete_last_record_without_newline_is_not_lost(tmp_path):
    path=tmp_path/'items.jsonl';path.write_text(json.dumps(row(0)),encoding='utf-8')
    assert len(e.jsonlines(path,repair=True))==1 and path.read_bytes().endswith(b'\n')


def test_resumed_item_not_computed_twice_and_metrics_saved(tmp_path,monkeypatch):
    items=[item(0),item(1)];done={'0':{k:v for k,v in row(0).items() if k!='generation'}}
    (tmp_path/'items.jsonl').write_text(json.dumps(done['0'])+'\n',encoding='utf-8')
    calls=[]
    def score(model,tok,r,prompted):
        calls.append(r['id']);return {m:[-1.,-2.,-3.] for m in ('answer_sum','answer_mean','full_sequence_mean')}
    monkeypatch.setattr(e,'scores',score)
    e.item_stage(SimpleNamespace(eval=lambda:None),None,items,tmp_path,'scoring',done,10)
    assert calls==['1','1']
    assert e.read(tmp_path/'metrics.json')['complete']
    assert len(e.jsonlines(tmp_path/'items.jsonl'))==2


def test_budget_pause_saves_provisional_summary_and_is_resumable(tmp_path):
    items=[item(0)];done={}
    e.item_stage(SimpleNamespace(eval=lambda:None),None,items,tmp_path,'scoring',done,0)
    assert e.read(tmp_path/'status.json')['status']=='paused_budget'
    assert not e.read(tmp_path/'metrics.json')['complete']


def test_import_rejects_bad_classification(tmp_path):
    old,items=legacy(tmp_path,1)
    r=row(0);r['scoring']['paper_prompt/answer_sum']['correct']=0
    (old/'items.jsonl').write_text(json.dumps(r)+'\n',encoding='utf-8')
    out=tmp_path/'new';out.mkdir()
    with pytest.raises(ValueError,match='classification'):e.prepare_items(out,items,'scoring',{},old)


def test_ppl_windows_match_full_token_count_and_do_not_repeat(tmp_path):
    import math
    from baselines.runtime import load
    original=load().capability
    core=SimpleNamespace(WIKI_REVISION='fixed',capability=SimpleNamespace(
        load_wikitext=lambda revision:('text',{'revision':revision}),token_windows=original.token_windows))
    class Tok:
        bos_token_id=None
        def __call__(self,*args,**kw):return SimpleNamespace(input_ids=torch.tensor([[0,1,2,0,1,2,0,1,2]]))
    class Model:
        config=SimpleNamespace(max_position_embeddings=5)
        device='cpu'
        calls=0
        def eval(self):pass
        def __call__(self,input_ids,**kw):
            self.calls+=1
            return SimpleNamespace(logits=torch.zeros(1,input_ids.shape[1],3))
    model=Model();e.ppl_stage(core,model,Tok(),tmp_path,10)
    result=e.read(tmp_path/'metrics.json');calls=model.calls
    assert result['complete'] and result['scored_tokens']==8
    assert result['token_ppl']==pytest.approx(3.,abs=1e-6)
    e.ppl_stage(core,model,Tok(),tmp_path,10)
    assert model.calls==calls


def test_mmlu_completed_subjects_are_reused(tmp_path,monkeypatch):
    import sys
    names=[f'mmlu_subject{i:02d}' for i in range(57)];calls=[]
    class Manager:
        task_index={n:{'type':'task','yaml_path':str(tmp_path/'mmlu/default'/(n+'.yaml'))} for n in names}
    class Task:
        dataset={'test':SimpleNamespace(_fingerprint='fixed')}
        def test_docs(self):return [dict(answer=1)]
    def simple(**kw):
        name=kw['tasks'][0];calls.append(name)
        return dict(samples={name:[{'acc':1}]},results={name:{'acc,none':1.}})
    monkeypatch.setitem(sys.modules,'lm_eval',SimpleNamespace(simple_evaluate=simple))
    monkeypatch.setitem(sys.modules,'lm_eval.models.huggingface',SimpleNamespace(HFLM=lambda *a,**kw:None))
    monkeypatch.setitem(sys.modules,'lm_eval.tasks',SimpleNamespace(TaskManager=Manager,
        get_task_dict=lambda names,**kw:{names[0]:Task()}))
    e.write(tmp_path/(names[0]+'.json'),dict(n=1,accuracy=1.))
    e.mmlu_stage(None,None,tmp_path,20)
    assert len(calls)==56 and names[0] not in calls
    assert e.read(tmp_path/'metrics.json')['complete']
    e.mmlu_stage(None,None,tmp_path,20)
    assert len(calls)==56


def test_atomic_status_retries_transient_windows_lock(tmp_path,monkeypatch):
    from pathlib import Path
    original=Path.replace;attempts=[]
    def locked_once(self,path):
        attempts.append(str(path))
        if len(attempts)==1:raise PermissionError('transient reader')
        return original(self,path)
    monkeypatch.setattr(Path,'replace',locked_once)
    e.write(tmp_path/'status.json',dict(status='completed'))
    assert len(attempts)==2 and e.read(tmp_path/'status.json')['status']=='completed'
