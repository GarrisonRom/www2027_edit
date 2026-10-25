import copy
import json
from types import SimpleNamespace
import pytest
import torch
from baselines.recipes import recipe,validate,search_plan,FAITHUN
from baselines.protocol import prompt,option_order,classify
from baselines.objectives import dpo,faithun_dpo_loss,DiskReference,activation_mse
from baselines.train import micro_weights
from baselines.train import learning_rate
from baselines.evaluate import aggregate,text_metrics,paper_metrics


def item():return dict(id='a',question='Where?',answer='Paris',false_options=['Rome','Lima'])


def test_exact_figure8_instruction_and_stable_options():
    p=prompt(item())
    assert p.startswith('Answer the following question by simply selecting a proper answer among the given options.')
    assert 'You must generate only the exact word without an explanation.' in p
    assert p.endswith('Your Answer:') and option_order(item())==option_order(item())
    assert option_order(item())==option_order(dict(item(),id='same-QA-different-record-ID'))


def test_every_candidate_uses_the_same_prompt(monkeypatch):
    import baselines.protocol as protocol
    contexts=[]
    def fake_encode(tok,row,device,prompted):
        contexts.append(row['question'])
        return dict(input_ids=torch.tensor([[0,1,2]]),labels=torch.tensor([[-100,-100,1]]),
                    attention_mask=torch.ones(1,3,dtype=torch.long))
    class M:
        device='cpu'
        def __call__(self,**kw):return SimpleNamespace(logits=torch.zeros(1,3,4))
    monkeypatch.setattr(protocol,'encode',fake_encode)
    protocol.scores(M(),None,item(),True)
    assert contexts==[prompt(item())]*3


def test_source_parameters_and_native_gate():
    assert recipe('RMU')['steering_coeff']==20
    assert recipe('RMU')['alpha_grid']==[20,50,100,150,200,300]
    assert recipe('DPOrej')['retain_loss_type']=='reverse_DPO'
    assert recipe('DPOrej')['rejection']=='I can’t answer the question.'
    assert recipe('SatImp')['gamma']==1 and recipe('SatImp')['alpha']==.1
    assert recipe('UNDIAL')['track']=='lora' and recipe('UNDIAL')['batch_size']==32
    assert recipe('KLUE')['stop_ua']==.33 and 11/33>.33
    for m in FAITHUN:
        c=recipe(m);c['track']='lora'
        with pytest.raises(ValueError):validate(c)


def test_pdu_cannot_run_without_calibration_or_dual_budget():
    c=recipe('PDU')
    with pytest.raises(ValueError):validate(c)
    c['retain_loss_eps']=6.;validate(c)
    c['epochs']=5
    with pytest.raises(ValueError):validate(c)


def test_search_is_source_bounded_and_not_test_ranked():
    plan=search_plan('RMU')
    assert len(plan)==60 and min(r['learning_rate'] for r in plan)==1e-7
    assert max(r['learning_rate'] for r in plan)==.003


def test_simnpo_author_config_not_open_unlearning_default():
    c=recipe('SimNPO')
    assert c['beta']==2.5 and c['gamma']==.1375 and c['seed']==1001
    assert c['batch_size']==4 and c['weight_decay']==.01
    assert learning_rate(c,0,9)==0
    assert learning_rate(c,9,9)==c['learning_rate']
    assert learning_rate(c,90,9)==0


def batch(n):
    return dict(labels=torch.tensor([[-100]+list(range(1,n+1))]))


@pytest.mark.parametrize('method',['GA','GradDiff','WGA','SatImp','UNDIAL','PDU','CEU'])
def test_token_weighted_microbatch_matches_pooled_mean(method):
    weights=micro_weights(method,[batch(2),batch(4)],[batch(1),batch(3)])
    if method=='CEU':assert weights[0][0]==.25
    else:assert weights[0][0]==pytest.approx(1/3)
    assert weights[0][1]==.25
    assert sum(w[0] for w in weights)==1 and sum(w[1] for w in weights)==1


@pytest.mark.parametrize('method',['NPO','SimNPO','DPOmis','DPOrej'])
def test_sequence_weighted_methods_do_not_become_token_weighted(method):
    weights=micro_weights(method,[batch(2),batch(4)],[batch(1),batch(3)])
    assert [w[0] for w in weights]==[.5,.5]


def test_rmu_includes_question_tokens_and_weights_all_valid_positions():
    weights=micro_weights('RMU',[batch(2),batch(4)],[batch(1),batch(3)])
    assert weights[0][0]==3/8 and weights[0][1]==pytest.approx(1/3)
    h=torch.tensor([[[3.],[2.],[100.]]],requires_grad=True)
    v=activation_mse(h,torch.zeros_like(h),torch.tensor([[1,1,0]]))
    v.backward()
    assert v.item()==6.5 and h.grad[0,0].item()!=0 and h.grad[0,2].item()==0


class Tiny(torch.nn.Module):
    def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.randn(3,6))
    def forward(self,input_ids,labels=None,**kw):
        logits=self.w[input_ids]
        loss=None if labels is None else torch.nn.functional.cross_entropy(logits[:,:-1].transpose(1,2),labels[:,1:],ignore_index=-100)
        return SimpleNamespace(logits=logits,loss=loss)


def encoded(answer):
    return dict(input_ids=torch.tensor([[0,1,2]]),labels=torch.tensor([[-100,1,answer]]))


def test_dpo_retain_is_reverse_preference_not_nll():
    torch.manual_seed(3);model=Tiny();ref=copy.deepcopy(model)
    f=encoded(1);a=encoded(2);cfg=recipe('DPOmis')
    actual=faithun_dpo_loss(model,ref,f,a,f,a,cfg)
    expected=.7*dpo(model,ref,a,f,.1)+dpo(model,ref,f,a,.1)
    assert torch.allclose(actual,expected)
    g=torch.autograd.grad(actual,model.w)[0]
    assert g.abs().sum()>0
    model2=copy.deepcopy(model)
    rev=faithun_dpo_loss(model2,ref,f,a,f,a,dict(cfg,gamma=0.))
    ga=torch.autograd.grad(rev,model2.w)[0]
    lose=dpo(model2,ref,a,f,.1)
    gb=torch.autograd.grad(lose,model2.w)[0]
    assert torch.allclose(ga,-gb,atol=1e-6)


def test_disk_reference_exact_and_immutable(tmp_path):
    m=Tiny();b=encoded(1);ref=DiskReference(tmp_path/'reference')
    ref.capture(m,[b]);before=m(**b).logits.detach().clone()
    with torch.no_grad():m.w.add_(3.)
    assert torch.equal(ref(**b).logits,before)
    assert not torch.equal(m(**b).logits,before)


def test_metrics_keep_path_macro_and_generation_separate():
    rows=[]
    for i,(path,correct) in enumerate([('p',1),('p',1),('q',0)]):
        rows.append(dict(group='forget/hop2_test',covered_F1=True,path=path,
            scoring={'paper_prompt/answer_sum':dict(correct=correct,margin=1,tie=False)}))
    a=aggregate(rows)['forget/hop2_test']['paper_prompt/answer_sum']
    assert a['accuracy']==pytest.approx(2/3) and a['path_macro_accuracy']==.5
    assert text_metrics('Paris.','Paris')['exact_match']==1
    assert classify([0,0,-1])['correct']==0


def test_paper_metric_formula_uses_base_TA_and_balanced_multihop():
    values={'forget/hop1_unlearn':.2,'forget/hop1_test':.3,'test/hop1_unlearn':.8,
            'forget/hop2_test':.4,'test/hop2_test':.9,'forget/same_object':.6,'test/same_object':.7}
    data={g:{'paper_prompt/answer_sum':dict(n=10,accuracy=v)} for g,v in values.items()}
    m=paper_metrics(data)
    assert m['TA']==.8 and m['UA_extended']==.3
    assert m['MA']==pytest.approx(.75)
    assert m['SA_pooled']==pytest.approx(.65)
    assert m['Score_pooledSA']==pytest.approx((.7+.8+.65+.75)/4)


def test_missing_capabilities_cannot_be_reported_complete(tmp_path,monkeypatch):
    import baselines.evaluate as ev
    import baselines.runtime as rt
    groups=['forget/hop1_unlearn','forget/hop1_test','test/hop1_unlearn',
            'forget/hop2_test','test/hop2_test','forget/same_object','test/same_object']
    rows=[dict(id=str(i),group=g,covered_F1=False,path=None) for i,g in enumerate(groups)]
    monkeypatch.setattr(ev,'records',lambda inputs:rows)
    monkeypatch.setattr(ev,'scores',lambda *args:{'answer_sum':[1,0,0],'answer_mean':[1,0,0],'full_sequence_mean':[1,0,0]})
    def fail(*args,**kw):raise ConnectionError('dataset unavailable')
    monkeypatch.setattr(rt,'load',lambda:SimpleNamespace(WIKI_REVISION='pinned',
        capability=SimpleNamespace(load_wikitext=fail,mmlu_eval=fail)))
    result=ev.evaluate(SimpleNamespace(eval=lambda:None),None,None,tmp_path/'eval')
    assert not result['evaluation_complete'] and result['capability_status']=='incomplete'
    assert result['mmlu']['status']=='failed' and result['ppl']['status']=='failed'


def test_selection_does_not_read_final_test_scores(tmp_path):
    from baselines.manage import select
    for name,lr in [('a',1e-5),('b',1e-4)]:
        run=tmp_path/name;run.mkdir()
        (run/'status.json').write_text(json.dumps(dict(status='training_completed')))
        (run/'training_summary.json').write_text(json.dumps(dict(qualified=True,final_training_ua=.2)))
        (run/'config.json').write_text(json.dumps(dict(method='GA',model='same',learning_rate=lr,
             beta=.1,alpha=1,seed=42,smoke=False)))
        (run/'metrics.json').write_text('INVALID TEST RESULTS MUST NOT BE READ')
    result=select(tmp_path)
    assert not result['uses_final_test_metrics']
    assert result['selected'][0]['learning_rate']==1e-5


def test_full_mmlu_does_not_silently_sample(tmp_path,monkeypatch):
    import sys
    from baselines.capabilities import full_mmlu
    names=[f'mmlu_subject{i}' for i in range(57)]
    class Task:
        dataset={'test':SimpleNamespace(_fingerprint='frozen')}
        def __init__(self,name):self.name=name
        def dump_config(self):return dict(task=self.name)
    calls=[]
    def simple(**kwargs):
        calls.append(kwargs)
        return dict(results={n:{'acc,none':.5} for n in names},
                    samples={n:[{'acc':1},{'acc':0}] for n in names})
    monkeypatch.setitem(sys.modules,'lm_eval',SimpleNamespace(simple_evaluate=simple))
    monkeypatch.setitem(sys.modules,'lm_eval.models.huggingface',SimpleNamespace(HFLM=lambda *a,**kw:object()))
    monkeypatch.setitem(sys.modules,'lm_eval.tasks',SimpleNamespace(TaskManager=lambda:object(),
        get_task_dict=lambda *a,**kw:{'mmlu':{n:Task(n) for n in names}}))
    result=full_mmlu(None,None,tmp_path)
    assert result['full'] and result['evaluated_samples']==114
    assert result['macro_subject_accuracy']==.5
    assert 'samples' not in calls[0] and 'limit' not in calls[0]
    assert calls[0]['num_fewshot']==5


def test_source_lock_paths_transfer_between_windows_and_linux(tmp_path):
    from baselines.bootstrap import relative_path
    assert relative_path(tmp_path,'vendor/runtime/train.py')==relative_path(tmp_path,'vendor'+chr(92)+'runtime'+chr(92)+'train.py')
