import copy
import importlib.util
from pathlib import Path
import sys
import random
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import decisive_controls as c
import reproduction_losses as loss


def test_random_component_histogram_preserved_within_module_type():
    weights={f'L{i:02d}.mlp.{module}':float(i)/10+.5 for i in range(8) for module in ('up_proj','down_proj')}
    randomized=c.permute_within_type(weights,42)
    assert randomized==c.permute_within_type(weights,42)
    assert randomized!=weights
    for module in ('up_proj','down_proj'):
        assert sorted(v for k,v in weights.items() if k.endswith(module))==sorted(v for k,v in randomized.items() if k.endswith(module))


def test_direction_null_preserves_rank_none_and_projector_identities():
    p=torch.nn.Parameter(torch.zeros(8));q=torch.nn.Parameter(torch.zeros(5))
    basis=torch.linalg.qr(torch.randn(8,2)).Q
    result,meta=c.random_subspaces({p:basis,q:None},{p:'a',q:'b'},44)
    repeated,_=c.random_subspaces({p:basis,q:None},{p:'a',q:'b'},44)
    assert result[q] is None
    assert result[p].shape==basis.shape
    assert torch.equal(result[p],repeated[p])
    assert torch.allclose(result[p].T@result[p],torch.eye(2),atol=1e-6)
    assert not torch.allclose(result[p]@result[p].T,basis@basis.T)


def test_no_fallback_winner_when_development_fails_gate():
    r={'status':'completed','direct_accuracy':.4,'retain_accuracy':.8,'original_retain_accuracy':.9,
       'multihop_macro':0.,'learning_rate':1e-4}
    assert c.select_trial([r])['status']=='no_feasible_configuration'
    r['direct_accuracy']=.3;r['retain_accuracy']=.81
    winner=c.select_trial([r])
    assert winner['selected']==r
    assert c.select_trial([])['selected'] is None


def test_development_disjointness_actual_frozen_splits():
    import run_remote_baselines as b
    inputs=b.protocol.load_inputs()
    plan=c.unused_development_split(inputs['clusters'],inputs['indices'])
    assert len(plan['forget'])==len(plan['retain'])==24
    assert not set(plan['forget'])&set(plan['retain'])
    excluded=set(plan['official_excluded'])
    assert not (set(plan['forget'])|set(plan['retain']))&excluded
    assert plan==c.unused_development_split(inputs['clusters'],inputs['indices'])


def test_fixed_job_matrix_and_development_commands(tmp_path):
    from run_decisive_validation import initial_jobs
    jobs=initial_jobs(tmp_path,tmp_path/'development_plan.json')
    dev=[j for j in jobs if j['kind']=='development']
    formal=[j for j in jobs if j['kind']=='formal' and not j['smoke']]
    assert len(dev)==30 and len(formal)==30
    assert {j['method'] for j in formal}=={'KLUE','GA-F0F1','Cg','C1R','C2D'}
    assert all('--development_plan' in j['command'] for j in dev)
    for method in ('KLUE','GA-F0F1','Cg','C1R','C2D'):
        assert sorted(j['seed'] for j in formal if j['method']==method)==[42,42,43,43,44,44]
    assert all('--sdpa_math' in j['command'] for j in jobs if j['method']=='KLUE')
    assert all('F0F1' in j['command'] for j in jobs if j['method']=='GA-F0F1')


def test_development_inputs_have_no_official_training_or_F1_targets():
    import run_remote_baselines as b
    import development_evaluation as d
    inputs=b.protocol.load_inputs()
    plan=c.unused_development_split(inputs['clusters'],inputs['indices'])
    task=d.task_inputs(inputs,plan)
    assert task['F1']==[] and len(task['F0'])==24 and len(task['retain_all'])==24
    assert not {x['id'] for x in task['F0']}&{x['id'] for x in inputs['F0']}
    assert {x['cluster_index'] for x in task['F0']}==set(plan['forget'])


@pytest.mark.parametrize('method',[m for m in loss.METHODS if m!='KLUE'])
@pytest.mark.parametrize('track',['native','matched_lora'])
def test_all_learning_rates_numeric(method,track):
    cfg=loss.profile(method,track)
    assert isinstance(cfg['learning_rate'],float)
    assert cfg['learning_rate']>0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('family',['llama','qwen3'])
def test_math_sdpa_streaming_two_forward_padded_backward(family):
    from native_cpu_streaming import NativeCPUStreaming
    from transformers import LlamaConfig,LlamaForCausalLM,Qwen3Config,Qwen3ForCausalLM
    from torch.nn.attention import sdpa_kernel,SDPBackend
    C,M=(LlamaConfig,LlamaForCausalLM) if family=='llama' else (Qwen3Config,Qwen3ForCausalLM)
    config=C(vocab_size=127,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
             num_attention_heads=4,num_key_value_heads=2,head_dim=4,attention_dropout=0.,use_cache=False)
    torch.manual_seed(42);model=M(config).to(torch.bfloat16);ref=copy.deepcopy(model).cuda()
    stream=NativeCPUStreaming(model)
    ids=torch.randint(0,127,(4,18),device='cuda');mask=torch.ones_like(ids);mask[0,7:]=0;mask[1,13:]=0
    labels=ids.clone();labels[:,:2]=-100;labels[mask==0]=-100
    b=dict(input_ids=ids,attention_mask=mask,labels=labels)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            actual=model(**b).loss+model(**b).loss
            expected=ref(**b).loss+ref(**b).loss
            actual.backward();expected.backward()
        assert torch.allclose(actual,expected,atol=1e-3)
        for (name,p),(_,q) in zip(model.named_parameters(),ref.named_parameters()):
            assert torch.allclose(p.grad.cpu().float(),q.grad.cpu().float(),atol=.01,rtol=.02),name
    finally:stream.context.__exit__(None,None,None)

