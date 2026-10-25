import copy
from pathlib import Path
import sys
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from native_cpu_streaming import NativeCPUStreaming,StreamingAdamW
from test_complete_reproduction import Toy,batch
import diagnose_gsu_memory_guided as protocol


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_streaming_forward_and_gradients_equal_reference(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(7)
    model=Toy();reference=copy.deepcopy(model).to(device)
    model.get_input_embeddings=lambda:model.embed
    stream=NativeCPUStreaming(model,device)
    try:
        b={key:value.to(device) for key,value in batch().items()}
        actual=model(**b);expected=reference(**b)
        assert torch.allclose(actual.logits,expected.logits,atol=1e-5)
        actual.loss.backward();expected.loss.backward()
        for (n,p),(rn,rp) in zip(model.named_parameters(),reference.named_parameters()):
            assert n==rn
            assert torch.allclose(p.grad.cpu(),rp.grad.cpu(),atol=1e-5),n
        assert model.model.layers[0].mlp.up_proj.weight.device.type=='cpu'
    finally:stream.context.__exit__(None,None,None)


def test_streaming_adam_fp32_matches_adamw():
    p=torch.nn.Parameter(torch.tensor([1.,2.]))
    q=torch.nn.Parameter(p.detach().clone())
    stream=StreamingAdamW([p],.01,.01)
    other=torch.optim.AdamW([q],lr=.01,weight_decay=.01)
    for _ in range(4):
        p.grad=torch.tensor([.2,-.3]);q.grad=p.grad.clone()
        stream.step();other.step()
        assert torch.allclose(p,q,atol=1e-7)


def test_seed_changes_schedule_without_changing_fixed_data():
    items=[{'id':str(i),'question':'q'+str(i),'answer':'a','source':'F0'} for i in range(33)]
    plans=[protocol.make_update_schedule(items,114,seed) for seed in (42,43,44)]
    assert len({tuple(x['item_id'] for x in p) for p in plans})==3
    assert all(len(p)==114 for p in plans)
    assert {x['item_id'] for x in plans[0]}=={x['item_id'] for x in plans[1]}


def test_8b_job_matrix_contains_every_seed_method_and_track():
    from run_8b_multiseed import jobs_for,SEEDS,MODELS
    from reproduction_losses import METHODS
    jobs=jobs_for(Path('/runs/frozen'))
    trained=[job for job in jobs if job['kind']=='training' and not job['smoke']]
    assert len(trained)==162
    for key,_ in MODELS:
        for seed in SEEDS:
            subset=[j for j in trained if j['model_key']==key and j['seed']==seed]
            assert {j['method'] for j in subset}==set(METHODS)|{'C1','C2'}
            assert len(subset)==27
            assert all('--seed' in j['command'] and str(seed) in j['command'] for j in subset)
    originals=[j for j in jobs if j['method']=='Original']
    assert len(originals)==2 and all(j['seed'] is None for j in originals)


@pytest.mark.parametrize('family',['llama','qwen3'])
def test_native_streaming_actual_decoder_architecture(family):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from transformers import LlamaConfig,LlamaForCausalLM,Qwen3Config,Qwen3ForCausalLM
    config_cls,model_cls=(LlamaConfig,LlamaForCausalLM) if family=='llama' else (Qwen3Config,Qwen3ForCausalLM)
    config=config_cls(vocab_size=127,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=4,max_position_embeddings=128,
                      attention_dropout=0.,use_cache=False)
    torch.manual_seed(13)
    model=model_cls(config).float();reference=copy.deepcopy(model).cuda()
    stream=NativeCPUStreaming(model)
    try:
        b={key:value.cuda() for key,value in batch().items()}
        actual=model(**b);expected=reference(**b)
        assert torch.allclose(actual.logits,expected.logits,atol=1e-5)
        actual.loss.backward();expected.loss.backward()
        for (n,p),(rn,rp) in zip(model.named_parameters(),reference.named_parameters()):
            assert n==rn
            assert torch.allclose(p.grad.cpu(),rp.grad.cpu(),atol=1e-5),n
    finally:stream.context.__exit__(None,None,None)

