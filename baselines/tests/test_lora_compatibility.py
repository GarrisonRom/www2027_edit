import copy
import pytest
import torch
from peft import LoraConfig,get_peft_model
from transformers import LlamaConfig,LlamaForCausalLM
from baselines.recipes import recipe,lora_adapt,lora_targets,validate,LORA_METHODS
from baselines.objectives import DiskReference,faithun_dpo_loss,rmu_loss


def test_explicit_adaptation_preserves_source_objective_and_excludes_klue():
    for method in LORA_METHODS:
        original=recipe(method);c=lora_adapt(original)
        if method=='PDU':c['retain_loss_eps']=10.
        validate(c)
        for key in ('gamma','alpha','beta','epochs','batch_size','stop_ua','learning_rate'):
            assert c[key]==original[key]
        assert original==recipe(method)
    with pytest.raises(ValueError,match='no verified equivalent'):lora_adapt(recipe('KLUE'))


def test_rmu_lora_keeps_source_component_restriction():
    names=[f'model.layers.{i}.mlp.{field}_proj' for i in range(8) for field in ('up','down','gate')]
    targets=lora_targets(names,lora_adapt(recipe('RMU')))
    assert targets==[f'model.layers.{i}.mlp.down_proj' for i in (5,6,7)]


def test_native_and_lora_trials_are_never_selected_as_one_group(tmp_path):
    import json
    from baselines.manage import select
    for mode in ('native','lora'):
        run=tmp_path/mode;run.mkdir()
        c=recipe('GA') if mode=='native' else lora_adapt(recipe('GA'))
        c.update(model='same',smoke=False)
        (run/'config.json').write_text(json.dumps(c),encoding='utf-8')
        (run/'status.json').write_text(json.dumps(dict(status='training_completed')))
        (run/'training_summary.json').write_text(json.dumps(dict(qualified=True,final_training_ua=.2)))
    result=select(tmp_path)
    assert len(result['selected'])==2
    assert {r['track'] for r in result['selected']}=={'native','lora'}


@pytest.fixture
def cpu_threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize('method',[m for m in LORA_METHODS if m!='Original'])
def test_objective_updates_lora_but_never_backbone(method,tmp_path,cpu_threads):
    from baselines.runtime import load
    core=load();torch.manual_seed(42)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=8,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64))
    c=lora_adapt(recipe(method))
    if method=='PDU':c['retain_loss_eps']=10.
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,task_type='CAUSAL_LM',
        target_modules=lora_targets(core.base.target_modules(model),c),bias='none'))
    model.enable_input_require_grads()
    def b(ids):
        ids=torch.tensor([ids]);labels=ids.clone();labels[:,:2]=-100
        return dict(input_ids=ids,labels=labels,attention_mask=torch.ones_like(ids))
    f=b([1,2,3,4,5]);r=b([6,7,8,9,10]);fa=b([1,2,11,12,13]);ra=b([6,7,14,15,16])
    bank=None
    if method in ('NPO','RMU','DPOmis','DPOrej','UNDIAL'):
        bank=DiskReference(tmp_path/'reference')
        bank.capture(model,[f,r,fa,ra],core.rl.decoder_layer(model,7) if method=='RMU' else None)
    cfg=core.rl.profile(method,'matched_lora');cfg.update(c)
    trainer=core.rl.bind_trainer(cfg,model,bank)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    adapters={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    if method.startswith('DPO'):loss=faithun_dpo_loss(model,bank,f,fa,r,ra,c)
    elif method=='RMU':loss=rmu_loss(trainer,model,bank,f,r,c['gamma'],c['alpha'])
    else:loss=trainer.compute_loss(model,dict(forget=f,retain=r))
    assert torch.isfinite(loss)
    loss.backward()
    params=[p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in params)
    opt=torch.optim.SGD(params,lr=.001);opt.step()
    current=dict(model.named_parameters())
    assert all(torch.equal(value,current[n]) for n,value in frozen.items())
    assert any(not torch.equal(value,current[n]) for n,value in adapters.items())
