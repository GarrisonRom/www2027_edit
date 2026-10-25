import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_cloud_gsu_c2 as c2


def test_basis_span_projection_and_zero_fallback():
    basis=c2.orthonormal_basis(torch.tensor([[1.,2.],[0.,0.]]))
    assert basis.shape==(2,1)
    assert torch.allclose(c2.project(torch.tensor([2.,3.]),basis),torch.tensor([2.,0.]),atol=1e-5)
    assert c2.orthonormal_basis(torch.zeros(2,2)) is None
    assert torch.equal(c2.project(torch.tensor([2.,3.]),None),torch.tensor([2.,3.]))


def test_route_contracts_and_shares_trust_radius():
    parameter=torch.nn.Parameter(torch.zeros(2))
    raw={parameter:torch.tensor([2.,3.])}
    basis=torch.tensor([[1.],[0.]])
    result,stats=c2.route_step([parameter],raw,{parameter:'x'},{'x':.5},{parameter:basis})
    assert stats['route_correction_ratio']<=.3+1e-6
    assert result[parameter].norm()<=raw[parameter].norm()


def test_f0_guard_falls_back_without_optimizer_advance():
    parameter=torch.nn.Parameter(torch.zeros(1))
    alpha,lp,trials,descent=c2.choose_guard([parameter],[torch.zeros(1)],
        {parameter:torch.tensor([-1.])},{parameter:torch.tensor([-.7])},
        lambda:parameter.item(),0.,-1.)
    assert alpha==.25 and descent
    assert lp<=-.9+1e-6 and len(trials)==3


def test_nondescent_raw_has_no_false_guarantee():
    parameter=torch.nn.Parameter(torch.zeros(1))
    alpha,lp,trials,descent=c2.choose_guard([parameter],[torch.zeros(1)],
        {parameter:torch.tensor([1.])},{parameter:torch.tensor([0.])},
        lambda:parameter.item(),0.,1.)
    assert alpha==0. and not descent and parameter.item()==1.


def test_groups_support_hybrid_modules_and_reject_unexpected_trainable():
    model=torch.nn.Module()
    module=torch.nn.Module()
    module.lora_A=torch.nn.ModuleDict({'default':torch.nn.Linear(2,1,bias=False)})
    module.lora_B=torch.nn.ModuleDict({'default':torch.nn.Linear(1,2,bias=False)})
    model.linear_attn=module
    params,groups,mapping,names=c2.components(model)
    assert len(groups)==1 and len(params)==2
    model.other=torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(RuntimeError,match='Unexpected'):
        c2.components(model)
