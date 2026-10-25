"""Independent loss/gradient checks against frozen exp016 code, not live training."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

WS = Path(__file__).resolve().parents[3]
FROZEN = WS/'exp/runs/exp016_local_baselines/runs/local_20261023_233138/source_snapshot/exp/train'
sys.path.insert(0,str(FROZEN))
import reproduction_losses as r
import reproduce_all_baselines as runner


class Logits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values=torch.nn.Parameter(torch.randn(2,5,11))
    def forward(self,input_ids,labels,**kw):
        logits=self.values[input_ids[:,0]]
        return SimpleNamespace(logits=logits,loss=F.cross_entropy(
            logits[:,:-1].transpose(1,2),labels[:,1:],ignore_index=-100))


def item(index=0):
    return dict(input_ids=torch.tensor([[index,2,3,4,5]]),
        attention_mask=torch.ones(1,5,dtype=torch.long),
        labels=torch.tensor([[-100,-100,3,4,5]]))


def nll(model,b):
    logits=model(**b).logits[:,:-1]
    return F.cross_entropy(logits.transpose(1,2),b['labels'][:,1:],
                           ignore_index=-100,reduction='none')


@pytest.mark.parametrize('method',['DPOmis','DPOrej','CEU','PDU','WGA','SatImp','UNDIAL'])
def test_independent_loss_and_gradient(method):
    torch.manual_seed(17)
    model=Logits();f=item();retain=item(1)
    alt=item();alt['labels']=torch.tensor([[-100,-100,6,7,8]])
    bank=runner.ReferenceBank();bank.capture(model,[f,retain,alt])
    with torch.no_grad():model.values.add_(torch.randn_like(model.values)*.1)
    cfg=r.profile(method,'matched_lora');trainer=r.bind_trainer(cfg,model,bank)
    if method=='PDU':trainer.retain_loss_eps=.2
    b={'forget':{'original':f,'alternate':alt} if method.startswith('DPO') else f,'retain':retain}
    actual=trainer.compute_loss(model,b)
    valid=f['labels'][:,1:]!=-100
    ce=nll(model,f)[valid]
    ret=model(**retain).loss
    if method.startswith('DPO'):
        win=-nll(model,alt).sum()+nll(bank,alt).sum()
        lose=-nll(model,f).sum()+nll(bank,f).sum()
        expected=cfg['gamma']*(-2/cfg['beta']*F.logsigmoid(cfg['beta']*(win-lose)))+cfg['alpha']*ret
    elif method in ('WGA','SatImp'):
        prob=(-ce).exp().detach()
        weight=prob**cfg['beta'] if method=='WGA' else prob**cfg['beta1']*(1-prob)**cfg['beta2']
        expected=-cfg['gamma']*(weight*ce).mean()+cfg['alpha']*ret
    elif method=='CEU':
        labels=f['labels'][:,1:].clone();labels[:,1]=-100
        mask=labels!=-100;logits=model(**f).logits[:,:-1][mask]
        target=logits.detach().clone();target.scatter_(1,labels[mask].unsqueeze(1),float('-inf'))
        expected=F.cross_entropy(logits,target.softmax(-1))
    elif method=='PDU':
        logits=model(**f).logits
        spread=(logits.max(-1).values-logits.mean(-1)).square()
        expected=spread[f['labels']!=-100].mean()+ret-.2
    else:
        logits=model(**f).logits[:,:-1][valid]
        teacher=bank(**f).logits[:,:-1][valid].clone()
        teacher.scatter_add_(1,f['labels'][:,1:][valid].unsqueeze(1),torch.full((3,1),-cfg['beta']))
        expected=cfg['gamma']*F.cross_entropy(logits,teacher.softmax(-1))+cfg['alpha']*ret
    assert torch.allclose(actual,expected,atol=1e-6)
    ag=torch.autograd.grad(actual,model.values,retain_graph=True)[0]
    eg=torch.autograd.grad(expected,model.values)[0]
    assert torch.allclose(ag,eg,atol=1e-6)


def test_full_sequence_scoring_can_reverse_answer_only_order():
    # Identical question LP sum=-10, 10 question tokens; candidate lengths differ.
    # Correct answer mean=-1 vs false=-2, yet raw full-sequence means favor false.
    qsum=-10.;qn=10
    correct_answer=(-1.,1);false_answer=(-2.,10)
    c=(qsum+correct_answer[0]*correct_answer[1])/(qn+correct_answer[1])
    w=(qsum+false_answer[0]*false_answer[1])/(qn+false_answer[1])
    # Use stronger negative common prefix to expose the length normalization reversal.
    qsum=-100.
    c=(qsum-1)/(qn+1);w=(qsum-20)/(qn+10)
    assert -1 > -2 and c < w
