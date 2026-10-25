"""Paper-specific preference directions; pinned upstream losses for other methods."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F


def sequence_nll(model,b):
    out=model(**b)
    nll=F.cross_entropy(out.logits[:,:-1].float().transpose(1,2),b['labels'][:,1:],
                        ignore_index=-100,reduction='none').sum(-1)
    return nll,out


def dpo(model,reference,win,lose,beta):
    wn,_=sequence_nll(model,win);ln,_=sequence_nll(model,lose)
    with torch.no_grad():wr,_=sequence_nll(reference,win);lr,_=sequence_nll(reference,lose)
    return -2/beta*F.logsigmoid(beta*((wr-wn)-(lr-ln))).mean()


def faithun_dpo_loss(model,reference,forget,forget_alt,retain,retain_alt,cfg):
    # Forget: alternate wins. Retain: genuine wins. Do not replace retain DPO by NLL.
    return cfg['gamma']*dpo(model,reference,forget_alt,forget,cfg['beta']) + cfg['alpha']*dpo(
        model,reference,retain,retain_alt,cfg['beta'])


def activation_mse(current,target,attention_mask):
    per_token=(current.float()-target.float()).square().mean(-1)
    return (per_token*attention_mask).sum()/attention_mask.sum()


def rmu_loss(trainer,model,reference,forget,retain,gamma,alpha):
    f,_=trainer.forward_with_cache(model,forget,trainer.model_module,no_grad=False)
    control=trainer.get_control_vector(f.shape[-1]).to(f.device,f.dtype)
    r,_=trainer.forward_with_cache(model,retain,trainer.model_module,no_grad=False)
    target,_=trainer.forward_with_cache(reference,retain,trainer.ref_module,no_grad=True)
    return gamma*activation_mse(f,control,forget['attention_mask'])+alpha*activation_mse(r,target,retain['attention_mask'])


class DiskReference(nn.Module):
    """Exact fixed-base outputs on disk; no moving reference, no full CPU logits bank."""
    def __init__(self,directory):
        super().__init__();self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.layer=nn.Identity()
    def key(self,ids):
        return hashlib.sha256(ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    @torch.no_grad()
    def capture(self,model,batches,layer=None):
        previous=model.training;model.eval()
        for b in batches:
            path=self.directory/(self.key(b['input_ids'])+'.pt')
            if path.exists():continue
            hidden=[];handles=[]
            if layer is not None:handles.append(layer.register_forward_hook(
                lambda m,a,o:hidden.append((o[0] if isinstance(o,tuple) else o).detach().cpu())))
            try:
                out=model(**b)
                # Preserve exactly the output values returned by the fixed base model.
                torch.save(dict(logits=out.logits.detach().cpu(),hidden=hidden[0] if hidden else None),path)
            finally:
                for h in handles:h.remove()
        model.train(previous)
    def forward(self,input_ids,labels=None,**kw):
        if input_ids.shape[0]!=1:raise ValueError('Disk reference expects single-example microbatches')
        row=torch.load(self.directory/(self.key(input_ids)+'.pt'),map_location='cpu',weights_only=True)
        logits=row['logits'].to(input_ids.device).float()
        if row['hidden'] is not None:self.layer(row['hidden'].to(input_ids.device))
        loss=None if labels is None else F.cross_entropy(logits[:,:-1].transpose(1,2),labels[:,1:],ignore_index=-100)
        return SimpleNamespace(logits=logits,loss=loss)
