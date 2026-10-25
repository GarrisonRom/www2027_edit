"""Native BF16 weights/gradients stay CPU; stream modules to GPU, offload saved tensors.
No quantization, LoRA conversion, or change in the native trainable parameter domain.
"""
import types
import weakref
import torch
from torch.nn import functional as F


class NativeCPUStreaming:
    def __init__(self,model,device='cuda'):
        self.model=model
        self.device=torch.device(device)
        self.original={}
        self.transfers={}
        model.to('cpu')
        model.get_input_embeddings().to(self.device)
        # Input placement is GPU. Parameters may reside CPU; explicit dispatch below.
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.Linear):
                self.original[module]=module.forward
                def linear(mod,x,owner=self):
                    weight=owner.move(mod.weight)
                    bias=owner.move(mod.bias) if mod.bias is not None else None
                    return F.linear(x.to(owner.device),weight,bias)
                module.forward=types.MethodType(linear,module)
            elif isinstance(module,torch.nn.Embedding):
                self.original[module]=module.forward
                def embedding(mod,x,owner=self):
                    return F.embedding(x.to(owner.device),owner.move(mod.weight),mod.padding_idx,
                        mod.max_norm,mod.norm_type,mod.scale_grad_by_freq,mod.sparse)
                module.forward=types.MethodType(embedding,module)
            elif ('RMSNorm' in type(module).__name__ or isinstance(module,torch.nn.LayerNorm)):
                # Norm weights are tiny; avoid specialized RMSNorm kernel device mismatches.
                module.to(self.device)
            elif 'RotaryEmbedding' in type(module).__name__:
                module.to(self.device)
        self.context=torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack)
        self.context.__enter__()

    def move(self,parameter):
        value=parameter.to(self.device)
        self.transfers[value.untyped_storage().data_ptr()]=(parameter,weakref.ref(value))
        return value

    def pack(self,tensor):
        if tensor.device.type!='cuda':return ('plain',tensor)
        match=self.transfers.get(tensor.untyped_storage().data_ptr())
        if match is not None and match[1]() is not None:
            parameter,_=match
            # A saved transposed linear weight shares the transferred tensor's storage.
            return ('weight',parameter,tensor.shape,tensor.stride(),tensor.storage_offset(),tensor.device)
        return ('tensor',tensor.detach().cpu(),tensor.device)

    @staticmethod
    def unpack(item):
        if item[0]=='plain':return item[1]
        if item[0]=='tensor':return item[1].to(item[2])
        _,parameter,shape,stride,offset,device=item
        return parameter.as_strided(shape,stride,offset).to(device)

    def clear_transfers(self):
        self.transfers.clear()


class StreamingAdamW:
    """Native BF16 parameter storage, FP32 CPU moments; no duplicate FP32 masters.
    Full gradient norm clipping is applied by the shared trainer before this step.
    """
    def __init__(self,params,lr,weight_decay):
        self.params=list(params);self.lr=lr;self.weight_decay=weight_decay
        self.states={};self.step_count=0
        self.param_groups=[{'lr':lr}]
    @torch.no_grad()
    def step(self):
        self.step_count+=1;t=self.step_count;b1,b2=.9,.999
        lr=self.param_groups[0]['lr']
        for p in self.params:
            if p.grad is None:continue
            grad=p.grad.detach().float().cpu()
            if p not in self.states:
                self.states[p]=(torch.zeros_like(grad),torch.zeros_like(grad))
            m,v=self.states[p]
            m.mul_(b1).add_(grad,alpha=1-b1)
            v.mul_(b2).addcmul_(grad,grad,value=1-b2)
            value=p.detach().float().cpu()
            value.mul_(1-lr*self.weight_decay)
            value.addcdiv_(m/(1-b1**t),(v/(1-b2**t)).sqrt().add_(1e-8),value=-lr)
            p.copy_(value.to(p.device,p.dtype));p.grad=None

