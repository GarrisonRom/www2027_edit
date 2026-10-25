"""Frozen upstream losses, explicit profiles, and KLUE native neuron primitives.
No historical trainer is patched. Upstream AST executes original function bodies.
"""
import ast
import contextlib
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
import yaml

WS = Path(__file__).resolve().parents[2]
UPSTREAM = WS / 'code/open-unlearning'
METHODS = ('GA', 'GradDiff', 'NPO', 'SimNPO', 'RMU', 'DPOmis', 'DPOrej',
           'CEU', 'PDU', 'WGA', 'SatImp', 'UNDIAL', 'KLUE')
HANDLERS = {'GA': 'GradAscent', 'DPOmis': 'DPO', 'DPOrej': 'DPO'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upstream_namespace(root=UPSTREAM):
    env = {'torch': torch, 'nn': nn, 'F': F, 'UnlearnTrainer': object,
           'TrainerCallback': object, 'deepspeed': SimpleNamespace(DeepSpeedEngine=type(None))}
    files = [('src/trainer/utils.py', None),
             ('src/trainer/unlearn/grad_diff.py', {'GradDiff'}),
             ('src/trainer/unlearn/grad_ascent.py', {'GradAscent'}),
             ('src/trainer/unlearn/npo.py', {'NPO'}),
             ('src/trainer/unlearn/simnpo.py', {'SimNPO'}),
             ('src/trainer/unlearn/rmu.py', {'RMU'}),
             ('src/trainer/unlearn/dpo.py', {'DPO'}),
             ('src/trainer/unlearn/ceu.py', None),
             ('src/trainer/unlearn/pdu.py', None),
             ('src/trainer/unlearn/wga.py', {'WGA'}),
             ('src/trainer/unlearn/satimp.py', {'SatImp'}),
             ('src/trainer/unlearn/undial.py', {'UNDIAL'})]
    manifest = {}
    for relative, names in files:
        path = root / relative
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and (names is None or n.name in names)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
        manifest[relative] = digest(path)
    return env, manifest


def profile(method, track):
    if method not in METHODS or track not in ('matched_lora', 'native'):
        raise ValueError((method, track))
    if method == 'KLUE':
        if track != 'native':
            raise ValueError('KLUE is native neuron masking, not a disguised LoRA mask')
        return dict(method=method, track=track, gamma=.7, alpha=1., batch_size=4,
                    epochs=150, learning_rate=1e-5, weight_decay=0., beta=1.,
                    attribution_alpha=10., mismatch_count=5, neuron_fraction=.05,
                    early_stop_ua=1/3, max_updates=1350,
                    max_grad_norm=1., lr_scheduler='constant',
                    convention='gated-MLP pre-down neurons; gate/up rows and down columns; global top5%',
                    fidelity='paper reimplementation; gated neuron mapping and optimizer explicit, no official code')
    handler = HANDLERS.get(method, method)
    path = UPSTREAM / 'configs/trainer' / (handler + '.yaml')
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    args = dict(raw.get('method_args') or {})
    base = yaml.safe_load((UPSTREAM/'configs/trainer/finetune.yaml').read_text())['args']
    config = dict(method=method, track=track, gamma=1., alpha=1., beta=1., delta=0.,
                  learning_rate=base['learning_rate'], weight_decay=base['weight_decay'],
                  epochs=base['num_train_epochs'], batch_size=4, max_updates=None,
                  max_grad_norm=1., lr_scheduler='linear',
                  retain_loss_type='NLL', config_sha256=digest(path), raw_upstream=raw,
                  fidelity='exact frozen upstream loss; architecture/data/optimizer profile disclosed')
    config.update(args)
    overrides = raw.get('args') or {}
    config.update({k: v for k, v in overrides.items() if k in ('learning_rate', 'num_train_epochs')})
    if 'num_train_epochs' in config:
        config['epochs'] = config.pop('num_train_epochs')
    if track == 'matched_lora':
        config.update(batch_size=1, max_updates=114, epochs=4,
                      learning_rate=5e-5, weight_decay=.01, max_grad_norm=None, lr_scheduler='constant')
    if method == 'PDU':
        config['retain_loss_eps'] = 'initial_retain_nll'
        # Keep the upstream five-epoch warmup. No-dual 114-step results are labeled, not hidden.
        config['dual_active_in_matched_budget'] = False
    if method == 'RMU':
        config['steering_coeff'] = args['steering_coeff']
        config['semantic_repairs']=['per-sequence masked MSE avoids upstream BxB broadcast at batch>1']
    if method == 'UNDIAL':
        config['semantic_repairs']=['answer-only teacher targets, exclude -100 labels from indexing and CE']
    if 'semantic_repairs' in config:
        config['fidelity']='frozen upstream algorithm with explicit tested semantic repairs; not bitwise source loss parity'
    return config


def bind_trainer(config, model, reference=None, env=None):
    if env is None:
        env, _ = upstream_namespace()
    if config['method']=='UNDIAL':
        env['compute_undial_loss']=undial_answer_loss
    handler = HANDLERS.get(config['method'], config['method'])
    trainer = object.__new__(env[handler])
    trainer.__dict__.update(config)
    trainer.model = model
    trainer.ref_model = reference
    trainer.log = lambda *a, **k: None
    trainer.preferences = [config['gamma'], config['alpha']]
    trainer.loss_names = ['forget_loss', 'retain_loss']
    trainer.can_update = config.get('dual_warmup_epochs', 0) == 0
    trainer.primal_dual = config.get('primal_dual', False)
    trainer.dual_step_size = config.get('dual_step_size', 1.)
    trainer.dual_update_upon = config.get('dual_update_upon', 'step')
    trainer.retain_loss_eps = config.get('retain_loss_eps', 0.)
    trainer.ignore_first_n_answer_tokens = config.get('ignore_first_n_answer_tokens', 1)
    if config['method'] == 'RMU':
        trainer.control_vec = None
        trainer.model_module = decoder_layer(model, 7)
        trainer.ref_module = reference.layer
        trainer.compute_activation_loss=masked_activation_loss
    return trainer


def masked_activation_loss(current,target,mask):
    counts=mask.sum(-1)
    if bool((counts==0).any()):raise ValueError('RMU empty answer sequence')
    per_token=(current.float()-target.float()).square().mean(-1)
    return ((per_token*mask).sum(-1)/counts).mean()


def undial_answer_loss(model,ref_model,inputs,beta):
    outputs=model(**inputs)
    labels=inputs['labels'][:,1:]
    valid=labels!=-100
    if not bool(valid.any()):raise ValueError('UNDIAL no answer targets')
    logits=outputs.logits[:,:-1].float()[valid]
    with torch.no_grad():
        teacher=ref_model(**inputs).logits[:,:-1].float()[valid].clone()
        teacher.scatter_add_(1,labels[valid].unsqueeze(1),torch.full_like(labels[valid].unsqueeze(1),-beta,dtype=teacher.dtype))
        targets=teacher.softmax(-1)
    return F.cross_entropy(logits,targets),outputs


def decoder_layer(model, index):
    found = [m for n, m in model.named_modules() if n.endswith('.layers.'+str(index))
             and not any(x in n for x in ('visual', 'vision', 'mtp'))]
    if len(found) != 1:
        raise RuntimeError(f'Expected unique decoder layer {index}: {len(found)}')
    return found[0]


def gated_mlps(model):
    groups = {}
    for name, module in model.named_modules():
        if name.endswith('.mlp') and not any(x in name for x in ('visual', 'vision', 'mtp')):
            if not all(isinstance(getattr(module, key, None), nn.Linear)
                       for key in ('gate_proj', 'up_proj', 'down_proj')):
                raise RuntimeError('KLUE gated MLP contract not satisfied: '+name)
            if module.up_proj.out_features != module.down_proj.in_features:
                raise RuntimeError('Inconsistent gated neuron axis')
            groups[name] = module
    if not groups:
        raise RuntimeError('No native FFN groups')
    return groups


def answer_nll(output, labels):
    return F.cross_entropy(output.logits[:, :-1].float().transpose(1, 2), labels[:, 1:],
                           ignore_index=-100, reduction='none').sum()


def probability_attribution(model, batch, mlps):
    saved, handles = {}, []
    for name, mlp in mlps.items():
        def hook(module, args, key=name):
            saved[key] = args[0]
        handles.append(mlp.down_proj.register_forward_pre_hook(hook))
    try:
        output = model(**batch)
        log_probability = -answer_nll(output, batch['labels'])
        probability = torch.exp(log_probability.double())
        values = list(saved.values())
        grads = torch.autograd.grad(probability, values, allow_unused=False)
        scores = {name: (activation.detach().double()*grad.detach().double()).amax(dim=1).squeeze(0).cpu()
                  for (name, activation), grad in zip(saved.items(), grads)}
        if not all(bool(torch.isfinite(v).all()) for v in scores.values()):
            raise RuntimeError('Nonfinite probability attribution')
        if not any(bool(v.abs().max() > 0) for v in scores.values()):
            raise RuntimeError('Zero/underflowed probability attribution; do not silently switch to logprob')
        return scores, float(log_probability.detach())
    finally:
        for handle in handles:
            handle.remove()


def regularized_importance(genuine, mismatches, alpha):
    if not mismatches:
        raise ValueError('KLUE requires mismatched-question regularization')
    return {key: genuine[key] - alpha*torch.stack([row[key].clamp_min(0) for row in mismatches]).mean(0)
            for key in genuine}


def global_neurons(scores, fraction=.05):
    if not 0 < fraction <= 1:
        raise ValueError('Invalid neuron fraction')
    keys = sorted(scores)
    flat = torch.cat([scores[key].double() for key in keys])
    if not bool(torch.isfinite(flat).all()):
        raise ValueError('Nonfinite neuron scores')
    # Stable ordering fixes ties by sorted module name then neuron index.
    count = max(1, math.ceil(flat.numel()*fraction))
    indices = torch.argsort(flat, descending=True, stable=True)[:count]
    selected, start = {}, 0
    for key in keys:
        length = scores[key].numel()
        selected[key] = indices[(indices >= start) & (indices < start+length)]-start
        start += length
    return selected


class NativeNeuronAdam:
    """CPU FP32 master/moments per visited neuron; only selected native entries move.

    Inactive moments decay lazily to match zero-gradient Adam moments, but inactive
    parameter displacements are suppressed. No LoRA is used. Weight decay is zero.
    """
    def __init__(self, mlps, lr, betas=(.9, .999), eps=1e-8):
        self.mlps, self.lr, self.betas, self.eps = mlps, lr, betas, eps
        self.states = {}
        self.step_count = 0

    @torch.no_grad()
    def step(self, selected):
        self.step_count += 1
        t, (b1, b2) = self.step_count, self.betas
        for key, mlp in self.mlps.items():
            indices = selected[key].long().cpu()
            if not len(indices):
                continue
            for field, axis in (('gate_proj', 0), ('up_proj', 0), ('down_proj', 1)):
                parameter = getattr(mlp, field).weight
                if parameter.grad is None:
                    raise RuntimeError('Missing KLUE native gradient')
                gpu_indices = indices.to(parameter.device)
                grad = parameter.grad.index_select(axis, gpu_indices)
                if axis == 1:
                    grad = grad.T
                grad = grad.detach().float().cpu()
                token = (key, field)
                if token not in self.states:
                    master = parameter.detach().float().cpu()
                    if axis == 1:
                        master = master.T.contiguous()
                    self.states[token] = dict(master=master.clone(), m=torch.zeros_like(master),
                        v=torch.zeros_like(master), last=torch.zeros(master.shape[0], dtype=torch.long),
                        visited=torch.zeros(master.shape[0], dtype=torch.bool), axis=axis)
                state = self.states[token]
                dt = (t-state['last'][indices]).double().unsqueeze(1)
                m = state['m'][indices]*(b1**dt).float()+(1-b1)*grad
                v = state['v'][indices]*(b2**dt).float()+(1-b2)*grad.square()
                master = state['master'][indices]
                master.addcdiv_(m/(1-b1**t), (v/(1-b2**t)).sqrt().add_(self.eps), value=-self.lr)
                state['master'][indices] = master
                state['m'][indices] = m; state['v'][indices] = v
                state['last'][indices] = t; state['visited'][indices] = True
                update = master if axis == 0 else master.T
                parameter.index_copy_(axis, gpu_indices, update.to(parameter.device, parameter.dtype))

    def state_dict(self):
        # Only visited native weight slices are needed to reconstruct the final model.
        # Adam moments stay in RAM; this is a model checkpoint, not optimizer resume.
        sparse = {}
        for (key, field), state in self.states.items():
            indices = state['visited'].nonzero().flatten()
            sparse[key+'.'+field] = {'indices': indices, 'axis': state['axis'],
                                    'weights': state['master'][indices].to(getattr(self.mlps[key], field).weight.dtype).clone()}
        return {'step_count': self.step_count, 'slices': sparse,
                'optimizer_resume': False,
                'optimizer': 'native masked Adam with CPU FP32 masters; not LoRA'}


@torch.no_grad()
def restore_native_neurons(model, checkpoint):
    mlps=gated_mlps(model)
    for key,entry in checkpoint['slices'].items():
        name,field=key.rsplit('.',1)
        parameter=getattr(mlps[name],field).weight
        values=entry['weights'] if entry['axis']==0 else entry['weights'].T
        parameter.index_copy_(entry['axis'],entry['indices'].to(parameter.device),
                              values.to(parameter.device,parameter.dtype))

