"""Source-specific recipes. A transferred model/dataset is never author-exact."""
from copy import deepcopy
from itertools import product

FAITHUN=('GA','GradDiff','DPOmis','DPOrej','NPO','RMU','KLUE')
EXTENDED=('SimNPO','CEU','PDU','WGA','SatImp','UNDIAL')
METHODS=('Original',)+FAITHUN+EXTENDED
LORA_METHODS=tuple(m for m in METHODS if m!='KLUE')
LR_GRID=(1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3)


def recipe(method,lr=None):
    if method not in METHODS:raise ValueError(method)
    c=dict(method=method,seed=42,track='native',batch_size=4,microbatch=1,
        epochs=150,learning_rate=1e-5,weight_decay=0.,max_grad_norm=1.,
        lr_scheduler='constant',gamma=.7,alpha=1.,beta=.1,delta=0.,
        stop_ua=.33,stop_scope='training F0 MCQA only; checked each epoch',
        retain_loss_type='NLL',source='FaithUn Section4 and AppendixB.3',
        fidelity='paper algorithm transferred to new model; explicit implementation conventions',
        deviations=['Qwen/FaithUn task transfer; not author-exact model replication',
            'AdamW/betas(.9,.999)/eps1e-8,norm1,constant LR are explicit conventions',
            'paper gives LR range, not exact grid: LR_GRID is our frozen discretization',
            'raw QA training without added template follows FaithUn Section3.4'],
        retain_pool='all66; independent evaluation on test464, not last33 retain',
        use_retain=method not in ('GA','CEU','UNDIAL','Original'))
    if method=='Original':c.update(track='none',epochs=0)
    if method in ('DPOmis','DPOrej'):
        c.update(retain_loss_type='reverse_DPO',rejection='I can’t answer the question.',
                 beta_grid=[.1,.5])
    if method=='NPO':c['beta_grid']=[.1,.5]
    if method=='RMU':
        c.update(steering_coeff=20.,alpha=20.,alpha_grid=[20,50,100,150,200,300],
            module_layer=7,retain_loss_type='EMBED_DIFF',trainable_layers=[5,6,7],
            trainable_fields=['down_proj'],parameter_rule='original WMDP RMU restriction')
        c.update(activation_scope='full QA non-padding tokens, not answer-only')
        c['deviations']+=['WMDP computes full-sequence MSE; QA transfer excludes padding and accumulates by valid token count']
    if method=='KLUE':
        c.update(track='native_neurons',attribution_alpha=10.,mismatch_count=5,
            neuron_fraction=.05,early_stop_ua=.33,convention='pre-down gated neurons,gate/up rows,down columns')
        c['deviations']+=['gated-neuron mapping,FP32 attribution,masked Adam are reimplementation conventions']
    if method in EXTENDED:
        c.update(epochs=10,stop_ua=None,gamma=1.,alpha=1.,beta=1.,
            source='open-unlearning author-contributed community recipe, pinned commit17cbbc8',
            fidelity='author/source algorithm transfer; not original dataset/model result')
    if method=='SimNPO':
        c.update(beta=2.5,gamma=.1375,delta=0.,batch_size=4,seed=1001,weight_decay=.01,
            lr_scheduler='linear',warmup_epochs=1,
            source='OPTML-Group/Unlearn-Simple TOFU/config/forget.yaml blob9aeeac3c1a2f258cbe1dc90e94f8788c8ab038a3')
        c['deviations']+=['author repository default recipe, not an author-reported optimum for Qwen/FaithUn']
    if method in ('WGA','SatImp'):
        c.update(batch_size=16,epochs=5,learning_rate=1e-5,lr_grid=[1e-5,2e-5])
        c['deviations']+=['paper reports effective batch16; contributed shell uses32; paper batch takes precedence']
    if method=='WGA':c['alpha_grid']=[.1,1.,10.]
    if method=='SatImp':c.update(beta1=5.,beta2=1.,gamma=1.,alpha=.1,alpha_grid=[.01,.1,1.])
    if method=='UNDIAL':
        c.update(track='lora',batch_size=32,epochs=10,learning_rate=1e-4,beta=10.,alpha=0.,
            lora_r=8,lora_alpha=16,lora_dropout=0.,lr_grid=[1e-5,1e-4,3e-4],beta_grid=[3,10,30])
    if method=='CEU':c.update(batch_size=32,ignore_first_n_answer_tokens=1)
    if method=='PDU':
        c.update(batch_size=128,epochs=10,alpha=100.,primal_dual=True,
            dual_step_size=5.,dual_warmup_epochs=5,dual_update_upon='step',
            retain_loss_eps=None)
        c['deviations']+=['author TOFU nominal batch4*accum4*8GPUs=128; small final batches recorded explicitly']
        c['deviations']+=['PDU retain epsilon must be set from calibration, never silently guessed']
    if lr is not None:c['learning_rate']=lr
    return deepcopy(c)


def search_plan(method):
    c=recipe(method)
    lrs=LR_GRID if method in FAITHUN else c.get('lr_grid',[c['learning_rate']])
    betas=c.get('beta_grid',[c['beta']]);alphas=c.get('alpha_grid',[c['alpha']])
    return [dict(learning_rate=l,beta=b,alpha=a) for l,b,a in product(lrs,betas,alphas)]


def lora_adapt(c):
    c=deepcopy(c)
    if c['method']=='KLUE':raise ValueError('KLUE native neuron masks have no verified equivalent LoRA implementation; excluded')
    if c['method']=='Original':return c
    c.setdefault('source_track',c['track'])
    c.update(track='lora',lora_adapted=True,lora_r=8,lora_alpha=16,lora_dropout=0.,
        fidelity='explicit LoRA-adapted comparison; source objective retained, not native paper reproduction')
    note='User-authorized LoRA adaptation; source loss, batch, schedule and stopping preserved; parameter domain changed'
    if note not in c['deviations']:c['deviations'].append(note)
    return c


def lora_targets(names,c):
    if c['method']!='RMU':return list(names)
    return [name for name in names if name.endswith('.mlp.down_proj') and
            any(f'.layers.{i}.' in name for i in c['trainable_layers'])]


def validate(c):
    expected='none' if c['method']=='Original' else 'lora' if c['method']=='UNDIAL' else 'native_neurons' if c['method']=='KLUE' else 'native'
    explicit_lora=c.get('lora_adapted',False) and c['method'] in LORA_METHODS and c['method']!='Original'
    if c['track']!=('lora' if explicit_lora else expected):raise ValueError('Track differs from registered source recipe: '+expected)
    if c.get('lora_adapted') and c['method']=='KLUE':raise ValueError('KLUE excluded from LoRA comparison')
    if c['method']=='PDU':
        if c['retain_loss_eps'] is None:raise ValueError('Calibrate and explicitly set PDU retain_loss_eps')
        if c['epochs']<=c['dual_warmup_epochs']:raise ValueError('PDU budget never enables dual updates')
    if c['batch_size']<c['microbatch'] or c['batch_size']%c['microbatch']:
        raise ValueError('Microbatch must divide effective batch size')
    if c['microbatch']!=1:raise ValueError('Current memory-safe runner implements single-example microbatches only')
    if c['method'] in FAITHUN and c['track']=='lora' and not explicit_lora:raise ValueError('FaithUn track must not silently become LoRA')
