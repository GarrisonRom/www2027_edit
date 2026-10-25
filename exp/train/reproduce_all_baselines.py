"""Auditable upstream/native baseline queue. Never overwrites legacy experiments."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

import reproduction_losses as rl
import native_delta_checkpoints as native_delta
import run_remote_baselines as base
import run_cloud_gsu_c2 as routing
import evaluate_cloud_capabilities as capability

WS = Path(__file__).resolve().parents[2]
VENV = '/root/autodl-tmp/venvs/lm_eval_0411_capability_20261023_v2/bin/python'
MODELS = [('llama32_3b','/root/autodl-tmp/models/Llama-3.2-3B-Instruct'),
          ('qwen35_4b','/root/autodl-tmp/models/Qwen3.5-4B')]
WIKI_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def write(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def fp32_loss_outputs(module,args,output):
    # HF Accelerate promotes model outputs for loss computation in its BF16 wrapper.
    # Keep hidden hooks native, but do not compute NPO/PDU/weighted CE in BF16 logits.
    output.logits=output.logits.float()
    if getattr(output,'loss',None) is not None:output.loss=output.loss.float()
    return output


def encode(tok, item, device, pad_to=None):
    full, context = base.losses.te.enc_qa(tok,item['question'],item['answer'])
    ids = torch.tensor([full],device=device)
    labels = ids.clone(); labels[:,:context] = -100
    if not bool((labels[:,1:] != -100).any()):
        raise ValueError('No answer labels')
    mask=torch.ones_like(ids)
    if pad_to is not None:
        extra=pad_to-ids.shape[1]
        if extra<0:raise ValueError('Training padding limit must not truncate examples')
        pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if pad is None:raise ValueError('Need tokenizer pad or EOS for native batch padding')
        ids=torch.cat([ids,torch.full((1,extra),pad,dtype=ids.dtype,device=device)],dim=1)
        labels=torch.cat([labels,torch.full((1,extra),-100,dtype=labels.dtype,device=device)],dim=1)
        mask=torch.cat([mask,torch.zeros((1,extra),dtype=mask.dtype,device=device)],dim=1)
    return dict(input_ids=ids,attention_mask=mask,labels=labels)


def collate(tok,items,device,pad_to=None):
    if pad_to is None:
        pad_to=max(len(base.losses.te.enc_qa(tok,x['question'],x['answer'])[0]) for x in items)
    encoded=[encode(tok,item,device,pad_to) for item in items]
    return {key:torch.cat([row[key] for row in encoded],dim=0) for key in encoded[0]}


def alternate(item, pool, method, rng):
    result = dict(item)
    if method == 'DPOrej':
        result['answer'] = "I don't know."
    elif method == 'DPOmis':
        eligible = [row['answer'] for row in pool if row['answer'] != item['answer']]
        if not eligible: raise ValueError('No mismatched answer')
        result['answer'] = rng.choice(eligible)
    else:
        raise ValueError(method)
    return result


class ReferenceBank(torch.nn.Module):
    """Frozen base logits/activations cached before any update, never a moving reference."""
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Identity()
        self.entries = {}

    @staticmethod
    def key(batch):
        return tuple(batch['input_ids'].detach().cpu().flatten().tolist())

    @torch.no_grad()
    def capture(self, model, batches, use_hidden=False):
        previous = model.training
        model.eval()
        for batch in batches:
            key = self.key(batch)
            if key in self.entries: continue
            hidden, handles = [], []
            if use_hidden:
                handles.append(rl.decoder_layer(model,7).register_forward_hook(
                    lambda module,args,out: hidden.append((out[0] if isinstance(out,tuple) else out).detach().cpu())))
            try:
                output = model(**batch)
                self.entries[key] = (output.logits.detach().cpu(),hidden[0] if hidden else None)
            finally:
                for handle in handles: handle.remove()
        model.train(previous)

    def forward(self,input_ids,labels=None,**kwargs):
        entries=[self.entries[self.key({'input_ids':row.unsqueeze(0)})] for row in input_ids]
        logits=torch.cat([entry[0] for entry in entries],dim=0).to(input_ids.device)
        if entries[0][1] is not None:
            self.layer(torch.cat([entry[1] for entry in entries],dim=0).to(input_ids.device))
        loss = None if labels is None else torch.nn.functional.cross_entropy(
            logits[:,:-1].float().transpose(1,2),labels[:,1:],ignore_index=-100)
        return SimpleNamespace(logits=logits,loss=loss)


class CPUAdamW:
    """AdamW on CPU FP32 master weights, copied back without a second GPU model."""
    def __init__(self, parameters, lr, weight_decay):
        self.pairs = [(p,torch.nn.Parameter(p.detach().float().cpu().clone())) for p in parameters]
        self.optimizer = torch.optim.AdamW([m for _,m in self.pairs],lr=lr,
                                            weight_decay=weight_decay,betas=(.9,.999),eps=1e-8)
    @torch.no_grad()
    def step(self):
        for p, master in self.pairs:
            master.grad = None if p.grad is None else p.grad.detach().float().cpu()
        self.optimizer.step()
        for p, master in self.pairs: p.copy_(master.to(p.device,p.dtype))
        self.optimizer.zero_grad(set_to_none=True)


def source_manifest():
    paths = list((WS/'exp/train').glob('*.py')) + list((rl.UPSTREAM/'src/trainer').rglob('*.py'))
    paths += list((rl.UPSTREAM/'configs/trainer').glob('*.yaml'))
    return {str(p.relative_to(WS)):rl.digest(p) for p in sorted(paths)}


def remembered(model,tok,items):
    previous=model.training; model.eval()
    with torch.no_grad():
        result=[]
        for item in items:
            scores=base.losses.te.score_choices(model,tok,item['question'],
                                                [item['answer']]+list(item['false_options']))
            if scores[0] > max(scores[1:]):result.append(item)
    model.train(previous)
    return result


def klue_selection(model,tok,items,pool,mlps,cfg,rng):
    aggregate={key:torch.zeros(m.up_proj.out_features,dtype=torch.float64) for key,m in mlps.items()}
    metadata=[]
    for item in items:
        genuine,lp=rl.probability_attribution(model,encode(tok,item,model.device),mlps)
        eligible=[row for row in pool if row['question']!=item['question'] and row['answer']!=item['answer']]
        if len(eligible)<cfg['mismatch_count']:raise RuntimeError('Mismatch pool too small')
        wrong=[]; ids=[]
        for sample in rng.sample(eligible,cfg['mismatch_count']):
            mismatch=dict(item,question=sample['question'])
            scores,_=rl.probability_attribution(model,encode(tok,mismatch,model.device),mlps)
            wrong.append(scores);ids.append(sample['id'])
        importance=rl.regularized_importance(genuine,wrong,cfg['attribution_alpha'])
        for key in aggregate:aggregate[key]+=importance[key]/len(items)
        metadata.append({'id':item['id'],'log_probability':lp,'mismatch_ids':ids})
    return rl.global_neurons(aggregate,cfg['neuron_fraction']),metadata


def paper_scope(model,tok,inputs,smoke):
    """Disclose our raw/full-sequence scoring, not a hidden claim of identical paper prompts."""
    result={}
    for split in ('forget','test'):
        direct=[];multi=[]
        indices=inputs['indices'][split][:1] if smoke else inputs['indices'][split]
        for ci in indices:
            cluster=inputs['clusters'][ci][2]
            for q in cluster['hop1_test']['questions']:
                f=cluster['hop1_test'];s=base.losses.te.score_choices(model,tok,q,[f['answers']]+list(f['false_answer_options']))
                direct.append(int(s[0]>max(s[1:])))
            f=cluster.get('hop2_test',{})
            for pi,questions in enumerate(f.get('questions',[])):
                for q in questions:
                    s=base.losses.te.score_choices(model,tok,q,[f['answers'][pi]]+list(f['false_answer_options'][pi]))
                    multi.append(int(s[0]>max(s[1:])))
        result[split]={'direct_n':len(direct),'direct_accuracy':sum(direct)/len(direct),
                       'multihop_n':len(multi),'multihop_accuracy':sum(multi)/len(multi)}
    result['AA']=(1-result['forget']['direct_accuracy']+result['test']['direct_accuracy'])/2
    result['MA']=(1-result['forget']['multihop_accuracy']+result['test']['multihop_accuracy'])/2
    result['scoring']='raw QA full-sequence mean LP, not asserted identical to author prompt'
    return result


def evaluate_all(model,tok,inputs,run,smoke,retain_is_heldout=True):
    model.eval()
    behavior=base.evaluate(model,tok,inputs)
    behavior['full_hop2']=routing.full_hop2_audit(model,tok,inputs)
    write(run/'behavior.json',behavior)
    corpus,meta=capability.load_wikitext(WIKI_REVISION)
    ppl=capability.token_ppl(model,tok,corpus[:24000] if smoke else corpus,
                              token_limit=10000 if smoke else None)
    write(run/'ppl.json',{'dataset':meta,'metric':ppl})
    mmlu=capability.mmlu_eval(model,tok,limit=1 if smoke else 10,output_dir=run)
    write(run/'mmlu.json',mmlu)
    # Smoke excludes the expensive test464 audit but still checks representative test input.
    write(run/'paper_scope.json',paper_scope(model,tok,inputs,smoke))
    return {'all89':behavior['full_hop2']['all_hop2']['summary']['path_macro_accuracy'],
            'f0':behavior['groups']['F0']['summary']['mcqa_accuracy'],
            'retain':behavior['groups']['retain_heldout']['summary']['mcqa_accuracy'],
            'retain_is_heldout':retain_is_heldout,
            'mmlu':mmlu['macro_subject_accuracy'],'ppl':ppl['token_ppl'],
            'smoke':smoke}


def model_manifest(model_path):
    root = Path(model_path).resolve()
    # Local ModelScope downloads do not have the cloud server's global manifest.
    files = [p for p in sorted(root.iterdir()) if p.is_file()]
    return {'path': str(root), 'files': {
        p.name: {'bytes': p.stat().st_size, 'sha256': rl.digest(p)} for p in files}}


def worker(args):
    run=Path(args.run_dir).resolve();run.mkdir(parents=True,exist_ok=False)
    cfg=rl.profile(args.method,args.track)
    if getattr(args,'learning_rate',None) is not None:cfg['learning_rate']=float(args.learning_rate)
    seed=getattr(args,'seed',42)
    cfg.update(model=args.model,seed=seed,smoke=args.smoke,
               lr_search='no test-based tuning; configured source recipe, additional trials separate',
               optimizer='native neuron CPU masked Adam' if args.method=='KLUE' else
                         'CPU-master AdamW' if args.track=='native' else 'torch AdamW',
               prompt='raw question-space-answer, no chat template; source-paper prompt not certified identical',
               native_compatibility='full parameter update except native KLUE gated FFN neuron mask')
    if args.smoke: cfg['max_updates']=2;cfg['epochs']=1
    cfg['supervision']=getattr(args,'supervision','F0')
    cfg['sdpa_math']=getattr(args,'sdpa_math',False)
    cfg['development']=getattr(args,'development_plan',None)
    write(run/'config.json',cfg)
    write(run/'manifest.json',{'created':base.protocol.timestamp(),'source':source_manifest(),
                               'inputs':base.protocol.input_manifest(),'command':sys.argv,
                               'torch':torch.__version__,'cuda':torch.version.cuda,
                               'model_manifest':model_manifest(args.model)})
    actual_manifest=Path(args.model)/'download_manifest_8b.json'
    if actual_manifest.exists():
        manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        manifest['actual_model_manifest']=json.loads(actual_manifest.read_text(encoding='utf-8'))
        write(run/'manifest.json',manifest)
    (run/'environment.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    (run/'run_log.md').write_text('# Baseline reimplementation\n\nConfig and manifest frozen before loading.\n'
        'No result is an official author-code replication. Source loss and declared adaptations are separate.\n',encoding='utf-8')
    start=time.time();write(run/'status.json',{'status':'running','phase':'loading'})
    try:
        if getattr(args,'sdpa_math',False):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda,'enable_cudnn_sdp'):torch.backends.cuda.enable_cudnn_sdp(False)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed)
        inputs=base.protocol.load_inputs()
        if getattr(args,'development_plan',None):
            import development_evaluation as dev_eval
            dev_plan=json.loads(Path(args.development_plan).read_text(encoding='utf-8'))
            inputs=dev_eval.task_inputs(inputs,dev_plan)
            write(run/'development_plan.json',dev_plan)
        forget=inputs['F0'];retain=inputs['retain_all'][:33] if args.track=='matched_lora' else inputs['retain_all']
        if getattr(args,'supervision','F0')=='F0F1':forget=forget+inputs['F1']
        if args.track=='matched_lora' and cfg['max_updates']:
            cfg['epochs']=max(cfg['epochs'],math.ceil(cfg['max_updates']/len(forget)))
        tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        model=base.load_model(args.model)
        streaming=None
        if getattr(args,'gradient_checkpointing',False):
            if args.track=='native':
                model.enable_input_require_grads()
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.config.use_cache=False
        config=getattr(model.config,'text_config',model.config)
        if hasattr(config,'use_cache'):config.use_cache=False
        for p in model.parameters():p.requires_grad=False
        if args.track=='matched_lora':
            targets=base.target_modules(model)
            model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,bias='none',
                                                   task_type='CAUSAL_LM',target_modules=targets))
        elif args.method=='KLUE':
            targets=[]
            mlps=rl.gated_mlps(model)
            for mlp in mlps.values():
                for field in ('gate_proj','up_proj','down_proj'):
                    getattr(mlp,field).weight.requires_grad=True
                    targets.append(field)
        else:
            targets=[]
            for name,p in model.named_parameters():
                if not any(x in name for x in ('visual','vision','mtp')):p.requires_grad=True;targets.append(name)
        params=[p for p in model.parameters() if p.requires_grad]
        if getattr(args,'native_streaming',False):
            from native_cpu_streaming import NativeCPUStreaming
            streaming=NativeCPUStreaming(model)
            cfg['optimizer']='CPU FP32 moments, native BF16 weights, no extra FP32 masters'
            cfg['execution']='native CPU parameter/gradient streaming; same parameter domain and loss'
        if getattr(args,'gradient_checkpointing',False) and args.track=='matched_lora':
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        cfg['gradient_checkpointing']=getattr(args,'gradient_checkpointing',False)
        cfg['native_streaming']=getattr(args,'native_streaming',False)
        model.register_forward_hook(fp32_loss_outputs)
        original_development=None
        if getattr(args,'development_plan',None):
            original_development=dev_eval.evaluate(model,tok,inputs)
            write(run/'development_original.json',original_development)
        cfg['loss_precision']='BF16 backbone, FP32 returned logits/loss matching Accelerate output conversion'
        write(run/'modules.json',{'targets':targets,'trainable_parameters':sum(p.numel() for p in params),
                                  'model_parameters':sum(p.numel() for p in model.parameters()),
                                  'gpu':torch.cuda.get_device_name(0)})
        native_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
        if args.track=='native' and args.method!='KLUE' and shutil.disk_usage(run).free < 2*1024**3:
            raise RuntimeError('Native checkpoint storage gate: no silent conversion to LoRA')
        if args.track=='native' and args.method!='KLUE' and not args.smoke:
            if streaming is not None:
                from model_8b_resources import DiskBaseWeights
                native_base=DiskBaseWeights(model,args.model)
            else:native_base=native_delta.capture_base(model)
        else:native_base=None
        rng=random.Random(seed)
        alternates={item['id']:alternate(item,retain,args.method,rng) for item in forget} if args.method.startswith('DPO') else {}
        aligned_schedule=None
        if getattr(args,'aligned_schedule',False) and args.track=='matched_lora':
            aligned_schedule=base.protocol.make_update_schedule(forget,cfg['max_updates'],seed)
            rng=random.Random(seed)
            write(run/'aligned_schedule.json',{'forget':aligned_schedule,'seed':seed,
                   'contract':'same F0 order as C1/C2, separate training-retain RNG'})
        write(run/'data_plan.json',{'forget_ids':[r['id'] for r in forget],
            'retain_ids':[r['id'] for r in retain],'alternates':alternates,
            'mismatch_rule':'distinct question and differing true answer, drawn only from forget/retain'})
        pad_to=None
        if args.track=='native':
            pad_to=max(len(base.losses.te.enc_qa(tok,x['question'],x['answer'])[0])
                       for x in forget+retain+list(alternates.values()))
        cfg['training_pad_to']=pad_to
        cfg['batching']='actual padded batch, not serial per-example gradient averaging'
        write(run/'config.json',cfg)
        bank=None
        if args.method in ('NPO','DPOmis','DPOrej','RMU','UNDIAL'):
            bank=ReferenceBank()
            bank.capture(model,(encode(tok,x,model.device,pad_to) for x in forget+retain+list(alternates.values())),args.method=='RMU')
        env,upstream=rl.upstream_namespace()
        write(run/'upstream_hashes.json',upstream)
        trainer=None if args.method=='KLUE' else rl.bind_trainer(cfg,model,bank,env)
        if trainer is not None and args.method=='PDU':
            with torch.no_grad():
                initial=[float(model(**encode(tok,x,model.device,pad_to)).loss) for x in retain]
            trainer.retain_loss_eps=sum(initial)/len(initial)
            write(run/'pdu_constraint.json',{'retain_loss_eps':trainer.retain_loss_eps,
                  'source':'initial train-retain mean NLL; source YAML unresolved ??? filled explicitly'})
        if args.method=='KLUE':optimizer=rl.NativeNeuronAdam(mlps,cfg['learning_rate'])
        elif args.track=='native':
            if streaming is not None:
                from native_cpu_streaming import StreamingAdamW
                optimizer=StreamingAdamW(params,cfg['learning_rate'],cfg['weight_decay'])
            else:optimizer=CPUAdamW(params,cfg['learning_rate'],cfg['weight_decay'])
        else:optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        update=0;attempted=0;skipped=[];stop=None
        model.train()
        with (run/'diagnostics.jsonl').open('x',encoding='utf-8') as log:
            for epoch in range(cfg['epochs']):
                if aligned_schedule is not None:
                    plan=aligned_schedule[epoch*len(forget):(epoch+1)*len(forget)]
                    byid={item['id']:item for item in forget}
                    active=[byid[item['item_id']] for item in plan]
                else:active=remembered(model,tok,forget) if args.method=='KLUE' else list(forget)
                if args.method=='KLUE':
                    write(run/f'remembered_epoch_{epoch}.json',{'ids':[x['id'] for x in active],
                          'ua':len(active)/len(forget),'criterion':'raw full-sequence MCQA correct'})
                    if len(active)/len(forget)<=cfg['early_stop_ua'] and not args.smoke:
                        stop='training_UA_threshold';break
                if aligned_schedule is None:rng.shuffle(active)
                if trainer is not None and args.method=='PDU':
                    trainer.can_update=epoch>=cfg.get('dual_warmup_epochs',0)
                for pos in range(0,len(active),cfg['batch_size']):
                    if time.time()-start>getattr(args,'timeout_hours',6)*3600:raise TimeoutError('Per-run training budget')
                    current=active[pos:pos+cfg['batch_size']]
                    model.zero_grad(set_to_none=True)
                    selected=None;attr=None
                    if args.method=='KLUE':
                        selected,attr=klue_selection(model,tok,current,forget+retain,mlps,cfg,rng)
                    valid_items=[]
                    for item in current:
                        attempted+=1
                        single=encode(tok,item,model.device)
                        if args.method=='CEU' and int((single['labels'][:,1:]!=-100).sum())<=trainer.ignore_first_n_answer_tokens:
                            skipped.append({'id':item['id'],'reason':'upstream CEU ignores all answer labels'})
                        else:valid_items.append(item)
                    if not valid_items:continue
                    keeps=[rng.choice(retain) for _ in valid_items]
                    f=collate(tok,valid_items,model.device,pad_to)
                    r=collate(tok,keeps,model.device,pad_to)
                    if args.method=='KLUE':
                        loss=-cfg['gamma']*model(**f).loss+cfg['alpha']*model(**r).loss
                    else:
                        batch={'forget':f,'retain':r}
                        if alternates:
                            batch['forget']={'original':f,'alternate':collate(tok,[alternates[x['id']] for x in valid_items],model.device,pad_to)}
                        loss=trainer.compute_loss(model,batch)
                    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss; no formula fallback')
                    loss.backward()
                    total=float(loss.detach());valid=len(valid_items)
                    if not all(bool(torch.isfinite(p.grad).all()) for p in params if p.grad is not None):
                        raise RuntimeError('Nonfinite gradients')
                    if selected is not None:
                        for key,mlp in mlps.items():
                            grad_device=mlp.up_proj.weight.grad.device
                            mask=torch.zeros(mlp.up_proj.out_features,device=grad_device)
                            mask[selected[key].to(grad_device)]=1
                            mlp.gate_proj.weight.grad.mul_(mask[:,None])
                            mlp.up_proj.weight.grad.mul_(mask[:,None])
                            mlp.down_proj.weight.grad.mul_(mask[None,:])
                    if cfg['max_grad_norm'] is not None:
                        torch.nn.utils.clip_grad_norm_(params,cfg['max_grad_norm'],error_if_nonfinite=True)
                    effective_lr=cfg['learning_rate']
                    if cfg['lr_scheduler']=='linear':
                        planned=math.ceil(len(forget)/cfg['batch_size'])*cfg['epochs']
                        effective_lr*=max(0.,1-update/planned)
                    if args.method=='KLUE':optimizer.lr=effective_lr
                    else:
                        inner=optimizer.optimizer if isinstance(optimizer,CPUAdamW) else optimizer
                        for group in inner.param_groups:group['lr']=effective_lr
                    if args.method=='KLUE':optimizer.step(selected)
                    else:optimizer.step()
                    if streaming is not None:streaming.clear_transfers()
                    update+=1
                    row={'update':update,'epoch':epoch,'loss':total,'valid':valid,
                         'learning_rate':effective_lr,
                         'ids':[x['id'] for x in current],'seconds':time.time()-start}
                    if selected is not None:row.update(neuron_indices={k:v.tolist() for k,v in selected.items()},attribution=attr)
                    if args.method=='PDU':row.update(dual_enabled=trainer.can_update,preferences=trainer.preferences)
                    log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                    write(run/'status.json',{'status':'running','phase':'training','update':update,'epoch':epoch})
                    print('UPDATE',args.method,args.track,update,'loss',row['loss'],flush=True)
                    if cfg['max_updates'] and update>=cfg['max_updates']:stop='update_budget';break
                if stop:break
        write(run/'training_summary.json',{'updates':update,'attempted_examples':attempted,
            'skipped':skipped,'stop_reason':stop or 'epoch_budget',
            'reference':'base outputs cached before first parameter update' if bank else None})
        if update==0:raise RuntimeError('No optimizer updates; do not report a trained baseline')
        model.zero_grad(set_to_none=True)
        if not args.smoke:
            if getattr(args,'development_plan',None):pass
            elif args.method=='KLUE':torch.save(optimizer.state_dict(),run/'native_neuron_checkpoint.pt')
            elif args.track=='native':native_delta.save_delta(model,native_base,run/'native_delta')
            else:model.save_pretrained(run/'adapter')
        del optimizer,bank,trainer
        del native_base
        gc.collect();torch.cuda.empty_cache()
        write(run/'status.json',{'status':'running','phase':'evaluation','updates':update})
        if getattr(args,'development_plan',None):
            metrics=dev_eval.evaluate(model,tok,inputs)
            metrics.update(original_retain_accuracy=original_development['retain_accuracy'],
                           learning_rate=cfg['learning_rate'],method=args.method,status='completed')
            write(run/'development_metrics.json',metrics)
        else:metrics=evaluate_all(model,tok,inputs,run,args.smoke,args.track=='matched_lora')
        write(run/'metrics.json',metrics)
        write(run/'status.json',{'status':'completed','updates':update,'seconds':time.time()-start,
              'ended':base.protocol.timestamp(),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
    except BaseException:
        write(run/'status.json',{'status':'failed','ended':base.protocol.timestamp(),
                                'seconds':time.time()-start,'traceback':traceback.format_exc()})
        raise


def queue(args):
    root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    jobs=[]
    # KLUE native first: the previously missing baseline gets real code and smoke before formal jobs.
    specs=[('KLUE','native')]+[(m,'matched_lora') for m in rl.METHODS if m!='KLUE']
    specs += [(m,'native') for m in rl.METHODS if m!='KLUE']
    for method,track in specs:
        for key,model in MODELS:
            for smoke in (True,False):
                label=f'{key}_{method}_{track}'+('_smoke' if smoke else '')
                command=[sys.executable,str(Path(__file__).resolve()),'--method',method,'--track',track,
                         '--model',model,'--run_dir',str(root/label)]+(['--smoke'] if smoke else [])
                jobs.append({'label':label,'method':method,'track':track,'smoke':smoke,'command':command})
    manifest={'created':base.protocol.timestamp(),'source':source_manifest(),'jobs':jobs,
              'dependency':args.after,'max_queue_hours':72,
              'profiles':[rl.profile(m,t) for m,t in specs],
              'legacy_results':'not overwritten or relabeled as new reproduction'}
    write(root/'batch_manifest.json',manifest)
    summarize(root)
    if getattr(args,'plan_only',False):
        print(json.dumps({'formal_jobs':sum(not j['smoke'] for j in jobs),
                          'smoke_jobs':sum(j['smoke'] for j in jobs),'run_dir':str(root)}))
        return
    done=[];failures=[];start=time.time();failed_smokes=set()
    if args.after:
        while True:
            dependency=json.loads(Path(args.after).read_text(encoding='utf-8'))
            if dependency['status']=='completed':break
            if dependency['status'] in ('failed','interrupted'):
                write(root/'status.json',{'status':'dependency_failed','dependency':dependency});return
            write(root/'status.json',{'status':'waiting_for_dependency','dependency':args.after})
            time.sleep(20)
    for job in jobs:
        signature=job['label'].removesuffix('_smoke')
        if not job['smoke'] and signature in failed_smokes:
            failures.append({'label':job['label'],'reason':'smoke_failed_not_run'});continue
        if time.time()-start>72*3600:
            write(root/'status.json',{'status':'budget_exhausted','completed':done,'failures':failures});return
        for path,sha in manifest['source'].items():
            if rl.digest(WS/path)!=sha:raise RuntimeError('Frozen source changed: '+path)
        write(root/'status.json',{'status':'running','current':job['label'],'completed':done,'failures':failures})
        with (root/(job['label']+'.stdout.log')).open('x') as out,(root/(job['label']+'.stderr.log')).open('x') as err:
            try:code=subprocess.run(job['command'],cwd=WS,stdout=out,stderr=err,timeout=7*3600).returncode
            except subprocess.TimeoutExpired:code=124
        if code:
            failures.append({'label':job['label'],'exit_code':code})
            if job['smoke']:failed_smokes.add(signature)
        else:done.append(job['label'])
        summarize(root)
    write(root/'status.json',{'status':'completed' if not failures else 'completed_with_failures',
                             'completed':done,'failures':failures,'ended':base.protocol.timestamp()})


def summarize(root):
    rows=[]
    for folder in root.iterdir():
        if not folder.is_dir() or not (folder/'metrics.json').exists():continue
        cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'));state=json.loads((folder/'status.json').read_text(encoding='utf-8'))
        if cfg['smoke'] or state['status']!='completed':continue
        row=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        row.update(method=cfg['method'],track=cfg['track'],model=Path(cfg['model']).name,run_id=folder.name,
                   evidence=str(folder/'metrics.json'),sha256=rl.digest(folder/'metrics.json'))
        rows.append(row)
    write(root/'comparison.json',{'rows':rows,'note':'separate native and LoRA tracks; no combined rank'})
    lines=['# Complete baseline reproduction: measured endpoints only','',
           '| Model | Method | Track | All89 recall down | MMLU-10/subject up | WikiText2 PPL down |',
           '|---|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['model']} | {r['method']} | {r['track']} | {r['all89']:.4f} | {r['mmlu']:.4f} | {r['ppl']:.4f} |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status_rows=[]
    manifest=json.loads((root/'batch_manifest.json').read_text(encoding='utf-8'))
    for job in manifest['jobs']:
        if job['smoke']:continue
        status_file=root/job['label']/'status.json'
        state=json.loads(status_file.read_text(encoding='utf-8')) if status_file.exists() else {'status':'not_started'}
        status_rows.append({'run_id':job['label'],'method':job['method'],'track':job['track'],**state})
    write(root/'method_status.json',{'runs':status_rows,'measured':rows})
    for target in ('report_v2','report_v3_klue_pending'):
        path=WS/'exp/analysis/cloud_comparison_20261022'/target/'README.md'
        if not path.exists():continue
        text=path.read_text(encoding='utf-8')
        marker='## 完整baseline复现实验（exp015）'
        if marker in text:text=text[:text.index(marker)].rstrip()+'\n'
        progress=[marker,'','独立原生/LoRA两轨，先导旧结果不冒充复现。失败和未运行不填分数。','',
                  '| 方法 | 原生代码 | 云端完成正式端点数 | 状态来源 |','|---|---|---:|---|']
        for method in rl.METHODS:
            count=sum(row['method']==method for row in rows)
            progress.append(f'| {method} | 已实现，需真实模型门检 | {count} | exp015/method_status.json |')
        progress+=['']+lines[2:]
        path.write_text(text+'\n'+'\n'.join(progress)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--queue',action='store_true');parser.add_argument('--after')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--gradient_checkpointing',action='store_true')
    parser.add_argument('--native_streaming',action='store_true')
    parser.add_argument('--timeout_hours',type=float,default=6.)
    parser.add_argument('--aligned_schedule',action='store_true')
    parser.add_argument('--supervision',choices=('F0','F0F1'),default='F0')
    parser.add_argument('--learning_rate',type=float)
    parser.add_argument('--sdpa_math',action='store_true')
    parser.add_argument('--development_plan')
    parser.add_argument('--plan_only',action='store_true')
    parser.add_argument('--method',choices=rl.METHODS);parser.add_argument('--track',choices=('matched_lora','native'))
    parser.add_argument('--model');parser.add_argument('--run_dir',required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.queue:queue(args)
    else:
        if not args.method or not args.track or not args.model:parser.error('worker requires method, track and model')
        worker(args)

if __name__=='__main__':main()

