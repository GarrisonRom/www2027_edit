"""Independent native/source-specific runner. No silent LoRA or budget fallback."""
import argparse
import gc
import json
import math
import random
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from .bootstrap import ROOT,sha,verify
from .recipes import recipe,search_plan,validate,METHODS,lora_adapt,lora_targets
from .protocol import encode,scores,classify
from .objectives import DiskReference,faithun_dpo_loss,rmu_loss
from .evaluate import evaluate
from .runtime import load


def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def git_commit():
    try:
        result=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True,capture_output=True)
        return result.stdout.strip() if result.returncode==0 else None
    except FileNotFoundError:return None


def alternative(item,pool,method,rng,rejection):
    answer=rejection
    if method=='DPOmis':
        answers=sorted({r['answer'] for r in pool if r['answer']!=item['answer']})
        if not answers:raise ValueError('No distinct alternative answer')
        answer=rng.choice(answers)
    return dict(item,answer=answer)


def target_count(batch,skip=0,shift=True):
    labels=batch['labels'][:,1:] if shift else batch['labels']
    return max(0,int((labels!=-100).sum())-skip)


def learning_rate(cfg,update,steps_per_epoch):
    if cfg['lr_scheduler']=='constant':return cfg['learning_rate']
    warmup=cfg.get('warmup_epochs',0)*steps_per_epoch
    total=cfg['epochs']*steps_per_epoch
    if update<warmup:return cfg['learning_rate']*update/max(1,warmup)
    return cfg['learning_rate']*max(0.,(total-update)/max(1,total-warmup))


def micro_weights(method,forgets,retains):
    # Match source reductions across variable answer lengths; simple mean-of-means is wrong.
    ft=[target_count(x,1 if method=='CEU' else 0,method!='PDU') for x in forgets]
    rt=[target_count(x) for x in retains]
    if method=='RMU':
        ft=[int(x.get('attention_mask',torch.ones_like(x['labels'])).sum()) for x in forgets]
        rt=[int(x.get('attention_mask',torch.ones_like(x['labels'])).sum()) for x in retains]
    if method in ('NPO','SimNPO','DPOmis','DPOrej'):
        fw=[1/len(ft)]*len(ft)
    else:fw=[x/sum(ft) for x in ft]
    rw=[1/len(rt)]*len(rt) if method in ('DPOmis','DPOrej') else [x/sum(rt) for x in rt]
    return list(zip(fw,rw))


def remembered(model,tok,items):
    model.eval();active=[]
    for item in items:
        if classify(scores(model,tok,item,True)['answer_sum'])['correct']:active.append(item)
    model.train();return active


def resource_gate(params,track):
    import psutil
    n=sum(p.numel() for p in params)
    required=(16*n+2*1024**3) if track!='lora' else (16*n+512*1024**2)
    available=psutil.virtual_memory().available
    if available<required:
        raise RuntimeError(f'Host RAM gate: need estimated{required}bytes, available{available}; no LoRA fallback')
    return dict(estimated_cpu_bytes=required,available_cpu_bytes=available)


def worker(args):
    core=load();cfg=recipe(args.method,args.learning_rate)
    if args.config:cfg.update(json.loads(Path(args.config).read_text(encoding='utf-8')))
    if args.lora_adapted:cfg=lora_adapt(cfg)
    if cfg['method']!=args.method:raise ValueError('Method/config mismatch')
    if args.retain_eps is not None:cfg['retain_loss_eps']=args.retain_eps
    validate(cfg)
    run=Path(args.run_dir).resolve();run.mkdir(parents=True,exist_ok=False)
    cfg.update(model=args.model,smoke=args.smoke,
        evaluation='FaithUn prompt + conditional mean/sum + raw sensitivity + full test464 + MMLU/PPL',
        checkpoint='native full model or original-source LoRA; never discard on failed evaluation')
    write(run/'config.json',cfg)
    files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if 'runs' not in p.relative_to(ROOT).parts}
    write(run/'manifest.json',dict(source=files,dependencies=verify(),command=sys.argv,
        seed=cfg['seed'],model=core.model_manifest(args.model),created=core.base.protocol.timestamp(),
        git_commit=git_commit(),
        deterministic_algorithms=False,cudnn_benchmark=torch.backends.cudnn.benchmark))
    (run/'environment.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    if git_commit() is not None:
        with (run/'patch.diff').open('wb') as f:subprocess.run(['git','diff','--binary'],cwd=ROOT.parent,stdout=f,check=True)
    else:(run/'patch.diff').write_text('Source snapshot without Git checkout; exact source hashes recorded in manifest.json.',encoding='utf-8')
    (run/'run_log.md').write_text('Source-specific algorithm transfer. Config/dependency hashes frozen before loading. No author-exact claim. Failed/incomplete capabilities remain explicit.',encoding='utf-8')
    start=time.time();write(run/'status.json',dict(status='running',phase='loading'))
    try:
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        random.seed(cfg['seed']);rng=random.Random(cfg['seed']);alt_rng=random.Random(cfg['seed']+2718)
        data=core.base.protocol.load_inputs();forget=data['F0'];retain=data['retain_all']
        tok=core.AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        model=core.base.load_model(args.model);model.config.use_cache=False
        if args.method=='Original':
            total_nll=0.;total_tokens=0;model.eval()
            with torch.no_grad():
                for item in retain:
                    b=encode(tok,item,model.device);count=target_count(b)
                    total_nll+=float(model(**b).loss)*count;total_tokens+=count
            write(run/'retain_calibration.json',dict(initial_retain_token_nll=total_nll/total_tokens,
                training_retain_count=len(retain),tokens=total_tokens,
                pdu_rule='epsilon must exceed this value; select using registered training/development criteria'))
            metrics=evaluate(model,tok,data,run/'evaluation',limit=1 if args.smoke else None,
                generation=args.generation,capabilities=not args.behavior_only,mmlu_limit=None if args.mmlu_full else 10)
            result_status='smoke_passed' if args.smoke and metrics['evaluation_complete'] else 'completed' if metrics['evaluation_complete'] else 'evaluation_incomplete'
            write(run/'status.json',dict(status=result_status,seconds=time.time()-start))
            return
        for p in model.parameters():p.requires_grad=False
        if cfg['track']=='lora':
            from peft import get_peft_model,LoraConfig
            model=get_peft_model(model,LoraConfig(r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],
                lora_dropout=cfg['lora_dropout'],task_type='CAUSAL_LM',bias='none',
                target_modules=lora_targets(core.base.target_modules(model),cfg)))
            model.enable_input_require_grads()
        elif cfg['track']=='native_neurons':
            for mlp in core.rl.gated_mlps(model).values():
                for field in ('gate_proj','up_proj','down_proj'):getattr(mlp,field).weight.requires_grad=True
        elif args.method=='RMU':
            for name,p in model.named_parameters():
                p.requires_grad=any(f'.layers.{i}.mlp.down_proj.weight' in name for i in cfg['trainable_layers'])
        else:
            for name,p in model.named_parameters():p.requires_grad=not any(x in name for x in ('visual','vision','mtp'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.register_forward_hook(core.fp32_loss_outputs)
        params=[p for p in model.parameters() if p.requires_grad]
        write(run/'resources.json',resource_gate(params,cfg['track']))
        write(run/'modules.json',dict(trainable_parameters=sum(p.numel() for p in params),
            names=[name for name,p in model.named_parameters() if p.requires_grad],gpu=torch.cuda.get_device_name(0)))
        if cfg['method']=='PDU':
            model.eval()
            total=0.;count=0
            with torch.no_grad():
                for row in retain:
                    b=encode(tok,row,model.device);n=target_count(b)
                    total+=float(model(**b).loss)*n;count+=n
            initial=total/count
            if cfg['retain_loss_eps']<=initial:raise ValueError('PDU epsilon must exceed calibrated initial retain NLL')
            write(run/'calibration.json',dict(initial_retain_nll=initial,epsilon=cfg['retain_loss_eps']))
        alternates={}
        if args.method.startswith('DPO'):
            alternates={r['id']:alternative(r,forget+retain,args.method,alt_rng,cfg['rejection']) for r in forget+retain}
        write(run/'data_plan.json',dict(forget_ids=[r['id'] for r in forget],retain_ids=[r['id'] for r in retain],alternates=alternates))
        bank=None
        if args.method in ('NPO','DPOmis','DPOrej','RMU','UNDIAL'):
            bank=DiskReference(run/'fixed_reference')
            hidden=core.rl.decoder_layer(model,7) if args.method=='RMU' else None
            bank.capture(model,(encode(tok,r,model.device) for r in forget+retain+list(alternates.values())),hidden)
        bound=core.rl.profile(args.method,'matched_lora' if cfg['track']=='lora' else 'native')
        bound.update(cfg)
        trainer=None if args.method=='KLUE' else core.rl.bind_trainer(bound,model,bank)
        mlps=core.rl.gated_mlps(model) if args.method=='KLUE' else None
        if cfg['track']=='native_neurons':optimizer=core.rl.NativeNeuronAdam(mlps,cfg['learning_rate'])
        elif cfg['track']=='lora':optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        else:optimizer=core.CPUAdamW(params,cfg['learning_rate'],cfg['weight_decay'])
        updates=0;skipped=[];stop=None;ua_history=[]
        with (run/'diagnostics.jsonl').open('x',encoding='utf-8') as log:
            for epoch in range(cfg['epochs']):
                active=remembered(model,tok,forget) if args.method=='KLUE' else list(forget)
                ua=len(active)/len(forget) if args.method=='KLUE' else None
                if cfg['stop_ua'] is not None:
                    if ua is None:ua=len(remembered(model,tok,forget))/len(forget)
                    ua_history.append(dict(epoch=epoch,ua=ua))
                    if ua<=cfg['stop_ua'] and not args.smoke:stop='training_ua_threshold';break
                rng.shuffle(active);model.train()
                for begin in range(0,len(active),cfg['batch_size']):
                    if time.time()-start>args.hours*3600:raise TimeoutError('Explicit run time budget')
                    current=active[begin:begin+cfg['batch_size']]
                    if args.method=='CEU':
                        valid=[]
                        for r in current:
                            if target_count(encode(tok,r,model.device),cfg['ignore_first_n_answer_tokens'])<=0:skipped.append(r['id'])
                            else:valid.append(r)
                        current=valid
                    if not current:continue
                    keeps=[rng.choice(retain) for _ in current]
                    fs=[encode(tok,r,model.device) for r in current];rs=[encode(tok,r,model.device) for r in keeps]
                    weights=micro_weights(args.method,fs,rs)
                    model.zero_grad(set_to_none=True);selected=None
                    if args.method=='KLUE':
                        # FP32 activation/backprop avoids BF16 attribution underflow when feasible.
                        model.float()
                        try:selected,attr=core.klue_selection(model,tok,current,forget+retain,mlps,cfg,alt_rng)
                        finally:model.to(torch.bfloat16)
                    total=0.;retain_nll=0.
                    dual_on=args.method=='PDU' and epoch>=cfg['dual_warmup_epochs']
                    for j,(f,r,(wf,wr)) in enumerate(zip(fs,rs,weights)):
                        if args.method.startswith('DPO'):
                            loss=faithun_dpo_loss(model,bank,f,encode(tok,alternates[current[j]['id']],model.device),
                                r,encode(tok,alternates[keeps[j]['id']],model.device),dict(cfg,gamma=cfg['gamma']*wf,alpha=cfg['alpha']*wr))
                        elif args.method=='KLUE':loss=-cfg['gamma']*wf*model(**f).loss+cfg['alpha']*wr*model(**r).loss
                        elif args.method=='RMU':loss=rmu_loss(trainer,model,bank,f,r,cfg['gamma']*wf,cfg['alpha']*wr)
                        else:
                            trainer.gamma=cfg['gamma']*wf;trainer.alpha=cfg['alpha']*wr
                            if args.method=='PDU':
                                previous=list(trainer.preferences);trainer.preferences=[previous[0]*wf,previous[1]*wr];trainer.can_update=False
                            if args.method=='GA':loss=-wf*model(**f).loss
                            else:
                                loss=trainer.compute_loss(model,dict(forget=f,retain=r))
                                if args.method=='CEU':loss=loss*wf
                            if args.method=='PDU':
                                trainer.preferences=previous
                                with torch.no_grad():retain_nll+=wr*float(model(**r).loss)
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
                        loss.backward();total+=float(loss.detach())
                    if not all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):raise RuntimeError('Nonfinite gradient')
                    if selected is not None:
                        for key,mlp in mlps.items():
                            mask=torch.zeros(mlp.up_proj.out_features,device=model.device);mask[selected[key].to(model.device)]=1
                            mlp.gate_proj.weight.grad.mul_(mask[:,None]);mlp.up_proj.weight.grad.mul_(mask[:,None]);mlp.down_proj.weight.grad.mul_(mask[None,:])
                    torch.nn.utils.clip_grad_norm_(params,cfg['max_grad_norm'],error_if_nonfinite=True)
                    effective_lr=learning_rate(cfg,updates,math.ceil(len(forget)/cfg['batch_size']))
                    if selected is not None:optimizer.lr=effective_lr
                    else:
                        inner=optimizer.optimizer if hasattr(optimizer,'optimizer') else optimizer
                        for group in inner.param_groups:group['lr']=effective_lr
                    optimizer.step(selected) if selected is not None else optimizer.step()
                    if dual_on:trainer.preferences[1]=max(0.,trainer.preferences[1]+cfg['dual_step_size']*(retain_nll-cfg['retain_loss_eps']))
                    updates+=1;row=dict(update=updates,epoch=epoch,loss=total,ids=[r['id'] for r in current],dual_enabled=dual_on,learning_rate=effective_lr)
                    if args.method=='PDU':row['preferences']=list(trainer.preferences)
                    if selected is not None:row.update(attribution=attr,neuron_indices={k:v.tolist() for k,v in selected.items()})
                    log.write(json.dumps(row)+'\n');log.flush();write(run/'status.json',dict(status='running',phase='training',**row))
                    if args.smoke and updates>=2:stop='smoke_two_updates';break
                if stop:break
        if updates==0 and stop!='training_ua_threshold':raise RuntimeError('No optimizer updates')
        final_ua=len(remembered(model,tok,forget))/len(forget) if cfg['stop_ua'] is not None else None
        write(run/'training_summary.json',dict(updates=updates,skipped=skipped,stop=stop or 'epoch_budget',
            ua_history=ua_history,final_training_ua=final_ua,
            qualified=final_ua is not None and final_ua<=cfg['stop_ua'] and updates>0))
        if not args.smoke:model.save_pretrained(run/'checkpoint',safe_serialization=True)
        model.zero_grad(set_to_none=True);del optimizer,bank,trainer;gc.collect();torch.cuda.empty_cache()
        if args.smoke:write(run/'status.json',dict(status='smoke_passed',updates=updates));return
        if args.train_only:
            write(run/'status.json',dict(status='training_completed',updates=updates,seconds=time.time()-start));return
        metrics=evaluate(model,tok,data,run/'evaluation',generation=args.generation,capabilities=not args.behavior_only,
                         mmlu_limit=None if args.mmlu_full else 10)
        write(run/'status.json',dict(status='completed' if metrics['evaluation_complete'] else 'evaluation_incomplete',updates=updates,seconds=time.time()-start))
    except BaseException:
        write(run/'status.json',dict(status='failed',traceback=traceback.format_exc(),seconds=time.time()-start));raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=METHODS,required=True);parser.add_argument('--model')
    parser.add_argument('--run-dir');parser.add_argument('--config');parser.add_argument('--learning-rate',type=float)
    parser.add_argument('--retain-eps',type=float);parser.add_argument('--hours',type=float,default=6)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--behavior-only',action='store_true')
    parser.add_argument('--generation',action='store_true');parser.add_argument('--plan',action='store_true')
    parser.add_argument('--train-only',action='store_true')
    parser.add_argument('--mmlu-full',action='store_true')
    parser.add_argument('--lora-adapted',action='store_true')
    args=parser.parse_args()
    if args.plan:
        cfg=recipe(args.method)
        if args.lora_adapted:cfg=lora_adapt(cfg)
        print(json.dumps(dict(recipe=cfg,search=search_plan(args.method)),indent=2,ensure_ascii=False));return
    if not args.model or not args.run_dir:parser.error('Training/evaluation requires --model and --run-dir')
    worker(args)


if __name__=='__main__':main()
