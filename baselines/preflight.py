"""Read model tensor shapes and dependency readiness without allocating full weights."""
import argparse
import json
import math
from pathlib import Path
import psutil
from safetensors import safe_open
from .bootstrap import verify
from .recipes import recipe,METHODS,lora_adapt,lora_targets


def check(method,model,data_check=False,lora=False):
    verify();c=lora_adapt(recipe(method)) if lora else recipe(method);n=0
    for file in Path(model).glob('*.safetensors'):
        with safe_open(str(file),framework='pt',device='cpu') as reader:
            for key in reader.keys():
                shape=reader.get_slice(key).get_shape();include=True
                if c['track']=='lora':
                    if '.layers.' in key and len(shape)==2 and any(x in key for x in ('_proj.weight',)):
                        if lora_targets([key.removesuffix('.weight')],c):n+=c['lora_r']*sum(shape)
                    continue
                if method=='RMU':include=any(f'.layers.{i}.mlp.down_proj.weight' in key for i in c['trainable_layers'])
                elif method=='KLUE':include='.mlp.' in key and any(f in key for f in ('gate_proj.weight','up_proj.weight','down_proj.weight'))
                if include:n+=math.prod(shape)
    if not n:raise ValueError('No supported model safetensors/parameters found')
    required=0 if method=='Original' else 16*n+(512*1024**2 if c['track']=='lora' else 2*1024**3)
    free=psutil.virtual_memory().available
    errors=[]
    try:
        from lm_eval.tasks import TaskManager
        mmlu_registered='mmlu' in TaskManager().all_tasks
    except Exception as e:errors.append(repr(e));mmlu_registered=False
    result=dict(method=method,track=c['track'],trainable_parameter_estimate=0 if method=='Original' else n,
        cpu_required_bytes=required,cpu_free_bytes=free,resource_ready=free>=required,
        mmlu_registered=mmlu_registered,dependency_errors=errors,
        datasets_ready='unverified until actual MMLU/WikiText downloads and scoring succeed',
        source_fidelity=c['fidelity'],pdu_calibration_required=method=='PDU')
    if data_check:
        from .runtime import load
        from .evaluate import records
        from .protocol import option_order,prompt,encode
        core=load();inputs=core.base.protocol.load_inputs();rows=records(inputs)
        tok=core.AutoTokenizer.from_pretrained(model,local_files_only=True)
        errors=[]
        for row in rows:
            try:
                option_order(row)
                for q in (row['question'],prompt(row)):
                    for answer in [row['answer']]+list(row['false_options']):
                        encode(tok,dict(row,question=q,answer=answer),'cpu')
            except Exception as e:errors.append(dict(id=row['id'],error=repr(e)))
        result['data_check']=dict(rows=len(rows),errors=errors,passed=not errors)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--model',required=True);p.add_argument('--output')
    p.add_argument('--data',action='store_true')
    p.add_argument('--lora-adapted',action='store_true')
    a=p.parse_args();result=check(a.method,a.model,a.data,a.lora_adapted)
    text=json.dumps(result,indent=2,ensure_ascii=False)
    if a.output:
        path=Path(a.output)
        if path.exists():raise FileExistsError(path)
        path.write_text(text,encoding='utf-8')
    print(text)
