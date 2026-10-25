"""Re-evaluate a selected checkpoint without retraining or changing its weights."""
import argparse
import json
from pathlib import Path
from .runtime import load
from .evaluate import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-model',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--generation',action='store_true')
    p.add_argument('--mmlu-full',action='store_true')
    a=p.parse_args();core=load();checkpoint=Path(a.checkpoint)
    tok=core.AutoTokenizer.from_pretrained(a.base_model,local_files_only=True)
    if (checkpoint/'adapter_config.json').exists():
        from peft import PeftModel
        model=PeftModel.from_pretrained(core.base.load_model(a.base_model),str(checkpoint))
    else:model=core.base.load_model(str(checkpoint))
    result=evaluate(model,tok,core.base.protocol.load_inputs(),Path(a.output),
                    limit=1 if a.smoke else None,generation=a.generation,capabilities=True,
                    mmlu_limit=None if a.mmlu_full else 10)
    core.write(Path(a.output)/'checkpoint_source.json',dict(base=a.base_model,checkpoint=str(checkpoint),
        files={str(f.relative_to(checkpoint)):core.rl.digest(f) for f in checkpoint.iterdir() if f.is_file()}))
    print(json.dumps(dict(complete=result['evaluation_complete'],scope=result['scope'])))


if __name__=='__main__':main()
