"""8B loader and native checkpoint reference; resource changes never alter loss/data."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from safetensors import safe_open


def load_native(model_path):
    return AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,
        device_map='cpu',low_cpu_mem_usage=True,local_files_only=True)


class DiskBaseWeights:
    def __init__(self,model,model_path):
        path=Path(model_path)
        index=path/'model.safetensors.index.json'
        if index.exists():self.mapping=json.loads(index.read_text(encoding='utf-8'))['weight_map']
        else:
            with safe_open(path/'model.safetensors',framework='pt',device='cpu') as f:
                self.mapping={key:'model.safetensors' for key in f.keys()}
        self.root=path
        self.names=[name for name,p in model.named_parameters() if p.requires_grad]
    def items(self):
        for name in self.names:
            if name not in self.mapping:
                raise RuntimeError('No exact base tensor for native delta: '+name)
            with safe_open(self.root/self.mapping[name],framework='pt',device='cpu') as f:
                value=f.get_tensor(name)
            yield name,value

