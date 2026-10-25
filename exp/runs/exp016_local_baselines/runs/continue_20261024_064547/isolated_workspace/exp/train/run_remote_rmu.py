"""Isolated RMU-LoRA representation-matching follow-up to exp008."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
import run_remote_baselines as baseline

ORIGINAL_LOSS = baseline.losses.compute_loss


def attach_rmu(model):
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear)
               and re.search(r'\.layers\.(5|6|7)\.mlp\.down_proj$', name)
               and not any(key in name for key in ('visual', 'vision', 'mtp'))]
    if len(targets) != 3:
        raise RuntimeError(f'Expected exactly three RMU target matrices: {targets}')
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
                                           bias='none', target_modules=targets))
    model.config.use_cache = False
    return model, sorted(targets)


def activation(model, tokenizer, q, a):
    ids, context = baseline.losses.te.enc_qa(tokenizer, q, a)
    tensor = torch.tensor([ids], device=model.device)
    modules = [module for name, module in model.named_modules()
               if re.search(r'\.layers\.7$', name)
               and not any(key in name for key in ('visual', 'vision', 'mtp'))]
    if len(modules) != 1:
        raise RuntimeError('RMU layer7 hook must be unique')
    cache = []
    def hook(module, args, output):
        cache.append(output[0] if isinstance(output, tuple) else output)
    handle = modules[0].register_forward_hook(hook)
    try:
        model(input_ids=tensor)
    finally:
        handle.remove()
    mask = torch.zeros_like(tensor, dtype=torch.bool)
    mask[:, context:] = True
    return cache[0], mask


def masked_mse(current, target, mask):
    if not mask.any():
        raise RuntimeError('Empty RMU answer mask')
    return ((current.float() - target.float()).square().mean(-1) * mask).sum() / mask.sum()


def rmu_loss(method, model, tok, q, a, rq, ra, state):
    if method != 'RMU':
        return ORIGINAL_LOSS(method, model, tok, q, a, rq, ra, state)
    forget, mask = activation(model, tok, q, a)
    if 'ctrl' not in state:
        generator = torch.Generator().manual_seed(42)
        direction = torch.rand(1, 1, forget.shape[-1], generator=generator)
        state['ctrl'] = (20. * direction / direction.norm()).to(forget.device)
    forget_loss = masked_mse(forget, state['ctrl'], mask)
    retain, retain_mask = activation(model, tok, rq, ra)
    with model.disable_adapter(), torch.no_grad():
        reference, _ = activation(model, tok, rq, ra)
    return forget_loss + masked_mse(retain, reference, retain_mask)


def configure():
    baseline.attach_lora = attach_rmu
    baseline.losses.compute_loss = rmu_loss
    baseline.METHODS = ('Original', 'RMU')


def queue(args):
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=False)
    baseline.write(root / 'queue_config.json', vars(args))
    sources = {str(path): baseline.protocol.sha256_file(path)
               for folder in ('train', 'damage')
               for path in (Path(__file__).parent.parent / folder).rglob('*.py')}
    baseline.write(root / 'batch_manifest.json', {
        'created': baseline.protocol.timestamp(), 'source': sources,
        'inputs': baseline.protocol.input_manifest(), 'torch': torch.__version__,
        'model_manifest': json.loads(Path('/root/autodl-tmp/models/download_manifest.json').read_text())})
    (root / 'environment.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    baseline.write(root / 'status.json', {'status': 'waiting_for_dependency',
                                        'dependency': args.after})
    dependency = Path(args.after)
    while True:
        status = json.loads(dependency.read_text())
        if status['status'] == 'completed':
            break
        if status['status'] in ('failed', 'interrupted'):
            baseline.write(root / 'status.json', {'status': 'dependency_failed',
                                                  'dependency_status': status})
            return
        time.sleep(30)
    completed = []
    jobs = [(name, path, smoke) for smoke in (True, False) for name, path in (
        ('qwen35_4b', '/root/autodl-tmp/models/Qwen3.5-4B'),
        ('llama32_3b', '/root/autodl-tmp/models/Llama-3.2-3B-Instruct'))]
    for name, model, smoke in jobs:
        for source, digest in sources.items():
            if baseline.protocol.sha256_file(Path(source)) != digest:
                raise RuntimeError(f'Frozen source changed: {source}')
        label = name + ('_smoke' if smoke else '_RMU_F0')
        command = [sys.executable, str(Path(__file__).resolve()), '--model', model,
                   '--run_dir', str(root / label)] + (['--smoke'] if smoke else [])
        baseline.write(root / 'status.json', {'status': 'running', 'current': label,
                                              'completed': completed, 'command': command})
        with (root / (label + '.stdout.log')).open('x') as out, \
             (root / (label + '.stderr.log')).open('x') as err:
            code = subprocess.run(command, stdout=out, stderr=err).returncode
        if code:
            baseline.write(root / 'status.json', {'status': 'failed', 'current': label,
                                                  'exit_code': code, 'completed': completed})
            return
        completed.append(label)
    baseline.write(root / 'status.json', {'status': 'completed', 'completed': completed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--after')
    parser.add_argument('--model')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    configure()
    if args.after:
        queue(args)
    else:
        args.method, args.supervision, args.seed = 'RMU', 'F0', 42
        baseline.worker(args)
