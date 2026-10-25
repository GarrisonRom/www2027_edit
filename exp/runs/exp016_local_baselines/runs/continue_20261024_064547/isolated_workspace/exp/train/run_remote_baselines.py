"""Logged cross-architecture baseline screen; independent of frozen GSU runs."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

import diagnose_gsu_memory_guided as protocol
import train_baselines as losses

METHODS = ('Original', 'GA', 'GradDiff', 'NPO', 'SimNPO')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding='utf-8')


def load_model(path):
    config = json.loads((Path(path) / 'config.json').read_text())
    loader = (AutoModelForImageTextToText if config['model_type'] == 'qwen3_5'
              else AutoModelForCausalLM)
    return loader.from_pretrained(path, dtype=torch.bfloat16, device_map='cuda',
                                  local_files_only=True)


def target_modules(model):
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or '.layers.' not in name:
            continue
        if any(part in name for part in ('visual', 'vision', 'mtp')):
            continue
        if '.self_attn.' in name or '.linear_attn.' in name or '.mlp.' in name:
            targets.append(name)
    if not targets:
        raise RuntimeError('No language decoder linear modules found')
    return sorted(targets)


def attach_lora(model):
    targets = target_modules(model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias='none', target_modules=targets))
    model.config.use_cache = False
    return model, targets


def evaluate(model, tokenizer, inputs):
    groups = {'F0': inputs['F0'], 'F1': inputs['F1'],
              'retain_train_pool': inputs['retain_all'][:33],
              'retain_heldout': inputs['retain_all'][33:]}
    paraphrases = []
    for ci in inputs['indices']['forget']:
        fact = inputs['clusters'][ci][2]['hop1_test']
        for qi, question in enumerate(fact['questions']):
            paraphrases.append({'id': f'paraphrase:{ci}:{qi}', 'source': 'F0_paraphrases', 'question': question,
                                'answer': fact['answers'],
                                'false_options': fact['false_answer_options']})
    groups['F0_paraphrases'] = paraphrases
    return {'sets': protocol.collect_set_audit(model, tokenizer, inputs['sets'], inputs['qa']),
            'groups': protocol.collect_group_audits(model, tokenizer, groups)}


def worker(args):
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    write(run / 'config.json', vars(args))
    write(run / 'status.json', {'status': 'running', 'started': protocol.timestamp()})
    (run / 'run_log.md').write_text(
        '# Baseline run\n\nSource/environment/data/model provenance: ../batch_manifest.json.\n'
        'Protocol: snapshot exp/runs/exp008_remote_baselines/card.md.\n'
        'Per-item evidence: metrics.json.\n', encoding='utf-8')
    try:
        inputs = protocol.load_inputs()
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        print('LOADING_MODEL', args.model, flush=True)
        model = load_model(args.model)
        for parameter in model.parameters():
            parameter.requires_grad = False
        targets = []
        if args.method != 'Original':
            model, targets = attach_lora(model)
        write(run / 'modules.json', {'targets': targets,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
        if args.smoke:
            model.train()
            q, a = inputs['F0'][0]['question'], inputs['F0'][0]['answer']
            rq, ra = inputs['retain_all'][0]['question'], inputs['retain_all'][0]['answer']
            for method in METHODS[1:]:
                model.zero_grad(set_to_none=True)
                loss = losses.compute_loss(method, model, tokenizer, q, a, rq, ra, {})
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite smoke loss: {method}')
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):
                    raise RuntimeError(f'Invalid smoke gradients: {method}')
                print('SMOKE_PASSED', method, float(loss.detach()), flush=True)
            audit = protocol.collect_item_audit(model, tokenizer, inputs['F0'][:1])
            if len(audit) != 1:
                raise RuntimeError('Audit smoke check failed')
            model.save_pretrained(run / 'adapter_smoke')
            metrics = {'smoke': 'passed', 'methods': METHODS[1:]}
        else:
            if args.method != 'Original':
                items = inputs['F0'] + (inputs['F1'] if args.supervision == 'F0F1' else [])
                schedule = protocol.make_update_schedule(items, 114, args.seed)
                rng = random.Random(args.seed)
                retain_indices = [rng.randrange(33) for _ in schedule]
                write(run / 'schedule.json', {'forget': schedule, 'retain_indices': retain_indices})
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad], lr=5e-5,
                    weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
                model.train()
                with (run / 'diagnostics.jsonl').open('x', encoding='utf-8') as log:
                    for item, ri in zip(schedule, retain_indices):
                        retain = inputs['retain_all'][ri]
                        optimizer.zero_grad(set_to_none=True)
                        loss = losses.compute_loss(args.method, model, tokenizer,
                            item['question'], item['answer'], retain['question'], retain['answer'], {})
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite training loss')
                        loss.backward()
                        if not all(torch.isfinite(p.grad).all() for p in model.parameters()
                                   if p.grad is not None):
                            raise RuntimeError('Nonfinite gradient')
                        optimizer.step()
                        row = {'update': item['update'], 'loss': float(loss.detach()),
                               'seconds': round(time.time() - started, 2)}
                        log.write(json.dumps(row) + '\n')
                        log.flush()
                        if item['update'] % 10 == 0:
                            print(row, flush=True)
                model.save_pretrained(run / 'adapter')
            model.eval()
            print('EVALUATION_STARTED', args.method, flush=True)
            metrics = evaluate(model, tokenizer, inputs)
        write(run / 'metrics.json', metrics)
        write(run / 'status.json', {'status': 'completed', 'ended': protocol.timestamp(),
              'seconds': time.time() - started,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})
    except BaseException:
        write(run / 'status.json', {'status': 'failed', 'ended': protocol.timestamp(),
                                    'traceback': traceback.format_exc()})
        raise


def batch(args):
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=False)
    models = [('qwen35_4b', '/root/autodl-tmp/models/Qwen3.5-4B'),
              ('llama32_3b', '/root/autodl-tmp/models/Llama-3.2-3B-Instruct')]
    jobs = [(name, path, method, 'F0') for method in METHODS for name, path in models]
    jobs += [(name, path, 'GA', 'F0F1') for name, path in models]
    if args.smoke:
        jobs = [(name, path, 'GA', 'F0') for name, path in models]
    manifest = {'created': protocol.timestamp(), 'seed': args.seed, 'jobs': jobs,
        'command': sys.argv, 'inputs': protocol.input_manifest(),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'source': {},
        'model_manifest': json.loads(Path('/root/autodl-tmp/models/download_manifest.json').read_text())}
    for folder in ('train', 'damage'):
        for path in (Path(__file__).parent.parent / folder).rglob('*.py'):
            manifest['source'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(root / 'batch_manifest.json', manifest)
    (root / 'environment.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    completed = []
    summaries = []
    for name, model, method, supervision in jobs:
        for source, digest in manifest['source'].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Source changed during batch: {source}')
        run = root / f'{name}_{method}_{supervision}'
        command = [sys.executable, str(Path(__file__).resolve()), '--model', model,
                   '--method', method, '--supervision', supervision, '--run_dir', str(run),
                   '--seed', str(args.seed)] + (['--smoke'] if args.smoke else [])
        write(root / 'status.json', {'status': 'running', 'current': run.name,
                                    'completed': completed, 'command': command})
        with (root / f'{run.name}.stdout.log').open('x') as out, \
             (root / f'{run.name}.stderr.log').open('x') as err:
            result = subprocess.run(command, stdout=out, stderr=err)
        if result.returncode:
            write(root / 'status.json', {'status': 'failed', 'current': run.name,
                                         'completed': completed, 'exit_code': result.returncode})
            raise RuntimeError(f'Failed job: {run}')
        completed.append(run.name)
        if not args.smoke:
            metrics = json.loads((run / 'metrics.json').read_text())
            summaries.append({'run': run.name, 'model': name, 'method': method,
                'supervision': supervision,
                'F1_hop2': metrics['sets']['F1_hop2']['summary'],
                'groups': {key: value['summary'] for key, value in metrics['groups'].items()}})
            write(root / 'comparison.json', {'status': 'partial', 'rows': summaries})
    if not args.smoke:
        write(root / 'comparison.json', {'status': 'completed', 'rows': summaries,
              'interpretation': 'Single seed fixed-input transfer; no causal or significance claim'})
    write(root / 'status.json', {'status': 'completed', 'completed': completed,
                                 'ended': protocol.timestamp()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--method', choices=METHODS, default='GA')
    parser.add_argument('--supervision', choices=('F0', 'F0F1'), default='F0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_dir', required=True)
    arguments = parser.parse_args()
    if arguments.batch:
        batch(arguments)
    else:
        worker(arguments)
