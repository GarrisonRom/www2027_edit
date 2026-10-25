"""Local F0-only baseline queue; cloud entrypoint and old runs remain untouched."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import reproduce_all_baselines as runner

WS = Path(__file__).resolve().parents[2]
METHODS = ('GA', 'KLUE', 'GradDiff', 'NPO', 'SimNPO', 'RMU', 'DPOmis', 'DPOrej',
           'CEU', 'PDU', 'WGA', 'SatImp', 'UNDIAL')


def evaluate_behavior(model, tok, inputs, run, smoke, retain_is_heldout=True):
    # Smoke tests finite training updates, not final performance.
    if smoke:
        return {'smoke': True, 'performance_evaluation': 'not_run',
                'mmlu': None, 'ppl': None}
    model.eval()
    behavior = runner.base.evaluate(model, tok, inputs)
    behavior['full_hop2'] = runner.routing.full_hop2_audit(model, tok, inputs)
    runner.write(run / 'behavior.json', behavior)
    groups = behavior['groups']
    return {'all89': behavior['full_hop2']['all_hop2']['summary']['path_macro_accuracy'],
            'f0': groups['F0']['summary']['mcqa_accuracy'],
            'f1': groups['F1']['summary']['mcqa_accuracy'],
            'retain': groups['retain_heldout']['summary']['mcqa_accuracy'],
            'retain_is_heldout': retain_is_heldout, 'smoke': False,
            'mmlu': None, 'ppl': None,
            'capability_status': 'not_run_in_local_behavior_protocol'}


def verify_sources(expected):
    for relative, sha in expected.items():
        if runner.rl.digest(WS / relative) != sha:
            raise RuntimeError('Frozen source changed: ' + relative)


def queue(args):
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = runner.source_manifest()
    source.update({str(p.relative_to(WS)): runner.rl.digest(p)
                   for p in (WS / 'exp/damage').rglob('*.py')})
    for relative in source:
        target = root / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WS / relative, target)
    with (root / 'patch.diff').open('wb') as out:
        subprocess.run(['git', 'diff', '--binary'], cwd=WS, stdout=out, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WS, text=True).strip()
    (root / 'environment.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    runner.write(root / 'manifest.json', {'created': runner.base.protocol.timestamp(),
        'commit': commit, 'source': source, 'inputs': runner.base.protocol.input_manifest(),
        'model': runner.model_manifest(args.model), 'command': sys.argv, 'pid': os.getpid(),
        'seed': 42, 'queue_hours': 24, 'job_hours': 6,
        'methods': METHODS, 'evaluation': 'behavior only; MMLU/PPL pending',
        'native_KLUE_retain': 'all66; last33 NOT heldout',
        'matched_budget': 'F0 only, 114 updates, LoRA r8 alpha16, lr5e-5'})
    (root / 'run_log.md').write_text(
        '# Local baseline comparison\n\nFrozen config/source/input hashes precede jobs. '
        'Each formal run requires its own two-update smoke. Failures are retained. '
        'KLUE is a paper reimplementation, not official author code. '
        'Native KLUE and matched LoRA are separate tracks; no equal-budget claim.\n', encoding='utf-8')
    jobs = []
    start = time.time()
    for method in METHODS:
        track = 'native' if method == 'KLUE' else 'matched_lora'
        for smoke in (True, False):
            if time.time() - start >= 24 * 3600:
                runner.write(root / 'queue_status.json', {'status': 'budget_exhausted', 'jobs': jobs})
                return
            verify_sources(source)
            name = method + ('_smoke' if smoke else '_formal')
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                       '--method', method, '--track', track, '--model', args.model,
                       '--run-dir', str(root / name)]
            if smoke:
                command.append('--smoke')
            row = {'name': name, 'command': command, 'status': 'running'}
            jobs.append(row)
            runner.write(root / 'queue_status.json', {'status': 'running', 'jobs': jobs})
            print('START', name, flush=True)
            with (
                (root / (name + '.stdout.log')).open('x', encoding='utf-8') as out,
                (root / (name + '.stderr.log')).open('x', encoding='utf-8') as err,
            ):
                try:
                    remaining = max(1, 24 * 3600 - (time.time() - start))
                    code = subprocess.run(command, cwd=WS, stdout=out, stderr=err,
                                          timeout=min(6 * 3600, remaining)).returncode
                except subprocess.TimeoutExpired:
                    code = -1
            row.update(status='completed' if code == 0 else 'failed', returncode=code)
            runner.write(root / 'queue_status.json', {'status': 'running', 'jobs': jobs})
            if code != 0:
                break
    runner.write(root / 'queue_status.json', {'status': 'finished', 'jobs': jobs})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--track', choices=('matched_lora', 'native'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        runner.evaluate_all = evaluate_behavior
        runner.worker(args)
    else:
        queue(args)


if __name__ == '__main__':
    main()
