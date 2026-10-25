"""Resume missing baselines using the previous batch's exact isolated sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

WS = Path(__file__).resolve().parents[2]
METHODS = ('DPOmis', 'DPOrej', 'CEU', 'PDU', 'WGA', 'SatImp', 'UNDIAL')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def collect_rows(roots):
    rows = []
    for root in roots:
        for folder in root.glob('*_formal'):
            status = folder / 'status.json'
            if not status.exists() or json.loads(status.read_text(encoding='utf-8'))['status'] != 'completed':
                continue
            m = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
            b = json.loads((folder / 'behavior.json').read_text(encoding='utf-8'))
            rows.append(dict(method=folder.name.removesuffix('_formal'),
                all89=m['all89'], covered75=round(75*b['sets']['F1_hop2']['summary']['mcqa_accuracy']),
                f0=round(33*m['f0']), f1=round(5*m['f1']), retain=round(33*m['retain']),
                heldout=m['retain_is_heldout'], source=str(folder / 'metrics.json')))
    croot = WS / 'exp/runs/exp010_gsu_step_routing/runs'
    for name in ('C0', 'Cg', 'C1', 'C2', 'C3', 'C3R', 'C3D', 'C4'):
        batch = 'local_continue_20261023_205325' if name in ('C3R','C3D','C4') else 'batch_20261023_092800_Cgroup_restart1'
        path = croot / batch / name / 'metrics.json'
        m = json.loads(path.read_text(encoding='utf-8'))
        rows.append(dict(method=name, all89=m['full_hop2']['all_hop2']['summary']['path_macro_accuracy'],
            covered75=round(75*m['sets']['F1_hop2']['summary']['mcqa_accuracy']),
            f0=round(33*m['groups']['F0']['summary']['mcqa_accuracy']),
            f1=round(5*m['groups']['F1']['summary']['mcqa_accuracy']),
            retain=round(33*m['groups']['retain_heldout']['summary']['mcqa_accuracy']),
            heldout=True, source=str(path)))
    return sorted(rows, key=lambda row: row['all89'])


def summarize(roots, output):
    rows = collect_rows(roots)
    write(output / 'comparison.json', rows)
    lines = ['# Local C-group and baseline comparison', '',
        'Qwen2.5-1.5B, seed42; provisional descriptive ranking, no significance claim.',
        'KLUE is a separate native paper reimplementation track; its retain33 were trained.',
        'C variants use F1 guidance; C0 omits the F0 guard. Data access and optimization differ.',
        'MMLU/PPL pending. Missing/failed methods have no performance row.', '',
        '| Method | All89 recall ↓ | Covered /75 ↓ | F0 /33 ↓ | F1 /5 ↓ | Retain /33 ↑ | Heldout |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['all89']:.2%} | {r['covered75']} | {r['f0']} | {r['f1']} | {r['retain']} | {r['heldout']} |")
    lines += ['', '## Raw sources', '']
    lines += [f"- {r['method']}: {r['source']}" for r in rows]
    (output / 'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    old = Path(args.previous).resolve()
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    prior = json.loads((old / 'manifest.json').read_text(encoding='utf-8'))
    isolated = root / 'isolated_workspace'
    shutil.copytree(old / 'source_snapshot', isolated)
    for relative, expected in prior['source'].items():
        if sha(isolated / relative) != expected:
            raise RuntimeError('Previous snapshot hash mismatch: ' + relative)
    for relative, expected in prior['inputs'].items():
        source = WS / relative
        if sha(source) != expected:
            raise RuntimeError('Input changed: ' + relative)
        target = isolated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(Path(__file__), root / 'continuation_source.py')
    (root / 'environment.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    with (root / 'patch.diff').open('wb') as f:
        subprocess.run(['git', 'diff', '--binary'], cwd=WS, stdout=f, check=True)
    write(root / 'manifest.json', dict(previous=str(old), source=prior['source'], inputs=prior['inputs'],
        command=sys.argv, methods=METHODS, model=args.model, seed=42, budget_hours=24,
        implementation='exact previous frozen sources; isolated input copies',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WS,text=True).strip()))
    (root / 'run_log.md').write_text('Continuation: exact old source hashes, independent smoke gates, F0-only114updates; capability metrics pending.', encoding='utf-8')
    jobs = []
    start = time.time()
    summarize([old, root], root)
    try:
        for method in METHODS:
            for smoke in (True, False):
                if time.time()-start >= 24*3600:
                    raise TimeoutError('Continuation24hour budget')
                for relative, expected in {**prior['source'], **prior['inputs']}.items():
                    if sha(isolated / relative) != expected:
                        raise RuntimeError('Isolated source/input changed: '+relative)
                name = method + ('_smoke' if smoke else '_formal')
                command = [sys.executable, '-B', str(isolated/'exp/train/run_local_baseline_queue.py'),
                    '--worker','--method',method,'--track','matched_lora',
                    '--model',args.model,'--run-dir',str(root/name)]
                if smoke:
                    command.append('--smoke')
                row = dict(name=name, status='running', command=command)
                jobs.append(row)
                write(root/'queue_status.json', dict(status='running',jobs=jobs))
                print('START',name,flush=True)
                with (root/(name+'.stdout.log')).open('x',encoding='utf-8') as out, (root/(name+'.stderr.log')).open('x',encoding='utf-8') as err:
                    try:
                        code = subprocess.run(command,cwd=isolated,stdout=out,stderr=err,
                            timeout=min(6*3600,max(1,24*3600-(time.time()-start)))).returncode
                    except subprocess.TimeoutExpired:
                        code = -1
                row.update(status='completed' if code==0 else 'failed',returncode=code)
                write(root/'queue_status.json',dict(status='running',jobs=jobs))
                summarize([old,root],root)
                if code:
                    break
        write(root/'queue_status.json',dict(status='finished',jobs=jobs))
    except BaseException as error:
        write(root/'queue_status.json',dict(status='interrupted',error=repr(error),jobs=jobs))
        raise


if __name__ == '__main__':
    main()
