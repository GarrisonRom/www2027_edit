"""Continue only missing exp010 variants using a hash-verified frozen snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

WS = Path(__file__).resolve().parents[2]
OLD = WS / 'exp/runs/exp010_gsu_step_routing/runs/batch_20261023_092800_Cgroup_restart1'

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    completed = []
    write(root/'status.json', {'status':'preflight', 'parent':str(OLD)})
    try:
        manifest = json.loads((OLD/'batch_manifest.json').read_text(encoding='utf-8'))
        for gate in ('c0_replay_gate.json','c0_metric_gate.json'):
            if not json.loads((OLD/gate).read_text(encoding='utf-8'))['passed']:
                raise RuntimeError('Historical C0 gate not passed')
        used = 0.
        for label in ('smoke_C0','C0','Cg','C1','C2','C3'):
            state=json.loads((OLD/label/'status.json').read_text(encoding='utf-8'))
            if state['status'] != 'completed':
                raise RuntimeError('Historical completed job is not complete: '+label)
            used += state['seconds']
        used += json.loads((OLD/'C3R/status.json').read_text(encoding='utf-8'))['elapsed_seconds']
        budget = manifest['contract']['batch_budget_seconds'] - used
        if budget <= 0:
            raise RuntimeError('Original 16-hour compute budget exhausted')
        snapshot=root/'snapshot'
        for folder in ('train','damage'):
            shutil.copytree(WS/'exp'/folder, snapshot/'exp'/folder,
                            ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
        for filename, expected in manifest['source_sha256'].items():
            source=Path(filename)
            raw=source.read_bytes()
            if sha(raw)!=expected and source.name=='run_cloud_gsu_c2.py':
                text=raw.decode('utf-8')
                text=text.replace('                    if getattr(args, "variant", "C2") == "C1":\n                        bases = {parameter: None for parameter in params}\n                        basis_meta = {"variant": "C1", "span_route": False}\n                    else:\n                        bases,basis_meta=refresh_basis(model,tokenizer,inputs,params,names)\n',
                    '                    bases,basis_meta=refresh_basis(model,tokenizer,inputs,params,names)\n')
                text=text.replace("    parser.add_argument('--variant', choices=('C1', 'C2'), default='C2')\n",'')
                raw=text.encode('utf-8')
            if sha(raw)!=expected:
                raise RuntimeError('Cannot reconstruct exact frozen source: '+str(source))
            target=snapshot/source.relative_to(WS)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
        for rel, expected in manifest['input_manifest'].items():
            source=WS/rel
            if sha(source.read_bytes())!=expected:
                raise RuntimeError('Input changed: '+rel)
            target=snapshot/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
        for filename, expected in manifest['legacy_B0_reference_sha256'].items():
            if sha(Path(filename).read_bytes())!=expected:
                raise RuntimeError('Historical B0 reference changed')
        attribution=snapshot/'exp/results/attribution'
        attribution.mkdir(parents=True,exist_ok=True)
        for name in ('ffn_G1.json','ffn_G2.json','stats_stage2.json','ablation_G2.jsonl'):
            source=WS/'exp/results/attribution'/name
            shutil.copy2(source,attribution/name)
        env=subprocess.check_output([sys.executable,'-m','pip','freeze'])
        oldenv=(OLD/'environment.txt').read_bytes().splitlines()
        if sorted(env.splitlines())!=sorted(oldenv):
            raise RuntimeError('Python package freeze differs from parent batch')
        (root/'environment.txt').write_bytes(env)
        (root/'patch.diff').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=WS))
        write(root/'config.json',{'parent':str(OLD),'variants':['C3R','C3D','C4'],
            'restart_from_initialization':True,'optimizer_checkpoint_available':False,
            'original_budget_seconds':57600,'accounted_previous_seconds':used,
            'remaining_budget_seconds':budget,'frozen_manifest':manifest,
            'snapshot_files':{str(p.relative_to(snapshot)):sha(p.read_bytes()) for p in snapshot.rglob('*') if p.is_file()},
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=WS).decode().strip()})
        (root/'run_log.md').write_text('# exp010 continuation\n\nSame frozen algorithm/data/environment as parent; new run IDs. C3R restarts from initialization, not from update36. No extra seed or tuning. See config.json for hashes and budget. Parent results are read-only.\n',encoding='utf-8')
        for variant in ('C3R','C3D','C4'):
            remaining=budget-(time.time()-started)
            if remaining<=0:
                raise TimeoutError('Original remaining budget exhausted')
            command=[sys.executable,'-B',str(snapshot/'exp/train/run_local_gsu_c_group.py'),
                '--model',manifest['model'],'--variant',variant,'--run_dir',str(root/variant)]
            write(root/'status.json',{'status':'running','current':variant,'completed':completed,
                'command':command,'parent':str(OLD)})
            with (root/(variant+'.stdout.log')).open('x') as out,(root/(variant+'.stderr.log')).open('x') as err:
                subprocess.run(command,cwd=snapshot,stdout=out,stderr=err,check=True,timeout=remaining)
            completed.append(variant)
        write(root/'status.json',{'status':'completed','completed':completed,'seconds':time.time()-started})
    except BaseException:
        write(root/'status.json',{'status':'failed','completed':completed,'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':
    main()
