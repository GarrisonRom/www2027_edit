"""Read-only training/data/metric audit against immutable exp016 sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

WS=Path(__file__).resolve().parents[2]
OLD=WS/'exp/runs/exp016_local_baselines/runs/local_20261023_233138'
CONT=WS/'exp/runs/exp016_local_baselines/runs/continue_20261024_064547'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    snapshot=OLD/'source_snapshot'
    sys.path.insert(0,str(snapshot/'exp/train'))
    import reproduce_all_baselines as runner
    # Point frozen loaders at an independently copied workspace with verified data.
    sys.path.pop(0)
    isolated=CONT/'isolated_workspace'
    proto=runner.base.protocol;te=runner.base.losses.te;ca=proto.ca
    te.DATA=isolated/'data/faithun/FaithUn_Dataset.json'
    te.SPLITS=isolated/'data/faithun/splits';proto.OUT=isolated/'exp/results'
    ca.OUT=proto.OUT;ca.ATT=proto.OUT/'attribution';ca.SPLITS=te.SPLITS
    ca.ep.DATA=te.DATA
    inputs=proto.load_inputs()
    tok=runner.AutoTokenizer.from_pretrained('E:/12models/modelscope/Qwen2.5-1.5B-Instruct',local_files_only=True)
    token_report={}
    for name,rows in [('F0',inputs['F0']),('F1',inputs['F1']),('retain',inputs['retain_all'])]:
        details=[]
        for row in rows:
            full,ctx=te.enc_qa(tok,row['question'],row['answer'])
            q=tok(row['question'],add_special_tokens=True)['input_ids']
            details.append(dict(id=row['id'],answer_tokens=len(full)-ctx,
                prefix_matches=full[:ctx]==q,ceu_fully_skipped=len(full)-ctx<=1))
        token_report[name]=dict(n=len(rows),prefix_mismatches=sum(not x['prefix_matches'] for x in details),
            ceu_fully_skipped=sum(x['ceu_fully_skipped'] for x in details),details=details)
    prior=read(OLD/'manifest.json')
    source_mismatch=[p for p,h in prior['source'].items() if sha(snapshot/p)!=h]
    input_mismatch=[p for p,h in prior['inputs'].items() if sha(isolated/p)!=h]
    runs=[]
    for root in (OLD,CONT):
        for folder in root.glob('*_formal'):
            if not (folder/'status.json').exists():continue
            status=read(folder/'status.json');cfg=read(folder/'config.json')
            logs=[json.loads(x) for x in (folder/'diagnostics.jsonl').read_text(encoding='utf-8').splitlines()] if (folder/'diagnostics.jsonl').exists() else []
            summary=read(folder/'training_summary.json') if (folder/'training_summary.json').exists() else None
            r=dict(run=str(folder),status=status,config=cfg,summary=summary,
                logged_updates=len(logs),consecutive_updates=[x['update'] for x in logs]==list(range(1,len(logs)+1)))
            if logs:r.update(first_loss=logs[0]['loss'],last_loss=logs[-1]['loss'],
                dual_active_steps=sum(bool(x.get('dual_enabled')) for x in logs))
            if (folder/'data_plan.json').exists():
                plan=read(folder/'data_plan.json')
                r['forget_retain_overlap']=sorted(set(plan['forget_ids'])&set(plan['retain_ids']))
                r['forget_count']=len(plan['forget_ids']);r['retain_count']=len(plan['retain_ids'])
            if (folder/'metrics.json').exists():
                m=read(folder/'metrics.json');b=read(folder/'behavior.json');r['metrics']=m
                items=b['full_hop2']['all_hop2']['items']
                r['all89_recomputed']=sum(x['mcqa_correct'] for x in items)/len(items)
                r['metric_consistent']=abs(r['all89_recomputed']-m['all89'])<1e-10
                r['choice_ties']=sum(len(set(x['mcqa_scores']))!=len(x['mcqa_scores']) for x in items)
            runs.append(r)
    result=dict(source_mismatch=source_mismatch,input_mismatch=input_mismatch,
        tokenization=token_report,runs=runs,frozen_runner=str(snapshot/'exp/train/reproduce_all_baselines.py'))
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'environment.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    print(json.dumps(dict(source_mismatch=source_mismatch,input_mismatch=input_mismatch,
        tokenization={k:{a:b for a,b in v.items() if a!='details'} for k,v in token_report.items()},
        runs=[dict(method=r['config']['method'],status=r['status']['status'],updates=r['logged_updates'],
                   consistent=r.get('metric_consistent'),dual_steps=r.get('dual_active_steps')) for r in runs]),indent=2))


if __name__=='__main__':main()
