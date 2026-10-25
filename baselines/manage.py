"""Generate reviewable configs/search plans and select using training UA only."""
import argparse
import json
from pathlib import Path
from .recipes import METHODS,LORA_METHODS,recipe,search_plan,lora_adapt


def prepare(output,refresh=False,lora=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=refresh)
    for method in LORA_METHODS if lora else METHODS:
        c=lora_adapt(recipe(method)) if lora else recipe(method)
        (output/(method+'.json')).write_text(json.dumps(c,indent=2,ensure_ascii=False),encoding='utf-8')
        (output/(method+'.search.json')).write_text(json.dumps(search_plan(method),indent=2),encoding='utf-8')


def select(root):
    candidates=[]
    for summary in Path(root).rglob('training_summary.json'):
        run=summary.parent
        if not (run/'status.json').exists():continue
        status=json.loads((run/'status.json').read_text(encoding='utf-8'))['status']
        if status not in ('training_completed','completed','evaluation_incomplete'):continue
        s=json.loads(summary.read_text(encoding='utf-8'))
        c=json.loads((run/'config.json').read_text(encoding='utf-8'))
        if s.get('qualified') and not c.get('smoke'):
            candidates.append(dict(method=c['method'],model=c['model'],learning_rate=c['learning_rate'],
                beta=c['beta'],alpha=c['alpha'],seed=c['seed'],ua=s['final_training_ua'],run=str(run),
                track=c.get('track','unknown'),lora_adapted=bool(c.get('lora_adapted'))))
    groups={}
    for row in sorted(candidates,key=lambda r:(r['learning_rate'],r['beta'],r['alpha'])):
        key=(row['method'],row['model'],row['seed'],row['track'],row['lora_adapted'])
        if key not in groups:groups[key]=row
    return dict(selection='lowest qualified LR among completed trials; provisional until registered grid complete',
        uses_final_test_metrics=False,tie_rule='lower beta then lower alpha; preregistered convention',
        candidates=candidates,selected=list(groups.values()))


def lock_references():
    from .bootstrap import ROOT,verify,sha
    lock={name.replace(chr(92),'/'):h for name,h in verify().items()}
    lock.update({f.relative_to(ROOT).as_posix():sha(f) for f in (ROOT/'references').rglob('*') if f.is_file()})
    (ROOT/'source_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','select','lock-references']);p.add_argument('path',nargs='?')
    p.add_argument('--refresh',action='store_true')
    p.add_argument('--lora-adapted',action='store_true')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.path,a.refresh,a.lora_adapted)
    elif a.action=='lock-references':lock_references()
    else:print(json.dumps(select(a.path),ensure_ascii=False,indent=2))
