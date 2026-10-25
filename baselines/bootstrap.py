"""Provision immutable dependencies/data copies; never download model weights."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
WS=ROOT.parent
OLD=WS/'exp/runs/exp016_local_baselines/runs/local_20261023_233138'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def relative_path(root,value):
    return root.joinpath(*str(value).replace(chr(92),'/').split('/'))


def provision():
    target=ROOT/'vendor/runtime'
    if target.exists():
        verify();return target
    prior=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    files={**prior['source'],**prior['inputs']}
    for relative,expected in files.items():
        source=relative_path(OLD/'source_snapshot',relative) if relative in prior['source'] else relative_path(WS,relative)
        if sha(source)!=expected:raise RuntimeError('Source/input hash mismatch: '+relative)
    shutil.copytree(OLD/'source_snapshot',target)
    for relative in prior['inputs']:
        dst=relative_path(target,relative);dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(relative_path(WS,relative),dst)
    for source in (WS/'code/open-unlearning').glob('LICENSE*'):
        shutil.copy2(source,target/'code/open-unlearning'/source.name)
    refs=ROOT/'references';refs.mkdir(parents=True,exist_ok=True)
    for source in (WS/'code/open-unlearning/community/methods').rglob('*'):
        if source.is_file():
            dst=refs/'open-unlearning-community'/source.relative_to(WS/'code/open-unlearning/community/methods')
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
    for filename in ('faithun_2502_19207v2_audit.html','faithun_mcqa_template.svg'):
        shutil.copy2(WS/'tmp'/filename,refs/filename)
    lock={p.relative_to(ROOT).as_posix():sha(p) for base in (target,refs) for p in base.rglob('*') if p.is_file()}
    (ROOT/'source_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    return target


def verify():
    lock=json.loads((ROOT/'source_lock.json').read_text(encoding='utf-8'))
    bad=[name for name,expected in lock.items() if not relative_path(ROOT,name).is_file() or sha(relative_path(ROOT,name))!=expected]
    if bad:raise RuntimeError('Pinned baseline dependency changed: '+repr(bad))
    return lock


if __name__=='__main__':
    print(provision())
