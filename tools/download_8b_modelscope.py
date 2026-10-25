"""Sequential pinned ModelScope download; no deletion and no hidden cache duplicate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
from modelscope.hub.api import HubApi
from modelscope.hub.file_download import model_file_download

MODELS=[('Qwen/Qwen3-8B','/root/models8b/Qwen3-8B'),
        ('LLM-Research/Meta-Llama-3.1-8B-Instruct','/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct')]


def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def wanted(path):
    return '/' not in path and (path.endswith(('.safetensors','.json','.model','.txt','.tiktoken'))
                               or path in ('README.md','LICENSE','LICENSE.txt','USE_POLICY.md'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run_dir',required=True);a=p.parse_args()
    root=Path(a.run_dir);root.mkdir(parents=True,exist_ok=False)
    done=[]
    try:
        api=HubApi()
        for repo,destination in MODELS:
            detail=api.get_valid_revision_detail(repo,revision='master')
            revision=(detail.get('Revision') or detail.get('Name') or 'master') if isinstance(detail,dict) else str(detail)
            allfiles=api.get_model_files(repo,revision=revision,recursive=True)
            files=[row for row in allfiles if wanted(row['Path'])]
            weights=[row for row in files if row['Path'].endswith('.safetensors')]
            if not weights:raise RuntimeError('No safetensors files returned for '+repo)
            dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
            missing=sum(int(row['Size']) for row in files if not (dest/row['Path']).exists())
            if shutil.disk_usage(dest).free<missing+4*1024**3:
                raise RuntimeError('Disk gate: missing weights plus4GiB reserve: '+str(dest))
            write(root/(dest.name+'_plan.json'),{'repo':repo,'revision':revision,'files':files,'path':str(dest)})
            for row in files:
                write(root/'status.json',{'status':'downloading','repo':repo,'file':row['Path'],'completed':done})
                model_file_download(repo,row['Path'],revision=revision,local_dir=str(dest))
                path=dest/row['Path']
                if path.stat().st_size!=int(row['Size']):raise RuntimeError('File size mismatch: '+str(path))
            collected=[]
            for row in files:
                path=dest/row['Path'];sha=digest(path)
                expected=row.get('Sha256') or row.get('SHA256')
                if expected and sha.lower()!=expected.lower():raise RuntimeError('Source hash mismatch')
                collected.append({'path':row['Path'],'bytes':path.stat().st_size,'sha256':sha})
            write(dest/'download_manifest_8b.json',{'repo':repo,'revision':revision,'files':collected,
                   'source_snapshot':allfiles,'completed_unix':time.time()})
            done.append({'repo':repo,'path':str(dest)})
        write(root/'status.json',{'status':'completed','models':done})
    except BaseException:
        write(root/'status.json',{'status':'failed','completed':done,'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()

