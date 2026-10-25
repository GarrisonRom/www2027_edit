"""Checkpoint-only retention: dry-run plan or apply exact hash-checked file unlinks."""
import argparse,hashlib,json,os,stat,time
from pathlib import Path
from collections import defaultdict
BASE=Path('/root/autodl-tmp');SKIP_PREFIXES=('repro_unpack.','exp016_unpack.');EXTS=('.safetensors','.pt','.bin','.xor.zlib')
SNAPSHOT_PRIORITY={'www2027_sync_20261024_8b_v1':50,'www2027_sync_20261023_baselines_complete_v3':40,'www2027_sync_20261023_c1_v3_local_aligned':30,'www2027_sync_20261023_capability_eval_v3':20,'www2027_baselines_20261022_v2':15,'www2027_rmu_20261022':10}
STALE_RUNNING_SECONDS=3600

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def method_name(cfg,run):
 value=cfg.get('method') or cfg.get('variant') or ''
 if value:return str(value)
 for c in ('GSU-C2','GSU-C1','KLUE','DPOmis','DPOrej','GradDiff','SimNPO','UNDIAL','SatImp','RMU','NPO','GA','CEU','PDU','WGA','C1','C2','C3R','C3D','C3','C4','Cg','C0'):
  if c.lower() in run.name.lower():return c
 return 'unknown'

def inventory():
 rows=[]
 for top in sorted(BASE.iterdir()):
  if not top.is_dir() or not top.name.startswith(('www2027_','www2027_sync_')) or any(x in top.name for x in SKIP_PREFIXES):continue
  exp=top/'exp'/'runs'
  if not exp.exists():continue
  for directory,dirs,files in os.walk(exp):
   for filename in files:
    if not any(filename.endswith(ext) for ext in EXTS):continue
    path=Path(directory)/filename
    try:
     st=path.lstat()
     if not stat.S_ISREG(st.st_mode):continue
    except FileNotFoundError:continue
    run=None;cfg={};status={};cur=path.parent
    while cur!=exp.parent and cur.is_relative_to(exp):
     if (cur/'config.json').exists():
      run=cur
      try:cfg=json.loads((cur/'config.json').read_text(encoding='utf-8'))
      except Exception:cfg={}
      try:status=json.loads((cur/'status.json').read_text(encoding='utf-8'))
      except Exception:status={}
      break
     cur=cur.parent
    if run is None:continue
    model=Path(str(cfg.get('model') or cfg.get('model_path') or 'unknown')).name
    rows.append(dict(path=str(path),size=st.st_size,device=st.st_dev,inode=st.st_ino,nlink=st.st_nlink,model=model,
      method=method_name(cfg,run),seed=str(cfg.get('seed','42')),track=str(cfg.get('track') or cfg.get('supervision') or 'unspecified'),
      smoke=bool(cfg.get('smoke')) or 'smoke' in run.name.lower() or 'adapter_smoke' in str(path).lower(),
      status=status.get('status','unknown'),started=status.get('started',''),ended=status.get('ended',''),run=str(run),
      run_rel=str(run.relative_to(exp)),snapshot=top.name,rel_file=str(path.relative_to(run))))
 return rows

def process_has_path(needle):
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='ignore')
  except (OSError,PermissionError):continue
  if needle in cmd:return True
 return False

def build_plan(rows):
 groups=defaultdict(list)
 for row in rows:
  if row['status']=='completed' and not row['smoke'] and row['method']!='unknown' and row['model']!='unknown':groups[(row['model'],row['method'],row['seed'],row['track'])].append(row)
 winners={}
 for key,items in groups.items():winners[key]=max(items,key=lambda r:(r['ended'],r['run_rel']))['run_rel']
 keep=[];deletes=[];protected=[]
 for key,items in groups.items():
  run_rel=winners[key];winning=[r for r in items if r['run_rel']==run_rel];by_file=defaultdict(list)
  for row in winning:by_file[row['rel_file']].append(row)
  for rel_file,copies in by_file.items():
   chosen=max(copies,key=lambda r:(SNAPSHOT_PRIORITY.get(r['snapshot'],0),r['path']))
   keep.append(dict(key=key,run=chosen['run'],path=chosen['path'],size=chosen['size'],sha256=digest(chosen['path'])))
   for copy in copies:
    if copy['path']!=chosen['path']:deletes.append(dict(**copy,reason='duplicate_snapshot_link',keep_path=chosen['path']))
  for row in items:
   if row['run_rel']!=run_rel:deletes.append(dict(**row,reason='superseded_completed_run',keep_run=run_rel))
 for row in rows:
  if any(d['path']==row['path'] for d in deletes) or any(k['path']==row['path'] for k in keep):continue
  if row['status'] in ('failed','interrupted','preflight_failed','dependency_failed'):deletes.append(dict(**row,reason='failed_run_checkpoint'))
  elif row['smoke'] and row['status']=='completed':deletes.append(dict(**row,reason='smoke_checkpoint_no_final_model'))
  elif row['status']=='running':
   key=(row['model'],row['method'],row['seed'],row['track'])
   match=next((k for k in keep if tuple(k['key'])==key and k['run'].endswith('/'+row['run_rel'])),None)
   age=time.time()-Path(row['path']).stat().st_mtime;batch=Path(row['run']).parent/'status.json'
   try:batch_done=json.loads(batch.read_text(encoding='utf-8')).get('status')=='completed'
   except Exception:batch_done=False
   if match and age>STALE_RUNNING_SECONDS and not process_has_path(row['path']) and not process_has_path(row['run']):deletes.append(dict(**row,reason='stale_running_duplicate_of_completed_run',keep_path=match['path']))
   else:protected.append(dict(path=row['path'],reason='run_status_running_or_active',run=row['run']))
  elif row['path'] not in {k['path'] for k in keep}:protected.append(dict(path=row['path'],reason='unknown_or_unmatched_not_deleted',run=row['run'],status=row['status']))
 seen=set();dedup=[]
 for item in deletes:
  if item['path'] not in seen:
   seen.add(item['path']);item['sha256']=digest(item['path']);dedup.append(item)
 deletes=dedup;all_inode=defaultdict(list);del_inode=defaultdict(list)
 for row in rows:all_inode[(row['device'],row['inode'])].append(row)
 for row in deletes:del_inode[(row['device'],row['inode'])].append(row)
 freed=[]
 for inode,items in del_inode.items():
  nlink=max(r['nlink'] for r in all_inode[inode])
  if len(items)>=nlink:freed.append(dict(device=inode[0],inode=inode[1],size=items[0]['size'],links_removed=len(items),nlink=nlink))
 return dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),policy='one newest successful final checkpoint per model/method/seed/track; one canonical snapshot link; smoke, stale orphan, failed and superseded weights removed; base models and all records retained',checkpoint_paths=len(rows),retained_final_files=keep,delete_files=deletes,protected_files=protected,failed_checkpoint_count=sum(x['reason']=='failed_run_checkpoint' for x in deletes),expected_freed_bytes=sum(x['size'] for x in freed),freed_inode_plan=freed)

def atomic_write(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(tmp,path)

def apply(plan_file,manifest_file):
 p=json.loads(Path(plan_file).read_text(encoding='utf-8'));audit=dict(policy=p['policy'],created=p['created'],retained_final_files=p['retained_final_files'],planned_delete_count=len(p['delete_files']),results=[],expected_freed_bytes=p['expected_freed_bytes']);atomic_write(manifest_file,audit);allowed=BASE.resolve()
 for target in p['delete_files']:
  path=Path(target['path']);resolved=path.resolve(strict=True)
  if not resolved.is_relative_to(allowed) or '/exp/runs/' not in str(resolved):raise RuntimeError('Refusing checkpoint outside exp/runs: '+str(path))
  if path.is_symlink() or not path.is_file() or not any(path.name.endswith(ext) for ext in EXTS):raise RuntimeError('Refusing non-checkpoint/symlink target: '+str(path))
  if process_has_path(target['run']) or process_has_path(str(path)):raise RuntimeError('Refusing checkpoint used by active process: '+str(path))
  if path.stat().st_size!=target['size'] or digest(path)!=target['sha256']:raise RuntimeError('Checkpoint changed since audit: '+str(path))
  status=json.loads((Path(target['run'])/'status.json').read_text(encoding='utf-8'));reason=target['reason']
  if reason=='stale_running_duplicate_of_completed_run':
   if status.get('status')!='running' or not Path(target['keep_path']).is_file():raise RuntimeError('Stale duplicate verification failed')
  elif reason=='failed_run_checkpoint':
   if status.get('status') not in ('failed','interrupted','preflight_failed','dependency_failed'):raise RuntimeError('Failed status changed')
  elif status.get('status')!='completed':raise RuntimeError('Run is not completed; preserve checkpoint')
  rec=dict(path=str(path),reason=reason,size=target['size'],sha256=target['sha256'],run=target['run'],status_before=status.get('status'),removed=False);audit['results'].append(rec);atomic_write(manifest_file,audit);path.unlink();rec['removed']=True;atomic_write(manifest_file,audit)
 audit['completed']=True;atomic_write(manifest_file,audit);return audit

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan');ap.add_argument('--apply');ap.add_argument('--manifest',required=True);a=ap.parse_args()
 if a.plan:
  p=build_plan(inventory());atomic_write(a.plan,p);summary={k:v for k,v in p.items() if k not in ('retained_final_files','delete_files','protected_files','freed_inode_plan')};summary['retained_final_count']=len(p['retained_final_files']);summary['delete_reason_counts']={r:sum(x['reason']==r for x in p['delete_files']) for r in sorted(set(x['reason'] for x in p['delete_files']))};summary['protected_count']=len(p['protected_files']);print(json.dumps(summary,ensure_ascii=False))
 elif a.apply:print(json.dumps(apply(a.apply,a.manifest),ensure_ascii=False))
 else:ap.error('Use --plan or --apply')
if __name__=='__main__':main()

