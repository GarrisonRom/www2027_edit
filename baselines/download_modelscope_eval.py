"""Download and validate ModelScope capability corpora without executing remote code."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import sys
import shutil
import tarfile
import time
import traceback
import zipfile
import requests


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def download(url,path,expected=None):
    if path.exists():
        if expected and sha(path)!=expected:raise ValueError('Existing file checksum mismatch: '+str(path))
        return
    partial=path.with_suffix(path.suffix+'.part')
    for attempt in range(4):
        try:
            with requests.get(url,timeout=(15,45),stream=True) as response:
                response.raise_for_status()
                with partial.open('wb') as f:
                    for block in response.iter_content(1024*1024):
                        if block:f.write(block)
            if expected and sha(partial)!=expected:raise ValueError('Downloaded checksum mismatch: '+str(path))
            partial.replace(path);print('DOWNLOADED',path.name,path.stat().st_size,flush=True);return
        except Exception:
            if attempt==3:raise
            print('RETRY',attempt+1,path.name,flush=True);time.sleep(1)


def safe_name(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive name: '+name)
    return p


def prepare(root,reuse_raw=None):
    import datasets
    import pyarrow as pa
    import pyarrow.parquet as pq
    raw=root/'raw';raw.mkdir();prepared=root/'prepared';prepared.mkdir()
    if reuse_raw:
        for source in Path(reuse_raw).iterdir():
            if source.is_file() and not source.name.endswith('.part'):shutil.copy2(source,raw/source.name)
    api='https://modelscope.cn/api/v1/datasets/modelscope/'
    sources={}
    for dataset in ('mmlu','wikitext'):
        tree=raw/(dataset+'_tree.json')
        download(api+dataset+'/repo/tree?Revision=master&Root=',tree)
        listing=json.loads(tree.read_text(encoding='utf-8'))
        if listing.get('Code')!=200:raise ValueError('ModelScope tree API failed')
        entries={x['Path']:x for x in listing['Data']['Files']}
        for filename in ('dataset_infos.json',dataset+'.py'):
            entry=entries[filename];dst=raw/(dataset+'_'+filename)
            download(api+dataset+'/repo?Revision='+entry['Revision']+'&FilePath='+filename,dst,entry.get('Sha256'))
        sources[dataset]=dict(api=api+dataset,files=entries)
    mmlu_archive=raw/'mmlu_data.tar';entry=sources['mmlu']['files']['data.tar']
    download(api+'mmlu/repo?Revision='+entry['Revision']+'&FilePath=data.tar',mmlu_archive,entry['Sha256'])
    infos=json.loads((raw/'mmlu_dataset_infos.json').read_text(encoding='utf-8'))
    if len(infos)!=57:raise ValueError('MMLU must have57 configurations')
    tables={};counts={};files=[];source_count_notes=[];duplicate_choices=[]
    with tarfile.open(mmlu_archive,'r:*') as archive:
        for member in archive.getmembers():
            safe_name(member.name)
            if member.issym() or member.islnk():raise ValueError('Archive links not permitted')
            if not member.isfile() or not member.name.endswith('.csv'):continue
            name=PurePosixPath(member.name).name
            for subject in infos:
                for split,suffix in (('dev','dev'),('validation','val'),('test','test'),('auxiliary_train','train')):
                    if name!=f'{subject}_{suffix}.csv':continue
                    rows=[]
                    with io.TextIOWrapper(archive.extractfile(member),encoding='utf-8',newline='') as f:
                        for line in csv.reader(f):
                            if len(line)!=6 or line[5] not in 'ABCD' or len(line[5])!=1:raise ValueError('Invalid MMLU row: '+name)
                            if len(set(line[1:5]))<4:duplicate_choices.append(dict(file=name,row=len(rows)))
                            rows.append(dict(question=line[0],choices=line[1:5],answer='ABCD'.index(line[5])))
                    source_split='train' if split=='dev' else split
                    expected=infos[subject]['splits'][source_split]['num_examples']
                    # The old ModelScope loader uses pandas.read_csv(header=infer),
                    # losing the first data row. Raw MMLU CSV has NO header.
                    if len(rows) not in (expected,expected+1):raise ValueError(f'MMLU count mismatch:{subject}/{split}:{len(rows)}!={expected}(+1)')
                    source_count_notes.append(dict(subject=subject,split=split,source_metadata=expected,
                        raw_csv_rows=len(rows),first_question_preserved=True))
                    folder=prepared/'mmlu'/subject;folder.mkdir(parents=True,exist_ok=True)
                    path=folder/(split+'.parquet');pq.write_table(pa.Table.from_pylist(rows),path)
                    files.append(dict(path=path.relative_to(root).as_posix(),sha256=sha(path),rows=len(rows)))
                    counts.setdefault(subject,{})[split]=len(rows)
    for subject in infos:
        if not all(s in counts.get(subject,{}) for s in ('dev','validation','test')):
            raise ValueError('Missing MMLU evaluation split: '+subject)
    wiki_infos=json.loads((raw/'wikitext_dataset_infos.json').read_text(encoding='utf-8'))['wikitext-2-raw-v1']
    checks=wiki_infos['download_checksums']
    if len(checks)!=1:raise ValueError('Unexpected WikiText source count')
    url,metadata=next(iter(checks.items()));wiki_archive=raw/'wikitext-2-raw-v1.zip'
    download(url,wiki_archive,metadata['checksum'])
    wiki_counts={}
    with zipfile.ZipFile(wiki_archive) as archive:
        for name in archive.namelist():safe_name(name)
        for split,suffix in (('train','train'),('validation','valid'),('test','test')):
            names=[n for n in archive.namelist() if n.endswith('/wiki.'+suffix+'.raw')]
            if len(names)!=1:raise ValueError('WikiText split missing: '+split)
            text=archive.read(names[0]).decode('utf-8');lines=text.splitlines(keepends=True)
            if len(lines)!=wiki_infos['splits'][split]['num_examples']:raise ValueError('WikiText split count mismatch')
            # HF WikiText keeps source line endings in the text field. Do not strip/normalize.
            folder=prepared/'wikitext-2-raw-v1';folder.mkdir(parents=True,exist_ok=True)
            path=folder/(split+'.parquet');pq.write_table(pa.Table.from_pylist([dict(text=x if x.strip() else '') for x in lines]),path)
            files.append(dict(path=path.relative_to(root).as_posix(),sha256=sha(path),rows=len(lines)))
            wiki_counts[split]=len(lines)
    totals={split:sum(c.get(split,0) for c in counts.values()) for split in ('dev','validation','test')}
    if totals!={'dev':285,'validation':1531,'test':14042}:raise ValueError('MMLU standard total mismatch')
    manifest=dict(source='ModelScope',sources=sources,prepared_files=files,
        mmlu=dict(subject_count=57,counts=counts,totals=totals,answer_mapping='A/B/C/D to0/1/2/3; option order unchanged',
            source_count_notes=source_count_notes,duplicate_choice_rows=duplicate_choices,
            loader_repair='Parse headerless original CSV; do not reproduce ModelScope pandas header-inference row loss'),
        wikitext=dict(config='wikitext-2-raw-v1',counts=wiki_counts,archive_sha256=sha(wiki_archive),
            text_handling='Match source loader: preserve nonempty raw line including newline, whitespace-only line becomes empty text; join fields with two newlines'),
        format='local Parquet, generated without executing remote dataset scripts',
        limitations='ModelScope content hash frozen; HF pinned revision byte-level equivalence not certified')
    write(root/'manifest.json',manifest)
    print(json.dumps(dict(mmlu=totals,wikitext=wiki_counts,files=len(files))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--reuse-raw');a=p.parse_args()
    root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=False)
    write(root/'status.json',dict(status='downloading',started=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    write(root/'run_manifest.json',dict(command=sys.argv,source_sha256=sha(Path(__file__)),python=sys.executable))
    try:prepare(root,a.reuse_raw);write(root/'status.json',dict(status='completed',ended=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    except BaseException:
        write(root/'status.json',dict(status='failed',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
