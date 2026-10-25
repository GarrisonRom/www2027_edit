"""Verified local ModelScope corpora; no Hugging Face network requests."""
import hashlib
import json
from pathlib import Path


def root_path(root=None):
    if root:return Path(root)
    config=json.loads((Path(__file__).parent/'capability_data.json').read_text(encoding='utf-8'))
    return Path(config['root'])


def manifest(root=None):
    root=root_path(root)
    data=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if json.loads((root/'status.json').read_text(encoding='utf-8'))['status']!='completed':
        raise ValueError('Capability download not complete')
    return root,data


def checked_files(dataset,subject=None,root=None):
    root,data=manifest(root)
    prefix='prepared/mmlu/'+subject+'/' if dataset=='mmlu' else 'prepared/wikitext-2-raw-v1/'
    records=[r for r in data['prepared_files'] if r['path'].startswith(prefix)]
    files={}
    for row in records:
        path=root/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Local dataset hash mismatch: '+str(path))
        files[path.stem]=str(path)
    if not all(s in files for s in (('dev','validation','test') if dataset=='mmlu' else ('train','validation','test'))):
        raise ValueError('Missing local dataset split')
    return files


def load_mmlu(subject,root=None):
    from datasets import load_dataset
    base=root_path(root)
    return load_dataset('parquet',data_files=checked_files('mmlu',subject,base),
                        cache_dir=str(base/'datasets_cache'))


def mmlu_task_config(manager,name,root=None):
    from copy import deepcopy
    if not name.startswith('mmlu_'):raise ValueError('Not an MMLU subject')
    subject=name.removeprefix('mmlu_');base=root_path(root)
    c=deepcopy(manager._get_config(name))
    if c.get('dataset_name')!=subject:raise ValueError('MMLU subject/config mismatch')
    c.update(dataset_path='parquet',dataset_name=None,dataset_kwargs=dict(
        data_files=checked_files('mmlu',subject,base),cache_dir=str(base/'datasets_cache')))
    return c


def load_wikitext(root=None):
    import pyarrow.parquet as pq
    root,data=manifest(root);files=checked_files('wikitext',root=root)
    lines=pq.read_table(files['test']).column('text').to_pylist()
    corpus='\n\n'.join(lines)
    return corpus,dict(source='ModelScope',config='wikitext-2-raw-v1',split='test',
        document_count=len(lines),text_sha256=hashlib.sha256(corpus.encode()).hexdigest(),
        archive_sha256=data['wikitext']['archive_sha256'],
        source_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        hf_pinned_revision_equivalence='not certified; explicit ModelScope source track')
