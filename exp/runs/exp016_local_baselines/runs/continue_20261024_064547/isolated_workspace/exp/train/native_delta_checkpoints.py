"""Lossless native-weight XOR checkpoints; frozen base model is required for restore."""
import hashlib
import json
from pathlib import Path
import shutil
import zlib
import numpy as np
import torch


def bytes_of(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def sha(data):return hashlib.sha256(data).hexdigest()


def capture_base(model):
    return {name:parameter.detach().cpu().clone() for name,parameter in model.named_parameters()
            if parameter.requires_grad}


def save_delta(model,original,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    records=[]
    params=dict(model.named_parameters())
    for index,(name,initial) in enumerate(original.items()):
        current=params[name]
        if current.shape!=initial.shape or current.dtype!=initial.dtype:
            raise RuntimeError('Native checkpoint dtype/shape changed')
        a=bytes_of(initial);b=bytes_of(current)
        record={'name':name,'shape':list(initial.shape),'dtype':str(initial.dtype),
                'base_sha256':sha(a),'final_sha256':sha(b)}
        if a!=b:
            delta=np.bitwise_xor(np.frombuffer(a,dtype=np.uint8),np.frombuffer(b,dtype=np.uint8))
            compressed=zlib.compress(delta.tobytes(),level=6)
            if shutil.disk_usage(destination).free<len(compressed)+512*1024**2:
                raise RuntimeError('Insufficient space for lossless delta; do not drop checkpoint')
            filename=f'tensor_{index:05d}.xor.zlib'
            (destination/filename).write_bytes(compressed)
            record.update(file=filename,compressed_sha256=sha(compressed),bytes=len(compressed))
        records.append(record)
    manifest={'format':'native-xor-zlib-v1','requires_exact_base':True,'tensors':records,
              'final_parameter_digest':sha(json.dumps([(r['name'],r['final_sha256']) for r in records]).encode())}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


@torch.no_grad()
def restore_delta(model,destination):
    destination=Path(destination)
    manifest=json.loads((destination/'manifest.json').read_text(encoding='utf-8'))
    params=dict(model.named_parameters())
    for record in manifest['tensors']:
        parameter=params[record['name']]
        initial=bytes_of(parameter)
        if sha(initial)!=record['base_sha256']:raise RuntimeError('Wrong base checkpoint: '+record['name'])
        if 'file' not in record:continue
        path=(destination/record['file']).resolve()
        if path.parent!=destination.resolve():raise ValueError('Unsafe checkpoint path')
        data=path.read_bytes()
        if sha(data)!=record['compressed_sha256']:raise RuntimeError('Corrupted native delta')
        delta=zlib.decompress(data)
        if len(delta)!=len(initial):raise RuntimeError('Wrong delta size')
        final=np.bitwise_xor(np.frombuffer(initial,dtype=np.uint8),np.frombuffer(delta,dtype=np.uint8)).copy()
        if sha(final.tobytes())!=record['final_sha256']:raise RuntimeError('Native restore hash mismatch')
        value=torch.from_numpy(final).view(parameter.dtype).reshape(parameter.shape)
        parameter.copy_(value.to(parameter.device))
    return manifest

