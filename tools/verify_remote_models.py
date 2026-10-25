"""Verify downloaded ModelScope files and write a reproducibility manifest."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from modelscope.hub.api import HubApi


def main():
    root = Path('/root/autodl-tmp/models')
    models = ['Qwen/Qwen3.5-4B', 'LLM-Research/Llama-3.2-3B-Instruct']
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'models': []}
    api = HubApi()
    for model_id in models:
        folder = root / model_id.split('/')[-1]
        files = []
        for entry in api.get_model_files(model_id, recursive=True):
            relative = entry.get('Path', '')
            if entry.get('Type') == 'tree' or relative == 'original' or relative.startswith('original/'):
                continue
            if relative.endswith(('.bin', '.pth')):
                continue
            path = folder / relative
            if not path.is_file():
                raise RuntimeError(f'Missing file: {path}')
            if path.stat().st_size != entry['Size']:
                raise RuntimeError(f'Size mismatch: {path}')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            files.append({'path': relative, 'bytes': path.stat().st_size,
                          'sha256': digest.hexdigest()})
        index = json.loads((folder / 'model.safetensors.index.json').read_text())
        for shard in set(index['weight_map'].values()):
            if not (folder / shard).is_file():
                raise RuntimeError(f'Missing weight shard: {shard}')
        manifest['models'].append({'model_id': model_id, 'directory': str(folder),
                                   'source': 'ModelScope', 'files': files})
        print(f'VERIFIED {model_id}: {len(files)} files', flush=True)
    output = root / 'download_manifest.json'
    output.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
