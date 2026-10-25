"""Snapshot source to SSH and archive completed cloud evidence locally.

Credentials are prompted, not saved. No model transfer, deletion or job launch.
"""
import argparse
import base64
import getpass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST_KEY = 'liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc='
BATCHES = {
    'exp008': '/root/autodl-tmp/www2027_baselines_20261022_v2/exp/runs/exp008_remote_baselines/runs/batch_20261022_01',
    'exp009': '/root/autodl-tmp/www2027_rmu_20261022/exp/runs/exp009_remote_rmu/runs/batch_20261022_01',
}


class PinnedHostKey(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        fingerprint = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode()
        if fingerprint != HOST_KEY:
            raise RuntimeError('SSH host fingerprint changed; verify before sending credentials')


def connect(password, host='connect.nmb1.seetacloud.com', port=33935, username='root'):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(PinnedHostKey())
    client.connect(host, port=port, username=username, password=password,
                   timeout=20, look_for_keys=False, allow_agent=False)
    return client


def safe_unpack(content, destination):
    """Only regular files beneath a new explicitly selected destination."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            target = destination.joinpath(*relative.parts).resolve()
            if (not member.isfile() or relative.is_absolute() or '..' in relative.parts
                    or not target.is_relative_to(destination)):
                raise RuntimeError(f'Unsafe archive entry: {member.name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('xb') as output:
                output.write(source.read())


def pull_evidence(client, destination):
    selection = []
    with client.open_sftp() as sftp:
        for experiment, root in BATCHES.items():
            with sftp.open(root + '/status.json') as stream:
                status = json.load(stream)
            if status['status'] != 'completed':
                raise RuntimeError(f'Cannot archive incomplete batch: {experiment}')
            for filename in ('status.json', 'batch_manifest.json', 'environment.txt'):
                selection.append((root + '/' + filename, experiment + '/' + filename))
            for run in status['completed']:
                if run.endswith('_smoke'):
                    continue
                for filename in ('metrics.json', 'config.json', 'status.json', 'modules.json',
                                 'diagnostics.jsonl', 'schedule.json', 'run_log.md'):
                    path = root + '/' + run + '/' + filename
                    try:
                        sftp.stat(path)
                    except FileNotFoundError:
                        if filename in ('diagnostics.jsonl', 'schedule.json'):
                            continue
                        raise
                    selection.append((path, experiment + '/' + run + '/' + filename))
    program = (
        'import sys,tarfile,json; files=json.loads(sys.stdin.read()); '
        'archive=tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz"); '
        '[(archive.add(path,arcname=name,recursive=False)) for path,name in files]; archive.close()'
    )
    stdin, stdout, stderr = client.exec_command(
        '/root/miniconda3/bin/python -c ' + shlex.quote(program), timeout=180)
    stdin.write(json.dumps(selection))
    stdin.channel.shutdown_write()
    content = stdout.read()
    error = stderr.read().decode(errors='replace')
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error)
    safe_unpack(content, destination)
    manifest = {'archive_sha256': hashlib.sha256(content).hexdigest(), 'files': []}
    for remote, relative in selection:
        local = Path(destination) / relative
        manifest['files'].append({'remote': remote, 'relative': relative,
            'bytes': local.stat().st_size, 'sha256': hashlib.sha256(local.read_bytes()).hexdigest()})
    (Path(destination) / 'collection_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Collected {len(selection)} evidence files ({len(content)} compressed bytes)', flush=True)
    return manifest


def source_paths(root=ROOT):
    allowed = {'.py', '.ps1', '.sh', '.md', '.tex', '.bib', '.cls', '.sty', '.bst',
               '.json', '.jsonl', '.yaml', '.yml', '.toml', '.txt', '.sha256'}
    tracked = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root)
    paths = tracked.decode('utf-8').split('\0')
    submodule = root / 'code/open-unlearning'
    if submodule.is_dir():
        upstream = subprocess.check_output(
            ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=submodule)
        paths.extend('code/open-unlearning/' + name
                     for name in upstream.decode('utf-8').split('\0') if name)
    selected = []
    for name in sorted(set(paths)):
        path = root / name
        if not name or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            continue
        parts = PurePosixPath(name).parts
        if any(part in ('.git', '__pycache__', '.pytest_cache', 'node_modules') for part in parts):
            continue
        if name.startswith(('tmp/', 'exp/sync/', 'exp/adapters/')) or '盲审' in name:
            continue
        if name.startswith('exp/runs/') and path.name not in ('card.md', 'README.md', 'probe01.md'):
            continue
        if path.name.startswith('.env') or path.name in ('local_config.json', '_kaiti_text.txt', '_paper_text.txt'):
            continue
        if path.suffix.lower() not in allowed and path.name not in ('.gitignore', '.gitmodules', 'LICENSE', 'UPSTREAM_LICENSE'):
            continue
        if path.stat().st_size > 10 * 1024 * 1024:
            continue
        selected.append(name)
    required = ['exp/train/run_remote_baselines.py', 'exp/train/run_remote_rmu.py',
                'data/faithun/FaithUn_Dataset.json', 'exp/results/F1.json',
                'exp/results/per_item_G2.jsonl', 'exp/results/attribution/items.json']
    if not set(required).issubset(selected):
        raise RuntimeError('Required source/input missing from payload')
    return selected


def push_snapshot(client, remote, receipt):
    if not remote.startswith('/root/autodl-tmp/www2027_sync_') or '..' in PurePosixPath(remote).parts:
        raise ValueError('Snapshot destination must be a new www2027_sync_* directory on data disk')
    files = []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for relative in source_paths():
            raw = (ROOT / relative).read_bytes()
            info = tarfile.TarInfo(relative)
            info.size, info.mode = len(raw), 0o644
            archive.addfile(info, io.BytesIO(raw))
            files.append({'path': relative, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        manifest = {'files': files, 'local_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
            'local_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT).decode('utf-8'),
            'remote_snapshot': remote, 'direction': 'local source to new remote snapshot',
            'excluded': 'models/adapters/.git/credentials/builds/live run trees/tmp'}
        extras = {'sync_manifest.json': json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8'),
                  'local_dirty.patch': subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT)}
        for relative, raw in extras.items():
            info = tarfile.TarInfo(relative)
            info.size, info.mode = len(raw), 0o644
            archive.addfile(info, io.BytesIO(raw))
    with client.open_sftp() as sftp:
        try:
            sftp.stat(remote)
        except FileNotFoundError:
            sftp.mkdir(remote)
        else:
            raise FileExistsError(f'Will not overwrite snapshot: {remote}')
        buffer.seek(0)
        sftp.putfo(buffer, remote + '/.sync_payload.tar.gz')
    program = '''import pathlib,tarfile,json,hashlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
with tarfile.open(root/'.sync_payload.tar.gz') as archive:
 for member in archive.getmembers():
  path=(root/member.name).resolve()
  if not member.isfile() or not path.is_relative_to(root):raise RuntimeError('Unsafe member')
  path.parent.mkdir(parents=True,exist_ok=True)
  with archive.extractfile(member) as src,path.open('xb') as dst:dst.write(src.read())
manifest=json.loads((root/'sync_manifest.json').read_text())
bad=[f['path'] for f in manifest['files'] if hashlib.sha256((root/f['path']).read_bytes()).hexdigest()!=f['sha256']]
if bad:raise RuntimeError(str(bad))
print(json.dumps({'verified_files':len(manifest['files']),'mismatches':bad,'snapshot':str(root)}))
'''
    _, stdout, stderr = client.exec_command('/root/miniconda3/bin/python -c ' + shlex.quote(program)
                                          + ' ' + shlex.quote(remote), timeout=120)
    result, error = stdout.read().decode(), stderr.read().decode()
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error)
    verified = json.loads(result)
    receipt = Path(receipt)
    receipt.mkdir(parents=True, exist_ok=False)
    (receipt / 'sync_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (receipt / 'verification.json').write_text(json.dumps(verified, indent=2), encoding='utf-8')
    print(result, flush=True)
    return verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pull_evidence')
    parser.add_argument('--push_snapshot')
    parser.add_argument('--receipt')
    args = parser.parse_args()
    if not args.pull_evidence and not args.push_snapshot:
        parser.error('Choose pull_evidence and/or push_snapshot')
    if args.push_snapshot and not args.receipt:
        parser.error('push_snapshot requires receipt')
    client = connect(getpass.getpass('SSH password: '))
    try:
        if args.pull_evidence:
            pull_evidence(client, Path(args.pull_evidence))
        if args.push_snapshot:
            push_snapshot(client, args.push_snapshot, args.receipt)
    finally:
        client.close()


if __name__ == '__main__':
    main()
