"""Read-only SSH collection of completed baseline metrics for descriptive audits.

SSH password is prompted, never persisted. No training or remote writes.
"""
import argparse
import getpass
import hashlib
import json
from pathlib import Path

import paramiko
from audit_gsu_pareto import mark_frontier


def collect_remote(host, port, username, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=username, password=password,
                   timeout=20, look_for_keys=False, allow_agent=False)
    sources = []
    def read(sftp, path):
        with sftp.open(path, 'rb') as stream:
            content = stream.read()
        sources.append({'path': path, 'sha256': hashlib.sha256(content).hexdigest()})
        return json.loads(content)
    rows = []
    try:
        with client.open_sftp() as sftp:
            batches = [
                ('exp008', '/root/autodl-tmp/www2027_baselines_20261022_v2/exp/runs/exp008_remote_baselines/runs/batch_20261022_01'),
                ('exp009', '/root/autodl-tmp/www2027_rmu_20261022/exp/runs/exp009_remote_rmu/runs/batch_20261022_01')]
            for exp_id, root in batches:
                status = read(sftp, root + '/status.json')
                if status['status'] != 'completed':
                    raise RuntimeError(f'Batch not complete: {root}')
                for run_id in status['completed']:
                    if run_id.endswith('_smoke'):
                        continue
                    metrics = read(sftp, root + '/' + run_id + '/metrics.json')
                    config = read(sftp, root + '/' + run_id + '/config.json')
                    groups = metrics['groups']
                    sets = metrics['sets']
                    rows.append({'method': config['method'] + '_' + config['supervision'],
                        'model': Path(config['model']).name,
                        'supervision': 'none' if config['method'] == 'Original' else config['supervision'],
                        'seed': config['seed'], 'run_id': run_id, 'exp_id': exp_id,
                        'multihop_recall': sets['F1_hop2']['summary']['mcqa_accuracy'],
                        'retain_heldout': groups['retain_heldout']['summary']['mcqa_accuracy'],
                        'direct_recall': groups['F0']['summary']['mcqa_accuracy'],
                        'all_hop2_recall': None, 'source': root + '/' + run_id + '/metrics.json'})
    finally:
        client.close()
    return {'rows': rows, 'sources': sources,
            'limitations': ['Single-seed fixed-input transfer, not tuned baseline optima',
                            'Current remote evaluator lacks ALL89-path full-hop2 metric',
                            'Adapters permit separate full-hop2 evaluation without retraining',
                            'No same-model GSU run yet; no cross-model common frontier',
                            'Retain heldout is not semantically unrelated by construction'],
            'per_model': {
                model: {'all_supervisions': mark_frontier(
                    [row for row in rows if row['model'] == model],
                    {'multihop_recall': 'min', 'retain_heldout': 'max'}),
                    'F0_only_and_original': mark_frontier(
                    [row for row in rows if row['model'] == model and row['supervision'] != 'F0F1'],
                    {'multihop_recall': 'min', 'retain_heldout': 'max'})}
                for model in sorted({row['model'] for row in rows})}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=22)
    parser.add_argument('--username', default='root')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    report = collect_remote(args.host, args.port, args.username, getpass.getpass('SSH password: '))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'remote_pareto_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Collected {len(report["rows"])} completed outcomes')


if __name__ == '__main__':
    main()
