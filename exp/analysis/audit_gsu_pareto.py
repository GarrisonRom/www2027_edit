"""Descriptive Pareto audit of frozen A/B outcomes; never renders a figure."""
import argparse
import hashlib
import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]


def dominates(left, right, objectives):
    """Objectives map numeric field names to min or max; equality is not dominance."""
    oriented = [(left[key], right[key], direction) for key, direction in objectives.items()]
    return (all(a <= b if direction == 'min' else a >= b for a, b, direction in oriented)
            and any(a < b if direction == 'min' else a > b for a, b, direction in oriented))


def mark_frontier(rows, objectives):
    result = []
    for row in rows:
        dominated_by = [other['method'] for other in rows
                        if dominates(other, row, objectives)]
        result.append({**row, 'dominated_by': dominated_by,
                       'empirically_nondominated': not dominated_by})
    return result


def collect():
    result = []
    batches = [('exp006_gsu_memory_guided', 'batch_20261021_132023_431bf0'),
               ('exp007_gsu_integrated_guidance', 'batch_20261022_121504_28dccc')]
    for experiment, batch in batches:
        folder = WORKSPACE / 'exp/runs' / experiment / 'runs'
        for path in sorted(folder.glob(f'run_{batch}_*/metrics.json')):
            metrics = json.loads(path.read_text(encoding='utf-8'))
            variant = metrics['diagnostics']['variant']
            audits = metrics['logprob_audits']
            result.append({
                'method': variant, 'model': 'Qwen2.5-1.5B-Instruct', 'seed': 42,
                'supervision': 'F0F1' if variant in ('A1', 'A4', 'B1') else 'F0',
                'multihop_recall': metrics['set_logprob_audits']['F1_hop2']['summary']['mcqa_accuracy'],
                'retain_heldout': audits['retain_heldout']['summary']['mcqa_accuracy'],
                'direct_recall': audits['F0']['summary']['mcqa_accuracy'],
                'all_hop2_recall': metrics['five_set']['hop2'],
                'run_id': path.parent.name, 'source': str(path.relative_to(WORKSPACE)),
                'metrics_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    if len(result) != 13:
        raise RuntimeError(f'Expected all seven A and six B variants; got {len(result)}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = collect()
    objectives = {'multihop_recall': 'min', 'retain_heldout': 'max'}
    report = {
        'status': 'exploratory_posthoc_coordinate_choice',
        'objectives': objectives,
        'limitations': ['single seed, finite observed candidate set; no global optimality',
                        'retain heldout is not proven semantically unrelated',
                        'F1_hop2 is historical coverage-selected, not independent confirmation',
                        'A0=B0 and A1=B1 are duplicate outcomes, not independent seeds',
                        'no pooling across model/environment/supervision families',
                        'no confidence intervals or curves inferred from single endpoints'],
        'all_supervisions': mark_frontier(rows, objectives),
        'F0_only': mark_frontier([row for row in rows if row['supervision'] == 'F0'], objectives),
        'three_objectives': mark_frontier(rows, {**objectives, 'direct_recall': 'min'}),
        'all_hop2_objectives': mark_frontier(rows, {'all_hop2_recall': 'min', 'retain_heldout': 'max'}),
        'F0_only_all_hop2': mark_frontier([row for row in rows if row['supervision'] == 'F0'],
                                          {'all_hop2_recall': 'min', 'retain_heldout': 'max'}),
    }
    (output / 'pareto_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name in ('all_supervisions', 'F0_only'):
        print(name, [row['method'] for row in report[name] if row['empirically_nondominated']])


if __name__ == '__main__':
    main()
