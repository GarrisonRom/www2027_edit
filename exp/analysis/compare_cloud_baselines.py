"""Build traceable Markdown comparison/ranking tables from real cloud evidence.

The unrun GSU candidate is never assigned a score or rank. Targets are not forecasts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ranks(values, direction):
    if direction not in ('min', 'max'):
        return {}
    finite = {key: value for key, value in values.items()
              if value is not None and math.isfinite(value)}
    result = {}
    for key, value in finite.items():
        better = sum(other < value - 1e-10 if direction == 'min' else other > value + 1e-10
                     for other in finite.values())
        result[key] = better + 1
    return result


def load_records(evidence):
    records = []
    for experiment in ('exp008', 'exp009'):
        batch = evidence / experiment
        status = read(batch / 'status.json')
        if status['status'] != 'completed':
            raise RuntimeError('Incomplete archived batch')
        for run_id in status['completed']:
            if run_id.endswith('_smoke'):
                continue
            folder = batch / run_id
            config, metrics = read(folder / 'config.json'), read(folder / 'metrics.json')
            state, modules = read(folder / 'status.json'), read(folder / 'modules.json')
            if state['status'] != 'completed':
                raise RuntimeError('Incomplete archived run')
            method = config['method']
            supervision = 'none' if method == 'Original' else config['supervision']
            record = {'model': Path(config['model']).name, 'method': method,
                'label': method + (' [F0+F1]' if supervision == 'F0F1' else ''),
                'supervision': supervision, 'seed': config['seed'], 'run_id': run_id,
                'rank_eligible': supervision == 'F0', 'groups': metrics['groups'],
                'sets': metrics['sets'], 'elapsed_seconds': state['seconds'],
                'peak_memory_gib': state['peak_cuda_allocated_bytes'] / (1024 ** 3),
                'trainable_parameters': modules['trainable_parameters'],
                'source': str(folder / 'metrics.json'),
                'source_sha256': hashlib.sha256((folder / 'metrics.json').read_bytes()).hexdigest()}
            records.append(record)
    if len(records) != 14:
        raise RuntimeError(f'Expected14 actual outcomes, found{len(records)}')
    c2root = evidence / 'exp011'
    if c2root.exists():
        for folder in sorted(c2root.iterdir()):
            if not folder.is_dir() or not (folder/'metrics.json').exists(): continue
            config, metrics = read(folder/'config.json'), read(folder/'metrics.json')
            state, modules = read(folder/'status.json'), read(folder/'modules.json')
            if state['status'] != 'completed': raise RuntimeError('Incomplete C2 run')
            records.append({'model':Path(config['model']).name,'method':'GSU-C2','label':'GSU-C2',
                'supervision':'F0','seed':42,'run_id':folder.name,'rank_eligible':True,
                'groups':metrics['groups'],'sets':metrics['sets'],'full_hop2':metrics['full_hop2'],
                'elapsed_seconds':state['seconds'],
                'peak_memory_gib':state['peak_cuda_allocated_bytes']/(1024**3),
                'trainable_parameters':modules['trainable_parameters'],
                'source':str(folder/'metrics.json'),
                'source_sha256':hashlib.sha256((folder/'metrics.json').read_bytes()).hexdigest()})
    for model in {row['model'] for row in records}:
        original = next(row for row in records if row['model'] == model and row['method'] == 'Original')
        for row in [r for r in records if r['model'] == model]:
            for section in ('groups', 'sets'):
                for key, audit in row[section].items():
                    initial = original[section][key]
                    if [item['id'] for item in audit['items']] != [item['id'] for item in initial['items']]:
                        raise RuntimeError('Cannot compare unmatched evaluation items')
                    audit['summary']['answer_lp_delta_vs_original'] = (
                        audit['summary']['mean_answer_logprob'] - initial['summary']['mean_answer_logprob'])
    return records


def group_metric(row, section, group, field):
    return row.get(section, {}).get(group, {}).get('summary', {}).get(field)


def pretty(value, percent=False):
    if value is None:
        return '未测'
    return f'{100*value:.2f}%' if percent else f'{value:.4f}'


def ranked_table(rows, title, section, group, field, direction, percent=False):
    values = {row['label']: group_metric(row, section, group, field) for row in rows}
    position = ranks({row['label']: values[row['label']] for row in rows if row['rank_eligible']}, direction)
    reverse = direction == 'max'
    trained = sorted([row for row in rows if row['rank_eligible']],
                     key=lambda row: values[row['label']], reverse=reverse)
    references = [row for row in rows if not row['rank_eligible']]
    lines = [f'## {title}', '', '| 方法 | 实测值 | F0-only排名 | 口径 |', '|---|---:|---:|---|']
    for row in trained + references:
        role = 'F0-only已测' if row['rank_eligible'] else '不训练参照' if row['method'] == 'Original' else '扩展监督参照'
        lines.append(f"| {row['label']} | {pretty(values[row['label']],percent)} | {position.get(row['label'],'—')} | {role} |")
    lines += ['| KLUE（必须补充） | 待复现 | — | 无实测，不参加排名 |', '']
    return lines


def build_model_report(model, rows):
    lines = [f'# {model}：云端全指标比较', '',
        '以下均为seed42的实际结果。只在六种F0损失方法（含访问F1指导的C2）训练方法中逐指标排名；并列采用竞赛排名。',
        'Original和F0+F1单列参照，不参加同监督排名；不存在可信的跨指标总分。',
        'C2已纳入实测排名；额外F1指导/retain反馈导致信息访问和计算预算不同，排名仅作描述。', '']
    for section, group, label, direction in (
        ('groups', 'F0', 'F0直接回忆', 'min'),
        ('groups', 'F0_paraphrases', 'F0改写回忆', 'min'),
        ('sets', 'F1_hop2', '历史F1覆盖多跳回忆（非全部hop2）', 'min'),
        ('groups', 'retain_train_pool', '训练保护池（非独立测试）', 'max'),
        ('groups', 'retain_heldout', '非锚点保持（不等同语义无关）', 'max')):
        arrow = '↓ 越低越好' if direction == 'min' else '↑ 越高越好'
        for field, metric, percent in (
            ('mcqa_accuracy', '选择题正确率', True),
            ('mean_answer_logprob', '答案平均logprob（nats）', False),
            ('mean_mcqa_margin', '全串MCQA margin', False),
            ('answer_lp_delta_vs_original', '答案LP相对原始模型变化（nats）', False)):
            lines += ranked_table(rows, f'{label}：{metric} {arrow}', section, group, field, direction, percent)
    lines += ['## 支持事实与历史切片：审计，不设统一优劣方向', '',
        'F1直接事实改变既可能表示关联影响，也可能表示不必要的支持事实破坏；当前主线不强制其完全保持。',
        'D类切片来自旧损伤/恢复研究，不能一面把同一hop2题标为应遗忘，一面标为必须保持。',
        '因此以下仅列实测，不给“越高越好/越低越好”的伪统一结论或总榜。', '']
    for section, group in [('groups', 'F1')] + [('sets', key) for key in (
        'D_so', 'D_hop2', 'K', 'D_hop2_F1_overlap', 'D_hop2_no_F1_overlap')]:
        lines += [f'### {group}（审计）', '',
                  '| 方法 | 正确率 | 答案LP | MCQA margin | 全串正确选项LP |',
                  '|---|---:|---:|---:|---:|']
        for row in rows:
            stats = row[section][group]['summary']
            lines.append(f"| {row['label']} | {pretty(stats['mcqa_accuracy'],True)} | "
                f"{pretty(stats['mean_answer_logprob'])} | {pretty(stats['mean_mcqa_margin'])} | "
                f"{pretty(stats['mean_correct_answer_logprob'])} |")
        lines += ['| KLUE（必须补充） | 待复现 | 待复现 | 待复现 | 待复现 |', '']
    lines += ['## 资源代价（↓成本更低，不等于算法效果更好）', '',
              '墙钟包含底模加载、训练和既定评测，未拆分定位/纯训练时间；各方法单次测量没有稳定性区间。',
              '通用LoRA与受限层RMU参数预算不同，必须与效果一起解释。', '',
              '括号为六种F0损失方法（含访问F1指导的C2）方法内的成本排名；Original和扩展参照不排名。', '',
              '| 方法 | 总墙钟秒 ↓ | 峰值显存GiB ↓ | 可训练参数数 ↓ |', '|---|---:|---:|---:|']
    cost_ranks = {field: ranks({row['label']: row[field] for row in rows if row['rank_eligible']}, 'min')
                  for field in ('elapsed_seconds', 'peak_memory_gib', 'trainable_parameters')}
    for row in rows:
        rank_labels = {field: f" (#{ranking[row['label']]})" if row['label'] in ranking else ''
                       for field, ranking in cost_ranks.items()}
        lines.append(f"| {row['label']} | {row['elapsed_seconds']:.2f}{rank_labels['elapsed_seconds']} | "
            f"{row['peak_memory_gib']:.3f}{rank_labels['peak_memory_gib']} | "
            f"{row['trainable_parameters']}{rank_labels['trainable_parameters']} |")
    lines += ['| KLUE（必须补充） | 未测 | 未测 | 待复现冻结 |', '',
              'C2已测全部89路径，但baseline统一补评、低关联分层、生成行为、定位开销拆分、多seed误差仍缺，不能填0或用本地1.5B替代。']
    return lines


def generate(evidence, output):
    records = load_records(evidence)
    output.mkdir(parents=True, exist_ok=True)
    readme = ['# 云端baseline全指标比较与GSU候选评估', '',
        '证据：exp008/exp009/exp011，共16个已完成行为端点、两个模型、seed42。',
        'GSU-C2已完成并纳入全部行为与成本表；MMLU/PPL正式评测尚未完成。旧筛选门槛只是事前目标。', '',
        '## 指标方向', '', '| 指标族 | 方向 | 用途 |', '|---|---|---|',
        '| F0直接/改写正确率、目标答案LP、目标margin | ↓ | 遗忘行为；LP与选择题排序分开 |',
        '| 覆盖多跳正确率、目标答案LP、目标margin | ↓ | 现有覆盖切片，不代表全部多跳 |',
        '| 训练保护池/非锚点正确率、答案LP、margin | ↑ | 保持审计；非锚点不等于无关 |',
        '| F1直接事实、历史D/K切片 | 审计，无统一方向 | 不制造虚假的跨任务总排名 |',
        '| 墙钟、显存、参数量 | ↓表示成本更低 | 不单独等同方法质量 |',
        '| 全部89路径、关联分层、生成式行为、多seed | 未测 | 不补造或从别的模型估算 |', '',
        '## 每模型主要指标', '']
    targets = []
    models = sorted({row['model'] for row in records})
    for model in models:
        rows = [row for row in records if row['model'] == model]
        readme += [f'### {model}', '', '| 方法 | F0正确率 ↓ | 改写正确率 ↓ | 覆盖多跳 ↓ | 非锚点保持 ↑ |', '|---|---:|---:|---:|---:|']
        for row in rows:
            values = [group_metric(row,'groups','F0','mcqa_accuracy'),
                      group_metric(row,'groups','F0_paraphrases','mcqa_accuracy'),
                      group_metric(row,'sets','F1_hop2','mcqa_accuracy'),
                      group_metric(row,'groups','retain_heldout','mcqa_accuracy')]
            readme.append('| '+row['label']+' | '+' | '.join(pretty(x,True) for x in values)+' |')
        readme += ['| KLUE（必须补充） | 待复现 | 待复现 | 待复现 | 待复现 |', '']
        detailed = build_model_report(model, rows)
        (output / (model + '_all_metrics.md')).write_text('\n'.join(detailed)+'\n', encoding='utf-8')
        reference = next(row for row in rows if row['method'] == 'GradDiff')
        direct = reference['groups']['F0']['summary']
        multi = reference['sets']['F1_hop2']['summary']
        utility = reference['groups']['retain_heldout']['summary']
        target = {'model':model, 'status':'design_target_not_prediction', 'reference':'GradDiff_F0',
                  'direct_correct_max':round(direct['mcqa_accuracy']*direct['n']),
                  'covered_multihop_correct_max':round(multi['mcqa_accuracy']*multi['n'])-1,
                  'heldout_correct_min':round(utility['mcqa_accuracy']*utility['n']),
                  'direct_n':direct['n'],'covered_n':multi['n'],'heldout_n':utility['n']}
        targets.append(target)
    readme += ['## 全89路径与能力补评状态', '',
        '| 模型 | 方法 | 全89路径宏回忆 ↓ | MMLU-10/科目 ↑ | WikiText-2 PPL ↓ |',
        '|---|---|---:|---:|---:|']
    for row in records:
        all89=row.get('full_hop2',{}).get('all_hop2',{}).get('summary',{}).get('path_macro_accuracy')
        readme.append('| '+row['model']+' | '+row['label']+' | '+pretty(all89,True)+' | 待正式评测 | 待正式评测 |')
    readme += ['', 'C2全路径结果已测；其余方法全89路径与所有端点MMLU/PPL由exp012补齐，禁止以覆盖75题替代全部路径。', '']
    readme += ['## 必须补充的定位baseline：KLUE', '',
        'FaithUn已经是当前数据集；KLUE是该论文的方法，尚未包含在14个实测端点内。',
        '官方yny0506/faithun仓库当前递归清单只见数据、split、README和LICENSE，没有可执行KLUE训练实现。',
        '需按论文核对神经元归因、失配问答正则、未遗忘样本筛选和掩码更新，验证后另开run。',
        '已把KLUE列为必须补齐项，所有比较表保留待复现行，不伪造分数或混同CLUE。', '',
        '## C2候选：可检验门槛，不是预估成绩', '',
        '当前已测C2为模型风险评分+共享方向子空间+实际位移路由+F0进展护栏；以下为历史事前门槛，不是当前效果结论。',
        '先以GradDiff作为平衡型参照定义一个可证伪区域，另完整报告GA等更强遗忘参照：', '',
        '| 模型 | F0答对最多 | 覆盖多跳答对最多 | 非锚点答对至少 |', '|---|---:|---:|---:|']
    for target in targets:
        readme.append(f"| {target['model']} | {target['direct_correct_max']}/{target['direct_n']} | "
            f"{target['covered_multihop_correct_max']}/{target['covered_n']} | {target['heldout_correct_min']}/{target['heldout_n']} |")
    readme += ['', '满足这一矩形区域只说明在这三个已定义指标上不弱于/局部优于该参照，',
        '不保证支配所有方法，更不代表全部89路径或无关知识上的帕累托最优。',
        '正式方法仍需全部hop2、真实/随机组件方向对照和多seed；不能用候选区间伪装实测行。', '',
        '## 排名与缺失规则', '',
        '只在同模型五个F0-only训练方法内排名；并列为1,2,2,4式竞赛排名。',
        '不训练Original和扩展监督单列。C2已测并参与描述性排名；没有统一综合分，也不合并不同模型。',
        '单seed只作描述，不能以单次排名声称统计显著；C2访问F1作为指导且额外反馈计算，不能把同F0损失当成同信息/同计算预算。每个模型详细文件含20张单指标排序表、6张审计表和资源表。', '',
        '溯源：相邻evidence目录保存完整逐题metrics、config、modules、status、schedule及采集哈希。']
    (output / 'README.md').write_text('\n'.join(readme)+'\n', encoding='utf-8')
    (output / 'comparison.json').write_text(json.dumps({'records':records,'candidate_targets':targets,
        'rank_rules':'within model / F0-only / competition ties / no unmeasured ranks'},
        ensure_ascii=False,indent=2,allow_nan=False), encoding='utf-8')
    print('ACTUAL_OUTCOMES',len(records),'MODEL_REPORTS',len(models),'GSU_SCORES_FABRICATED',0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    generate(args.evidence,args.output)
