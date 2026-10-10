"""Audit combined runs and plot measured task success without curve fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from .packet_loss_report import audit_condition


COMPATIBLE_KEYS = [
    'experiment', 'mtu', 'header_budget', 'feature_payload_per_full_packet',
    'seed', 'random_seeds', 'questions_sha256', 'model_sha256', 'projector_sha256',
    'tokenizer_sha256', 'runtime', 'runtime_dll_sha256', 'language_quantization',
    'vision_weights', 'wire_dtype', 'context', 'max_tokens', 'temperature',
    'prompt_template', 'scope', 'preprocessing',
]


def audit_sources(sources, questions):
    manifest = Path(questions).read_bytes()
    labels = {row['question_id']: row for row in json.loads(manifest)}
    if len(labels) != 125:
        raise ValueError('Expected 125 unique fixed questions')
    configs, conditions, summaries = [], {}, []
    for source in sources:
        source = Path(source)
        config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
        if hashlib.sha256(manifest).hexdigest() != config['questions_sha256']:
            raise ValueError('Question manifest identity mismatch')
        if configs:
            for key in COMPATIBLE_KEYS:
                if config[key] != configs[0][key]:
                    raise ValueError(f'Cannot combine runs with different {key}')
        configs.append(config)
        for mode in ['single', 'native4']:
            for rate in config['loss_rates']:
                if (mode, rate) in conditions:
                    raise ValueError(f'Duplicate condition: {mode}/{rate}')
                predictions, summary = audit_condition(
                    source / mode / f'loss_{rate:g}', labels, rate, mode, config)
                conditions[mode, rate] = predictions
                summaries.append(summary)
    for summary in summaries:
        mode, rate = summary['mode'], summary['nominal_loss_rate']
        baseline, current = conditions[mode, 0], conditions[mode, rate]
        for qid in labels:
            if (baseline[qid]['original_feature_sha256'] != current[qid]['original_feature_sha256']
                    or baseline[qid]['frames'] != current[qid]['frames']):
                raise ValueError(f'Visual inputs changed across runs: {qid}')
        summary.update(
            correct_to_wrong=sum(baseline[q]['correct'] and not current[q]['correct'] for q in labels),
            wrong_to_correct=sum(not baseline[q]['correct'] and current[q]['correct'] for q in labels),
            changed_answers=sum(baseline[q]['prediction'] != current[q]['prediction'] for q in labels))
    summaries.sort(key=lambda row: (row['mode'], row['nominal_loss_rate']))
    return conditions, summaries, configs


def draw_curves(summaries, output, focus_max, text_accuracy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'single': '#2364AA', 'native4': '#DA7C30'}
    names = {'single': 'Single native frame', 'native4': 'Four consecutive native frames'}
    all_rates = sorted({s['nominal_loss_rate'] for s in summaries})
    for full_range in [False, True]:
        figure, axis = plt.subplots(figsize=(8.4, 5.2), layout='constrained')
        for mode in ['single', 'native4']:
            rows = [s for s in summaries if s['mode'] == mode
                    and s['nominal_loss_rate'] <= focus_max]
            axis.plot([s['nominal_loss_rate'] * 100 for s in rows],
                      [s['accuracy_percent'] for s in rows], color=colors[mode],
                      marker='o' if mode == 'single' else 's', markersize=5,
                      linewidth=1.9, label=names[mode])
            if full_range:
                extra = [s for s in summaries if s['mode'] == mode
                         and s['nominal_loss_rate'] > focus_max]
                # Dashed spans distinguish the unsampled interval from the
                # denser measured interval; markers remain the actual data.
                for previous, current in zip([rows[-1]] + extra, extra):
                    axis.plot([previous['nominal_loss_rate'] * 100, current['nominal_loss_rate'] * 100],
                              [previous['accuracy_percent'], current['accuracy_percent']],
                              color=colors[mode], linestyle='--', linewidth=1.1)
                    axis.scatter(current['nominal_loss_rate'] * 100, current['accuracy_percent'],
                                 color=colors[mode], marker='o' if mode == 'single' else 's', s=40)
        if text_accuracy is not None:
            axis.axhline(text_accuracy, color='#737373', linestyle=':', linewidth=1.3,
                         label=f'Text-only control ({text_accuracy:g}%)')
        maximum = max(all_rates) * 100 if full_range else focus_max * 100
        axis.set(xlim=(0, maximum), ylim=(0, 100),
                 xlabel='Nominal feature-packet erasure rate (%)',
                 ylabel='Task success rate: correct / 125 (%)')
        axis.set_title('MiniCPM-o 2.6 / StreamingBench\n'
                       + ('All measured conditions, including all-zero control' if full_range
                          else f'Prespecified detail range: 0–{focus_max * 100:g}%'))
        axis.grid(alpha=.22)
        axis.legend(loc='lower left', fontsize=9)
        note = 'Independent packet erasure; missing F32 values set to zero.\n'
        note += ('Markers are measurements; dashed span is sparsely sampled.' if full_range
                 else 'Markers are measurements; lines only connect adjacent points.\n'
                      '100% all-zero control is shown in the companion full-range figure.')
        figure.text(.5, -.04, note, ha='center', va='top', fontsize=8, color='#555555')
        name = 'task_success_full_range' if full_range else 'task_success_0_50'
        for extension in ['png', 'pdf', 'svg']:
            path = output / f'{name}.{extension}'
            figure.savefig(path, dpi=200, bbox_inches='tight')
            if extension == 'svg':
                # Matplotlib path data contain trailing spaces; normalize
                # only line endings/whitespace for clean source publication.
                path.write_bytes(('\n'.join(line.rstrip() for line in
                                            path.read_text(encoding='utf-8').splitlines()) + '\n').encode('utf-8'))
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--focus-max', type=float, default=.5)
    parser.add_argument('--text-control', help='Optional verified text-only control summary.json')
    args = parser.parse_args()
    if args.focus_max != .5:
        parser.error('This prespecified follow-up figure uses a fixed 0–50% range')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    conditions, summaries, configs = audit_sources(args.sources, args.questions)
    expected_rates = [0, .001, .01, .05, .1, .2, .25, .3, .35, .4, .45, .5, 1]
    if set(conditions) != {(mode, rate) for mode in ['single', 'native4'] for rate in expected_rates}:
        raise ValueError('This follow-up report requires all 26 prespecified mode/rate conditions')
    control_accuracy = None
    if args.text_control:
        control = json.loads(Path(args.text_control).read_text(encoding='utf-8'))['text_only']
        if control['questions'] != 125:
            raise ValueError('Text-only control must have 125 questions')
        control_accuracy = control['accuracy'] * 100
    audit = {
        'status': 'passed', 'conditions': len(conditions), 'real_model_answers': len(conditions) * 125,
        'question_manifest_sha256': configs[0]['questions_sha256'],
        'cross_run_inputs': 'all original feature SHA-256 and frame metadata identical at each question/mode',
        'mask_audit': 'all masks regenerated; lost bytes and server feature receipt hashes checked',
        'primary_figure_range': [0, 50], 'primary_range_chosen_before_new_results': True,
        'full_range_control_retained': True, 'curve_fitting': False, 'monotonic_enforcement': False,
        'source_configs': configs, 'summary': summaries,
    }
    (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    with (output / 'condition_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    rates = sorted({rate for _, rate in conditions})
    lookup = {(s['mode'], s['nominal_loss_rate']): s for s in summaries}
    lines = [
        '# MiniCPM-o 2.6：20%–50% 丢包补测与任务成功率曲线', '',
        '日期：2026-10-10。新增 25%、30%、35%、40%、45%、50% 六个丢包率，'
        '每点单帧与连续四帧各 125 题，共新增 1500 次真实推理。',
        f'合并既有条件后共 {len(summaries)} 个完整条件、{len(summaries) * 125} 次回答。', '',
        '“任务成功率”定义为固定 125 道四选一问题的答题准确率；'
        '横轴是视觉特征包丢失率，不是比特错误率或解码错误率。', '',
        '## 0%–50% 主图', '',
        '![Task success, 0–50%](task_success_0_50.png)', '',
        '主图范围在新增结果产生前固定为 0%–50%，包含该区间全部实测点。'
        '没有按结果形状筛选点、强制单调或拟合“先平后陡”曲线。', '',
        '## 全范围对照', '',
        '![Task success, full range](task_success_full_range.png)', '',
        '100% 是已有的全部视觉特征置零对照。50%–100% 之间没有实测点，'
        '虚线只连接端点，不代表该区间形状或存在何处的平台。', '',
        '## 全部实测点', '',
        '| 丢包率 | 单帧答对/125 | 单帧成功率 | 四帧答对/125 | 四帧成功率 |',
        '|---:|---:|---:|---:|---:|',
    ]
    for rate in rates:
        one, four = lookup['single', rate], lookup['native4', rate]
        lines.append(f'| {rate * 100:g}% | {one["correct"]} | {one["accuracy_percent"]:.1f}% | '
                     f'{four["correct"]} | {four["accuracy_percent"]:.1f}% |')
    lines += ['', '## 实测变化', '']
    for mode, name in [('single', '单帧'), ('native4', '连续四帧')]:
        baseline = lookup[mode, 0]['accuracy_percent']
        middle = lookup[mode, .2]['accuracy_percent']
        end = lookup[mode, .5]['accuracy_percent']
        measured = [lookup[mode, r]['accuracy_percent'] for r in rates if r <= .5]
        lines.append(f'{name}：0%、20%、50% 丢包率的成功率分别为 '
                     f'{baseline:.1f}%、{middle:.1f}%、{end:.1f}%；'
                     f'整个 0%–50% 区间的实测值在 {min(measured):.1f}%–{max(measured):.1f}% 之间。')
    lines.append('曲线形状以这些实测点为准；上述分数没有经过平滑、单调化或按预期形状筛选。')
    lines.append('个别点高于无损基线只反映本轮逐题正确/错误变化，不能据此证明丢包提升能力。')
    lines += ['', '## 新增点的逐题变化', '',
              '| 输入 | 丢包率 | 正确变错误 | 错误变正确 | 答案改变 |',
              '|---|---:|---:|---:|---:|']
    for row in summaries:
        rate = row['nominal_loss_rate']
        if .2 < rate <= .5:
            name = '单帧' if row['mode'] == 'single' else '连续四帧'
            lines.append(f'| {name} | {rate * 100:g}% | {row["correct_to_wrong"]} | '
                         f'{row["wrong_to_correct"]} | {row["changed_answers"]} |')
    lines += ['', '## 核验与解释边界', '',
        '沿用 MTU 1500 字节建模预算，每包 1432 字节 F32 特征，'
        '缺包覆盖的完整特征位置置零。无重传、FEC 或前帧补填。',
        '模型、运行时、量化、提示词、随机种子、视频帧及输入特征哈希均与既有实验一致。'
        '新增运行重新编码，但逐题哈希确认特征字节与原运行相同。',
        '每题一个固定随机掩码，各率的缺包集合嵌套；全部原始响应、标签、缺包位置及哈希保存。'
        '独立重评分及随机掩码审计覆盖全部条件，无效响应计错，分母固定 125。',
        '每题仅一个种子、仅这 25 个视频，局部波动不代表统计显著或普遍趋势；'
        '没有保证曲线单调下降，也没有预设明显下降的拐点。',
        '纯文本对照为 56.0%，全零视觉特征为 58.4%。残余总分不能直接解释为视觉容错；'
        '仍应结合逐题正确变错误/错误变正确以及语言基线解读。',
        '本轮仍为离线特征擦除，经可靠本地 TCP 输入语言模型；不是原版 Testbed 的实际网卡网络 profile 测试。', '',
        '汇总及逐题变化见 condition_summary.csv；完整审计与各运行配置见 audit.json。'
        'PNG/PDF/SVG 均随报告发布。', '']
    (output / 'report_zh.md').write_text('\n'.join(lines), encoding='utf-8')
    draw_curves(summaries, output, args.focus_max, control_accuracy)
    print(json.dumps({'audit': 'passed', 'conditions': len(summaries), 'summary': summaries}, indent=2))


if __name__ == '__main__':
    main()
