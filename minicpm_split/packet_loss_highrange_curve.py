"""Audit the 50--100% follow-up and plot every measured erasure rate."""
import argparse
import csv
import json
from pathlib import Path

from .packet_loss_curve import audit_sources


def draw_curves(summaries, output, text_accuracy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for lower, name in [(50, 'task_success_50_100'), (0, 'task_success_full_range')]:
        figure, axis = plt.subplots(figsize=(9, 5.5), layout='constrained')
        for mode, label, color, marker in [
            ('single', 'Single native frame', '#2364AA', 'o'),
            ('native4', 'Four consecutive native frames', '#DA7C30', 's'),
        ]:
            rows = [r for r in summaries if r['mode'] == mode
                    and r['nominal_loss_rate'] * 100 >= lower]
            axis.plot([r['nominal_loss_rate'] * 100 for r in rows],
                      [r['accuracy_percent'] for r in rows],
                      label=label, color=color, marker=marker, markersize=5, linewidth=1.8)
        if text_accuracy is not None:
            axis.axhline(text_accuracy, color='#737373', linestyle=':', linewidth=1.3,
                         label=f'Text-only control ({text_accuracy:g}%)')
        axis.set(xlim=(lower, 100), ylim=(0, 100),
                 xlabel='Nominal feature-packet erasure rate (%)',
                 ylabel='Task success rate: correct / 125 (%)',
                 title=f'MiniCPM-o 2.6 / StreamingBench\nMeasured erasure rates: {lower}–100%')
        axis.set_xticks(range(lower, 101, 5 if lower else 10))
        axis.grid(alpha=.22)
        axis.legend(loc='lower left', fontsize=9)
        figure.text(.5, -.04,
                    '125 fixed questions per point; one nested random mask per question/mode.\n'
                    'Missing F32 features set to zero. Markers are measurements; lines connect points only.',
                    ha='center', va='top', fontsize=8, color='#555555')
        for extension in ['png', 'pdf', 'svg']:
            path = output / f'{name}.{extension}'
            figure.savefig(path, dpi=200, bbox_inches='tight')
            if extension == 'svg':
                path.write_bytes(('\n'.join(line.rstrip() for line in
                                           path.read_text(encoding='utf-8').splitlines()) + '\n').encode('utf-8'))
        plt.close(figure)


def adjacent_changes(conditions, summaries):
    changes = []
    for mode in ['single', 'native4']:
        rows = [r for r in summaries if r['mode'] == mode and r['nominal_loss_rate'] >= .5]
        for previous, current in zip(rows, rows[1:]):
            start, end = previous['nominal_loss_rate'], current['nominal_loss_rate']
            before, after = conditions[mode, start], conditions[mode, end]
            decline = previous['accuracy_percent'] - current['accuracy_percent']
            changes.append({
                'mode': mode, 'from_loss_percent': start * 100, 'to_loss_percent': end * 100,
                'before_accuracy_percent': previous['accuracy_percent'],
                'after_accuracy_percent': current['accuracy_percent'],
                'decline_percentage_points': round(decline, 8),
                'decline_pp_per_10_loss_points': round(decline / ((end - start) * 100) * 10, 8),
                'correct_to_wrong': sum(before[q]['correct'] and not after[q]['correct'] for q in before),
                'wrong_to_correct': sum(not before[q]['correct'] and after[q]['correct'] for q in before),
                'changed_answers': sum(before[q]['prediction'] != after[q]['prediction'] for q in before),
            })
    return changes


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--text-control', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    conditions, summaries, configs = audit_sources(args.sources, args.questions)
    expected_rates = [0, .001, .01, .05, .1, .2, .25, .3, .35, .4, .45, .5,
                      .55, .6, .65, .7, .75, .8, .85, .9, .95, .99, 1]
    if set(conditions) != {(m, r) for m in ['single', 'native4'] for r in expected_rates}:
        raise ValueError('Expected all 46 complete mode/rate conditions')
    control = json.loads(Path(args.text_control).read_text(encoding='utf-8'))['text_only']
    if control['questions'] != 125:
        raise ValueError('Text control must contain 125 questions')
    text_accuracy = control['accuracy'] * 100
    changes = adjacent_changes(conditions, summaries)
    audit = {
        'status': 'passed', 'conditions': len(conditions), 'real_model_answers': len(conditions) * 125,
        'new_model_answers': 2500, 'question_manifest_sha256': configs[0]['questions_sha256'],
        'cross_run_inputs': 'all original feature SHA-256 and frame metadata identical at each question/mode',
        'mask_audit': 'all masks regenerated; lost bytes and server feature receipt hashes checked',
        'primary_figure_range': [50, 100], 'full_range_control_retained': True,
        'curve_fitting': False, 'monotonic_enforcement': False,
        'source_configs': configs, 'summary': summaries, 'adjacent_changes': changes,
    }
    (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    write_csv(output / 'condition_summary.csv', summaries)
    write_csv(output / 'adjacent_changes.csv', changes)
    lookup = {(r['mode'], r['nominal_loss_rate']): r for r in summaries}
    lines = [
        '# MiniCPM-o 2.6：50%–100% 特征丢包补测', '',
        '日期：2026-10-10。新增 55%、60%、65%、70%、75%、80%、85%、90%、95%、99% 十个点，'
        '单帧与连续四帧各 125 题，共新增 2500 次真实模型推理。'
        '与既有结果合并为 23 个丢包率、46 个条件、5750 次回答；另有 375 次独立对照不计入该数字。', '',
        '任务成功率指固定 125 道四选一问题的准确率。横轴是视觉特征包擦除概率，'
        '不是比特错误率；所有缺包对应的 F32 特征置零。', '',
        '## 50%–100% 实测曲线', '', '![50–100%](task_success_50_100.png)', '',
        '所有新增点和既有 50%、100% 端点均展示；连线仅连接实测点，不是拟合。', '',
        '## 下降集中在哪些相邻采样区间', '',
    ]
    for mode, name in [('single', '单帧'), ('native4', '连续四帧')]:
        rows = [r for r in changes if r['mode'] == mode]
        maximum = max(r['decline_percentage_points'] for r in rows)
        largest = [r for r in rows if r['decline_percentage_points'] == maximum]
        descriptions = [f'{r["from_loss_percent"]:g}%→{r["to_loss_percent"]:g}%：'
                        f'{r["before_accuracy_percent"]:.1f}%→{r["after_accuracy_percent"]:.1f}%' for r in largest]
        lines.append(f'{name}最大的相邻点绝对下降为 {maximum:.1f} 个百分点，发生于 '
                     + '；'.join(descriptions) + '。')
        begin, end = lookup[mode, .5], lookup[mode, 1]
        lines.append(f'{name}从 50% 到 100% 丢包，准确率从 {begin["accuracy_percent"]:.1f}% '
                     f'变为 {end["accuracy_percent"]:.1f}%，净下降 '
                     f'{begin["accuracy_percent"] - end["accuracy_percent"]:.1f} 个百分点。')
    lines += [
        '', '相邻区间宽度不全相同：95%→99% 为 4 个百分点，99%→100% 为 1 个百分点。'
        '下表同时列出实际绝对变化及按丢包率每增加 10 个百分点归一化的斜率。'
        '后者只是两点差商，不代表未测区间的真实形状或精确拐点。', '',
        '| 输入 | 丢包率区间 | 准确率下降（百分点） | 每增加 10 点丢包的下降 | 正确→错误 | 错误→正确 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for r in changes:
        name = '单帧' if r['mode'] == 'single' else '四帧'
        lines.append(f'| {name} | {r["from_loss_percent"]:g}%–{r["to_loss_percent"]:g}% | '
                     f'{r["decline_percentage_points"]:.1f} | {r["decline_pp_per_10_loss_points"]:.1f} | '
                     f'{r["correct_to_wrong"]} | {r["wrong_to_correct"]} |')
    lines += ['', '## 全范围与全部数据', '',
              '![All measured rates](task_success_full_range.png)', '',
              '| 丢包率 | 单帧答对/125 | 单帧准确率 | 四帧答对/125 | 四帧准确率 |',
              '|---:|---:|---:|---:|---:|']
    for rate in expected_rates:
        one, four = lookup['single', rate], lookup['native4', rate]
        lines.append(f'| {rate * 100:g}% | {one["correct"]} | {one["accuracy_percent"]:.1f}% | '
                     f'{four["correct"]} | {four["accuracy_percent"]:.1f}% |')
    lines += ['', '## 方法与解释边界', '',
        '沿用 Q4_K_M 语言模型、F16 视觉模型、llama.cpp b11516、固定提示词及贪心解码。'
        '视频 1–25，每视频 5 问；单帧取提问前最近原生帧，四帧取该帧及之前连续三帧。',
        'IP MTU 1500 字节预算：IPv4/UDP/RTP/自定义头/SRTP 标签分别为 20/8/12/16/10 字节，'
        '按 float32 对齐后每完整包载荷 1432 字节；单帧 1923 包、四帧 7689 包。'
        '各包独立 Bernoulli 擦除，无 FEC、重传或历史特征补填。问题与元数据不擦除。',
        '基础种子 20261009，逐题/输入方式派生固定种子，各丢包率掩码嵌套。'
        '新旧实验逐题原始特征 SHA-256 与帧元数据完全一致；全部 5750 条原始响应独立重评分，'
        '缺包掩码、置零字节数和服务器接收特征哈希通过核验。无效响应计错，分母固定 125。',
        f'全部条件共 {sum(r["invalid_responses"] for r in summaries)} 个无效响应；'
        '逐条件数量保存在 condition_summary.csv 的 invalid_responses 列，未删除或补跑。',
        f'纯文本对照为 {text_accuracy:.1f}%，全零视觉为 58.4%。语言先验/选项线索可能支撑残余得分；'
        '不能将全零得分解释为视觉容错，也不能仅凭该实验判断训练集重叠。',
        '每题仅一个固定随机掩码，仅 25 个视频。相邻点下降是此轮实测差异，'
        '没有多种子置信区间或统计显著性结论；未强制单调、平滑、筛选题目或隐藏平台。',
        '本轮为离线特征擦除，通过可靠本地 TCP 发送受损全长张量；'
        '尚不是原版 Testbed 的网卡、RTC 与 network profile 测试。', '',
        'condition_summary.csv 为各点汇总，adjacent_changes.csv 为相邻点变化，'
        'audit.json 保存完整核验与配置；图表同时提供 PNG、PDF、SVG。', '']
    (output / 'report_zh.md').write_text('\n'.join(lines), encoding='utf-8')
    draw_curves(summaries, output, text_accuracy)
    print(json.dumps({'audit': 'passed', 'conditions': len(conditions),
                      'real_model_answers': len(conditions) * 125, 'adjacent_changes': changes}, indent=2))


if __name__ == '__main__':
    main()
