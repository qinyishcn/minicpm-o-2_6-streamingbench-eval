"""Build a descriptive report/plots from independently audited RTC results.

Requires matplotlib. No inference or network configuration is performed here.
Before this script, run scripts/audit_testbed_profiles.py --reencode with
the public Testbed patch on PYTHONPATH.
"""
import argparse
import csv
import json
import re
from pathlib import Path

PROFILES = ['no_emulation', 'lossy', '5qi_7', 'congested']
MODES = ['single', 'native4']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prediction(raw):
    match = re.fullmatch(r'\s*\(?([ABCD])\)?[.)]?\s*', raw) if isinstance(raw, str) else None
    return match.group(1) if match else None


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    audit = read(root / 'audit/audit.json')
    if (audit['status'] != 'passed' or audit['question_slots'] != 1000
            or not audit['reencoded_feature_audit']
            or not all(r['zero_fill_hash_reconstructed'] for r in audit['question_checks'] if r['answer_received'])):
        raise ValueError('Expected all 1000 question slots scored and all saved input hashes re-encoded')
    summary = {(r['mode'], r['profile']): r for r in audit['summary']}
    network = read(root / 'network_evidence.json')
    reused_baseline = (root / 'baseline_reuse.json').exists()
    if set(summary) != {(mode, profile) for mode in MODES for profile in PROFILES}:
        raise ValueError('Expected exactly eight complete conditions')
    output.mkdir(parents=True, exist_ok=True)
    paired, videos, transitions, positions = [], [], [], []
    for mode in MODES:
        baseline = {}
        for profile in PROFILES:
            sessions = {r['video_id']: r for r in network[f'{mode}/{profile}']['original_video_sessions']}
            rows = []
            for video in range(1, 26):
                path = root / 'dataset' / mode / f'sample_{video}' / f'questions_results_{profile}.json'
                available = read(path)[0]['questions'] if path.exists() else []
                if profile == 'no_emulation':
                    subset = available
                else:
                    indexed = {r['question_id']: r for r in available}
                    subset = [indexed.get(r['question_id'], {k: v for k, v in r.items()
                        if k not in ['minicpm', 'minicpm_details']})
                        for r in baseline.values() if r['video_id'] == video]
                if len(subset) != 5:
                    raise ValueError('Expected five fixed question slots per video')
                videos.append({'mode': mode, 'profile': profile, 'video_id': video,
                    'correct': sum(prediction(r.get('minicpm_details', {}).get('raw_response', '')) == r['answer'] for r in subset),
                    'answers_saved': sum('minicpm_details' in r for r in subset),
                    'missing_answers': sum('minicpm_details' not in r for r in subset),
                    'client_session_success': sessions[video]['client_success'],
                    'client_answers_received': sessions[video].get('client_answers_received', ''),
                    'questions': 5})
                rows.extend(subset)
            if profile == 'no_emulation':
                baseline = {r['question_id']: r for r in rows}
            gained = lost = 0
            zeroed_by_quarter = [0, 0, 0, 0]
            bytes_by_quarter = [0, 0, 0, 0]
            for row in rows:
                detail = row.get('minicpm_details')
                original = baseline[row['question_id']]
                actual = prediction(detail['raw_response']) if detail else None
                before = prediction(original['minicpm_details']['raw_response'])
                correct, before_correct = actual == row['answer'], before == row['answer']
                gained += correct and not before_correct
                lost += before_correct and not correct
                common = {'mode': mode, 'profile': profile, 'question_id': row['question_id'],
                    'video_id': row['video_id'], 'label': row['answer'],
                    'prediction': actual or '', 'raw_response': detail['raw_response'] if detail else '',
                    'answer_saved': bool(detail), 'correct': correct,
                    'baseline_prediction': before, 'baseline_correct': before_correct,
                    'answer_changed': actual != before}
                if detail is None:
                    paired.append({**common, **{key: '' for key in ['feature_packets',
                        'missing_feature_packets', 'zeroed_feature_percent', 'original_feature_sha256',
                        'received_feature_sha256', 'edge_encode_seconds',
                        'server_inference_seconds_including_queue', 'server_metadata_to_result_seconds']}})
                    continue
                length = detail['feature_bytes']
                boundaries = [length * index // 4 for index in range(5)]
                for quarter in range(4):
                    begin, end = boundaries[quarter:quarter + 2]
                    bytes_by_quarter[quarter] += end - begin
                size = detail['loss']['feature_payload_per_packet']
                for index in detail['loss']['missing_packet_indices']:
                    begin, end = index * size, min((index + 1) * size, length)
                    for quarter in range(4):
                        zeroed_by_quarter[quarter] += max(0, min(end, boundaries[quarter + 1])
                                                         - max(begin, boundaries[quarter]))
                paired.append({**common, 'feature_packets': detail['loss']['total_packets'],
                    'missing_feature_packets': detail['loss']['missing_packets'],
                    'zeroed_feature_percent': 100 * detail['loss']['zeroed_feature_fraction'],
                    'original_feature_sha256': detail['original_feature_sha256'],
                    'received_feature_sha256': detail['feature_sha256'],
                    'edge_encode_seconds': detail['edge_encode_seconds'],
                    'server_inference_seconds_including_queue': detail['server_inference_seconds'],
                    'server_metadata_to_result_seconds': detail['metadata_to_result_seconds']})
            condition = summary[mode, profile]
            if sum(zeroed_by_quarter) != condition['zeroed_feature_bytes']:
                raise ValueError('Feature-position erasure totals disagree with independent audit')
            positions.append({'mode': mode, 'profile': profile, **{
                f'buffer_bytes_{quarter * 25}_{(quarter + 1) * 25}_percent_zeroed':
                100 * zeroed_by_quarter[quarter] / bytes_by_quarter[quarter] if bytes_by_quarter[quarter] else None
                for quarter in range(4)}})
            baseline_score = summary[mode, 'no_emulation']['correct']
            if gained - lost != condition['correct'] - baseline_score:
                raise ValueError('Paired score delta disagrees with independent audit')
            transitions.append({'mode': mode, 'profile': profile, 'wrong_to_correct': gained,
                'correct_to_wrong': lost, 'net_correct_change': gained - lost,
                'accuracy_delta_percentage_points': 100 * (gained - lost) / 125})
    write_csv(output / 'paired_predictions.csv', paired)
    write_csv(output / 'per_video.csv', videos)
    write_csv(output / 'paired_changes.csv', transitions)
    write_csv(output / 'feature_position_erasure.csv', positions)
    completion = []
    for mode in MODES:
        for profile in PROFILES:
            metrics = read(root / 'network' / mode / profile / 'testbed_report.json')[
                'scenarios']['realtime_video_understanding'][profile]
            if (metrics['samples'] != 25 or metrics['task_accuracy']['real_total'] != 125
                    or metrics['task_accuracy']['real_correct'] != summary[mode, profile]['correct']):
                raise ValueError('Independent scores disagree with original Testbed results')
            rate = metrics['qoe_metrics']['success_rate_pct']
            sessions = network[f'{mode}/{profile}']['original_video_sessions']
            if sum(r['client_success'] for r in sessions) != round(metrics['samples'] * rate / 100):
                raise ValueError('Client completion logs disagree with original Testbed aggregate')
            client_answers = (sum(r['client_answers_received'] for r in sessions)
                if all('client_answers_received' in r for r in sessions) else None)
            completion.append({'mode': mode, 'profile': profile,
                'video_sessions': metrics['samples'], 'client_complete_video_percent': rate,
                'client_complete_videos': round(metrics['samples'] * rate / 100),
                'original_video_response_latency_seconds': metrics['qoe_metrics']['response_latency_ms'] / 1000,
                'client_answers_received': client_answers,
                'server_answer_records': summary[mode, profile]['answers_received'],
                'missing_server_answers': summary[mode, profile]['missing_answers']})
    write_csv(output / 'client_completion.csv', completion)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(len(PROFILES))
    colors = {'single': '#2563eb', 'native4': '#ea580c'}
    labels = {'single': 'Single native frame', 'native4': 'Four consecutive native frames'}
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), constrained_layout=True)
    for mode, offset in [('single', -.19), ('native4', .19)]:
        accuracy = [summary[mode, profile]['accuracy_percent'] for profile in PROFILES]
        loss = [summary[mode, profile]['missing_feature_percent'] for profile in PROFILES]
        for ax, values in zip(axes, [accuracy, loss]):
            bars = ax.bar(x + offset, values, .36, color=colors[mode], label=labels[mode])
            ax.bar_label(bars, labels=[f'{value:.1f}%' for value in values], padding=3, fontsize=9)
    for ax in axes:
        ax.set_ylim(0, 110)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xticks(x, PROFILES)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_title('MiniCPM-o 2.6 / StreamingBench: original Testbed RTC profiles')
    axes[0].set_ylabel('Task accuracy (%)')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[1].set_ylabel('Feature loss in saved answer records (%)')
    axes[1].set_xlabel('Original network profile (categorical, not a loss-rate scale)')
    fig.get_layout_engine().set(rect=(0, .055, 1, .945))
    fig.text(.5, .012, '125 questions per condition; one trial per question; selected query-frame accelerated transfer.\n'
        'Missing answers count wrong; feature-loss statistics exclude unrecorded inputs.',
        ha='center', fontsize=9)
    fig.savefig(output / 'network_profile_comparison.png', dpi=220)
    fig.savefig(output / 'network_profile_comparison.pdf')
    plt.close(fig)

    table = ['| 原版 profile | 单帧答对/125 | 单帧准确率 | 四帧答对/125 | 四帧准确率 |',
             '|---|---:|---:|---:|---:|']
    for profile in PROFILES:
        single, four = summary['single', profile], summary['native4', profile]
        table.append(f'| {profile} | {single["correct"]}/125 | {single["accuracy_percent"]:.1f}% | '
                     f'{four["correct"]}/125 | {four["accuracy_percent"]:.1f}% |')
    comparison = []
    for profile in PROFILES[1:]:
        changes = [summary[mode, profile]['accuracy_percent']
                   - summary[mode, 'no_emulation']['accuracy_percent'] for mode in MODES]
        comparison.append(f'- `{profile}` 相对无扰动：单帧{changes[0]:+.1f}个百分点，'
                          f'四帧{changes[1]:+.1f}个百分点。')
    lines = ['# MiniCPM-o 2.6：原版 Testbed 完整 network profile 测试', '',
        f'固定视频 1–25，每视频 5 题。两种输入 × 四种网络条件，每条件 125 题，共1000个题位。'
        f'已保存服务器答案{audit["answers"]}条，缺失{audit["missing_answers"]}条按错误计分。'
        + ('本轮新增750个题位的真实RTC测试，另复用此前250题无扰动基线并逐文件核验SHA-256。'
           if reused_baseline else '此结果根目录未声明复用此前基线；全部条件从该目录逐题记录汇总。'), '',
        *table, '', '![准确率与实际特征分片缺失](network_profile_comparison.png)', '',
        *comparison, '',
        '以上是本次固定125题、每条件一次运行的描述性结果。微小的上升或下降不能证明丢包带来收益，'
        '也不能证明存在统计显著差异。拥塞条件中的答案缺失与格式无效同时计入主表，'
        '不能将准确率变化全部归因于视觉特征置零。', '',
        '## 条件与计分', '',
        '单帧为提问前最近原生帧，四帧为该帧与紧邻前三帧。复用相同已核验的帧缓存，'
        '无时间抽样或FPS变换；四帧覆盖的时间窗为3/fps秒。'
        '每题独立清空语言模型KV，标准答案仅用于评分。'
        'Q4_K_M 语言模型 / F16 视觉编码器及 resampler，llama.cpp b11516 CUDA 12.8，'
        '本机 WSL Ubuntu / RTX 3070 8GB。客户端和服务器是两个进程，共享 GPU。', '',
        '执行公开源码补丁对应的干净 Testbed 副本；保持原版 orchestrator、RTC、'
        'UDP/RTP/SRTP、netemu、抓包、SQLite 以及每视频 20 秒间隔。'
        'TCP/18124 仅信令，可靠 DataChannel 传问题/特征布局/答案。'
        'lo 只做 egress shaping，回环上两个方向各经过一次原队列。', '',
        '**当前按提问点加速，仅传指定单帧/四帧特征；不是连续原生 FPS 流量或原视频时间轴回放。**', '',
        '主表按保存的服务器推理答案计分，每条件分母固定125，无效或缺失答案计错。'
        '服务器生成答案与客户端成功收齐答案是不同事件，客户端完成率另见下表。'
        '使用与此前相同的严格 A–D 响应格式，'
        '未删除错误、选择有利重复运行、强制结果单调或隐藏语言先验基线。'
        '每视频每条件运行一次，不重跑补齐缺失答案；netem 未另设固定种子，'
        '未做多次重复或显著性检验。', '',
        '## 原版网络配置与实际特征缺失', '',
        '| profile | 配置摘要 |', '|---|---|',
        '| no_emulation | 无 tc/netem 扰动 |',
        '| lossy | 随机丢包 10%，无时延和带宽限制 |',
        '| 5qi_7 | 80ms 延迟、10ms jitter，配置丢包 0.1%，相关 20%，队列 1000 包；原重排保留 |',
        '| congested | 200ms 延迟、50ms jitter，gemodel 3% 设定，相关 40%，1Mbps、队列 200 包；原重排/重复保留 |', '',
        '以上值是原 YAML 参数，不是实测模型缺失率；完整配置见 `configs/original_profiles.yaml`。'
        '随机丢包和队列溢出、重组丢弃、迟到/截止舍弃共同决定模型可用特征。', '',
        '| 输入 | profile | 特征统计记录数 | 缺失分片/总数 | 不可用特征包 | 特征置零字节比例 | 无效/缺失回答 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for mode in MODES:
        for profile in PROFILES:
            row = summary[mode, profile]
            lines.append(f'| {mode} | {profile} | {row["answers_received"]}/125 | {row["missing_feature_packets"]}/{row["feature_packets"]} | '
                f'{row["missing_feature_percent"]:.3f}% | {row["zeroed_feature_percent"]:.3f}% | '
                f'{row["invalid_responses"]}/{row["missing_answers"]} |')
    lines += ['', '**特征缺失、置零位置和推理计时仅统计有服务器答案及特征记录的题目。**'
        '缺失答案对应的实际模型输入未知，没有推定为全零，也没有放入特征丢包率分母。'
        '因此这些特征统计包含记录可用性的选择效应，不能代表所有125题的完整链路丢包率。', '',
        '### 抓包与队列观测', '',
        '| 输入 | profile | 抓到的特征消息组/125 | qdisc最大观测drops（分层列示） | RTP最大IP包长 |',
        '|---|---|---:|---|---:|']
    for mode in MODES:
        for profile in PROFILES:
            row = network[f'{mode}/{profile}']
            drops = ', '.join(f'{kind}={value}' for kind, value in
                row['qdisc_max_observed_drops'].items() if kind != 'noqueue') or 'noqueue=0'
            lines.append(f'| {mode} | {profile} | {row["rtp_message_groups"]}/125 | '
                f'{drops} | {row["maximum_captured_rtp_ip_bytes"]} B |')
    lines += ['', 'qdisc值为执行期间的最大观测计数，父子队列不能相加；抓包可见不等于模型收到。'
        '队列丢弃计数也不能直接当作模型特征缺失比例。分片缺失以接收端保存的实际输入为准，'
        '控制消息和媒体同样经过网卡扰动。'
        f'抓包过滤排除了{sum(r["dns_packets_matching_rtp_prefix_excluded"] for r in network.values())}'
        '个字节前缀偶合RTP的DNS包（UDP/53）；原始捕获保留，不影响模型输入或答题结果。', '',
        '每包 F32 特征 1080 字节；单帧 2549 包、四帧 10195 包。'
        '保留 Liquid 的 1100 字节 RTP 载荷上限，媒体 IP 包小于 MTU 1500；'
        '缺包按原字节位置置零，不从其他问题补特征。RTX 不协商，实际 fork 的 RTX 正/负控制已验证；'
        'RTCP/NACK 保留，未添加可靠媒体重传或 FEC。', '',
        'MiniCPM codec 固定传 F32 特征，未根据 target_bitrate 做额外量化或特征删减；'
        '原版传输和反馈路径保留，但此结果只覆盖该固定特征负载适配。', '',
        '单帧每题2,752,512字节（约2.75MB），四帧11,010,048字节（约11.01MB）；'
        '四帧每问的F32特征字节量恰为单帧的4倍。在相同原版带宽和队列预算下，'
        '输入增加的视觉信息和增加的传输负担会共同影响准确率。', '',
        '这个 RTC 实验与此前 1432 字节分片的离线 IID 擦除实验不同：实际网络的突发、有限队列、'
        '重排及控制消息也参与链路行为。相同“特征缺失比例”不代表相同缺失位置或同一个损伤分布。'
        '总正确率中的残余语言先验不能直接解释为视觉容错；此前 Windows 纯文本/全零对照只作背景，'
        '不能冒充这里的 Linux 网络条件。', '',
        '### 缺失位置分布', '',
        '将每题原始 F32 特征缓冲按字节位置等分为四段，逐个缺包区间计算交集。'
        '表中是各段内部的置零比例，用于观察前缀/尾部缺失差异；这些区间不是图像空间区域。', '',
        '| 输入 | profile | 0–25% 字节段 | 25–50% | 50–75% | 75–100% |',
        '|---|---|---:|---:|---:|---:|']
    for row in positions:
        values = [row[f'buffer_bytes_{q * 25}_{(q + 1) * 25}_percent_zeroed'] for q in range(4)]
        lines.append(f'| {row["mode"]} | {row["profile"]} | ' +
                     ' | '.join(f'{value:.2f}%' for value in values) + ' |')
    lines += ['',
        '## 相对无扰动的逐题变化', '',
        '| 输入 | profile | 错→对 | 对→错 | 准确率变化（百分点） |',
        '|---|---|---:|---:|---:|']
    for row in transitions:
        lines.append(f'| {row["mode"]} | {row["profile"]} | {row["wrong_to_correct"]} | '
            f'{row["correct_to_wrong"]} | {row["accuracy_delta_percentage_points"]:+.1f} |')
    lines += ['', '## 原版客户端完成率', '',
        '原版客户端以每视频收齐5条回答判断成功。服务器已保存答案不保证答案已回传到客户端；'
        '下表沿用原版视频级完成率，与主表的服务器推理准确率分别报告。'
        '受损链路上的控制/完成消息超时及缺失回答均保留，没有延长超时或重跑替换失败视频。', '',
        '| 输入 | profile | 客户端收齐5条回答的视频数 | 完成率 | 原版视频级response_latency | 客户端答案接收数 | 已保存服务器答案 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in completion:
        lines.append(f'| {row["mode"]} | {row["profile"]} | {row["client_complete_videos"]}/'
            f'{row["video_sessions"]} | {row["client_complete_video_percent"]:.1f}% | '
            f'{row["original_video_response_latency_seconds"]:.3f}s | '
            f'{row["client_answers_received"] if row["client_answers_received"] is not None else "未知"}/125 | '
            f'{row["server_answer_records"]}/125 |')
    lines += ['', '## 核验与计时边界', '',
        '1000个固定题位独立重评分，125个问题身份、选项和标准label固定；有记录的跨profile原生帧索引、'
        '每帧 SHA-256 与原始特征 SHA-256 均一致。250 组输入（125 题 × 两种方式）重新编码，'
        f'按各条件记录的缺包索引重建实际模型输入，{audit["answers"]}条已保存的原始/受损特征字节哈希全部通过；'
        f'{audit["missing_answers"]}条无记录的输入没有哈希核验结论。'
        '此前完整 250 题无扰动的答案、生成 token IDs 与首步完整 logits 哈希，'
        '已与同 Linux 运行时独立未拆分 helper 核对一致。', '',
        '| 输入 | profile | 平均端侧编码 | 平均语言推理（含排队） | 服务器元数据登记至完整答案 |',
        '|---|---|---:|---:|---:|']
    for mode in MODES:
        for profile in PROFILES:
            row = summary[mode, profile]
            lines.append(f'| {mode} | {profile} | {row["mean_edge_encode_seconds"]:.3f}s | '
                f'{row["mean_server_inference_seconds"]:.3f}s | {row["mean_metadata_to_result_seconds"]:.3f}s |')
    lines += ['', '上述计时不是端到端延迟：服务器计时不含端侧编码或答案回传，语言模型调用含共享 GPU '
        '串行等待。答案一次性返回，未测逐 token TTFT。原版视频级 report 的 TTFT/TTLT/stall '
        '字段保留原名，但这里是五个独立问题的整条答案事件，不能当成 token 级时延或卡顿。', '',
        '[完整汇总 CSV](audit/condition_summary.csv) · [1000 题逐题对照](paired_predictions.csv) · '
        '[按视频汇总](per_video.csv) · [相对基线变化](paired_changes.csv) · '
        '[客户端完成率](client_completion.csv) · '
        '[逐项特征审计](audit/audit.json) · [网络证据](network_evidence.json) · '
        '[源码与复现说明](../../docs/testbed_rtc.md)', '',
        '原始回答在 `dataset/`，原版网络配置、实际 CLI、source patch 身份、基线复用哈希与'
        '网络快照一并保留。PCAP/SQLite 的大小和 SHA-256 在网络摘要，原文件留在本机，'
        '不上传模型、视频、二进制或含短期 ICE 信令的原抓包。', '']
    (output / 'report_zh.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'conditions': 8, 'question_slots': len(paired), 'answers': audit['answers'],
        'summary': list(summary.values())}, indent=2))


if __name__ == '__main__':
    main()
