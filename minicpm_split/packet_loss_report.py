"""Independently rescore saved answers and audit masks, hashes and completeness."""
import argparse
import csv
import hashlib
import json
import random
import re
import statistics
from pathlib import Path

from .packet_loss import OVERHEAD, packet_ranges


def audit_condition(path, labels, rate, mode, config):
    rows = [json.loads(line) for line in (path/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    by_id = {r['question_id']:r for r in rows}
    if len(rows)!=125 or len(by_id)!=125 or set(by_id)!=set(labels):
        raise ValueError(f'Incomplete or duplicate question set: {path}')
    correct = invalid = total_packets = lost_packets = zeroed_bytes = 0
    for row in rows:
        question_id = row['question_id']
        match = re.fullmatch(r'\s*\(?([ABCD])\)?[.)]?\s*',row['raw_response'])
        answer = match.group(1) if match else None
        expected_correct = answer == labels[question_id]['answer']
        if (row['label']!=labels[question_id]['answer'] or row['prediction']!=answer or
                row['correct']!=expected_correct or row['mode']!=mode):
            raise ValueError(f'Prediction/label mismatch: {question_id}')
        loss = row['loss']
        seed = int.from_bytes(hashlib.sha256(f'{config["seed"]}|{mode}|{question_id}'.encode()).digest()[:8],'big')
        rng = random.Random(seed)
        ranges = packet_ranges(row['feature_bytes'],config['mtu'])
        expected_lost = [index for index in range(len(ranges)) if rng.random()<rate]
        expected_zeroed = sum(ranges[index][1] for index in expected_lost)
        if (loss['seed']!=seed or loss['nominal_loss_rate']!=rate or
                loss['lost_packet_indices']!=expected_lost or loss['lost_packets']!=len(expected_lost) or
                loss['total_packets']!=len(ranges) or loss['zeroed_bytes']!=expected_zeroed or
                row['feature_sha256']!=row['damaged_feature_sha256']):
            raise ValueError(f'Packet mask/feature receipt mismatch: {question_id}')
        if rate==0 and row['original_feature_sha256']!=row['damaged_feature_sha256']:
            raise ValueError('Zero-loss condition changed the feature buffer')
        if rate==1 and row['damaged_feature_sha256']!=hashlib.sha256(bytes(row['feature_bytes'])).hexdigest():
            raise ValueError('All-loss control did not send an entirely zero buffer')
        if len(row['frames'])!=(1 if mode=='single' else 4):
            raise ValueError('Wrong frame count')
        indices = [f['frame_index'] for f in row['frames']]
        if len(indices)==4 and indices!=list(range(indices[0],indices[0]+4)):
            raise ValueError('Nonconsecutive source frames')
        if any(f['actual_seconds']>labels[question_id]['timestamp_seconds']+1e-9 for f in row['frames']):
            raise ValueError('Future frame detected')
        correct += expected_correct
        invalid += answer is None
        total_packets += len(ranges)
        lost_packets += len(expected_lost)
        zeroed_bytes += expected_zeroed
    return by_id, {'mode':mode,'nominal_loss_rate':rate,'evaluated':125,'correct':correct,
        'accuracy_percent':100*correct/125,'invalid_responses':invalid,
        'total_packets':total_packets,'lost_packets':lost_packets,
        'observed_packet_loss_percent':100*lost_packets/total_packets,
        'zeroed_feature_percent':100*zeroed_bytes/sum(r['feature_bytes'] for r in rows),
        'mean_server_inference_seconds':statistics.mean(r['server_inference_seconds'] for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results',required=True)
    parser.add_argument('--questions',required=True)
    parser.add_argument('--allzero-control')
    parser.add_argument('--split-baseline',help='Optional local split baseline directory with single/native4 subdirectories')
    args = parser.parse_args()
    root = Path(args.results)
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    questions = Path(args.questions).read_bytes()
    if hashlib.sha256(questions).hexdigest()!=config['questions_sha256']:
        raise ValueError('Question manifest identity mismatch')
    labels = {r['question_id']:r for r in json.loads(questions)}
    if len(labels)!=125:
        raise ValueError('Expected 125 fixed labels')
    conditions, summaries = {}, []
    sources = [(root,config)]
    if args.allzero_control:
        control = Path(args.allzero_control)
        control_config = json.loads((control/'config.json').read_text(encoding='utf-8'))
        for key in ['seed','mtu','questions_sha256','model_sha256','projector_sha256',
                    'tokenizer_sha256','runtime_dll_sha256','context','max_tokens','prompt_template']:
            if control_config[key]!=config[key]:
                raise ValueError(f'All-zero control differs in {key}')
        sources.append((control,control_config))
    for source,settings in sources:
        for mode in ['single','native4']:
            for rate in settings['loss_rates']:
                if (mode,rate) in conditions:
                    raise ValueError('Duplicate condition')
                by_id,summary = audit_condition(source/mode/f'loss_{rate:g}',labels,rate,mode,settings)
                conditions[mode,rate] = by_id
                summaries.append(summary)
    baseline_audit = []
    for summary in summaries:
        mode,rate = summary['mode'],summary['nominal_loss_rate']
        base,current = conditions[mode,0],conditions[mode,rate]
        summary.update(wrong_to_correct=sum(not base[q]['correct'] and current[q]['correct'] for q in labels),
                       correct_to_wrong=sum(base[q]['correct'] and not current[q]['correct'] for q in labels),
                       changed_answers=sum(base[q]['prediction']!=current[q]['prediction'] for q in labels),
                       changed_first_logits=sum(base[q]['result']['first_logits_sha256']!=
                                                current[q]['result']['first_logits_sha256'] for q in labels))
        for q in labels:
            if base[q]['original_feature_sha256']!=current[q]['original_feature_sha256'] or base[q]['frames']!=current[q]['frames']:
                raise ValueError('Inputs changed across loss rates')
    if args.split_baseline:
        for mode in ['single','native4']:
            previous = {r['question_id']:r for r in map(json.loads,(Path(args.split_baseline)/mode/'predictions.jsonl').read_text().splitlines())}
            for q in labels:
                actual = conditions[mode,0][q]
                old = previous[q]
                baseline_audit.append({'mode':mode,'question_id':q,
                    'answer_equal':actual['prediction']==old['prediction'],
                    'feature_bytes_equal':actual['original_feature_sha256']==old['feature_sha256'],
                    'first_logits_bitwise_equal':actual['result']['first_logits_sha256']==old['result']['first_logits_sha256']})
        if not all(all(row[key] for key in ['answer_equal','feature_bytes_equal','first_logits_bitwise_equal']) for row in baseline_audit):
            raise ValueError('Zero-loss pipeline differs from the verified split baseline')
    audit = {'status':'passed','conditions':len(summaries),'real_model_answers':len(summaries)*125,
             'fixed_manifest_questions':125,'mask_reproduction':'all packet masks and erased byte counts match',
             'server_feature_hashes':'all damaged-feature hashes match server receipt',
             'frame_selection':'all frames are past-only; four-frame indices consecutive',
             'zero_loss_split_parity':baseline_audit,'summary':summaries}
    (root/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    with (root/'condition_summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(summaries[0]))
        writer.writeheader();writer.writerows(summaries)
    flat = []
    for (mode,rate),by_id in conditions.items():
        for question_id,row in by_id.items():
            flat.append({'mode':mode,'loss_percent':rate*100,'question_id':question_id,'label':row['label'],
                         'prediction':row['prediction'],'raw_response':row['raw_response'],'correct':row['correct'],
                         'total_packets':row['loss']['total_packets'],'lost_packets':row['loss']['lost_packets'],
                         'zeroed_bytes':row['loss']['zeroed_bytes'],'seed':row['loss']['seed'],
                         'original_feature_sha256':row['original_feature_sha256'],
                         'damaged_feature_sha256':row['damaged_feature_sha256']})
    with (root/'question_results.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    rates = sorted({rate for _,rate in conditions})
    lookup = {(s['mode'],s['nominal_loss_rate']):s for s in summaries}
    lines = ['# MiniCPM-o 2.6 视觉特征分包丢包置零实验', '',
        '日期：2026-10-09。对前 25 个视频的 125 个问题，分别使用提问时刻单帧及该帧与紧邻的前三个原生帧。',
        f'所有 {len(summaries)} 个条件均完成 125 题，共 {len(flat)} 次真实模型回答。', '',
        '| 设定丢包率 | 单帧答对/125 | 单帧准确率 | 四帧答对/125 | 四帧准确率 |',
        '|---:|---:|---:|---:|---:|']
    for rate in rates:
        one,four = lookup['single',rate],lookup['native4',rate]
        lines.append(f'| {rate*100:g}% | {one["correct"]}/125 | {one["accuracy_percent"]:.2f}% | {four["correct"]}/125 | {four["accuracy_percent"]:.2f}% |')
    lines += ['', '## 结果解释', '']
    if .2 in rates:
        one,four = lookup['single',.2],lookup['native4',.2]
        lines.append(f'20% 分包丢失时，单帧准确率为 {one["accuracy_percent"]:.1f}%，'
            f'比无丢包下降 {lookup["single",0]["accuracy_percent"]-one["accuracy_percent"]:.1f} 个百分点；'
            f'四帧为 {four["accuracy_percent"]:.1f}%，下降 '
            f'{lookup["native4",0]["accuracy_percent"]-four["accuracy_percent"]:.1f} 个百分点。')
    if 1 in rates:
        lines.append(f'全部视觉特征置零后，单帧和四帧分别仍答对 '
            f'{lookup["single",1]["correct"]} 和 {lookup["native4",1]["correct"]} 题。'
            '这表明问题与选项的语言先验可能贡献相当部分准确率；'
            '不能把残余准确率都归因于视觉特征抗丢包能力。')
    lines += ['在这 125 题和本轮固定随机掩码下，0%–20% 丢包的准确率变化较小。'
        '1% 与 5% 等点位不必单调，需结合逐题变化解释；本轮没有多种子重复，不能宣称统计显著或普遍容错。', '',
        '![Accuracy versus packet loss](accuracy_vs_packet_loss.png)', '',
        '## 实验方法', '',
        '视觉编码器及 resampler/projector 在端侧生成连续 F32 特征，服务器运行 Q4_K_M 语言模型。',
        '特征在每题每种输入方式下只编码一次，各丢包率复用完全相同的原始特征。',
        '问题、图像切片标记、特征形状及位置不变，语言 KV 缓存每次清空；采用原有官方选择题提示词和贪心解码。', '',
        '**分包预算：**1500 字节是 IP MTU，不含以太网头；扣除 IPv4 20、UDP 8、RTP 12、特征分片头 16、',
        'SRTP 认证标签 10 字节后，向下对齐到完整 float32，得到每个完整包 1432 字节特征（358 个数值）。',
        '模型每帧产生 3×64×3584 个数值，即 2,752,512 字节；四帧 11,010,048 字节。',
        '按连续特征缓冲区分包，单帧 1923 包、四帧 7689 包。最后一个包允许不足 1432 字节。', '',
        '**丢包与填补：**每包独立 Bernoulli 丢失，丢失包覆盖的全部特征字节设为零，保持原始张量形状和位置。',
        '不填补上一帧、不重传、不使用 FEC。没有制造位翻转或不完整 float32，也没有直接丢弃整个图像。',
        '问题文本和控制元数据受到保护，不参与本次丢包。100% 丢包（若列于表中）是所有视觉特征置零对照，',
        '保留了图像标记与问题文本，不等同于删除图像 token 的纯文本提示。', '',
        '**可复现随机性：**基础种子 20261009，每个问题/输入方式通过 SHA-256 派生独立种子。',
        '不同丢包率使用相同的均匀随机数，高丢包率的缺失集合包含低丢包率集合。',
        '这是每题一个随机掩码的探索性实验，没有重复多个随机种子；小幅准确率差异不代表稳定增益。', '',
        '总准确率不变不代表模型内部输出没有变化；condition_summary.csv 另记录首步 logits 哈希发生变化的题数。', '',
        '## 实际丢包与答案变化', '',
        '| 输入 | 设定丢包率 | 实际丢包率 | 置零特征比例 | 正确变错误 | 错误变正确 | 答案发生变化 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for summary in sorted(summaries,key=lambda s:(s['mode'],s['nominal_loss_rate'])):
        s = summary
        lines.append(f'| {"单帧" if s["mode"]=="single" else "四帧"} | {s["nominal_loss_rate"]*100:g}% | {s["observed_packet_loss_percent"]:.4f}% | {s["zeroed_feature_percent"]:.4f}% | {s["correct_to_wrong"]} | {s["wrong_to_correct"]} | {s["changed_answers"]} |')
    lines += ['', '## 核验与结论边界', '',
        f'独立重评分验证了全部 {len(flat)} 条回答、固定标签、随机掩码、置零字节数和服务器收到的特征哈希。',
        '无效回答计为错误，所有完整条件准确率以 125 为分母；原始回答及逐题掩码均保留。',
        f'0% 条件与此前无丢包拆分基线逐题比较了 {len(baseline_audit)} 个样例的特征哈希、答案和首步完整 logits 哈希，结果见 audit.json。',
        '输入帧沿用已核验缓存，空间预处理为 588×336 JPEG，时间轴没有抽样；不得将结果称为原始分辨率输入。', '',
        '**本次是离线特征包擦除实验。**MTU 和协议开销是分片大小的建模假设，没有声称发送真实 SRTP 包。',
        '受损特征通过可靠本地 TCP 送入语言模型，以隔离特征缺失本身对准确率的影响。',
        '置零后的缓冲区长度保持不变，并非实际减少网络发送字节；这里的耗时不能用于评价丢包链路传输效率。',
        '本次没有真实网卡丢包、乱序、延迟、拥塞、RTC 抖动缓冲、重传或 FEC，因此不能当作 Testbed 网络 profile 实测。',
        '后续正式网络测试应沿用原版 Testbed/Liquid 的场景、信令、aiortc RTP、netem、抓包及日志流程。',
        '高丢包率下残余准确率还可能来自问题/选项的语言先验，不能仅据答题准确率断言视觉信息完全保留。', '',
        '## 复核', '',
        '逐题结果：question_results.csv；汇总：condition_summary.csv；完整审计：audit.json。',
        '各模式/loss_* 目录保存原始响应、标签、丢包位置、种子、特征哈希及逐条件汇总。',
        '模型与运行时的固定 SHA-256、提示词和 MTU 预算在 config.json 中。', '',
        '上游：[作者 GGUF](https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf)、',
        '[StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench)、',
        '[llama.cpp b11516](https://github.com/ggml-org/llama.cpp/releases/tag/b11516)。', '']
    (root/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes = plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
    for axis,selected,title in [(axes[0],[r for r in rates if r<=.2],'0–20% loss (detail)'),
                                (axes[1],rates,'Including all-zero visual features')]:
        for mode,label in [('single','Single native frame'),('native4','Four consecutive native frames')]:
            axis.plot([r*100 for r in selected],[lookup[mode,r]['accuracy_percent'] for r in selected],marker='o',label=label)
        axis.set(xlabel='Nominal feature packet loss (%)',ylabel='Accuracy on 125 questions (%)',title=title)
        axis.grid(alpha=.25)
    detail = [lookup[m,r]['accuracy_percent'] for m in ['single','native4'] for r in rates if r<=.2]
    axes[0].set_ylim(min(detail)-2,max(detail)+2)
    axes[1].set_ylim(0,100)
    axes[1].legend(loc='lower left',fontsize=8)
    figure.suptitle('MiniCPM-o 2.6: missing feature packets set to zero')
    figure.savefig(root/'accuracy_vs_packet_loss.png',dpi=180)
    figure.savefig(root/'accuracy_vs_packet_loss.pdf')
    plt.close(figure)
    print(json.dumps({'audit':'passed','conditions':len(summaries),'real_answers':len(flat),
                      'zero_loss_bitwise_parity':len(baseline_audit),'summary':summaries},indent=2))


if __name__=='__main__':
    main()
