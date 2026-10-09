"""Paired accuracy and runtime comparison for the two user-requested protocols."""
import collections
import csv
import json
from pathlib import Path

from evaluate import ROOT


def main():
    folders=[ROOT/'results/q4_k_m_single_native',ROOT/'results/q4_k_m_four_native']
    summaries=[json.loads((f/'summary.json').read_text(encoding='utf-8')) for f in folders]
    assert all(s['status']=='complete' and s['evaluated']==125 for s in summaries)
    predictions=[{r['question_id']:r for r in [json.loads(l) for l in (f/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]} for f in folders]
    assert predictions[0].keys()==predictions[1].keys()
    paired=[];transitions=collections.Counter();spans=[];fps=set()
    for qid,single in predictions[0].items():
        four=predictions[1][qid]
        assert single['frames'][0]['sha256']==four['frames'][-1]['sha256']
        assert single['frames'][0]['frame_index']==four['frames'][-1]['frame_index']
        assert len(single['frames'])==1 and len(four['frames'])==4
        indices=[x['frame_index'] for x in four['frames']]
        assert indices==list(range(indices[-1]-3,indices[-1]+1))
        assert all(x['actual_seconds']<=four['timestamp_seconds']+1e-8 for x in four['frames'])
        transitions[(single['correct'],four['correct'])]+=1
        spans.append(four['frames'][-1]['actual_seconds']-four['frames'][0]['actual_seconds'])
        fps.add(four['frames'][0]['source_fps'])
        paired.append(dict(question_id=qid,video_id=single['video_id'],task_type=single['task_type'],
                           time_stamp=single['time_stamp'],label=single['answer'],
                           single_prediction=single['prediction'],four_prediction=four['prediction'],
                           single_correct=int(single['correct']),four_correct=int(four['correct']),
                           single_seconds=single['request_to_response_s'],four_seconds=four['request_to_response_s'],
                           source_fps=four['frames'][0]['source_fps'],
                           native_frame_indices=json.dumps(indices),
                           native_frame_pts=json.dumps([x['actual_seconds'] for x in four['frames']])))
    audit=json.loads((folders[1]/'frame_audit.json').read_text(encoding='utf-8'))
    assert audit['passed'] and audit['question_count']==125 and audit['total_question_frames']==500
    output=ROOT/'results/native_comparison';output.mkdir(exist_ok=True)
    delta=round(summaries[1]['accuracy_percent']-summaries[0]['accuracy_percent'],10)
    result=dict(status='complete',questions_per_mode=125,single=summaries[0],four_consecutive=summaries[1],
                accuracy_delta_percentage_points=delta,
                four_only_correct=transitions[(False,True)],single_only_correct=transitions[(True,False)],
                both_correct=transitions[(True,True)],both_wrong=transitions[(False,False)],
                four_frame_span_seconds_min=min(spans),four_frame_span_seconds_max=max(spans),
                source_fps=sorted(fps),frame_audit_passed=True)
    (output/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (output/'paired_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=paired[0].keys());w.writeheader();w.writerows(paired)
    a,b=summaries
    text=f'''# MiniCPM-o 2.6：单帧与原生连续四帧比较

已完成同一批 25 个视频、125 道题的两组实测，共 250 次模型回答。

| 输入方式 | 答对/总题数 | 准确率 | 单题平均请求耗时 | 单题中位耗时 |
|---|---:|---:|---:|---:|
| 提问时刻对应的单帧 | {a['correct']}/125 | {a['accuracy_percent']:.2f}% | {a['mean_request_to_response_s']:.3f} 秒 | {a['median_request_to_response_s']:.3f} 秒 |
| 该帧及紧邻它之前的三帧 | {b['correct']}/125 | {b['accuracy_percent']:.2f}% | {b['mean_request_to_response_s']:.3f} 秒 | {b['median_request_to_response_s']:.3f} 秒 |

四帧相对单帧准确率变化 **{delta:+.2f} 个百分点**。
四帧正确、单帧错误 **{transitions[(False,True)]}** 题；
单帧正确、四帧错误 **{transitions[(True,False)]}** 题；
两者均正确 **{transitions[(True,True)]}** 题，两者均错误 **{transitions[(False,False)]}** 题。
两组无效输出分别为 **{a['invalid_responses']}** 和 **{b['invalid_responses']}**，均计为错误。

## 帧选择与实验配置

提问时间若恰好对应原始帧，则使用该帧；若位于两帧之间，使用不晚于该时间的最近一帧。
四帧严格按源视频解码显示序号 **k−3、k−2、k−1、k** 排列，**不进行时间抽样、插帧或 FPS 变换**。
单帧与四帧的最后一帧逐题哈希完全相同。
原视频 FPS 包含 {', '.join(f'{x:.6g}' for x in sorted(fps))}。
四帧首尾实际跨度为 **{min(spans)*1000:.2f}–{max(spans)*1000:.2f} 毫秒**。

模型采用作者发布的 MiniCPM-o 2.6 **GGUF Q4_K_M 语言模型 + F16 视觉编码器/投影器**，
RTX 3070 8 GB，Windows，llama.cpp b11516 Vulkan，F16 KV cache。
两组使用相同的官方选择题提示词、CSV 标签、视觉预处理和贪心解码，
temperature=0，seed=20261009，max_tokens=16，关闭提示词缓存。
仅输入图像与题目/选项，参考标签不进入模型请求，不使用音频或先前问答。
空间预处理保持原宽高比近似不变，将面积缩至不超过 448² 并对齐至 28 像素；JPEG quality=95。
该空间缩放与源帧时间序列选择是两个独立步骤。

## 分任务准确率

| 任务 | 题数 | 单帧 | 四帧 |
|---|---:|---:|---:|
'''
    for task,x in a['per_task'].items():
        y=b['per_task'][task]
        text+=f"| {task} | {x['total']} | {100*x['accuracy']:.2f}% | {100*y['accuracy']:.2f}% |\n"
    text+='\n## 分视频准确率\n\n| 视频 | 单帧答对/5 | 四帧答对/5 |\n|---|---:|---:|\n'
    for n in range(1,26):
        text+=f"| {n} | {a['per_video'][str(n)]['correct']} | {b['per_video'][str(n)]['correct']} |\n"
    text+='''
## 验证与限制

所有视频与原始 ZIP 的成员 CRC 匹配，模型及运行时通过官方 SHA-256 校验。
最终独立从视频起点顺序解码，核对四帧组的全部 500 帧、实际时间戳、源帧序号与 JPEG 哈希，全部一致；
单帧组的 125 张图像逐题等于四帧组末帧，未使用未来帧。
准确率由独立评分程序读取原始输出和原始 CSV 标签重新计算。

耗时包含本地 HTTP、视觉编码、语言模型预填充和答案生成；不包含抽帧、模型加载或下载。
这不是端到端持续流式延迟测量。固定的 25 视频子集、4 位量化、仅视觉输入和 llama.cpp 预处理
限制了结果的适用范围，不能等同于完整 StreamingBench 或全精度 Transformers 的成绩。
CSV 与官方示例 JSON 的标注差异已记录于 `../annotation_comparison.json`，本次固定使用 CSV。
此前 60 秒窗口实验在用户更改协议后停止，其部分结果不纳入本次比较。

文件：`paired_predictions.csv` 为逐题配对结果，`comparison.json` 为汇总数据；
两组原始回答和配置分别在 `../q4_k_m_single_native/`、`../q4_k_m_four_native/`。
复现命令见项目根目录 `README.md`，来源版本和哈希见 `../provenance.json`。

来源：[作者权重](https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf)、
[指定数据集](https://huggingface.co/datasets/mjuicem/StreamingBench)、
[StreamingBench 官方代码](https://github.com/THUNLP-MT/StreamingBench)。
'''
    (output/'report_zh.md').write_text(text,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','questions_per_mode','accuracy_delta_percentage_points','four_only_correct','single_only_correct']},indent=2))


if __name__=='__main__':
    main()
