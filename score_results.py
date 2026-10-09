"""Recompute accuracy from raw outputs and original labels, independent of run flags."""
import argparse
import collections
import csv
import json
import math
import statistics
from pathlib import Path

from evaluate import ROOT, select_questions, parse_answer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',nargs='?',default='results/q4_k_m_60s')
    args=parser.parse_args()
    folder=ROOT/args.directory
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    labels={r['question_id']:r for r in select_questions(ROOT/'data/Real_Time_Visual_Understanding.csv')}
    predictions=[json.loads(line) for line in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len({p['question_id'] for p in predictions})==len(predictions),'Duplicate predictions'
    per_task=collections.defaultdict(lambda:[0,0])
    per_video=collections.defaultdict(lambda:[0,0])
    invalid=0
    correct=0
    flat=[]
    for p in predictions:
        reference=labels[p['question_id']]
        answer=parse_answer(p['raw_response'])
        hit=answer==reference['answer']
        assert p['answer']==reference['answer']
        assert p['correct']==hit and p['prediction']==answer
        end=reference['timestamp_seconds']
        frames=p['frames']
        if config['mode'] in ['single','native4']:
            count=1 if config['mode']=='single' else 4
            assert len(frames)==count==p['frame_count']
            assert all(f['actual_seconds']<=end+1e-8 for f in frames)
            assert [f['frame_index'] for f in frames]==list(range(frames[-1]['frame_index']-count+1,frames[-1]['frame_index']+1))
        else:
            assert len(frames)==min(60,end)==p['frame_count']
            assert all(max(0,end-60)<=f['actual_seconds']<end for f in frames)
        correct+=hit
        invalid+=answer is None
        per_task[reference['task_type']][0]+=hit
        per_task[reference['task_type']][1]+=1
        per_video[reference['video_id']][0]+=hit
        per_video[reference['video_id']][1]+=1
        flat.append(dict(question_id=p['question_id'],video_id=reference['video_id'],
                         task_type=reference['task_type'],time_stamp=reference['time_stamp'],
                         question=reference['question'],options=json.dumps(reference['options']),
                         label=reference['answer'],prediction=answer,raw_response=p['raw_response'],
                         correct=int(hit),frame_count=p['frame_count'],
                         request_to_response_s=p['request_to_response_s']))
    def breakdown(d):
        return {str(k):dict(correct=a,total=n,accuracy=a/n) for k,(a,n) in sorted(d.items())}
    n=len(predictions)
    times=[p['request_to_response_s'] for p in predictions]
    summary=dict(status='complete' if n==125 else 'partial',expected=125,evaluated=n,
                 correct=correct,incorrect=n-correct,invalid_responses=invalid,
                 accuracy=correct/n if n else None,accuracy_percent=100*correct/n if n else None,
                 per_task=breakdown(per_task),per_video=breakdown(per_video),
                 mean_request_to_response_s=statistics.mean(times) if times else None,
                 median_request_to_response_s=statistics.median(times) if times else None,
                 total_request_to_response_s=sum(times),
                 latency_scope='Includes local HTTP, image encoding, prefill and generation; excludes frame extraction and model loading. Offline per-question processing, not streaming latency.',
                 limitations=['25 fixed videos, not a random or representative full benchmark sample',
                              'Q4_K_M GGUF with F16 visual projector, not full precision',
                              'vision-only, independent per-question inputs, no persistent streaming state',
                              'runtime preprocessing and JPEG frame transport differ from reference Transformers adapter'])
    (folder/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    if flat:
        with (folder/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=flat[0].keys());writer.writeheader();writer.writerows(flat)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
