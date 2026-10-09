"""Audit cached images and query boundaries by decoding original videos sequentially."""
import concurrent.futures
import argparse
import hashlib
import io
import json
import math
import time

import av
from PIL import Image

from evaluate import ROOT


def verify_video(video_id, questions, cache_root):
    expected={}
    for question in questions:
        for frame in question['frames']:
            cached=cache_root/question['question_id'].replace(' ','_')/frame['file']
            assert hashlib.sha256(cached.read_bytes()).hexdigest()==frame['sha256'],cached
            pts=round(frame['actual_seconds'],8)
            if pts in expected:
                assert expected[pts]==frame['sha256']
            expected[pts]=frame['sha256']
    remaining=dict(expected)
    differences=[]
    pending=sorted(questions,key=lambda q:q['timestamp_seconds'])
    boundary_count=0
    previous_pts=None
    with av.open(str(ROOT/'data/videos'/f'{video_id:03d}.mp4')) as container:
        stream=container.streams.video[0]
        stream.thread_type='AUTO'
        stream.codec_context.thread_count=2
        for source_index,frame in enumerate(container.decode(stream)):
            pts=round(float(frame.pts*stream.time_base),8)
            assert previous_pts is None or pts>previous_pts,(video_id,previous_pts,pts)
            while boundary_count<len(pending) and pts>pending[boundary_count]['timestamp_seconds']+1e-8:
                question=pending[boundary_count]
                final=question['frames'][-1]
                assert final['frame_index']==source_index-1,(question['question_id'],source_index,final)
                assert round(final['actual_seconds'],8)==previous_pts
                assert previous_pts<=question['timestamp_seconds']+1e-8
                boundary_count+=1
            previous_pts=pts
            if pts not in remaining:
                if not remaining and boundary_count==len(pending):
                    break
                continue
            for question in questions:
                for item in question['frames']:
                    if round(item['actual_seconds'],8)==pts and 'frame_index' in item:
                        assert item['frame_index']==source_index,(video_id,source_index,item)
            image=frame.to_image().convert('RGB')
            scale=min(1,448/math.sqrt(image.width*image.height))
            size=tuple(max(28,int(x*scale/28)*28) for x in image.size)
            image=image.resize(size,Image.Resampling.BICUBIC)
            buffer=io.BytesIO()
            image.save(buffer,format='JPEG',quality=95)
            digest=hashlib.sha256(buffer.getvalue()).hexdigest()
            if digest!=remaining[pts]:
                differences.append(pts)
            del remaining[pts]
            if not remaining and boundary_count==len(pending):
                break
    assert boundary_count==len(pending),(video_id,boundary_count,len(pending))
    result=dict(video_id=video_id,unique_frames=len(expected),query_boundaries_verified=boundary_count,
                missing_pts=list(remaining),hash_differences=differences)
    print('verified sequential frames',video_id,len(expected),'differences',len(differences),'missing',len(remaining),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',nargs='?',default='results/q4_k_m_four_native')
    args=parser.parse_args()
    folder=ROOT/args.directory
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    cache_root=ROOT/('data/native_frames' if config['mode'] in ['single','native4'] else 'data/frames')
    begin=time.perf_counter()
    rows=[json.loads(line) for line in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==125,'Require a complete evaluation before final frame audit'
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        futures=[pool.submit(verify_video,n,[r for r in rows if r['video_id']==n],cache_root) for n in range(1,26)]
        results=[f.result() for f in futures]
    valid=all(not r['missing_pts'] and not r['hash_differences'] for r in results)
    audit=dict(passed=valid,method='Independent sequential decoding from video start; exact PTS, display indices, JPEG SHA256 and next-frame query boundary comparison',
               question_count=len(rows),total_question_frames=sum(r['frame_count'] for r in rows),videos=results,
               wall_seconds=time.perf_counter()-begin)
    (folder/'frame_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    assert valid,audit
    print('All sequential frame comparisons passed',flush=True)


if __name__=='__main__':
    main()
