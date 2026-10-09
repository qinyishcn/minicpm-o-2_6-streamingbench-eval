"""Use exact native display frames: last frame at/before query, and its predecessors."""
import bisect
import collections
import hashlib
import io
import json
import math

from evaluate import ROOT, select_questions


def preceding_positions(times,query,count):
    last=bisect.bisect_right(times,query+1e-8)-1
    if last<count-1:
        raise ValueError('Not enough preceding frames')
    return list(range(last-count+1,last+1))


def prepare_video(video_id,questions):
    import av
    from PIL import Image
    pending=sorted(questions,key=lambda r:r['timestamp_seconds'])
    recent=collections.deque(maxlen=4)
    i=0
    with av.open(str(ROOT/'data/videos'/f'{video_id:03d}.mp4')) as container:
        stream=container.streams.video[0]
        stream.thread_type='AUTO'
        stream.codec_context.thread_count=4
        fps=float(stream.average_rate)
        def save(row):
            assert len(recent)==4
            folder=ROOT/'data/native_frames'/row['question_id'].replace(' ','_')
            folder.mkdir(parents=True,exist_ok=True)
            metadata=[]
            for k,(frame_index,actual,frame) in enumerate(recent):
                assert actual<=row['timestamp_seconds']+1e-8
                image=frame.to_image().convert('RGB');original=image.size
                scale=min(1,448/math.sqrt(image.width*image.height))
                size=tuple(max(28,int(x*scale/28)*28) for x in image.size)
                image=image.resize(size,Image.Resampling.BICUBIC)
                buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=95)
                payload=buffer.getvalue();name=f'{k:03d}.jpg'
                (folder/name).write_bytes(payload)
                metadata.append(dict(file=name,frame_index=frame_index,actual_seconds=actual,
                                     query_seconds=row['timestamp_seconds'],source_fps=fps,
                                     source_fps_rational=str(stream.average_rate),
                                     original_size=original,size=size,sha256=hashlib.sha256(payload).hexdigest()))
            assert [m['frame_index'] for m in metadata]==list(range(metadata[-1]['frame_index']-3,metadata[-1]['frame_index']+1))
            (folder/'frames.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        for frame_index,frame in enumerate(container.decode(stream)):
            actual=float(frame.pts*stream.time_base)
            while i<len(pending) and pending[i]['timestamp_seconds']<actual-1e-8:
                save(pending[i]);i+=1
            if i==len(pending):
                break
            recent.append((frame_index,actual,frame))
        while i<len(pending):
            assert recent[-1][1]<=pending[i]['timestamp_seconds']+1e-8
            assert pending[i]['timestamp_seconds']-recent[-1][1]<=1/fps+1e-5
            save(pending[i]);i+=1
    print('native frames prepared',video_id,'fps',fps,'questions',len(questions),flush=True)


def main():
    import concurrent.futures
    rows=select_questions(ROOT/'data/Real_Time_Visual_Understanding.csv')
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        futures=[pool.submit(prepare_video,n,[r for r in rows if r['video_id']==n]) for n in range(1,26)]
        for f in futures:f.result()
    print('Prepared all 125 native four-frame groups',flush=True)


if __name__=='__main__':
    main()
