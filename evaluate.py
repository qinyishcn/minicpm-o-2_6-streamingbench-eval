"""Past-only visual QA evaluation against a local llama.cpp server."""
import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import math
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPT = '''You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}
{}
{}
{}\n\nThe best option is:'''


def select_questions(path):
    rows=[]
    with open(path,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            match=re.fullmatch(r'Real-Time Visual Understanding_sample_(\d+)_(\d+)',r['question_id'])
            if not match:
                raise ValueError(r['question_id'])
            video_id,question_number=map(int,match.groups())
            if 1 <= video_id <= 25:
                r.update(video_id=video_id,question_number=question_number,
                         options=ast.literal_eval(r['options']))
                r['timestamp_seconds']=sum(int(x)*60**i for i,x in enumerate(reversed(r['time_stamp'].split(':'))))
                assert r['answer'] in 'ABCD' and len(r['options'])==4
                rows.append(r)
    rows.sort(key=lambda r:(r['video_id'],r['question_number']))
    assert len(rows)==125 and len({r['question_id'] for r in rows})==125
    for n in range(1,26):
        assert [r['question_number'] for r in rows if r['video_id']==n]==[1,2,3,4,5]
    return rows


def frame_targets(end,context=60):
    start=max(0,end-context)
    # The reference MiniCPMo adapter samples clip-end-n+0.9, ..., clip-end-0.1.
    return [round(start+i+0.9,6) for i in range(int(end-start))]


def make_prompt(row):
    return PROMPT.format(row['question'],*row['options'])


def parse_answer(text):
    match=re.fullmatch(r'\s*\(?([ABCD])\)?[.)]?\s*',text)
    return match.group(1) if match else None


def extract_frames(video, row, frame_root):
    import av
    from PIL import Image
    folder=frame_root/row['question_id'].replace(' ','_')
    index=folder/'frames.json'
    if index.exists():
        metadata=json.loads(index.read_text(encoding='utf-8'))
        assert all((folder/x['file']).exists() for x in metadata)
        return folder,metadata
    folder.mkdir(parents=True,exist_ok=True)
    targets=frame_targets(row['timestamp_seconds'])
    assert targets
    metadata=[]
    with av.open(str(video)) as container:
        stream=container.streams.video[0]
        stream.thread_type='AUTO'
        container.seek(int(targets[0]/stream.time_base),stream=stream,backward=True)
        index_target=0
        for frame in container.decode(stream):
            actual=float(frame.pts*stream.time_base)
            if actual+1e-7 < targets[index_target]:
                continue
            if actual >= row['timestamp_seconds']:
                break
            image=frame.to_image().convert('RGB')
            original=image.size
            # Keep aspect ratio; floor dimensions to patch multiples, area <=448².
            # llama.cpp may add one local view when the long edge exceeds 448.
            scale=min(1,448/math.sqrt(image.width*image.height))
            size=tuple(max(28,int(x*scale/28)*28) for x in image.size)
            image=image.resize(size,Image.Resampling.BICUBIC)
            name=f'{index_target:03d}.jpg'
            buffer=io.BytesIO()
            image.save(buffer,format='JPEG',quality=95)
            payload=buffer.getvalue()
            (folder/name).write_bytes(payload)
            metadata.append(dict(file=name,target_seconds=targets[index_target],
                                 actual_seconds=actual,original_size=original,size=size,
                                 sha256=hashlib.sha256(payload).hexdigest()))
            index_target+=1
            if index_target==len(targets):
                break
    assert len(metadata)==len(targets),(video,len(metadata),len(targets))
    assert all(max(0,row['timestamp_seconds']-60)<=x['actual_seconds']<row['timestamp_seconds'] for x in metadata)
    index.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return folder,metadata


def main():
    import requests
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=125)
    parser.add_argument('--output',default='results/q4_k_m_60s')
    parser.add_argument('--url',default='http://127.0.0.1:18086')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--mode',choices=['history60','single','native4'],default='history60')
    args=parser.parse_args()
    output=ROOT/args.output
    output.mkdir(parents=True,exist_ok=True)
    rows=select_questions(ROOT/'data/Real_Time_Visual_Understanding.csv')
    (output/'questions_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    config=dict(model='openbmb/MiniCPM-o-2_6-gguf',revision='565cac3bb01ac51e985fff9f5d7a52322b826c23',
                quantization='Q4_K_M',vision_projector='F16',mode='vision-only past-context QA',
                context_seconds=60,sampling_fps=1,max_frames=60,frame_target_offset=0.9,
                frame_size='aspect preserved, area <=448^2, dimensions floored to multiples of 28',
                image_format='JPEG quality=95',temperature=0,seed=20261009,max_tokens=16,
                repeat_penalty=1.0,answer_parser='strict letter plus optional punctuation',
                llama_cpp_release='b11516',backend='Vulkan',kv_cache='F16',
                vision_views='runtime global view plus possible local slice',
                label_access='scorer only; never included in model request',
                prompt_template=PROMPT,total_expected=125)
    if args.mode!='history60':
        config.update(mode=args.mode,context_seconds=None,sampling_fps=None,
                      max_frames=1 if args.mode=='single' else 4,frame_target_offset=None,
                      frame_selection='last native frame at/before query; optionally previous three consecutive native frames',
                      native_frame_cache='data/native_frames')
    config_path=output/'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding='utf-8'))==config,'Cannot resume with a changed config'
    config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    result_path=output/'predictions.jsonl'
    done=set()
    if result_path.exists():
        done={json.loads(line)['question_id'] for line in result_path.read_text(encoding='utf-8').splitlines()}
    for row in rows[:args.limit]:
        if row['question_id'] in done:
            continue
        video=ROOT/'data/videos'/f"{row['video_id']:03d}.mp4"
        while not video.exists():
            print('waiting for video',row['video_id'],flush=True)
            time.sleep(20)
        begin=time.perf_counter()
        if args.mode=='history60':
            folder,metadata=extract_frames(video,row,ROOT/'data/frames')
        else:
            folder=ROOT/'data/native_frames'/row['question_id'].replace(' ','_')
            metadata_path=folder/'frames.json'
            while not metadata_path.exists():
                print('waiting for native frames',row['question_id'],flush=True)
                time.sleep(2)
            metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
            if args.mode=='single':metadata=metadata[-1:]
        extraction_s=time.perf_counter()-begin
        if args.prepare_only:
            print('frames ready',row['question_id'],len(metadata),flush=True)
            continue
        content=[]
        for frame in metadata:
            encoded=base64.b64encode((folder/frame['file']).read_bytes()).decode('ascii')
            content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+encoded}})
        content.append({'type':'text','text':make_prompt(row)})
        payload=dict(messages=[{'role':'user','content':content}],temperature=0,seed=20261009,
                     max_tokens=16,repeat_penalty=1.0,stream=False,cache_prompt=False)
        request_hash=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        begin=time.perf_counter()
        response=requests.post(args.url+'/v1/chat/completions',json=payload,timeout=(20,1200))
        inference_s=time.perf_counter()-begin
        response.raise_for_status()
        body=response.json()
        raw=body['choices'][0]['message']['content']
        prediction=parse_answer(raw)
        result=dict(**row,prediction=prediction,raw_response=raw,correct=prediction==row['answer'],
                    input_mode=args.mode,
                    frame_count=len(metadata),frames=metadata,extraction_s=extraction_s,
                    request_to_response_s=inference_s,request_sha256=request_hash,
                    usage=body.get('usage'),timings=body.get('timings'),
                    finish_reason=body['choices'][0].get('finish_reason'))
        with result_path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(result,ensure_ascii=False)+'\n')
            f.flush()
        print(f"{len(done)+1}/125 {row['question_id']} prediction={prediction} label={row['answer']} correct={result['correct']} frames={len(metadata)} seconds={inference_s:.2f}",flush=True)
        done.add(row['question_id'])
    if not args.prepare_only:
        print('run finished',len(done),'of 125',flush=True)


if __name__=='__main__':
    main()
