"""Real edge encoder and fixed-manifest evaluation over an activation-only link."""
import argparse
import hashlib
import json
import os
import re
import socket
import statistics
import time
from pathlib import Path

from .native import EdgeVision, MODEL_SHA, PROJECTOR_SHA, TOKENIZER_SHA, RUNTIME_SHA
from .protocol import encode_message, receive_message

PROMPT = '''You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}
{}
{}
{}\n\nThe best option is:'''


def request(host,port,metadata,payload=b''):
    packet = encode_message(metadata,payload)
    begin = time.perf_counter()
    with socket.create_connection((host,port),timeout=120) as connection:
        connection.sendall(packet)
        response,extra,response_bytes = receive_message(connection)
    if extra or not response.get('ok'):
        raise RuntimeError(response.get('error','Unexpected server response'))
    if payload and response.get('feature_sha256')!=hashlib.sha256(payload).hexdigest():
        raise RuntimeError('Server received different visual activations')
    return response,len(packet),response_bytes,time.perf_counter()-begin


class MiniCPMSplitClient:
    """Edge-only provider using a separate language server, with measured bytes."""
    provider = 'minicpm-split'

    def __init__(self, runtime, tokenizer, projector, host='127.0.0.1', port=18087, cpu_vision=False):
        self.host, self.port = host, port
        self.health, _, _, _ = request(host, port, {'op': 'health'})
        if self.health['vision_weights_loaded'] or self.health['pid'] == os.getpid():
            raise RuntimeError('Expected a separate language-only server process')
        self.edge = EdgeVision(runtime, tokenizer, projector, not cpu_vision)

    def infer(self, images, prompt, question_id, frames):
        timestamp, begin = time.time(), time.perf_counter()
        metadata, payload = self.edge.encode(images, prompt)
        metadata.update(question_id=question_id, frames=frames, max_tokens=16)
        response, sent, received, duration = request(self.host, self.port, metadata, payload)
        return dict(response, feature_bytes=len(payload), request_bytes=sent, response_bytes=received,
                    edge_encode_seconds=metadata['edge_encode_seconds'],
                    request_to_response_seconds=duration, total_seconds=time.perf_counter()-begin,
                    timestamp=timestamp,
                    embedding_shapes=[s['shape'] for s in metadata['chunks'] if s['kind']=='embedding'])

    def close(self):
        self.edge.close()


def load_frames(folder, row, mode):
    frames = json.loads((folder/'frames.json').read_text(encoding='utf-8'))
    if len(frames) != 4 or mode not in ('single', 'native4'):
        raise ValueError('Expected a cache of four consecutive native frames')
    indices = [frame['frame_index'] for frame in frames]
    if indices != list(range(indices[0], indices[0]+4)):
        raise ValueError('Cached frames are not consecutive native frames')
    timestamps = [frame['actual_seconds'] for frame in frames]
    if any(a >= b for a,b in zip(timestamps, timestamps[1:])):
        raise ValueError('Frame PTS are not in chronological order')
    for frame in frames:
        if (frame['query_seconds'] != row['timestamp_seconds'] or
                frame['actual_seconds'] > row['timestamp_seconds']+1e-9):
            raise ValueError('Incorrect query time or future video frame')
        if Path(frame['file']).name != frame['file']:
            raise ValueError('Frame filename must stay inside the question cache')
        if hashlib.sha256((folder/frame['file']).read_bytes()).hexdigest() != frame['sha256']:
            raise ValueError('Cached frame integrity mismatch')
    return frames[-1:] if mode == 'single' else frames


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',required=True)
    parser.add_argument('--projector',required=True)
    parser.add_argument('--tokenizer',default='models/minicpm-o-tokenizer-only.gguf')
    parser.add_argument('--questions',required=True)
    parser.add_argument('--frames',required=True)
    parser.add_argument('--mode',choices=['single','native4'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=18087)
    parser.add_argument('--limit',type=int,default=125)
    parser.add_argument('--cpu-vision',action='store_true')
    args=parser.parse_args()
    if not 1<=args.limit<=125:
        parser.error('--limit must be between 1 and 125')
    run_evaluation(args)


def run_evaluation(args, on_result=None):
    if not 1 <= args.limit <= 125:
        raise ValueError('Question limit must be between 1 and 125')
    rows=json.loads(Path(args.questions).read_text(encoding='utf-8'))
    if (len(rows)!=125 or len({r['question_id'] for r in rows})!=125 or
            {r['video_id'] for r in rows}!=set(range(1,26)) or
            any(r['answer'] not in 'ABCD' or len(r['options'])!=4 for r in rows)):
        raise ValueError('Expected the fixed 25-video, 125-question labeled manifest')
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    path=output/'predictions.jsonl'
    if path.exists():
        raise FileExistsError('Use a new output directory for a fresh split inference run')
    client=MiniCPMSplitClient(args.runtime,args.tokenizer,args.projector,args.host,args.port,args.cpu_vision)
    config={'mode':args.mode,'transport':'TCP binary activation envelope v1','wire_dtype':'<f4',
            'images_transmitted':False,'split_after':'visual encoder + resampler/projector',
            'edge_language_weight_tensors':0,'server_vision_weight_tensors':0,
            'edge_pid':os.getpid(),'server_pid':client.health['pid'],'runtime':'llama.cpp b11516',
            'llm_quantization':'Q4_K_M','visual_weights':'F16','context':4096,'max_tokens':16,
            'temperature':0,'expected_questions':125,'prompt_template':PROMPT,
            'vision_device':'CPU' if args.cpu_vision else 'Vulkan',
            'questions_sha256':hashlib.sha256(Path(args.questions).read_bytes()).hexdigest(),
            'frame_selection':'latest source PTS <= query, previous 3 native frame indices',
            'frame_preprocessing':'reused verified JPEG cache, 588x336, no temporal resampling',
            'server_host':args.host,'server_port':args.port}
    config.update(model_sha256=MODEL_SHA, projector_sha256=PROJECTOR_SHA,
                  tokenizer_sha256=TOKENIZER_SHA, runtime_dll_sha256=RUNTIME_SHA)
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    predictions=[]
    try:
        for row in rows[:args.limit]:
            folder=Path(args.frames)/row['question_id'].replace(' ','_')
            frames=load_frames(folder,row,args.mode)
            response=client.infer([folder/f['file'] for f in frames],
                                  PROMPT.format(row['question'],*row['options']),row['question_id'],frames)
            raw=response['result']['text']
            match=re.fullmatch(r'\s*\(?([ABCD])\)?[.)]?\s*',raw)
            answer=match.group(1) if match else None
            result=dict(response,video_id=row['video_id'],label=row['answer'],
                        raw_response=raw,prediction=answer,correct=answer==row['answer'],frames=frames)
            with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(result)+'\n')
            predictions.append(result)
            if on_result:
                on_result(result)
            print(f"{len(predictions)}/{args.limit} {args.mode} {row['question_id']} {answer}/{row['answer']} features={result['feature_bytes']}B seconds={result['total_seconds']:.2f}",flush=True)
    finally:
        client.close()
    summary={'status':'complete' if len(predictions)==125 else 'partial','evaluated':len(predictions),
             'correct':sum(r['correct'] for r in predictions),
             'accuracy_percent':100*sum(r['correct'] for r in predictions)/len(predictions),
             'invalid_responses':sum(r['prediction'] is None for r in predictions),
             'mean_feature_bytes':statistics.mean(r['feature_bytes'] for r in predictions),
             'mean_total_seconds':statistics.mean(r['total_seconds'] for r in predictions),
             'mean_edge_encode_seconds':statistics.mean(r['edge_encode_seconds'] for r in predictions),
             'mean_server_inference_seconds':statistics.mean(r['server_inference_seconds'] for r in predictions),
             'mean_request_to_response_seconds':statistics.mean(r['request_to_response_seconds'] for r in predictions)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':main()
