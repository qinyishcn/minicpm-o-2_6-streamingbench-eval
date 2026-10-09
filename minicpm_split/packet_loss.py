"""Offline MTU-sized feature erasure, followed by real language-model inference."""
import argparse
import hashlib
import json
import math
import random
import re
import statistics
import time
from pathlib import Path

from .client import PROMPT, load_frames, request
from .native import EdgeVision, MODEL_SHA, PROJECTOR_SHA, TOKENIZER_SHA, RUNTIME_SHA

# IP MTU excludes the Ethernet header. Budget a minimal IPv4/UDP/RTP packet,
# a 16-byte feature-fragment header and the 10-byte SRTP AES-CM auth tag.
# This is an offline size model, not a claim to encrypt or send SRTP packets.
OVERHEAD = {'ipv4':20, 'udp':8, 'rtp':12, 'feature_fragment_header':16, 'srtp_auth_tag':10}


def packet_ranges(length, mtu=1500):
    available = (mtu-sum(OVERHEAD.values()))//4*4
    if length < 0 or length % 4 or available < 4:
        raise ValueError('Feature buffer and packet budget must hold whole float32 elements')
    return [(offset,min(available,length-offset)) for offset in range(0,length,available)]


def zero_packets(payload, loss_rate, seed, mtu=1500):
    if not math.isfinite(loss_rate) or not 0<=loss_rate<=1:
        raise ValueError('Loss rate must be between zero and one')
    ranges = packet_ranges(len(payload),mtu)
    rng, damaged, lost, zeroed = random.Random(seed), bytearray(payload), [], 0
    for index,(offset,size) in enumerate(ranges):
        if rng.random() < loss_rate:
            damaged[offset:offset+size] = bytes(size)
            lost.append(index)
            zeroed += size
    return bytes(damaged), {'mtu':mtu, 'feature_payload_per_full_packet':ranges[0][1] if ranges else 0,
        'header_budget':OVERHEAD, 'nominal_loss_rate':loss_rate, 'seed':seed,
        'total_packets':len(ranges), 'lost_packets':len(lost), 'lost_packet_indices':lost,
        'observed_packet_loss_rate':len(lost)/len(ranges) if ranges else 0,
        'zeroed_bytes':zeroed, 'zeroed_float32_elements':zeroed//4,
        'zeroed_feature_fraction':zeroed/len(payload) if payload else 0}


def summarize(predictions):
    correct = sum(r['correct'] for r in predictions)
    packets = sum(r['loss']['total_packets'] for r in predictions)
    lost = sum(r['loss']['lost_packets'] for r in predictions)
    return {'status':'complete' if len(predictions)==125 else 'partial', 'evaluated':len(predictions),
            'correct':correct, 'accuracy_percent':100*correct/125,
            'completed_question_accuracy_percent':100*correct/len(predictions) if predictions else 0,
            'missing_questions':125-len(predictions),
            'invalid_responses':sum(r['prediction'] is None for r in predictions),
            'total_packets':packets, 'lost_packets':lost,
            'observed_packet_loss_rate':lost/packets if packets else 0,
            'zeroed_feature_fraction':sum(r['loss']['zeroed_bytes'] for r in predictions)/
                sum(r['feature_bytes'] for r in predictions) if predictions else 0,
            'mean_request_to_response_seconds':statistics.mean(r['request_to_response_seconds'] for r in predictions)
                if predictions else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--projector', required=True)
    parser.add_argument('--tokenizer', default='models/minicpm-o-tokenizer-only.gguf')
    parser.add_argument('--questions', required=True)
    parser.add_argument('--frames', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18087)
    parser.add_argument('--mtu', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--loss-rates', type=float, nargs='+', default=[0,.001,.01,.05,.1,.2])
    parser.add_argument('--limit', type=int, default=125)
    args = parser.parse_args()
    if not 1<=args.limit<=125 or len(set(args.loss_rates))!=len(args.loss_rates):
        parser.error('Expected 1..125 questions and unique loss rates')
    for rate in args.loss_rates:
        zero_packets(bytes(4),rate,1,args.mtu)
    manifest = Path(args.questions).read_bytes()
    rows = json.loads(manifest)
    if len(rows)!=125 or len({r['question_id'] for r in rows})!=125:
        raise ValueError('Expected the verified 125-question manifest')
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    config = {'experiment':'offline Bernoulli feature packet erasure; erased bytes -> positive float32 zero',
              'mtu':args.mtu,'header_budget':OVERHEAD,'feature_payload_per_full_packet':packet_ranges(10000,args.mtu)[0][1],
              'loss_rates':args.loss_rates,'seed':args.seed,'random_seeds':1,
              'mask_pairing':'same per-question uniforms at all rates; higher-rate masks contain lower-rate masks',
              'question_seed_derivation':'first 8 bytes of SHA256(seed|mode|question_id), big endian',
              'expected_questions_per_condition':125,'modes':['single','native4'],
              'questions_sha256':hashlib.sha256(manifest).hexdigest(),
              'model_sha256':MODEL_SHA,'projector_sha256':PROJECTOR_SHA,'tokenizer_sha256':TOKENIZER_SHA,
              'runtime':'llama.cpp b11516 Windows Vulkan x64','runtime_dll_sha256':RUNTIME_SHA,
              'language_quantization':'Q4_K_M','vision_weights':'F16','wire_dtype':'<f4',
              'context':4096,'max_tokens':16,'temperature':0,'prompt_template':PROMPT,
              'network_emulation':False,'actual_transport':'local TCP; erasure happens before sending',
              'images_transmitted':False,'feature_concealment':'zero only; no retransmission/FEC/temporal reuse',
              'scope':'first 25 videos, 5 questions each; latest past native frame or previous 3 plus current',
              'preprocessing':'verified 588x336 JPEG cache, no temporal resampling'}
    config_path = output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Existing experiment configuration differs; use a new output directory')
    config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    health,_,_,_ = request(args.host,args.port,{'op':'health'})
    if health['vision_weights_loaded']:
        raise ValueError('Expected a language-only server')
    conditions = {}
    for mode in config['modes']:
        for rate in args.loss_rates:
            folder = output/mode/f'loss_{rate:g}'
            folder.mkdir(parents=True,exist_ok=True)
            path = folder/'predictions.jsonl'
            existing = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
            by_id = {r['question_id']:r for r in existing}
            if len(by_id)!=len(existing) or not set(by_id)<=set(r['question_id'] for r in rows):
                raise ValueError('Duplicate or unknown existing prediction')
            conditions[mode,rate] = folder,existing,by_id
    edge = EdgeVision(args.runtime,args.tokenizer,args.projector)
    try:
        for mode in config['modes']:
            for index,row in enumerate(rows[:args.limit],1):
                todo = [rate for rate in args.loss_rates if row['question_id'] not in conditions[mode,rate][2]]
                if not todo:
                    continue
                folder = Path(args.frames)/row['question_id'].replace(' ','_')
                frames = load_frames(folder,row,mode)
                metadata, original = edge.encode([folder/f['file'] for f in frames],
                                                  PROMPT.format(row['question'],*row['options']))
                metadata.update(question_id=row['question_id'],frames=frames,max_tokens=16)
                seed = int.from_bytes(hashlib.sha256(f'{args.seed}|{mode}|{row["question_id"]}'.encode()).digest()[:8],'big')
                for rate in todo:
                    damaged,loss = zero_packets(original,rate,seed,args.mtu)
                    response,sent,received,duration = request(args.host,args.port,metadata,damaged)
                    raw = response['result']['text']
                    match = re.fullmatch(r'\s*\(?([ABCD])\)?[.)]?\s*',raw)
                    answer = match.group(1) if match else None
                    result = dict(response,mode=mode,question_id=row['question_id'],video_id=row['video_id'],
                        label=row['answer'],prediction=answer,raw_response=raw,correct=answer==row['answer'],
                        loss=loss,feature_bytes=len(original),request_bytes=sent,response_bytes=received,
                        original_feature_sha256=hashlib.sha256(original).hexdigest(),
                        damaged_feature_sha256=hashlib.sha256(damaged).hexdigest(),
                        edge_encode_seconds=metadata['edge_encode_seconds'],request_to_response_seconds=duration,
                        frames=frames,timestamp=time.time(),server_pid=health['pid'])
                    target,predictions,by_id = conditions[mode,rate]
                    with (target/'predictions.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(result)+'\n')
                    predictions.append(result)
                    by_id[row['question_id']] = result
                    (target/'summary.json').write_text(json.dumps(summarize(predictions),indent=2),encoding='utf-8')
                    print(f'{mode} {index}/125 loss={rate:g} {answer}/{row["answer"]} '
                          f'lost={loss["lost_packets"]}/{loss["total_packets"]} time={duration:.3f}',flush=True)
    finally:
        edge.close()
    summary = [{'mode':mode,'nominal_loss_rate':rate,**summarize(values[1])}
               for (mode,rate),values in conditions.items()]
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
