"""Unsplit reference using upstream mtmd_helper_eval_chunks directly."""
import argparse
import ctypes as C
import json
from pathlib import Path

from .client import PROMPT
from .native import EdgeVision,LanguageDecoder,I


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--projector',required=True)
    parser.add_argument('--tokenizer',default='models/minicpm-o-tokenizer-only.gguf')
    parser.add_argument('--questions',required=True)
    parser.add_argument('--frames',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--limit',type=int,default=3)
    args=parser.parse_args()
    decoder=LanguageDecoder(args.runtime,args.model)
    edge=EdgeVision(args.runtime,args.tokenizer,args.projector)
    rows=json.loads(Path(args.questions).read_text(encoding='utf-8'))
    results=[]
    try:
        for mode in ['single','native4']:
            for row in rows[:args.limit]:
                folder=Path(args.frames)/row['question_id'].replace(' ','_')
                frames=json.loads((folder/'frames.json').read_text(encoding='utf-8'))
                if mode=='single':frames=frames[-1:]
                decoder.reset()
                with edge.chunks([folder/f['file'] for f in frames],PROMPT.format(row['question'],*row['options'])) as chunks:
                    position=I()
                    status=edge.api.mtmd.mtmd_helper_eval_chunks(edge.vision,decoder.context,chunks,0,0,512,True,C.byref(position))
                    if status:raise RuntimeError(f'Unsplit upstream reference failed: {status}')
                    result=decoder.finish(position.value)
                results.append(dict(mode=mode,question_id=row['question_id'],**result))
                print('UNSPLIT',mode,row['question_id'],result['text'],flush=True)
    finally:
        edge.close();decoder.close()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
