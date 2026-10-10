"""Fresh-process zero-visual and text-only controls; no production-path changes."""
import argparse,collections,ctypes as C,hashlib,json,random,re,time
from pathlib import Path
from minicpm_split.client import PROMPT,load_frames
from minicpm_split.native import EdgeVision,LanguageDecoder,MODEL_SHA,PROJECTOR_SHA,Z

p=argparse.ArgumentParser()
p.add_argument('--assets',required=True);p.add_argument('--runtime',required=True);p.add_argument('--output',required=True)
a=p.parse_args();assets=Path(a.assets);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
manifest=json.loads((assets/'results/q4_k_m_single_native/questions_manifest.json').read_text())
labels={r['question_id']:r['answer'] for r in manifest}
rows=[{k:r[k] for k in ('question_id','question','options','video_id','task_type','timestamp_seconds')} for r in manifest]
rows.sort(key=lambda r:hashlib.sha256(('zero-audit-20261009|'+r['question_id']).encode()).digest())
decoder=LanguageDecoder(a.runtime,assets/'models/Model-7.6B-Q4_K_M.gguf')
edge=EdgeVision(a.runtime,'models/minicpm-o-tokenizer-only.gguf',assets/'models/mmproj-model-f16.gguf')
records=[]
try:
 for condition in ('zero_single_fresh','text_only','text_only_rotated_options'):
  for row in rows:
   qid=row['question_id'];options=row['options'];order=list(range(4))
   if condition=='text_only_rotated_options':
    shift=1+int.from_bytes(hashlib.sha256(qid.encode()).digest()[:4],'big')%3
    order=order[shift:]+order[:shift]
    options=[f'{"ABCD"[i]}. '+re.sub(r'^[ABCD]\.\s*','',row['options'][j]) for i,j in enumerate(order)]
   prompt=PROMPT.format(row['question'],*options)
   images=[]
   if condition=='zero_single_fresh':
    folder=assets/'data/native_frames'/qid.replace(' ','_')
    images=[folder/f['file'] for f in load_frames(folder,row,'single')]
   segments=[];length=0
   with edge.chunks(images,prompt) as chunks:
    mt=edge.api.mtmd
    for index in range(mt.mtmd_input_chunks_size(chunks)):
     chunk=mt.mtmd_input_chunks_get(chunks,index);kind=mt.mtmd_input_chunk_get_type(chunk)
     if kind==0:
      size=Z();tokens=mt.mtmd_input_chunk_get_tokens_text(chunk,C.byref(size))
      segments.append({'kind':'tokens','ids':list(tokens[:size.value])})
     elif kind==1:
      count=mt.mtmd_input_chunk_get_n_tokens(chunk);nbytes=count*edge.width*4
      segments.append({'kind':'embedding','shape':[count,edge.width],'dtype':'<f4','offset':length,'bytes':nbytes,'n_pos':mt.mtmd_input_chunk_get_n_pos(chunk)})
      length+=nbytes
     else:raise ValueError(kind)
   # No visual encoder is invoked in any condition. All visual bytes are newly
   # allocated zero bytes, so no previous nonzero feature can be carried over.
   begin=time.perf_counter()
   if images:
    metadata={'protocol':1,'model_sha256':MODEL_SHA,'projector_sha256':PROJECTOR_SHA,'chunks':segments}
    result=decoder.generate(metadata,bytes(length),max_tokens=16)
   else:
    assert all(s['kind']=='tokens' for s in segments)
    decoder.reset();position=0
    for i,s in enumerate(segments):
     for start in range(0,len(s['ids']),512):
      position=decoder._decode(s['ids'][start:start+512],position,output=i==len(segments)-1 and start+512>=len(s['ids']))
    result=decoder.finish(position,16)
   match=re.fullmatch(r'\s*\(?([ABCD])\)?[.)]?\s*',result['text']);answer=match.group(1) if match else None
   record={'condition':condition,'question_id':qid,'task_type':row['task_type'],'raw_response':result['text'],'prediction':answer,'option_order':order,'visual_bytes':length,'visual_sha256':hashlib.sha256(bytes(length)).hexdigest(),'input_text':prompt,'first_logits_sha256':result['first_logits_sha256'],'seconds':time.perf_counter()-begin}
   records.append(record)
   with (out/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
   print(condition,len([r for r in records if r['condition']==condition]),qid,answer,flush=True)
finally:edge.close();decoder.close()
# Labels are consulted only here, after every model answer has been generated.
summary={}
for condition in ('zero_single_fresh','text_only','text_only_rotated_options'):
 group=[r for r in records if r['condition']==condition]
 correct=sum(r['prediction'] is not None and 'ABCD'[r['option_order']['ABCD'.index(r['prediction'])]]==labels[r['question_id']] for r in group)
 summary[condition]={'questions':len(group),'correct':correct,'accuracy':correct/125,'predictions':dict(collections.Counter(r['prediction'] for r in group))}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)