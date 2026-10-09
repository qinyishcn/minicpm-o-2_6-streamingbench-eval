"""Local language-only server accepting MiniCPM visual activation envelopes."""
import argparse
import hashlib
import json
import os
import socketserver
import threading
import time
from pathlib import Path

from .native import LanguageDecoder
from .protocol import encode_message, receive_message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18087)
    parser.add_argument('--log', default='results/minicpm_split/server_requests.jsonl')
    args = parser.parse_args()
    model = LanguageDecoder(args.runtime,args.model)
    log = Path(args.log)
    log.parent.mkdir(parents=True,exist_ok=True)
    lock = threading.Lock()

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(120)
            begin = time.perf_counter()
            try:
                metadata,payload,received = receive_message(self.request)
                if metadata.get('op')=='health':
                    self.request.sendall(encode_message({'ok':True,'pid':os.getpid(),
                        'vision_weights_loaded':False,'language_hidden_size':model.width}))
                    return
                with lock:
                    infer_begin = time.perf_counter()
                    result = model.generate(metadata,payload,metadata.get('max_tokens',16))
                    inference = time.perf_counter()-infer_begin
                response = {'ok':True,'question_id':metadata.get('question_id'),'result':result,
                            'feature_sha256':hashlib.sha256(payload).hexdigest(),
                            'server_inference_seconds':inference,'server_receive_to_result_seconds':time.perf_counter()-begin}
                packet = encode_message(response)
                self.request.sendall(packet)
                record = dict(response,request_bytes=received,response_bytes=len(packet),feature_bytes=len(payload),
                              chunks=[{'kind':s['kind'],'shape':s.get('shape')} for s in metadata['chunks']],
                              server_pid=os.getpid(),vision_weights_loaded=False)
                with lock,log.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(record)+'\n')
            except Exception as error:
                self.request.sendall(encode_message({'ok':False,'error':str(error)}))

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    try:
        with Server((args.host,args.port),Handler) as server:
            print(json.dumps({'status':'ready','host':args.host,'port':args.port,'pid':os.getpid(),
                              'vision_weights_loaded':False,'language_hidden_size':model.width}),flush=True)
            server.serve_forever()
    finally:
        model.close()


if __name__=='__main__':
    main()
