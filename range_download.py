"""Resumable, bounded HTTP-range downloads; reject servers ignoring ranges."""
import concurrent.futures
import json
import time
from pathlib import Path

import requests


def fetch_range(url, start, end, total, attempts=8):
    for attempt in range(attempts):
        try:
            with requests.get(url, headers={'Range': f'bytes={start}-{end}'},
                              stream=True, timeout=(20, 30)) as r:
                r.raise_for_status()
                if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                    raise ValueError('Range not honored: ' + str(r.headers.get('Content-Range')))
                pieces = []
                begin = time.monotonic()
                for chunk in r.iter_content(65536):
                    pieces.append(chunk)
                    if time.monotonic() - begin > 300:
                        raise TimeoutError('Total chunk time exceeded')
                content = b''.join(pieces)
                if len(content) != end-start+1:
                    raise ValueError('Truncated range')
                return content
        except Exception as exc:
            print('range retry',start,end,attempt,type(exc).__name__,str(exc)[:120],flush=True)
            if attempt == attempts-1:
                raise
            time.sleep(min(10, attempt+1))


def download_ranges(url, path, total, workers=12, start=0, length=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    length = total-start if length is None else length
    if path.exists() and path.stat().st_size == length:
        return
    parts = path.parent/(path.name+'.parts')
    parts.mkdir(exist_ok=True)
    chunk_size = 1024*1024
    blocks = [(i,offset,min(start+length-1,offset+chunk_size-1))
              for i,offset in enumerate(range(start,start+length,chunk_size))]
    done = sum((parts/f'{i:06d}').exists() for i,_,_ in blocks)
    begin = time.monotonic()
    last = begin

    def fetch(block):
        i,a,b = block
        target = parts/f'{i:06d}'
        if target.exists() and target.stat().st_size == b-a+1:
            return
        content = fetch_range(url,a,b,total)
        tmp = target.with_suffix('.partial')
        tmp.write_bytes(content)
        tmp.replace(target)

    pending = [block for block in blocks if not (parts/f'{block[0]:06d}').exists()]
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures=[pool.submit(fetch,block) for block in pending]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            done += 1
            if time.monotonic()-last >= 25:
                print(path.name, f'{done}/{len(blocks)} chunks', flush=True)
                last = time.monotonic()
    partial = path.with_suffix(path.suffix+'.assembled')
    with partial.open('wb') as f:
        for i,_,_ in blocks:
            f.write((parts/f'{i:06d}').read_bytes())
    assert partial.stat().st_size == length
    partial.replace(path)
    # Retain chunks until the caller has checked the full digest/CRC.
    print('assembled', path.name, length, flush=True)


if __name__ == '__main__':
    import argparse
    import zipfile
    from prepare_assets import ROOT, RELEASE, sha256
    models = [('Model-7.6B-Q4_K_M.gguf',4681260960,'4f635fc0c0bb88d50ccd9cf1f1e5892b5cb085ff88fe0d8e1148fd9a8a836bc2'),
              ('mmproj-model-f16.gguf',1044425152,'efa4f7d96aa0f838f2023fc8d28e519179b16f1106777fa9280b32628191aa3e')]
    def model_job(entry):
        name,size,digest = entry
        url = f'https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6-gguf/resolve/757c343a19621613f01954e31f97ba8f61c5be34/{name}'
        p = ROOT/'models'/name
        download_ranges(url,p,size,workers=12)
        assert sha256(p) == digest
        print('SHA256 verified',name,flush=True)
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['vulkan','cuda'],default='vulkan')
    args=parser.parse_args()
    def runtime_job():
        assets = [(f'llama-{RELEASE}-bin-win-vulkan-x64.zip',33445878,'01dff69c02650624d401be0b170348ea38f4bcdca24cb97a7415cddb7315414b')] if args.backend=='vulkan' else [
            (f'llama-{RELEASE}-bin-win-cuda-12.4-x64.zip',265198243,'bd8960a861786b8d9e13624abc98beb23ce0ded97d3ed9add85a5d0a720883c7'),
            ('cudart-llama-bin-win-cuda-12.4-x64.zip',391443627,'8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6')]
        for name,size,digest in assets:
            p=ROOT/'runtime'/name
            download_ranges(f'https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/{name}',p,size,workers=8)
            assert sha256(p)==digest
            with zipfile.ZipFile(p) as z:
                assert z.testzip() is None
                z.extractall(ROOT/'runtime'/('vulkan' if args.backend=='vulkan' else 'llama'))
            print('runtime verified',name,sha256(p),flush=True)
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        futures=[pool.submit(model_job,x) for x in models]+[pool.submit(runtime_job)]
        for f in concurrent.futures.as_completed(futures):
            f.result()
