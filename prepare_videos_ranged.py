"""Fetch just the selected ZIP member bytes, then verify against original CRCs."""
import concurrent.futures
import json
import os
import re
import struct
import zlib
from pathlib import Path

from prepare_assets import ROOT, DATA_REV, sha256, RemoteZip
from range_download import download_ranges, fetch_range

URL = f'https://huggingface.co/datasets/mjuicem/StreamingBench/resolve/{DATA_REV}/Real-Time%20Visual%20Understanding_1-50.zip'
TOTAL = 9080551796


def main():
    import requests
    import zipfile
    (ROOT/'data').mkdir(exist_ok=True)
    labels=ROOT/'data/Real_Time_Visual_Understanding.csv'
    if not labels.exists():
        r=requests.get(f'https://huggingface.co/datasets/mjuicem/StreamingBench/resolve/{DATA_REV}/StreamingBench/Real_Time_Visual_Understanding.csv',timeout=60)
        r.raise_for_status()
        labels.write_bytes(r.content)
    inventory_path=ROOT/'data/archive_inventory.json'
    if not inventory_path.exists():
        with zipfile.ZipFile(RemoteZip(URL)) as z:
            entries=[dict(name=i.filename,size=i.file_size,compressed=i.compress_size,
                          crc32=f'{i.CRC:08x}',offset=i.header_offset) for i in z.infolist()]
        inventory_path.write_text(json.dumps(entries,indent=2),encoding='utf-8')
    inventory = json.loads((ROOT/'data/archive_inventory.json').read_text())
    selected=[]
    for entry in inventory:
        m = re.fullmatch(r'sample_(\d+)/video.mp4',entry['name'])
        if m and 1 <= int(m[1]) <= 25:
            selected.append((int(m[1]),entry))
    assert sorted(n for n,_ in selected) == list(range(1,26))
    def job(pair):
        n,e = pair
        p = ROOT/'data/videos'/f'{n:03d}.mp4'
        p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            a=e['offset']
            header=fetch_range(URL,a,a+29,TOTAL)
            assert header[:4] == b'PK\x03\x04'
            method=struct.unpack_from('<H',header,8)[0]
            filename_len,extra_len=struct.unpack_from('<HH',header,26)
            start=a+30+filename_len+extra_len
            compressed=ROOT/'data/compressed'/f'{n:03d}.deflate'
            download_ranges(URL,compressed,TOTAL,workers=int(os.environ.get('STREAMINGBENCH_DOWNLOAD_WORKERS','24')),start=start,length=e['compressed'])
            payload=compressed.read_bytes()
            content=zlib.decompress(payload,-15) if method==8 else payload
            assert len(content)==e['size']
            assert f'{zlib.crc32(content):08x}' == e['crc32']
            partial=p.with_suffix('.partial')
            partial.write_bytes(content)
            partial.replace(p)
        assert p.stat().st_size==e['size']
        assert f'{zlib.crc32(p.read_bytes()):08x}'==e['crc32']
        result=dict(video_id=n,path=str(p.relative_to(ROOT)),archive_member=e['name'],
                    bytes=p.stat().st_size,sha256=sha256(p),crc32=e['crc32'])
        print('video verified',n,result['sha256'],flush=True)
        return result
    results=[]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for f in concurrent.futures.as_completed([pool.submit(job,x) for x in sorted(selected)]):
            results.append(f.result())
            (ROOT/'data/videos_manifest.json').write_text(json.dumps(sorted(results,key=lambda x:x['video_id']),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
