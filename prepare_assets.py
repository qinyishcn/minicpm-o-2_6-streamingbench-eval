"""Pinned asset constants, SHA256, and a small HTTP reader for ZIP inventories."""
import hashlib
import io
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
MODEL_REV = '565cac3bb01ac51e985fff9f5d7a52322b826c23'
DATA_REV = '48872fa707124474ce4c5172ddc58efb8bc88058'
RELEASE = 'b11516'
ARCHIVE = 'Real-Time Visual Understanding_1-50.zip'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.position = 0
        r = requests.get(url, headers={'Range': 'bytes=-65536'}, timeout=90)
        r.raise_for_status()
        if r.status_code != 206:
            raise ValueError('ZIP server must support byte ranges')
        self.size = int(r.headers['Content-Range'].split('/')[-1])
        self.tail_start = self.size - len(r.content)
        self.tail = r.content

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, n=-1):
        n = self.size - self.position if n < 0 else min(n, self.size - self.position)
        if n <= 0:
            return b''
        start, end = self.position, self.position + n - 1
        if start >= self.tail_start:
            content = self.tail[start-self.tail_start:start-self.tail_start+n]
        else:
            for attempt in range(5):
                try:
                    r = requests.get(self.url, headers={'Range': f'bytes={start}-{end}'}, timeout=(30, 120))
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                        raise ValueError('Unexpected Content-Range')
                    content = r.content
                    if len(content) != n:
                        raise ValueError('Truncated ZIP range')
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2)
        self.position += len(content)
        return content
