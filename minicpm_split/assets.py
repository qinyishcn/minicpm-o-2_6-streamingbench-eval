"""Extract tokenizer/hyperparameter metadata without distributing LLM tensors."""
import argparse
import struct
from pathlib import Path

_WIDTH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _number(stream, fmt):
    size = struct.calcsize(fmt)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError('Truncated GGUF metadata')
    return struct.unpack(fmt, data)[0]


def _skip_value(stream, kind):
    if kind in _WIDTH:
        stream.seek(_WIDTH[kind], 1)
    elif kind == 8:
        stream.seek(_number(stream, '<Q'), 1)
    elif kind == 9:
        element, count = _number(stream, '<I'), _number(stream, '<Q')
        if element in _WIDTH:
            stream.seek(_WIDTH[element] * count, 1)
        else:
            for _ in range(count):
                _skip_value(stream, element)
    else:
        raise ValueError(f'Unsupported GGUF metadata type {kind}')


def extract_tokenizer_gguf(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Tokenizer destination must differ from source weights')
    with source.open('rb') as stream:
        magic, version, _, entries = struct.unpack('<4sIQQ', stream.read(24))
        if magic != b'GGUF' or version not in (2, 3):
            raise ValueError('Unsupported GGUF header')
        for _ in range(entries):
            _skip_value(stream, 8)
            _skip_value(stream, _number(stream, '<I'))
        end = stream.tell()
        if end > source.stat().st_size:
            raise ValueError('GGUF metadata exceeds source size')
        stream.seek(24)
        metadata = stream.read(end - 24)
    content = struct.pack('<4sIQQ', magic, version, 0, entries) + metadata
    content += b'\0' * (-len(content) % 32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return len(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('output')
    args = parser.parse_args()
    print('Tokenizer-only GGUF bytes:', extract_tokenizer_gguf(args.model, args.output))
