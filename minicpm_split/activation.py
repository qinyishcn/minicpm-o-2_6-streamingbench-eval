"""Validate an entire activation request before calling native model code."""
import numpy as np


def validate_layout(metadata, payload, width, vocab_size, context_size, max_tokens):
    if metadata.get('protocol') != 1:
        raise ValueError('Unsupported activation protocol')
    segments = metadata.get('chunks')
    if not isinstance(segments, list) or not 1 <= len(segments) <= 128:
        raise ValueError('Invalid number of split segments')
    if type(max_tokens) is not int or not 1 <= max_tokens <= 128:
        raise ValueError('Invalid generation limit')
    total, cursor, images = 0, 0, 0
    for segment in segments:
        if segment.get('kind') == 'embedding':
            shape = segment.get('shape')
            if not isinstance(shape, list) or len(shape) != 2:
                raise ValueError('Invalid visual embedding shape')
            n, columns = shape
            if (type(n) is not int or not 0 < n <= context_size or columns != width or
                    segment.get('dtype') != '<f4' or segment.get('n_pos') != n):
                raise ValueError('Unsupported visual embedding layout')
            size = n * width * 4
            if (segment.get('offset') != cursor or segment.get('bytes') != size or
                    cursor + size > len(payload)):
                raise ValueError('Invalid visual embedding byte range or ordering')
            if not np.isfinite(np.frombuffer(payload, dtype='<f4', count=n*width, offset=cursor)).all():
                raise ValueError('Non-finite visual features')
            cursor += size
            total += n
            images += 1
        elif segment.get('kind') == 'tokens':
            tokens = segment.get('ids')
            if (not isinstance(tokens, list) or not tokens or
                    not all(type(token) is int and 0 <= token < vocab_size for token in tokens)):
                raise ValueError('Invalid or empty text token segment')
            total += len(tokens)
        else:
            raise ValueError('Unexpected split segment type')
    if cursor != len(payload) or not images:
        raise ValueError('Missing visual features or unreferenced payload bytes')
    if total + max_tokens > context_size:
        raise ValueError('Split request exceeds context limit')
    return total
