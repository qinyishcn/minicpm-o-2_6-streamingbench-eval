"""Bounded binary framing for model activations; never carries image pixels."""
import hashlib
import json
import struct

MAGIC = b'MCPMSP01'
HEADER = struct.Struct('!8sIQ32s')
MAX_METADATA = 2 * 1024 * 1024
MAX_PAYLOAD = 128 * 1024 * 1024


def encode_message(metadata, payload=b''):
    encoded = json.dumps(metadata, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(encoded) > MAX_METADATA or len(payload) > MAX_PAYLOAD:
        raise ValueError('Split message exceeds size limit')
    digest = hashlib.sha256(encoded + payload).digest()
    return HEADER.pack(MAGIC, len(encoded), len(payload), digest) + encoded + payload


def decode_message(packet):
    if len(packet) < HEADER.size:
        raise ValueError('Truncated split envelope')
    magic, meta_size, data_size, digest = HEADER.unpack_from(packet)
    if magic != MAGIC or meta_size > MAX_METADATA or data_size > MAX_PAYLOAD:
        raise ValueError('Invalid split envelope')
    if len(packet) != HEADER.size + meta_size + data_size:
        raise ValueError('Split envelope length mismatch')
    body = packet[HEADER.size:]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError('Split envelope integrity mismatch')
    metadata = json.loads(body[:meta_size])
    if not isinstance(metadata, dict):
        raise ValueError('Split metadata must be an object')
    return metadata, bytes(body[meta_size:])


def _receive_exact(connection, count):
    result = bytearray()
    while len(result) < count:
        block = connection.recv(min(count - len(result), 1024 * 1024))
        if not block:
            raise ValueError('Connection ended before split message completed')
        result.extend(block)
    return bytes(result)


def receive_message(connection):
    header = _receive_exact(connection, HEADER.size)
    magic, meta_size, data_size, _ = HEADER.unpack(header)
    if magic != MAGIC or meta_size > MAX_METADATA or data_size > MAX_PAYLOAD:
        raise ValueError('Invalid split envelope')
    packet = header + _receive_exact(connection, meta_size + data_size)
    metadata, payload = decode_message(packet)
    return metadata, payload, len(packet)
