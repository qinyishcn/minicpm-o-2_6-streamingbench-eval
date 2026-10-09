import copy
import socket
import struct
import tempfile
import unittest
from pathlib import Path

from minicpm_split.protocol import encode_message, receive_message, decode_message
from minicpm_split.assets import extract_tokenizer_gguf
from minicpm_split.activation import validate_layout


class SplitProtocolTests(unittest.TestCase):
    def test_tensor_bytes_survive_framing(self):
        metadata = {'chunks': [{'kind': 'embedding', 'shape': [2, 3]}]}
        features = struct.pack('<6f', 0.1, -2, 0, 4, 7, 8)
        packet = encode_message(metadata, features)
        actual, payload = decode_message(packet)
        self.assertEqual(actual, metadata)
        self.assertEqual(payload, features)

    def test_corruption_and_truncation_are_rejected(self):
        packet = encode_message({'question_id': 'test'}, b'features')
        damaged = bytearray(packet)
        damaged[-1] ^= 1
        with self.assertRaises(ValueError):
            decode_message(damaged)
        with self.assertRaises(ValueError):
            decode_message(packet[:-1])

    def test_socket_receive_handles_fragmented_writes(self):
        packet = encode_message({'test': True}, b'abc')
        first, second = socket.socketpair()
        with first, second:
            first.sendall(packet[:7])
            first.sendall(packet[7:])
            self.assertEqual(receive_message(second), ({'test': True}, b'abc', len(packet)))

    def test_metadata_only_asset_contains_no_weight_tensors(self):
        key = b'general.architecture'
        value = b'qwen2'
        kv = struct.pack('<Q', len(key)) + key + struct.pack('<I', 8) + struct.pack('<Q', len(value)) + value
        source = struct.pack('<4sIQQ', b'GGUF', 3, 123, 1) + kv + b'PRIVATE_WEIGHT_BYTES'
        with tempfile.TemporaryDirectory() as folder:
            original, result = Path(folder)/'full.gguf', Path(folder)/'tokenizer.gguf'
            original.write_bytes(source)
            extract_tokenizer_gguf(original, result)
            output = result.read_bytes()
        self.assertEqual(struct.unpack('<4sIQQ', output[:24]), (b'GGUF', 3, 0, 1))
        self.assertEqual(output[24:24+len(kv)], kv)
        self.assertNotIn(b'PRIVATE_WEIGHT_BYTES', output)


class ActivationLayoutTests(unittest.TestCase):
    def setUp(self):
        self.payload = struct.pack('<6f', 1, 2, 3, 4, 5, 6)
        self.metadata = {'protocol': 1, 'chunks': [
            {'kind': 'tokens', 'ids': [1, 2]},
            {'kind': 'embedding', 'shape': [2, 3], 'dtype': '<f4',
             'offset': 0, 'bytes': 24, 'n_pos': 2},
            {'kind': 'tokens', 'ids': [3]}]}

    def validate(self, metadata=None, payload=None):
        return validate_layout(metadata or self.metadata, self.payload if payload is None else payload,
                               width=3, vocab_size=10, context_size=32, max_tokens=16)

    def test_ordered_mixed_segments_and_context_count(self):
        self.assertEqual(self.validate(), 5)

    def test_overlap_holes_and_unreferenced_payload_are_rejected(self):
        for offset in (4, -4):
            metadata = copy.deepcopy(self.metadata)
            metadata['chunks'][1]['offset'] = offset
            with self.assertRaises(ValueError):
                self.validate(metadata, self.payload+b'\0'*4)
        metadata = copy.deepcopy(self.metadata)
        metadata['chunks'].insert(2, copy.deepcopy(metadata['chunks'][1]))
        with self.assertRaises(ValueError):
            self.validate(metadata)
        with self.assertRaises(ValueError):
            self.validate(payload=self.payload+b'\0'*4)

    def test_nonfinite_shape_position_and_token_errors_are_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(payload=struct.pack('<6f', float('nan'), 2, 3, 4, 5, 6))
        for key, value in [('shape', [2, 4]), ('n_pos', 3), ('dtype', '<f2')]:
            metadata = copy.deepcopy(self.metadata)
            metadata['chunks'][1][key] = value
            with self.assertRaises(ValueError):
                self.validate(metadata)
        metadata = copy.deepcopy(self.metadata)
        metadata['chunks'][0]['ids'] = [10]
        with self.assertRaises(ValueError):
            self.validate(metadata)

    def test_protocol_context_and_trailing_empty_tokens_are_rejected(self):
        metadata = copy.deepcopy(self.metadata)
        metadata['protocol'] = 2
        with self.assertRaises(ValueError):
            self.validate(metadata)
        metadata = copy.deepcopy(self.metadata)
        metadata['chunks'][-1]['ids'] = []
        with self.assertRaises(ValueError):
            self.validate(metadata)
        with self.assertRaises(ValueError):
            validate_layout(self.metadata, self.payload, 3, 10, 20, 16)


if __name__ == '__main__':
    unittest.main()
