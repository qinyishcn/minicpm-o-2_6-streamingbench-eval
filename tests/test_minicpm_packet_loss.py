import struct
import unittest

from minicpm_split.packet_loss import packet_ranges, zero_packets


class PacketLossTests(unittest.TestCase):
    def test_mtu_budget_and_float_alignment(self):
        ranges = packet_ranges(10000, 1500)
        self.assertEqual(ranges[0], (0, 1432))
        self.assertEqual(sum(size for _,size in ranges), 10000)
        self.assertTrue(all(size % 4 == 0 and size+66<=1500 for _,size in ranges))
        self.assertEqual([offset for offset,_ in ranges], list(range(0,10000,1432)))

    def test_zero_and_total_loss_preserve_shape_and_exact_bytes(self):
        payload = struct.pack('<800f', *range(800))
        unchanged, stats = zero_packets(payload, 0, 42)
        self.assertEqual(unchanged, payload)
        self.assertEqual(stats['lost_packets'], 0)
        empty, stats = zero_packets(payload, 1, 42)
        self.assertEqual(empty, bytes(len(payload)))
        self.assertEqual(stats['zeroed_bytes'], len(payload))
        self.assertEqual(stats['lost_packets'], stats['total_packets'])

    def test_repeatable_nested_masks_and_complete_packet_zeroing(self):
        payload = struct.pack('<20000f', *([1.25]*20000))
        low, a = zero_packets(payload, .2, 17)
        high, b = zero_packets(payload, .7, 17)
        self.assertEqual(zero_packets(payload,.2,17),(low,a))
        self.assertTrue(set(a['lost_packet_indices'])<=set(b['lost_packet_indices']))
        for index,(offset,size) in enumerate(packet_ranges(len(payload),1500)):
            self.assertEqual(low[offset:offset+size], bytes(size) if index in a['lost_packet_indices']
                             else payload[offset:offset+size])
        self.assertEqual(len(high),len(payload))

    def test_invalid_packet_parameters_fail(self):
        for size, mtu in [(3,1500),(100,66),(100,69)]:
            with self.assertRaises(ValueError):
                packet_ranges(size,mtu)
        for rate in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):
                zero_packets(bytes(100),rate,1)


if __name__=='__main__':
    unittest.main()
