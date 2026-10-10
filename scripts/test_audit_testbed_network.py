"""Packet-level regression for DNS transaction IDs resembling RTP headers."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import dpkt

from audit_testbed_network import audit


class CaptureFilterTests(unittest.TestCase):
    def run_capture(self, dns_ports=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            observations = {'qdisc': [{'kind': 'noqueue', 'drops': 0}],
                'snmp': 'Udp: InDatagrams OutDatagrams\nUdp: 0 0\n'}
            (root / 'network_observations.jsonl').write_text(json.dumps(observations) + '\n')
            (root / 'orchestrator.log').write_text(
                'Running realtime_video_understanding [1/1]\n'
                'Completed: success=True, latency=1.0s\n')
            (root / 'traffic.db').write_bytes(b'')
            packets = [(40000, 40001, struct.pack('!BBHII', 0x80, 109, 1, 123, 456) + bytes(64))]
            if dns_ports:
                # A valid DNS query whose transaction ID matches the two-byte RTP filter.
                dns = dpkt.dns.DNS(id=0x856d, qd=[dpkt.dns.DNS.Q(name='example.com')])
                packets.append((*dns_ports, bytes(dns)))
            with (root / 'test.pcap').open('wb') as stream:
                writer = dpkt.pcap.Writer(stream)
                for sport, dport, payload in packets:
                    udp = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
                    udp.ulen = len(udp)
                    ip = dpkt.ip.IP(src=b'\x7f\0\0\x01', dst=b'\x7f\0\0\x01',
                        p=17, ttl=64, data=udp)
                    ip.len = len(ip)
                    frame = dpkt.ethernet.Ethernet(type=0x0800, data=ip)
                    writer.writepkt(bytes(frame), ts=1.0)
            return audit(root, root, expected_messages=1)

    def test_dns_query_with_rtp_like_transaction_id_is_excluded(self):
        row = self.run_capture((37929, 53))
        self.assertEqual(row['captured_rtp_packets_before_deduplication'], 1)

    def test_dns_response_port_with_rtp_like_transaction_id_is_excluded(self):
        row = self.run_capture((53, 37929))
        self.assertEqual(row['captured_rtp_packets_before_deduplication'], 1)

    def test_real_media_is_retained(self):
        row = self.run_capture()
        self.assertEqual(row['captured_unique_rtp_sequences_per_message'], 1)
        self.assertEqual(row['rtp_message_groups'], 1)


if __name__ == '__main__':
    unittest.main()
