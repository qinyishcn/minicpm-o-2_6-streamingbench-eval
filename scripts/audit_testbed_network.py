"""Summarize local RTC captures/counters without publishing SDP or PCAP bytes.

Run in the existing Testbed Linux environment (dpkt is required).
Only fixed experiment-owned paths are read. No network settings are changed.
"""
import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path

import dpkt


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def udp_counters(observation):
    lines = observation['snmp'].splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith('Udp:'))
    return dict(zip(lines[index].split()[1:], map(int, lines[index + 1].split()[1:])))


def audit(capture_folder, evidence_folder, expected_messages):
    groups = collections.defaultdict(set)
    maximum, packets = 0, 0
    captures = sorted(capture_folder.glob('*.pcap'))
    if not captures:
        raise ValueError(f'No captures: {capture_folder}')
    manifest = []
    for capture in captures:
        manifest.append({'name': capture.name, 'bytes': capture.stat().st_size,
                         'sha256': sha256(capture)})
        with capture.open('rb') as stream:
            reader = dpkt.pcap.Reader(stream)
            for _, data in reader:
                if reader.datalink() == 1:
                    packet = dpkt.ethernet.Ethernet(data).data
                elif reader.datalink() == 113:
                    packet = dpkt.sll.SLL(data).data
                else:
                    raise ValueError(f'Unsupported link type: {reader.datalink()}')
                if not isinstance(packet, dpkt.ip.IP) or not isinstance(packet.data, dpkt.udp.UDP):
                    continue
                payload = bytes(packet.data.data)
                if len(payload) < 12 or payload[0] >> 6 != 2 or payload[1] & 127 != 109:
                    continue
                sequence, timestamp, ssrc = struct.unpack_from('!HII', payload, 2)
                groups[packet.data.sport, packet.data.dport, ssrc, timestamp].add(sequence)
                maximum = max(maximum, packet.len)
                packets += 1
    observations = [json.loads(line) for line in
                    (evidence_folder / 'network_observations.jsonl').read_text().splitlines()]
    if not observations or maximum > 1500 or len(groups) != expected_messages:
        raise ValueError('Missing counter snapshots, oversized RTP, or wrong message count')
    qdiscs = [q for observation in observations for q in observation['qdisc']]
    before, after = udp_counters(observations[0]), udp_counters(observations[-1])
    return {'capture_files': manifest, 'rtp_payload_type': 109,
            'captured_rtp_packets_before_deduplication': packets,
            'captured_unique_rtp_sequences_per_message': sum(map(len, groups.values())),
            'rtp_message_groups': len(groups), 'maximum_captured_rtp_ip_bytes': maximum,
            'network_snapshots': len(observations),
            'qdisc_max_observed_drops': {
                kind: max(q.get('drops', 0) for q in qdiscs if q['kind'] == kind)
                for kind in sorted({q['kind'] for q in qdiscs})},
            'udp_global_counter_deltas': {key: after[key] - before[key] for key in before},
            'initial_qdisc': observations[0]['qdisc'], 'final_qdisc': observations[-1]['qdisc'],
            'sqlite_log': {'bytes': (evidence_folder / 'traffic.db').stat().st_size,
                           'sha256': sha256(evidence_folder / 'traffic.db')},
            'interpretation': 'Captured RTP is not necessarily usable model input. '
                'UDP counters are namespace-wide. Parent/child qdisc drops are not additive. '
                'Original PCAP/SQLite retained locally; raw signaling is not published.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    captures, results = Path(args.captures), Path(args.results)
    output = {}
    for mode in ['single', 'native4']:
        output[f'baseline/{mode}/no_emulation'] = audit(
            captures / 'baseline' / mode / 'no_emulation',
            results / 'minicpm_rtc_baseline_20261010' / mode / 'no_emulation', 125)
    for profile in ['no_emulation', 'lossy', '5qi_7', 'congested']:
        output[f'readiness/native4/{profile}'] = audit(
            captures / 'readiness/native4' / profile,
            results / 'minicpm_rtc_20261010/readiness/native4' / profile, 5)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({key: {field: row[field] for field in
        ['rtp_message_groups', 'maximum_captured_rtp_ip_bytes', 'qdisc_max_observed_drops',
         'udp_global_counter_deltas']} for key, row in output.items()}, indent=2))


if __name__ == '__main__':
    main()
