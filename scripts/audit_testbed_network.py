"""Summarize local RTC captures/counters without publishing SDP or PCAP bytes.

Run in the existing Testbed Linux environment (dpkt is required).
Only fixed experiment-owned paths are read. No network settings are changed.
"""
import argparse
import collections
import hashlib
import json
import re
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


def video_outcomes(path):
    """Retain allowlisted failure events, without publishing RTC signaling."""
    records, current = [], None
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = re.sub(r'\x1b\[[0-9;]*m', '', line)
        start = re.search(r'Running realtime_video_understanding \[(\d+)/(\d+)\]', line)
        if start:
            current = {'video_id': int(start[1]), 'expected_videos': int(start[2]), 'events': []}
            records.append(current)
        if current is None:
            continue
        done = re.search(r'Completed: success=(True|False), latency=([0-9.]+)s', line)
        if done:
            current.update(client_success=done[1] == 'True', original_logged_latency_seconds=float(done[2]))
            if current['client_success']:
                current['client_answers_received'] = 5
        count = re.search(r'Expected 5 model answers, received (\d+)', line)
        if count:
            current['client_answers_received'] = int(count[1])
        for event in ['Timeout waiting for VIDEO_ANALYSIS_COMPLETE',
                      'Exception in callback RTCDataChannel.send', 'aiortc.exceptions.InvalidStateError',
                      'Feature metadata was not acknowledged over the original data channel']:
            if event in line and event not in current['events']:
                current['events'].append(event)
    if (not records or len(records) != records[-1]['expected_videos']
            or [r['video_id'] for r in records] != list(range(1, len(records) + 1))
            or any('client_success' not in r for r in records)):
        raise ValueError('Incomplete or repeated original video session outcomes')
    return records


def audit(capture_folder, evidence_folder, expected_messages=None):
    groups = collections.defaultdict(set)
    maximum, packets, dns_collisions = 0, 0, 0
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
                # The original all-UDP capture also contains WSL DNS traffic.
                # DNS transaction IDs can mimic version/PT; RTC binds ephemeral ports.
                if 53 in (packet.data.sport, packet.data.dport):
                    dns_collisions += 1
                    continue
                sequence, timestamp, ssrc = struct.unpack_from('!HII', payload, 2)
                groups[packet.data.sport, packet.data.dport, ssrc, timestamp].add(sequence)
                maximum = max(maximum, packet.len)
                packets += 1
    observations = [json.loads(line) for line in
                    (evidence_folder / 'network_observations.jsonl').read_text().splitlines()]
    if (not observations or maximum > 1500
            or expected_messages is not None and len(groups) != expected_messages):
        raise ValueError('Missing counter snapshots, oversized RTP, or wrong message count')
    qdiscs = [q for observation in observations for q in observation['qdisc']]
    before, after = udp_counters(observations[0]), udp_counters(observations[-1])
    return {'capture_files': manifest, 'rtp_payload_type': 109,
            'dns_packets_matching_rtp_prefix_excluded': dns_collisions,
            'captured_rtp_packets_before_deduplication': packets,
            'captured_unique_rtp_sequences_per_message': sum(map(len, groups.values())),
            'rtp_message_groups': len(groups), 'maximum_captured_rtp_ip_bytes': maximum,
            'network_snapshots': len(observations),
            'qdisc_max_observed_drops': {
                kind: max(q.get('drops', 0) for q in qdiscs if q['kind'] == kind)
                for kind in sorted({q['kind'] for q in qdiscs})},
            'udp_global_counter_deltas': {key: after[key] - before[key] for key in before},
            'initial_qdisc': observations[0]['qdisc'], 'final_qdisc': observations[-1]['qdisc'],
            'original_video_sessions': video_outcomes(evidence_folder / 'orchestrator.log'),
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
    parser.add_argument('--profile-root', help='Completed full-profile result root; reuse earlier baseline captures')
    parser.add_argument('--profile-captures', help='Capture root for the six new profile conditions')
    args = parser.parse_args()
    if bool(args.profile_root) != bool(args.profile_captures):
        parser.error('--profile-root and --profile-captures must be provided together')
    captures, results = Path(args.captures), Path(args.results)
    output = {}
    for mode in ['single', 'native4']:
        key = f'{mode}/no_emulation' if args.profile_root else f'baseline/{mode}/no_emulation'
        output[key] = audit(
            captures / 'baseline' / mode / 'no_emulation',
            results / 'minicpm_rtc_baseline_20261010' / mode / 'no_emulation', 125)
    if args.profile_root:
        for mode in ['single', 'native4']:
            for profile in ['lossy', '5qi_7', 'congested']:
                row = audit(Path(args.profile_captures) / mode / profile,
                            Path(args.profile_root) / mode / profile)
                if row['rtp_message_groups'] > 125:
                    raise ValueError('Unexpected extra RTP message identities')
                row.update(expected_feature_messages=125,
                           messages_without_any_captured_rtp=125 - row['rtp_message_groups'])
                output[f'{mode}/{profile}'] = row
    else:
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
