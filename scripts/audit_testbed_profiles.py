"""Audit original RTC results while retaining network-failed, missing answers.

Run with the published Testbed patch on PYTHONPATH. Model/network code is
unchanged. Missing answers count wrong with the fixed manifest denominator;
their unobserved received features are never fabricated or hash-verified.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from minicpm_split.rtc_report import mean_field, score_rows, validate_loss


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def saved_answers(rows, expected, mode):
    seen, received = set(), []
    for row in rows:
        qid = row['question_id']
        if qid in seen or qid not in expected:
            raise ValueError('Duplicate or unexpected question identity')
        seen.add(qid)
        question = expected[qid]
        for key in ['answer', 'question', 'options', 'timestamp_seconds', 'video_id']:
            if row[key] != question[key]:
                raise ValueError(f'Changed benchmark field: {qid}/{key}')
        detail = row.get('minicpm_details')
        if detail is None:
            if row.get('minicpm') is not None:
                raise ValueError('Answer without auditable model feature record')
            continue
        if detail['question_id'] != qid or detail['mode'] != mode:
            raise ValueError('Feature identity mismatch')
        if detail['raw_response'] != row['minicpm']:
            raise ValueError('Raw response mismatch')
        if 'result' in detail and detail['result']['text'] != detail['raw_response']:
            raise ValueError('Native result mismatch')
        validate_loss(detail)
        received.append(row)
    return received


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--questions')
    parser.add_argument('--modes', nargs='+', default=['single', 'native4'])
    parser.add_argument('--profiles', nargs='+', default=['no_emulation', 'lossy', '5qi_7', 'congested'])
    parser.add_argument('--reencode', action='store_true')
    args = parser.parse_args()
    if (len(set(args.modes)) != len(args.modes) or not set(args.modes) <= {'single', 'native4'}
            or len(set(args.profiles)) != len(args.profiles) or 'no_emulation' not in args.profiles):
        parser.error('Expected unique modes/profiles including no_emulation')
    root, output = Path(args.root), Path(args.output)
    summaries, checks, configs = [], [], {}
    for mode in args.modes:
        config = read(root / 'configs' / f'{mode}_model.json')
        configs[mode] = config
        manifest = Path(args.questions or config['questions']).read_bytes()
        if hashlib.sha256(manifest).hexdigest() != config['questions_sha256']:
            raise ValueError('Manifest hash mismatch')
        questions = json.loads(manifest)
        expected = {q['question_id']: q for q in questions}
        if len(expected) != len(questions) or len(expected) != 125:
            raise ValueError('Expected the fixed 125-question manifest')
        sources = {}
        for profile in args.profiles:
            raw = []
            for video in range(1, 26):
                path = root / 'dataset' / mode / f'sample_{video}' / f'questions_results_{profile}.json'
                if path.exists():
                    raw.extend(read(path)[0]['questions'])
            rows = saved_answers(raw, expected, mode)
            score = score_rows(rows, expected)
            if profile == 'no_emulation' and score['missing_answers']:
                raise ValueError('Complete baseline required')
            sources[profile] = {r['question_id']: r for r in rows}
            sums = {name: sum(r['minicpm_details']['loss'][key] for r in rows) for name, key in [
                ('feature_packets', 'total_packets'), ('missing_feature_packets', 'missing_packets'),
                ('zeroed_feature_bytes', 'zeroed_bytes')]}
            sums['feature_bytes'] = sum(r['minicpm_details']['feature_bytes'] for r in rows)
            summaries.append({'mode': mode, 'profile': profile, **score, **sums,
                'feature_stat_answer_records': len(rows),
                'feature_statistics_scope': 'saved server answer records only; absent inputs are unknown',
                'missing_feature_percent': 100 * sums['missing_feature_packets'] / sums['feature_packets']
                    if sums['feature_packets'] else None,
                'zeroed_feature_percent': 100 * sums['zeroed_feature_bytes'] / sums['feature_bytes']
                    if sums['feature_bytes'] else None,
                'mean_edge_encode_seconds': mean_field(rows, 'edge_encode_seconds'),
                'mean_server_inference_seconds': mean_field(rows, 'server_inference_seconds'),
                'mean_metadata_to_result_seconds': mean_field(rows, 'metadata_to_result_seconds')})
        vision = None
        if args.reencode:
            from minicpm_split.native import EdgeVision
            vision = EdgeVision(config['runtime'], config['tokenizer'], config['projector'])
        try:
            for qid, question in expected.items():
                baseline = sources['no_emulation'][qid]['minicpm_details']
                payload = None
                if vision:
                    from minicpm_split.client import PROMPT, load_frames
                    folder = Path(config['frames']) / qid.replace(' ', '_')
                    frames = load_frames(folder, question, mode)
                    if frames != baseline['frames']:
                        raise ValueError('Baseline native frame-cache mismatch')
                    _, payload = vision.encode([folder / f['file'] for f in frames],
                        PROMPT.format(question['question'], *question['options']))
                for profile in args.profiles:
                    row = sources[profile].get(qid)
                    if row is None:
                        checks.append({'mode': mode, 'profile': profile, 'question_id': qid,
                            'answer_received': False, 'zero_fill_hash_reconstructed': None,
                            'reason': 'No saved server answer/feature record; counted wrong'})
                        continue
                    detail = row['minicpm_details']
                    frames = detail['frames']
                    indices = [f['frame_index'] for f in frames]
                    if (len(frames) != (1 if mode == 'single' else 4)
                            or frames != baseline['frames']
                            or detail['original_feature_sha256'] != baseline['original_feature_sha256']
                            or indices != list(range(indices[0], indices[0] + len(frames)))
                            or any(f['actual_seconds'] > question['timestamp_seconds'] + 1e-6 for f in frames)):
                        raise ValueError(f'Native visual input mismatch: {qid}/{profile}')
                    if payload is not None:
                        if hashlib.sha256(payload).hexdigest() != detail['original_feature_sha256']:
                            raise ValueError('Re-encoded source feature hash mismatch')
                        damaged = bytearray(payload)
                        size = detail['loss']['feature_payload_per_packet']
                        for index in detail['loss']['missing_packet_indices']:
                            begin, end = index * size, min((index + 1) * size, len(damaged))
                            damaged[begin:end] = bytes(end - begin)
                        if hashlib.sha256(damaged).hexdigest() != detail['feature_sha256']:
                            raise ValueError('Actual zero-filled input hash mismatch')
                    checks.append({'mode': mode, 'profile': profile, 'question_id': qid,
                        'answer_received': True, 'original_feature_sha256': detail['original_feature_sha256'],
                        'feature_sha256': detail['feature_sha256'], 'missing_packets': detail['loss']['missing_packets'],
                        'zero_fill_hash_reconstructed': payload is not None})
        finally:
            if vision:
                vision.close()
    answers = sum(r['answers_received'] for r in summaries)
    audit = {'status': 'passed', 'conditions': len(summaries), 'question_slots': 125 * len(summaries),
        'answers': answers, 'missing_answers': 125 * len(summaries) - answers,
        'reencoded_feature_audit': args.reencode, 'frame_cache_verified': args.reencode,
        'distinct_input_sets_reencoded': 125 * len(args.modes) if args.reencode else 0,
        'verified_received_feature_records': answers if args.reencode else 0,
        'source_configs': configs, 'summary': summaries, 'question_checks': checks,
        'scoring': 'Fixed 125 denominator; no server answer counts wrong; no inferred missing feature hashes',
        'feature_statistics': 'Only saved server answer records; not all attempted questions',
        'client_completion': 'Original testbed_report.json separately measures all-five-answer client completion'}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    with (output / 'condition_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / 'report_zh.md').write_text(
        '# 完整网络条件结果审计\n\n'
        f'{audit["question_slots"]}个固定题位，{answers}条已保存服务器答案，'
        f'{audit["missing_answers"]}条缺失计错。\n\n'
        f'重新编码：{args.reencode}；实际已保存输入的特征哈希核验数：'
        f'{audit["verified_received_feature_records"]}。没有记录的特征不可核验，未推定其内容或丢包率。\n',
        encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k not in ['source_configs', 'question_checks']}, indent=2))


if __name__ == '__main__':
    main()
