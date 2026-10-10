"""Compare all saved RTC answers/logits with the independent unsplit helper."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    reference = json.loads((root / 'linux_unsplit_reference_full.json').read_text(encoding='utf-8'))
    indexed = {(row['mode'], row['question_id']): row for row in reference}
    if len(indexed) != len(reference) or len(reference) != 250:
        raise ValueError('Expected 250 unique unsplit reference answers')
    checks = []
    for mode in ['single', 'native4']:
        for video in range(1, 26):
            path = root / 'dataset' / mode / f'sample_{video}' / 'questions_results_no_emulation.json'
            questions = json.loads(path.read_text(encoding='utf-8'))[0]['questions']
            if len(questions) != 5:
                raise ValueError('Expected five questions per video')
            for row in questions:
                expected = indexed[mode, row['question_id']]
                actual = row['minicpm_details']['result']
                checks.append({'mode': mode, 'question_id': row['question_id'], 'label': row['answer'],
                    'rtc_text': actual['text'], 'unsplit_text': expected['text'],
                    'answer_equal': actual['text'] == expected['text'],
                    'token_ids_equal': actual['token_ids'] == expected['token_ids'],
                    'first_logits_sha256_equal': actual['first_logits_sha256'] == expected['first_logits_sha256'],
                    'rtc_first_logits_sha256': actual['first_logits_sha256'],
                    'unsplit_first_logits_sha256': expected['first_logits_sha256']})
    if len({(r['mode'], r['question_id']) for r in checks}) != 250:
        raise ValueError('Duplicate RTC question identities')
    fields = ['answer_equal', 'token_ids_equal', 'first_logits_sha256_equal']
    summary = {mode: {'questions': 125, **{
        key: sum(row[key] for row in checks if row['mode'] == mode) for key in fields}}
        for mode in ['single', 'native4']}
    passed = all(all(row[key] for key in fields) for row in checks)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'parity.json').write_text(json.dumps({
        'status': 'passed' if passed else 'mismatch', 'summary': summary,
        'reference': 'Same Linux CUDA model/frames/prompt, upstream mtmd_helper_eval_chunks; '
                     'does not use manual split embedding injection or RTC.',
        'definition': 'Exact answer, full generated token IDs and full first-step logits byte SHA-256.',
        'checks': checks}, indent=2), encoding='utf-8')
    with (output / 'parity.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    print(json.dumps({'status': 'passed' if passed else 'mismatch', 'summary': summary}, indent=2))
    if not passed:
        raise SystemExit('RTC/unsplit parity mismatch; retain and investigate both raw results')


if __name__ == '__main__':
    main()
