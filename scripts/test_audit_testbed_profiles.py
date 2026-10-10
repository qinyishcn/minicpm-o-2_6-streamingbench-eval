"""Regression checks for fixed-denominator scoring of failed RTC questions.

Run from the patched Testbed environment; no model/GPU is used.
"""
import copy
import json
import unittest
from pathlib import Path

from audit_testbed_profiles import saved_answers, score_rows


class MissingAnswerAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.expected = {q['question_id']: q for q in json.loads((root /
            'results/q4_k_m_single_native/questions_manifest.json').read_text(encoding='utf-8'))}
        cls.row = json.loads((root / 'results/testbed_rtc_baseline_20261010/dataset/single/'
            'sample_1/questions_results_no_emulation.json').read_text(encoding='utf-8'))[0]['questions'][0]

    def test_absent_question_records_keep_125_denominator(self):
        score = score_rows(saved_answers([], self.expected, 'single'), self.expected)
        self.assertEqual((score['questions'], score['missing_answers'], score['accuracy_percent']),
                         (125, 125, 0.0))

    def test_unanswered_placeholder_is_missing_not_a_model_record(self):
        unanswered = copy.deepcopy(next(q for q in self.expected.values()
            if q['question_id'] != self.row['question_id']))
        rows = saved_answers([self.row, unanswered], self.expected, 'single')
        score = score_rows(rows, self.expected)
        self.assertEqual((score['answers_received'], score['missing_answers']), (1, 124))

    def test_answer_without_feature_evidence_is_rejected(self):
        row = copy.deepcopy(self.row)
        row.pop('minicpm_details')
        with self.assertRaises(ValueError):
            saved_answers([row], self.expected, 'single')

    def test_changed_ground_truth_is_rejected(self):
        row = copy.deepcopy(self.row)
        row['answer'] = 'A' if row['answer'] != 'A' else 'B'
        with self.assertRaises(ValueError):
            saved_answers([row], self.expected, 'single')

    def test_duplicate_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            saved_answers([self.row, self.row], self.expected, 'single')

    def test_impossible_loss_layout_is_rejected(self):
        row = copy.deepcopy(self.row)
        row['minicpm_details']['loss']['missing_packet_indices'] = [999999]
        with self.assertRaises(ValueError):
            saved_answers([row], self.expected, 'single')


if __name__ == '__main__':
    unittest.main()
