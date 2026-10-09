import unittest

from evaluate import frame_targets, parse_answer, select_questions, make_prompt
from native_frames import preceding_positions


class EvaluationChecks(unittest.TestCase):
    def test_exact_subset(self):
        rows = select_questions('data/Real_Time_Visual_Understanding.csv')
        self.assertEqual(len(rows), 125)
        self.assertEqual(sorted({r['video_id'] for r in rows}), list(range(1, 26)))
        self.assertTrue(all(sum(r['video_id'] == n for r in rows) == 5 for n in range(1,26)))

    def test_past_only_one_fps(self):
        self.assertEqual(frame_targets(4), [0.9, 1.9, 2.9, 3.9])
        targets = frame_targets(128)
        self.assertEqual(len(targets), 60)
        self.assertAlmostEqual(targets[0], 68.9)
        self.assertAlmostEqual(targets[-1], 127.9)
        self.assertTrue(all(68 <= t < 128 for t in targets))

    def test_strict_answers(self):
        for s, answer in [('A','A'), (' B. ','B'), ('(C)','C'), ('D\n','D')]:
            self.assertEqual(parse_answer(s), answer)
        for s in ['', 'A or B', 'The image shows a dog.', 'The answer is B.', 'AB']:
            self.assertIsNone(parse_answer(s))

    def test_labels_not_used_in_prompt(self):
        row = select_questions('data/Real_Time_Visual_Understanding.csv')[0]
        first = make_prompt(row)
        row['answer'] = 'X_SECRET_LABEL'
        self.assertEqual(first, make_prompt(row))
        self.assertNotIn('X_SECRET_LABEL', first)

    def test_native_frames_are_adjacent_and_causal(self):
        times=[i/30 for i in range(65)]
        self.assertEqual(preceding_positions(times,2,1),[60])
        self.assertEqual(preceding_positions(times,2,4),[57,58,59,60])
        self.assertEqual(preceding_positions(times,2.01,4),[57,58,59,60])
        variable=[0,0.031,0.066,0.101,0.135,0.170]
        self.assertEqual(preceding_positions(variable,0.15,4),[1,2,3,4])
        self.assertTrue(all(variable[i]<=0.15 for i in preceding_positions(variable,0.15,4)))


if __name__ == '__main__':
    unittest.main()
