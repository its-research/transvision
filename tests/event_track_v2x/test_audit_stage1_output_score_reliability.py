"""Synthetic end-to-end audit fixtures and adversarial label mutations only."""
import gzip
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[2] / 'tools/event_track_v2x'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('full_score_audit', TOOLS / 'audit_stage1_output_score_reliability.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def box(tid, x, score=0.):
    return dict(track_id=tid, mean=[x, 0., 0., 4., 2., 1., 0., 0., 0.], score=score)


def fixture():
    g, p = [], []
    gb = [[box('g1', 0), box('g2', 10)], [box('g1', 0)], [], []]
    pb = [[box('p1', 0, .8), box('p2', 30, .2)], [], [box('p3', 30, 1.)], []]
    for i in range(4):
        g.append(dict(sid='s', fid=str(i), time=100000 * (i + 1), boxes=gb[i]))
        p.append(dict(sid='s', fid=str(i), time=100000 * (i + 1), boxes=pb[i]))
    rows = []
    for gf, pf in zip(g, p):
        for b in pf['boxes']:
            y = int(b['track_id'] == 'p1')
            rows.append(dict(sequence_id='s', frame_id=gf['fid'], box_reference_timestamp_us=gf['time'],
                             predicted_track_id=b['track_id'], emitted_score=b['score'], TP_label=y,
                             matched_GT_track_id='g1' if y else None, matched_BEV_IoU=1. if y else None))
    return g, p, rows


class CompleteAuditTests(unittest.TestCase):
    def call(self, mutate=None):
        g, p, rows = fixture()
        if mutate:
            mutate(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.jsonl.gz'
            with gzip.open(path, 'wt') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')
            stats = m.scalar_statistics([(.8, 1), (.2, 0), (1., 0)], 4, 3)
            expected = dict(aggregate=stats, per_sequence={'s': stats}, complete_sequence_frame_counts={'s': 4},
                            label_rows_sha256=m.sha(path), strict_replay_pass=False,
                            accepted_three_seed_statistics_eligible=False)
            with patch.object(m, 'load_frames', return_value=(g, p, {'s': 4})):
                return m.audit_run(dict(run_id='failed-fixture', strict_replay_pass=False), 'unused', 'unused', {},
                                   path, expected, 1e-10)

    def test_full_labels_empty_frames_infinite_NLL_and_failed_status_preserved(self):
        r = self.call()
        self.assertEqual(r['audited_frames'], 4)
        self.assertEqual(r['audited_prediction_labels'], 3)
        a = r['independently_recomputed_scalars']['aggregate']
        self.assertEqual((a['TP'], a['FP'], a['FN'], a['GT_objects']), (1, 2, 2, 3))
        self.assertEqual(a['recall'], 1 / 3)
        self.assertAlmostEqual(a['Brier'], .36)
        self.assertAlmostEqual(a['ECE'], 1.4 / 3)
        self.assertIsNone(a['NLL'])
        self.assertEqual(a['NLL_status'], 'positive_infinity')
        self.assertEqual(a['infinite_NLL_observations'], 1)
        self.assertAlmostEqual(a['finite_observation_NLL_mean'], -math.log(.8))
        self.assertTrue(r['original_failed_replay_status_preserved'])

    def test_bad_score_or_frame_or_identity_rejected(self):
        for key, value in [('emitted_score', .81), ('frame_id', 'wrong'), ('matched_GT_track_id', 'g2'),
                           ('emitted_score', True), ('matched_BEV_IoU', float('nan'))]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.call(lambda rows: rows[0].update({key: value}))

    def test_false_negative_optimal_match_rejected_independently(self):
        with self.assertRaisesRegex(ValueError, 'alternating assignment cycle'):
            self.call(lambda rows: rows[0].update(TP_label=0, matched_GT_track_id=None, matched_BEV_IoU=None))

    def test_missing_and_extra_rows_rejected(self):
        for mutate in [lambda rows: rows.pop(), lambda rows: rows.append(dict(rows[-1]))]:
            with self.assertRaises(ValueError):
                self.call(mutate)

    def test_scalar_comparison_rejects_count_type_or_finite_NLL_substitution(self):
        for a, b in [(dict(TP=1), dict(TP=True)), (dict(NLL=None), dict(NLL=100.)),
                     (dict(Brier=.1), dict(Brier=.2)), (dict(ECE=.1), dict(ECE=float('nan')))]:
            with self.assertRaises(ValueError):
                m.compare(a, b, 1e-10)


if __name__ == '__main__':
    unittest.main()
