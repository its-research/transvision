"""Synthetic checks only; no experiment inputs or local scientific replay."""
import importlib.util
import itertools
import math
from pathlib import Path
import random
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'tools/event_track_v2x/stage1_output_score_reliability.py'
spec = importlib.util.spec_from_file_location('score_reliability', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def objective(matrix, labels):
    return (sum(x is not None for x in labels),
            math.fsum(matrix[x][i] for i, x in enumerate(labels) if x is not None))


def exhaustive(matrix, predictions):
    best = (0, 0.)
    for candidate in itertools.product([None] + list(range(len(matrix))), repeat=predictions):
        assigned = [x for x in candidate if x is not None]
        if len(assigned) != len(set(assigned)):
            continue
        if any(x is not None and matrix[x][i] < .5 for i, x in enumerate(candidate)):
            continue
        best = max(best, objective(matrix, candidate))
    return best


class ReliabilityTests(unittest.TestCase):
    def test_assignment_matches_exhaustive_small_rectangular_cases(self):
        rng = random.Random(7)
        for ng in range(5):
            for np in range(5):
                for _ in range(15):
                    matrix = [[rng.choice([0., .49, .5, .6, .9, 1.]) for _ in range(np)] for _ in range(ng)]
                    actual = objective(matrix, m.assignment_labels(matrix, np))
                    expected = exhaustive(matrix, np)
                    self.assertEqual(actual[0], expected[0])
                    self.assertAlmostEqual(actual[1], expected[1], places=13)

    def test_cardinality_precedes_greedy_largest_IoU(self):
        self.assertEqual(m.assignment_labels([[1., .5], [.5, 0.]], 2), [1, 0])

    def test_duplicate_predictions_only_one_positive(self):
        labels = m.assignment_labels([[1., 1.]], 2)
        self.assertEqual(sum(x is not None for x in labels), 1)
        self.assertEqual(labels, m.assignment_labels([[1., 1.]], 2))

    def test_empty_and_gate_boundaries(self):
        self.assertEqual(m.assignment_labels([], 2), [None, None])
        self.assertEqual(m.assignment_labels([[]], 0), [])
        self.assertEqual(m.assignment_labels([[.499999, .5]], 2), [None, 0])

    def test_invalid_matching_values_rejected(self):
        for matrix, count in [([[float('nan')]], 1), ([[1.1]], 1), ([[.5]], 2), ([], -1)]:
            with self.assertRaises(ValueError):
                m.assignment_labels(matrix, count)

    def test_NLL_preserves_infinite_endpoints_without_clipping(self):
        a = m.ScoreAccumulator()
        for score, label in [(0., 0), (1., 1), (0., 1), (1., 0)]:
            a.add(score, label)
        r = a.result()
        self.assertIsNone(r['NLL'])
        self.assertEqual(r['NLL_status'], 'positive_infinity')
        self.assertEqual(r['infinite_NLL_observations'], 2)
        self.assertEqual(r['finite_observation_NLL_mean'], 0.)
        self.assertEqual(r['Brier'], .5)
        self.assertEqual(r['ECE'], .5)
        self.assertFalse(r['clipping_applied'])

    def test_finite_scores_bins_and_explicit_missed_GT(self):
        a = m.ScoreAccumulator()
        a.gt, a.frames = 3, 2
        a.add(.2, 0)
        a.add(.8, 1)
        r = a.result()
        self.assertAlmostEqual(r['Brier'], .04)
        self.assertAlmostEqual(r['NLL'], -math.log(.8))
        self.assertAlmostEqual(r['ECE'], .2)
        self.assertEqual(r['FN'], 2)
        self.assertEqual(r['recall'], 1 / 3)
        self.assertEqual(r['reliability_bins'][2]['count'], 1)
        self.assertEqual(r['reliability_bins'][8]['count'], 1)

    def test_empty_predictions_distinct_from_perfect_calibration(self):
        a = m.ScoreAccumulator()
        a.gt, a.frames = 2, 1
        r = a.result()
        self.assertEqual(r['FN'], 2)
        self.assertEqual(r['recall'], 0.)
        for k in ['NLL', 'Brier', 'ECE']:
            self.assertIsNone(r[k])
        self.assertEqual(r['NLL_status'], 'empty')

    def test_invalid_scores_rejected(self):
        a = m.ScoreAccumulator()
        for score, label in [(True, 1), (-.1, 0), (1.1, 1), (float('nan'), 0), (.5, True)]:
            with self.assertRaises(ValueError):
                a.add(score, label)

    def test_geometry_and_strict_ROI_fixture(self):
        box = [100000., 100000., 0., 4., 2., 1., .7, 0., 0.]
        self.assertAlmostEqual(m.bev_iou(box, box), 1.)
        remote = list(box)
        remote[0] += 100
        self.assertEqual(m.bev_iou(box, remote), 0.)
        items = [dict(track_id='a', class_label='car', mean=[49.999, 0., 0., 4., 2., 1., 0., 0., 0.], score=.5),
                 dict(track_id='b', class_label='car', mean=[50., 0., 0., 4., 2., 1., 0., 0., 0.], score=.5)]
        self.assertEqual([x['track_id'] for x in m.car_rows(items, [0., 0., 0.])], ['a'])


    def test_adjacent_float_bin_boundaries_obey_declared_intervals(self):
        for i in range(1, 10):
            edge = i / 10
            for score, index in [(math.nextafter(edge, -math.inf), i - 1),
                                 (edge, i), (math.nextafter(edge, math.inf), i)]:
                a = m.ScoreAccumulator()
                a.add(score, 0)
                bins = a.result()['reliability_bins']
                self.assertEqual([b['index'] for b in bins if b['count']], [index])


    def test_population_count_gate_uses_exact_native_post_ROI_counts(self):
        result = dict(GT_objects=3, emitted_predictions=2)
        sequences = {'s': dict(frames=4, GT_objects=3, emitted_predictions=2)}
        metrics = {'trackeval': {'car': {'sequences': {'s': dict(frames=4, gt_objects=3, predicted_objects=2)}}}}
        # The pre-ROI car census may exceed the scientifically evaluated population.
        m.verify_population(result, sequences, metrics, 5)
        for wrong in [dict(frames=4, GT_objects=3, emitted_predictions=1),
                      dict(frames=3, GT_objects=3, emitted_predictions=2),
                      dict(frames=4, GT_objects=2, emitted_predictions=2)]:
            with self.assertRaises(ValueError):
                m.verify_population(result, {'s': wrong}, metrics, 5)
        with self.assertRaises(ValueError):
            m.verify_population(result, {'wrong': sequences['s']}, metrics, 5)
        with self.assertRaises(ValueError):
            m.verify_population(dict(GT_objects=3, emitted_predictions=3), sequences, metrics, 5)
        with self.assertRaises(ValueError):
            m.verify_population(result, sequences, metrics, 1)


if __name__ == '__main__':
    unittest.main()
