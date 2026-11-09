"""Independent synthetic optimality checker against exhaustive assignments."""
import importlib.util
import itertools
import math
from pathlib import Path
import random
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'tools/event_track_v2x/stage1_score_matching_audit.py'
spec = importlib.util.spec_from_file_location('matching_audit', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def objective(matrix, labels):
    return sum(i is not None for i in labels), math.fsum(matrix[i][j] for j, i in enumerate(labels) if i is not None)


class AssignmentAuditTests(unittest.TestCase):
    def test_all_feasible_small_assignments_vs_exhaustive_objective(self):
        rng = random.Random(1337)
        checked = 0
        for ng in range(4):
            for np in range(4):
                for _ in range(12):
                    matrix = [[rng.choice([0., .49, .5, .6, .9, 1.]) for _ in range(np)] for _ in range(ng)]
                    feasible = []
                    for labels in itertools.product([None] + list(range(ng)), repeat=np):
                        matched = [x for x in labels if x is not None]
                        if len(matched) == len(set(matched)) and all(i is None or matrix[i][j] >= .5 for j, i in enumerate(labels)):
                            feasible.append(labels)
                    best = max(objective(matrix, labels) for labels in feasible)
                    for labels in feasible:
                        actual = objective(matrix, labels)
                        optimal = actual[0] == best[0] and abs(actual[1] - best[1]) < 1e-12
                        if optimal:
                            self.assertTrue(m.audit_assignment(matrix, labels)['no_improving_cycle'])
                        else:
                            with self.assertRaises(ValueError):
                                m.audit_assignment(matrix, labels)
                        checked += 1
        self.assertGreater(checked, 500)

    def test_free_column_improvement_is_detected(self):
        with self.assertRaises(ValueError):
            m.audit_assignment([[.5, 1.]], [0, None])

    def test_cardinality_and_alternating_swap_improvements_detected(self):
        with self.assertRaises(ValueError):
            m.audit_assignment([[1., .5], [.5, 0.]], [0, None])
        with self.assertRaises(ValueError):
            m.audit_assignment([[.5, .9], [.9, .5]], [0, 1])
        self.assertTrue(m.audit_assignment([[.5, .9], [.9, .5]], [1, 0])['no_improving_cycle'])

    def test_infeasible_matches_and_invalid_tolerance(self):
        for matrix, labels in [([[1., 1.]], [0, 0]), ([[.49]], [0]), ([[1.]], [True]), ([], [0])]:
            with self.assertRaises(ValueError):
                m.audit_assignment(matrix, labels)
        with self.assertRaises(ValueError):
            m.audit_assignment([], [], tolerance=-1)

    def test_empty_and_exact_ties(self):
        self.assertTrue(m.audit_assignment([], [None])['no_improving_cycle'])
        self.assertTrue(m.audit_assignment([[]], [])['no_improving_cycle'])
        self.assertTrue(m.audit_assignment([[1., 1.]], [0, None])['no_improving_cycle'])
        self.assertTrue(m.audit_assignment([[1., 1.]], [None, 0])['no_improving_cycle'])


if __name__ == '__main__':
    unittest.main()
