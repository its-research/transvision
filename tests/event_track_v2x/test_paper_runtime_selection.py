"""Prevent an exclusive experiment from silently entering the legacy runtime."""
import unittest
from dataclasses import asdict

from transvision.models.event_track_v2x import exclusive_paper_runtime, paper_runtime
from transvision.models.event_track_v2x.paper_runtime_selection import allocation_configuration, runtime


class RuntimeSelectionTest(unittest.TestCase):
    def test_exclusive_teacher_and_learned_keep_every_limit(self):
        for policy in ('teacher', 'learned', 'bound'):
            config = exclusive_paper_runtime.default_configuration(allocation=policy)
            self.assertIs(runtime(config), exclusive_paper_runtime)
            bound = asdict(allocation_configuration(config))
            self.assertEqual(bound.pop('state'), config['state'])
            self.assertEqual(bound, config['limits'])

    def test_historical_configuration_retains_runtime(self):
        config = paper_runtime.default_configuration(allocation='teacher')
        self.assertIs(runtime(config), paper_runtime)
        bound = asdict(allocation_configuration(config))
        self.assertEqual(bound.pop('state'), config['state'])
        self.assertEqual(bound, config['limits'])

    def test_unknown_or_incompatible_backend_is_rejected(self):
        for config in ({'backend': 'typo', 'method': 'rbf'},
                       {'backend': exclusive_paper_runtime.BACKEND, 'method': 'mht'}):
            with self.assertRaises(ValueError):
                runtime(config)


if __name__ == '__main__':
    unittest.main()
