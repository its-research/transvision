"""Synthetic Linux interface probes; no resource measurement or GPU activity."""
import pathlib
import tempfile
import unittest

from tools.event_track_v2x.native_cost_resource_guard import (
    parse_GPU_processes, read_cgroup_memory, read_process_rss, resource_violations)


class ResourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        for name, value in {'memory.current': '100', 'memory.peak': '200', 'memory.max': '1024',
                            'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n',
                            'memory.stat': 'anon 50\nfile 50\n'}.items():
            (self.root / name).write_text(value)
        (self.root / 'status').write_text('Name: synthetic\nVmRSS: 1 kB\nVmHWM: 1 kB\n')

    def tearDown(self):
        self.temporary.cleanup()

    def test_context_resident_and_CPU_accounting_are_distinct(self):
        cgroup = read_cgroup_memory(self.root, 1024)
        process = read_process_rss(self.root / 'status')
        GPU = parse_GPU_processes('GPU-owned, 123, 2\nGPU-other, 99, 999\n', 'GPU-owned', 123)
        self.assertEqual(process['process_RSS_high_water_bytes'], 1024)
        self.assertEqual(GPU['owned_GPU_processes'][0]['GPU_resident_bytes'], 2 * 1024**2)
        self.assertEqual(resource_violations(cgroup, process, GPU, 1024, 2 * 1024**2), [])
        self.assertTrue(GPU['GPU_process_resident_observation_is_sampled'])

    def test_foreign_GPU_process_and_OOM_are_violations(self):
        (self.root / 'memory.events').write_text('max 1\noom 0\noom_kill 1\n')
        cgroup = read_cgroup_memory(self.root, 1024)
        GPU = parse_GPU_processes('GPU-owned, 123, 3\nGPU-owned, 456, 1\n', 'GPU-owned', 123)
        violations = resource_violations(cgroup, read_process_rss(self.root / 'status'), GPU, 1024, 2 * 1024**2)
        self.assertEqual(set(violations), {'owned_cgroup_hard_limit_or_OOM_event',
                                         'other_process_appeared_on_same_physical_GPU',
                                         'owned_GPU_process_total_resident_cap'})

    def test_cap_change_or_unavailable_counters_are_not_accepted(self):
        (self.root / 'memory.max').write_text('max')
        with self.assertRaises(ValueError):
            read_cgroup_memory(self.root, 1024)
        with self.assertRaises(ValueError):
            parse_GPU_processes('GPU-owned, 123, [Not Supported]', 'GPU-owned', 123)

    def test_process_high_water_cap_is_independent_of_cgroup_charge(self):
        (self.root / 'status').write_text('VmRSS: 1 kB\nVmHWM: 2 kB\n')
        violations = resource_violations(read_cgroup_memory(self.root, 1024),
                                        read_process_rss(self.root / 'status'),
                                        parse_GPU_processes('', 'GPU-owned', 123), 1024, 100)
        self.assertEqual(violations, ['owned_native_process_RSS_high_water_cap'])


if __name__ == '__main__':
    unittest.main()
