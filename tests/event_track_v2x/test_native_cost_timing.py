"""Software-only guards; no GPU, dataset, or scientific cost qualification."""
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools.event_track_v2x.native_cost_timing import NativeFrameTiming, instrument_author_source, execute_author_source


SOURCE = b'''def train_one_epoch(tracker, fail=False):
    seq_name = '0048'
    min_frame, max_frame = 0, 2
    values = []
    for frame in range(min_frame, max_frame + 1):
        values.append(frame)
        result = tracker.track_multi_sensor_differentiable_kalman_filter(frame)
        if fail and frame == 1:
            raise ValueError('native failure')
        values.append(result)
    return values
'''


class Tracker:
    def track_multi_sensor_differentiable_kalman_filter(self, frame):
        return frame + 10


class NativeTimingTests(unittest.TestCase):
    def execute(self, fail=False):
        output = io.StringIO()
        ticks = iter(range(100))
        syncs = []
        timer = NativeFrameTiming(output, lambda: syncs.append(True), lambda: next(ticks))
        namespace = {'_rbf_native_timer': timer}
        code = instrument_author_source(SOURCE, 'author.py', hashlib.sha256(SOURCE).hexdigest())
        exec(code, namespace)
        if fail:
            with self.assertRaisesRegex(ValueError, 'native failure'):
                namespace['train_one_epoch'](Tracker(), True)
        else:
            original = {}
            exec(SOURCE, original)
            self.assertEqual(namespace['train_one_epoch'](Tracker()), original['train_one_epoch'](Tracker()))
        return timer, [json.loads(line) for line in output.getvalue().splitlines()], syncs

    def test_full_frame_coverage_and_unchanged_native_return(self):
        timer, rows, syncs = self.execute()
        self.assertEqual([r['frame'] for r in rows], [0, 1, 2])
        self.assertEqual([(r['core_seconds'], r['frame_seconds']) for r in rows], [(1, 3)] * 3)
        self.assertEqual((timer.completed, timer.failed, len(syncs)), (3, 0, 12))

    def test_native_failure_stays_failed_and_is_not_complete(self):
        timer, rows, _ = self.execute(True)
        self.assertEqual([r['status'] for r in rows], ['returned', 'failed'])
        self.assertEqual((timer.completed, timer.failed), (1, 1))
        self.assertIsNone(timer.current)

    def test_rejects_source_and_ambiguous_core(self):
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            instrument_author_source(SOURCE, 'author.py', '0' * 64)
        changed = SOURCE.replace(b'        if fail', b'        extra = tracker.track_multi_sensor_differentiable_kalman_filter(frame)\n        if fail')
        with self.assertRaisesRegex(ValueError, 'unique top-level'):
            instrument_author_source(changed, 'author.py', hashlib.sha256(changed).hexdigest())

    def test_main_module_and_argv_semantics_restored_on_native_exception(self):
        original_module, original_argv = sys.modules['__main__'], list(sys.argv)
        source = SOURCE + b'''if __name__ == '__main__':
    import __main__, sys
    assert __main__.__file__ == sys.argv[0]
    assert getattr(__main__, '_rbf_native_timer') is globals()['_rbf_native_timer']
    raise RuntimeError('author entry failure')
'''
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / 'author.py'
            path.write_bytes(source)
            with self.assertRaisesRegex(RuntimeError, 'author entry failure'):
                execute_author_source(path, hashlib.sha256(source).hexdigest(), object())
        self.assertIs(sys.modules['__main__'], original_module)
        self.assertEqual(sys.argv, original_argv)

    def test_full_warmup_measurement_pair_without_GPU_or_scientific_inputs(self):
        # This tests native-entry integration only, with synthetic statements
        # and a fake CUDA clock. It produces no scientific prediction or cost.
        from tools.event_track_v2x import native_cost_resource_guard
        with mock.patch.dict(sys.modules, {'native_cost_timing': sys.modules[NativeFrameTiming.__module__],
                                         'native_cost_resource_guard': native_cost_resource_guard}):
            from tools.event_track_v2x import run_DMSTrack_native_cost_entry as entry
        author = SOURCE.replace(b'max_frame = 0, 2', b'max_frame = 0, 1992') + b'''if __name__ == '__main__':
    class Tracker:
        def track_multi_sensor_differentiable_kalman_filter(self, frame):
            return frame
    result = train_one_epoch(Tracker())
    assert len(result) == 3986
'''
        original = b'''import json, pathlib, runpy, sys
b = json.loads(pathlib.Path(sys.argv[1]).read_bytes())
sys.argv = b['native_argv'][1:]
runpy.run_path(b['author_source'] + '/DMSTrack/main_dkf.py', run_name='__main__')
'''
        cuda = types.SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                                     max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
        fake_torch = types.SimpleNamespace(cuda=cuda)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / 'author/DMSTrack/main_dkf.py'
            source.parent.mkdir(parents=True)
            source.write_bytes(author)
            original_path = root / 'original_entry.py'
            original_path.write_bytes(original)
            pair = root / 'pair'
            pair.mkdir()
            binding = dict(original_native_binding=dict(
                entry_source_sha256=hashlib.sha256(original).hexdigest(),
                author_source=str(root / 'author'),
                original_source_files={'DMSTrack/main_dkf.py': {'sha256': hashlib.sha256(author).hexdigest()}},
                native_argv=['python', 'main_dkf.py', '--save_dir_prefix', '/unused'],
                CPU_threads=2, original_native_num_epochs=0, GPU_UUID='synthetic',
                GPU_memory_cap_bytes=1, CPU_memory_cap_bytes=1),
                pair_output=str(pair), repetition_index=0, original_entry_path=str(original_path),
                container_cgroup_mount='/sys/fs/cgroup',
                owned_process_guard=dict(GPU_UUID='synthetic', CPU_threads=2,
                    GPU_process_resident_cap_bytes=1, CPU_process_resident_cap_bytes=1,
                    live_monitor_PID=1, original_deadline_controller_restarted=False))
            path = root / 'binding.json'
            path.write_text(json.dumps(binding))
            old_argv = list(sys.argv)
            old_main = sys.modules['__main__']
            cgroup = dict(cgroup_accounted_peak_bytes=0, memory_events={'oom': 0, 'oom_kill': 0, 'max': 0})
            process = dict(process_RSS_high_water_bytes=0)
            with mock.patch.object(sys, 'platform', 'linux'), mock.patch.dict(sys.modules, {'torch': fake_torch}), \
                 mock.patch.object(entry, 'read_cgroup_memory', return_value=cgroup), \
                 mock.patch.object(entry, 'read_process_rss', return_value=process):
                entry.main(path)
            receipt = json.loads((pair / 'native-cost-pair-returned.json').read_bytes())
            self.assertEqual([x['phase'] for x in receipt['phases']], ['warmup', 'measured'])
            self.assertFalse(receipt['formal_cost_accepted'])
            for phase in ('warmup', 'measured'):
                rows = [json.loads(x) for x in (pair / phase / 'native-frame-times.jsonl').read_text().splitlines()]
                self.assertEqual([r['frame'] for r in rows], list(range(1993)))
            self.assertEqual(sys.argv, old_argv)
            self.assertIs(sys.modules['__main__'], old_main)


if __name__ == '__main__':
    unittest.main()
