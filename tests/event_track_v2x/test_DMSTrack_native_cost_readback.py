"""Readback failure probes with tiny synthetic outputs; no scientific stats."""
import hashlib
import json
import pathlib
import tempfile
import unittest

from tools.event_track_v2x.verify_DMSTrack_native_cost_phase import verify_phase, latency_summary, verify_resource_samples


class ReadbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.native = self.root / 'native'
        self.native.mkdir()
        self.manifest = {'members': {}}
        self.rows = []
        for sequence, frames in [('0000', 2), ('0001', 1)]:
            for frame in range(frames):
                name = 'trk_withid_0/%s/%06d.txt' % (sequence, frame)
                path = self.native / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic %s %d\n' % (sequence, frame))
                raw = path.read_bytes()
                self.manifest['members']['native/' + name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                self.rows.append(dict(sequence=sequence, frame=frame, status='returned', core_seconds=1, frame_seconds=2))
        self.save_rows()
        (self.root / 'native-cost-phase-observation.json').write_text(json.dumps(dict(
            phase='measured', completed_frame_rows=3, failed_frame_rows=0, full_entry_wall_seconds=7)))

    def tearDown(self):
        self.temporary.cleanup()

    def save_rows(self):
        (self.root / 'native-frame-times.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in self.rows))

    def verify(self):
        return verify_phase(self.root, self.native, self.manifest, expected_frames=3, expected_sequences=2)

    def test_complete_readback_cannot_accept_formal_cost(self):
        receipt = self.verify()
        self.assertEqual(receipt['native_frames'], 3)
        self.assertFalse(receipt['formal_cost_accepted'])

    def test_duplicate_same_count_cannot_hide_missing_frame(self):
        self.rows[-1] = dict(self.rows[0])
        self.save_rows()
        with self.assertRaisesRegex(ValueError, 'repeated'):
            self.verify()

    def test_prediction_mutation_is_rejected(self):
        (self.native / 'trk_withid_0/0000/000000.txt').write_text('mutated prediction')
        with self.assertRaisesRegex(ValueError, 'scientific output mismatch'):
            self.verify()

    def test_bad_numeric_and_scope_values_are_rejected(self):
        for changes in [{'core_seconds': float('nan')}, {'core_seconds': 3}, {'status': 'failed'}]:
            with self.subTest(changes=changes):
                old = dict(self.rows[0])
                self.rows[0].update(changes)
                self.save_rows()
                with self.assertRaises(ValueError):
                    self.verify()
                self.rows[0] = old

    def test_unexpected_native_output_is_rejected(self):
        (self.native / 'extra.txt').write_text('extra')
        with self.assertRaisesRegex(ValueError, 'member set'):
            self.verify()

    def test_nonuniform_latency_quantiles_and_distinct_throughput_scopes(self):
        self.assertEqual(latency_summary([10, 1, 2])['p50_seconds'], 2)
        self.assertAlmostEqual(latency_summary([10, 1, 2])['p95_seconds'], 9.2)
        self.assertAlmostEqual(latency_summary([10, 1, 2])['p99_seconds'], 9.84)
        self.assertIsNone(latency_summary([0, 0])['frames_per_scope_second'])
        receipt = self.verify()
        self.assertEqual(receipt['frame_latency']['frames_per_scope_second'], 0.5)
        self.assertEqual(receipt['core_latency']['frames_per_scope_second'], 1)
        self.assertAlmostEqual(receipt['full_entry_frames_per_wall_second'], 3 / 7)
        self.assertTrue(receipt['core_throughput_not_end_to_end_FPS'])

    def resource_rows(self):
        return [dict(monotonic_seconds=i + 1, active_phases=[phase],
                     GPU_free_reserve=dict(GPU_UUID='GPU-68c03b8c-acc3-421c-0e7a-8da0bb9ad3fa',
                                           required_reserve_bytes=2 * 1024**3, free_bytes=3 * 1024**3),
                     CPU=dict(cgroup_accounted_current_bytes=20, cgroup_accounted_peak_bytes=30,
                              cgroup_hard_cap_bytes=6 * 1024**3, memory_events=dict(max=0, oom=0, oom_kill=0)),
                     process=dict(process_RSS_bytes=10, process_RSS_high_water_bytes=15),
                     GPU=dict(other_same_GPU_processes=[], owned_GPU_processes=[dict(PID=123, GPU_resident_bytes=25)]))
                for i, phase in enumerate(('warmup', 'measured'))]

    def check_resources(self, rows):
        path = self.root / 'resource-samples.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return verify_resource_samples(path)

    def test_resource_readback_requires_GPU_in_both_phases(self):
        receipt = self.check_resources(self.resource_rows())
        self.assertEqual(receipt['owned_GPU_phase_samples'], dict(warmup=1, measured=1))
        self.assertEqual(receipt['peaks']['cgroup_bytes'], 30)
        self.assertEqual(receipt['peaks']['process_RSS_bytes'], 15)
        self.assertFalse(receipt['formal_cost_accepted'])
        rows = self.resource_rows()
        rows[-1]['GPU']['owned_GPU_processes'] = []
        with self.assertRaisesRegex(ValueError, 'sample missing'):
            self.check_resources(rows)

    def test_resource_readback_rejects_contamination_and_bad_counters(self):
        for mutation in ('foreign', 'clock', 'OOM', 'GPU_cap', 'PID', 'CPU_bool', 'reserve'):
            with self.subTest(mutation=mutation):
                rows = self.resource_rows()
                if mutation == 'foreign':
                    rows[-1]['GPU']['other_same_GPU_processes'] = [dict(PID=456, GPU_resident_bytes=1)]
                elif mutation == 'clock':
                    rows[-1]['monotonic_seconds'] = 1
                elif mutation == 'OOM':
                    rows[-1]['CPU']['memory_events']['oom'] = 1
                elif mutation == 'GPU_cap':
                    rows[-1]['GPU']['owned_GPU_processes'][0]['GPU_resident_bytes'] = 6 * 1024**3 + 1
                elif mutation == 'PID':
                    rows[-1]['GPU']['owned_GPU_processes'][0]['PID'] = 456
                elif mutation == 'reserve':
                    rows[-1]['GPU_free_reserve']['free_bytes'] = 2 * 1024**3 - 1
                else:
                    rows[-1]['CPU']['cgroup_accounted_current_bytes'] = True
                with self.assertRaises(ValueError):
                    self.check_resources(rows)


if __name__ == '__main__':
    unittest.main()
