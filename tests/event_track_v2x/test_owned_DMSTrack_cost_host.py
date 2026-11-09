"""Admission rejection and owned-identity probes without remote execution."""
import hashlib
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools.event_track_v2x import native_cost_resource_guard

with mock.patch.dict(sys.modules, {'native_cost_resource_guard': native_cost_resource_guard}):
    from tools.event_track_v2x import run_owned_DMSTrack_cost_pair_host as host


class OwnedHostTests(unittest.TestCase):
    def fixture(self, root):
        bundle, pair, control = root / 'bundle', root / 'pair', root / 'control'
        for path in (bundle, pair, control):
            path.mkdir()
        names = ['run_owned_DMSTrack_cost_pair_host.py', 'run_DMSTrack_native_cost_entry.py',
                 'native_cost_timing.py', 'native_cost_resource_guard.py', 'original-qualified-GPU-entry.py']
        for name in names:
            (bundle / name).write_text('synthetic source ' + name)
        sources = {name: host.sha(bundle / name) for name in names}
        b = dict(container_name='old-original', remote_output=str(root / 'old'), CPU_threads=2,
                 CPU_memory_cap_bytes=6 * 1024**3, GPU_memory_cap_bytes=6 * 1024**3,
                 owned_total_output_cap_bytes=2 * 1024**3, host_disk_reserve_bytes=8 * 1024**3, GPU_free_reserve_bytes=2 * 1024**3,
                 host_RAM_reserve_bytes=6 * 1024**3, maximum_native_execution_seconds=21600,
                 GPU_UUID='GPU-68c03b8c-acc3-421c-0e7a-8da0bb9ad3fa',
                 entry_source_sha256=sources['original-qualified-GPU-entry.py'], image='synthetic',
                 runtime_site='/synthetic/runtime/site', original_CPU_site_packages='/synthetic/CPU/site',
                 author_source='/synthetic/author', original_CPU_root='/synthetic/CPU',
                 original_Python_prefix='/synthetic/Python', runtime_RAM_root='/synthetic/RAM',
                 original_CPU_python='/synthetic/Python/bin/python')
        spec = dict(original_native_binding=b, pair_output=str(pair), control_output=str(control),
                    new_owned_container_name='rbf-native-cost-pair-synthetic',
                    original_deadline_controller_restarted=False, source_bundle=str(bundle),
                    source_files=sources, sampling_interval_seconds=1, repetition_index=0)
        path = root / 'spec.json'
        path.write_text(json.dumps(spec))
        return path, bundle, pair, control, spec

    def test_disk_rejection_creates_failure_receipt_without_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            bundle, pair, control = root / 'bundle', root / 'pair', root / 'control'
            for path in (bundle, pair, control):
                path.mkdir()
            names = ['run_owned_DMSTrack_cost_pair_host.py', 'run_DMSTrack_native_cost_entry.py',
                     'native_cost_timing.py', 'native_cost_resource_guard.py', 'original-qualified-GPU-entry.py']
            for name in names:
                (bundle / name).write_text('synthetic source ' + name)
            sources = {name: host.sha(bundle / name) for name in names}
            b = dict(container_name='old-original', remote_output=str(root / 'old'), CPU_threads=2,
                     CPU_memory_cap_bytes=6 * 1024**3, GPU_memory_cap_bytes=6 * 1024**3,
                     owned_total_output_cap_bytes=2 * 1024**3, host_disk_reserve_bytes=8 * 1024**3, GPU_free_reserve_bytes=2 * 1024**3,
                     maximum_native_execution_seconds=21600,
                     GPU_UUID='GPU-68c03b8c-acc3-421c-0e7a-8da0bb9ad3fa',
                     entry_source_sha256=sources['original-qualified-GPU-entry.py'])
            spec = dict(original_native_binding=b, pair_output=str(pair), control_output=str(control),
                        new_owned_container_name='rbf-native-cost-pair-synthetic',
                        original_deadline_controller_restarted=False, source_bundle=str(bundle),
                        source_files=sources, sampling_interval_seconds=1, repetition_index=0)
            path = root / 'spec.json'
            path.write_text(json.dumps(spec))
            with mock.patch.object(host, '__file__', str(bundle / names[0])), \
                 mock.patch.object(host, 'run', return_value='10.100.35.112'), \
                 mock.patch.object(host.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=9451745280)), \
                 mock.patch.object(host.subprocess, 'Popen') as launch, \
                 mock.patch.object(host, 'check_inputs') as check_inputs:
                with self.assertRaisesRegex(ValueError, 'remaining full-phase retention'):
                    host.main(path)
            launch.assert_not_called()
            check_inputs.assert_not_called()
            receipt = json.loads((control / 'owned-native-cost-pair-failure-preserved.json').read_bytes())
            self.assertTrue(receipt['original_jobs_untouched'])
            self.assertFalse(receipt['automatic_retry'])
            self.assertEqual(list(pair.iterdir()), [])

    def test_only_exact_Docker_creation_ID_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / 'owned.cid'
            path.write_text('a' * 64 + '\n')
            self.assertEqual(host.owned_container_ID(path), 'a' * 64)
            for value in ['old-original-container', 'b' * 63, 'g' * 64]:
                path.write_text(value)
                with self.assertRaises(ValueError):
                    host.owned_container_ID(path)

    def test_foreign_GPU_appearance_stops_only_own_Docker_CID(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, bundle, pair, control, spec = self.fixture(pathlib.Path(temporary))
            CID = 'a' * 64

            class FakeProcess:
                returncode = None

                def poll(self):
                    return self.returncode

                def wait(self, timeout):
                    return self.returncode

            process = FakeProcess()

            def launch(command, **kwargs):
                cidfile = pathlib.Path(command[command.index('--cidfile') + 1])
                cidfile.write_text(CID)
                return process

            def commands(arguments):
                if arguments[0] == 'hostname':
                    return '10.100.35.112'
                if arguments[0] == 'nvidia-smi' and '--query-gpu=uuid,memory.free' in arguments:
                    return spec['original_native_binding']['GPU_UUID'] + ', 4096'
                if arguments[:3] == ['docker', 'image', 'inspect']:
                    return 'synthetic'
                return ''

            def stop(CID_to_stop):
                self.assertEqual(CID_to_stop, CID)
                process.returncode = -15

            inspection = types.SimpleNamespace(returncode=0, stdout=json.dumps([dict(
                Id=CID, Config={'Labels': {'rbf.cost.execution': spec['new_owned_container_name']}},
                State={'Pid': 123})]))
            sample = (dict(cgroup_accounted_peak_bytes=0, memory_events={'max': 0, 'oom': 0, 'oom_kill': 0}),
                      dict(process_RSS_high_water_bytes=0),
                      dict(owned_GPU_processes=[], other_same_GPU_processes=[{'PID': 999, 'GPU_resident_bytes': 1}]))
            with mock.patch.object(host, '__file__', str(bundle / 'run_owned_DMSTrack_cost_pair_host.py')), \
                 mock.patch.object(host, 'run', side_effect=commands), \
                 mock.patch.object(host.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=30 * 1024**3)), \
                 mock.patch.object(host, 'available_host_RAM', return_value=30 * 1024**3), \
                 mock.patch.object(host, 'check_inputs'), \
                 mock.patch.object(host.subprocess, 'Popen', side_effect=launch), \
                 mock.patch.object(host.subprocess, 'run', return_value=inspection), \
                 mock.patch.object(host, 'observe_native_process', return_value=sample), \
                 mock.patch.object(host, 'owned_stop', side_effect=stop) as stop_call:
                with self.assertRaisesRegex(ValueError, 'other_process_appeared'):
                    host.main(path)
            stop_call.assert_called_once_with(CID)
            self.assertTrue((control / 'owned-native-cost-pair-failure-preserved.json').is_file())
            self.assertFalse((control / 'owned-native-cost-pair-terminal.json').exists())

    def test_extra_source_payload_cannot_hide_in_runtime_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, bundle, pair, control, spec = self.fixture(pathlib.Path(temporary))
            (bundle / 'unapproved_extra.py').write_text('synthetic extra payload')
            with mock.patch.object(host, 'run', return_value='10.100.35.112'), \
                 mock.patch.object(host.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'outside public-only allowlist'):
                    host.main(path)
            launch.assert_not_called()
            self.assertEqual(list(pair.iterdir()), [])
            self.assertEqual(list(control.iterdir()), [])

    def test_disk_probe_never_launches_even_if_disk_later_becomes_sufficient(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, bundle, pair, control, spec = self.fixture(pathlib.Path(temporary))
            spec['disk_admission_probe_only'] = True
            path.write_text(json.dumps(spec))
            with mock.patch.object(host, '__file__', str(bundle / 'run_owned_DMSTrack_cost_pair_host.py')), \
                 mock.patch.object(host, 'run', return_value='10.100.35.112') as commands, \
                 mock.patch.object(host.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=30 * 1024**3)), \
                 mock.patch.object(host.subprocess, 'Popen') as launch, \
                 mock.patch.object(host, 'check_inputs') as check_inputs:
                host.main(path)
            commands.assert_called_once_with(['hostname', '-I'])
            launch.assert_not_called()
            check_inputs.assert_not_called()
            receipt = json.loads((control / 'native-controller-disk-admission-only-returned.json').read_bytes())
            self.assertTrue(receipt['disk_admitted'])
            self.assertFalse(receipt['full_resource_admission'])
            self.assertFalse(receipt['native_process_created'])
            self.assertEqual(list(pair.iterdir()), [])

    def test_GPU_free_reserve_rejects_before_input_readback_or_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, bundle, pair, control, spec = self.fixture(pathlib.Path(temporary))

            def commands(arguments):
                if arguments[0] == 'hostname':
                    return '10.100.35.112'
                if arguments[:3] == ['docker', 'image', 'inspect']:
                    return 'synthetic'
                if '--query-gpu=uuid,memory.free' in arguments:
                    return spec['original_native_binding']['GPU_UUID'] + ', 2047'
                return ''

            with mock.patch.object(host, '__file__', str(bundle / 'run_owned_DMSTrack_cost_pair_host.py')), \
                 mock.patch.object(host, 'run', side_effect=commands), \
                 mock.patch.object(host.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=30 * 1024**3)), \
                 mock.patch.object(host, 'available_host_RAM', return_value=30 * 1024**3), \
                 mock.patch.object(host, 'check_inputs') as inputs, \
                 mock.patch.object(host.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'GPU free reserve not admitted'):
                    host.main(path)
            inputs.assert_not_called()
            launch.assert_not_called()

    def test_last_exit_output_growth_cannot_escape_cap_between_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, bundle, pair, control, spec = self.fixture(pathlib.Path(temporary))
            CID = 'a' * 64

            class FakeProcess:
                returncode = None
                polls = 0

                def poll(self):
                    self.polls += 1
                    if self.polls == 3:
                        with (pair / 'measured/last-output').open('wb') as handle:
                            handle.truncate(2 * 1024**3 + 1)
                        self.returncode = 0
                    return self.returncode

            process = FakeProcess()

            def launch(command, **kwargs):
                (control / 'owned-container.cid').write_text(CID)
                for phase in ('warmup', 'measured'):
                    (pair / phase).mkdir()
                (pair / 'warmup/native-cost-phase-start.json').write_text(json.dumps({'started_monotonic_seconds': host.time.monotonic()}))
                return process

            def next_phase(seconds):
                (pair / 'warmup/native-cost-phase-observation.json').write_text('{}')
                (pair / 'measured/native-cost-phase-start.json').write_text(json.dumps({'started_monotonic_seconds': host.time.monotonic()}))

            inspection = types.SimpleNamespace(returncode=0, stdout=json.dumps([dict(
                Id=CID, Config={'Labels': {'rbf.cost.execution': spec['new_owned_container_name']}}, State={'Pid': 123})]))
            sample = (dict(cgroup_accounted_peak_bytes=0, memory_events={'max': 0, 'oom': 0, 'oom_kill': 0}),
                      dict(process_RSS_high_water_bytes=0),
                      dict(owned_GPU_processes=[{'PID': 123, 'GPU_resident_bytes': 1024}], other_same_GPU_processes=[]))
            with mock.patch.object(host, '__file__', str(bundle / 'run_owned_DMSTrack_cost_pair_host.py')), \
                 mock.patch.object(host, 'run', side_effect=lambda argv: '10.100.35.112' if argv[0] == 'hostname' else ('synthetic' if argv[:3] == ['docker', 'image', 'inspect'] else '')), \
                 mock.patch.object(host.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=30 * 1024**3)), \
                 mock.patch.object(host, 'available_host_RAM', return_value=30 * 1024**3), \
                 mock.patch.object(host, 'check_inputs'), \
                 mock.patch.object(host, 'check_GPU_reserve', return_value=dict(
                     GPU_UUID=spec['original_native_binding']['GPU_UUID'],
                     free_bytes=3 * 1024**3, required_reserve_bytes=2 * 1024**3)), \
                 mock.patch.object(host, 'parse_GPU_processes', return_value={'other_same_GPU_processes': []}), \
                 mock.patch.object(host.subprocess, 'Popen', side_effect=launch), \
                 mock.patch.object(host.subprocess, 'run', return_value=inspection), \
                 mock.patch.object(host, 'observe_native_process', return_value=sample), \
                 mock.patch.object(host.time, 'sleep', side_effect=next_phase), \
                 mock.patch.object(host, 'owned_stop'):
                with self.assertRaisesRegex(ValueError, 'per-phase output cap exceeded'):
                    host.main(path)
            self.assertTrue((control / 'owned-native-cost-pair-failure-preserved.json').exists())
            self.assertFalse((control / 'owned-native-cost-pair-terminal.json').exists())


if __name__ == '__main__':
    unittest.main()
