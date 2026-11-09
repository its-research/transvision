"""One full native warmup and one measured repetition in the same process.

Requires the original, independently qualified CUDA entry and its full binding.
This is a preparation entry, not a resource-admission/dispatch controller.
It never reports formal cost acceptance; the external owned-process guard and
independent output/coverage readback remain mandatory. Run three such pairs
under the same frozen host contract for the minimum resource repetitions.
"""
import argparse
import datetime
import gc
import hashlib
import json
import pathlib
import resource
import runpy
import sys
import time

from native_cost_timing import NativeFrameTiming, execute_author_source
from native_cost_resource_guard import read_cgroup_memory, read_process_rss, resource_violations


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main(binding_path):
    if sys.platform != 'linux':
        raise ValueError('Linux native author runtime required')
    specification = json.loads(pathlib.Path(binding_path).read_bytes())
    base = specification['original_native_binding']
    if specification['container_cgroup_mount'] != '/sys/fs/cgroup':
        raise ValueError('original private container cgroup mount required')
    root = pathlib.Path(specification['pair_output'])
    guard = specification['owned_process_guard']
    if guard['GPU_UUID'] != base['GPU_UUID'] or guard['CPU_threads'] != 2:
        raise ValueError('owned process guard identity mismatch')
    if guard['GPU_process_resident_cap_bytes'] != base['GPU_memory_cap_bytes']:
        raise ValueError('owned process GPU cap changed')
    if guard['CPU_process_resident_cap_bytes'] != base['CPU_memory_cap_bytes']:
        raise ValueError('owned process CPU cap changed')
    # These bindings do not replace the live external monitor. The controller
    # must verify the monitor PID and terminate only its owned container if
    # total residency, outputs, deadline or exclusive-GPU admission fail.
    if guard['live_monitor_PID'] <= 0 or guard['original_deadline_controller_restarted']:
        raise ValueError('new owned live monitor required')
    # The host guard must create and own this exclusive pair root before entry.
    if not root.is_dir() or any(root.iterdir()):
        raise ValueError('existing empty owned pair root required')
    if specification['repetition_index'] not in (0, 1, 2):
        raise ValueError('original three repetition index required')
    original_entry = pathlib.Path(specification['original_entry_path'])
    if digest(original_entry) != base['entry_source_sha256']:
        raise ValueError('original qualified runtime entry SHA mismatch')
    source = pathlib.Path(base['author_source']) / 'DMSTrack/main_dkf.py'
    expected = base['original_source_files']['DMSTrack/main_dkf.py']['sha256']
    if digest(source) != expected:
        raise ValueError('original qualified author main SHA mismatch')
    if base['CPU_threads'] != 2 or base['original_native_num_epochs'] != 0:
        raise ValueError('native inference/threads contract changed')
    original_run_path = runpy.run_path
    original_argv = list(sys.argv)
    phase_receipts = []
    try:
        for phase in ('warmup', 'measured'):
            directory = root / phase
            directory.mkdir(exist_ok=False)
            bound = dict(base)
            bound['remote_output'] = str(directory)
            argv = list(base['native_argv'])
            option = argv.index('--save_dir_prefix')
            argv[option + 1] = str(directory / 'native-output/native-results')
            bound['native_argv'] = argv
            phase_binding = directory / 'original-native-phase-binding.json'
            with phase_binding.open('x') as handle:
                json.dump(bound, handle, indent=2, allow_nan=False)
            sys.argv = [str(original_entry), str(phase_binding)]
            with (directory / 'native-cost-phase-start.json').open('x') as handle:
                json.dump(dict(phase=phase, started_monotonic_seconds=time.monotonic()), handle)
            # Torch is imported from the exact host-admitted original runtime.
            import torch
            memory_before = read_cgroup_memory('/sys/fs/cgroup', base['CPU_memory_cap_bytes'])
            process_before = read_process_rss('/proc/self/status')
            violations = resource_violations(memory_before, process_before,
                dict(owned_GPU_processes=[], other_same_GPU_processes=[]),
                base['CPU_memory_cap_bytes'], base['GPU_memory_cap_bytes'])
            if violations:
                raise ValueError('owned CPU resource violation before phase: ' + repr(violations))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            with (directory / 'native-frame-times.jsonl').open('x') as timing_output:
                timer = NativeFrameTiming(timing_output, torch.cuda.synchronize)

                def timed_run_path(path_name, init_globals=None, run_name=None):
                    if pathlib.Path(path_name).resolve() == source.resolve():
                        if init_globals is not None or run_name != '__main__':
                            raise ValueError('unexpected original native run_path invocation')
                        return execute_author_source(source, expected, timer)
                    return original_run_path(path_name, init_globals=init_globals, run_name=run_name)

                runpy.run_path = timed_run_path
                torch.cuda.synchronize()
                started = time.perf_counter()
                try:
                    original_run_path(str(original_entry), run_name='__main__')
                finally:
                    runpy.run_path = original_run_path
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
            if timer.completed != 1993 or timer.failed:
                raise ValueError('incomplete native phase; no cost acceptance')
            memory_after = read_cgroup_memory('/sys/fs/cgroup', base['CPU_memory_cap_bytes'])
            process_after = read_process_rss('/proc/self/status')
            violations = resource_violations(memory_after, process_after,
                dict(owned_GPU_processes=[], other_same_GPU_processes=[]),
                base['CPU_memory_cap_bytes'], base['GPU_memory_cap_bytes'])
            if violations:
                raise ValueError('owned CPU resource violation after phase: ' + repr(violations))
            phase_receipts.append(dict(
                phase=phase, full_entry_wall_seconds=elapsed,
                completed_frame_rows=timer.completed, failed_frame_rows=timer.failed,
                CPU_process_lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                CPU_RSS_unit='Linux ru_maxrss KiB; lifetime includes prior warmup',
                CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                cgroup_memory_before_native_phase=memory_before,
                cgroup_memory_after_native_phase=memory_after,
                native_process_RSS_before_phase=process_before,
                native_process_RSS_after_phase=process_after,
                GPU_total_process_resident_memory_not_proven_by_allocator=True,
                checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            with (directory / 'native-cost-phase-observation.json').open('x') as handle:
                json.dump(phase_receipts[-1], handle, indent=2, allow_nan=False)
            gc.collect()
            # Add no cache clearing here. The unchanged original CUDA entry
            # already empties the allocator cache after its smoke in each
            # phase; report that policy, rather than claiming warm allocation.
    finally:
        runpy.run_path = original_run_path
        sys.argv = original_argv
    with (root / 'native-cost-pair-returned.json').open('x') as handle:
        json.dump(dict(repetition_index=specification['repetition_index'], phases=phase_receipts,
                       full_warmup_and_measurement_same_process=True,
                       independent_output_parity_and_unique_frame_coverage_pending=True,
                       total_owned_process_memory_guard_required=True,
                       formal_cost_accepted=False, same_resource_accepted=False),
                  handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('binding')
    main(parser.parse_args().binding)
