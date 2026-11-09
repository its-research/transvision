"""Independent full-frame and native prediction readback for one cost phase.

Actual scientific readback/statistics run on L40 CPU under its admitted task.
This component cannot accept total resource cost without the host monitor,
input/runtime bindings and the complete three-repetition campaign.
"""
import hashlib
import json
import math
import pathlib
import re


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def latency_summary(values):
    ordered = sorted(values)
    if not ordered or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in ordered):
        raise ValueError('invalid complete latency sample')
    total = math.fsum(ordered)

    def quantile(probability):
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        fraction = position - lower
        return ordered[lower] + fraction * (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower])

    return dict(count=len(ordered), total_seconds=total,
                p50_seconds=quantile(0.50), p95_seconds=quantile(0.95),
                p99_seconds=quantile(0.99), maximum_seconds=ordered[-1],
                frames_per_scope_second=len(ordered) / total if total > 0 else None,
                zero_duration_throughput_not_invented=total == 0,
                quantile_definition='linear interpolation at (n-1)*p; complete frame cohort')


def verify_resource_samples(path, CPU_cap_bytes=6 * 1024**3, GPU_cap_bytes=6 * 1024**3,
                            byte_cap=256 * 1024**2):
    path = pathlib.Path(path)
    if path.is_symlink() or path.stat().st_size > byte_cap:
        raise ValueError('resource log cap or symlink violation')
    previous = None
    counts = dict(warmup=0, measured=0)
    peaks = dict(cgroup_bytes=0, process_RSS_bytes=0, sampled_GPU_resident_bytes=0)
    samples = 0
    owned_PID = None

    def integer(value):
        if type(value) is not int or value < 0:
            raise ValueError('invalid resource byte counter')
        return value

    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            clock = row['monotonic_seconds']
            if type(clock) not in (int, float) or not math.isfinite(clock) or (previous is not None and clock <= previous):
                raise ValueError('resource samples not strictly monotonic')
            previous = clock
            phases = row['active_phases']
            if type(phases) is not list or len(phases) > 1 or any(p not in counts for p in phases):
                raise ValueError('ambiguous or unknown native resource phase')
            CPU, process, GPU = row['CPU'], row['process'], row['GPU']
            reserve = row['GPU_free_reserve']
            if reserve['GPU_UUID'] != 'GPU-68c03b8c-acc3-421c-0e7a-8da0bb9ad3fa' or integer(reserve['required_reserve_bytes']) != 2 * 1024**3 or integer(reserve['free_bytes']) < reserve['required_reserve_bytes']:
                raise ValueError('original physical GPU reserve not evidenced')
            peak = integer(CPU['cgroup_accounted_peak_bytes'])
            current = integer(CPU['cgroup_accounted_current_bytes'])
            rss_peak = integer(process['process_RSS_high_water_bytes'])
            rss_current = integer(process['process_RSS_bytes'])
            if integer(CPU['cgroup_hard_cap_bytes']) != CPU_cap_bytes or current > peak or peak > CPU_cap_bytes or rss_current > rss_peak or rss_peak > CPU_cap_bytes:
                raise ValueError('native CPU counters violate frozen resource contract')
            events = CPU['memory_events']
            if any(integer(events[k]) != 0 for k in ('max', 'oom', 'oom_kill')):
                raise ValueError('native cgroup reached hard limit or OOM')
            if GPU['other_same_GPU_processes'] or len(GPU['owned_GPU_processes']) > 1:
                raise ValueError('nonexclusive native GPU samples')
            for owned in GPU['owned_GPU_processes']:
                PID = integer(owned['PID'])
                if PID == 0 or (owned_PID is not None and PID != owned_PID):
                    raise ValueError('owned native GPU PID changed')
                owned_PID = PID
                resident = integer(owned['GPU_resident_bytes'])
                if resident > GPU_cap_bytes:
                    raise ValueError('native GPU resident cap exceeded')
                peaks['sampled_GPU_resident_bytes'] = max(peaks['sampled_GPU_resident_bytes'], resident)
                for phase in phases:
                    counts[phase] += 1
            peaks['cgroup_bytes'] = max(peaks['cgroup_bytes'], peak)
            peaks['process_RSS_bytes'] = max(peaks['process_RSS_bytes'], rss_peak)
            samples += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError('owned GPU sample missing from a complete native phase')
    return dict(resource_log_sha256=sha(path), sampled_observations=samples,
                owned_GPU_phase_samples=counts, owned_GPU_PID=owned_PID, peaks=peaks,
                GPU_peak_is_sampled_not_exact_high_water=True,
                CPU_peaks_are_lifetime_including_warmup=True,
                CPU_cgroup_and_RSS_are_separate_not_summed=True,
                source_input_runtime_deadline_and_full_output_checks_pending=True,
                formal_cost_accepted=False)


def native_reference(manifest):
    references = {}
    keys = set()
    for name, record in manifest['members'].items():
        if name.startswith('native/'):
            relative = name[len('native/'):]
            path = pathlib.PurePosixPath(relative)
            if path.is_absolute() or '..' in path.parts or str(path) != relative:
                raise ValueError('unsafe reference native path')
            references[relative] = record
            match = re.fullmatch(r'trk_withid_0/(\d{4})/(\d{6})\.txt', relative)
            if match:
                keys.add((match[1], int(match[2])))
    return references, keys


def verify_phase(phase_directory, native_directory, reference_manifest,
                 expected_frames=1993, expected_sequences=9,
                 native_output_cap_bytes=2 * 1024**3, timing_cap_bytes=2 * 1024**2):
    phase = pathlib.Path(phase_directory)
    native = pathlib.Path(native_directory)
    references, expected_keys = native_reference(reference_manifest)
    if len(expected_keys) != expected_frames or len({s for s, _ in expected_keys}) != expected_sequences:
        raise ValueError('reference is not the complete native frame cohort')
    if phase.is_symlink() or native.is_symlink():
        raise ValueError('symlink phase/native root forbidden')
    files = {}
    for path in native.rglob('*'):
        if path.is_symlink():
            raise ValueError('symlink native member forbidden')
        if path.is_file():
            files[path.relative_to(native).as_posix()] = path
    if set(files) != set(references):
        raise ValueError('complete native output member set differs')
    if sum(path.stat().st_size for path in files.values()) > native_output_cap_bytes:
        raise ValueError('original native output cap exceeded')
    readback = {}
    for name, path in sorted(files.items()):
        actual = dict(bytes=path.stat().st_size, sha256=sha(path))
        readback[name] = actual
        # Author PDF bytes may include creation timestamps. They are fully
        # read and retained, but do not establish scientific prediction parity.
        if not name.endswith('.pdf'):
            expected = references[name]
            if actual != {k: expected[k] for k in ('bytes', 'sha256')}:
                raise ValueError('native scientific output mismatch: ' + name)
    timings = phase / 'native-frame-times.jsonl'
    if timings.is_symlink() or timings.stat().st_size > timing_cap_bytes:
        raise ValueError('timing file cap or symlink violation')
    observed = set()
    sequence_last_frame = {}
    frame_total = 0.0
    core_total = 0.0
    frame_values = []
    core_values = []
    with timings.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['status'] != 'returned' or not isinstance(row['sequence'], str):
                raise ValueError('partial or failed timing row')
            if type(row['frame']) is not int:
                raise ValueError('native frame key is not an integer')
            key = (row['sequence'], row['frame'])
            if key not in expected_keys or key in observed:
                raise ValueError('unexpected or repeated native frame key')
            if key[1] <= sequence_last_frame.get(key[0], -1):
                raise ValueError('native sequence timing order reversed')
            sequence_last_frame[key[0]] = key[1]
            for name in ('core_seconds', 'frame_seconds'):
                value = row[name]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('nonfinite or invalid latency')
            if row['core_seconds'] > row['frame_seconds']:
                raise ValueError('core latency exceeds enclosing frame scope')
            observed.add(key)
            frame_total += row['frame_seconds']
            core_total += row['core_seconds']
            frame_values.append(row['frame_seconds'])
            core_values.append(row['core_seconds'])
    if observed != expected_keys:
        raise ValueError('missing native timing frame keys')
    observation_path = phase / 'native-cost-phase-observation.json'
    if observation_path.is_symlink():
        raise ValueError('symlink phase receipt forbidden')
    observation = json.loads(observation_path.read_bytes())
    if observation['completed_frame_rows'] != expected_frames or observation['failed_frame_rows'] != 0:
        raise ValueError('phase count receipt contradicts full readback')
    elapsed = observation['full_entry_wall_seconds']
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < frame_total:
        raise ValueError('entry wall clock contradicts frame timings')
    if observation['phase'] not in ('warmup', 'measured'):
        raise ValueError('unknown cost phase')
    return dict(kind='DMSTrack_native_cost_phase_prediction_and_frame_readback_component',
                phase=observation['phase'], native_frames=len(observed),
                native_sequences=len(sequence_last_frame), native_files=len(readback),
                all_native_output_members_fully_read=readback,
                non_PDF_scientific_outputs_exact_original_bytes=True,
                PDF_byte_identity_not_required_or_claimed=True,
                frame_keys_exact_complete_no_duplicates=True,
                timing_file_sha256=sha(timings), phase_receipt_sha256=sha(observation_path),
                full_entry_wall_seconds=elapsed,
                frame_total_seconds=frame_total, core_total_seconds=core_total,
                frame_latency=latency_summary(frame_values), core_latency=latency_summary(core_values),
                full_entry_frames_per_wall_second=len(observed) / elapsed,
                latency_scope='author native frame: feature/transformation loads, GT diagnostics, loss and output writes; core is the original tracker call; setup/evaluation outside frame',
                core_throughput_not_end_to_end_FPS=True,
                total_resident_memory_and_input_runtime_admission_pending=True,
                complete_three_repetition_campaign_pending=True,
                formal_cost_accepted=False, same_resource_accepted=False)
