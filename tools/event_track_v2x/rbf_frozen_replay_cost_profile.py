"""External cProfile observer for a separately named, source-bound replay.

This module never replaces a scorer, modifies replay globals or changes an
event. Profiling is an explicitly separate debugging run: its overhead and
function-wall times do not qualify paper latency or GPU kernel occupancy.
"""
from __future__ import annotations

import cProfile
import datetime
import hashlib
import json
import marshal
import math
from pathlib import Path
import pstats
import sys
import time
import types


ALLOWED_MODULES = frozenset((
    'exclusive_paper_runtime.py', 'paper_runtime.py', 'persistent_cache_stream.py',
    'forest_potentials.py', 'recoverable_identity.py', 'learned_identity.py',
    'forest_row_context.py', 'forest_tracking.py', 'prediction_features.py',
    'tracking_v2.py', 'persistent_forest.py', 'persistent_component_tracking.py',
    'persistent_component_store.py', 'covered_completion_tracking.py',
    'exclusive_completion_tracking.py', 'residual_frontier.py', 'paper_decision.py',
    'identity_forest.py', 'hypothesis_bank.py', 'paper_resources.py',
))
ALLOWED_NATIVE_NAMES = frozenset((
    "<method 'cpu' of 'torch._C.TensorBase' objects>",
    "<method 'to' of 'torch._C.TensorBase' objects>",
    "<method 'tolist' of 'torch._C.TensorBase' objects>",
    "<method 'execute' of 'sqlite3.Connection' objects>",
    "<method 'executemany' of 'sqlite3.Connection' objects>",
    "<method 'fetchall' of 'sqlite3.Cursor' objects>",
    "<method 'fetchone' of 'sqlite3.Cursor' objects>",
))
RUNTIME_KEYS = frozenset((
    'events_from_independently_admitted_real_schedule', 'causal_event_bytes_unchanged',
    'read_only_model_loaded_from_frozen_checkpoint', 'actual_device', 'GPU_uuid',
    'TF32_matmul', 'TF32_cudnn', 'torch_version', 'numpy_version', 'python_version',
    'CUDA_version', 'platform', 'world_size', 'rank',
))
MODEL_KEYS = frozenset((
    'candidate_protocol', 'dataset', 'fit_split', 'checkpoint_sha256',
    'model_sha256', 'seed', 'frozen_cache_identity',
))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _new_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def _sources(source_root, source_manifest):
    root = Path(source_root).resolve(strict=True)
    if not isinstance(source_manifest, dict) or not source_manifest:
        raise ValueError('explicit source file identities required')
    found = {}
    for relative, expected in source_manifest.items():
        name = Path(relative)
        if name.is_absolute() or '..' in name.parts or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('invalid source identity')
        path = root / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root) or _sha(path) != expected:
            raise ValueError('frozen source differs: '+relative)
        found[str(path.resolve())] = dict(relative=relative, sha256=expected)
    return root, found


def _selected_stats(profile, source_files):
    """Persist allowlisted code names and aggregate numbers, never call args."""
    selected = []
    if not profile.getstats():
        return selected
    for (filename, line, name), (primitive, calls, own, cumulative, callers) in pstats.Stats(profile).stats.items():
        identity = source_files.get(filename)
        if identity is not None and Path(filename).name in ALLOWED_MODULES:
            origin = dict(source=identity['relative'], source_sha256=identity['sha256'], line=line)
        elif filename == '~' and name in ALLOWED_NATIVE_NAMES:
            origin = dict(source='allowed-native-function', source_sha256=None, line=line)
        else:
            continue
        if not all(type(v) is int and v >= 0 for v in (primitive, calls)) or not all(
                type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (own, cumulative)):
            raise ValueError('invalid profile statistics')
        selected.append(dict(**origin, function=name, primitive_calls=primitive, total_calls=calls,
                             function_own_wall_seconds=own, function_inclusive_wall_seconds=cumulative))
    return sorted(selected, key=lambda row: (-row['function_inclusive_wall_seconds'], row['source'], row['line'], row['function']))


def profile_replay(replay, *, source_root, source_manifest, cache, events, output,
                   protocol, configuration, scorer, model_binding,
                   runtime_identity, schedule_path, schedule_sha256,
                   selected_sequences, per_sequence_prefix_count, synchronize=None):
    """Call the original replay once; preserve every original event and value.

    `events` must be selected from an independently admitted real schedule by
    the caller. This observer requires that provenance and refuses to claim
    that a measured prefix is a full sequence or final evaluation.
    """
    root, sources = _sources(source_root, source_manifest)
    if (not callable(replay) or replay.__name__ != 'replay'
            or replay.__code__.co_filename not in sources
            or Path(replay.__code__.co_filename).name not in ('exclusive_paper_runtime.py', 'paper_runtime.py')):
        raise ValueError('original source-bound replay callable required')
    compiled = compile(Path(replay.__code__.co_filename).read_bytes(), replay.__code__.co_filename,
                       'exec', dont_inherit=True, optimize=sys.flags.optimize)
    candidates = [code for code in compiled.co_consts if isinstance(code, types.CodeType) and code.co_name == 'replay']
    if len(candidates) != 1 or marshal.dumps(replay.__code__) != marshal.dumps(candidates[0]):
        raise ValueError('replay function bytecode differs from the frozen source')
    if (not isinstance(events, (list, tuple)) or not events
            or not isinstance(runtime_identity, dict)
            or runtime_identity.get('events_from_independently_admitted_real_schedule') is not True
            or runtime_identity.get('causal_event_bytes_unchanged') is not True
            or runtime_identity.get('TF32_matmul') is not False
            or runtime_identity.get('TF32_cudnn') is not False
            or not isinstance(runtime_identity.get('actual_device'), str)):
        raise ValueError('real schedule binding, actual device and disabled TF32 required')
    if runtime_identity.get('actual_device', '').startswith('cuda') and not callable(synchronize):
        raise ValueError('CUDA wall time requires an explicit device synchronization callback')
    if runtime_identity.get('read_only_model_loaded_from_frozen_checkpoint') is not True:
        raise ValueError('frozen checkpoint load binding required')
    if set(runtime_identity)-RUNTIME_KEYS or not isinstance(model_binding, dict) or set(model_binding)-MODEL_KEYS:
        raise ValueError('unallowlisted runtime or checkpoint fields')
    if (model_binding.get('candidate_protocol') != 'rbf-all-class-top64-v1'
            or model_binding.get('dataset') != 'spd' or model_binding.get('fit_split') != 'train'):
        raise ValueError('this debugging observer is restricted to all-class SPD train')
    schedule_path = Path(schedule_path)
    if schedule_path.is_symlink() or _sha(schedule_path) != schedule_sha256:
        raise ValueError('original admitted real schedule bytes differ')
    schedule = json.loads(schedule_path.read_bytes())
    if len(schedule['events']) != 7445 or len(schedule['origin_us_by_sequence']) != 46:
        raise ValueError('full original paired train schedule required; no inferred empty events')
    if (not isinstance(selected_sequences, (list, tuple)) or not selected_sequences
            or list(selected_sequences) != sorted(set(selected_sequences))
            or not set(selected_sequences).issubset(schedule['origin_us_by_sequence'])
            or (per_sequence_prefix_count is not None and
                (type(per_sequence_prefix_count) is not int or per_sequence_prefix_count < 1))):
        raise ValueError('explicit sequence IDs and fixed causal prefix count required')
    counts = dict.fromkeys(selected_sequences, 0)
    expected_events = []
    for event in schedule['events']:
        sequence = event['sequence_id']
        if sequence in counts:
            counts[sequence] += 1
            if per_sequence_prefix_count is None or counts[sequence] <= per_sequence_prefix_count:
                expected_events.append(event)
    if events != expected_events:
        raise ValueError('measured events must be unchanged complete per-sequence causal prefixes')
    # Canonical event bytes are an immutable request identity. The actual
    # objects passed to replay are the same objects; no filtering or padding.
    event_bytes = json.dumps(events, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    profile = cProfile.Profile()
    request = dict(kind='rbf_original_replay_external_function_profile_request_v1',
        source_root=str(root), source_files=source_manifest,
        replay_bytecode_sha256=hashlib.sha256(marshal.dumps(replay.__code__)).hexdigest(),
        event_bytes_sha256=hashlib.sha256(event_bytes).hexdigest(), event_count=len(events),
        admitted_full_schedule_sha256=schedule_sha256,
        selected_sequences=selected_sequences, per_sequence_prefix_count=per_sequence_prefix_count,
        sequence_ids=sorted({event['sequence_id'] for event in events}),
        checkpoint_binding=model_binding, runtime_identity=runtime_identity,
        observer_sha256=_sha(__file__), observer_entrypoint='profile_replay',
        online_model_inputs_or_source_changed=False, measurement_instrumented=True,
        profiler_inclusive_times_must_not_be_added_as_disjoint_cost=True,
        paper_latency_admission=False, full_Stage2_admission=False)
    _new_json(directory/'request.json', request)
    failure = None
    result = None
    wait_before = wait_after = 0.
    try:
        if synchronize is not None:
            before = time.perf_counter()
            synchronize()
            wait_before = time.perf_counter()-before
        started, cpu_started = time.perf_counter(), time.process_time()
        profile.enable()
        try:
            result = replay(cache, events, directory/'original-replay', protocol=protocol,
                            configuration=configuration, scorer=scorer, model_binding=model_binding, fixture=False)
        finally:
            profile.disable()
        if synchronize is not None:
            before = time.perf_counter()
            synchronize()
            wait_after = time.perf_counter()-before
        elapsed, cpu_elapsed = time.perf_counter()-started, time.process_time()-cpu_started
        _sources(root, source_manifest)
        if json.dumps(events, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() != event_bytes:
            raise ValueError('event request changed during measurement')
        if _sha(schedule_path) != schedule_sha256:
            raise ValueError('original schedule bytes changed during measurement')
    except BaseException as error:
        failure = dict(error_type=type(error).__name__, error=str(error))
        elapsed = time.perf_counter()-started if 'started' in locals() else None
        cpu_elapsed = time.process_time()-cpu_started if 'cpu_started' in locals() else None
        raise
    finally:
        # cProfile names/numbers only. No raw pstats, remote environment, task
        # config, frame payloads or credential-bearing function arguments.
        stats = _selected_stats(profile, sources)
        _new_json(directory/'function-profile.json', dict(kind='rbf_allowlisted_original_replay_function_profile_v1',
            functions=stats, raw_call_arguments_persisted=False,
            unallowlisted_functions_or_paths_persisted=False,
            function_wall_times_include_blocking_and_are_not_GPU_kernel_times=True))
        _new_json(directory/'measurement-receipt.json', dict(kind='rbf_original_replay_external_function_profile_v1',
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            status='failed' if failure else 'profile_produced_pending_independent_readback',
            request_sha256=_sha(directory/'request.json'), function_profile_sha256=_sha(directory/'function-profile.json'),
            original_replay_receipt=result, failure=failure,
            profiled_replay_wall_seconds=elapsed, current_process_CPU_seconds=cpu_elapsed,
            device_wait_before_seconds=wait_before, device_wait_after_seconds=wait_after,
            native_function_statistics_are_allowlisted_and_may_be_absent=True,
            wall_seconds_include_profile_overhead_and_output_writing=True,
            function_profile_is_not_independent_physical_GPU_occupancy=True,
            profiler_overhead_quantified=False, full_schedule_coverage_claimed=False,
            paper_cost_or_performance_accepted=False, independent_readback_pending=True))
    return result
