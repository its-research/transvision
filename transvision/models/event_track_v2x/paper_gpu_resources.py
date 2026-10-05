"""Explicit CUDA resource-scan contract and separately measured warm-up/replay.

This is measurement plumbing, not an independent numerical or same-resource
certificate. It never changes candidates, model parameters, batching or limits.
"""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from .detection_cache_v2 import canonical, sha_file

KIND = 'rbf_single_GPU_resource_runtime_v1'


def validate_contract(value):
    required = {'kind', 'device', 'gpu_uuid', 'warmup_events', 'torch_threads', 'torch_interop_threads',
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'forward_batch_protocol', 'TF32', 'memory_target_percent'}
    if not isinstance(value, dict) or set(value) != required or value['kind'] != KIND:
        raise ValueError('explicit complete GPU resource runtime contract required')
    if not isinstance(value['device'], str) or re.fullmatch(r'cuda:[0-9]+', value['device']) is None:
        raise ValueError('explicit indexed CUDA device required')
    if not isinstance(value['gpu_uuid'], str) or re.fullmatch(r'GPU-[A-Za-z0-9-]+', value['gpu_uuid']) is None:
        raise ValueError('physical GPU UUID required')
    for key in ('warmup_events', 'torch_threads', 'torch_interop_threads', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError('positive predeclared warm-up and thread counts required')
    if value['forward_batch_protocol'] != 'native_causal_frontier_one_call_at_a_time_v1' or value['TF32'] is not False:
        raise ValueError('frozen native causal batching and disabled TF32 required')
    if value['memory_target_percent'] != [75, 80]:
        raise ValueError('memory target is recorded separately from achieved usage')
    return value


def load_contract(path, digest, device):
    path = Path(path)
    if not path.is_file() or sha_file(path) != digest:
        raise ValueError('GPU resource runtime contract changed')
    value = validate_contract(json.loads(path.read_bytes()))
    if value['device'] != device:
        raise ValueError('CLI and contract CUDA device differ')
    return value


def configure(value):
    """Before model loading; no fallback to a different GPU or CPU."""
    validate_contract(value)
    import torch
    if not torch.cuda.is_available():
        raise ValueError('requested CUDA runtime unavailable')
    device = torch.device(value['device'])
    if device.index >= torch.cuda.device_count():
        raise ValueError('requested CUDA index unavailable')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        if os.environ.get(key) != str(value[key]):
            raise ValueError('runtime thread environment differs from contract')
    torch.set_num_threads(value['torch_threads'])
    torch.set_num_interop_threads(value['torch_interop_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if str(getattr(props, 'uuid', None)) != value['gpu_uuid']:
        raise ValueError('physical GPU differs from resource contract')
    if 'L40' in props.name.upper():
        raise ValueError('L40 workers are reserved for CPU tasks')
    torch.cuda.synchronize(device)
    return dict(device=str(device), GPU_uuid=str(props.uuid), GPU_name=props.name,
                GPU_total_memory_bytes=props.total_memory, torch_version=torch.__version__, CUDA_version=torch.version.cuda,
                torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32)


def measured_replay(replay, cache, events, output, *, contract, contract_sha256, hardware, setup_seconds, **kwargs):
    from tools.event_track_v2x.rbf_gpu_replay_measurement import measure_replay
    import torch
    validate_contract(contract)
    output = Path(output).absolute(); warmup = output.with_name(output.name+'-warmup')
    warmup_log = output.with_name(output.name+'-warmup.stdout')
    if any(p.exists() for p in (output, warmup, warmup_log)) or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('fresh GPU measurement and warm-up outputs required')
    events = tuple(events); count = contract['warmup_events']
    if not 0 < count <= len(events):
        raise ValueError('warm-up prefix exceeds supplied causal schedule')
    signature = kwargs['scorer'].signature
    started = time.monotonic()
    with warmup_log.open('x') as log, contextlib.redirect_stdout(log):
        warmup_receipt = replay(cache, events[:count], warmup, **kwargs)
    torch.cuda.synchronize(contract['device'])
    warmup_seconds = time.monotonic()-started
    if warmup_receipt['completed_events'] != count or kwargs['scorer'].signature != signature:
        raise ValueError('warm-up did not finish or altered scorer identity')
    # replay constructs fresh trackers/databases; only cache/model/allocator state
    # is warm. No forest branch or past commit is carried into the measured run.
    result = measure_replay(replay, device=contract['device'], output=output,
                            cache=cache, events=events, **kwargs)
    if result['completed_events'] != len(events) or kwargs['scorer'].signature != signature:
        raise ValueError('measured replay incomplete or scorer changed')
    measurement = output/'GPU-runtime-measurement.json'
    value = dict(kind='rbf_GPU_resource_replay_binding_v1', contract=contract, contract_sha256=contract_sha256,
        hardware=hardware, setup_seconds=setup_seconds, warmup_seconds=warmup_seconds,
        warmup=dict(path=str(warmup), receipt_sha256=sha_file(warmup/'receipt.json'),
                    plan_sha256=sha_file(warmup/'plan.json'), resources_sha256=sha_file(warmup/'resources.json'),
                    log_sha256=sha_file(warmup_log), events=count,
                    event_prefix_sha256=hashlib.sha256(canonical(events[:count])).hexdigest()),
        measured_events=len(events), measured_events_sha256=hashlib.sha256(canonical(events)).hexdigest(),
        replay_receipt_sha256=sha_file(output/'receipt.json'), measurement_sha256=sha_file(measurement),
        scorer_signature=signature, warmup_forest_state_reused=False, warmup_timing_in_formal_event_latency=False,
        process_lifetime_host_peak_includes_setup_and_warmup=True,
        GPU_device_usage_includes_other_processes=True, exclusive_GPU_use_independently_verified=False,
        memory_target_independently_accepted=False, full_dataset_verified=False, equal_resources_claimed=False,
        paper_results_verified=False)
    with (output/'GPU-resource-binding.json').open('xb') as stream: stream.write(canonical(value))
    return result


def trial_files():
    return {'replay/GPU-runtime-measurement.json', 'replay/GPU-resource-binding.json',
            'replay-warmup/receipt.json', 'replay-warmup/plan.json', 'replay-warmup/resources.json', 'replay-warmup.stdout'}


def validate_trial(root, contract, digest, events, replay_plan, receipt):
    """Read actual per-trial hardware and warm-up binding before train selection."""
    validate_contract(contract); root = Path(root)
    binding = json.loads((root/'replay/GPU-resource-binding.json').read_bytes())
    measurement_path = root/'replay/GPU-runtime-measurement.json'
    value = json.loads(measurement_path.read_bytes())
    if (value['kind'] != 'rbf_real_replay_per_device_memory_measurement_v1'
            or value['target_percent'] != contract['memory_target_percent']
            or value['target_independently_accepted'] is not False
            or value['isolated_performance_or_paper_acceptance'] is not False
            or value['device_usage_includes_all_processes'] is not True):
        raise ValueError('GPU measurement scope or acceptance claim differs')
    if binding['kind'] != 'rbf_GPU_resource_replay_binding_v1' or binding['contract'] != contract or binding['contract_sha256'] != digest:
        raise ValueError('GPU runtime binding differs')
    if binding['replay_receipt_sha256'] != sha_file(root/'replay/receipt.json') or binding['measurement_sha256'] != sha_file(measurement_path):
        raise ValueError('GPU measurements are not bound to measured replay')
    warm = binding['warmup']; warm_path = root/'replay-warmup'
    for field, filename in (('receipt_sha256','receipt.json'),('plan_sha256','plan.json'),('resources_sha256','resources.json')):
        if warm[field] != sha_file(warm_path/filename): raise ValueError('warm-up evidence changed')
    if warm['path'] != str(warm_path) or warm['log_sha256'] != sha_file(root/'replay-warmup.stdout'):
        raise ValueError('warm-up output binding differs')
    warm_plan = json.loads((warm_path/'plan.json').read_bytes()); warm_receipt = json.loads((warm_path/'receipt.json').read_bytes())
    prefix_digest = hashlib.sha256(canonical(events[:contract['warmup_events']])).hexdigest()
    all_digest = hashlib.sha256(canonical(events)).hexdigest()
    if (warm['events'] != contract['warmup_events'] or warm['event_prefix_sha256'] != prefix_digest
            or warm_plan['events_sha256'] != prefix_digest or warm_plan['expected_events'] != contract['warmup_events']
            or warm_receipt['completed_events'] != contract['warmup_events'] or warm_receipt['status'] != 'software_replay_completed'
            or binding['measured_events'] != receipt['completed_events'] or binding['measured_events'] != len(events)
            or binding['measured_events_sha256'] != replay_plan['events_sha256'] or binding['measured_events_sha256'] != all_digest):
        raise ValueError('warm-up or formal schedule scope differs')
    for key in ('configuration','cache_sha256','model_binding','scorer_signature','fixture','source_sha256','protocol'):
        if warm_plan[key] != replay_plan[key]: raise ValueError('warm-up changed inference contract')
    if binding['scorer_signature'] != replay_plan['scorer_signature']:
        raise ValueError('GPU scorer differs')
    for flag in ('warmup_forest_state_reused','warmup_timing_in_formal_event_latency','exclusive_GPU_use_independently_verified',
                 'memory_target_independently_accepted','full_dataset_verified','equal_resources_claimed','paper_results_verified'):
        if binding[flag] is not False: raise ValueError('GPU measurement has unsupported acceptance claim')
    for flag in ('process_lifetime_host_peak_includes_setup_and_warmup', 'GPU_device_usage_includes_other_processes'):
        if binding[flag] is not True: raise ValueError('GPU or host measurement scope differs')
    hardware = binding['hardware']
    if (hardware['device'] != value['device'] or hardware['device'] != contract['device']
            or hardware['GPU_uuid'] != value['GPU_uuid'] or hardware['GPU_uuid'] != contract['gpu_uuid']
            or hardware['GPU_name'] != value['GPU_name'] or 'L40' in hardware['GPU_name'].upper()
            or value['sampler_finished'] is not True or value['sampler_errors'] or value['replay_failure'] is not None
            or value['completed_events'] != len(events) or not value['samples']
            or hardware['torch_threads'] != contract['torch_threads'] or hardware['torch_interop_threads'] != contract['torch_interop_threads']):
        raise ValueError('GPU device, progress or sampler evidence differs')
    for key in ('torch_version','CUDA_version'):
        if hardware[key] != value[key]: raise ValueError('GPU runtime version differs')
    for key in ('TF32_matmul','TF32_cudnn'):
        if hardware[key] is not False or value[key] is not False: raise ValueError('TF32 must remain disabled')
    total = hardware['GPU_total_memory_bytes']
    if type(total) is not int or total <= 0: raise ValueError('GPU capacity missing')
    previous_time = -1.
    for row in value['samples']:
        if (type(row['device_total_bytes']) is not int or row['device_total_bytes'] != total
                or type(row['device_free_bytes']) is not int or not 0 <= row['device_free_bytes'] <= total):
            raise ValueError('GPU memory sample differs')
        elapsed = row['elapsed_seconds']; used = row['device_used_percent']
        if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < previous_time
                or elapsed < 0 or type(used) not in (int, float) or not math.isfinite(used)
                or not math.isclose(used, 100*(total-row['device_free_bytes'])/total, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('GPU sample timing or occupancy contradicts bytes')
        previous_time = elapsed
        for key in ('process_tensor_allocated_bytes','process_allocator_reserved_bytes'):
            if type(row[key]) is not int or not 0 <= row[key] <= total: raise ValueError('invalid process GPU memory')
        if row['process_tensor_allocated_bytes'] > row['process_allocator_reserved_bytes']:
            raise ValueError('GPU tensor allocation exceeds allocator reserve')
    numbers = dict(peak_device_tensor_bytes=value['peak_process_tensor_allocated_bytes'],
                   peak_device_reserved_bytes=value['peak_process_allocator_reserved_bytes'],
                   replay_wall_seconds=value['elapsed_seconds_including_observer'],
                   setup_seconds=binding['setup_seconds'],warmup_seconds=binding['warmup_seconds'])
    if any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in numbers.values()):
        raise ValueError('invalid measured GPU runtime cost')
    if (numbers['replay_wall_seconds'] <= 0 or previous_time > numbers['replay_wall_seconds']
            or type(numbers['peak_device_tensor_bytes']) is not int or type(numbers['peak_device_reserved_bytes']) is not int
            or numbers['peak_device_tensor_bytes'] > numbers['peak_device_reserved_bytes']):
        raise ValueError('GPU peak allocation or elapsed time contradicts samples')
    rate = value['events_per_second_with_observer']
    if (type(rate) not in (int, float) or not math.isfinite(rate)
            or not math.isclose(rate, len(events)/numbers['replay_wall_seconds'], rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError('GPU throughput contradicts measured event count or time')
    if not max(row['process_tensor_allocated_bytes'] for row in value['samples']) <= numbers['peak_device_tensor_bytes'] <= total:
        raise ValueError('GPU tensor peak contradicts samples')
    if not max(row['process_allocator_reserved_bytes'] for row in value['samples']) <= numbers['peak_device_reserved_bytes'] <= total:
        raise ValueError('GPU allocator peak contradicts samples')
    return numbers, hardware
