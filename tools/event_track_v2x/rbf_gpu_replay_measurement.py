"""Measure real replay work; device occupancy is distinct from tensor allocation."""
import json
from pathlib import Path
import threading
import time


def measure_replay(replay, *, device, output, **kwargs):
    import torch

    device = torch.device(device)
    assert device.type == 'cuda' and device.index is not None
    props = torch.cuda.get_device_properties(device)
    uuid = str(getattr(props, 'uuid', 'unavailable'))
    assert uuid not in ('unavailable', '', 'None')
    output = Path(output)
    assert not output.exists()
    samples, errors = [], []
    stop = threading.Event()
    started = time.monotonic()

    def sample():
        free, total = torch.cuda.mem_get_info(device)
        assert total > 0 and 0 <= free <= total
        samples.append(dict(elapsed_seconds=time.monotonic()-started,
            device_free_bytes=free, device_total_bytes=total,
            device_used_percent=100*(total-free)/total,
            process_tensor_allocated_bytes=torch.cuda.memory_allocated(device),
            process_allocator_reserved_bytes=torch.cuda.memory_reserved(device)))

    def observe():
        try:
            torch.cuda.set_device(device)
            while not stop.wait(1.0):
                sample()
        except Exception as error:
            errors.append(dict(type=type(error).__name__, message=str(error)))

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    sample()
    worker = threading.Thread(target=observe, daemon=True)
    worker.start()
    receipt = None
    failure = None
    try:
        receipt = replay(output=output, **kwargs)
        return receipt
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        stop.set()
        worker.join(timeout=10)
        sampler_finished = not worker.is_alive()
        try:
            torch.cuda.synchronize(device)
            if sampler_finished:
                sample()
        except Exception as error:
            errors.append(dict(type=type(error).__name__, message=str(error)))
        elapsed = time.monotonic()-started
        output.mkdir(exist_ok=True)
        evidence = dict(kind='rbf_real_replay_per_device_memory_measurement_v1',
            device=str(device), GPU_uuid=uuid, GPU_name=props.name,
            torch_version=torch.__version__, CUDA_version=torch.version.cuda,
            TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
            TF32_cudnn=torch.backends.cudnn.allow_tf32,
            elapsed_seconds_including_observer=elapsed,
            completed_events=receipt.get('completed_events') if receipt else None,
            events_per_second_with_observer=(receipt['completed_events']/elapsed
                if receipt and elapsed > 0 else None),
            peak_process_tensor_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_process_allocator_reserved_bytes=torch.cuda.max_memory_reserved(device),
            samples=list(samples), sampler_errors=errors,
            sampler_finished=sampler_finished, replay_failure=failure,
            target_percent=[75, 80], target_independently_accepted=False,
            device_usage_includes_all_processes=True,
            isolated_performance_or_paper_acceptance=False,
            measured_scope='real complete sequence including CPU forest work; no synthetic allocations')
        with (output/'GPU-runtime-measurement.json').open('x') as stream:
            json.dump(evidence, stream, sort_keys=True, allow_nan=False)
            stream.write('\n')
