"""Hardware admission for priority training; device names are not a whitelist."""


def validate_runtime(runtimes, *, require_full_train):
    if type(require_full_train) is not bool:
        raise TypeError('explicit full-train requirement required')
    if not runtimes or any(r['world_size'] != len(runtimes) for r in runtimes):
        raise ValueError('complete actual rank inventory required')
    if sorted(r['rank'] for r in runtimes) != list(range(len(runtimes))) or len({r['host'] for r in runtimes}) != 1:
        raise ValueError('distinct ranks on one worker required')
    if not require_full_train:
        return
    if (len(runtimes) < 4
            or any(r['backend'] != 'nccl' or not r['device'].startswith('cuda:') for r in runtimes)
            or len({r['device'] for r in runtimes}) != len(runtimes)
            or len({r.get('gpu_uuid') for r in runtimes}) != len(runtimes)
            or any(not r.get('gpu_uuid') or r['gpu_uuid'] in ('unavailable', 'None') for r in runtimes)):
        raise ValueError('at least four distinct actual CUDA/NCCL GPUs required')
    for runtime in runtimes:
        capability = runtime.get('capability')
        if (not runtime.get('gpu_name') or 'L40' in runtime['gpu_name']
                or not isinstance(capability, (list, tuple)) or len(capability) != 2
                or any(type(v) is not int or v < 0 for v in capability)
                or 'sm_' + ''.join(map(str, capability)) not in runtime.get('native_architectures', ())
                or type(runtime.get('total_memory_bytes')) is not int or runtime['total_memory_bytes'] <= 0
                or runtime.get('TF32_matmul') is not False or runtime.get('TF32_cudnn') is not False):
            raise ValueError('native GPU architecture, actual device memory and disabled TF32 required; L40S is CPU-only')
