"""New, isolated public-native cost pair; never restart an old controller.

Preparation only until the exact source bundle, new binding and all admission
gates are independently checked. Only this owned Docker container can be
stopped. No upload, retries, source mutation, cleanup or paper acceptance.
"""
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
from native_cost_resource_guard import parse_GPU_processes, read_cgroup_memory, read_process_rss, resource_violations


def sha(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    with pathlib.Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def run(arguments):
    return subprocess.check_output(arguments, universal_newlines=True, timeout=20)


def bytes_in(directory):
    total = 0
    for path in pathlib.Path(directory).rglob('*'):
        if path.is_symlink():
            raise ValueError('owned output contains symlink')
        if path.is_file():
            total += path.stat().st_size
    return total


def available_host_RAM():
    return int(next(line.split()[1] for line in pathlib.Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024


def check_GPU_reserve(binding):
    rows = [line.strip().split(',') for line in run([
        'nvidia-smi', '--id=' + binding['GPU_UUID'],
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits']).splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][0].strip() != binding['GPU_UUID']:
        raise ValueError('original physical GPU free-memory observation unavailable')
    free = rows[0][1].strip()
    if not free.isdecimal() or int(free) * 1024**2 < binding['GPU_free_reserve_bytes']:
        raise ValueError('original GPU free reserve not admitted')
    return dict(GPU_UUID=binding['GPU_UUID'], free_bytes=int(free) * 1024**2,
                required_reserve_bytes=binding['GPU_free_reserve_bytes'])


def check_output_resources(pair, control, binding):
    if shutil.disk_usage(pair).free < binding['host_disk_reserve_bytes']:
        raise ValueError('original host disk reserve violated')
    control_bytes = bytes_in(control)
    if control_bytes > 256 * 1024**2:
        raise ValueError('fixed pair control/log byte cap exceeded')
    if bytes_in(pair) + control_bytes > 2 * binding['owned_total_output_cap_bytes']:
        raise ValueError('combined original two-phase output budget exceeded')
    for phase in ('warmup', 'measured'):
        directory = pair / phase
        if directory.exists() and bytes_in(directory) > binding['owned_total_output_cap_bytes']:
            raise ValueError('original per-phase output cap exceeded')


def observe_native_process(PID, binding, GPU_query):
    lines = pathlib.Path('/proc/%d/cgroup' % PID).read_text().splitlines()
    groups = [line.split(':', 2)[2] for line in lines if line.startswith('0::')]
    if len(groups) != 1 or '..' in pathlib.PurePosixPath(groups[0]).parts:
        raise ValueError('owned native cgroup is unavailable')
    CPU = read_cgroup_memory(pathlib.Path('/sys/fs/cgroup') / groups[0].lstrip('/'), binding['CPU_memory_cap_bytes'])
    RSS = read_process_rss('/proc/%d/status' % PID)
    GPU = parse_GPU_processes(run(GPU_query), binding['GPU_UUID'], PID)
    return CPU, RSS, GPU


def check_inputs(binding):
    b = binding
    if sha(b['original_CPU_python']) != b['original_CPU_python_sha256']:
        raise ValueError('original Python bytes differ')
    if sha(b['original_environment_lock']) != b['original_environment_lock_sha256']:
        raise ValueError('original environment lock differs')
    source = pathlib.Path(b['author_source'])
    if run(['git', '-C', str(source), 'rev-parse', 'HEAD']).strip() != b['author_commit']:
        raise ValueError('original author commit differs')
    if run(['git', '-C', str(source), 'diff', 'HEAD', '--']):
        raise ValueError('author source worktree changed')
    for name, record in b['original_source_files'].items():
        path = source / name
        if record['mode'] == '120000':
            if not path.is_symlink() or hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest() != record['sha256']:
                raise ValueError('author symlink differs')
        elif path.is_symlink() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('author source bytes differ: ' + name)
    for name, record in b['original_feature_files'].items():
        path = pathlib.Path(b['original_feature_root']) / name
        if path.is_symlink() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('native feature bytes differ: ' + name)
    for record in b['original_assets'].values():
        path = pathlib.Path(record['path'])
        if path.is_symlink() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('original asset bytes differ')
    runtime = json.loads(pathlib.Path(b['runtime_extracted_receipt']).read_bytes())
    if not runtime['all9775_members_exact'] or len(runtime['all_files']) != 9775:
        raise ValueError('original runtime member qualification differs')
    for record in runtime['all_files']:
        path = pathlib.Path(record['path'])
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('runtime bytes differ')
    for record in b['adapter_files'].values():
        if sha(record['remote_path']) != record['sha256']:
            raise ValueError('original output adapter differs')


def owned_stop(container_name):
    # container_name is the exclusive validated name for this single new pair.
    subprocess.run(['docker', 'stop', '--time', '15', container_name], check=False, timeout=40)


def owned_container_ID(path):
    value = pathlib.Path(path).read_text().strip()
    if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('owned Docker CID file invalid')
    return value


def main(specification_path):
    specification = json.loads(pathlib.Path(specification_path).read_bytes())
    b = specification['original_native_binding']
    pair = pathlib.Path(specification['pair_output'])
    control = pathlib.Path(specification['control_output'])
    name = specification['new_owned_container_name']
    repetition = specification['repetition_index']
    if type(repetition) is not int or repetition not in (0, 1, 2):
        raise ValueError('original three repetition index required')
    if not name.startswith('rbf-native-cost-pair-') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name):
        raise ValueError('new owned container name required')
    if name == b['container_name'] or specification['original_deadline_controller_restarted']:
        raise ValueError('original terminal container/controller cannot be reused')
    if '10.100.35.112' not in run(['hostname', '-I']).split():
        raise ValueError('single GPU cost host must be 35.112')
    if pair.resolve() == pathlib.Path(b['remote_output']).resolve():
        raise ValueError('original owned output cannot be reused')
    if not pair.is_dir() or any(pair.iterdir()) or not control.is_dir() or any(control.iterdir()):
        raise ValueError('new empty owned pair/control directories required')
    if pair.is_symlink() or control.is_symlink() or pair.resolve() == control.resolve():
        raise ValueError('invalid owned output roots')
    bundle = pathlib.Path(specification['source_bundle'])
    if not bundle.is_dir() or bundle.is_symlink():
        raise ValueError('new exact public source bundle required')
    required = {'run_owned_DMSTrack_cost_pair_host.py', 'run_DMSTrack_native_cost_entry.py',
                'native_cost_timing.py', 'native_cost_resource_guard.py', 'original-qualified-GPU-entry.py'}
    if set(specification['source_files']) != required:
        raise ValueError('explicit public-only source allowlist required')
    if {path.name for path in bundle.iterdir()} != required:
        raise ValueError('source bundle contains files outside public-only allowlist')
    for relative, expected in specification['source_files'].items():
        path = bundle / relative
        if path.is_symlink() or sha(path) != expected:
            raise ValueError('cost source bundle SHA differs')
    if pathlib.Path(__file__).resolve() != (bundle / 'run_owned_DMSTrack_cost_pair_host.py').resolve():
        raise ValueError('controller not running from exact frozen bundle')
    if sha(bundle / 'original-qualified-GPU-entry.py') != b['entry_source_sha256']:
        raise ValueError('original qualified CUDA entry changed')
    if b['CPU_threads'] != 2 or b['GPU_memory_cap_bytes'] != 6 * 1024**3 or b['CPU_memory_cap_bytes'] != 6 * 1024**3:
        raise ValueError('original CPU/GPU resource caps changed')
    if b['owned_total_output_cap_bytes'] != 2 * 1024**3 or b['host_disk_reserve_bytes'] != 8 * 1024**3:
        raise ValueError('original phase output/reserve contract changed')
    if b['GPU_free_reserve_bytes'] != 2 * 1024**3:
        raise ValueError('original GPU free reserve contract changed')
    if b['maximum_native_execution_seconds'] != 21600 or specification['sampling_interval_seconds'] != 1:
        raise ValueError('frozen per-phase deadline or sampling interval changed')
    if b['GPU_UUID'] != 'GPU-68c03b8c-acc3-421c-0e7a-8da0bb9ad3fa':
        raise ValueError('original physical 35.112 GPU required')
    process = None
    CID_file = control / 'owned-container.cid'
    try:
        # Reserve all remaining phase budgets plus the unchanged host reserve.
        # Pair controls also fit within the combined two-phase output budget.
        remaining_capacity = (3 - repetition) * 2 * b['owned_total_output_cap_bytes'] + b['host_disk_reserve_bytes']
        if shutil.disk_usage(pair).free < remaining_capacity:
            raise ValueError('remaining full-phase retention and original reserve not admitted')
        if specification.get('disk_admission_probe_only') is True:
            write(control / 'native-controller-disk-admission-only-returned.json', dict(
                required_available_bytes=remaining_capacity, disk_admitted=True,
                full_resource_admission=False, native_process_created=False))
            return
        available = available_host_RAM()
        if available < b['CPU_memory_cap_bytes'] + b['host_RAM_reserve_bytes']:
            raise ValueError('original host RAM reserve not admitted')
        if run(['docker', 'image', 'inspect', '--format', '{{.Id}}', b['image']]).strip() != b['image']:
            raise ValueError('original Docker image differs')
        if run(['docker', 'ps', '-a', '--filter', 'name=^/' + name + '$', '--format', '{{.ID}}']).strip():
            raise ValueError('new owned container name already occupied')
        GPU_query = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits']
        if parse_GPU_processes(run(GPU_query), b['GPU_UUID'], -1)['other_same_GPU_processes']:
            raise ValueError('original physical GPU is occupied')
        check_GPU_reserve(b)
        check_inputs(b)
        write(control / 'all-original-inputs-preexecution-byte-readback.json', dict(all_original_inputs_full_SHA_exact=True))
        entry_binding = dict(original_native_binding=b, pair_output=str(pair),
            repetition_index=specification['repetition_index'], container_cgroup_mount='/sys/fs/cgroup',
            original_entry_path=str(bundle / 'original-qualified-GPU-entry.py'),
            owned_process_guard=dict(GPU_UUID=b['GPU_UUID'], CPU_threads=2,
                GPU_process_resident_cap_bytes=b['GPU_memory_cap_bytes'],
                CPU_process_resident_cap_bytes=b['CPU_memory_cap_bytes'], live_monitor_PID=os.getpid(),
                original_deadline_controller_restarted=False))
        entry_binding_path = control / 'pair-entry-binding.json'
        write(entry_binding_path, entry_binding)
        paths = [str(bundle), b['runtime_site'], b['original_CPU_site_packages'], b['author_source'],
                 b['author_source'] + '/DMSTrack', b['author_source'] + '/AB3DMOT',
                 b['author_source'] + '/AB3DMOT/Xinshuo_PyToolbox', b['author_source'] + '/V2V4Real']
        command = ['docker', 'run', '--rm', '--name', name, '--cidfile', str(CID_file),
            '--label', 'rbf.cost.execution=' + name, '--read-only', '--network', 'none',
            '--cgroupns', 'private',
            '--user', '1000:1000', '--cpus', '2', '--memory', str(b['CPU_memory_cap_bytes']),
            '--memory-swap', str(b['CPU_memory_cap_bytes']), '--pids-limit', '128',
            '--tmpfs', '/tmp:rw,size=128m', '--gpus', 'device=' + b['GPU_UUID']]
        for path, mode in [(b['original_CPU_root'], 'ro'), (b['original_Python_prefix'], 'ro'),
                           (b['runtime_RAM_root'], 'ro'), (str(bundle), 'ro'), (str(pair), 'rw'), (str(control), 'rw')]:
            command.extend(['-v', path + ':' + path + ':' + mode])
        command.extend(['-w', b['author_source'] + '/DMSTrack'])
        for value in ['PYTHONPATH=' + ':'.join(paths), 'PYTHONNOUSERSITE=1', 'PYTHONDONTWRITEBYTECODE=1',
                      'CUDA_VISIBLE_DEVICES=0', 'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2',
                      'OPENBLAS_NUM_THREADS=2', 'MPLCONFIGDIR=/tmp/matplotlib']:
            command.extend(['-e', value])
        command.extend(['--entrypoint', b['original_CPU_python'], b['image'],
                        str(bundle / 'run_DMSTrack_native_cost_entry.py'), str(entry_binding_path)])
        write(control / 'new-owned-container-command.json', dict(argv=command))
        started = time.monotonic()
        owned_ID = None
        samples = 0
        GPU_phase_samples = dict(warmup=0, measured=0)
        violation = None
        with (control / 'native-pair.log').open('xb') as log, (control / 'resource-samples.jsonl').open('x') as observations:
            # Full input hashing can take time. Recheck immediately before creation.
            if parse_GPU_processes(run(GPU_query), b['GPU_UUID'], -1)['other_same_GPU_processes']:
                raise ValueError('original physical GPU became occupied during input readback')
            check_GPU_reserve(b)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if time.monotonic() - started > 2 * b['maximum_native_execution_seconds']:
                    raise ValueError('new pair total fixed deadline exceeded')
                check_output_resources(pair, control, b)
                GPU_reserve = check_GPU_reserve(b)
                available = available_host_RAM()
                if available < b['host_RAM_reserve_bytes']:
                    raise ValueError('original host RAM reserve violated')
                active_phases = []
                for phase in ('warmup', 'measured'):
                    directory = pair / phase
                    begin, end = directory / 'native-cost-phase-start.json', directory / 'native-cost-phase-observation.json'
                    if begin.is_file() and not end.is_file():
                        active_phases.append(phase)
                        phase_start = json.loads(begin.read_bytes())['started_monotonic_seconds']
                        if time.monotonic() - phase_start > b['maximum_native_execution_seconds']:
                            raise ValueError('original per-phase fixed deadline exceeded')
                inspect = subprocess.run(['docker', 'inspect', name], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         universal_newlines=True, timeout=20)
                if inspect.returncode == 0:
                    item = json.loads(inspect.stdout)[0]
                    if (item['Config']['Labels'] or {}).get('rbf.cost.execution') != name:
                        raise ValueError('Docker owned execution label mismatch')
                    if not CID_file.is_file() or owned_container_ID(CID_file) != item['Id']:
                        raise ValueError('container ID does not match own Docker creation')
                    if owned_ID is None:
                        owned_ID = item['Id']
                    elif item['Id'] != owned_ID:
                        raise ValueError('owned container identity changed')
                    PID = item['State']['Pid']
                    if PID:
                        CPU, RSS, GPU = observe_native_process(PID, b, GPU_query)
                        violation = resource_violations(CPU, RSS, GPU, b['CPU_memory_cap_bytes'], b['GPU_memory_cap_bytes'])
                        observations.write(json.dumps(dict(monotonic_seconds=time.monotonic(), active_phases=active_phases,
                            CPU=CPU, process=RSS, GPU=GPU, GPU_free_reserve=GPU_reserve), allow_nan=False) + '\n')
                        observations.flush()
                        samples += 1
                        if GPU['owned_GPU_processes']:
                            for phase in active_phases:
                                GPU_phase_samples[phase] += 1
                        if violation:
                            raise ValueError('owned native resource violation: ' + repr(violation))
                elif time.monotonic() - started > 30:
                    raise ValueError('cannot observe owned container identity')
                time.sleep(specification['sampling_interval_seconds'])
        if process.returncode != 0 or samples == 0:
            raise ValueError('native pair failed or no resource observations available')
        if any(count == 0 for count in GPU_phase_samples.values()):
            raise ValueError('owned GPU resource observations missing for a native phase')
        check_output_resources(pair, control, b)
        if not (pair / 'native-cost-pair-returned.json').is_file():
            raise ValueError('complete native pair return receipt missing')
        check_inputs(b)
        write(control / 'all-original-inputs-postexecution-byte-readback.json', dict(all_original_inputs_full_SHA_exact=True))
        write(control / 'owned-native-cost-pair-terminal.json', dict(returncode=process.returncode,
            elapsed_seconds=time.monotonic() - started, sampled_resource_observations=samples,
            owned_GPU_phase_samples=GPU_phase_samples,
            GPU_peak_resolution='sampled only; no exact high-water claim', formal_cost_accepted=False,
            independent_L40_output_and_three_repetition_readback_pending=True))
        # Include the final readback and terminal receipt bytes in the same caps.
        check_output_resources(pair, control, b)
    except BaseException:
        original_failure = traceback.format_exc()
        cleanup_failure = None
        try:
            if process is not None and CID_file.is_file():
                # A Docker client may exit before its container does. Target
                # only the CID from this invocation even if the client exited.
                owned_stop(owned_container_ID(CID_file))
                if process.poll() is None:
                    process.wait(timeout=60)
        except BaseException:
            cleanup_failure = traceback.format_exc()
        write(control / 'owned-native-cost-pair-failure-preserved.json', dict(
            traceback=original_failure, owned_cleanup_failure=cleanup_failure, partial_outputs_retained=True,
            original_jobs_untouched=True, automatic_retry=False, formal_cost_accepted=False))
        raise


if __name__ == '__main__':
    main(sys.argv[1])
