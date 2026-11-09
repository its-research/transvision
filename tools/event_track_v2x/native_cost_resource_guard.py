"""Read-only Linux/Docker resource observations for an owned native cost run.

The caller owns dispatch, isolation, deadline, disk/output checks and stopping
its container. GPU resident peaks below are sampled, never exact high-water
claims. CPU cgroup accounting and process RSS are reported separately.
"""
import pathlib


def _counter(path):
    value = pathlib.Path(path).read_text().strip()
    if not value.isdecimal():
        raise ValueError('non-numeric Linux memory counter: ' + str(path))
    return int(value)


def _key_values(path):
    rows = {}
    for line in pathlib.Path(path).read_text().splitlines():
        cells = line.split()
        if len(cells) != 2 or cells[0] in rows or not cells[1].isdecimal():
            raise ValueError('invalid Linux memory accounting row')
        rows[cells[0]] = int(cells[1])
    return rows


def read_cgroup_memory(directory, expected_cap_bytes):
    root = pathlib.Path(directory)
    current = _counter(root / 'memory.current')
    peak = _counter(root / 'memory.peak')
    maximum = _counter(root / 'memory.max')
    if maximum != expected_cap_bytes:
        raise ValueError('container hard memory cap differs from frozen contract')
    if peak < current:
        # Reads are not atomic: if memory grew between them, peak is still
        # required to cover the earlier current value. Never repair the data.
        raise ValueError('cgroup peak contradicts current memory counter')
    events = _key_values(root / 'memory.events')
    if not {'oom', 'oom_kill', 'max'} <= set(events):
        raise ValueError('required cgroup memory events missing')
    stats = _key_values(root / 'memory.stat')
    return dict(cgroup_accounted_current_bytes=current,
                cgroup_accounted_peak_bytes=peak, cgroup_hard_cap_bytes=maximum,
                memory_events=events, memory_stat_current_snapshot=stats,
                current_categories_not_peak_decomposition=True,
                shared_file_page_accounting_not_process_RSS=True)


def read_process_rss(status_path):
    rows = {}
    for line in pathlib.Path(status_path).read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            cells = line.split()
            if len(cells) != 3 or cells[2] != 'kB' or not cells[1].isdecimal():
                raise ValueError('invalid Linux process RSS unit/counter')
            rows[cells[0][:-1]] = int(cells[1]) * 1024
    if set(rows) != {'VmRSS', 'VmHWM'} or rows['VmHWM'] < rows['VmRSS']:
        raise ValueError('missing or contradictory Linux RSS high-water mark')
    return dict(process_RSS_bytes=rows['VmRSS'], process_RSS_high_water_bytes=rows['VmHWM'])


def parse_GPU_processes(csv_text, GPU_UUID, owned_native_PID):
    owned = []
    external = []
    for line in csv_text.splitlines():
        if not line.strip():
            continue
        cells = [x.strip() for x in line.split(',')]
        if len(cells) != 3:
            raise ValueError('invalid nvidia-smi process row')
        if cells[0] != GPU_UUID:
            continue
        if not cells[1].isdecimal() or not cells[2].isdecimal():
            raise ValueError('GPU PID or resident memory unavailable')
        row = dict(PID=int(cells[1]), GPU_resident_bytes=int(cells[2]) * 1024**2)
        (owned if row['PID'] == owned_native_PID else external).append(row)
    if len(owned) > 1:
        raise ValueError('duplicate owned GPU process row')
    return dict(owned_GPU_processes=owned, other_same_GPU_processes=external,
                GPU_process_resident_observation_is_sampled=True,
                CUDA_context_included_in_resident_counter=True)


def resource_violations(cgroup, process, GPU, CPU_cap_bytes, GPU_cap_bytes):
    violations = []
    if cgroup['cgroup_accounted_peak_bytes'] > CPU_cap_bytes:
        violations.append('owned_cgroup_memory_peak_cap')
    if process['process_RSS_high_water_bytes'] > CPU_cap_bytes:
        violations.append('owned_native_process_RSS_high_water_cap')
    if any(cgroup['memory_events'].get(key, 0) for key in ('oom', 'oom_kill', 'max')):
        violations.append('owned_cgroup_hard_limit_or_OOM_event')
    if GPU['other_same_GPU_processes']:
        violations.append('other_process_appeared_on_same_physical_GPU')
    if any(row['GPU_resident_bytes'] > GPU_cap_bytes for row in GPU['owned_GPU_processes']):
        violations.append('owned_GPU_process_total_resident_cap')
    return violations
