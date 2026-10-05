"""Collect native public-baseline bytes without treating them as accepted metrics."""
import hashlib
import math
import os
from pathlib import Path

DMSTRACK_CONFIG_SHA256 = '99fb6da379a46825d16e7e5b19355b5596cc1fc2131125793ea6607816e2e005'
DMSTRACK_RUN = 'evaluation_multi_sensor_differentiable_kalman_filter_Car_val_all_H1_epoch_0'
LENGTHS = (147, 114, 144, 198, 180, 310, 304, 221, 375)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def regular(path, *, nonempty=False):
    path = Path(path)
    if (not path.is_file() or any(p.is_symlink() for p in (path, *path.parents))
            or (nonempty and path.stat().st_size == 0)):
        raise ValueError('missing, linked or empty native file: '+str(path))
    return path


def tree_files(root):
    root = Path(root)
    if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('missing or linked native output directory')
    files = {}
    for directory, names, filenames in os.walk(root, followlinks=False):
        directory = Path(directory)
        if any((directory/n).is_symlink() for n in names):
            raise ValueError('linked directory in native output')
        for name in sorted(filenames):
            p = regular(directory/name)
            files[str(p.relative_to(root))] = dict(sha256=digest(p), bytes=p.stat().st_size)
    return files


def contract(plan, output):
    source, output = Path(plan['source']).absolute(), Path(output).absolute()
    if plan['method'] == 'CoopTrack':
        # tools/test.py overwrites kwargs['jsonfile_prefix'] with this native
        # config-derived path even when --eval-options supplies another path.
        stem = Path(plan['configuration']['path']).name.split('.')[-2]
        return dict(kind='cooptrack_native_output_paths_v1', root=str(output),
                    raw_predictions='native-output.pkl',
                    evaluation_root=str(source/'test'/stem), native_code_changed=False)
    if plan['method'] == 'DMSTrack':
        config = regular(source/'DMSTrack/configs/v2v4real.yml', nonempty=True)
        if digest(config) != DMSTRACK_CONFIG_SHA256:
            raise ValueError('fixed DMSTrack native configuration changed')
        # Absolute --save_dir_prefix wins over cfg.save_root in os.path.join.
        return dict(kind='dmstrack_native_output_paths_v1', root=str(output/'native-results'),
                    run=DMSTRACK_RUN, configuration=dict(path=str(config), sha256=digest(config)),
                    sequence_frames={f'{i:04d}': n for i, n in enumerate(LENGTHS)},
                    native_code_changed=False)
    raise ValueError('unsupported public output contract')


def validate_destination(value):
    if value['kind'] == 'cooptrack_native_output_paths_v1':
        path = Path(value['evaluation_root'])
        if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('new isolated CoopTrack native evaluation root required')


def parse_dmstrack(directory):
    """Validate all nine native sequence files; no fabricated per-frame records."""
    directory = Path(directory)
    expected = {f'{i:04d}.txt': n for i, n in enumerate(LENGTHS)}
    if not directory.is_dir() or set(p.name for p in directory.iterdir()) != set(expected):
        raise ValueError('all nine native prediction sequence files required')
    counts = {}
    for name, length in expected.items():
        p = regular(directory/name)
        identities = set()
        rows = p.read_text().splitlines()
        for line in rows:
            cells = line.split()
            if len(cells) != 18 or cells[2] != 'Car':
                raise ValueError('native prediction requires 18 KITTI fields and merged Car alias')
            try:
                frame, ident = int(cells[0]), int(cells[1])
                values = [float(x) for x in cells[3:]]
            except ValueError as e:
                raise ValueError('invalid native prediction numeric field') from e
            if (not 0 <= frame < length or ident < 0 or (frame, ident) in identities
                    or not all(math.isfinite(x) for x in values)
                    or min(values[7:10]) <= 0
                    or values[3] > values[5] or values[4] > values[6]):
                raise ValueError('invalid native frame, identity, box or finite value')
            identities.add((frame, ident))
        counts[name] = dict(rows=len(rows), declared_frames=length,
                            frames_with_predictions=len({f for f, _ in identities}))
    return counts


def read_summary(path):
    """Parse the exact official three-value summary, not a metric recomputation."""
    lines = regular(path, nonempty=True).read_text().splitlines()
    header = [i for i, line in enumerate(lines) if line.split() == ['sAMOTA', 'AMOTA', 'AMOTP']]
    if len(header) != 1 or header[0]+1 >= len(lines):
        raise ValueError('native averaged tracking summary header missing or duplicated')
    try:
        values = [float(x) for x in lines[header[0]+1].split()]
    except ValueError as e:
        raise ValueError('invalid native averaged summary') from e
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise ValueError('three finite native averaged metrics required')
    return dict(zip(('sAMOTA', 'AMOTA', 'AMOTP'), values))


def collect(value):
    root = Path(value['root'])
    result = dict(root=str(root), official_metrics_verified=False,
                  output_semantics_verified=False, unpickled=False)
    if value['kind'] == 'cooptrack_native_output_paths_v1':
        raw = regular(root/value['raw_predictions'], nonempty=True)
        raw_sha256 = digest(raw)
        evaluation = Path(value['evaluation_root'])
        if not evaluation.is_dir() or evaluation.is_symlink():
            raise ValueError('CoopTrack native evaluation output missing')
        runs = list(evaluation.iterdir())
        if len(runs) != 1 or not runs[0].is_dir():
            raise ValueError('one isolated CoopTrack evaluation timestamp required')
        files = tree_files(runs[0])
        if not files:
            raise ValueError('empty CoopTrack native evaluation output')
        if (digest(regular(raw, nonempty=True)) != raw_sha256
                or tree_files(runs[0]) != files
                or list(evaluation.iterdir()) != runs):
            raise ValueError('CoopTrack native outputs changed while being collected')
        result.update(raw_predictions=str(raw), raw_sha256=raw_sha256,
                      evaluation_root=str(runs[0]), files=files)
    else:
        config = value['configuration']
        if digest(regular(config['path'])) != config['sha256']:
            raise ValueError('DMSTrack native configuration changed during execution')
        run = root/value['run']
        before = tree_files(run)
        counts = parse_dmstrack(run/'data_0')
        metrics = read_summary(run/'summary_car_average_eval3D.txt')
        if tree_files(run) != before:
            raise ValueError('native outputs changed while being collected')
        result.update(files=before, run=str(run), sequence_output_schema=counts,
                      reported_metrics=metrics, metric_direction={k:'higher' for k in metrics},
                      metric_definition='native V2V4Real recall-averaged 3D IoU at 0.25',
                      frame_coverage_independently_verified=False)
    return result
