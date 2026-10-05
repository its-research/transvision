#!/usr/bin/env python3
"""Create a deterministic diagnostic package from the pinned historical source.

This is a local preparation step. It neither publishes to ClearML nor enqueues
training. The official source tree is copied byte-for-byte from its pinned tar.
"""
import argparse
import gzip
import hashlib
import io
from pathlib import Path, PurePosixPath
import tarfile

HISTORICAL_SHA256 = {
    '7968e53e7ab244cc4780926e25ae33fc707c33d61f771c697c3d92a1f1b616ff',
    '7e797d124d4e5ae90cea4306558e6aa5e0396ebc89fcc1c18e3df6bce641477f',
}
EXPECTED_REPLACEMENTS = {
    'project/tools/event_track_v2x/run_v2v4real_detector_clearml.py',
    'project/tools/event_track_v2x/train_v2v4real_detector.py',
}
ADDITIONS = {
    'project/tools/event_track_v2x/materialize_v2v4real_overlap_controlled.py',
    'project/transvision/models/event_track_v2x/experiment_progress.py',
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build(historical: Path, output: Path, replacements: dict[str, Path]) -> str:
    if digest(historical) not in HISTORICAL_SHA256:
        raise ValueError('historical source archive differs')
    if set(replacements) != EXPECTED_REPLACEMENTS | ADDITIONS:
        raise ValueError('candidate source member set differs')
    if output.exists() or output.is_symlink():
        raise ValueError('fresh output required')
    original = {}
    with tarfile.open(historical, 'r:gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or not member.isfile()
                    or name.as_posix() in original
                    or name.parts[0] not in {'project', 'official'}):
                raise ValueError('unsafe historical source member')
            original[name.as_posix()] = archive.extractfile(member).read()
    if not EXPECTED_REPLACEMENTS <= set(original) or ADDITIONS & set(original):
        raise ValueError('historical member identities differ')
    if any(not path.is_file() or path.is_symlink() for path in replacements.values()):
        raise ValueError('ordinary candidate source files required')
    for name, path in replacements.items():
        original[name] = path.read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=9, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w') as archive:
                for name, data in sorted(original.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    archive.addfile(info, io.BytesIO(data))
    return digest(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical', required=True, type=Path)
    p.add_argument('--runner', required=True, type=Path)
    p.add_argument('--trainer', required=True, type=Path)
    p.add_argument('--materializer', required=True, type=Path)
    p.add_argument('--progress', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    print(build(a.historical, a.output, {
        'project/tools/event_track_v2x/run_v2v4real_detector_clearml.py': a.runner,
        'project/tools/event_track_v2x/train_v2v4real_detector.py': a.trainer,
        'project/tools/event_track_v2x/materialize_v2v4real_overlap_controlled.py': a.materializer,
        'project/transvision/models/event_track_v2x/experiment_progress.py': a.progress,
    }))


if __name__ == '__main__':
    main()
