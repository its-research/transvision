#!/usr/bin/env python3
"""Verify a pinned release ZIP and extract native inputs into a fresh directory.

The supplied release snapshot is a provenance trust input, not an authentication
service. One verified volume never certifies the whole official split. Raw YAML
may contain GT and must remain outside the inference-only projection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.event_track_v2x.prepare_v2v4real_inputs import inventory_native_root

MAX_FILES = 200_000
MAX_FILE_BYTES = 128 * 1024**2
MAX_TOTAL_BYTES = 32 * 1024**3


def digest_file(path):
    hashes = hashlib.sha1(), hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            for digest in hashes:
                digest.update(block)
    return tuple(d.hexdigest() for d in hashes)


def ordinary(path, *, directory=False):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink input or destination is not allowed')
    mode = path.stat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError('ordinary file/directory required')
    return path


def stamp(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def checked_members(archive):
    entries = archive.infolist()
    if not entries or len(entries) > MAX_FILES:
        raise ValueError('ZIP entry count exceeds limit or is empty')
    seen = set()
    total = 0
    for row in entries:
        name = row.filename
        path = name[:-1] if row.is_dir() else name
        parts = path.split('/')
        if (not path or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', p) or p in ('.', '..') for p in parts)
                or name != row.orig_filename or path.casefold() in seen
                or row.flag_bits & 1 or row.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
            raise ValueError('unsafe, duplicate or unsupported ZIP entry')
        seen.add(path.casefold())
        mode = stat.S_IFMT(row.external_attr >> 16)
        if mode not in ((0, stat.S_IFDIR) if row.is_dir() else (0, stat.S_IFREG)):
            raise ValueError('only ordinary ZIP files/directories allowed')
        if row.is_dir():
            if len(parts) not in (1, 2) or row.file_size != 0:
                raise ValueError('unexpected native directory')
        elif (len(parts) != 3 or re.fullmatch(r'[0-9]+', parts[1]) is None
              or re.fullmatch(r'[0-9]+\.(pcd|yaml)', parts[2]) is None
              or not 0 < row.file_size <= MAX_FILE_BYTES):
            raise ValueError('expected bounded sequence/CAV/frame.yaml or .pcd')
        total += row.file_size
        if total > MAX_TOTAL_BYTES:
            raise ValueError('ZIP uncompressed size exceeds limit')
    return entries, total


def extract_volume(archive_path, release_path, release_sha256, output):
    archive_path, release_path = ordinary(archive_path), ordinary(release_path)
    if re.fullmatch(r'[0-9a-f]{64}', release_sha256) is None:
        raise ValueError('explicit release SHA-256 required')
    raw = release_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != release_sha256:
        raise ValueError('release snapshot SHA-256 differs')
    meta = json.loads(raw)
    if meta.get('kind') != 'v2v4real_official_box_metadata_snapshot_v1' or meta.get('dataset') != 'V2V4Real':
        raise ValueError('V2V4Real release snapshot required')
    records = [r for r in meta['files'] if r['name'] == archive_path.name]
    if len(records) != 1 or records[0]['split'] not in ('train', 'test'):
        raise ValueError('exactly one authorized train/test volume required')
    record = records[0]
    original_stamp = stamp(archive_path)
    if (type(record['size_bytes']) is not int or record['size_bytes'] != original_stamp[2]
            or re.fullmatch(r'[0-9a-f]{40}', record['reported_sha1']) is None):
        raise ValueError('archive length or expected SHA-1 differs')
    source_sha1, source_sha256 = digest_file(archive_path)
    if source_sha1 != record['reported_sha1'] or stamp(archive_path) != original_stamp:
        raise ValueError('archive SHA-1 differs or source changed')
    output = Path(output).absolute()
    ordinary(output.parent, directory=True)
    if output.exists() or output.is_symlink():
        raise ValueError('fresh output directory required')
    with zipfile.ZipFile(archive_path) as archive:
        entries, total = checked_members(archive)
        if shutil.disk_usage(output.parent).free < total + 1024**3:
            raise ValueError('insufficient free space for extracted volume and 1 GiB reserve')
        staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
        try:
            payload = staging / 'payload'
            payload.mkdir()
            inventory = []
            for row in entries:
                target = payload / row.filename
                if row.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                digest, size = hashlib.sha256(), 0
                with archive.open(row) as inp, target.open('xb') as out:
                    while block := inp.read(1024**2):
                        size += len(block)
                        if size > row.file_size:
                            raise ValueError('ZIP payload exceeds declared size')
                        digest.update(block)
                        out.write(block)
                if size != row.file_size:
                    raise ValueError('ZIP payload size differs')
                inventory.append(dict(path=row.filename, bytes=size, sha256=digest.hexdigest()))
            native = inventory_native_root(payload)
            if stamp(archive_path) != original_stamp or digest_file(archive_path) != (source_sha1, source_sha256):
                raise ValueError('source changed during extraction')
            receipt = dict(kind='v2v4real_verified_volume_v1', dataset='V2V4Real', split=record['split'],
                archive_record=record, archive_sha256=source_sha256, release_sha256=release_sha256,
                extractor_sha256=digest_file(Path(__file__))[1], payload_verified=True,
                full_official_split_verified=False, raw_yaml_may_contain_GT=True,
                inference_ready=False, paper_eligible=False, uncompressed_bytes=total,
                files=inventory, native_inventory=native)
            (staging / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
            (staging / 'release-snapshot.json').write_bytes(raw)
            if output.exists() or output.is_symlink():
                raise ValueError('destination appeared during extraction')
            # mkdir() claims the pathname without replacing existing output;
            # publish the receipt last so a partial move is not a success.
            output.mkdir()
            try:
                for item in (staging / 'release-snapshot.json', payload, staging / 'receipt.json'):
                    os.rename(item, output / item.name)
            except BaseException:
                raise RuntimeError(f'incomplete output retained for inspection: {output}')
            return receipt
        finally:
            shutil.rmtree(staging)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--release-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    result = extract_volume(a.archive, a.release, a.release_sha256, a.output)
    print(json.dumps({k: result[k] for k in ('archive_sha256', 'payload_verified', 'uncompressed_bytes',
        'full_official_split_verified', 'inference_ready', 'paper_eligible')}, sort_keys=True))


if __name__ == '__main__':
    main()
