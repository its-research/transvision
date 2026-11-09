#!/usr/bin/env python3
"""Read every pinned Git blob without following author dataset symlinks."""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    def git(*command):
        return subprocess.check_output(['git', '-C', str(args.source), *command])
    assert git('rev-parse', 'HEAD').decode().strip() == args.commit
    assert not git('diff', 'HEAD', '--') and not git('ls-files', '--others', '--exclude-standard')
    records = {}
    links = []
    total = 0
    for row in git('ls-tree', '-r', '-z', args.commit).split(b'\0'):
        if not row:
            continue
        info, raw_name = row.split(b'\t', 1)
        mode, kind, oid = info.decode().split()
        name = raw_name.decode()
        assert kind == 'blob' and mode in ('100644', '100755', '120000')
        path = args.source / name
        if mode == '120000':
            assert path.is_symlink()
            raw = os.fsencode(os.readlink(path))
            links.append(dict(path=name, literal_target=os.fsdecode(raw), target_dereferenced=False))
        else:
            assert path.is_file() and not path.is_symlink()
            raw = path.read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        assert actual == oid
        records[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                             git_blob_oid=oid, mode=mode)
        total += len(raw)
    assert git('rev-parse', 'HEAD').decode().strip() == args.commit and not git('diff', 'HEAD', '--')
    value = dict(kind='rbf_complete_pinned_author_source_independent_tracked_blob_readback_v2',
                 source=str(args.source), author_commit=args.commit,
                 tracked_files=len(records), tracked_bytes=total, files=records,
                 author_symlink_blobs=links, author_symlink_targets_dereferenced=False,
                 all_tracked_git_blobs_and_SHA_independently_read=True,
                 git_HEAD_clean=True, untracked_files=[],
                 auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 native_environment_qualified=False, native_predictions_produced=False,
                 paper_complete=False,
                 checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    args.output.open('x').write(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(tracked_files=len(records), tracked_bytes=total,
                          author_symlink_blobs=len(links), receipt=str(args.output))))


if __name__ == '__main__':
    main()
