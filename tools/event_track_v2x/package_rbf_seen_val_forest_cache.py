"""Create-once full V2 transport archive with independent streamed readback."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

from rbf_nested_seen_val_v2_common import R, new, register, sha

BASE = 'rbf-seen-val-forest-input-bridge-v1-20261005'
NAME = 'rbf-seen-val-forest-cache-transport-v1-20261005'


def checked_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or p.as_posix() != relative:
        raise ValueError('unsafe archive member')
    path = root/p
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()) or any(
            q.is_symlink() for q in (path, *path.parents) if q.is_relative_to(root) and q != root):
        raise ValueError('missing or symlinked cache payload')
    return path


def progress(stage, done, total, started):
    print(json.dumps(dict(stage=stage, completed_bytes=done, total_bytes=total,
        ETA_seconds=(time.monotonic()-started)*(total-done)/done if done else None,
        ETA_scope='current local archive byte pass only')), flush=True)


def create_archive(root, files, output):
    if len({r['path'] for r in files}) != len(files):
        raise ValueError('duplicate inventory')
    total, done, started, last = sum(r['bytes'] for r in files), 0, time.monotonic(), 0
    with tarfile.open(output, 'x', format=tarfile.PAX_FORMAT) as archive:
        for record in files:
            path = checked_path(root, record['path'])
            # Hash the exact file handle consumed by tar; post-read size and hash
            # reject mutation while packaging without rereading the source file.
            with path.open('rb') as raw:
                h = hashlib.sha256()

                class Reader:
                    def read(self, size=-1):
                        block = raw.read(size)
                        h.update(block)
                        return block

                info = tarfile.TarInfo('cache/'+record['path'])
                info.size, info.mode, info.mtime = record['bytes'], 0o644, 0
                archive.addfile(info, Reader())
                if raw.read(1) or h.hexdigest() != record['sha256']:
                    raise ValueError('cache payload changed or truncated')
            done += record['bytes']
            if time.monotonic()-last > 20 or done == total:
                progress('cache_transport_write', done, total, started)
                last = time.monotonic()


def read_archive(archive_path, files):
    expected = {'cache/'+r['path']:r for r in files}
    if len(expected) != len(files):
        raise ValueError('duplicate inventory')
    seen, done, total, started, last = set(), 0, sum(r['bytes'] for r in files), time.monotonic(), 0
    with tarfile.open(archive_path, 'r|*') as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError('unexpected, linked, or duplicate archive member')
            record = expected[member.name]
            if member.size != record['bytes']:
                raise ValueError('archive member size differs')
            seen.add(member.name)
            h, count = hashlib.sha256(), 0
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(8*1024**2), b''):
                    h.update(block)
                    count += len(block)
            if count != record['bytes'] or h.hexdigest() != record['sha256']:
                raise ValueError('archive member bytes differ')
            done += count
            if time.monotonic()-last > 20 or done == total:
                progress('cache_transport_independent_readback', done, total, started)
                last = time.monotonic()
    if seen != set(expected):
        raise ValueError('incomplete archive')
    return dict(members=len(seen), payload_bytes=done, all_member_bytes_independently_read=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(1337,2027,3407))
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    freeze = json.loads((source/'source-freeze.json').read_bytes())
    assert freeze['kind'] == NAME
    for name, digest in freeze['sources'].items():
        assert sha(source/name) == digest
    base = R/'artifacts'/BASE/f'seed{args.seed}'
    binding_path = base/'input-binding.json'
    binding = json.loads(binding_path.read_bytes())
    ledger = json.loads((R/'receipts/20260928-execution-ledger.json').read_bytes())
    assert any(e.get('receipt') == str(binding_path) and e.get('receipt_sha256') == sha(binding_path)
               for e in ledger['entries'])
    assert binding['full_original_schedule_to_forest_event_binding_passed'] is True
    assert binding['seed'] == args.seed and binding['events'] == 3316 and binding['sequences'] == 21
    assert sha(base/'cache-inventory.json') == binding['inventory_sha256']
    assert sha(base/'events.json') == binding['events_sha256']
    inventory = json.loads((base/'cache-inventory.json').read_bytes())
    root = Path(inventory['cache_root'])
    assert root == R/f'artifacts/rbf-nested-seen-val-matching-V2-full-admission-v1-20261004/seed{args.seed}/cache'
    assert sha(root/'manifest.json') == inventory['cache_manifest_sha256']
    manifest = json.loads((root/'manifest.json').read_bytes())
    assert manifest['split'] == 'val' and manifest['gt_in_cache'] is False
    records = [dict(role=role, **frame[role]) for frame in manifest['frames'] for role in ('arrays','metadata')]
    assert inventory['files'] == records and len(records) == 2*7189
    files = [dict(path='manifest.json', bytes=(root/'manifest.json').stat().st_size,
                  sha256=inventory['cache_manifest_sha256'])] + records
    destination = R/'artifacts'/NAME/f'seed{args.seed}'
    destination.mkdir(parents=True, exist_ok=False)
    try:
        archive = destination/'cache.tar'
        create_archive(root, files, archive)
        readback = read_archive(archive, files)
        receipt = dict(kind=NAME, seed=args.seed, input_binding_sha256=sha(binding_path),
            cache_manifest_sha256=inventory['cache_manifest_sha256'], events_sha256=sha(base/'events.json'),
            archive=dict(path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive)),
            **readback, source_sha256=sha(__file__), source_freeze_sha256=sha(source/'source-freeze.json'),
            compression='none; existing NPZ payloads preserved byte-for-byte',
            cache_rebuilt=False, NN_forward_repeated=False, uploaded=False, GPU_task_created=False,
            full_forest_independently_accepted=False, paper_performance_complete=False)
        new(destination/'local-transport-readback.json', receipt)
        register(destination/'local-transport-readback.json', NAME)
        print(json.dumps(receipt), flush=True)
    except BaseException as error:
        failure = destination/'failure.json'
        new(failure, dict(seed=args.seed, exception_type=type(error).__name__, message=str(error),
                         partials_preserved=True, no_automatic_retry=True, accepted=False))
        register(failure, NAME+'_failure')
        raise


if __name__ == '__main__':
    main()
