"""Read the registered label-free SPD val package for producer admission.

No model or evaluator is run. Archive byte identity is distinct from the
pending dataset schema, producer/runtime and complete inference acceptance.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time

import requests
from clearml import Task
from clearml.backend_api.session import Session

R = Path('/Volumes/Data/test/recover-before-fuse')
TASK_ID = '5717bf5d1ead4fcba37f3d9101b11b20'
KEY = 'cache-inputs'
DIGEST = '7a58ac98f96b7c0ea07429b4951744c34a6e2048bca4ad7bff6aa0952936b9d0'
SIZE = 1796367315
MANIFEST_SHA = 'd7a7688321145ad4bc5ed5692858b2136ac2c6aa8376c200d45b8f908a3f4877'
OUT = R / 'artifacts/rbf-final-refit-SPD-seen-val-label-free-package-byte-recovery-v1-20261004'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def register(path, kind):
    ledger = R / 'receipts/20260928-execution-ledger.json'
    with open(str(ledger) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = json.loads(ledger.read_bytes())
        if not any(x.get('receipt') == str(path) for x in value['entries']):
            value['entries'].append(dict(kind=kind, receipt=str(path), receipt_sha256=sha(path),
                goal_status='active', checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            tmp = ledger.with_suffix(ledger.suffix + '.tmp')
            tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
            os.replace(tmp, ledger)


def main():
    OUT.mkdir(exist_ok=False)
    count = 0
    stage = 'registered_archive_byte_readback'
    try:
        task = Task.get_task(task_id=TASK_ID)
        assert str(task.status) == 'completed'
        artifact = task.artifacts[KEY]
        assert artifact.hash == DIGEST and artifact.size == SIZE
        manifest_path = R / ('artifacts/rbf-final-refit-SPD-seen-val-matching-producer-input-recovery-v1-20261004/'
                             + TASK_ID + '/package-manifest')
        assert sha(manifest_path) == MANIFEST_SHA
        manifest = json.loads(manifest_path.read_bytes())
        assert manifest['data_sha256'] == DIGEST and manifest['data_bytes'] == SIZE
        assert manifest['val_payloads_included'] and not manifest['test_payloads_included']
        assert not manifest['gt_payloads_included'] and manifest['validation_sequences'] == 21
        partial = OUT / 'cache-inputs.tar.gz.partial'
        digest = hashlib.sha256()
        start, last = time.monotonic(), 0
        url = artifact.url.replace('10.100.35.118:8081', '10.100.34.118:8081')
        with requests.get(url, headers={'Authorization': 'Bearer ' + Session().token},
                          timeout=(10, 120), stream=True) as response:
            if response.status_code != 200:
                raise RuntimeError('artifact HTTP status ' + str(response.status_code))
            with partial.open('xb') as f:
                for block in response.iter_content(1024**2):
                    f.write(block)
                    digest.update(block)
                    count += len(block)
                    assert count <= SIZE
                    if time.monotonic() - last > 20:
                        print(json.dumps(dict(stage=stage, completed_bytes=count, total_bytes=SIZE,
                            ETA_seconds=(time.monotonic() - start) * (SIZE - count) / count,
                            ETA_scope='archive_transfer_only', experiment_accepted=False)), flush=True)
                        last = time.monotonic()
        assert count == SIZE and digest.hexdigest() == DIGEST
        archive = OUT / 'cache-inputs.tar.gz'
        partial.rename(archive)
        stage = 'archive_member_inventory_and_frozen_source_readback'
        members, sources, seen = [], {}, set()
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                name = PurePosixPath(member.name)
                assert not name.is_absolute() and '..' not in name.parts and '\\' not in member.name
                assert not member.issym() and not member.islnk() and (member.isfile() or member.isdir())
                assert member.name not in seen
                seen.add(member.name)
                members.append(dict(path=member.name, bytes=member.size, type='file' if member.isfile() else 'directory'))
                if member.isfile() and name.name in manifest['scripts']:
                    raw = tar.extractfile(member).read()
                    assert hashlib.sha256(raw).hexdigest() == manifest['scripts'][name.name]
                    assert name.name not in sources
                    path = OUT / name.name
                    with path.open('xb') as f:
                        f.write(raw)
                    sources[name.name] = dict(path=str(path), bytes=len(raw), sha256=sha(path))
        assert set(sources) == set(manifest['scripts'])
        new(OUT / 'archive-members.json', members)
        result = dict(kind='rbf_SPD_val_registered_label_free_archive_independent_byte_readback_v1',
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            task_id=TASK_ID, artifact_key=KEY, archive=str(archive), bytes=count, sha256=DIGEST,
            producer_manifest_sha256=MANIFEST_SHA, archive_members=len(members),
            member_inventory_sha256=sha(OUT / 'archive-members.json'), frozen_sources=sources,
            full_registered_archive_bytes_read=True, schema_and_all_payload_independent_acceptance=False,
            matching_three_seed_detector_inference_started=False, paper_performance_complete=False,
            source_sha256=sha(__file__), ETA_seconds=0, ETA_scope='archive_readback_only')
        new(OUT / 'byte-readback-receipt.json', result)
        register(OUT / 'byte-readback-receipt.json', result['kind'])
        print(json.dumps(dict(archive_byte_identity_pass=True, receipt=str(OUT / 'byte-readback-receipt.json'),
                             next='independent_schema_and_frozen_source_runtime_admission')), flush=True)
    except BaseException as error:
        path = OUT / 'execution-failure.json'
        new(path, dict(stage=stage, completed_bytes=count, exception_type=type(error).__name__,
                       message=str(error), source_sha256=sha(__file__)))
        register(path, 'rbf_SPD_val_archive_byte_recovery_failure_preserved')
        raise


if __name__ == '__main__':
    main()
