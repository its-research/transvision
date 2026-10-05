#!/usr/bin/env python3
"""Independently hash streamed ClearML bytes of one canonical SPD OOF package."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlparse

from run_cooptrack_official_oof_gpu4 import validate_cohort


def readback(artifact, expected_bytes, expected_sha):
    from clearml.storage.helper import StorageHelper
    url = artifact.url
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.netloc != '10.100.35.118:8081':
        raise ValueError('artifact host differs from the locally registered file service')
    helper = StorageHelper.get(url)
    container = helper._driver._containers['http://' + parsed.netloc]
    offset = 0
    h = hashlib.sha256()
    started = time.monotonic()
    while offset < expected_bytes:
        end = min(offset + 64 * 1024 * 1024, expected_bytes) - 1
        for attempt in range(3):
            response = None
            try:
                headers = dict(container.get_headers(url))
                headers['Range'] = f'bytes={offset}-{end}'
                headers['Accept-Encoding'] = 'identity'
                response = container.session.get(url, headers=headers, timeout=(10, 120), stream=True)
                if (response.headers.get('Content-Encoding', 'identity') not in ('identity', '')
                        or response.status_code != 206
                        or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{expected_bytes}'
                        or int(response.headers.get('Content-Length', '-1')) != end-offset+1):
                    raise ValueError('file server did not return the exact requested range')
                candidate = h.copy()
                received = 0
                for block in response.iter_content(chunk_size=1024*1024):
                    if block:
                        candidate.update(block)
                        received += len(block)
                if received != end-offset+1:
                    raise ValueError('short artifact range')
                h = candidate
                offset += received
                break
            except Exception:
                if attempt == 2:
                    raise
            finally:
                if response is not None:
                    response.close()
        if offset % (256*1024*1024) < 64*1024*1024 or offset == expected_bytes:
            elapsed = max(time.monotonic()-started, 0.001)
            eta = (expected_bytes-offset)/(offset/elapsed)
            print(f'OOF package independent readback {offset}/{expected_bytes} bytes ETA={eta:.1f}s', flush=True)
    if h.hexdigest() != expected_sha:
        raise ValueError('independent artifact byte hash differs')
    return {'bytes': offset, 'sha256': h.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--package', type=Path, required=True)
    args = parser.parse_args()
    out = args.package / 'clearml-independent-readback.json'
    if out.exists():
        raise FileExistsError('independent readback receipt is create-once')
    from clearml import Task
    manifest_path = args.package / 'package-manifest.json'
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    validate_cohort(manifest)
    upload = json.loads((args.package / 'clearml-upload-acceptance.json').read_bytes())
    if (manifest['fold_id'] != args.fold or upload['fold_id'] != args.fold
            or upload['manifest_sha256'] != manifest_sha
            or upload['status'] != 'uploaded_and_hash_verified'):
        raise ValueError('local uploaded package identity differs')
    task = Task.get_task(task_id=upload['task_id'])
    if task.status != 'completed':
        raise RuntimeError('package upload task is not completed')
    archive = next(item for item in manifest['inventory'] if item['path'] == 'train-inputs.tar.gz')
    expected = [('package-manifest', len(raw), manifest_sha),
                ('train-inputs', archive['bytes'], archive['sha256'])]
    results = {}
    for name, size, sha in expected:
        artifact = task.artifacts[name]
        if artifact.hash != sha or artifact.size != size:
            raise ValueError('registered uploaded artifact identity differs')
        results[name] = readback(artifact, size, sha)
    receipt = {'kind': 'spd_official_oof_fold_clearml_independent_readback_v1',
               'status': 'independent_bytes_verified', 'task_id': task.id,
               'fold_id': args.fold, 'manifest_sha256': manifest_sha,
               'artifacts': results, 'detector_training_started': False,
               'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    with out.open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print('OOF_PACKAGE_INDEPENDENTLY_ACCEPTED', hashlib.sha256(out.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
