#!/usr/bin/env python3
"""Acquire an exact public author feature archive, preserving partial failures."""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import requests


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    assert sha(__file__) == plan['producer_sha256']
    assert sha(plan['author_folder_html']['path']) == plan['author_folder_html']['sha256']
    assert sha(plan['author_data_document']['path']) == plan['author_data_document']['sha256']
    assert sha(plan['HEAD_receipt']['path']) == plan['HEAD_receipt']['sha256']
    spec = next(r for r in plan['files'] if r['name'] == args.name)
    html = Path(plan['author_folder_html']['path']).read_text()
    assert spec['file_id'] in html and spec['name'] in html
    output = Path(plan['output']) / spec['name']
    part = output.with_name(output.name + '.partial')
    receipt = output.with_name(output.name + '.byte-receipt.json')
    assert not any(p.exists() or p.is_symlink() for p in [output, part, receipt])
    assert shutil.disk_usage(output.parent).free > 2 * sum(r['bytes'] for r in plan['files'])
    h = hashlib.sha256()
    count = 0
    start = last = time.monotonic()
    with requests.get(spec['url'], stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        assert 'text/' not in response.headers.get('Content-Type', '')
        assert int(response.headers['Content-Length']) == spec['bytes']
        assert spec['name'] in response.headers.get('Content-Disposition', '')
        with part.open('xb') as stream:
            for block in response.iter_content(8 * 1024**2):
                if not block:
                    continue
                if count == 0:
                    assert block.startswith(b'PK\x03\x04'), 'Expected author ZIP file header'
                stream.write(block)
                h.update(block)
                count += len(block)
                assert count <= spec['bytes']
                now = time.monotonic()
                if now - last >= 30:
                    print(json.dumps(dict(name=spec['name'], actual_bytes=count,
                                          total_bytes=spec['bytes'], elapsed_seconds=now-start)), flush=True)
                    last = now
            stream.flush()
            os.fsync(stream.fileno())
    assert count == spec['bytes']
    os.rename(part, output)
    assert sha(output) == h.hexdigest()
    value = dict(kind='rbf_DMSTrack_public_author_native_feature_archive_complete_byte_acquisition',
                 name=spec['name'], path=str(output), bytes=count, sha256=h.hexdigest(),
                 plan_sha256=sha(args.plan), producer_sha256=sha(__file__),
                 public_author_file_id=spec['file_id'], source=spec,
                 author_published_checksum_available=False,
                 saved_file_independently_reread=True, native_feature_members_qualified=False,
                 native_predictions_produced=False, native_method_accepted=False,
                 checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    receipt.open('x').write(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(name=spec['name'], bytes=count, sha256=h.hexdigest(),
                         author_feature_archive_byte_acquisition_complete=True)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__,
                              partial_files_preserved=True, automatic_retry=False)), flush=True)
        raise
