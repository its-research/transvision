#!/usr/bin/env python3
"""Take one bounded macOS upload-process transport sample with approximate ETA."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--expected-bytes', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    command = subprocess.check_output(['ps', '-p', str(args.pid), '-o', 'command='], text=True).strip()
    if 'upload_spd_official_oof_fold_package.py' not in command:
        raise RuntimeError('PID is not the expected package uploader')
    output = subprocess.check_output(['nettop', '-P', '-L', '1', '-p', str(args.pid),
                                      '-J', 'bytes_in,bytes_out'], text=True, timeout=10)
    lines = [line.split(',') for line in output.splitlines() if line.strip()]
    matches = [row for row in lines if row[0].endswith('.' + str(args.pid))]
    if len(matches) != 1:
        raise RuntimeError('nettop did not return one uploader sample')
    sent = int(matches[0][2])
    now = datetime.now(timezone.utc)
    args.output.mkdir(parents=True, exist_ok=True)
    previous = []
    for path in args.output.glob('sample-*.json'):
        row = json.loads(path.read_bytes())
        if row['pid'] == args.pid and row['task_id'] == args.task_id:
            previous.append(row)
    rate = None
    eta = None
    if previous:
        prior = max(previous, key=lambda row: row['checked_at_utc'])
        elapsed = (now - datetime.fromisoformat(prior['checked_at_utc'])).total_seconds()
        delta = sent - prior['process_network_bytes_sent']
        if elapsed > 0 and delta > 0:
            rate = delta / elapsed
            eta = max(args.expected_bytes - sent, 0) / rate
    result = {'kind': 'clearml_package_upload_transport_sample_v1', 'pid': args.pid,
              'task_id': args.task_id, 'expected_archive_bytes': args.expected_bytes,
              'process_network_bytes_sent': sent, 'estimated_bytes_per_second': rate,
              'estimated_transport_eta_seconds': eta, 'eta': 'unknown' if eta is None else 'approximate',
              'scope': 'process network counters; not artifact receipt or accepted experiment',
              'checked_at_utc': now.isoformat()}
    path = args.output / ('sample-' + now.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    with path.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
