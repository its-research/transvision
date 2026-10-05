#!/usr/bin/env python3
"""Read current-phase ETA from timestamped training logs, never arrival time.

No task mutations. An estimate is not a completion or acceptance receipt.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

PATTERN = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}).*?mmdet - INFO - Iter \[(\d+)/(\d+)\]')


def estimate(messages):
    samples = []
    for message in messages:
        for line in message.splitlines():
            match = PATTERN.search(line)
            if not match:
                continue
            stamp, done, total = match.groups()
            point = (datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S,%f'), int(done), int(total))
            if not 0 < point[1] <= point[2]:
                raise ValueError('invalid source progress')
            if samples and point == samples[-1]:
                continue
            # A side/phase transition or resumed counter invalidates prior rate.
            if samples and (point[2] != samples[-1][2] or point[1] <= samples[-1][1]
                            or point[0] <= samples[-1][0]):
                samples = []
            samples.append(point)
    result = {'eta_seconds': None, 'eta_status': 'unknown',
              'method': 'training_source_clock_cumulative_and_recent_window',
              'scope': 'observed current phase only; excludes publication and other side',
              'overall_eta': 'unknown', 'experiment_complete': False,
              'source_clock_timezone': 'unspecified in mmdet log'}
    if samples:
        last = samples[-1]
        result.update(iteration=last[1], total_iterations=last[2],
                      latest_source_timestamp=last[0].isoformat())
    if len(samples) >= 2:
        last = samples[-1]
        rates = []
        for first in (samples[0], samples[max(0, len(samples)-5)]):
            elapsed = (last[0]-first[0]).total_seconds()
            if elapsed > 0 and last[1] > first[1]:
                rates.append((last[1]-first[1])/elapsed)
        if rates:
            result.update(eta_seconds=(last[2]-last[1])/min(rates),
                          eta_status='estimated', source_samples=len(samples))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from clearml import Task
    task = Task.get_task(task_id=a.task_id)
    messages = task.get_reported_console_output(number_of_reports=30)
    row = dict(estimate(messages), task_id=task.id, status=str(task.status),
               checked_at_utc=datetime.now(timezone.utc).isoformat())
    if str(task.status) != 'in_progress':
        row.update(eta_seconds=None, eta_status='unknown')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(row, allow_nan=False))


if __name__ == '__main__':
    main()
