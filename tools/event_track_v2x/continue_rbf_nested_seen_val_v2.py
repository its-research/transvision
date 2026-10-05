"""Bounded once-only handoff after complete raw-side independent acceptance."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from rbf_nested_seen_val_v2_common import OUT, R, RAW, SIDES, admitted_seed, new, register, sha, source_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch-seconds', type=int, default=3600)
    args = parser.parse_args()
    source, _ = source_gate()
    started, handled = time.monotonic(), set()
    while True:
        for seed in (1337, 2027, 3407):
            if seed in handled:
                continue
            launch = R / f'receipts/rbf-nested-seen-val-seed{seed}-matching-V2-full-admission-launch-20261004.json'
            if launch.exists() or (OUT / f'seed{seed}').exists():
                handled.add(seed)
                continue
            side_paths = [RAW / f'seed{seed}' / side for side in SIDES]
            if any((path / 'execution-failure.json').exists() for path in side_paths):
                print(json.dumps(dict(seed=seed, raw_reader_failure_preserved=True, new_cache_not_launched=True)), flush=True)
                handled.add(seed)
                continue
            if not all((path / 'independent-acceptance.json').exists() for path in side_paths):
                continue
            gate = admitted_seed(seed)
            logs = OUT / 'launch-logs'
            logs.mkdir(parents=True, exist_ok=True)
            log = logs / f'seed{seed}.log'
            command = [sys.executable, '-B', str(source / 'seal_rbf_nested_seen_val_v2.py'), '--seed', str(seed)]
            # Exclusive launch receipt creation prevents concurrent listeners from
            # launching a second CPU cache writer for the same scientific identity.
            with launch.open('x') as receipt_stream:
                with log.open('x') as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True, env=dict(os.environ, OMP_NUM_THREADS='1',
                            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
                value = dict(kind='rbf_nested_seen_val_complete_raw_gate_matching_V2_once_launch_v1',
                    seed=seed, pid=process.pid, command=command, log=str(log), inputs=gate,
                    launcher_sha256=sha(__file__), seal_source_sha256=sha(source / 'seal_rbf_nested_seen_val_v2.py'),
                    checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    ETA='unknown until task-level full-frame progress', GPU_task_created=False,
                    independent_V2_acceptance=False, paper_performance_complete=False)
                json.dump(value, receipt_stream, indent=2, allow_nan=False)
                receipt_stream.write('\n')
            register(launch, value['kind'])
            handled.add(seed)
            print(json.dumps(dict(seed=seed, matching_V2_CPU_pid=process.pid)), flush=True)
        if len(handled) == 3 or time.monotonic() - started >= args.watch_seconds:
            print(json.dumps(dict(handled_seeds=sorted(handled), bounded_listener_finished=True,
                                 remaining_seed_inputs_pending=sorted(set((1337, 2027, 3407)) - handled))), flush=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
