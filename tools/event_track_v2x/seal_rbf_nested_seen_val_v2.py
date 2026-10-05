"""Build and independently admit each complete matching seen-val cache once."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from rbf_nested_seen_val_v2_common import OUT, admitted_seed, new, register, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(1337, 2027, 3407), required=True)
    args = parser.parse_args()
    gate = admitted_seed(args.seed)
    destination = OUT / f'seed{args.seed}'
    destination.mkdir(parents=True, exist_ok=False)
    new(destination / 'input-gate.json', gate)
    register(destination / 'input-gate.json', 'rbf_nested_seen_val_V2_complete_raw_and_calibration_input_gate_v1')
    stage = 'frozen_four_shard_builder'
    try:
        source = Path(gate['source_root'])
        command = [sys.executable, '-I', '-B', str(source / 'code/tools/event_track_v2x/build_detection_cache_v2.py'),
            '--calibration', gate['calibration'], '--calibration-sha256', gate['calibration_sha256'],
            '--inputs', gate['inputs'], '--output', str(destination / 'cache'),
            '--receipt', str(destination / 'builder-component-readback.json')]
        for raw in gate['raw_roots']:
            command.extend(['--raw-root', raw])
        new(destination / 'commands.json', dict(builder=command,
            independent=[sys.executable, '-B', str(source / 'accept_rbf_nested_seen_val_v2.py'), '--seed', str(args.seed)],
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            producer_component_pass_is_not_independent_acceptance=True, full_online_RBF_accepted=False))
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        subprocess.run(command, check=True, env=env)
        stage = 'independent_full_raw_to_V2_numeric_and_schedule_admission'
        subprocess.run([sys.executable, '-B', str(source / 'accept_rbf_nested_seen_val_v2.py'),
                        '--seed', str(args.seed)], check=True, env=env)
    except BaseException as error:
        path = destination / 'execution-failure.json'
        new(path, dict(seed=args.seed, stage=stage, exception_type=type(error).__name__,
            command_receipt_sha256=sha(destination / 'commands.json') if (destination / 'commands.json').exists() else None,
            partials_preserved=True, no_automatic_retry=True, independent_acceptance=False))
        register(path, 'rbf_nested_seen_val_V2_attempt_failure_preserved_v1')
        raise


if __name__ == '__main__':
    main()
