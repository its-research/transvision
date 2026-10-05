"""Execute one source-bound independent val input audit, retaining failures."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from rbf_nested_seen_val_v2_common import R, new, register, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(1337, 2027, 3407))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    prepared = json.loads((root/'preparation.json').read_bytes())
    for name, record in prepared['sources'].items():
        assert sha(root/name) == record['sha256'] and (root/name).stat().st_size == record['bytes']
    gate_path = root/'software-gate.json'
    gate = json.loads(gate_path.read_bytes())
    assert gate['software_cases'] == 96 and gate['source_sha256'] == sha(root/'accept_rbf_seen_val_prediction_rows.py')
    directory = R/f'artifacts/rbf-matching-seen-val-target-free-original-row-contexts-v1-20261004/seed{args.seed}'
    assert (directory/'manifest.json').exists()
    command = [sys.executable, '-B', str(root/'accept_rbf_seen_val_prediction_rows.py'), '--seed', str(args.seed),
        '--software-gate', str(gate_path)]
    command_path = directory/'independent-audit-command.json'
    new(command_path, dict(seed=args.seed, command=command, preparation_sha256=sha(root/'preparation.json'),
        software_gate_sha256=sha(gate_path), checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    register(command_path, 'rbf_seen_val_target_free_complete_input_independent_audit_command_v1')
    try:
        subprocess.run(command, check=True, env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
        acceptance = directory/'independent-acceptance.json'
        register(acceptance, 'rbf_seen_val_target_free_complete_input_independent_acceptance_v1')
    except BaseException as error:
        failure = directory/'independent-audit-failure.json'
        new(failure, dict(seed=args.seed, exception_type=type(error).__name__, command_receipt_sha256=sha(command_path),
            failure_preserved=True, no_automatic_retry=True, independent_input_admission=False))
        register(failure, 'rbf_seen_val_target_free_complete_input_independent_audit_failure_v1')
        raise


if __name__ == '__main__':
    main()
