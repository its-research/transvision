#!/usr/bin/env python3
"""Read-only continuation from one actual public download into feature admission."""
import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

from rbf_nested_seen_val_v2_common import sha, register, new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    assert sha(__file__) == plan['continuation_sha256']
    for ref in plan['references'].values():
        assert sha(ref['path']) == ref['sha256']
    directory = Path(plan['output'])
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        prerequisite = Path(plan['byte_receipt'])
        while not prerequisite.exists():
            assert time.monotonic() - started < plan['prerequisite_wait_seconds']
            os.kill(plan['download_pid'], 0)
            command = subprocess.check_output(['ps', '-p', str(plan['download_pid']), '-o', 'args='], text=True)
            assert plan['references']['downloader']['path'] in command
            assert f"--name {plan['name']}" in command
            progress = dict(stage='specific_author_download_in_progress',
                            actual_download_pid=plan['download_pid'],
                            download_command_verified=True, native_features_accepted=False,
                            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            temporary = directory / 'progress.new.json'
            temporary.write_text(json.dumps(progress, indent=2) + '\n')
            os.replace(temporary, directory / 'progress.json')
            time.sleep(30)
        proof = json.loads(prerequisite.read_bytes())
        assert proof['name'] == plan['name'] and proof['saved_file_independently_reread']
        assert sha(plan['references']['auditor']['path']) == plan['references']['auditor']['sha256']
        command = [plan['python'], plan['references']['auditor']['path'], '--plan',
                   plan['references']['audit_plan']['path'], '--name', plan['name']]
        with (directory / 'feature-audit.log').open('xb') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        assert result.returncode == 0
        accepted = Path(plan['expected_audit_receipt'])
        value = json.loads(accepted.read_bytes())
        assert value['all_members_independently_CRC_size_SHA_verified']
        assert value['all_native_feature_shapes_counts_and_values_verified']
        assert value['byte_receipt_sha256'] == sha(prerequisite)
        register(prerequisite, proof['kind'])
        register(accepted, value['kind'])
        new(directory / 'process-exit.json', dict(returncode=0, input_receipt=str(accepted),
            input_receipt_sha256=sha(accepted), native_predictions_produced=False,
            native_method_accepted=False, source_modified=False, cache_or_NN_rebuilt=False))
        print(json.dumps(dict(name=plan['name'], native_feature_input_audit_complete=True,
                              native_method_accepted=False)), flush=True)
    except BaseException as error:
        new(directory / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
            accepted=False, partial_bytes_and_failed_audit_preserved=True, automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
