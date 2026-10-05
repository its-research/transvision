"""Stage already registered rank archives; never admit an unfinished experiment.

The unchanged completed-task reader rechecks these exact bytes later. Do not run
that reader concurrently with this process. This helper creates no ClearML task,
uploads nothing, unpacks nothing, and never writes a forest acceptance receipt.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


R = Path('/Volumes/Data/test/recover-before-fuse')
READER = R / 'source-freezes/rbf-final-refit-main-topK-independent-output-reader-v1-20261004'
PINNED = {
    'read_rbf_final_refit_forest_outputs.py': '53747489a446c8990cfc6d9310eb686a69b39d99bde226c8945665a974d83f61',
    'rbf_nested_seen_val_v2_common.py': '58e616f116155b9bd2a9f0ed52b6b0ebc47895c6e9957271a6d88db51b6502ec',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def rank_inventory(task, world):
    allowed = {'receipt', 'exclusive-source-manifest'} | {f'replay-rank{i}' for i in range(world)}
    assert set(task.artifacts) <= allowed
    result = {}
    for rank in range(world):
        key = f'replay-rank{rank}'
        if key not in task.artifacts:
            continue
        artifact = task.artifacts[key]
        assert isinstance(artifact.size, int) and artifact.size > 0
        assert len(artifact.hash) == 64 and all(c in '0123456789abcdef' for c in artifact.hash)
        result[key] = {'bytes': artifact.size, 'sha256': artifact.hash}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(1337, 2027, 3407))
    parser.add_argument('--parallel-downloads', type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    for name, digest in PINNED.items():
        assert sha(READER / name) == digest
    sys.path.insert(0, str(READER))
    spec = importlib.util.spec_from_file_location('frozen_forest_reader', READER / 'read_rbf_final_refit_forest_outputs.py')
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    from clearml import Task

    journal = R / 'receipts/rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json'
    job = next(v for v in json.loads(journal.read_bytes())['jobs'] if v['seed'] == args.seed)
    plan = job['plan']
    assert plan['method'] == 'rbf' and plan['world_size'] in (4, 8)
    assert hashlib.sha256(reader.canonical(plan)).hexdigest() == job['recipe_sha256']
    task = Task.get_task(task_id=job['task_id'])
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == plan['bootstrap_sha256']
    snapshot = rank_inventory(task, plan['world_size'])
    assert snapshot, 'No registered rank artifact is available to stage'
    cache = R / f'artifacts/rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/rbf/seed{args.seed}'
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'rank-byte-prefetch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (cache / 'independent-byte-coverage-factor-admission.json').exists()
        assert not (cache / 'independent-failure-byte-readback.json').exists()
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        prefix = R / f'receipts/rbf-final-refit-rank-byte-prefetch-seed{args.seed}-{stamp}'
        started = prefix.with_name(prefix.name + '-started.json')
        binding = dict(
            kind='rbf_final_refit_registered_rank_byte_prefetch_v1',
            seed=args.seed, task_id=task.id, task_status_at_start=str(task.status),
            recipe_sha256=job['recipe_sha256'], dispatch_journal=str(journal),
            dispatch_journal_sha256=sha(journal), command=sys.argv, pid=os.getpid(),
            source_path=str(Path(__file__).resolve()), source_sha256=sha(__file__),
            frozen_reader_hashes=PINNED, registered_snapshot=snapshot,
            cache_directory=str(cache), parallel_downloads=args.parallel_downloads,
            completed_task_reader_must_wait_for_this_process=True,
            experiment_accepted=False, forest_semantics_accepted=False,
            paper_performance_complete=False, whole_experiment_ETA='unknown',
        )
        reader.new(started, binding)
        reader.register(started, binding['kind'] + '_started')
        print(json.dumps(dict(stage='registered_rank_byte_prefetch_started', receipt=str(started),
                             pid=os.getpid(), artifacts=len(snapshot), bytes=sum(v['bytes'] for v in snapshot.values()),
                             experiment_accepted=False)), flush=True)
        results = {}
        failures = {}
        with ThreadPoolExecutor(max_workers=args.parallel_downloads) as pool:
            futures = {pool.submit(reader.read_artifact, task, key, cache / (key + '.tar.gz')): key for key in snapshot}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result()
                    assert result == snapshot[key]
                    results[key] = result
                    print(json.dumps(dict(stage='registered_rank_byte_prefetch_hash_matched', artifact=key,
                                          **result, experiment_accepted=False)), flush=True)
                except Exception as error:
                    # No arbitrary SDK/HTTP messages, paths with tokens, or credentials.
                    failures[key] = type(error).__name__
                    print(json.dumps(dict(stage='registered_rank_byte_prefetch_failed', artifact=key,
                                          error_type=type(error).__name__, partial_preserved=True)), flush=True)
        remote_unchanged = False
        end_status = 'unobserved'
        try:
            current = Task.get_task(task_id=task.id)
            end_status = str(current.status)
            after = rank_inventory(current, plan['world_size'])
            assert all(after.get(key) == value for key, value in snapshot.items())
            assert hashlib.sha256(current.data.script.diff.encode()).hexdigest() == plan['bootstrap_sha256']
            remote_unchanged = True
        except Exception as error:
            failures['remote_metadata_recheck'] = type(error).__name__
        complete = prefix.with_name(prefix.name + '-finished.json')
        proof = dict(binding, task_status_at_end=end_status, independently_read_artifacts=results,
                     failures=failures, registered_snapshot_unchanged_at_end=remote_unchanged,
                     all_snapshot_bytes_match=not failures and results == snapshot,
                     extraction_started=False, task_mutated=False, uploads_performed=False)
        reader.new(complete, proof)
        reader.register(complete, binding['kind'] + '_finished')
        print(json.dumps(dict(stage='registered_rank_byte_prefetch_finished', receipt=str(complete),
                              all_snapshot_bytes_match=proof['all_snapshot_bytes_match'],
                              experiment_accepted=False)), flush=True)
        return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
