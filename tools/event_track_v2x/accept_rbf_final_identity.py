"""Independent readback and unchanged float64 reference for new RBF refits.

The reference implementation is the byte-frozen, previously admitted NumPy
checker. Its weights decoder and numerical equations are imported unchanged.
These outputs are candidates on the paired train schedule, not online results.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

R = Path('/Volumes/Data/test/recover-before-fuse')
JOURNAL = R / 'receipts/rbf-new-final-refit-original-joint-all-row-GPU-forward-dispatch-20261004.json'
REFERENCE = R / 'source-freezes/rbf-joint-identity-full-independent-numpy-v1-20261001/verify.py'
REFERENCE_SHA = 'cb86ff0ef14d78f75bc15de78683e7d2888310ae995e6447f70cbf41f15d0cf4'
ROOT = R / 'artifacts/rbf-final-refit-all-row-independent-numeric-v1-20261004'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
                checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), goal_status='active'))
            tmp = ledger.with_suffix(ledger.suffix + '.tmp')
            tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
            os.replace(tmp, ledger)


def inputs(seed):
    job = next(x for x in json.loads(JOURNAL.read_bytes())['jobs'] if x['seed'] == seed)
    rows_root = R / f'artifacts/rbf-joint-identity-full-independent-numpy-v1-20261001/seed{seed}'
    train_root = R / f'artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004/seed{seed}'
    train = json.loads((train_root / 'acceptance.json').read_bytes())
    assert train['task_id'] == job['parent_refit_task_id']
    assert train['artifacts']['checkpoint'] == job['plan']['checkpoint']
    assert train['artifacts']['identity-training'] == job['plan']['weights_archive']
    for key, spec in train['artifacts'].items():
        assert sha(train_root / key) == spec['sha256']
    checkpoint = json.loads((train_root / 'checkpoint').read_bytes())
    manifest_path = rows_root / 'manifest'
    manifest = json.loads(manifest_path.read_bytes())
    assert sha(manifest_path) == job['plan']['manifest']['sha256'] == checkpoint['dataset_sha256']
    assert manifest['row_protocol'] == checkpoint['row_protocol']
    assert len(manifest['shards']) == len(manifest['sequences']) == 46
    assert checkpoint['partition']['fit'] == manifest['sequences'] and checkpoint['partition']['holdout'] == []
    paths = {}
    for rec in manifest['shards']:
        hits = list((rows_root / 'rows-unpack').rglob(rec['path']))
        assert len(hits) == 1 and sha(hits[0]) == rec['sha256']
        assert rec['sequence_id'] not in paths
        paths[rec['sequence_id']] = hits[0]
    wp = train_root / f'archive-unpack/training/seed-{seed}/weights.pt'
    assert sha(wp) == checkpoint['weights']['sha256'] == train['weights_sha256']
    return job, manifest, paths, wp, train_root / 'acceptance.json'


def readback(seed):
    import requests
    from clearml import Task
    from clearml.backend_api.session import Session
    job, manifest, _, wp, train_proof = inputs(seed)
    task = Task.get_task(task_id=job['task_id'])
    task.reload()
    assert str(task.status) == 'completed'
    expected = {'receipt', *(f'predictions-rank{i}' for i in range(4))}
    assert set(task.artifacts) == expected
    directory = ROOT / f'seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    proof = directory / 'independent-byte-coverage-receipt.json'
    if proof.exists():
        old = json.loads(proof.read_bytes())
        assert old['task_id'] == task.id and old['plan_sha256'] == hashlib.sha256(
            json.dumps(job['plan'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        for key, rec in old['artifacts'].items():
            artifact = task.artifacts[key]
            path = directory / (key + ('.json' if key == 'receipt' else '.jsonl'))
            assert sha(path) == artifact.hash == rec['sha256']
            assert path.stat().st_size == artifact.size == rec['bytes']
        print(json.dumps(dict(seed=seed, byte_coverage='prior_accepted_preserved')), flush=True)
        return
    inventory = {}
    for key in sorted(expected):
        artifact = task.artifacts[key]
        path = directory / (key + ('.json' if key == 'receipt' else '.jsonl'))
        if not path.exists():
            partial = directory / (path.name + '.partial-' + str(time.time_ns()))
            url = artifact.url.replace('10.100.35.118:8081', '10.100.34.118:8081')
            with requests.get(url, headers={'Authorization': 'Bearer ' + Session().token},
                              timeout=(10, 90), stream=True) as response:
                if response.status_code != 200:
                    raise RuntimeError('artifact HTTP status ' + str(response.status_code))
                with partial.open('xb') as f:
                    for chunk in response.iter_content(1024**2):
                        f.write(chunk)
            assert partial.stat().st_size == artifact.size and sha(partial) == artifact.hash
            partial.rename(path)
        assert path.stat().st_size == artifact.size and sha(path) == artifact.hash
        inventory[key] = dict(task=task.id, key=key, bytes=artifact.size, sha256=artifact.hash)
    producer = json.loads((directory / 'receipt.json').read_bytes())
    assert producer['completed_all_rows'] and producer['plan'] == job['plan'] and producer['seed'] == seed
    records = {x['sequence_id']: x for x in manifest['shards']}
    counts, identities, total = {}, set(), 0
    for rank in range(4):
        runtime = producer['ranks'][rank]
        assert runtime['tf32_matmul'] is runtime['tf32_cudnn'] is False
        expected_seq = {x['sequence_id']: x for x in runtime['sequences']}
        got = {}
        for line in (directory / f'predictions-rank{rank}.jsonl').open():
            row = json.loads(line)
            seq = row['sequence_id']
            assert seq in expected_seq and seq in records and row['row'] == got.get(seq, 0)
            got[seq] = got.get(seq, 0) + 1
            identity = seq, row['node_id']
            assert identity not in identities
            identities.add(identity)
            context = row['context_indices']
            assert context[-1] == row['row'] and context == sorted(set(context))
            assert 1 <= len(context) <= 9 and len(context) == len(row['logits'])
            assert all(isinstance(x, (float, int)) and math.isfinite(x) for x in row['logits'])
            total += 1
        assert got == {seq: rec['rows'] for seq, rec in expected_seq.items()}
        assert not set(counts) & set(got)
        counts.update(got)
    assert counts == {seq: rec['nodes'] for seq, rec in records.items()}
    assert total == sum(x['rows'] for x in producer['ranks'])
    result = dict(kind='rbf_new_final_refit_all_row_independent_byte_coverage_admission_v1',
        seed=seed, task_id=task.id, checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        artifacts=inventory, rows=total, sequence_counts=counts, weights_sha256=sha(wp),
        plan_sha256=hashlib.sha256(json.dumps(job['plan'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        train_byte_admission=str(train_proof), train_byte_admission_sha256=sha(train_proof),
        all_registered_output_bytes_verified=True, all_46_train_sequences_row_coverage_verified=True,
        runtime_devices=[x['device'] for x in producer['ranks']], source_sha256=sha(__file__),
        full_independent_numerical_acceptance=False, whole_forest_replay_complete=False,
        paper_performance_complete=False)
    new(proof, result)
    register(proof, result['kind'])
    print(json.dumps(dict(seed=seed, task_id=task.id, byte_coverage_pass=True, rows=total)), flush=True)


def numeric(seed):
    assert sha(REFERENCE) == REFERENCE_SHA
    spec = importlib.util.spec_from_file_location('frozen_independent_joint_reference', REFERENCE)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    np = ref.np
    job, manifest, paths, wp, train_proof = inputs(seed)
    directory = ROOT / f'seed{seed}'
    proof_path = directory / 'independent-byte-coverage-receipt.json'
    proof = json.loads(proof_path.read_bytes())
    assert proof['task_id'] == job['task_id'] and proof['weights_sha256'] == sha(wp)
    assert proof['train_byte_admission_sha256'] == sha(train_proof)
    w = ref.weights(wp)
    records = {x['sequence_id']: x for x in manifest['shards']}
    out = directory / 'numeric-v1'
    out.mkdir(exist_ok=False)
    plan = dict(kind='rbf_new_final_refit_full_independent_float64_joint_reference_v1', seed=seed,
        source_sha256=sha(__file__), unchanged_reference=str(REFERENCE), unchanged_reference_sha256=REFERENCE_SHA,
        numpy=np.__version__, weights_sha256=sha(wp), prediction_proof_sha256=sha(proof_path),
        atol=1e-4, rtol=1e-4, all_rows_selected_without_GT=True,
        no_Torch_or_production_NN_assignment_motion_imports=True)
    new(out / 'plan.json', plan)
    started = time.monotonic()
    done, maximum, failed = 0, 0., 0
    results = []
    try:
        with (out / 'failures.jsonl').open('x') as failures:
            for rank in range(4):
                path = directory / f'predictions-rank{rank}.jsonl'
                assert sha(path) == proof['artifacts'][f'predictions-rank{rank}']['sha256']
                current, arrays, buffer = None, None, []
                seqcount, seqmax, seqfailed = 0, 0., 0

                def flush():
                    nonlocal buffer, done, maximum, failed, seqcount, seqmax, seqfailed
                    if not buffer:
                        return
                    rows = np.asarray([r['row'] for r in buffer], dtype=np.int64)
                    vectors = ref.reference(w, arrays, rows)
                    for row, vector in zip(buffer, vectors):
                        n = int(arrays['lengths'][row['row']])
                        assert row['context_indices'] == arrays['contexts'][row['row'], :n].tolist()
                        assert row['node_id'] == str(arrays['node_id'][row['row']])
                        assert row['decision_us'] == int(arrays['decision_us'][row['row']])
                        got = np.asarray(row['logits'])
                        assert len(got) == len(vector)
                        error = float(np.max(np.abs(got - vector)))
                        maximum, seqmax = max(maximum, error), max(seqmax, error)
                        if not np.allclose(got, vector, atol=1e-4, rtol=1e-4):
                            failed += 1
                            seqfailed += 1
                            failures.write(json.dumps(dict(sequence_id=current, row=row['row'],
                                max_absolute_error=error, GPU=got.tolist(), independent_float64=vector.tolist())) + '\n')
                    done += len(buffer)
                    seqcount += len(buffer)
                    buffer = []

                def completed_sequence():
                    assert seqcount == records[current]['nodes']
                    results.append(dict(sequence_id=current, rows=seqcount,
                                        max_absolute_error=seqmax, failures=seqfailed))
                    print(json.dumps(dict(seed=seed, completed_rows=done, total_rows=proof['rows'],
                        numeric_failures=failed, ETA_seconds=(time.monotonic() - started) *
                        (proof['rows'] - done) / done)), flush=True)

                for line in path.open():
                    row = json.loads(line)
                    if row['sequence_id'] != current:
                        flush()
                        if current is not None:
                            completed_sequence()
                        current = row['sequence_id']
                        with np.load(paths[current], allow_pickle=False) as z:
                            arrays = {k: z[k] for k in ['features', 'source', 'information_us', 'arrival_us',
                                'mean', 'covariance', 'state_us', 'lengths', 'contexts', 'decision_us', 'node_id']}
                        seqcount, seqmax, seqfailed = 0, 0., 0
                    assert row['row'] == seqcount + len(buffer)
                    buffer.append(row)
                    if len(buffer) == 64:
                        flush()
                flush()
                completed_sequence()
        assert len(results) == len({x['sequence_id'] for x in results}) == 46
        assert done == proof['rows']
        result = dict(plan, full_independent_numeric_pass=failed == 0, rows=done, sequences=results,
            numeric_failed_rows=failed, max_absolute_error=maximum, seconds=time.monotonic() - started,
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            whole_forest_replay_complete=False, paper_performance_complete=False)
        new(out / 'completion.json', result)
        register(out / 'completion.json', result['kind'])
        print(json.dumps(dict(seed=seed, full_independent_numeric_pass=failed == 0,
                             rows=done, max_absolute_error=maximum)), flush=True)
    except BaseException as error:
        new(out / 'execution-failure.json', dict(exception_type=type(error).__name__, message=str(error),
            completed_rows=done, numeric_failed_rows=failed, source_sha256=sha(__file__)))
        register(out / 'execution-failure.json', 'rbf_final_refit_independent_numeric_execution_failure_preserved')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=[1337, 2027, 3407])
    parser.add_argument('--stage', choices=['readback', 'numeric'], required=True)
    args = parser.parse_args()
    (readback if args.stage == 'readback' else numeric)(args.seed)
