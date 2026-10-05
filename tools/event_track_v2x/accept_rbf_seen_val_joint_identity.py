"""Independent cloud bytes, full target-free val coverage, unchanged float64 NN."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

from rbf_nested_seen_val_v2_common import R, new, register, sha

JOURNAL = R/'receipts/rbf-matching-seen-val-three-seed-joint-identity-GPU-dispatch-20261004.json'
ROWS = R/'artifacts/rbf-matching-seen-val-target-free-original-row-contexts-v1-20261004'
CK = R/'artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004'
DEST = R/'artifacts/rbf-matching-seen-val-joint-identity-full-independent-output-admission-v1-20261004'
REFERENCE = R/'source-freezes/rbf-joint-identity-full-independent-numpy-v1-20261001/verify.py'
REFERENCE_SHA = 'cb86ff0ef14d78f75bc15de78683e7d2888310ae995e6447f70cbf41f15d0cf4'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def inputs(seed):
    job = next(job for job in json.loads(JOURNAL.read_bytes())['jobs'] if job['seed'] == seed)
    directory = ROWS/f'seed{seed}'
    manifest = json.loads((directory/'manifest.json').read_bytes())
    admission = json.loads((directory/'independent-acceptance.json').read_bytes())
    assert sha(directory/'manifest.json') == job['plan']['row-manifest']['sha256'] == admission['row_manifest_sha256']
    assert sha(directory/'independent-acceptance.json') == job['plan']['row-independent-admission']['sha256']
    assert admission['full_original_seen_val_features_and_contexts_independently_accepted'] is True
    assert admission['seed'] == manifest['seed'] == seed and manifest['split'] == 'val'
    assert manifest['original_events'] == 3316 and len(manifest['shards']) == len(manifest['sequences']) == 21
    checkpoint = CK/f'seed{seed}/checkpoint'
    assert sha(checkpoint) == job['plan']['checkpoint']['sha256'] == admission['checkpoint_sha256']
    metadata = json.loads(checkpoint.read_bytes())
    assert manifest['row_protocol'] == metadata['row_protocol'] and manifest['frozen_cache_identity'] == metadata['frozen_cache_identity']
    assert metadata['geometry_weight'] == 1 and metadata['row_protocol']['process_noise'] == .1
    assert metadata['row_protocol']['parent_limit'] == 8
    weights = CK/f'seed{seed}/archive-unpack/training/seed-{seed}/weights.pt'
    assert sha(weights) == metadata['weights']['sha256']
    for record in manifest['shards']:
        path = directory/record['path']
        assert not Path(record['path']).is_absolute() and '..' not in Path(record['path']).parts
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
    return job,directory,manifest,weights


def download(task,key,path):
    import requests
    from clearml.backend_api.session import Session
    artifact = task.artifacts[key]
    assert not path.exists(), 'existing output bytes retained; no implicit reread'
    started, completed = time.monotonic(), 0
    url = artifact.url.replace('10.100.35.118:8081','10.100.34.118:8081')
    with requests.get(url,stream=True,timeout=(10,120),headers={'Authorization':'Bearer '+Session().token}) as response:
        assert response.status_code == 200
        with path.open('xb') as stream:
            for block in response.iter_content(1024**2):
                stream.write(block);completed += len(block)
                if completed % (16*1024**2) < len(block):
                    print(json.dumps(dict(stage='seen_val_output_independent_byte_readback',key=key,
                        completed_bytes=completed,total_bytes=artifact.size,
                        ETA_seconds=(time.monotonic()-started)*(artifact.size-completed)/completed,
                        ETA_scope='current prediction artifact byte readback only')),flush=True)
    assert completed == artifact.size and sha(path) == artifact.hash
    return dict(task=task.id,key=key,sha256=artifact.hash,bytes=artifact.size)


def readback(seed):
    import numpy as np
    from clearml import Task
    job,directory,manifest,weights = inputs(seed)
    task = Task.get_task(task_id=job['task_id'])
    assert str(task.status) == 'completed'
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == job['plan']['producer_sha256']
    world = job['plan']['world_size']
    expected = {'receipt',*(f'predictions-rank{rank}' for rank in range(world))}
    assert set(task.artifacts) == expected
    output = DEST/f'seed{seed}'
    assert not output.exists(), 'accepted or partial output attempt retained; do not restart'
    output.mkdir(parents=True,exist_ok=False)
    inventory = {key:download(task,key,output/key) for key in sorted(expected)}
    producer = json.loads((output/'receipt').read_bytes())
    assert producer['kind'] == 'rbf_matching_seen_val_target_free_joint_identity_GPU_row_forward_candidate_v1'
    assert producer['plan'] == job['plan'] and producer['seed'] == seed
    assert producer['complete_admitted_input_row_forward'] is True and producer['validation_checkpoint_selection'] is False
    assert len(producer['ranks']) == world and {r['rank'] for r in producer['ranks']} == set(range(world))
    assert len({r['uuid'] for r in producer['ranks']}) == world
    counts, total = {}, 0
    for rank in range(world):
        runtime = next(r for r in producer['ranks'] if r['rank'] == rank)
        assert runtime['TF32_enabled'] is False and runtime['world_size'] == world
        assert runtime['optimizer_created'] is False and runtime['GT_read'] is False
        assigned = manifest['shards'][rank::world]
        assert [(x['sequence_id'],x['rows'],x['source_shard_sha256']) for x in runtime['sequences']] == [
            (x['sequence_id'],x['nodes'],x['sha256']) for x in assigned]
        with (output/f'predictions-rank{rank}').open() as stream:
            for record in assigned:
                with np.load(directory/record['path'],allow_pickle=False) as payload:
                    a = {key:payload[key] for key in ('node_id','contexts','lengths','decision_us')}
                for index in range(record['nodes']):
                    row = json.loads(next(stream))
                    assert set(row) == {'sequence_id','row','node_id','decision_us','context_indices','logits'}
                    assert type(row['row']) is int and row['row'] == index and row['sequence_id'] == record['sequence_id']
                    assert row['node_id'] == str(a['node_id'][index]) and row['decision_us'] == int(a['decision_us'][index])
                    length = int(a['lengths'][index])
                    assert row['context_indices'] == a['contexts'][index,:length].tolist()
                    assert len(row['logits']) == length and all(type(v) in (float,int) and math.isfinite(v) for v in row['logits'])
                assert record['sequence_id'] not in counts
                counts[record['sequence_id']] = record['nodes'];total += record['nodes']
            assert not stream.read().strip()
        assert runtime['rows'] == sum(x['nodes'] for x in assigned)
    assert total == manifest['rows'] and counts == {r['sequence_id']:r['nodes'] for r in manifest['shards']}
    proof = output/'independent-byte-coverage.json'
    new(proof,dict(kind='rbf_matching_seen_val_full_independent_output_bytes_and_row_coverage_v1',seed=seed,
        task_id=task.id,artifacts=inventory,all_cloud_output_bytes_independently_read=True,
        all_original_seen_val_input_rows_exactly_once=True,rows=total,sequences=counts,world_size=world,
        row_input_admission_sha256=sha(directory/'independent-acceptance.json'),row_manifest_sha256=sha(directory/'manifest.json'),
        weights_sha256=sha(weights),producer_sha256=job['plan']['producer_sha256'],runtime=producer['ranks'],
        source_sha256=sha(__file__),independent_full_NN_numeric_acceptance=False,paper_performance_complete=False))
    register(proof,'rbf_matching_seen_val_full_independent_output_bytes_and_row_coverage_v1')
    return output


def numeric(seed,output):
    assert sha(REFERENCE) == REFERENCE_SHA
    spec = importlib.util.spec_from_file_location('unchanged_joint_float64_reference',REFERENCE)
    reference = importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    np = reference.np
    job,directory,manifest,weights_path = inputs(seed)
    weights = reference.weights(weights_path)
    proof = json.loads((output/'independent-byte-coverage.json').read_bytes())
    assert proof['weights_sha256'] == sha(weights_path) and proof['task_id'] == job['task_id']
    started, done, maximum, failed, results = time.monotonic(),0,0.,0,[]
    numeric_output = output/'full-numeric';numeric_output.mkdir(exist_ok=False)
    plan = dict(kind='rbf_matching_seen_val_full_independent_float64_joint_NN_numeric_v1',seed=seed,
        prediction_byte_proof_sha256=sha(output/'independent-byte-coverage.json'),weights_sha256=sha(weights_path),
        row_input_admission_sha256=sha(directory/'independent-acceptance.json'),source_sha256=sha(__file__),
        unchanged_reference_sha256=REFERENCE_SHA,atol=1e-4,rtol=1e-4,numpy=np.__version__,
        no_Torch_or_production_NN_imports=True,all_rows_without_GT_selection=True)
    new(numeric_output/'plan.json',plan)
    with (numeric_output/'failed-rows.jsonl').open('x') as failures:
        for rank in range(proof['world_size']):
            path = output/f'predictions-rank{rank}'
            assert sha(path) == proof['artifacts'][path.name]['sha256']
            with path.open() as stream:
                for record in manifest['shards'][rank::proof['world_size']]:
                    with np.load(directory/record['path'],allow_pickle=False) as payload:
                        arrays = {key:payload[key] for key in ('features','source','information_us','arrival_us','mean',
                            'covariance','state_us','lengths','contexts','decision_us','node_id')}
                    scene_error, scene_failures = 0.,0
                    for start in range(0,record['nodes'],64):
                        rows = [json.loads(next(stream)) for _ in range(start,min(start+64,record['nodes']))]
                        indices = np.arange(start,start+len(rows),dtype=np.int64)
                        expected = reference.reference(weights,arrays,indices)
                        for index,row,vector in zip(indices,rows,expected):
                            assert row['sequence_id'] == record['sequence_id'] and row['row'] == int(index)
                            assert row['node_id'] == str(arrays['node_id'][index])
                            assert row['decision_us'] == int(arrays['decision_us'][index])
                            assert row['context_indices'] == arrays['contexts'][index,:int(arrays['lengths'][index])].tolist()
                            got = np.asarray(row['logits'],dtype=np.float64)
                            assert got.shape == vector.shape and np.isfinite(got).all() and np.isfinite(vector).all()
                            error = float(np.max(np.abs(got-vector)));maximum=max(maximum,error);scene_error=max(scene_error,error)
                            if not np.allclose(got,vector,atol=1e-4,rtol=1e-4):
                                failed += 1;scene_failures += 1
                                failures.write(json.dumps(dict(sequence_id=record['sequence_id'],row=int(index),
                                    max_absolute_error=error,GPU=got.tolist(),independent_float64=vector.tolist()))+'\n')
                        done += len(rows)
                    results.append(dict(sequence_id=record['sequence_id'],rows=record['nodes'],
                        numeric_failed_rows=scene_failures,max_absolute_error=scene_error))
                    print(json.dumps(dict(stage='complete_seen_val_independent_float64_joint_NN',seed=seed,
                        completed_rows=done,total_rows=proof['rows'],numeric_failed_rows=failed,
                        ETA_seconds=(time.monotonic()-started)*(proof['rows']-done)/done if done else None,
                        ETA_scope='remaining independent NN numeric rows only')),flush=True)
                assert not stream.read().strip()
    assert len(results) == len({x['sequence_id'] for x in results}) == 21 and done == proof['rows']
    receipt = numeric_output/'completion.json'
    new(receipt,dict(plan,full_independent_numeric_pass=failed==0,rows=done,sequences=results,
        numeric_failed_rows=failed,max_absolute_error=maximum,seconds=time.monotonic()-started,
        actual_recoverable_forest_replay_complete=False,full_online_RBF_accepted=False,paper_performance_complete=False))
    register(receipt,'rbf_matching_seen_val_full_independent_float64_joint_NN_numeric_v1')
    assert failed == 0, 'retain all numerical failures and original tolerance'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True,choices=(1337,2027,3407));args=parser.parse_args()
    root = Path(__file__).resolve().parent
    prepared = json.loads((root/'preparation.json').read_bytes())
    for name,record in prepared['sources'].items():
        assert sha(root/name) == record['sha256']
    output = DEST/f'seed{args.seed}'
    try:
        readback(args.seed);numeric(args.seed,output)
    except BaseException as error:
        if output.exists() and not (output/'execution-failure.json').exists():
            failure = output/'execution-failure.json'
            new(failure,dict(seed=args.seed,exception_type=type(error).__name__,source_sha256=sha(__file__),
                no_automatic_retry=True,partials_and_failed_rows_preserved=True,independent_NN_accepted=False))
            register(failure,'rbf_matching_seen_val_full_independent_output_attempt_failure_preserved_v1')
        raise


if __name__ == '__main__':
    main()
