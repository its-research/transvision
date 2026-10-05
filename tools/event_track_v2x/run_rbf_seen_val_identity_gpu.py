"""Generic GPU4/GPU8 forward over admitted target-free original val contexts.

This consumes a separately named prediction-only schema. It cannot interpret
train supervision arrays, select validation checkpoints, or claim full forest
replay/performance from candidate row logits.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import types

import requests
from clearml import Task
from clearml.backend_api.session import Session


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def fetch(specification, path):
    task = Task.get_task(task_id=specification['task'])
    assert str(task.status) == 'completed'
    artifact = task.artifacts[specification['key']]
    assert artifact.hash == specification['sha256'] and artifact.size == specification['bytes']
    url = artifact.url.replace('10.100.35.118:8081', '10.100.34.118:8081')
    with requests.get(url, headers={'Authorization': 'Bearer ' + Session().token},
                      stream=True, timeout=(10, 120)) as response:
        assert response.status_code == 200
        with path.open('xb') as stream:
            for block in response.iter_content(1024**2):
                stream.write(block)
    assert path.stat().st_size == artifact.size and sha(path) == artifact.hash


def unpack(path, destination):
    destination.mkdir()
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        assert len({m.name for m in members}) == len(members)
        assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            and not m.issym() and not m.islnk() and (m.isfile() or m.isdir()) for m in members)
        archive.extractall(destination, filter='data')


def packages(root):
    sys.path.insert(0, str(root))
    for name in ('transvision', 'transvision.models', 'transvision.models.event_track_v2x'):
        module = types.ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split('.')))]
        sys.modules[name] = module


class PredictionShard:
    def __init__(self, path, record, parent_limit):
        import numpy as np
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as payload:
            self.arrays = {key: payload[key] for key in payload.files}
        a, n = self.arrays, record['nodes']
        assert set(a) == {'features', 'mean', 'covariance', 'score', 'source', 'node_id', 'frame_id',
            'cache_sha256', 'information_us', 'arrival_us', 'state_us', 'detection_index',
            'contexts', 'lengths', 'decision_us'}
        assert all(len(value) == n and not value.dtype.hasobject for value in a.values())
        assert a['features'].shape == (n, 203) and a['mean'].shape == (n, 9) and a['covariance'].shape == (n, 9, 9)
        assert a['contexts'].shape == (n, parent_limit+1)
        assert all(np.isfinite(a[key]).all() for key in ('features', 'mean', 'covariance', 'score'))
        assert all(a[key].dtype == np.int64 for key in ('source', 'information_us', 'arrival_us', 'state_us',
            'detection_index', 'contexts', 'lengths', 'decision_us'))
        assert (a['state_us'] <= a['information_us']).all() and (a['information_us'] <= a['arrival_us']).all()
        assert (a['arrival_us'] <= a['decision_us']).all()
        assert ((a['source'] == 0) | (a['source'] == 1)).all() and (a['state_us'] >= 0).all()
        for row in range(n):
            length = int(a['lengths'][row])
            assert 1 <= length <= parent_limit+1
            indices = a['contexts'][row, :length]
            assert indices[-1] == row and (indices >= 0).all() and (np.diff(indices) > 0).all()
            assert (a['contexts'][row, length:] == -1).all()
        self.sequence_id, self.memo = record['sequence_id'], {}

    def observation(self, index):
        if index not in self.memo:
            from transvision.models.event_track_v2x.forest_tracking import RawIdentityDetection
            from transvision.models.event_track_v2x.identity_forest import IdentityNode
            a = self.arrays
            self.memo[index] = RawIdentityDetection(self.sequence_id,
                IdentityNode(str(a['node_id'][index]), int(a['source'][index]), int(a['information_us'][index]),
                             int(a['arrival_us'][index]), str(a['frame_id'][index])),
                int(a['detection_index'][index]), int(a['state_us'][index]), a['mean'][index], a['covariance'][index],
                float(a['score'][index]), a['features'][index], str(a['cache_sha256'][index]))
        return self.memo[index]

    def context(self, row):
        from transvision.models.event_track_v2x.forest_row_context import ForestRowContext
        a = self.arrays
        length = int(a['lengths'][row])
        assert 1 <= length <= a['contexts'].shape[1]
        indices = tuple(map(int, a['contexts'][row, :length]))
        assert indices[-1] == row and indices == tuple(sorted(set(indices))) and indices[0] >= 0
        assert (a['contexts'][row, length:] == -1).all()
        return ForestRowContext(indices, tuple(self.observation(i) for i in indices), int(a['decision_us'][row]))


def work(rank, base_string, plan):
    import numpy as np
    import torch
    base = Path(base_string)
    packages(base / 'source')
    from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
    from transvision.models.event_track_v2x.forest_training import batched_row_logits
    from transvision.models.event_track_v2x.forest_potentials import neural_parent_logits
    from transvision.models.event_track_v2x.recoverable_identity import model_digest
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    capability = torch.cuda.get_device_capability(rank)
    assert 'sm_%d%d' % capability in torch.cuda.get_arch_list(), 'native CUDA architecture unavailable'
    checkpoint = json.loads((base / 'checkpoint').read_bytes())
    manifest = json.loads((base / 'row-manifest').read_bytes())
    state = torch.load(base / 'weights.pt', map_location='cpu', weights_only=True)
    cpu = RecoverableIdentityModel(**checkpoint['architecture'])
    cpu.load_state_dict(state, strict=True)
    cpu.eval().requires_grad_(False)
    assert model_digest(cpu) == checkpoint['model_sha256']
    gpu = RecoverableIdentityModel(**checkpoint['architecture']).cuda(rank)
    gpu.load_state_dict(state, strict=True)
    gpu.eval().requires_grad_(False)
    assigned = manifest['shards'][rank::plan['world_size']]
    total, done, started, probes = sum(s['nodes'] for s in assigned), 0, time.monotonic(), []
    with (base / f'predictions-rank{rank}.jsonl').open('x') as stream, torch.inference_mode():
        for record in assigned:
            relative = Path(record['path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            shard = PredictionShard(base / 'rows' / relative, record, checkpoint['row_protocol']['parent_limit'])
            a, maximum_probe_error = shard.arrays, 0.
            for start in range(0, record['nodes'], 64):
                contexts = [shard.context(row) for row in range(start, min(start+64, record['nodes']))]
                logits = batched_row_logits(gpu, contexts, max_nodes=9, max_batch=64,
                    geometry_weight=checkpoint['geometry_weight'], process_noise=checkpoint['row_protocol']['process_noise'])
                for offset, (context, logit) in enumerate(zip(contexts, logits)):
                    values = logit.cpu().double().numpy()
                    assert np.isfinite(values).all()
                    row = start + offset
                    stream.write(json.dumps(dict(sequence_id=record['sequence_id'], row=row,
                        node_id=str(a['node_id'][row]), decision_us=context.decision_us,
                        context_indices=context.indices, logits=values.tolist()), allow_nan=False) + '\n')
                    if start == 0 and offset < 4:
                        support = tuple((-1,)+tuple(range(i)) for i in range(len(context.observations)))
                        reference = neural_parent_logits(cpu, context.observations, support, context.decision_us,
                            max_nodes=9, max_pairs=81, geometry_weight=checkpoint['geometry_weight'],
                            process_noise=checkpoint['row_protocol']['process_noise'])[-1].double().numpy()
                        assert np.allclose(values, reference, atol=1e-4, rtol=1e-4)
                        maximum_probe_error = max(maximum_probe_error, float(np.max(np.abs(values-reference))))
                done += len(contexts)
                if done % 1024 < 64 or done == total:
                    print(json.dumps(dict(stage='target_free_seen_val_joint_identity_GPU_forward', rank=rank,
                        completed_rows=done, total_rows=total, ETA_seconds=(time.monotonic()-started)*(total-done)/done,
                        ETA_scope='remaining assigned real seen-val NN rows only')), flush=True)
            probes.append(dict(sequence_id=record['sequence_id'], rows=record['nodes'],
                source_shard_sha256=record['sha256'], producer_CPU_probe_max_absolute_error=maximum_probe_error))
    properties = torch.cuda.get_device_properties(rank)
    receipt = dict(rank=rank, world_size=plan['world_size'], rows=done, sequences=probes,
        device=properties.name, uuid=str(properties.uuid), capability=list(capability), torch=torch.__version__,
        cuda=torch.version.cuda, numpy=np.__version__, TF32_enabled=False, seconds=time.monotonic()-started,
        optimizer_created=False, GT_read=False, producer_probes_are_not_full_independent_numeric_acceptance=True)
    (base / f'rank{rank}-receipt.json').write_bytes(canonical(receipt))


def main():
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    import torch
    task = Task.init(project_name='Thesis/Recover-Before-Fuse/Inference', task_name='seen-val joint identity rows',
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        output_uri='http://10.100.34.118:8081')
    parameters = task.get_parameters()
    plan = json.loads(parameters['General/plan'])
    assert hashlib.sha256(canonical(plan)).hexdigest() == parameters['General/recipe_sha256']
    assert plan['world_size'] in (4, 8) and torch.cuda.device_count() == plan['world_size']
    assert plan['seed'] in (1337, 2027, 3407)
    assert plan['independent_complete_feature_history_row_admission'] is True
    base = Path('rbf-matching-seen-val-joint-forward')
    base.mkdir()
    for key in ('source', 'checkpoint', 'weights_archive', 'row-manifest', 'rows', 'row-independent-admission'):
        fetch(plan[key], base / key)
    unpack(base / 'source', base / 'source-unpack')
    (base / 'source').unlink()
    (base / 'source-unpack').rename(base / 'source')
    unpack(base / 'rows', base / 'rows-unpack')
    (base / 'rows').unlink()
    (base / 'rows-unpack').rename(base / 'rows')
    unpack(base / 'weights_archive', base / 'weights-unpack')
    checkpoint = json.loads((base / 'checkpoint').read_bytes())
    candidates = list((base / 'weights-unpack').rglob('weights.pt'))
    assert len(candidates) == 1 and sha(candidates[0]) == checkpoint['weights']['sha256']
    candidates[0].rename(base / 'weights.pt')
    for relative, digest in checkpoint['source_sha256'].items():
        assert sha(base / 'source' / relative) == digest
    manifest = json.loads((base / 'row-manifest').read_bytes())
    admission = json.loads((base / 'row-independent-admission').read_bytes())
    assert admission['kind'] == 'rbf_matching_seen_val_target_free_full_independent_feature_context_admission_v1'
    assert admission['full_original_seen_val_features_and_contexts_independently_accepted'] is True
    assert admission['row_manifest_sha256'] == sha(base / 'row-manifest')
    assert admission['checkpoint_sha256'] == sha(base / 'checkpoint')
    assert manifest['kind'] == 'rbf_prediction_only_original_arrival_row_contexts_v1'
    assert manifest['split'] == 'val' and manifest['seed'] == checkpoint['seed'] == plan['seed']
    assert admission['seed'] == plan['seed'] and admission['cache_manifest_sha256'] == manifest['cache_manifest_sha256']
    assert admission['original_schedule_sha256'] == manifest['original_schedule_sha256']
    assert len(manifest['sequences']) == len(manifest['shards']) == 21 and manifest['original_events'] == 3316
    assert manifest['no_GT_or_dummy_supervision_fields'] is True and manifest['all_empty_arrival_events_preserved'] is True
    assert manifest['row_protocol'] == checkpoint['row_protocol']
    assert manifest['frozen_cache_identity'] == checkpoint['frozen_cache_identity']
    assert manifest['candidate_protocol'] == 'rbf-all-class-top64-v1'
    torch.multiprocessing.spawn(work, args=(str(base), plan), nprocs=plan['world_size'], join=True)
    ranks = [json.loads((base / f'rank{rank}-receipt.json').read_bytes()) for rank in range(plan['world_size'])]
    assert sum(rank['rows'] for rank in ranks) == manifest['rows'] == sum(s['nodes'] for s in manifest['shards'])
    receipt = dict(kind='rbf_matching_seen_val_target_free_joint_identity_GPU_row_forward_candidate_v1',
        seed=plan['seed'], plan=plan, ranks=ranks, complete_admitted_input_row_forward=True,
        independent_output_byte_and_numeric_acceptance=False, actual_forest_replay_complete=False,
        validation_checkpoint_selection=False, full_online_RBF_accepted=False, paper_performance_complete=False)
    (base / 'receipt.json').write_bytes(canonical(receipt))
    for rank in range(plan['world_size']):
        task.upload_artifact(f'predictions-rank{rank}', artifact_object=base/f'predictions-rank{rank}.jsonl', wait_on_upload=True)
    task.upload_artifact('receipt', artifact_object=base/'receipt.json', wait_on_upload=True)
    task.close()


if __name__ == '__main__':
    main()
