"""Stream train-only counterfactual records; fit/checkpoint a priority MLP.

Sequence holdout is internal to train, fixed before fitting; detector/identity
producers may still be in-sample, so this is NOT pipeline-isolated selection.
No validation/test results, identity labels, GT or raw embeddings are read.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .allocation_policy import FEATURES, RECIPE, TARGET, FrozenPriorityPolicy
from .detection_cache_v2 import canonical, contained_file, sha_file
from .forest_training_data import _new_json
from .forest_training_checkpoint import SCORING_SOURCES


KIND = 'component_priority_model_progress_checkpoint_v1'
DATA_KIND = 'component_priority_counterfactual_train_v1'
SOURCES = ('allocation_policy.py', 'learned_component_allocation.py', 'allocation_training.py',
           'persistent_component_tracking.py', 'persistent_component_store.py', 'persistent_forest.py',
           'frontier_completion.py', 'completion_component_tracking.py',
           'covered_proposal_capacity.py', 'covered_completion_tracking.py',
           'beam_recovery_tracking.py', 'beam_recovery_allocation.py', 'recovery_task_scope.py',
           'persistent_beam_tracking.py', 'persistent_cache_stream.py')
SEEDS = (1337, 2027, 3407)


def allocation_sources():
    return {'transvision/models/event_track_v2x/'+name: sha_file(Path(__file__).with_name(name)) for name in SOURCES}


def _directory(output):
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new nonsymlink output directory required')
    output.mkdir()
    return output


def export_training(replay, receipt_sha256, output):
    replay = Path(replay)
    receipt_path = contained_file(replay, 'receipt.json')
    if sha_file(receipt_path) != receipt_sha256:
        raise ValueError('teacher replay receipt changed')
    receipt = json.loads(receipt_path.read_bytes())
    if (receipt['status'] != 'complete' or receipt.get('cache_split') != 'train'
            or receipt.get('allocation_teacher') is not True or receipt['completed_frames'] != receipt['scheduled_frames']):
        raise ValueError('complete train-only teacher replay required')
    plan_path, trace_path = contained_file(replay, 'plan.json'), contained_file(replay, 'tracking.jsonl')
    if sha_file(plan_path) != receipt['plan_sha256'] or sha_file(trace_path) != receipt['tracking_sha256']:
        raise ValueError('teacher plan or trace changed')
    plan = json.loads(plan_path.read_bytes())
    sources = allocation_sources()
    if any(plan['source_sha256'].get(p) != h for p, h in sources.items()):
        raise ValueError('teacher solver/feature sources differ')
    if plan.get('full_official_train_verified') is True:
        final = json.loads(contained_file(replay, 'full-train-teacher-receipt.json').read_bytes())
        if (final.get('full_official_train_trace_completed') is not True
                or {k: v for k, v in final.items() if k != 'full_official_train_trace_completed'} != receipt):
            raise ValueError('final full-train teacher verification is missing or differs')
    output = _directory(output)
    inventory, sequence_order, counts, files = [], [], {}, {}
    frame_count = 0
    try:
        with trace_path.open('rb') as stream:
            for line in stream:
                audit = json.loads(line)['tracking']
                if audit.get('training_trace_only') is not True or audit.get('offline_counterfactual_probes') is not True:
                    raise ValueError('trace contains nonteacher inference')
                scene = audit['sequence_id']
                if scene not in files:
                    sequence_order.append(scene)
                    path = output/f'sequence-{len(files):04d}.jsonl'
                    files[scene] = path.open('xb'), path
                    counts[scene] = dict(groups=0, rows=0)
                trace_field = ('recovery_allocation_trace' if audit.get('beam_recovery_allocation') is True
                               else 'allocation_trace')
                for event in audit[trace_field]:
                    record = event['allocation_training']
                    if record['feature_recipe'] != RECIPE or record['target_recipe'] != TARGET:
                        raise ValueError('counterfactual recipe differs')
                    options = record['candidates']
                    x = np.asarray([r['features'] for r in options], dtype=float)
                    y = np.asarray([r['target'] for r in options], dtype=float)
                    _validate_group(x, y)
                    for r, features, target in zip(options, x, y):
                        expected = features[0]*(r['model_bound_before']-r['model_bound_after'])/max(1, r['charged_steps'])
                        if abs(expected-target) > 1e-12:
                            raise ValueError('counterfactual target arithmetic differs')
                    files[scene][0].write(canonical(dict(features=x.tolist(), targets=y.tolist()))+b'\n')
                    counts[scene]['groups'] += 1; counts[scene]['rows'] += len(y)
                frame_count += 1
        if frame_count != receipt['completed_frames'] or set(sequence_order) != set(receipt['sequence_heads']):
            raise ValueError('teacher trace does not cover its receipt')
    finally:
        for stream, _ in files.values():
            stream.close()
    for scene in sequence_order:
        path = files[scene][1]
        inventory.append(dict(sequence_id=scene, path=path.name, sha256=sha_file(path), **counts[scene]))
    if sha_file(trace_path) != receipt['tracking_sha256'] or allocation_sources() != sources:
        raise ValueError('teacher trace or sources changed during extraction')
    manifest = dict(kind=DATA_KIND, split='train', feature_recipe=RECIPE, feature_names=FEATURES,
        target_recipe=TARGET, shards=inventory, replay_receipt_sha256=receipt_sha256,
        full_official_train_trace=plan.get('full_official_train_verified') is True,
        binding=plan['allocation_training_binding'], source_sha256=sources,
        labels_are_model_not_true_risk=True, future_or_gt_inputs=False, paper_eligible=False)
    _new_json(output/'manifest.json', manifest)
    return manifest


def _validate_group(x, y):
    if (x.ndim != 2 or x.shape[1] != len(FEATURES) or not 1 <= len(x) <= 4096
            or y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(np.abs(y) > 1.+1e-12)):
        raise ValueError('invalid finite priority group or normalized target')


def groups(root, record):
    path = contained_file(root, record['path'])
    if sha_file(path) != record['sha256']:
        raise ValueError('priority training shard changed')
    count = rows = 0
    with path.open('rb') as stream:
        for line in stream:
            value = json.loads(line)
            if set(value) != {'features', 'targets'}:
                raise ValueError('extra fields in priority model inputs')
            x, y = np.asarray(value['features'], dtype=float), np.asarray(value['targets'], dtype=float)
            _validate_group(x, y)
            count += 1; rows += len(y)
            yield x, y
    if count != record['groups'] or rows != record['rows'] or sha_file(path) != record['sha256']:
        raise ValueError('priority shard group/row coverage changed')


def fit_priority(data, manifest_sha256, output, *, epochs=10, hidden=32, learning_rate=.001, require_full_train=True):
    import torch
    if (type(require_full_train) is not bool or type(epochs) is not int or epochs < 1 or type(hidden) is not int or not 1 <= hidden <= 256
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('valid priority fitting hyperparameters required')
    data = Path(data)
    path = contained_file(data, 'manifest.json')
    if sha_file(path) != manifest_sha256:
        raise ValueError('priority training manifest changed')
    manifest = json.loads(path.read_bytes())
    sources = allocation_sources()
    if (manifest['kind'] != DATA_KIND or manifest['split'] != 'train'
            or manifest['feature_recipe'] != RECIPE or manifest['feature_names'] != list(FEATURES)
            or manifest['target_recipe'] != TARGET or manifest['source_sha256'] != sources
            or require_full_train and manifest['full_official_train_trace'] is not True):
        raise ValueError('train-only current-source priority data required; fixture is not full train')
    records = manifest['shards']
    scenes = [r['sequence_id'] for r in records]
    if len(scenes) < 2 or len(set(scenes)) != len(scenes):
        raise ValueError('at least two distinct train sequences required for isolated head holdout')
    ordered = sorted(scenes, key=lambda s: (hashlib.sha256(('priority-holdout-v1:'+s).encode()).hexdigest(), s))
    holdout = set(ordered[:max(1, len(scenes)//5)])
    fit_records, held_records = ([r for r in records if (r['sequence_id'] in holdout) == held] for held in (False, True))
    if not sum(r['groups'] for r in fit_records) or not sum(r['groups'] for r in held_records):
        raise ValueError('train fit and sequence holdout both require nonempty trace groups')
    output = _directory(output)
    _new_json(output/'plan.json', dict(manifest_sha256=manifest_sha256, source_sha256=sources,
        seeds=SEEDS, epochs=epochs, hidden=hidden, learning_rate=learning_rate,
        fit_sequences=sorted(set(scenes)-holdout), holdout_sequences=sorted(holdout),
        objective='equal_group_mean_squared_one_operation_model_progress',
        selection='fixed_final_epoch_no_holdout_checkpoint_selection',
        strict_pipeline_isolated_selection=False, upstream_may_be_in_sample=True, paper_eligible=False))
    results = []
    try:
        for seed in SEEDS:
            directory = output/str(seed); directory.mkdir()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), hidden), torch.nn.Tanh(),
                    torch.nn.Linear(hidden, 1), torch.nn.Tanh()).double()
                initial = FrozenPriorityPolicy([v.detach().numpy() for v in
                    (model[0].weight, model[0].bias, model[2].weight, model[2].bias)]).signature
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                with (directory/'epochs.jsonl').open('xb') as log:
                    for epoch in range(epochs):
                        total = count = 0
                        for record in fit_records:
                            for x, y in groups(data, record):
                                optimizer.zero_grad(set_to_none=True)
                                loss = (model(torch.from_numpy(x))[:, 0]-torch.from_numpy(y)).square().mean()
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                                optimizer.step()
                                total += float(loss.detach()); count += 1
                        held_loss = held_count = 0
                        with torch.no_grad():
                            for record in held_records:
                                for x, y in groups(data, record):
                                    held_loss += float((model(torch.from_numpy(x))[:, 0]-torch.from_numpy(y)).square().mean())
                                    held_count += 1
                        log.write(canonical(dict(epoch=epoch+1, training_mse=total/count,
                            train_sequence_holdout_mse=held_loss/held_count, fit_groups=count, holdout_groups=held_count,
                            official_validation_or_test_read=False))+b'\n'); log.flush()
                weights = [v.detach().numpy() for v in (model[0].weight, model[0].bias, model[2].weight, model[2].bias)]
                policy = FrozenPriorityPolicy(weights)
                if policy.signature == initial:
                    raise ValueError('optimizer did not change priority model weights')
                with (directory/'weights.npz').open('xb') as stream:
                    np.savez(stream, **{'w'+str(i): w for i, w in enumerate(policy.weights)})
            checkpoint = dict(kind=KIND, feature_recipe=RECIPE, feature_names=FEATURES, target_recipe=TARGET,
                seed=seed, split='train', binding=manifest['binding'], source_sha256=sources,
                weights_sha256=sha_file(directory/'weights.npz'), policy_signature=policy.signature,
                full_official_train_trace=require_full_train, head_fit_sequence_isolated=True,
                head_fit_includes_holdout=False, initial_policy_signature=initial,
                fit_sequences=sorted(set(scenes)-holdout), holdout_sequences=sorted(holdout),
                strict_pipeline_isolated_selection=False, paper_eligible=False,
                training_manifest_sha256=manifest_sha256, plan_sha256=sha_file(output/'plan.json'))
            _new_json(directory/'checkpoint.json', checkpoint)
            results.append(dict(seed=seed, checkpoint_sha256=sha_file(directory/'checkpoint.json'),
                                policy_signature=policy.signature))
        if sha_file(path) != manifest_sha256 or allocation_sources() != sources:
            raise ValueError('priority data or source changed during fitting')
        receipt = dict(kind='component_priority_fit_receipt_v1', status='complete', seeds=results,
                       paper_eligible=False, real_tracking_validation=False)
        _new_json(output/'receipt.json', receipt)
        return receipt
    except BaseException as error:
        _new_json(output/'failure.json', dict(status='failed', error_type=type(error).__name__, error=str(error)))
        raise


def load_priority(root, manifest_sha256, *, binding, require_full_train=True):
    if type(require_full_train) is not bool:
        raise TypeError('explicit full-train trace requirement required')
    root = Path(root)
    path, weights = contained_file(root, 'checkpoint.json'), contained_file(root, 'weights.npz')
    if sha_file(path) != manifest_sha256:
        raise ValueError('allocation checkpoint identity differs')
    manifest = json.loads(path.read_bytes())
    if (manifest['kind'] != KIND or manifest['split'] != 'train' or manifest['binding'] != binding
            or manifest['source_sha256'] != allocation_sources() or manifest['feature_recipe'] != RECIPE
            or manifest['target_recipe'] != TARGET or manifest['feature_names'] != list(FEATURES)
            or require_full_train and manifest['full_official_train_trace'] is not True):
        raise ValueError('allocation model sources, train provenance or tracking binding differs')
    if sha_file(weights) != manifest['weights_sha256']:
        raise ValueError('allocation weights changed')
    with np.load(weights, allow_pickle=False) as archive:
        if set(archive.files) != {'w0', 'w1', 'w2', 'w3'}:
            raise ValueError('invalid allocation weights archive')
        policy = FrozenPriorityPolicy([archive['w'+str(i)] for i in range(4)])
    if sha_file(weights) != manifest['weights_sha256'] or policy.signature != manifest['policy_signature']:
        raise ValueError('allocation model signature differs')
    return policy, manifest


def training_binding(config, scorer_signature, frozen_cache_identity):
    return dict(configuration=asdict(config), factor_scorer_signature=scorer_signature,
        frozen_cache_identity=frozen_cache_identity,
        factor_implementation_sha256={name: sha_file(Path(__file__).with_name(name)) for name in
            (*SCORING_SOURCES, 'recoverable_states.py', 'fusion.py', 'arrays.py')})
