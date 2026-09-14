from dataclasses import replace
import json

import numpy as np
import pytest

from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows
from transvision.models.event_track_v2x.allocation_training import (
    allocation_sources, export_training, fit_priority, load_priority, training_binding,
)
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.learned_component_allocation import LearnedComponentTracker
from transvision.models.event_track_v2x.persistent_component_tracking import PersistentComponentConfig
from test_run_persistent_forest_v2 import replay_inputs
from test_detection_cache_v2 import sources
from test_forest_training_data import prepared_rows


@pytest.fixture
def priority_data(prepared_rows, tmp_path):
    _, _, cache, rows = prepared_rows
    config = PersistentComponentConfig(state=ForestTrackingConfig(active_limit=2, expansion_budget=16))
    binding = training_binding(config, GeometryForestScorer().signature, frozen_cache_identity(cache))
    output = tmp_path/'teacher'
    result = replay_rows(cache, rows, output, config, allocation_teacher=True,
        plan=dict(source_sha256=allocation_sources(), allocation_training_binding=binding,
                  full_official_train_verified=False))
    data = tmp_path/'priority-data'
    manifest = export_training(output, sha_file(output/'receipt.json'), data)
    assert result['allocation_teacher'] and result['cache_split'] == 'train'
    assert not manifest['full_official_train_trace']
    return data, config, binding, cache, rows


def test_teacher_forbids_validation_cache_before_payload_or_output(replay_inputs, tmp_path):
    cache, rows = replay_inputs
    with pytest.raises(ValueError, match='actual sealed train'):
        replay_rows(cache, rows, tmp_path/'forbidden', PersistentComponentConfig(), allocation_teacher=True)
    assert not (tmp_path/'forbidden').exists()


def test_three_seed_train_export_frozen_inference_and_sequence_holdout(priority_data, tmp_path):
    data, config, binding, cache, rows = priority_data
    manifest = json.loads((data/'manifest.json').read_bytes())
    for r in manifest['shards']:
        for line in (data/r['path']).read_bytes().splitlines():
            assert set(json.loads(line)) == {'features', 'targets'}
    seen = []
    for repeat in range(2):
        output = tmp_path/f'fit-{repeat}'
        result = fit_priority(data, sha_file(data/'manifest.json'), output,
            epochs=2, hidden=8, require_full_train=False)
        signatures = []
        plan = json.loads((output/'plan.json').read_bytes())
        assert set(plan['fit_sequences']).isdisjoint(plan['holdout_sequences'])
        assert set(plan['fit_sequences']) | set(plan['holdout_sequences']) == {r['sequence_id'] for r in manifest['shards']}
        for record in result['seeds']:
            root = output/str(record['seed'])
            policy, checkpoint = load_priority(root, record['checkpoint_sha256'], binding=binding, require_full_train=False)
            assert checkpoint['initial_policy_signature'] != policy.signature
            assert not checkpoint['head_fit_includes_holdout'] and not checkpoint['strict_pipeline_isolated_selection']
            signatures.append(policy.signature)
            if repeat == 0 and record['seed'] == 1337:
                replay = tmp_path/'allocated'
                allocated = replay_rows(cache, rows, replay, config, allocation_policy=policy)
                assert allocated['learned_allocation_enabled'] and not allocated['allocation_teacher']
                normal = tmp_path/'deterministic'
                replay_rows(cache, rows, normal, config)
                factors = [[json.loads(line)['tracking']['factor_rows_sha256'] for line in (p/'tracking.jsonl').read_bytes().splitlines()]
                           for p in (replay, normal)]
                assert factors[0] == factors[1]
                for head in allocated['sequence_heads'].values():
                    t = LearnedComponentTracker.open(replay/head['database'], allocation_policy=policy,
                        expected_prediction_sha256=head['prediction_sha256'], expected_database_sha256=head['database_sha256'])
                    t.close()
                with pytest.raises(ValueError, match='provenance'):
                    load_priority(root, record['checkpoint_sha256'], binding=binding)
                with pytest.raises(ValueError, match='tracking binding'):
                    load_priority(root, record['checkpoint_sha256'], binding=dict(binding, factor_scorer_signature='0'*64), require_full_train=False)
        seen.append(signatures)
    assert seen[0] == seen[1] and len(set(seen[0])) == 3


def test_production_fit_rejects_fixture_without_output(priority_data, tmp_path):
    data, *_ = priority_data
    with pytest.raises(ValueError, match='fixture is not full train'):
        fit_priority(data, sha_file(data/'manifest.json'), tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


def test_holdout_targets_cannot_change_fitted_weights(priority_data, tmp_path):
    data, *_ = priority_data
    first = tmp_path/'first'
    original = fit_priority(data, sha_file(data/'manifest.json'), first, epochs=1, hidden=4, require_full_train=False)
    heldout = set(json.loads((first/'plan.json').read_bytes())['holdout_sequences'])
    manifest_path = data/'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    for shard in manifest['shards']:
        if shard['sequence_id'] not in heldout:
            continue
        path = data/shard['path']
        groups = [json.loads(line) for line in path.read_bytes().splitlines()]
        for group in groups:
            group['targets'] = [.9 for _ in group['targets']]
        path.write_bytes(b'\n'.join(canonical(g) for g in groups)+b'\n')
        shard['sha256'] = sha_file(path)
    manifest_path.write_bytes(canonical(manifest))
    changed = fit_priority(data, sha_file(manifest_path), tmp_path/'changed', epochs=1, hidden=4, require_full_train=False)
    assert [r['policy_signature'] for r in original['seeds']] == [r['policy_signature'] for r in changed['seeds']]


def test_dataset_byte_mutation_is_not_accepted(priority_data, tmp_path):
    data, *_ = priority_data
    manifest = json.loads((data/'manifest.json').read_bytes())
    (data/manifest['shards'][0]['path']).write_bytes(b'{}\n')
    with pytest.raises(ValueError, match='shard changed'):
        fit_priority(data, sha_file(data/'manifest.json'), tmp_path/'failed-fit', epochs=1, require_full_train=False)
    assert not (tmp_path/'failed-fit'/'receipt.json').exists()
