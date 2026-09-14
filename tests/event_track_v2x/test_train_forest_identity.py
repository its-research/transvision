from dataclasses import replace
import json
from argparse import Namespace

import pytest
import torch

from tools.event_track_v2x.train_forest_identity import FitConfig, fit_dataset, training_sources, ROOT
from tools.event_track_v2x.prepare_forest_training import preparation_sources
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows
from tools.event_track_v2x.run_trained_persistent_forest_v2 import run as run_full_val
from tools.event_track_v2x.run_component_persistent_forest_v2 import run as run_component_val
from tools.event_track_v2x.run_fixed_beam_v2 import run as run_beam_val
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.forest_training_data import TrainingShard
from transvision.models.event_track_v2x.forest_training import batched_row_logits
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.persistent_forest import PersistentForestConfig
from transvision.models.event_track_v2x.persistent_component_tracking import PersistentComponentConfig
from transvision.models.event_track_v2x.persistent_beam_tracking import PersistentBeamConfig, PersistentRankedBeamConfig
from transvision.models.event_track_v2x.persistent_joint_beam import PersistentJointBeamConfig
from transvision.models.event_track_v2x.persistent_probabilistic_tracking import PersistentProbabilisticConfig
from tools.event_track_v2x.run_probabilistic_tracking_v2 import run as run_probabilistic_val
from transvision.models.event_track_v2x.allocation_training import (
    allocation_sources, export_training, fit_priority, load_priority, training_binding,
)
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
from test_forest_training_data import prepared_rows


def test_preparation_hash_contract_is_accepted_by_current_training_sources():
    trained = training_sources()
    assert all(trained[path.relative_to(ROOT).as_posix()] == sha for path, sha in preparation_sources().items())


def test_observation_reuse_keeps_all_three_seed_training_weights_identical(prepared_rows, tmp_path, monkeypatch):
    data, _, _, _ = prepared_rows
    config = FitConfig(epochs=2, batch_size=2, hidden=8, heads=2, dropout=.1)
    cached = TrainingShard._observation
    hashes = []
    for reuse in (True, False):
        with monkeypatch.context() as patch:
            if not reuse:
                def fresh(self, index):
                    self._observations.clear()
                    return cached(self, index)
                patch.setattr(TrainingShard, '_observation', fresh)
            output = tmp_path/('reuse' if reuse else 'rebuild')
            result = fit_dataset(data, sha_file(data/'manifest.json'), output,
                                 config=config, require_full_train=False)
            hashes.append([json.loads((output/r['checkpoint_manifest']).read_bytes())['model_sha256']
                           for r in result['seeds']])
    assert hashes[0] == hashes[1]


def test_three_seed_fixture_fit_checkpoint_inference_and_cpu_reproducibility(prepared_rows, tmp_path):
    data, manifest, cache, schedule = prepared_rows
    config = FitConfig(epochs=2, batch_size=3, hidden=8, heads=2, dropout=.1)
    hashes = []
    for repeat in range(2):
        output = tmp_path/f'fit-{repeat}'
        receipt = fit_dataset(data, sha_file(data/'manifest.json'), output, config=config, require_full_train=False)
        assert receipt['local_fixture_only'] and not receipt['full_official_train']
        assert [r['seed'] for r in receipt['seeds']] == [1337, 2027, 3407]
        current = []
        for result in receipt['seeds']:
            path = output/result['checkpoint_manifest']
            scorer, checkpoint = load_identity_checkpoint(path.parent, result['checkpoint_sha256'], config=ForestTrackingConfig())
            assert checkpoint['model_sha256'] != checkpoint['initial_model_sha256']
            assert not checkpoint['paper_eligible'] and not checkpoint['calibration_guarantee']
            shard = TrainingShard(data/manifest['shards'][0]['path'], manifest['shards'][0], 8)
            context, _ = shard.example(2)
            factors = scorer(context.observations, context.support, context.decision_us)
            with torch.inference_mode():
                logits = batched_row_logits(scorer.model, [context])[0]
            torch.testing.assert_close(torch.tensor([v for _, v in factors.rows[-1]]), logits.log_softmax(0))
            current.append(checkpoint['model_sha256'])
            if repeat == 0 and result['seed'] == 1337:
                replay = replay_rows(cache, schedule, tmp_path/'trained-replay', PersistentForestConfig(), learned_scorer=scorer)
                assert replay['learned_identity_enabled'] and not replay['geometry_development_baseline']
                assert replay['completed_frames'] == 2 and not replay['trained_paper_method']
                component_replay = replay_rows(cache, schedule, tmp_path/'trained-component-replay',
                    PersistentComponentConfig(), learned_scorer=scorer)
                assert component_replay['learned_identity_enabled'] and component_replay['persistent_component_allocation']
                assert component_replay['completed_frames'] == 2 and not component_replay['trained_paper_method']
                beam_replay = replay_rows(cache, schedule, tmp_path/'trained-beam-replay',
                    PersistentBeamConfig(), learned_scorer=scorer)
                assert beam_replay['learned_identity_enabled'] and beam_replay['irreversible_beam_enabled']
                assert beam_replay['completed_frames'] == 2 and not beam_replay['trained_paper_method']
                ranked_replay = replay_rows(cache, schedule, tmp_path/'trained-ranked-beam-replay',
                    PersistentRankedBeamConfig(), learned_scorer=scorer)
                assert ranked_replay['complete_batch_beam_ranking'] and ranked_replay['completed_frames'] == 2
                joint_replay = replay_rows(cache, schedule, tmp_path/'trained-joint-beam-replay',
                    PersistentJointBeamConfig(), learned_scorer=scorer)
                assert joint_replay['joint_cartesian_beam_ranking'] and joint_replay['completed_frames'] == 2
                for update in ('jpda-ci', 'jpda-kalman', 'pkf'):
                    probabilistic = replay_rows(cache, schedule, tmp_path/('trained-'+update),
                        PersistentProbabilisticConfig(update_rule=update), learned_scorer=scorer)
                    assert probabilistic['probabilistic_single_history_enabled'] and probabilistic['learned_identity_enabled']
                    assert probabilistic['completed_frames'] == 2 and not probabilistic['paper_eligible']
                # Train BOTH heads, with the allocator trained only on causal
                # train-side solver traces from this very identity scorer.
                allocation_config = PersistentComponentConfig()
                binding = training_binding(allocation_config, scorer.signature, frozen_cache_identity(cache))
                teacher = tmp_path/'trained-teacher'
                replay_rows(cache, schedule, teacher, allocation_config, learned_scorer=scorer,
                    allocation_teacher=True, plan=dict(source_sha256=allocation_sources(),
                        allocation_training_binding=binding, full_official_train_verified=False))
                priority_data, priority_fit = tmp_path/'priority-data', tmp_path/'priority-fit'
                export_training(teacher, sha_file(teacher/'receipt.json'), priority_data)
                priority_result = fit_priority(priority_data, sha_file(priority_data/'manifest.json'), priority_fit,
                    epochs=1, hidden=4, require_full_train=False)
                policy, _ = load_priority(priority_fit/'1337', priority_result['seeds'][0]['checkpoint_sha256'],
                    binding=binding, require_full_train=False)
                allocated = replay_rows(cache, schedule, tmp_path/'trained-allocated-replay',
                    allocation_config, learned_scorer=scorer, allocation_policy=policy)
                assert allocated['learned_identity_enabled'] and allocated['learned_allocation_enabled']
                # Scoring context and factors must remain byte-identical, not
                # just use models with equally named parameters.
                runs = [tmp_path/name/'tracking.jsonl' for name in ('trained-replay', 'trained-component-replay',
                        'trained-beam-replay', 'trained-ranked-beam-replay', 'trained-joint-beam-replay', 'trained-allocated-replay',
                        'trained-jpda-ci', 'trained-jpda-kalman', 'trained-pkf')]
                factors = [[json.loads(line)['tracking']['factor_rows_sha256'] for line in p.read_text().splitlines()] for p in runs]
                assert all(f == factors[0] for f in factors)
                with pytest.raises(ValueError, match='fixture checkpoint'):
                    run_full_val(Namespace(checkpoint=path.parent, checkpoint_sha256=result['checkpoint_sha256'], device='cpu'))
                with pytest.raises(ValueError, match='fixture checkpoint'):
                    run_component_val(Namespace(checkpoint=path.parent, checkpoint_sha256=result['checkpoint_sha256'], device='cpu'))
                with pytest.raises(ValueError, match='fixture checkpoint'):
                    run_beam_val(Namespace(width=4, checkpoint=path.parent, checkpoint_sha256=result['checkpoint_sha256'], device='cpu'))
                with pytest.raises(ValueError, match='fixture checkpoint'):
                    run_probabilistic_val(Namespace(checkpoint=path.parent, checkpoint_sha256=result['checkpoint_sha256'], device='cpu'))
        hashes.append(current)
    assert hashes[0] == hashes[1] and len(set(hashes[0])) == 3


def test_full_train_cli_gate_does_not_relabel_fixture(prepared_rows, tmp_path):
    data, *_ = prepared_rows
    with pytest.raises(ValueError, match='full official train'):
        fit_dataset(data, sha_file(data/'manifest.json'), tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


def test_checkpoint_protocol_and_bytes_are_pinned(prepared_rows, tmp_path):
    data, *_ = prepared_rows
    output = tmp_path/'fit'
    receipt = fit_dataset(data, sha_file(data/'manifest.json'), output,
        config=FitConfig(epochs=1, batch_size=4, hidden=8, heads=2, dropout=0.), require_full_train=False)
    result = receipt['seeds'][0]
    path = output/result['checkpoint_manifest']
    with pytest.raises(ValueError, match='protocol differs'):
        load_identity_checkpoint(path.parent, result['checkpoint_sha256'], config=ForestTrackingConfig(parent_limit=2))
    weights = path.parent/'weights.pt'
    weights.write_bytes(weights.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='weight identity'):
        load_identity_checkpoint(path.parent, result['checkpoint_sha256'], config=ForestTrackingConfig())
