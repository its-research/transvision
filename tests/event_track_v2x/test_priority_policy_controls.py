from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from tools.event_track_v2x import evaluate_priority_policy_controls as tool
from tools.event_track_v2x.train_allocation_policy_ddp import PriorityFitConfig, audit_data, priority_ddp_sources
from transvision.models.event_track_v2x.allocation_policy import FEATURES, RECIPE, TARGET, FrozenPriorityPolicy
from transvision.models.event_track_v2x.allocation_training import DATA_KIND, KIND, SEEDS, allocation_sources, groups
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file


def write(path, value):
    path.write_bytes(canonical(value))
    return sha_file(path)


def policies():
    # Static analytical/random fixtures, no parameter fitting or optimizer.
    result = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        result[seed] = FrozenPriorityPolicy((rng.normal(size=(2, len(FEATURES))) / 10,
            rng.normal(size=2) / 10, rng.normal(size=(1, 2)) / 10, np.zeros(1)))
    return result


def features(n):
    x = np.zeros((n, len(FEATURES)))
    x[:, -1] = np.arange(n) / max(1, n)
    return x


def test_equal_group_mse_does_not_weight_large_candidate_sets_more():
    profile = tool.new_profile()
    frozen = policies()
    tool.add_group((profile,), features(1), np.ones(1), frozen)
    tool.add_group((profile,), features(9), np.zeros(9), frozen)
    result = tool.profile_result(profile)
    assert result['all_groups'][tool.ZERO]['equal_group_mse'] == .5
    assert result['all_groups'][tool.ZERO]['candidate_rows'] == 10
    assert result['all_groups'][tool.RULE]['equal_group_mse'] is None
    assert result['informative_groups'][tool.ZERO]['groups'] == 0
    assert result['informative_groups'][tool.ZERO]['mean_selected_model_progress'] is None


def test_actions_use_features_only_and_exact_ties_keep_export_order():
    frozen = policies()
    x = features(3)
    x[:, -1] = .5
    before = tool.choices(x, frozen)
    for name, (selected, _) in before.items():
        assert selected == 0
    for targets in (np.array([0., 1., -1.]), np.array([1., -1., 0.])):
        profile = tool.new_profile()
        tool.add_group((profile,), x, targets, frozen)
        for name, (selected, _) in before.items():
            assert profile['all_groups'][name].target == targets[selected]
    # A larger feature value always wins the feature rule, even if the label is worse.
    x[-1, -1] += 1e-14
    assert tool.choices(x, frozen)[tool.RULE][0] == 2


def test_negative_labels_shortfall_and_win_tie_loss_are_signed():
    profile = tool.new_profile()
    frozen = policies()
    for y in ([-.2, -.8], [-.9, -.1], [.3, .3]):
        tool.add_group((profile,), features(2), np.array(y), frozen)
    zero = tool.profile_result(profile)['all_groups'][tool.ZERO]
    assert zero['wins_vs_feature_rule'] == zero['losses_vs_feature_rule'] == zero['ties_vs_feature_rule'] == 1
    assert zero['mean_progress_delta_vs_feature_rule'] == pytest.approx(-.2 / 3)
    assert zero['mean_one_step_model_progress_shortfall'] == pytest.approx(.8 / 3)
    assert zero['within_tolerance_of_best_count'] == 2


def test_near_zero_and_single_candidate_groups_are_retained():
    profile = tool.new_profile()
    for y in ([.2], [0., 5e-13], [0., 2e-12]):
        tool.add_group((profile,), features(len(y)), np.array(y), policies())
    result = tool.profile_result(profile)
    assert result['all_groups'][tool.ZERO]['groups'] == 3
    assert result['informative_groups'][tool.ZERO]['groups'] == 1


def test_lower_mse_than_zero_does_not_imply_better_priority_decision():
    class Reversed:
        def scores(self, _): return np.array([.0951, .0949])
    profile = tool.new_profile()
    tool.add_group((profile,), features(2), np.array([.09, .10]), {s: Reversed() for s in SEEDS})
    results = tool.profile_result(profile)['all_groups']
    learned = results['learned_1337']
    assert learned['equal_group_mse'] < results[tool.ZERO]['equal_group_mse']
    assert learned['mean_one_step_model_progress_shortfall'] == pytest.approx(.01)
    assert learned['losses_vs_feature_rule'] == 1
    assert learned['within_tolerance_of_best_count'] == 0


def test_model_progress_shortfall_has_full_two_unit_range():
    profile = tool.new_profile()
    tool.add_group((profile,), features(2), np.array([-1., 1.]), policies())
    assert tool.profile_result(profile)['all_groups'][tool.ZERO][
        'mean_one_step_model_progress_shortfall'] == 2.


@pytest.mark.parametrize('bad', ['5090', 'V100', 'cpu', 'duplicate_uuid', 'three_cards'])
def test_production_runtime_inventory_is_at_least_four_distinct_a100_cards(bad):
    runtimes = [dict(rank=r, world_size=4, host='fixture', backend='nccl', device='cuda:'+str(r),
                     gpu_uuid='fixture-'+str(r), gpu_name='NVIDIA A100') for r in range(4)]
    tool.current_runtime(runtimes, require_full_train=True)
    if bad in ('5090', 'V100'): runtimes[0]['gpu_name'] = 'NVIDIA '+bad
    if bad == 'cpu': runtimes[0].update(device='cpu', backend='gloo')
    if bad == 'duplicate_uuid': runtimes[0]['gpu_uuid'] = runtimes[1]['gpu_uuid']
    if bad == 'three_cards': runtimes.pop()
    with pytest.raises(ValueError): tool.current_runtime(runtimes, require_full_train=True)


def test_regression_to_one_step_shortfall_bound_on_exhaustive_small_vectors():
    import itertools
    profile = tool.Metrics()
    for y_tuple in itertools.product((-1., 0., 1.), repeat=3):
        y = np.array(y_tuple)
        for f_tuple in itertools.product((-1., 0., 1.), repeat=3):
            prediction = np.array(f_tuple)
            index = int(np.argmax(prediction))
            shortfall = float(y.max() - y[index])
            bound = min(2., float(np.sqrt(2 * np.sum((prediction-y)**2))))
            assert shortfall <= bound + 1e-12
            profile.add(y, (index, prediction), 0)
    assert profile.count == 729


@pytest.mark.parametrize('bad', ['nan_x', 'nan_y', 'target_range', 'shape', 'empty', 'score_shape'])
def test_invalid_inputs_fail_without_silent_score_fallback(bad):
    x, y, frozen = features(2), np.zeros(2), policies()
    if bad == 'nan_x': x[0, 0] = np.nan
    if bad == 'nan_y': y[0] = np.nan
    if bad == 'target_range': y[0] = 2.
    if bad == 'shape': x = x[:, :-1]
    if bad == 'empty': x, y = x[:0], y[:0]
    if bad == 'score_shape':
        class BadPolicy:
            def scores(self, _): return np.array([0.])
        frozen[1337] = BadPolicy()
    with pytest.raises(ValueError): tool.add_group((tool.new_profile(),), x, y, frozen)


@pytest.fixture
def campaign(tmp_path):
    data = tmp_path/'data'
    data.mkdir()
    shards = []
    for i in range(5):
        path = data/f'sequence-{i:04d}.jsonl'
        rows = [dict(features=features(n).tolist(), targets=(np.arange(n) / 5 + i / 100).tolist())
                for n in (1, 2)]
        path.write_bytes(b''.join(canonical(r) + b'\n' for r in rows))
        shards.append(dict(sequence_id=f'{i:04d}', path=path.name, sha256=sha_file(path), groups=2, rows=3))
    manifest = dict(kind=DATA_KIND, split='train', source_sha256=allocation_sources(),
        feature_recipe=RECIPE, target_recipe=TARGET, feature_names=list(FEATURES),
        labels_are_model_not_true_risk=True, future_or_gt_inputs=False, full_official_train_trace=False,
        replay_receipt_sha256='a'*64, shards=shards, binding={'fixture': 'not an actual training run'})
    data_sha = write(data/'manifest.json', manifest)
    manifest, fit, held, stats = audit_data(data, data_sha, require_full_train=False)
    frozen = policies()
    config = PriorityFitConfig(epochs=1, hidden=2, batch_groups=4)
    runs = []
    for seed in SEEDS:
        root = tmp_path/str(seed)
        directory = root/str(seed)
        directory.mkdir(parents=True)
        runtimes = [dict(rank=r, world_size=2, host='fixture', backend='gloo', device='cpu') for r in range(2)]
        plan = dict(kind='component_priority_ddp_plan_v1', seed=seed,
            training_manifest_sha256=data_sha, training_source_sha256=priority_ddp_sources(),
            source_sha256=manifest['source_sha256'], binding=manifest['binding'], statistics=stats,
            fit_config=asdict(config), full_official_train_trace=False, local_fixture_only=True,
            paper_eligible=False, strict_pipeline_isolated_selection=False, final_full_train_refit=False,
            selection='fixed_final_epoch_no_holdout_checkpoint_selection',
            objective='equal_group_mean_squared_one_operation_model_progress', dtype='float64',
            partition='strided_global_groups_no_padding_no_drop', global_batch_groups=4,
            fit_sequences=sorted(r['sequence_id'] for r in fit),
            holdout_sequences=sorted(r['sequence_id'] for r in held), world_size=2, rank_runtime=runtimes)
        plan_sha = write(root/'plan.json', plan)
        with (directory/'weights.npz').open('wb') as stream:
            np.savez(stream, **{'w'+str(i): w for i, w in enumerate(frozen[seed].weights)})
        cp = dict(kind=KIND, seed=seed, split='train', binding=manifest['binding'],
            source_sha256=allocation_sources(), feature_recipe=RECIPE, target_recipe=TARGET,
            feature_names=list(FEATURES), full_official_train_trace=False,
            weights_sha256=sha_file(directory/'weights.npz'), policy_signature=frozen[seed].signature,
            training_manifest_sha256=data_sha, training_source_sha256=priority_ddp_sources(),
            plan_sha256=plan_sha, fit_sequences=plan['fit_sequences'], holdout_sequences=plan['holdout_sequences'],
            head_fit_sequence_isolated=True, head_fit_includes_holdout=False, fixed_final_epoch=1,
            distributed_world_size=2, distributed_backend='gloo', initial_policy_signature='b'*64,
            paper_eligible=False, strict_pipeline_isolated_selection=False, final_full_train_refit=False)
        cp_sha = write(directory/'checkpoint.json', cp)
        losses = {name: sum(float(np.mean((frozen[seed].scores(x)-y)**2))
                            for record in records for x, y in groups(data, record))
                  for name, records in (('fit', fit), ('holdout', held))}
        progress = []
        for rank in range(2):
            p = dict(rank=rank, batches=2, nonempty_batches=2, positive_gradient_batches=0,
                policy_signature=frozen[seed].signature, group_loss_sum=losses['fit']/2,
                holdout_group_loss_sum=losses['holdout']/2)
            for part in ('fit', 'holdout'):
                for unit in ('groups', 'rows'):
                    p[part+'_'+unit] = stats[part][unit]//2 + int(rank < stats[part][unit]%2)
            progress.append(p)
        epoch = dict(epoch=1, official_validation_or_test_read=False, fit_groups=stats['fit']['groups'],
            holdout_groups=stats['holdout']['groups'], global_batches=2, rank_progress=progress,
            training_mse=losses['fit']/stats['fit']['groups'],
            train_sequence_holdout_mse=losses['holdout']/stats['holdout']['groups'])
        epoch_sha = write(directory/'epochs.jsonl', epoch)
        receipt = dict(kind='component_priority_ddp_receipt_v1', status='complete', seed=seed,
            full_official_train_trace=False, local_fixture_only=True, paper_eligible=False,
            tracking_validation_performed=False, plan_sha256=plan_sha, world_size=2, rank_runtime=runtimes,
            checkpoint_manifest=str(seed)+'/checkpoint.json', checkpoint_sha256=cp_sha, epochs_sha256=epoch_sha)
        runs.append((root, write(root/'receipt.json', receipt)))
    return data, data_sha, runs


def test_fixture_end_to_end_checks_all_seeds_and_final_holdout_loss(campaign):
    data, digest, runs = campaign
    result = tool.evaluate(data, digest, runs, require_full_train=False)
    assert result['seeds'] == list(SEEDS)
    assert result['local_fixture_only'] and not result['full_official_train']
    assert not result['tracking_validation_performed'] and not result['parameter_fitting_performed']
    assert not result['feature_rule_is_recorded_behavior_verified']
    assert not result['remote_execution_attested']
    assert result['partitions']['fit']['pooled']['all_groups'][tool.ZERO]['groups'] == 8
    assert result['partitions']['train_sequence_holdout']['pooled']['all_groups'][tool.ZERO]['groups'] == 2
    assert len(result['partitions']['fit']['per_sequence']) == 4
    # Even structurally consistent fixture receipts cannot enter the production CLI.
    with pytest.raises(ValueError, match='complete 46-sequence'):
        tool.evaluate(data, digest, runs)


@pytest.mark.parametrize('bad', ['partial', 'duplicate', 'receipt_hash', 'weights', 'shard',
                                 'heldout_loss', 'rank_counts', 'seed', 'holdout_in_fit', 'source', 'initial'])
def test_saved_artifact_and_partition_tampering_is_rejected(campaign, bad):
    data, digest, runs = campaign
    root = runs[0][0]
    receipt = json.loads((root/'receipt.json').read_bytes())
    if bad == 'partial': runs = runs[:2]
    if bad == 'duplicate': runs[1] = runs[0]
    if bad == 'receipt_hash': runs[0] = (root, '0'*64)
    if bad == 'weights': (root/'1337/weights.npz').write_bytes(b'invalid')
    if bad == 'shard': (data/'sequence-0000.jsonl').write_bytes(b'{}')
    if bad in ('heldout_loss', 'rank_counts'):
        path = root/'1337/epochs.jsonl'
        epoch = json.loads(path.read_bytes())
        if bad == 'heldout_loss':
            epoch['train_sequence_holdout_mse'] += .1
            for rank in epoch['rank_progress']: rank['holdout_group_loss_sum'] += .1
        else: epoch['rank_progress'][0]['fit_groups'] += 1
        receipt['epochs_sha256'] = write(path, epoch)
    if bad in ('seed', 'holdout_in_fit', 'source', 'initial'):
        path = root/'1337/checkpoint.json'
        cp = json.loads(path.read_bytes())
        if bad == 'seed': cp['seed'] = 2027
        if bad == 'holdout_in_fit': cp['head_fit_includes_holdout'] = True
        if bad == 'source': cp['source_sha256'] = {}
        if bad == 'initial': del cp['initial_policy_signature']
        receipt['checkpoint_sha256'] = write(path, cp)
    if bad in ('heldout_loss', 'rank_counts', 'seed', 'holdout_in_fit', 'source', 'initial'):
        runs[0] = (root, write(root/'receipt.json', receipt))
    with pytest.raises(ValueError): tool.evaluate(data, digest, runs, require_full_train=False)


def test_cli_rejects_existing_output_before_input_processing(tmp_path):
    output = tmp_path/'report.json'
    output.write_text('preserve')
    with pytest.raises(ValueError, match='new nonsymlink'):
        tool.main(['--data', str(tmp_path), '--manifest-sha256', 'a'*64,
                   '--run', str(tmp_path), 'b'*64, '--output', str(output)])
    assert output.read_text() == 'preserve'


def test_cli_has_no_fixture_flag(tmp_path):
    with pytest.raises(SystemExit):
        tool.main(['--data', str(tmp_path), '--manifest-sha256', 'a'*64,
                   '--run', str(tmp_path), 'b'*64, '--output', str(tmp_path/'out.json'),
                   '--allow-fixture'])
