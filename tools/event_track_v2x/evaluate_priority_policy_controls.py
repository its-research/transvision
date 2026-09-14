#!/usr/bin/env python3
"""Frozen priority-head controls on the fixed TRAIN sequence partition.

No fitting, label filtering, epoch/seed selection, GT, val/test or replay. The
CLI requires all three completed four-A100 runs and the complete train export.
Labels describe one-step MODEL-bound progress, not true risk or tracking gain.
Receipts establish offline consistency, not independent execution attestation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np

from tools.event_track_v2x.train_allocation_policy_ddp import (
    PriorityFitConfig, audit_data, priority_ddp_sources, validate_runtime,
)
from transvision.models.event_track_v2x.allocation_policy import FEATURES
from transvision.models.event_track_v2x.allocation_training import (
    SEEDS, _validate_group, groups, load_priority,
)
from transvision.models.event_track_v2x.detection_cache_v2 import contained_file, sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json

TOLERANCE = 1e-12
ZERO = 'zero_predictor'
RULE = 'weighted_eta_feature_rule'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def sources():
    return dict(priority_ddp_sources(), **{
        Path(__file__).relative_to(ROOT).as_posix(): sha_file(__file__)})


def current_runtime(runtimes, *, require_full_train):
    validate_runtime(runtimes, require_full_train=require_full_train)
    if require_full_train:
        require(all('A100' in r['gpu_name'] for r in runtimes), 'current policy requires A100 only')


def choices(features, policies):
    """No target argument: decisions cannot use the counterfactual labels."""
    x = np.asarray(features, dtype=float)
    require(x.ndim == 2 and x.shape[1] == len(FEATURES) and 1 <= len(x) <= 4096
            and np.isfinite(x).all(), 'finite bounded priority features required')
    predictions = {ZERO: np.zeros(len(x)), RULE: x[:, FEATURES.index('weighted_eta')]}
    for seed in SEEDS:
        predictions['learned_' + str(seed)] = np.asarray(policies[seed].scores(x), dtype=float)
    require(all(v.shape == (len(x),) and np.isfinite(v).all() for v in predictions.values()),
            'finite one-score-per-candidate outputs required')
    # np.argmax selects the FIRST index of exact ties. Candidate order is kept.
    return {name: (int(np.argmax(v)), None if name == RULE else v)
            for name, v in predictions.items()}


class Metrics:
    def __init__(self):
        self.count = self.rows = self.optimal = self.wins = self.ties = self.losses = 0
        self.target = self.shortfall = self.delta = self.mse = 0.
        self.has_mse = None

    def add(self, targets, selection, reference):
        index, prediction = selection
        selected, baseline, maximum = float(targets[index]), float(targets[reference]), float(targets.max())
        difference = selected - baseline
        self.count += 1
        self.rows += len(targets)
        self.target += selected
        self.shortfall += maximum - selected
        self.delta += difference
        self.optimal += maximum - selected <= TOLERANCE
        self.wins += difference > TOLERANCE
        self.losses += difference < -TOLERANCE
        self.ties += abs(difference) <= TOLERANCE
        has_mse = prediction is not None
        require(self.has_mse is None or self.has_mse == has_mse, 'prediction type changed')
        self.has_mse = has_mse
        if has_mse:
            self.mse += float(np.mean((prediction - targets) ** 2))

    def result(self):
        def mean(total):
            return total / self.count if self.count else None
        return dict(groups=self.count, candidate_rows=self.rows,
            equal_group_mse=mean(self.mse) if self.has_mse else None,
            mean_selected_model_progress=mean(self.target),
            mean_one_step_model_progress_shortfall=mean(self.shortfall),
            mean_progress_delta_vs_feature_rule=mean(self.delta),
            within_tolerance_of_best_count=self.optimal,
            within_tolerance_of_best_fraction=mean(self.optimal),
            wins_vs_feature_rule=self.wins, ties_vs_feature_rule=self.ties,
            losses_vs_feature_rule=self.losses)


def new_profile():
    return {subset: {name: Metrics() for name in (ZERO, RULE, *('learned_' + str(s) for s in SEEDS))}
            for subset in ('all_groups', 'informative_groups')}


def add_group(profiles, features, targets, policies):
    _validate_group(features, targets)
    chosen = choices(features, policies)
    # Target-dependent strata are descriptive ONLY, after all actions are fixed.
    informative = len(targets) >= 2 and float(np.ptp(targets)) > TOLERANCE
    for profile in profiles:
        for subset in ('all_groups', 'informative_groups') if informative else ('all_groups',):
            for name, selection in chosen.items():
                profile[subset][name].add(targets, selection, chosen[RULE][0])


def profile_result(profile):
    return {subset: {name: metrics.result() for name, metrics in methods.items()}
            for subset, methods in profile.items()}


def read_pinned(root, name, digest, pins):
    path = contained_file(root, name)
    require(sha_file(path) == digest, 'pinned artifact differs: ' + name)
    pins.append((path, digest))
    return json.loads(path.read_bytes())


def inspect_run(root, digest, data_sha, manifest, fit, held, statistics, *, require_full_train):
    """Check saved artifacts; a receipt is not remote/GPU execution attestation."""
    root = Path(root)
    pins = []
    receipt = read_pinned(root, 'receipt.json', digest, pins)
    require(receipt.get('kind') == 'component_priority_ddp_receipt_v1'
            and receipt.get('status') == 'complete', 'completed priority DDP receipt required')
    seed = receipt['seed']
    require(type(seed) is int and seed in SEEDS, 'predeclared priority seed required')
    require(receipt.get('full_official_train_trace') is require_full_train
            and receipt.get('local_fixture_only') is (not require_full_train)
            and receipt.get('paper_eligible') is False
            and receipt.get('tracking_validation_performed') is False,
            'honest complete-train/fixture receipt boundary required')
    plan = read_pinned(root, 'plan.json', receipt['plan_sha256'], pins)
    config = PriorityFitConfig(**plan['fit_config'])
    require(plan.get('kind') == 'component_priority_ddp_plan_v1'
            and plan.get('seed') == seed and plan.get('training_manifest_sha256') == data_sha
            and plan.get('training_source_sha256') == priority_ddp_sources()
            and plan.get('source_sha256') == manifest['source_sha256']
            and plan.get('binding') == manifest['binding'] and plan.get('statistics') == statistics,
            'priority plan data/source/binding differs')
    require(plan.get('full_official_train_trace') is require_full_train
            and plan.get('local_fixture_only') is (not require_full_train)
            and all(plan.get(k) is False for k in
                    ('paper_eligible', 'strict_pipeline_isolated_selection', 'final_full_train_refit'))
            and plan.get('selection') == 'fixed_final_epoch_no_holdout_checkpoint_selection'
            and plan.get('objective') == 'equal_group_mean_squared_one_operation_model_progress'
            and plan.get('dtype') == 'float64'
            and plan.get('partition') == 'strided_global_groups_no_padding_no_drop'
            and plan.get('global_batch_groups') == config.batch_groups,
            'fixed train-only priority fitting contract differs')
    fit_scenes, held_scenes = (sorted(r['sequence_id'] for r in records) for records in (fit, held))
    require(plan.get('fit_sequences') == fit_scenes and plan.get('holdout_sequences') == held_scenes,
            'fixed sequence partition differs')
    runtimes = receipt['rank_runtime']
    current_runtime(runtimes, require_full_train=require_full_train)
    world = len(runtimes)
    require(receipt['world_size'] == plan['world_size'] == world
            and plan['rank_runtime'] == runtimes, 'saved rank inventories differ')
    expected_path = str(seed) + '/checkpoint.json'
    require(receipt['checkpoint_manifest'] == expected_path, 'checkpoint seed directory differs')
    cp = read_pinned(root, expected_path, receipt['checkpoint_sha256'], pins)
    policy, loaded_cp = load_priority(root / str(seed), receipt['checkpoint_sha256'],
                                     binding=manifest['binding'], require_full_train=require_full_train)
    require(cp == loaded_cp, 'checkpoint changed during loading')
    require(cp.get('seed') == seed and cp.get('training_manifest_sha256') == data_sha
            and cp.get('training_source_sha256') == priority_ddp_sources()
            and cp.get('plan_sha256') == receipt['plan_sha256']
            and cp.get('fit_sequences') == fit_scenes and cp.get('holdout_sequences') == held_scenes
            and cp.get('head_fit_sequence_isolated') is True and cp.get('head_fit_includes_holdout') is False
            and cp.get('fixed_final_epoch') == config.epochs
            and cp.get('distributed_world_size') == world
            and cp.get('distributed_backend') == runtimes[0]['backend']
            and policy.weights[0].shape[0] == config.hidden
            and re.fullmatch(r'[0-9a-f]{64}', cp.get('initial_policy_signature') or '') is not None
            and cp.get('initial_policy_signature') != policy.signature
            and all(cp.get(k) is False for k in
                    ('paper_eligible', 'strict_pipeline_isolated_selection', 'final_full_train_refit')),
            'fixed final checkpoint provenance differs')
    pins.append((contained_file(root / str(seed), 'weights.npz'), cp['weights_sha256']))
    log = contained_file(root, str(seed) + '/epochs.jsonl')
    require(sha_file(log) == receipt['epochs_sha256'], 'epoch log hash differs')
    pins.append((log, receipt['epochs_sha256']))
    epochs = [json.loads(line) for line in log.read_bytes().splitlines()]
    require([e['epoch'] for e in epochs] == list(range(1, config.epochs + 1)), 'epoch coverage differs')
    expected_batches = math.ceil(statistics['fit']['groups'] / config.batch_groups)
    for epoch in epochs:
        progress = epoch['rank_progress']
        require(epoch.get('official_validation_or_test_read') is False
                and epoch['fit_groups'] == statistics['fit']['groups']
                and epoch['holdout_groups'] == statistics['holdout']['groups']
                and epoch['global_batches'] == expected_batches
                and sorted(p['rank'] for p in progress) == list(range(world))
                and len({p['policy_signature'] for p in progress}) == 1
                and all(p['batches'] == expected_batches for p in progress),
                'epoch rank or group coverage differs')
        for partition in ('fit', 'holdout'):
            for unit in ('groups', 'rows'):
                key = partition + '_' + unit
                require(sum(p[key] for p in progress) == statistics[partition][unit],
                        'epoch sample coverage differs')
        if require_full_train:
            require(all(0 < p['nonempty_batches'] <= expected_batches
                        and 0 < p['positive_gradient_batches'] <= expected_batches for p in progress),
                    'saved ranks have no optimizer work')
        for field, rank_field, partition in (
            ('training_mse', 'group_loss_sum', 'fit'),
            ('train_sequence_holdout_mse', 'holdout_group_loss_sum', 'holdout')):
            require(all(finite(p[rank_field]) and p[rank_field] >= 0 for p in progress)
                    and finite(epoch[field]) and epoch[field] >= 0, 'nonfinite or negative loss')
            expected = math.fsum(p[rank_field] for p in progress) / statistics[partition]['groups']
            require(math.isclose(epoch[field], expected, rel_tol=1e-10, abs_tol=1e-12),
                    'epoch loss reduction differs')
    require(all(p['policy_signature'] == policy.signature for p in epochs[-1]['rank_progress']),
            'final rank weights differ from checkpoint')
    return dict(seed=seed, policy=policy, plan=plan, pins=pins, receipt_sha256=digest,
                checkpoint_sha256=receipt['checkpoint_sha256'],
                heldout_mse=epochs[-1]['train_sequence_holdout_mse'])


def evaluate(data, data_sha, runs, *, require_full_train=True):
    require(type(require_full_train) is bool, 'explicit full-train boundary required')
    runs = list(runs)
    require(len(runs) == len(SEEDS) and len({Path(r[0]).resolve() for r in runs}) == len(SEEDS),
            'three distinct priority run directories required')
    before = sources()
    manifest, fit, held, statistics = audit_data(data, data_sha, require_full_train=require_full_train)
    checked = [inspect_run(root, digest, data_sha, manifest, fit, held, statistics,
                           require_full_train=require_full_train) for root, digest in runs]
    require(sorted(r['seed'] for r in checked) == list(SEEDS), 'exactly all three priority seeds required')
    common = [{k: v for k, v in r['plan'].items() if k not in ('seed', 'rank_runtime')} for r in checked]
    require(all(p == common[0] for p in common), 'seed fitting contracts differ')
    policies = {r['seed']: r['policy'] for r in checked}
    require(len({p.signature for p in policies.values()}) == len(SEEDS), 'independent saved models required')
    partitions = {}
    for name, records in (('fit', fit), ('train_sequence_holdout', held)):
        pooled = new_profile()
        scenes = {}
        for record in records:
            scene = new_profile()
            for x, y in groups(data, record):
                add_group((pooled, scene), x, y, policies)
            scenes[record['sequence_id']] = profile_result(scene)
        partitions[name] = dict(pooled=profile_result(pooled), per_sequence=scenes)
    for run in checked:
        observed = partitions['train_sequence_holdout']['pooled']['all_groups'][
            'learned_' + str(run['seed'])]['equal_group_mse']
        require(math.isclose(observed, run['heldout_mse'], rel_tol=1e-9, abs_tol=1e-12),
                'recomputed final holdout MSE differs from training log')
    pins = [(contained_file(data, 'manifest.json'), data_sha)]
    pins += [(contained_file(data, r['path']), r['sha256']) for r in manifest['shards']]
    pins += [pin for run in checked for pin in run['pins']]
    require(all(sha_file(path) == digest for path, digest in pins) and sources() == before,
            'evaluation inputs or source changed')
    return dict(kind='priority_policy_fixed_controls_v1', completed=True, partitions=partitions,
        training_manifest_sha256=data_sha, source_sha256=before,
        run_bindings=[dict(seed=r['seed'], receipt_sha256=r['receipt_sha256'],
                          checkpoint_sha256=r['checkpoint_sha256'], policy_signature=r['policy'].signature)
                      for r in sorted(checked, key=lambda r: r['seed'])],
        seeds=list(SEEDS), tolerance=TOLERANCE, exact_tie_break='first_exported_candidate_index',
        primary_partition='train_sequence_holdout', aggregation='equal_group_with_per_sequence_details',
        informative_definition='candidate_count >= 2 and target_range > tolerance; descriptive only',
        feature_rule_is_recorded_behavior_verified=False,
        feature_rule_mse_reported=False, labels_are_one_step_model_progress=True,
        training_group_filtering=False, parameter_fitting_performed=False,
        final_fit_mse_is_training_online_loss=False,
        best_seed_or_checkpoint_selected=False, full_official_train=require_full_train,
        local_fixture_only=not require_full_train, offline_artifact_consistency_only=True,
        remote_execution_attested=False, teacher_trace_reaudited=False,
        validation_or_test_read=False, tracking_validation_performed=False,
        sequential_control_benefit_established=False, strict_pipeline_isolated_selection=False,
        paper_eligible=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--run', nargs=2, action='append', required=True,
                        metavar=('DIRECTORY', 'RECEIPT_SHA256'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists() and not any(p.is_symlink() for p in (args.output, *args.output.parents)),
            'new nonsymlink report output required')
    report = evaluate(args.data, args.manifest_sha256, args.run)
    _new_json(args.output, report)
    print(json.dumps(dict(output=str(args.output), report_sha256=sha_file(args.output),
                          seeds=report['seeds'], tracking_validation_performed=False), sort_keys=True))


if __name__ == '__main__':
    main()
