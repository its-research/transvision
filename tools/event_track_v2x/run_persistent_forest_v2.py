#!/usr/bin/env python3
"""Development geometry baseline on the sealed full SPD official-val schedule.

This runner is not the trained paper method. It accepts no GT, checkpoint or
test input and publishes nothing. Pair snapshots arrive at reference+100ms;
sensor frames not available then are excluded before payload access, matching
the existing sealed source-availability rule. Output is create-once.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from transvision.models.event_track_v2x.experiment_progress import ExperimentProgress

from tools.event_track_v2x.run_tracking_v2 import schedule_rows, SPLIT_SHA
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery, VerifiedForestCache
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer, LearnedForestScorer
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.persistent_cache_stream import PersistentForestCacheStream
from transvision.models.event_track_v2x.persistent_forest import PersistentForestConfig, PersistentForestTracker
from transvision.models.event_track_v2x.persistent_component_tracking import PersistentComponentConfig, PersistentComponentTracker
from transvision.models.event_track_v2x.persistent_beam_tracking import (
    PersistentBeamConfig, PersistentBeamTracker, PersistentRankedBeamConfig, PersistentRankedBeamTracker,
)
from transvision.models.event_track_v2x.persistent_joint_beam import PersistentJointBeamConfig, PersistentJointBeamTracker
from transvision.models.event_track_v2x.persistent_class_bound_beam import (
    PersistentClassBoundBeamConfig, PersistentClassBoundBeamTracker,
)
from transvision.models.event_track_v2x.persistent_slot_bound_beam import (
    PersistentSlotBoundBeamConfig, PersistentSlotBoundBeamTracker,
)
from transvision.models.event_track_v2x.persistent_sparse_slot_bound_beam import (
    PersistentSparseSlotBoundBeamConfig, PersistentSparseSlotBoundBeamTracker,
)
from transvision.models.event_track_v2x.persistent_reachable_slot_bound_beam import (
    PersistentReachableSlotBoundBeamConfig, PersistentReachableSlotBoundBeamTracker,
)
from transvision.models.event_track_v2x.beam_recovery_tracking import BeamRecoveryConfig, BeamRecoveryTracker
from transvision.models.event_track_v2x.beam_recovery_allocation import LearnedBeamRecoveryTracker, BeamRecoveryTeacherTracker
from transvision.models.event_track_v2x.recovery_task_scope import SCOPE_MODE, VerifiedEgoPoseTable
from transvision.models.event_track_v2x.allocation_policy import FrozenPriorityPolicy
from transvision.models.event_track_v2x.learned_component_allocation import LearnedComponentTracker, AllocationTeacherTracker
from transvision.models.event_track_v2x.completion_component_tracking import (
    PersistentCompletionConfig, CompletionComponentTracker, CompletionTeacherTracker, CompletionLearnedTracker,
)
from transvision.models.event_track_v2x.covered_completion_tracking import (
    PersistentCoveredCompletionConfig, CoveredCompletionTracker, CoveredCompletionTeacher, CoveredCompletionLearned,
)
from transvision.models.event_track_v2x.persistent_probabilistic_tracking import (
    PersistentProbabilisticConfig, PersistentProbabilisticTracker,
)


def _new(path, payload):
    with path.open('xb') as stream:
        stream.write(canonical(payload))


def replay_rows(cache, rows, output, config, *, birth_logit=-4., plan=None, learned_scorer=None,
                allocation_policy=None, allocation_teacher=False, ego_poses=None):
    """Sequence-contiguous replay helper. The CLI separately locks full val."""
    tracker_types = {PersistentForestConfig: PersistentForestTracker, PersistentComponentConfig: PersistentComponentTracker,
                     PersistentCompletionConfig: CompletionComponentTracker,
                     PersistentCoveredCompletionConfig: CoveredCompletionTracker,
                     PersistentBeamConfig: PersistentBeamTracker, PersistentRankedBeamConfig: PersistentRankedBeamTracker,
                     PersistentJointBeamConfig: PersistentJointBeamTracker,
                     PersistentClassBoundBeamConfig: PersistentClassBoundBeamTracker,
                     PersistentSlotBoundBeamConfig: PersistentSlotBoundBeamTracker,
                     PersistentSparseSlotBoundBeamConfig: PersistentSparseSlotBoundBeamTracker,
                     PersistentReachableSlotBoundBeamConfig: PersistentReachableSlotBoundBeamTracker,
                     BeamRecoveryConfig: BeamRecoveryTracker,
                     PersistentProbabilisticConfig: PersistentProbabilisticTracker}
    if type(cache) is not VerifiedForestCache or type(config) not in tracker_types:
        raise TypeError('verified cache and persistent configuration required')
    tracker_type = tracker_types[type(config)]
    scoped = type(config) is BeamRecoveryConfig and config.recovery_allocation_scope == SCOPE_MODE
    if (scoped != (ego_poses is not None) or ego_poses is not None and
            (type(ego_poses) is not VerifiedEgoPoseTable or ego_poses.cache_sha256 != cache.manifest_sha256)):
        raise ValueError('explicit matching pose table only for task-scoped beam recovery')
    cache_split = json.loads(cache.manifest_json)['split']
    component_types = (PersistentComponentConfig, PersistentCompletionConfig, PersistentCoveredCompletionConfig)
    completion_types = (PersistentCompletionConfig, PersistentCoveredCompletionConfig)
    if (type(allocation_teacher) is not bool or allocation_teacher and allocation_policy is not None
            or (allocation_teacher or allocation_policy is not None) and
                (type(config) not in (*component_types, BeamRecoveryConfig) or
                 type(config) is BeamRecoveryConfig and not config.enable_recovery)):
        raise ValueError('allocation policy/teacher requires the recoverable component backend exclusively')
    if allocation_teacher:
        if cache_split != 'train':
            raise ValueError('allocation teacher requires actual sealed train cache before payload access')
        tracker_type = {PersistentComponentConfig: AllocationTeacherTracker,
            BeamRecoveryConfig: BeamRecoveryTeacherTracker,
            PersistentCompletionConfig: CompletionTeacherTracker,
            PersistentCoveredCompletionConfig: CoveredCompletionTeacher}[type(config)]
    if allocation_policy is not None:
        if type(allocation_policy) is not FrozenPriorityPolicy:
            raise TypeError('frozen allocation policy required')
        tracker_type = {PersistentComponentConfig: LearnedComponentTracker,
            BeamRecoveryConfig: LearnedBeamRecoveryTracker,
            PersistentCompletionConfig: CompletionLearnedTracker,
            PersistentCoveredCompletionConfig: CoveredCompletionLearned}[type(config)]
    if learned_scorer is not None and (type(learned_scorer) is not LearnedForestScorer
            or learned_scorer.options['process_noise'] != config.state.process_noise):
        raise ValueError('verified frozen learned scorer with matching motion model required')
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new output directory without symlink traversal required')
    # Check every public row against metadata before creating experiment files.
    previous, order = {}, []
    for row in rows:
        scene, reference = row['sequence_id'], row['box_reference_timestamp_us']
        if not scene or type(reference) is not int or reference <= previous.get(scene, -1):
            raise ValueError('invalid sequence/reference schedule')
        previous[scene] = reference
        order.append(scene)
        for side, key in (('vehicle-side', 'vehicle_frame'), ('infrastructure-side', 'infrastructure_frame')):
            entry, metadata = (json.loads(v) for v in cache.index[(scene, side, row[key])])
            if side == 'vehicle-side' and metadata['box_reference_timestamp_us'] != reference:
                raise ValueError('cache/schedule reference mismatch')
    if not rows or order != sorted(order):
        raise ValueError('nonempty sequence-contiguous schedule required')
    output.mkdir()
    if plan is None:
        plan = dict(kind='persistent_forest_subset_replay_plan_v1', cache_sha256=cache.manifest_sha256,
            configuration=asdict(config), birth_logit=birth_logit, scheduled_frames=len(rows),
            full_official_schedule_verified=False, paper_eligible=False,
            learned_scorer_signature=None if learned_scorer is None else learned_scorer.signature)
    # Record actual execution mode even for caller-supplied plans.
    plan = dict(plan, cache_split=cache_split, allocation_teacher=allocation_teacher,
        allocation_policy_signature=None if allocation_policy is None else allocation_policy.signature,
        ego_pose_table_sha256=None if ego_poses is None else ego_poses.manifest_sha256)
    _new(output/'plan.json', plan)
    prediction_path, audit_path = output/'predictions.jsonl', output/'tracking.jsonl'
    timing_path = output/'frame-timings.jsonl'
    timings, frame_timings, heads, rejected = [], [], {}, {'vehicle-side': 0, 'infrastructure-side': 0}
    tracker, stream, scene, frames = None, None, None, 0
    started = time.monotonic()
    progress = ExperimentProgress("persistent_forest_replay_frames", len(rows))
    try:
        with prediction_path.open('xb') as predictions, audit_path.open('xb') as audits, timing_path.open('xb') as timing_file:
            for row in rows:
                frame_started = time.monotonic()
                if row['sequence_id'] != scene:
                    if tracker is not None:
                        heads[scene]['database_sha256'] = tracker.close()
                    scene = row['sequence_id']
                    # Do not derive output paths from untrusted sequence strings.
                    database = output/f'sequence-{len(heads):04d}.sqlite'
                    tracker = tracker_type(database, sequence_id=scene, config=config,
                        **({} if allocation_policy is None else dict(allocation_policy=allocation_policy)))
                    origin = min(json.loads(meta)['box_reference_timestamp_us'] for key, (_, meta) in cache.index.items()
                                 if key[0] == scene)
                    stream = PersistentForestCacheStream(cache, tracker,
                        learned_scorer if learned_scorer is not None else
                        GeometryForestScorer(birth_logit=birth_logit, process_noise=config.state.process_noise),
                        origin_us=origin,ego_poses=ego_poses)
                    heads[scene] = dict(database=database.name, frames=0)
                reference = row['box_reference_timestamp_us']
                decision = reference+100_000
                deliveries, skipped = [], []
                for side, key in (('vehicle-side', 'vehicle_frame'), ('infrastructure-side', 'infrastructure_frame')):
                    entry, meta = (json.loads(v) for v in cache.index[(scene, side, row[key])])
                    information = max(meta['box_reference_timestamp_us'], meta['source_image_timestamp_us'])
                    if information <= decision:
                        deliveries.append(CacheDelivery(scene, side, row[key], decision, entry['frame_sha256']))
                    else:
                        rejected[side] += 1
                        skipped.append(dict(side=side, frame_id=row[key], information_us=information))
                before = time.monotonic()
                commit = stream.step(deliveries, frame_id=row['vehicle_frame'], reference_us=reference,
                                     decision_us=decision, event_id=row['vehicle_frame'])
                step_seconds = time.monotonic()-before
                timings.append(step_seconds)
                predictions.write(commit.prediction_json+b'\n')
                audits.write(canonical(dict(tracking=commit.tracking_audit,
                    source_unavailable_before_payload_read=skipped))+b'\n')
                frames += 1
                heads[scene].update(frames=heads[scene]['frames']+1,
                    prediction_sha256=commit.prediction['commit_sha256'])
                frame_seconds = time.monotonic()-frame_started
                frame_timings.append(frame_seconds)
                timing_file.write(canonical(dict(sequence_id=scene, frame_id=row['vehicle_frame'],
                    box_reference_timestamp_us=reference, decision_timestamp_us=decision,
                    step_seconds=step_seconds, frame_seconds=frame_seconds))+b'\n')
                progress.update(frames, force=frames == len(rows))
                if frames % 100 == 0:
                    print(json.dumps(dict(kind='persistent_forest_progress', frames=frames,
                        scheduled=len(rows), sequence=scene, elapsed_seconds=time.monotonic()-started)), flush=True)
            heads[scene]['database_sha256'] = tracker.close()
            tracker = None
        result = dict(kind='persistent_forest_scheduled_replay_v1', status='complete',
            scheduled_frames=len(rows), completed_frames=frames, sequence_heads=heads,
            plan_sha256=sha_file(output/'plan.json'),
            predictions_sha256=sha_file(prediction_path), tracking_sha256=sha_file(audit_path),
            frame_timings_sha256=sha_file(timing_path),
            source_unavailable=rejected, elapsed_seconds=time.monotonic()-started,
            latency_seconds_p50_p95_p99_max=np.quantile(timings, [.5, .95, .99, 1.]).tolist(),
            latency_scope='new_cache_load_and_scoring_inference_state_commit_not_startup_audit_or_output_file_io',
            frame_latency_seconds_p50_p95_p99_max=np.quantile(frame_timings, [.5, .95, .99, 1.]).tolist(),
            frame_latency_scope='sequence_transition_source_selection_step_buffered_output_io_excludes_timer_record_final_close_startup',
            latency_clock='time.monotonic', latency_quantile_method='numpy_linear',
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
            database_bytes=sum((output/r['database']).stat().st_size for r in heads.values()),
            geometry_development_baseline=learned_scorer is None, learned_identity_enabled=learned_scorer is not None,
            persistent_component_allocation=type(config) in component_types or
                type(config) is BeamRecoveryConfig and config.enable_recovery,
            frontier_completion_enabled=type(config) in completion_types or
                type(config) is BeamRecoveryConfig and config.enable_recovery and config.recovery_completions_per_component>0,
            coverage_aware_proposal_admission=type(config) is PersistentCoveredCompletionConfig or
                type(config) is BeamRecoveryConfig and config.enable_recovery and config.recovery_completions_per_component>0,
            cache_split=cache_split, cache_sha256=cache.manifest_sha256,
            allocation_teacher=allocation_teacher, learned_allocation_enabled=allocation_policy is not None,
            latency_includes_offline_teacher_probes=allocation_teacher,
            irreversible_beam_enabled=type(config) in (PersistentBeamConfig, PersistentRankedBeamConfig, PersistentJointBeamConfig, PersistentClassBoundBeamConfig, PersistentSlotBoundBeamConfig, PersistentSparseSlotBoundBeamConfig, PersistentReachableSlotBoundBeamConfig) or
                type(config) is BeamRecoveryConfig and not config.enable_recovery,
            beam_recovery_backbone_enabled=type(config) is BeamRecoveryConfig,
            additional_beam_recovery_enabled=type(config) is BeamRecoveryConfig and config.enable_recovery,
            recovery_allocation_scope=getattr(config,'recovery_allocation_scope',None),
            ego_pose_table_sha256=None if ego_poses is None else ego_poses.manifest_sha256,
            complete_batch_beam_ranking=type(config) in (PersistentRankedBeamConfig, PersistentJointBeamConfig, PersistentClassBoundBeamConfig, PersistentSlotBoundBeamConfig, PersistentSparseSlotBoundBeamConfig, PersistentReachableSlotBoundBeamConfig),
            joint_cartesian_beam_ranking=type(config) in (PersistentJointBeamConfig, PersistentClassBoundBeamConfig, PersistentSlotBoundBeamConfig, PersistentSparseSlotBoundBeamConfig, PersistentReachableSlotBoundBeamConfig),
            single_class_ranking_bound_enabled=type(config) in (PersistentClassBoundBeamConfig, PersistentSlotBoundBeamConfig, PersistentSparseSlotBoundBeamConfig, PersistentReachableSlotBoundBeamConfig),
            source_frame_assignment_bound_enabled=type(config) in (PersistentSlotBoundBeamConfig, PersistentSparseSlotBoundBeamConfig, PersistentReachableSlotBoundBeamConfig),
            sparse_assignment_decomposition_enabled=type(config) in (PersistentSparseSlotBoundBeamConfig, PersistentReachableSlotBoundBeamConfig),
            causal_root_reachability_bound_enabled=type(config) is PersistentReachableSlotBoundBeamConfig,
            probabilistic_single_history_enabled=type(config) is PersistentProbabilisticConfig,
            probabilistic_update_rule=config.update_rule if type(config) is PersistentProbabilisticConfig else None,
            probabilistic_association_algorithm=config.association_algorithm if type(config) is PersistentProbabilisticConfig else None,
            probabilistic_anchor_decoder=config.anchor_decoder if type(config) is PersistentProbabilisticConfig else None,
            trained_paper_method=False, paper_eligible=False,
            physical_deadline_enforced=False, test_payloads_read=False, gt_model_inputs=False,
            source_arrival_policy='scheduled_pair_snapshot_at_reference_plus_100ms',
            protocol_comparison_requires_same_scoring_context=True)
        _new(output/'receipt.json', result)
        return result
    except BaseException as error:
        if tracker is not None:
            tracker.close()
        _new(output/'failure.json', dict(kind='persistent_forest_replay_failure_v1', status='failed',
            completed_frames=frames, scheduled_frames=len(rows), error_type=type(error).__name__,
            error=str(error), plan_sha256=sha_file(output/'plan.json'),
            partial_outputs_not_final_results=True, paper_eligible=False))
        raise


def run(args):
    # This is the existing strict 3316-frame, 21-sequence, prediction-only parser.
    rows = schedule_rows(args.schedule, args.schedule_sha256)
    cache = VerifiedForestCache(args.cache, args.cache_sha256)
    manifest = json.loads(cache.manifest_json)
    if (manifest['split'] != 'val' or manifest['frame_count'] != 7189
            or set(manifest['sequences']) != {r['sequence_id'] for r in rows}):
        raise ValueError('full sealed SPD validation cache required; no test or partial cohort')
    configuration = PersistentForestConfig(state=ForestTrackingConfig(active_limit=4, expansion_budget=256))
    sources = [Path(__file__), ROOT/'tools/event_track_v2x/run_tracking_v2.py']+[
        ROOT/'transvision/models/event_track_v2x'/name for name in ('persistent_forest.py', 'persistent_cache_stream.py',
            'forest_cache_stream.py', 'forest_tracking.py', 'forest_potentials.py', 'detection_cache_v2.py',
            'identity_forest.py', 'hypothesis_bank.py', 'recoverable_states.py', 'tracking_v2.py',
            'prediction_features.py', 'fusion.py', 'arrays.py', 'forest_row_context.py',
            'persistent_component_store.py', 'persistent_component_tracking.py', 'persistent_beam_tracking.py',
            'persistent_joint_beam.py', 'persistent_class_bound_beam.py', 'persistent_slot_bound_beam.py',
            'persistent_sparse_slot_bound_beam.py',
            'persistent_reachable_slot_bound_beam.py',
            'allocation_policy.py', 'learned_component_allocation.py', 'experiment_progress.py')]
    sources += [ROOT/'transvision/models/event_track_v2x'/name for name in
                ('frontier_completion.py','completion_component_tracking.py',
                 'covered_proposal_capacity.py','covered_completion_tracking.py')]
    plan = dict(kind='persistent_forest_geometry_development_plan_v1', cache_sha256=args.cache_sha256,
        schedule_sha256=args.schedule_sha256, split_sha256=SPLIT_SHA, configuration=asdict(configuration),
        birth_logit=-4., scorer_context_recipe='persistent-forest-arrival-row-parent-context-v1',
        source_sha256={p.relative_to(ROOT).as_posix(): sha_file(p) for p in sources},
        class_scope=['car'], learned_method=False, train_selected_configuration=False,
        val_seen_during_research=True, validation_parameter_search=False, paper_eligible=False)
    # Source/config identities are captured BEFORE replay and must stay unchanged.
    result = replay_rows(cache, rows, args.output, configuration, plan=plan)
    if any(sha_file(ROOT/p) != sha for p, sha in plan['source_sha256'].items()):
        raise ValueError('runner sources changed during replay; result not frozen')
    _new(Path(args.output)/'full-validation-receipt.json', dict(result,
        full_official_validation_schedule_completed=result['completed_frames'] == 3316,
        plan_sha256=sha_file(Path(args.output)/'plan.json')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--cache-sha256', required=True)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--schedule-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({k: result[k] for k in ('status', 'completed_frames', 'geometry_development_baseline')}, sort_keys=True))


if __name__ == '__main__':
    main()
