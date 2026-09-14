#!/usr/bin/env python3
"""Complete TRAIN sequence for the fixed single-history JPDA/PKF adapters.

Uses the same verified schedule, learned raw factors and V2 replay as recovery.
State/time handling is explicitly different; this is not a recovery-only
ablation, a public-method reproduction, parameter training or validation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.event_track_v2x.collect_allocation_training import verified_train_rows, select_teacher_schedule
from tools.event_track_v2x.prepare_forest_training import TRAIN_CACHE_SHA256
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows, _new
from tools.event_track_v2x.run_train_inference_diagnostic import diagnostic_sources
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
from transvision.models.event_track_v2x.resource_sweep import configuration, THREAD_ENV, validate_execution_modes

BACKENDS = ('jpda_ci', 'jpda_kalman', 'pkf')
PAIR_SHA256 = '787b7e4dfa3fb9d97ba0a6a4a216a2990af990dc9226ceec01053e7d2e7da984'
PRODUCER = 'train_probabilistic_diagnostic_v1'


def sources():
    result = diagnostic_sources()
    result[Path(__file__).relative_to(ROOT).as_posix()] = sha_file(__file__)
    return result


def run(cache, cooperative_metadata, metadata_sha256, checkpoint, checkpoint_sha256,
        output, *, sequence, backend, allow_fixture=False):
    if type(allow_fixture) is not bool or backend not in BACKENDS:
        raise ValueError('known fixed probabilistic backend and explicit fixture mode required')
    if not isinstance(sequence, str) or not sequence:
        raise ValueError('one complete train sequence required')
    if not allow_fixture and metadata_sha256 != PAIR_SHA256:
        raise ValueError('sealed full official train cooperative metadata required')
    if any(os.environ.get(k) != v for k, v in THREAD_ENV.items()):
        raise ValueError('fresh-process thread environment is not explicitly pinned')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    bound = sources()
    rows = verified_train_rows(cache, cooperative_metadata, metadata_sha256,
                               require_full_train=not allow_fixture)
    rows, _ = select_teacher_schedule(rows, sequence)
    config = configuration(dict(backend=backend))
    scorer, metadata = load_identity_checkpoint(checkpoint, checkpoint_sha256,
                                                config=config.state, device='cpu')
    if (not allow_fixture and metadata.get('full_official_train') is not True
            or metadata['frozen_cache_identity'] != frozen_cache_identity(cache)):
        raise ValueError('matching frozen full-train identity checkpoint required')
    plan = dict(kind='train_sequence_inference_diagnostic_plan_v1', producer=PRODUCER,
        backend=backend, configuration=asdict(config), source_sha256=bound,
        source_scope='model_modules_initializers_and_declared_replay_tool_dependencies',
        cache_sha256=cache.manifest_sha256, cooperative_metadata_sha256=metadata_sha256,
        identity_checkpoint_sha256=checkpoint_sha256, identity_seed=metadata['seed'],
        scorer_signature=scorer.signature, cohort_mode='fixture-only' if allow_fixture else 'real-train-development',
        complete_input_train_cohort_verified=not allow_fixture, selected_sequence=sequence,
        selected_schedule=rows, scheduled_frames=len(rows), class_scope=['car'],
        full_official_train_trace_completed=False, upstream_in_sample=True,
        strict_pipeline_isolated_selection=False, real_tracking_evaluation=False,
        reproduced_public_method=False, irreversible_identity_anchors=True,
        raw_row_context_shared_with_recoverable=True, same_state_time_protocol_as_recoverable=False,
        state_policy='project_raw_measurements_to_output_reference_then_sequential_scan_updates',
        unmatched_mass_scales_detection_score=False, recovery_only_ablation=False,
        physically_equal_resources_claimed=False, three_seed_comparison=False, paper_eligible=False,
        thread_environment={k: os.environ[k] for k in THREAD_ENV},
        runtime=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                     host=platform.node(), pid=os.getpid(), torch=torch.__version__))
    output = Path(output)
    result = replay_rows(cache, rows, output, config, plan=plan, learned_scorer=scorer)
    validate_execution_modes(result, dict(backend=backend, configuration=asdict(config)))
    if (sources() != bound or sha_file(cooperative_metadata) != metadata_sha256
            or sha_file(Path(checkpoint)/'checkpoint.json') != checkpoint_sha256):
        _new(output/'diagnostic-failure.json', dict(status='failed', reason='source_or_input_changed',
            partial_outputs_not_final_results=True, paper_eligible=False))
        raise ValueError('probabilistic diagnostic source or input changed; no final receipt')
    final = dict(kind='train_sequence_inference_diagnostic_v1', producer=PRODUCER,
        status='complete', backend=backend, replay_receipt_sha256=sha_file(output/'receipt.json'),
        plan_sha256=sha_file(output/'plan.json'), completed_frames=result['completed_frames'],
        cohort_mode=plan['cohort_mode'], selected_sequence=sequence,
        complete_selected_sequence_verified=not allow_fixture,
        full_official_train_trace_completed=False, training_performed=False,
        offline_teacher_probes=False, real_tracking_evaluation=False,
        reproduced_public_method=False, same_state_time_protocol_as_recoverable=False,
        recovery_only_ablation=False, paper_eligible=False)
    _new(output/'development-inference-receipt.json', final)
    print(json.dumps(final, sort_keys=True))
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'cooperative-metadata', 'checkpoint', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('cooperative-metadata-sha256', 'checkpoint-sha256', 'sequence'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--backend', choices=BACKENDS, required=True)
    args = parser.parse_args()
    cache = VerifiedForestCache(args.cache, TRAIN_CACHE_SHA256)
    run(cache, args.cooperative_metadata, args.cooperative_metadata_sha256,
        args.checkpoint, args.checkpoint_sha256, args.output,
        sequence=args.sequence, backend=args.backend)


if __name__ == '__main__':
    main()
