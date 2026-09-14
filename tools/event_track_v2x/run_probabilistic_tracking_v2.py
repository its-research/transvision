#!/usr/bin/env python3
"""Frozen full-SPD-val car replay for single-history JPDA/PKF adapters.

This is a declared cooperative adaptation, not public-method reproduction.
Association solver, state updater, anchor decoder and resource limits are
explicit before replay. No validation selection, GT/test reads or publication.
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

from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows, _new
from tools.event_track_v2x.run_tracking_v2 import schedule_rows, SPLIT_SHA
from tools.event_track_v2x.train_forest_identity import training_sources
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
from transvision.models.event_track_v2x.jpda_marginals import JPDALimits
from transvision.models.event_track_v2x.persistent_probabilistic_tracking import PersistentProbabilisticConfig


THREAD_ENV = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', BLIS_NUM_THREADS='1',
    PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')


def validation_sources():
    # replay_rows imports several backends, including lazy allocation helpers.
    # Seal the whole model package, not only the selected backend's top file.
    paths = [ROOT/'transvision/__init__.py', ROOT/'transvision/models/__init__.py',
             ROOT/'transvision/register.py', ROOT/'transvision/version.py',
             Path(__file__), ROOT/'tools/event_track_v2x/run_persistent_forest_v2.py',
             ROOT/'tools/event_track_v2x/run_tracking_v2.py']
    paths.extend(sorted((ROOT/'transvision/models/event_track_v2x').glob('*.py')))
    return dict(training_sources(), **{p.relative_to(ROOT).as_posix(): sha_file(p) for p in paths})


def runtime_evidence(device):
    import numpy as np
    import torch
    if device != 'cpu':
        raise ValueError('this validation run requires explicit CPU inference; not GPU parameter training')
    if any(os.environ.get(key) != value for key, value in THREAD_ENV.items()):
        raise ValueError('fresh-process thread environment must be pinned before Python starts')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    return dict(python=sys.version, executable=sys.executable,
        executable_sha256=sha_file(Path(sys.executable).resolve()), platform=platform.platform(),
        host=platform.node(), pid=os.getpid(), torch=str(torch.__version__), numpy=np.__version__,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        thread_environment={key: os.environ[key] for key in THREAD_ENV}, device=device,
        exclusive_host=False, same_latency_or_memory_verified=False)


def input_files(args, checkpoint):
    paths = [Path(args.cache)/'manifest.json', Path(args.schedule)]
    if checkpoint is not None:
        paths.extend((Path(args.checkpoint)/'checkpoint.json',
                      Path(args.checkpoint)/checkpoint['weights']['path']))
    return {str(p.absolute()): sha_file(p) for p in paths}


def require_unchanged(sources, inputs):
    if validation_sources() != sources or any(sha_file(Path(p)) != h for p, h in inputs.items()):
        raise ValueError('validation sources or input files changed during replay')


def run(args):
    defaults = JPDALimits()
    inference = JPDALimits(**{name: getattr(args, name, getattr(defaults, name))
                              for name in JPDALimits.__dataclass_fields__})
    config = PersistentProbabilisticConfig(
        association_algorithm=getattr(args, 'association', 'lbp'),
        update_rule=getattr(args, 'update_rule', 'jpda-ci'),
        anchor_decoder=getattr(args, 'anchor_decoder', 'joint-map'), inference=inference,
        max_scan_tracks=getattr(args, 'max_scan_tracks', 2048))
    if (args.checkpoint is None) != (args.checkpoint_sha256 is None):
        raise ValueError('checkpoint path and hash must be supplied together')
    scorer, checkpoint = None, None
    if args.checkpoint is not None:
        scorer, checkpoint = load_identity_checkpoint(args.checkpoint, args.checkpoint_sha256,
                                                      config=config.state, device=args.device)
        if checkpoint.get('full_official_train') is not True:
            raise ValueError('fixture checkpoint cannot represent full-train probabilistic validation')
    rows = schedule_rows(args.schedule, args.schedule_sha256)
    cache = VerifiedForestCache(args.cache, args.cache_sha256)
    manifest = json.loads(cache.manifest_json)
    if (manifest['split'] != 'val' or manifest['frame_count'] != 7189
            or set(manifest['sequences']) != {r['sequence_id'] for r in rows}):
        raise ValueError('full sealed SPD validation cache required')
    if checkpoint is not None and frozen_cache_identity(cache) != checkpoint['frozen_cache_identity']:
        raise ValueError('validation upstream differs from frozen training producers')
    runtime = runtime_evidence(args.device)
    sources, inputs = validation_sources(), input_files(args, checkpoint)
    # Bind already-loaded content, not merely a fresh hash of a replaced file.
    if (inputs[str((Path(args.cache)/'manifest.json').absolute())] != args.cache_sha256
            or inputs[str(Path(args.schedule).absolute())] != args.schedule_sha256
            or checkpoint is not None and (
                inputs[str((Path(args.checkpoint)/'checkpoint.json').absolute())] != args.checkpoint_sha256
                or inputs[str((Path(args.checkpoint)/checkpoint['weights']['path']).absolute())]
                    != checkpoint['weights']['sha256'])):
        raise ValueError('validation inputs changed during preflight')
    plan = dict(kind='probabilistic_single_history_spd_val_adaptation_plan_v1',
        cache_sha256=args.cache_sha256, schedule_sha256=args.schedule_sha256, split_sha256=SPLIT_SHA,
        checkpoint_sha256=args.checkpoint_sha256, scorer_signature=None if scorer is None else scorer.signature,
        checkpoint_seed=None if checkpoint is None else checkpoint['seed'], geometry_baseline=scorer is None,
        configuration=asdict(config), source_sha256=sources, class_scope=['car'],
        source_scope='all_model_modules_initializers_and_declared_replay_tools',
        input_file_sha256=inputs, runtime=runtime, scheduled_frames=len(rows),
        reproduced_public_method=False, irreversible_identity_anchors=True,
        raw_row_context_shared_with_recoverable=True, same_state_time_protocol_as_recoverable=False,
        state_policy='project_raw_measurements_to_output_reference_then_sequential_scan_updates',
        same_latency_or_memory_verified=False, unmatched_mass_scales_detection_score=False,
        val_seen_during_research=True, validation_parameter_search=False, validation_checkpoint_selection=False,
        strict_pipeline_isolated_selection=False, learned_allocation=False, paper_eligible=False)
    result = replay_rows(cache, rows, args.output, config, plan=plan, learned_scorer=scorer)
    require_unchanged(sources, inputs)
    if (result['status'] != 'complete' or result['completed_frames'] != 3316
            or result['scheduled_frames'] != 3316 or len(result['sequence_heads']) != 21):
        raise ValueError('complete 21-sequence 3316-frame validation required for final receipt')
    _new(args.output/'full-validation-receipt.json', dict(result,
        full_official_validation_schedule_completed=True,
        checkpoint_sha256=args.checkpoint_sha256, association_algorithm=config.association_algorithm,
        update_rule=config.update_rule, anchor_decoder=config.anchor_decoder))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'schedule', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    for key in ('cache-sha256', 'schedule-sha256'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--association', choices=('exact', 'lbp'), default='lbp')
    parser.add_argument('--update-rule', choices=('jpda-ci', 'jpda-kalman', 'pkf'), default='jpda-ci')
    parser.add_argument('--anchor-decoder', choices=('joint-map', 'marginal-bayes'), default='joint-map')
    parser.add_argument('--max-scan-tracks', type=int, default=2048)
    for name, field in JPDALimits.__dataclass_fields__.items():
        parser.add_argument('--'+name.replace('_', '-'), type=float if name == 'tolerance' else int, default=field.default)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == '__main__':
    main()
