#!/usr/bin/env python3
"""Predeclared CPU event trajectories; evaluator identities never enter inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import dataclasses
import hashlib
import json
import random
import sys
import types
from pathlib import Path


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    assert sha(Path(__file__)) == plan['producer_sha256']
    root = Path(plan['frozen_backend_root'])
    for relative, digest in plan['backend_source_sha256'].items():
        assert sha(root / relative) == digest
    for name in ['transvision', 'transvision.models', 'transvision.models.event_track_v2x']:
        mod = types.ModuleType(name)
        mod.__path__ = [str(root.joinpath(*name.split('.')))]
        sys.modules[name] = mod
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from transvision.models.event_track_v2x.forest_tracking import RawIdentityDetection, PaperForestTrackingConfig
    from transvision.models.event_track_v2x.identity_forest import IdentityNode
    from transvision.models.event_track_v2x.exclusive_completion_tracking import ExclusiveCompletionTracker, PersistentExclusiveCompletionConfig
    args.output.mkdir(exist_ok=False)
    reports = []
    for seed in plan['seeds']:
        epsilon = random.Random(seed).uniform(-.001, .001)
        for case in plan['cases']:
            for width in plan['widths']:
                folder = args.output / f'{case}-seed{seed}-K{width}'
                folder.mkdir()
                config = PersistentExclusiveCompletionConfig(state=PaperForestTrackingConfig(
                    active_limit=width, expansion_budget=plan['expansion_budget'],
                    max_model_regret=.05, decision_mode='all-legal-hamming'))
                sequence = 'finite-stage2-cross-occlusion'
                tracker = ExclusiveCompletionTracker(folder / 'tracker.sqlite', sequence_id=sequence, config=config)
                def obs(name, source, frame, stamp, index, x, y, vx, arrival):
                    # Identical appearance descriptor; valid full covariance with
                    # cross terms exercises unequal-time transport and CI.
                    features = np.zeros(203)
                    features[138] = 1.; features[200:202] = .8
                    P = np.eye(9) * .2
                    P[0, 7] = P[7, 0] = .025
                    P[1, 8] = P[8, 1] = -.015
                    return RawIdentityDetection(sequence, IdentityNode(name, source, stamp, arrival, frame),
                        index, stamp, [x, y, 1., 4., 2., 1.5, 0., vx, 0.], P, .8, features, 'a' * 64)
                raw = [obs('a', 0, 'initial', 1000000, 0, -.2, -2., 1.5, 1000010),
                       obs('b', 0, 'initial', 1000000, 1, .2, 2., -1.5, 1000010),
                       obs('c', 1, 'initial', 1000000, 0, .19, 2., -1.5, 1000010)]
                rows = [((-1, 0.),), ((-1, 0.),), ((-1, -30.), (0, epsilon), (1, -10.))]
                schedule = [(3, 1100000, 'initial')]
                if case == 'occlusion_reappearance':
                    raw.append(obs('d', 0, 'occluded', 1200000, 0, .1, -2., 1.5, 1299990))
                    rows.append(((-1, -30.), (0, 0.)))
                    schedule.append((4, 1300000, 'one-object-visible'))
                    parent_a = 3
                else:
                    parent_a = 0
                ia = len(raw)
                raw.extend([obs('e', 0, 'after-crossing', 1400000, 0, .4, -2., 1.5, 1499990),
                            obs('f', 0, 'after-crossing', 1400000, 1, -.4, 2., -1.5, 1499990)])
                rows.extend([((-1, -30.), (parent_a, 0.)), ((-1, -30.), (2, 0.))])
                schedule.append((len(raw), 1500000, 'both-visible-after-crossing'))
                schedule.extend((len(raw), 1500100 + i * 100, f'compute-{i}') for i in range(plan['compute_pulses']))
                events = []; ingested = 0
                for available, stamp, name in schedule:
                    commit = tracker.step(tuple(raw[ingested:available]), tuple(rows[ingested:available]),
                        frame_id=name, event_id=name, reference_us=stamp, decision_us=stamp)
                    assert commit.audit['search_steps'] <= plan['expansion_budget']
                    events.append(dict(event_id=name, available_observations=available,
                        prediction=json.loads(commit.prediction_json), audit=json.loads(commit.audit_json),
                        prediction_sha256=hashlib.sha256(commit.prediction_json).hexdigest(),
                        audit_sha256=hashlib.sha256(commit.audit_json).hexdigest()))
                    ingested = available
                first = events[0]
                duplicate = tracker.step(tuple(raw[:3]), tuple(rows[:3]), frame_id='initial', event_id='initial', reference_us=1100000, decision_us=1100000)
                assert hashlib.sha256(duplicate.prediction_json).hexdigest() == first['prediction_sha256']
                assert hashlib.sha256(duplicate.audit_json).hexdigest() == first['audit_sha256']
                database_sha = tracker.close()
                truth = ['a', 'b', 'b'] + (['a'] if case == 'occlusion_reappearance' else []) + ['a', 'b']
                report = dict(case=case, seed=seed, K=width, sequence_id=sequence, configuration=dataclasses.asdict(config),
                    raw=[dataclasses.asdict(r) for r in raw], rows=rows, events=events,
                    database_path=str(folder / 'tracker.sqlite'), database_sha256=database_sha,
                    evaluator_only_true_roots=truth, truth_read_for_inference=False,
                    sealed_first_prediction_and_audit_byte_equal=True,
                    geometry='opposite-direction cars on distinct y=-2/+2 lanes swap x order; no physical intersection asserted',
                    absence_evidence='no observation of B at 1.2s; no existence-likelihood factor inserted',
                    finite_synthetic_factors=True, learned_priority_executed=False, full_Stage2_complete=False)
                (folder / 'candidate.json').write_bytes(canonical(report) + b'\n')
                reports.append(dict(run=folder.name, path=str(folder / 'candidate.json'), sha256=sha(folder / 'candidate.json')))
                print(json.dumps({'completed_new_runs': len(reports), 'total': len(plan['seeds']) * len(plan['cases']) * len(plan['widths'])}), flush=True)
    for relative, digest in plan['backend_source_sha256'].items():
        assert sha(root / relative) == digest
    assert not torch.cuda.is_initialized()
    receipt = dict(kind='rbf_stage2_crossing_occlusion_candidate_campaign_v1', runs=reports,
        source_plan_sha256=sha(args.plan), source_sha256=sha(Path(__file__)), CUDA_initialized=False,
        independent_admission=False, learned_priority_executed=False, isolated_resource_measurement=False,
        full_Stage2_complete=False, goal_status='active')
    (args.output / 'campaign-receipt.json').write_bytes(canonical(receipt) + b'\n')


if __name__ == '__main__':
    main()
