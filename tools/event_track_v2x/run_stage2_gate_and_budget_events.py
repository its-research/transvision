#!/usr/bin/env python3
"""Finite wrong-gate / exhausted-budget experiments on a frozen CPU backend.

These are new explicit-factor model cases, not learned or real-data results.
The evaluator-only identity target is never passed to the tracker.
"""
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
    root = Path(plan['frozen_backend_root'])
    for relative, digest in plan['backend_source_sha256'].items():
        assert sha(root / relative) == digest
    assert sha(Path(__file__)) == plan['producer_sha256']
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
    assert not torch.cuda.is_initialized()
    args.output.mkdir(exist_ok=False)
    reports = []
    for seed in plan['seeds']:
        epsilon = random.Random(seed).uniform(-.001, .001)
        for case in plan['cases']:
            for width in plan['widths']:
                run_name = f'{case}-seed{seed}-K{width}'
                reused = plan.get('reused_runs', {}).get(run_name)
                if reused is not None:
                    assert sha(Path(reused['path'])) == reused['sha256']
                    original = json.loads(Path(reused['path']).read_text())
                    assert original['case'] == case and original['seed'] == seed and original['K'] == width
                    assert sha(Path(original['database_path'])) == original['database_sha256']
                    reports.append(dict(reused, run=run_name))
                    continue
                # Only the supported persistent expansion budget is set.
                # The finite-window action_budget option is unused by this
                # backend; all actual persistent decision limits stay fixed.
                budget = 1 if case == 'exhausted_budget' else 3
                state = PaperForestTrackingConfig(active_limit=width, expansion_budget=budget,
                    max_model_regret=.05, decision_mode='all-legal-hamming')
                config = PersistentExclusiveCompletionConfig(state=state)
                folder = args.output / run_name
                folder.mkdir()
                tracker = ExclusiveCompletionTracker(folder / 'tracker.sqlite', sequence_id='finite-stage2-new', config=config)
                raw = []
                for i, (name, source, x) in enumerate([('a', 0, -.2), ('b', 0, .2), ('c', 1, .19)]):
                    features = np.zeros(203)
                    features[0] = x / 100
                    features[138] = 1
                    features[200:202] = .8
                    raw.append(RawIdentityDetection('finite-stage2-new', IdentityNode(name, source, 1000000, 1000010, 'frame0'),
                        i if source == 0 else 0, 1000000, [x, 0., 1., 4., 2., 1.5, 0., 0., 0.], np.eye(9) * .2, .8, features, 'a' * 64))
                rows = [((-1, 0.),), ((-1, 0.),), ((-1, -30.), (0, epsilon))]
                if case == 'exhausted_budget':
                    rows[2] += ((1, -.1 + epsilon),)
                assert (case == 'incorrect_gate') == (1 not in dict(rows[2]))
                events = []
                for i in range(plan['events_per_run']):
                    stamp = 1100000 + i * 100
                    commit = tracker.step(tuple(raw) if i == 0 else (), tuple(rows) if i == 0 else (),
                        frame_id=f'event{i}', event_id=f'event{i}', reference_us=stamp, decision_us=stamp)
                    assert commit.audit['search_steps'] <= budget
                    events.append({'event_id': f'event{i}', 'prediction': json.loads(commit.prediction_json),
                        'audit': json.loads(commit.audit_json), 'prediction_sha256': hashlib.sha256(commit.prediction_json).hexdigest(),
                        'audit_sha256': hashlib.sha256(commit.audit_json).hexdigest()})
                first = events[0]
                duplicate = tracker.step(tuple(raw), tuple(rows), frame_id='event0', event_id='event0', reference_us=1100000, decision_us=1100000)
                assert hashlib.sha256(duplicate.prediction_json).hexdigest() == first['prediction_sha256']
                database_sha = tracker.close()
                report = {'case': case, 'seed': seed, 'K': width, 'configuration': dataclasses.asdict(config),
                    'raw': [dataclasses.asdict(r) for r in raw], 'rows': rows, 'events': events,
                    'database_path': str(folder / 'tracker.sqlite'), 'database_sha256': database_sha,
                    'sealed_first_prediction_byte_equal': True, 'evaluator_only_true_roots': ['a', 'b', 'b'],
                    'truth_read_for_inference': False, 'full_Stage2_complete': False}
                (folder / 'candidate.json').write_bytes(canonical(report) + b'\n')
                reports.append({'run': folder.name, 'path': str(folder / 'candidate.json'), 'sha256': sha(folder / 'candidate.json')})
                print(json.dumps({'completed_new_runs': len(reports), 'total': len(plan['seeds']) * len(plan['cases']) * len(plan['widths'])}), flush=True)
    for relative, digest in plan['backend_source_sha256'].items():
        assert sha(root / relative) == digest
    assert not torch.cuda.is_initialized()
    receipt = {'kind': 'rbf_stage2_new_gate_and_budget_cpu_candidate_campaign_v1', 'runs': reports,
        'source_plan_sha256': sha(args.plan), 'source_sha256': sha(Path(__file__)), 'CUDA_initialized': False,
        'independent_admission': False, 'learned_priority_executed': False, 'isolated_resource_measurement': False,
        'full_Stage2_complete': False, 'goal_status': 'active'}
    (args.output / 'campaign-receipt.json').write_bytes(canonical(receipt) + b'\n')


if __name__ == '__main__':
    main()
