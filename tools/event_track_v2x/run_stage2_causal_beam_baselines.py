#!/usr/bin/env python3
"""Same-input irreversible beam experiments, not same-resource or MHT claims."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import dataclasses
import hashlib
import json
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
        mod = types.ModuleType(name); mod.__path__ = [str(root.joinpath(*name.split('.')))]; sys.modules[name] = mod
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from transvision.models.event_track_v2x.forest_tracking import RawIdentityDetection, PaperForestTrackingConfig
    from transvision.models.event_track_v2x.identity_forest import IdentityNode
    from transvision.models.event_track_v2x.persistent_beam_tracking import PersistentBeamTracker, PersistentBeamConfig, PersistentRankedBeamTracker, PersistentRankedBeamConfig
    from transvision.models.event_track_v2x.persistent_joint_beam import PersistentJointBeamTracker, PersistentJointBeamConfig
    backends = dict(node_beam=(PersistentBeamTracker, PersistentBeamConfig),
        batch_beam=(PersistentRankedBeamTracker, PersistentRankedBeamConfig),
        joint_beam=(PersistentJointBeamTracker, PersistentJointBeamConfig))
    args.output.mkdir(exist_ok=False); reports = []
    for incoming in plan['inputs']:
        p = Path(incoming['path']); assert sha(p) == incoming['sha256']
        data = json.loads(p.read_text())
        assert set(data) == {'sequence_id', 'raw', 'rows', 'schedule'}
        raw = []
        for item in data['raw']:
            item = dict(item); item['node'] = IdentityNode(**item['node']); raw.append(RawIdentityDetection(**item))
        rows = [tuple(tuple(v) for v in row) for row in data['rows']]
        for method in plan['methods']:
            Tracker, Config = backends[method]
            for width in plan['widths']:
                name = incoming['id'] + f'-{method}-K{width}'
                folder = args.output / name; folder.mkdir()
                # These backends reject recoverable expansion options. Their
                # independent default work limits remain fixed in the plan.
                config = Config(state=PaperForestTrackingConfig(active_limit=width))
                assert dataclasses.asdict(config) == plan['configurations'][method][str(width)]
                tracker = Tracker(folder / 'tracker.sqlite', sequence_id=data['sequence_id'], config=config)
                events = []; ingested = 0
                for event in data['schedule']:
                    assert set(event) == {'event_id', 'frame_id', 'reference_us', 'decision_us', 'available_observations'}
                    n = event['available_observations']
                    commit = tracker.step(tuple(raw[ingested:n]), tuple(rows[ingested:n]),
                        **{k: event[k] for k in ['event_id', 'frame_id', 'reference_us', 'decision_us']})
                    assert not commit.audit['recovery_enabled'] and not commit.audit['archived_prefixes_used_for_recovery']
                    events.append(dict(event_id=event['event_id'], available_observations=n,
                        prediction=json.loads(commit.prediction_json), audit=json.loads(commit.audit_json),
                        prediction_sha256=hashlib.sha256(commit.prediction_json).hexdigest(),
                        audit_sha256=hashlib.sha256(commit.audit_json).hexdigest()))
                    ingested = n
                first = data['schedule'][0]; n = first['available_observations']
                duplicate = tracker.step(tuple(raw[:n]), tuple(rows[:n]),
                    **{k: first[k] for k in ['event_id', 'frame_id', 'reference_us', 'decision_us']})
                assert hashlib.sha256(duplicate.prediction_json).hexdigest() == events[0]['prediction_sha256']
                assert hashlib.sha256(duplicate.audit_json).hexdigest() == events[0]['audit_sha256']
                database_sha = tracker.close()
                report = dict(method=method, width=width, input_id=incoming['id'], input_path=str(p), input_sha256=incoming['sha256'],
                    configuration=dataclasses.asdict(config), events=events, database_path=str(folder/'tracker.sqlite'),
                    database_sha256=database_sha, original_prediction_and_audit_byte_immutable=True,
                    inference_read_evaluator_truth_or_RBF_output=False, same_resource_measurement=False,
                    reproduced_classical_MHT=False, full_Stage2_complete=False)
                (folder/'candidate.json').write_bytes(canonical(report)+b'\n')
                reports.append(dict(run=name,path=str(folder/'candidate.json'),sha256=sha(folder/'candidate.json')))
                print(json.dumps({'new_closed_beam_runs':len(reports),'total':len(plan['inputs'])*len(plan['methods'])*len(plan['widths'])}),flush=True)
    for relative,digest in plan['backend_source_sha256'].items():assert sha(root/relative)==digest
    assert not torch.cuda.is_initialized()
    receipt=dict(kind='rbf_stage2_causal_same_input_irreversible_beam_candidates_v1',runs=reports,
        source_plan_sha256=sha(args.plan),source_sha256=sha(Path(__file__)),CUDA_initialized=False,
        independent_admission=False,same_resource_comparison=False,full_Stage2_complete=False,goal_status='active')
    (args.output/'campaign-receipt.json').write_bytes(canonical(receipt)+b'\n')


if __name__=='__main__':
    main()
