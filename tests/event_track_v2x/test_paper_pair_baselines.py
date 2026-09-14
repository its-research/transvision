"""Actual M0--M4 code under paper bindings; no real result numbers."""
import copy
import json
import os
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest
import torch
from test_detection_cache_v2 import _build, _sources
from test_paper_pipeline import native_frame
from test_source_mask_v2 import short_sequence
from test_tracking_v2 import FixedModel, calibration, frame, tracker

from tools.event_track_v2x.run_paper_pair_baselines import ROOT, run
from transvision.models.event_track_v2x.detection_cache_v2 import SIDES, canonical, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery, VerifiedForestCache
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
from transvision.models.event_track_v2x.paper_native_cache import NativePaperCache, write_native_cache
from transvision.models.event_track_v2x.paper_pair_baselines import FEATURE_RECIPE, METHODS, load_assets, make_tracker, restore, snapshot
from transvision.models.event_track_v2x.paper_protocol import PAPER, PaperProtocol
from transvision.models.event_track_v2x.predicted_association_v2 import CALIBRATION_SHA, PredictedAssociation
from transvision.models.event_track_v2x.tracking_v2 import TrackingConfigV2, physical_world


def binding(dataset='spd', calibration_sha=CALIBRATION_SHA):
    return dict(dataset=dataset, candidate_protocol=PAPER, fit_split='train', calibration_sha256=calibration_sha, fixture=True)


@pytest.mark.parametrize('method', METHODS)
def test_pair_adapters_keep_original_algebra_and_can_reopen(method, tmp_path):
    costs = {}
    current = make_tracker(method, FixedModel(), calibration(), 'seq01', 1_000_000, protocol=PaperProtocol('spd', 'val'), binding=binding(), counters=costs)
    from transvision.models.event_track_v2x.tracking_birth_score_v2 import BirthScoreDiagnosticTrackerV2
    from transvision.models.event_track_v2x.tracking_mechanisms_v2 import MechanismDiagnosticTrackerV2
    original = (
        tracker() if method == 'learned-ci' else BirthScoreDiagnosticTrackerV2(FixedModel(), calibration(), 'seq01', 1_000_000) if method == 'M4' else MechanismDiagnosticTrackerV2(
            FixedModel(), calibration(), 'seq01', 1_000_000, mode=method))
    assert current.step.__func__.__code__ is type(original).step.__code__
    original_globals = dict(type(original).step.__globals__)
    for index, pair in enumerate(short_sequence()):
        actual, expected = current.step(*pair), original.step(*pair)
        assert [canonical(r) for r in actual] == [canonical(r) for r in expected]
        if index == 1:
            detached = snapshot(current)
            detached['binding']['model']['dataset'] = 'not-the-original'
            if detached['tracks']:
                detached['tracks'][0]['identity_hypotheses'].clear()
            assert current.paper_binding['model']['dataset'] == 'spd'
            assert all(t['identity_hypotheses'] for t in current.tracks.values())
            path = tmp_path / 'state.json'
            path.write_bytes(canonical(snapshot(current)))
            new = make_tracker(method, FixedModel(), calibration(), 'seq01', 1_000_000, protocol=PaperProtocol('spd', 'val'), binding=binding(), counters=costs)
            current = restore(new, path, sha_file(path))
    assert all(type(original).step.__globals__[key] is value for key, value in original_globals.items())
    assert costs['temporal_assignment_solves'] > 0
    if method == 'M1':
        assert costs.get('cross_assignment_solves', 0) == 0
    else:
        assert costs['cross_assignment_solves'] > 0


def test_pair_candidate_gate_is_all_class_top64():
    left = frame(states=np.tile([0., 0., 1., 2., 4., 2., 0., 0., 0.], (65, 1)), scores=[.9] * 64 + [.8], classes=[2] * 64 + [0])
    current = make_tracker('M1', FixedModel(), calibration(), 'seq01', 1_000_000, protocol=PaperProtocol('spd', 'val'), binding=binding())
    result, _, diagnostics = current.step(left, frame('infrastructure-side', states=[]))
    assert result['selected_detections'] == [64, 0]
    assert {s['class_index'] for s in diagnostics['source_records']} == {2}
    assert not current.model.inputs


@pytest.mark.parametrize('dataset', ['spd', 'v2v4real'])
def test_dual_format_pair_suite_and_independent_evaluation(tmp_path, dataset):
    if dataset == 'spd':
        sources = _sources(tmp_path, split='train', empty_vehicle=False)
        calpath = sources[1]
        cal = json.loads(calpath.read_bytes())
    else:
        calpath = tmp_path / 'calibration.json'
        cal = dict(calibration(), kind='rbf_pair_calibration_v1')
    cal.update(dataset=dataset, fit_split='train', candidate_protocol=PAPER, fixture=True)
    calpath.write_bytes(canonical(cal))
    calsha = sha_file(calpath)
    if dataset == 'spd':
        cache = VerifiedForestCache(sources[-1], _build(sources))
    else:
        root = tmp_path / 'native'
        frames = [
            native_frame(1, sequence_id=seq, side=side, agent_mask=mask, calibration_sha256=calsha, dataset_split='train') for seq in ('0003', '0007')
            for side, mask in SIDES.items()
        ]
        cache = NativePaperCache(root, write_native_cache(root, frames, split='train', producer={'fit_split': 'train'}, fixture=True))
    events, gt = [], []
    for seq in ('0003', '0007'):
        ds, metas = [], {}
        for (scene, side, fid), (raw, meta) in cache.index.items():
            if scene != seq:
                continue
            raw, meta = json.loads(raw), json.loads(meta)
            metas[side] = meta
            ds.append(dict(sequence_id=seq, side=side, frame_id=fid, arrival_us=meta['box_reference_timestamp_us'] + 100_000, frame_sha256=raw['frame_sha256']))
        vm = metas['vehicle-side']
        reference = vm['box_reference_timestamp_us']
        events.append(dict(sequence_id=seq, frame_id=vm['frame_id'], reference_us=reference, decision_us=reference + 100_000, event_id='e', deliveries=ds))
        d = CacheDelivery(**next(d for d in ds if d['side'] == 'vehicle-side'))
        f = cache.load_arrived(d, d.arrival_us)
        means, _ = physical_world(f, list(range(f.count)), reference)
        gt.append(
            dict(
                sequence_id=seq,
                frame_id=vm['frame_id'],
                box_reference_timestamp_us=reference,
                ego_translation_world=vm['lidar_to_world_translation'],
                objects=[dict(track_id='fixture-' + str(i), class_label='car', mean=v.tolist()) for i, v in enumerate(means)]))
    checkpoint_dir = tmp_path / 'checkpoint'
    checkpoint_dir.mkdir()
    # Tiny synthetic optimizer step; not a real trained baseline or a paper result.
    torch.manual_seed(1337)
    model = PredictedAssociation(hidden=4, dropout=0.)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    logits = model(torch.zeros(1, 2, 203), torch.zeros(1, 2, 203))
    loss = sum(v.square().mean() for v in logits)
    loss.backward()
    optimizer.step()
    weightpath = checkpoint_dir / 'weights.pt'
    torch.save(model.state_dict(), weightpath)
    training = dict(
        dataset=dataset, fit_split='train', candidate_protocol=PAPER, fixture=True, seed=1337, weights_sha256=sha_file(weightpath), status='fixture_optimizer_step_only')
    trainpath = checkpoint_dir / 'training.json'
    trainpath.write_bytes(canonical(training))
    manifest = dict(
        kind='rbf_pair_checkpoint_v1',
        dataset=dataset,
        fit_split='train',
        candidate_protocol=PAPER,
        feature_recipe=FEATURE_RECIPE,
        seed=1337,
        calibration_sha256=calsha,
        fixture=True,
        frozen_cache_identity=frozen_cache_identity(cache),
        architecture=dict(hidden=4, dropout=0.),
        weights=dict(path='weights.pt', sha256=sha_file(weightpath)),
        training_receipt=dict(path='training.json', sha256=sha_file(trainpath)))
    ckpath = checkpoint_dir / 'checkpoint.json'
    ckpath.write_bytes(canonical(manifest))
    protocol = PaperProtocol(dataset, 'train')
    results = {}
    for method in METHODS:
        model, loaded, seal = load_assets(
            None if method == 'M1' else ckpath, None if method == 'M1' else sha_file(ckpath), calpath, calsha, cache=cache, protocol=protocol, fixture=True, method=method)
        destination = tmp_path / method
        result = run(
            cache,
            events,
            destination,
            protocol=protocol,
            configuration=dict(method=method, tracking=asdict(TrackingConfigV2())),
            model=model,
            calibration=loaded,
            binding=seal,
            fixture=True)
        assert result['fixture'] and not result['paper_results_verified']
        assert result['completed_events'] == 2 and len(result['states']) == 2
        resources = json.loads((destination / 'resources.json').read_bytes())
        assert resources['costs']['model_forward_calls'] == (0 if method == 'M1' else 2)
        results[method] = (destination / 'association.jsonl').read_bytes()
    assert results['learned-ci'] == results['M0'] == results['M2'] == results['M3'] == results['M4']
    assert (tmp_path / 'learned-ci/predictions.jsonl').read_bytes() == (tmp_path / 'M0/predictions.jsonl').read_bytes()
    schedule_path = tmp_path / 'schedule.jsonl'
    schedule_path.write_bytes(b''.join(canonical(e) + b'\n' for e in events))
    config_path = tmp_path / 'm0-config.json'
    config_path.write_bytes(canonical(dict(method='M0', tracking=asdict(TrackingConfigV2()))))
    cli = subprocess.run([
        sys.executable,
        str(ROOT / 'tools/event_track_v2x/run_paper_pair_baselines.py'), '--cache',
        str(cache.root), '--cache-sha256', cache.manifest_sha256, '--schedule',
        str(schedule_path), '--schedule-sha256',
        sha_file(schedule_path), '--configuration',
        str(config_path), '--configuration-sha256',
        sha_file(config_path), '--dataset', dataset, '--split', 'train', '--calibration',
        str(calpath), '--calibration-sha256', calsha, '--checkpoint',
        str(ckpath), '--checkpoint-sha256',
        sha_file(ckpath), '--fixture', '--output',
        str(tmp_path / 'cli')
    ],
                         capture_output=True,
                         text=True,
                         timeout=60)
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert (tmp_path / 'cli/predictions.jsonl').read_bytes() == (tmp_path / 'M0/predictions.jsonl').read_bytes()
    changed = dict(manifest, candidate_protocol='rbf-car-first-top64-v1')
    ckpath.write_bytes(canonical(changed))
    with pytest.raises(ValueError, match='protocol/producer'):
        load_assets(ckpath, sha_file(ckpath), calpath, calsha, cache=cache, protocol=protocol, fixture=True, method='M0')
    ckpath.write_bytes(canonical(manifest))
    invalid = copy.deepcopy(events)
    invalid[0]['deliveries'][0]['arrival_us'] += 1
    with pytest.raises(ValueError, match='not arrived'):
        run(cache,
            invalid,
            tmp_path / 'late',
            protocol=protocol,
            configuration=dict(method='M0', tracking=asdict(TrackingConfigV2())),
            model=model,
            calibration=loaded,
            binding=seal,
            fixture=True)
    assert not (tmp_path / 'late').exists()
    evaluator = os.environ.get('RBF_EVALUATOR_PYTHON')
    if evaluator:
        gtpath = tmp_path / 'gt.jsonl'
        gtpath.write_bytes(b''.join(canonical(g) + b'\n' for g in gt))
        gtmanifest = tmp_path / 'gt-manifest.json'
        gtmanifest.write_bytes(
            canonical(
                dict(
                    kind='rbf_paper_evaluation_gt_v1',
                    fixture=True,
                    protocol=asdict(protocol),
                    frames=2,
                    ground_truth=dict(path='gt.jsonl', sha256=sha_file(gtpath)),
                    roi=dict(kind='strict_radial_xy', radius_m=50.),
                    sequence_clusters={
                        '0003': 'a',
                        '0007': 'b'
                    })))
        process = subprocess.run([
            evaluator,
            str(ROOT / 'tools/event_track_v2x/evaluate_paper.py'), '--gt-manifest',
            str(gtmanifest), '--gt-sha256',
            sha_file(gtmanifest), '--replay',
            str(tmp_path / 'M0'), '--receipt-sha256',
            sha_file(tmp_path / 'M0/receipt.json'), '--output',
            str(tmp_path / 'evaluation')
        ],
                                 capture_output=True,
                                 text=True,
                                 timeout=60)
        assert process.returncode == 0, process.stdout + process.stderr
        report = json.loads((tmp_path / 'evaluation/report.json').read_bytes())
        assert report['fixture'] and report['status'] == 'evaluated'
