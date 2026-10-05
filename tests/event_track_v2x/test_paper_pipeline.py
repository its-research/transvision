import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_detection_cache_v2 import _frame

from transvision.models.event_track_v2x.detection_cache_v2 import ARRAYS, canonical, sha_file
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
from transvision.models.event_track_v2x.paper_native_cache import NATIVE_FEATURE, NO_IMAGE, NativeDetectionFrame, NativePaperCache, write_native_cache
from transvision.models.event_track_v2x.paper_protocol import PAPER, PaperProtocol
from transvision.models.event_track_v2x.paper_runtime import default_configuration, replay


def native_frame(n=2, **changes):
    base = _frame(n)
    meta = dict(base.metadata, dataset_split='official_test', feature_method=NATIVE_FEATURE, image_sha256=NO_IMAGE, source_image_timestamp_us=1_000_000)
    meta.update(changes)
    return NativeDetectionFrame(canonical(meta), **{k: getattr(base, k) for k in ARRAYS})


def test_native_cache_official_test_replay_and_gt_rejection(tmp_path):
    cache_path = tmp_path / 'native'
    sha = write_native_cache(cache_path, [native_frame()], split='official_test', producer={'fit_split': 'train'}, fixture=True)
    cache = NativePaperCache(cache_path, sha)
    entry = json.loads(next(iter(cache.index.values()))[0])
    event = dict(
        sequence_id='0003',
        frame_id='out',
        reference_us=1_100_000,
        decision_us=1_100_000,
        event_id='a',
        deliveries=[dict(sequence_id='0003', frame_id='000123', side='vehicle-side', arrival_us=1_100_000, frame_sha256=entry['frame_sha256'])])
    result = replay(
        cache, [event],
        tmp_path / 'replay',
        protocol=PaperProtocol('v2v4real', 'official_test'),
        configuration=default_configuration('geometry'),
        scorer=GeometryForestScorer(),
        model_binding={},
        fixture=True)
    assert result['fixture'] and result['completed_events'] == 1
    with pytest.raises(ValueError):
        NativeDetectionFrame(canonical(dict(native_frame().metadata, gt_ids=[1])), **{k: getattr(native_frame(), k) for k in ARRAYS})


def test_pointpillar_zero_regression_nms_features_and_physical_roundtrip():
    from transvision.models.event_track_v2x.paper_calibration import ExistenceCalibration
    from transvision.models.event_track_v2x.paper_pointpillar import decode_heads, native_arrays, rotated_nms
    anchors = torch.tensor([[[[1., 2., 3., 1., 2., 4., 0.], [1., 2., 3., 1., 2., 4., 0.]]]])
    scores = torch.tensor([[[[2.]], [[1.]]]])
    regression = torch.zeros(1, 14, 1, 1)
    boxes, p = decode_heads(scores, regression, anchors)
    torch.testing.assert_close(boxes, anchors.reshape(-1, 7))
    assert rotated_nms(boxes.numpy(), p.numpy()).tolist() == [0]
    arrays = native_arrays(
        scores,
        regression,
        anchors,
        torch.ones(1, 256, 2, 2),
        lidar_range=[0., 0., -5., 4., 4., 5.],
        covariance_diagonal=np.arange(1, 10),
        calibration=ExistenceCalibration(1., 0., 'train', ('train1', )))
    frame = NativeDetectionFrame(native_frame(1).metadata_json, **arrays)
    from transvision.models.event_track_v2x.forest_tracking import cache_detections
    raw = cache_detections(frame, arrival_us=1_000_000, decision_us=1_000_000, origin_us=0, candidate_protocol=PAPER)[0]
    np.testing.assert_allclose(raw.mean[:6], [2, 4, 6, 4, 2, 1])
    assert raw.mean[6] == pytest.approx(0)
    np.testing.assert_allclose(np.diag(raw.covariance), np.arange(1, 10))
    assert arrays['appearance_valid'].all()
    np.testing.assert_allclose(np.linalg.norm(arrays['appearance'], axis=1), 1.)


def test_calibration_and_probability_diagnostics_do_not_claim_joint_certificate():
    from transvision.models.event_track_v2x.paper_calibration import binary_diagnostics, fit_existence
    fit = fit_existence([.1, .3, .6, .9], [0, 1, 0, 1], protocol=PaperProtocol('spd', 'train'), groups=['a', 'a', 'b', 'b'], fit_groups=['a', 'b'])
    calibrated = fit.apply([.1, .3, .6, .9])
    assert np.all(np.diff(calibrated) >= 0)
    report = binary_diagnostics(calibrated, [0, 1, 0, 1])
    assert not report['joint_probability_certificate'] and sum(r['count'] for r in report['bins']) == 4
    with pytest.raises(ValueError):
        fit_existence([.1, .9], [0, 1], protocol=PaperProtocol('spd', 'val'), groups=['a', 'b'], fit_groups=['a', 'b'])


def test_both_identity_loss_terms_produce_gradients():
    from transvision.models.event_track_v2x.forest_row_context import ForestRowContext
    from transvision.models.event_track_v2x.forest_supervision import RowSupervision
    from transvision.models.event_track_v2x.forest_tracking import cache_detections
    from transvision.models.event_track_v2x.paper_calibration import separate_identity_losses
    a = cache_detections(_frame(1), arrival_us=1_100_000, decision_us=1_100_000, origin_us=0)[0]
    b = replace(a, node=replace(a.node, node_id='b', source_id=1))
    c = replace(a, node=replace(a.node, node_id='c', frame_id='later'))
    context = ForestRowContext((0, 1, 2), (a, b, c), 1_100_000)
    logits = torch.tensor([.2, .3, .4], requires_grad=True)
    target = RowSupervision((True, False, False), (True, True, True), 'birth')
    result = separate_identity_losses([logits], [context], [target])
    assert result['counts'] == dict(cross_source=1, temporal=1)
    for name in ('cross_source', 'temporal'):
        gradient = torch.autograd.grad(result[name], logits, retain_graph=True)[0]
        assert torch.linalg.vector_norm(gradient) > 0


def _paper_ddp_worker(rank, root, rendezvous):
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist

    from tools.event_track_v2x.train_forest_identity import FitConfig
    from tools.event_track_v2x.train_paper_identity import fit
    root = Path(root)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        fit(root / 'rows',
            sha_file(root / 'rows/manifest.json'),
            root / 'ddp-fit',
            protocol=PaperProtocol('spd', 'train'),
            groups={
                'a': 'recording-a',
                'b': 'recording-b'
            },
            config=FitConfig(epochs=1, batch_size=3, hidden=8, heads=2, dropout=0.),
            seeds=(1337, ),
            fixture=True)
    finally:
        dist.destroy_process_group()


def test_sequence_holdout_training_freeze_and_reload(tmp_path):
    from tools.event_track_v2x.train_forest_identity import FitConfig
    from tools.event_track_v2x.train_paper_identity import fit
    from transvision.models.event_track_v2x.forest_row_context import ForestRowContext
    from transvision.models.event_track_v2x.forest_supervision import RowSupervision
    from transvision.models.event_track_v2x.forest_tracking import PaperForestTrackingConfig as ForestTrackingConfig
    from transvision.models.event_track_v2x.forest_tracking import cache_detections
    from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
    from transvision.models.event_track_v2x.forest_training_data import DATA_KIND, row_protocol, write_shard
    config = ForestTrackingConfig(candidate_protocol=PAPER)
    data = tmp_path / 'rows'
    data.mkdir()
    records = []
    for sequence in ('a', 'b'):
        base = cache_detections(_frame(1), arrival_us=1_100_000, decision_us=1_100_000, origin_us=0, candidate_protocol=PAPER)[0]
        raw = tuple(replace(base, sequence_id=sequence, node=replace(base.node, node_id=sequence + str(i), source_id=i % 2, frame_id=str(i))) for i in range(3))
        contexts = tuple(ForestRowContext(tuple(range(i + 1)), raw[:i + 1], 1_100_000) for i in range(3))
        targets = tuple(RowSupervision((True, ) + (False, ) * i, (True, ) * (i + 1), 'birth') for i in range(3))
        records.append(write_shard(data / (sequence + '.npz'), sequence, raw, contexts, targets, config.parent_limit))
    manifest = dict(
        kind=DATA_KIND,
        split='train',
        sequences=['a', 'b'],
        shards=records,
        row_protocol=row_protocol(config),
        labels_in_model_inputs=False,
        provenance={'fixture_only': True},
        frozen_cache_identity={})
    (data / 'manifest.json').write_bytes(canonical(manifest))
    output = tmp_path / 'training'
    result = fit(
        data,
        sha_file(data / 'manifest.json'),
        output,
        protocol=PaperProtocol('spd', 'train'),
        groups={
            'a': 'recording-a',
            'b': 'recording-b'
        },
        config=FitConfig(epochs=2, batch_size=3, hidden=8, heads=2),
        seeds=(1337, ),
        fixture=True)
    assert result['fixture'] and not result['three_seed_training_complete']
    # Class interpretation is a transitive protocol dependency, not merely a
    # display label. Persist its bytes with the fitted checkpoint provenance.
    policy_source = 'transvision/models/event_track_v2x/paper_evaluation_policy.py'
    plan = json.loads((output / 'plan.json').read_bytes())
    assert plan['source_sha256'][policy_source] == sha_file(Path(__file__).resolve().parents[2] / policy_source)
    scorer, checkpoint = load_identity_checkpoint(output / 'seed-1337', result['seeds'][0]['sha256'], config=config)
    assert set(checkpoint['partition']['fit']).isdisjoint(checkpoint['partition']['holdout'])
    assert checkpoint['model_sha256'] != checkpoint['initial_model_sha256']
    assert not scorer.model.training and checkpoint['selected_epoch'] in (1, 2)
    import torch.multiprocessing as mp
    ddp_output = tmp_path / 'ddp-fit'
    mp.spawn(_paper_ddp_worker, args=(str(tmp_path), str(tmp_path / 'rendezvous')), nprocs=2, join=True)
    assert json.loads((ddp_output / 'plan.json').read_bytes())['world_size'] == 2
    assert json.loads((ddp_output / 'receipt.json').read_bytes())['status'] == 'software_training_completed'


def test_exact_joint_loss_sums_aliases_and_is_differentiable():
    from transvision.models.event_track_v2x.identity_forest import IdentityNode
    from transvision.models.event_track_v2x.paper_exact_loss import exact_forest_nll
    nodes = tuple(IdentityNode(str(i), i % 2, i, i, str(i)) for i in range(3))
    support = ((-1, ), (-1, 0), (-1, 0, 1))
    logits = [torch.zeros(len(row), requires_grad=True) for row in support]
    result = exact_forest_nll(nodes, support, logits, (0, 0, 0))
    assert result['legal_histories'] == 6 and result['target_parent_aliases'] == 2
    assert result['loss'].item() == pytest.approx(np.log(3))
    result['loss'].backward()
    assert logits[-1].grad.abs().sum() > 0
    with pytest.raises(ValueError, match='limit'):
        exact_forest_nll(nodes, support, logits, (0, 0, 0), maximum_histories=2)


def test_main_supervision_keeps_non_car_and_history_ablation_changes_forward():
    from transvision.models.event_track_v2x.forest_supervision import AnnotationIdentity, AnnotationIdentityIndex
    a = AnnotationIdentity('s', 0, 'f', 'ped1', 'token', 'pedestrian')
    assert AnnotationIdentityIndex([a], []).targets[a.reference].status == 'non_car'
    assert AnnotationIdentityIndex([a], [], class_scope=('car', 'bicycle', 'pedestrian')).targets[a.reference].status == 'matched'
    from transvision.models.event_track_v2x.forest_potentials import LearnedForestScorer
    from transvision.models.event_track_v2x.forest_tracking import cache_detections
    from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
    raw = cache_detections(_frame(1), arrival_us=1_100_000, decision_us=1_100_000, origin_us=0)
    previous = replace(raw[0], node=replace(raw[0].node, node_id='history', frame_id='prior'))
    torch.manual_seed(1337)
    model = RecoverableIdentityModel(hidden=8, heads=2, dropout=0).eval().requires_grad_(False)
    enabled = LearnedForestScorer(model)
    disabled = LearnedForestScorer(model, history_enabled=False)
    nodes = (previous, raw[0])
    support = ((-1, ), (-1, 0))
    left = enabled(nodes, support, 1_100_000)
    right = disabled(nodes, support, 1_100_000)
    assert enabled.signature != disabled.signature and left.rows != right.rows
    assert tuple(p for p, _ in left.rows[1]) == tuple(p for p, _ in right.rows[1])


def test_bootstrap_pairing_censoring_and_preflight(tmp_path):
    from tools.event_track_v2x.run_paper import preflight
    from transvision.models.event_track_v2x.paper_reports import duration_summary, genuine_recovery, paired_bootstrap
    a = np.arange(12).reshape(3, 4)
    result = paired_bootstrap(a + 2, a, clusters=['a', 'a', 'b', 'c'], draws=100)
    assert result['lower'] == result['upper'] == 2
    assert duration_summary([dict(start_us=0, end_us=2_000_000, censored=True)])['recovered_count'] == 0
    assert genuine_recovery([])['recovery_rate'] is None
    result = preflight(dict(protocol=asdict(PaperProtocol('spd', 'train')), assets=[dict(role='detector', path=str(tmp_path / 'missing.pt'), sha256='a' * 64)], python_modules=[]))
    assert not result['ready'] and not result['formal_success_receipt']
