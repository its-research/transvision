"""Run in the independent evaluator environment; never imports the tracker."""
import json
from dataclasses import asdict

import numpy as np
import pytest

pytest.importorskip('nuscenes')
pytest.importorskip('trackeval')
from test_train_inference_evaluator import metric_fixture  # noqa: E402

from tools.event_track_v2x.evaluate_paper import adapter, evaluate  # noqa: E402


@pytest.mark.parametrize('dataset,split', [('spd', 'val'), ('v2v4real', 'official_test')])
@pytest.mark.parametrize('case,expected', [('perfect', 1.), ('empty', 0.), ('identity_switch', 2**-.5)])
def test_independent_paper_evaluation_both_datasets(tmp_path, dataset, split, case, expected):
    from transvision.models.event_track_v2x.paper_protocol import PaperProtocol
    root, _, replay, _ = metric_fixture(tmp_path, case)
    m = adapter()
    protocol = asdict(PaperProtocol(dataset, split))
    (replay / 'plan.json').write_bytes(m.canonical(dict(protocol=protocol)))
    receipt = dict(status='software_replay_completed', fixture=True, completed_events=4, files={name: m.sha(replay / name) for name in ('plan.json', 'predictions.jsonl')})
    (replay / 'receipt.json').write_bytes(m.canonical(receipt))
    gtpath = root / 'ground-truth.jsonl'
    roi = dict(kind='strict_radial_xy', radius_m=50.)
    if dataset == 'v2v4real':
        rows = [dict(json.loads(line), world_to_ego_row_rotation=np.eye(3).tolist()) for line in gtpath.read_bytes().splitlines()]
        gtpath.write_bytes(b''.join(m.canonical(row) + b'\n' for row in rows))
        roi = dict(kind='ego_xy_rectangle', bounds_xy=[-100., -40., 100., 40.])
    manifest = dict(
        kind='rbf_paper_evaluation_gt_v1',
        fixture=True,
        protocol=protocol,
        frames=4,
        roi=roi,
        ground_truth=dict(path='ground-truth.jsonl', sha256=m.sha(gtpath)),
        sequence_clusters={'golden': 'recording1'})
    (root / 'manifest.json').write_bytes(m.canonical(manifest))
    result = evaluate(root / 'manifest.json', m.sha(root / 'manifest.json'), replay, m.sha(replay / 'receipt.json'), tmp_path / 'evaluation')
    assert result['metrics']['HOTA'] == pytest.approx(expected)
    assert result['per_sequence']['golden']['metrics'] == result['metrics']
    assert result['fixture'] and not result['native_protocol_reproduction']


def test_all_four_tables_require_actual_evaluator_bindings_and_three_seeds(tmp_path):
    from transvision.models.event_track_v2x.paper_protocol import PaperProtocol
    from transvision.models.event_track_v2x.paper_reports import METRICS, build_tables, write_tables
    m = adapter()
    prediction = tmp_path / 'predictions.jsonl'
    prediction.write_bytes(b'fixture-only')
    protocol = asdict(PaperProtocol('spd', 'val'))
    metrics = {k: 0. for k in METRICS}
    evaluation = tmp_path / 'evaluation.json'
    evaluation.write_bytes(m.canonical(dict(status='evaluated', metrics=metrics, protocol=protocol, fixture=True, input_sha256={str(prediction): m.sha(prediction)})))
    base = dict(
        status='evaluated',
        method='fixture',
        protocol=protocol,
        evidence_kind='fixture',
        deterministic=False,
        detector_sha256='a' * 64,
        embedding_sha256='b' * 64,
        motion='constant-velocity',
        roi='test-roi',
        frequency_hz=10,
        label_version='fixture',
        evaluator_sha256=m.sha(evaluation),
        metrics=metrics,
        resources={},
        amotp_definition='fixture only',
        artifacts=dict(prediction=dict(path=str(prediction), sha256=m.sha(prediction)), evaluation=dict(path=str(evaluation), sha256=m.sha(evaluation))))
    records = [dict(base, table=table, seed=seed) for table in range(1, 5) for seed in (1337, 2027, 3407)]
    result = write_tables(records, tmp_path / 'tables')
    assert not result['missing_tables']
    assert all(row[0]['sample_std']['HOTA'] == 0. for row in result['tables'].values())
    with pytest.raises(ValueError, match='missing or repeated seed'):
        build_tables(records[:-1])
    altered = [dict(r) for r in records]
    altered[0]['metrics'] = dict(metrics, HOTA=.9)
    with pytest.raises(ValueError, match='not bound'):
        build_tables(altered)
    missing_amotp = [dict(r) for r in records]
    del missing_amotp[0]['amotp_definition']
    with pytest.raises(ValueError, match='missing comparison fields'):
        build_tables(missing_amotp)
    mixed_amotp = [dict(r) for r in records]
    mixed_amotp[0]['amotp_definition'] = 'native IoU instead of center distance'
    with pytest.raises(ValueError, match='incomparable'):
        build_tables(mixed_amotp)
    mixed = [dict(r) for r in records]
    mixed[0]['roi'] = 'another'
    with pytest.raises(ValueError, match='incomparable'):
        build_tables(mixed)


@pytest.mark.parametrize('number', [1, 2, 4, 5])
def test_remaining_figure_generators(tmp_path, number):
    from tools.event_track_v2x.plot_paper import render
    m = adapter()
    source = tmp_path / 'fixture.json'
    source.write_text('{}')
    if number == 1:
        data = dict(gt_used_online=False, moments=[dict(timestamp_us=i, observations=[], ground_truth=[], predictions=[], branch_mass=[dict(id='a', mass=1.)]) for i in (0, 1, 2)])
    elif number == 2:
        data = dict(
            raw_factor_bytes=128,
            window_us=1_000_000,
            stages=[dict(name=str(i), explicit_leaves=['a'], disjoint_frontier=['b*'], action='a', audit_sha256='a' * 64) for i in (0, 1, 2)])
    elif number == 4:
        row = dict(
            method='mht',
            protocol={'dataset': 'spd'},
            detector_sha256='a' * 64,
            embedding_sha256='b' * 64,
            motion='fixture',
            roi='fixture',
            frequency_hz=10,
            label_version='fixture',
            evaluator_sha256='c' * 64,
            amotp_definition='fixture-only-center-distance',
            hardware_id='fixture',
            includes_raw_factors_frontier_states=True,
            HOTA=0.,
            IDF1=0.)
        data = dict(expected_mht_widths=[1, 4], rows=[dict(row, width=k, total_peak_memory_bytes=1024 * k, p95_latency_seconds=k / 1000) for k in (1, 4)])
    else:
        data = dict(
            episodes=[dict(start_us=0, end_us=1_000_000, censored=True), dict(start_us=0, end_us=500_000, censored=False)],
            recovery_events=[dict(timestamp_us=500_000, success=True), dict(timestamp_us=1_000_000, success=False)])
    result = render(dict(figure=number, evidence_kind='fixture', sources=[dict(path=str(source), sha256=m.sha(source))], data=data), tmp_path / 'figure')
    assert result['figure'] == number and len(result['artifacts']) == 3


def test_figure_generator_uses_bound_oracle_inputs_and_fixture_watermark(tmp_path):
    from tools.event_track_v2x.plot_paper import render
    m = adapter()
    source = tmp_path / 'oracle.json'
    source.write_text('{}')
    manifest = dict(
        figure=3,
        evidence_kind='fixture',
        sources=[dict(path=str(source), sha256=m.sha(source))],
        data=dict(independent_enumeration=True, rows=[dict(exact_omitted_mass=.25, upper_bound=.5, evidence_likelihood_ratio=2., eta_y=.5)]))
    result = render(manifest, tmp_path / 'figure')
    assert result['sources_verified']
    assert 'NOT PAPER RESULTS' in (tmp_path / 'figure/figure-3.svg').read_text()
    source.write_text('changed')
    with pytest.raises(ValueError, match='source changed'):
        render(manifest, tmp_path / 'changed')


from transvision.models.event_track_v2x.paper_evaluation_policy import (
    VEHICLE_PROTOCOL, NATIVE_VEHICLE_SELECTION, vehicle_binding,
)
from transvision.models.event_track_v2x.paper_protocol import PaperProtocol

def make_vehicle_evaluation(tmp_path, *, missing_binding=False):
    from test_train_inference_evaluator import metric_fixture
    from tools.event_track_v2x.evaluate_paper import adapter
    root, _, replay, _ = metric_fixture(tmp_path, 'perfect')
    m = adapter()
    protocol = asdict(PaperProtocol('v2v4real', 'official_test', evaluation_class='vehicle'))
    (replay/'plan.json').write_bytes(m.canonical(dict(protocol=protocol, model_binding={} if missing_binding else vehicle_binding())))
    (replay/'receipt.json').write_bytes(m.canonical(dict(status='software_replay_completed', fixture=True,
        completed_events=4, files={name: m.sha(replay/name) for name in ('plan.json', 'predictions.jsonl')})))
    gtpath = root/'ground-truth.jsonl'
    rows = [json.loads(line) for line in gtpath.read_bytes().splitlines()]
    for row in rows:
        for box in row['objects']:
            box.update(class_label='vehicle', raw_class='Truck')
    gtpath.write_bytes(b''.join(m.canonical(row)+b'\n' for row in rows))
    manifest = dict(kind='rbf_paper_evaluation_gt_v1', fixture=True, protocol=protocol,
        evaluation_protocol=VEHICLE_PROTOCOL, native_label_source=NATIVE_VEHICLE_SELECTION,
        frames=4, roi=dict(kind='strict_radial_xy', radius_m=50.),
        ground_truth=dict(path=gtpath.name, sha256=m.sha(gtpath)), sequence_clusters={'golden': 'recording1'})
    (root/'manifest.json').write_bytes(m.canonical(manifest))
    return root, replay, m


def test_independent_vehicle_metrics_are_explicitly_auxiliary_and_commits_unchanged(tmp_path):
    pytest.importorskip('nuscenes')
    pytest.importorskip('trackeval')
    from tools.event_track_v2x.evaluate_paper import evaluate
    root, replay, m = make_vehicle_evaluation(tmp_path)
    original = (replay/'predictions.jsonl').read_bytes()
    result = evaluate(root/'manifest.json', m.sha(root/'manifest.json'), replay,
                      m.sha(replay/'receipt.json'), tmp_path/'evaluation')
    assert result['metrics']['HOTA'] == pytest.approx(1.)
    assert result['protocol']['evaluation_class'] == 'vehicle'
    assert result['metric_backend_class_alias'] == {'vehicle': 'car'}
    assert result['evaluation_class_binding'] == vehicle_binding()
    assert not result['native_v2v4real_tracking_metrics'] and not result['native_protocol_reproduction']
    assert result['auxiliary_metrics_only']
    assert 'lower_is_better_not_native_AB3DMOT' in result['amotp_definition']
    assert (replay/'predictions.jsonl').read_bytes() == original
    from transvision.models.event_track_v2x.paper_reports import validate_result
    report_path = tmp_path/'evaluation/report.json'
    record = dict(status='evaluated', table=1, evidence_kind='fixture', deterministic=True, seed=None,
                  protocol=result['protocol'], metrics=result['metrics'], amotp_definition=result['amotp_definition'],
                  artifacts=dict(prediction=dict(path=str(replay/'predictions.jsonl'), sha256=m.sha(replay/'predictions.jsonl')),
                                 evaluation=dict(path=str(report_path), sha256=m.sha(report_path))))
    validate_result(record)
    with pytest.raises(ValueError, match='AMOTP definition'):
        validate_result(dict(record, amotp_definition='native_IoU_AMOTP_higher_is_better'))


def test_independent_vehicle_evaluation_rejects_unbound_legacy_predictions(tmp_path):
    from tools.event_track_v2x.evaluate_paper import evaluate
    root, replay, m = make_vehicle_evaluation(tmp_path, missing_binding=True)
    with pytest.raises(ValueError, match='vehicle calibration binding'):
        evaluate(root/'manifest.json', m.sha(root/'manifest.json'), replay,
                 m.sha(replay/'receipt.json'), tmp_path/'evaluation')


@pytest.mark.parametrize('bad', ['missing_obj_type', 'Pedestrian', 'car'])
def test_vehicle_gt_rejects_silent_class_relabeling_even_when_resealed(tmp_path, bad):
    root, replay, m = make_vehicle_evaluation(tmp_path)
    path = root/'ground-truth.jsonl'
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if bad == 'missing_obj_type':
        del rows[0]['objects'][0]['raw_class']
    elif bad == 'Pedestrian':
        rows[0]['objects'][0]['raw_class'] = 'Pedestrian'
    else:
        rows[0]['objects'][0]['class_label'] = 'car'
    path.write_bytes(b''.join(m.canonical(row)+b'\n' for row in rows))
    manifest = json.loads((root/'manifest.json').read_bytes())
    manifest['ground_truth']['sha256'] = m.sha(path)
    (root/'manifest.json').write_bytes(m.canonical(manifest))
    with pytest.raises(ValueError, match='obj_type'):
        evaluate(root/'manifest.json', m.sha(root/'manifest.json'), replay,
                 m.sha(replay/'receipt.json'), tmp_path/'evaluation')
