"""Small synthetic evidence bundles: no dataset, GT payload or metric rerun."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('mechanism_eval_test', ROOT/'tools/event_track_v2x/evaluate_mechanism_diagnostics_v2.py')
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(evaluation.canonical(value)+b'\n')


def metric_fixture(value):
    nu = {name: {'car': value} for group, name in evaluation.METRICS.values() if group == 'nuscenes'}
    te = {name: value for group, name in evaluation.METRICS.values() if group == 'trackeval'}
    return {'nuscenes': {'label_metrics': nu}, 'trackeval': {'car': {'summary': te,
        'sequences': {f'{i:04d}': {'HOTA': {k: [value]*19 for k in ('HOTA', 'AssA', 'DetA')},
                                  'Identity': {'IDF1': value}} for i in range(21)}}}}


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    source = evaluation.load_evaluator()
    monkeypatch.setattr(source, 'validate_runtime', Mock())
    monkeypatch.setattr(evaluation, 'load_evaluator', lambda: source)
    adapter = source.load_adapter()
    protocol = source.protocol(adapter)
    commits = {f'{i:04d}': 'f'*64 for i in range(21)}
    root, reports, reference_root = [tmp_path/x for x in ('experiment', 'reports', 'reference')]
    plan = {'kind': 'mechanism_diagnostics_v2_plan', 'runs': evaluation.RUNS, 'reporting_scope': 'car_only',
            'paper_eligible': False, 'gt_model_inputs': False, 'test_payloads_read': False,
            'val_parameter_fitting': False, 'seed_selection': False, 'official_validation_frames': 3316,
            'sequences': 21, 'current_source_hashes': evaluation.PREDICTION_SOURCE_PINS}
    put(root/'frozen-plan.json', plan)
    old_sources, old_values = {}, {}
    def report(folder, value, prediction):
        metrics = metric_fixture(value)
        for name, item in [('metrics.json', metrics), ('runtime.json', {}), ('golden-cases.json', {'passed': True})]:
            put(folder/name, item)
        data = {'kind': 'source_ablation_car_evaluation_v1', 'status': 'completed', 'reporting_scope': 'car_only',
                'protocol': protocol, 'protocol_sha256': hashlib.sha256(evaluation.canonical(protocol)).hexdigest(),
                'sealed_protocol_sha256': hashlib.sha256(evaluation.canonical(adapter.PROTOCOL)).hexdigest(),
                'ground_truth_manifest_sha256': source.GT_MANIFEST_SHA256, 'ground_truth_sha256': source.GT_SHA256,
                'paper_eligible': False, 'no_model_selection': True, 'no_test_payload': True,
                'no_validation_parameter_fitting': True, 'predictions_sha256': prediction['sha256'],
                'coverage': {'frames': 3316, 'sequences': 21, 'final_sequence_commit_sha256': commits},
                'input_sources': {'evaluation_cli': evaluation.evidence(evaluation.SOURCE_EVALUATOR),
                                 'sealed_adapter': evaluation.evidence(source.ADAPTER_PATH), 'predictions': prediction},
                'files': {name: evaluation.evidence(folder/name) for name in ('metrics.json', 'runtime.json', 'golden-cases.json')},
                'primary_car': source.car_metrics(metrics)}
        put(folder/'report.json', data)
        return data, {name: evaluation.evidence(folder/name) for name in ('metrics.json', 'runtime.json', 'golden-cases.json', 'report.json')}
    seed_values = {seed: .3 + i*.01 for i, seed in enumerate(evaluation.SEEDS)}
    for name in ['vehicle-only', 'infrastructure-only'] + [f'cooperative-seed-{s}' for s in evaluation.SEEDS]:
        value = .1 if name == 'vehicle-only' else (.7 if name == 'infrastructure-only' else seed_values[int(name.rsplit('-', 1)[1])])
        pred_file = tmp_path/'reference-predictions'/name/'predictions.jsonl'
        assoc_file = pred_file.with_name('association.jsonl')
        put(pred_file, {'prediction_value': value})
        put(assoc_file, {'association': name})
        data, files = report(reference_root/name, value, evaluation.evidence(pred_file))
        old_sources[name] = {'evaluation_files': files, 'runner_files': {'predictions.jsonl': evaluation.evidence(pred_file),
                                                                      'association.jsonl': evaluation.evidence(assoc_file)}}
        old_values[name] = source.metric_vector(data['primary_car'])
    reference = reference_root/'comparison'/'comparison.json'
    put(reference.with_name('source-manifest.json'), {'runs': old_sources})
    put(reference, {'kind': 'source_ablation_car_comparison_v1', 'status': 'completed',
                    'reporting_scope': 'car_only', 'paper_eligible': False, 'run_metrics': old_values,
                    'source_manifest_sha256': evaluation.sha(reference.with_name('source-manifest.json'))})
    monkeypatch.setattr(evaluation, 'REFERENCE_COMPARISON_SHA', evaluation.sha(reference))
    receipts = {}
    for spec in evaluation.RUNS:
        name, seed, mode = spec['run_id'], spec['seed'], spec['mode']
        value = seed_values[seed] + {'M0': 0., 'M1': .2, 'M2': .1, 'M3': -.1}[mode]
        folder = root/name
        pred_file = folder/'predictions.jsonl'
        put(pred_file, {'prediction_value': value})
        put(folder/'association.jsonl', {'association': f'cooperative-seed-{seed}' if mode != 'M1' else 'unmatched'})
        (folder/'diagnostics.jsonl.gz').write_bytes(b'synthetic-not-used-by-chain-audit')
        receipt = {**spec, 'frames': 3316, 'sequences': 21, 'sequence_commits': commits,
                   'diagnostic_sequence_commits': commits, 'weights_unchanged': True,
                   'sealed_association_parity': True if mode != 'M1' else None,
                   'sealed_prediction_parity': True if mode == 'M0' else None,
                   'predictions_sha256': evaluation.sha(pred_file), 'association_sha256': evaluation.sha(folder/'association.jsonl'),
                   'diagnostics_gzip_sha256': evaluation.sha(folder/'diagnostics.jsonl.gz'),
                   'diagnostics_gzip_bytes': (folder/'diagnostics.jsonl.gz').stat().st_size,
                   'events': {'birth': 4, 'miss': 3, 'kill_before_assignment': 1}, 'nodes': 4,
                   'temporal_matches': 3, 'predictions': 7}
        put(folder/'receipt.json', receipt)
        receipts[name] = receipt
        report(reports/name, value, evaluation.evidence(pred_file))
    put(root/'summary.json', {'kind': 'mechanism_diagnostics_v2_complete', 'status': 'completed',
        'reporting_scope': 'car_only', 'paper_eligible': False, 'weights_unchanged': True,
        'all_required_sealed_streams_match': True, 'plan_sha256': evaluation.sha(root/'frozen-plan.json'), 'runs': receipts})
    return SimpleNamespace(root=root, reports=reports, reference=reference, output=tmp_path/'comparison', source=source)


def test_all_ten_runs_pairing_n1_and_counter_boundaries(bundle):
    value = evaluation.compare(bundle.root, bundle.reports, bundle.reference, bundle.output)
    assert len(value['run_metrics']) == 10
    assert value['M1_repetitions'] == 1
    assert value['mode_descriptive']['M1']['HOTA']['n'] == 1
    assert value['mode_descriptive']['M1']['HOTA']['sample_sd'] is None
    assert value['mode_descriptive']['M0']['HOTA']['n'] == 3
    assert value['mode_descriptive']['M0']['HOTA']['sample_sd'] == pytest.approx(.01)
    for seed in evaluation.SEEDS:
        assert value['M2_M3_minus_same_seed_M0']['M2'][str(seed)]['metric_difference']['HOTA'] == pytest.approx(.1)
        assert value['M2_M3_minus_same_seed_M0']['M3'][str(seed)]['metric_difference']['HOTA'] == pytest.approx(-.1)
    assert value['paired_difference_descriptive']['M2']['HOTA']['mean'] == pytest.approx(.1)
    assert all(value['M0_matches_sealed_reference'].values())
    assert set(value['run_metrics']['M1-all-unmatched']) == set(evaluation.METRICS)
    assert value['metric_direction']['FP'] == value['metric_direction']['AMOTP_m'] == 'lower_is_better'
    assert 'not a birth-only' in value['M3_interpretation']
    assert value['paper_eligible'] is False
    assert value['independent_diagnostic_chain_audit_required'] is True
    assert 'not car metrics' in value['lifecycle_counters']['M1-all-unmatched']['scope']
    assert value['source_manifest_sha256'] == evaluation.sha(bundle.output/'source-manifest.json')
    with pytest.raises(ValueError, match='create-once'):
        evaluation.compare(bundle.root, bundle.reports, bundle.reference, bundle.output)


@pytest.mark.parametrize('kind', ['prediction', 'association', 'diagnostic', 'receipt', 'plan', 'source_pin'])
def test_changed_runner_evidence_is_rejected(bundle, kind):
    folder = bundle.root/'M2-seed-2027'
    if kind in ('prediction', 'association', 'diagnostic'):
        name = {'prediction': 'predictions.jsonl', 'association': 'association.jsonl', 'diagnostic': 'diagnostics.jsonl.gz'}[kind]
        with (folder/name).open('ab') as stream:
            stream.write(b'x')
    elif kind == 'receipt':
        data = evaluation.read_json(folder/'receipt.json'); data['frames'] = 3315
        put(folder/'receipt.json', data)
    else:
        data = evaluation.read_json(bundle.root/'frozen-plan.json')
        if kind == 'plan':
            data['runs'] = data['runs'][:-1]
        else:
            data['current_source_hashes'] = {k: '0'*64 for k in evaluation.PREDICTION_SOURCE_PINS}
        put(bundle.root/'frozen-plan.json', data)
        summary = evaluation.read_json(bundle.root/'summary.json')
        summary['plan_sha256'] = evaluation.sha(bundle.root/'frozen-plan.json')
        put(bundle.root/'summary.json', summary)
    with pytest.raises(ValueError):
        evaluation.compare(bundle.root, bundle.reports, bundle.reference, bundle.output)
    assert not bundle.output.exists()


@pytest.mark.parametrize('field,value', [('reporting_scope', 'all'), ('paper_eligible', True),
    ('no_model_selection', False), ('predictions_sha256', '0'*64), ('primary_car', {}), ('status', 'running')])
def test_corrupt_or_incompatible_car_report_rejected(bundle, field, value):
    path = bundle.reports/'M3-seed-3407'/'report.json'
    data = evaluation.read_json(path); data[field] = value; put(path, data)
    with pytest.raises(ValueError):
        evaluation.compare(bundle.root, bundle.reports, bundle.reference, bundle.output)


def test_modified_metric_artifact_rejected(bundle):
    put(bundle.reports/'M0-seed-1337'/'metrics.json', metric_fixture(.91))
    with pytest.raises(ValueError, match='artifact hash'):
        evaluation.compare(bundle.root, bundle.reports, bundle.reference, bundle.output)


def test_reference_manifest_link_rejected(bundle):
    value = evaluation.read_json(bundle.reference); value['source_manifest_sha256'] = '0'*64
    put(bundle.reference, value)
    with pytest.raises(ValueError, match='reference'):
        evaluation.compare(bundle.root, bundle.reports, bundle.reference, bundle.output)


def test_reference_report_tampering_rejected(bundle):
    path = bundle.reference.parent.parent/'cooperative-seed-2027'/'report.json'
    value = evaluation.read_json(path); value['coverage']['frames'] = 3000; put(path, value)
    with pytest.raises(ValueError):
        evaluation.compare(bundle.root, bundle.reports, bundle.reference, bundle.output)


@pytest.mark.parametrize('value', [None, True, float('nan'), float('inf'), '0'])
def test_invalid_metrics_fail_closed(value):
    with pytest.raises(ValueError):
        evaluation.finite(value)


def test_n1_is_not_seed_sd_and_paired_cohorts_are_required():
    vector = {k: .5 for k in evaluation.METRICS}
    assert evaluation.descriptives([vector])['HOTA'] == {'mean': .5, 'sample_sd': None, 'n': 1}
    with pytest.raises(ValueError, match='paired sequence'):
        evaluation.contrast(vector, vector, {'0000': {}}, {'0001': {}})


def test_worker_routes_only_evaluate_not_source_compare(monkeypatch, tmp_path):
    evaluate = Mock()
    forbidden_compare = Mock(side_effect=AssertionError('do not call hardcoded source compare'))
    monkeypatch.setattr(evaluation, 'load_evaluator', lambda: SimpleNamespace(evaluate=evaluate, compare=forbidden_compare))
    args = SimpleNamespace(ground_truth=tmp_path/'gt', predictions=tmp_path/'prediction', output=tmp_path/'output')
    evaluation.worker(args)
    evaluate.assert_called_once_with(args.ground_truth, args.predictions, args.output)
    forbidden_compare.assert_not_called()


@pytest.mark.parametrize('workers', [0, 4, True])
def test_worker_limit_precedes_any_evaluation(tmp_path, workers):
    with pytest.raises(ValueError, match='max-workers'):
        evaluation.run(SimpleNamespace(output=tmp_path/'output', max_workers=workers))


def test_existing_run_output_is_not_reused(tmp_path):
    output = tmp_path/'already'; output.mkdir()
    with pytest.raises(ValueError, match='create-once'):
        evaluation.run(SimpleNamespace(output=output, max_workers=3))


def test_reference_vectors_match_real_frozen_local_reports_without_metric_computation():
    path = Path('/Users/lbin/Desktop/Codes/thesis/evidence/clearml/eventtrack-full-train/source-ablation/results/evaluation/comparison/comparison.json')
    if not path.exists():
        pytest.skip('local optional reference artifact not present')
    source = evaluation.load_evaluator()
    reference, _ = evaluation.validate_reference(path, source)
    assert len(reference) == 5
    assert reference['infrastructure-only']['metrics']['HOTA'] == pytest.approx(.23917146446157758)
    assert reference['cooperative-seed-1337']['metrics']['FP'] == 5924
