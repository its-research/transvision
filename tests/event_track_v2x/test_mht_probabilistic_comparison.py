"""Synthetic aggregation and tamper cases, not real tracking measurements."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from tools.event_track_v2x import compare_mht_probabilistic_validation as tool
from tools.event_track_v2x.persistent_mht_tracking import PersistentMHTConfig


def resource_record():
    return dict(elapsed_seconds=100., process_peak_rss_bytes=20000, database_bytes=50000,
        latency_seconds_p50_p95_p99_max=[.1, .2, .3, .4],
        frame_latency_seconds_p50_p95_p99_max=[.2, .3, .4, .5],
        latency_scope='step', frame_latency_scope='whole-frame', latency_clock='time.monotonic',
        latency_quantile_method='numpy_linear')


@pytest.fixture
def cohort():
    configs = {r: asdict(tool.baseline.PersistentProbabilisticConfig(update_rule=r)) for r in tool.RULES}
    pc = dict(configuration_by_update_rule=configs, source_sha256={'shared': 'source'}, jobs=[], checkpoints=[])
    runtime = dict(pid=1, threads=1, exclusive_host=False, same_latency_or_memory_verified=False)
    mc = dict(configuration=asdict(PersistentMHTConfig()), jobs=[], checkpoints=[],
        source_sha256={**pc['source_sha256'], **{k: 'mht-source' for k in tool.EXTRA_SOURCES}},
        runtime={**runtime, **{k: 'mht-runtime' for k in tool.EXTRA_RUNTIME}})
    p_cells, m_cells = [], []
    for i, seed in enumerate(tool.SEEDS):
        cp = dict(seed=seed, sha256=f'cp-{seed}', model_sha256=f'model-{seed}', scorer_signature=f'scorer-{seed}')
        pc['checkpoints'].append(copy.deepcopy(cp)); mc['checkpoints'].append(copy.deepcopy(cp))
        for j, rule in enumerate((*tool.RULES, tool.MHT)):
            is_mht = rule == tool.MHT
            config = mc['configuration'] if is_mht else configs[rule]
            dest = f'/fixture/{seed}-{rule}'
            args = dict(cache_sha256='cache', schedule_sha256='schedule', checkpoint_sha256=cp['sha256'])
            job = dict(seed=seed, output=dest, arguments=args)
            if is_mht:
                job['width'] = 4
                args.update(width=4, output=dest, device='cpu')
                args.update({k: config[k] for k in config if k.startswith('max_assignment_') or k == 'max_generated_candidates'})
            else:
                job['update_rule'] = args['update_rule'] = rule
            (mc if is_mht else pc)['jobs'].append(job)
            r = dict(seed=seed, run_directory=dest, configuration=copy.deepcopy(config),
                width=4, checkpoint_sha256=cp['sha256'], scorer_signature=cp['scorer_signature'],
                cache_sha256='cache', schedule_sha256='schedule', inference_source_sha256=(mc if is_mht else pc)['source_sha256'],
                inference_runtime=copy.deepcopy(mc['runtime'] if is_mht else runtime),
                protocol=dict(kind='mht' if is_mht else 'jpda', roi='same', car=True),
                ground_truth_sha256='gt', ground_truth_manifest_sha256='manifest',
                paper_eligible=False, fair_resources_verified=False, reproduced_public_method=False,
                same_state_time_protocol_as_recoverable=False, validation_checkpoint_selection=False,
                validation_parameter_search=False, same_input_selection_as_legacy_source_ablation=False)
            if not is_mht: r.update(update_rule=rule, factor_stream_sha256=f'factor-{seed}')
            values = {m: .2 + i * .01 + j * .1 for m in tool.baseline.METRICS}
            cell = dict(report=r, primary=values, evaluator_runtime={'native': 'fixed'},
                factor_stream_sha256=f'factor-{seed}', resources=resource_record(),
                sequences={f'{s:04d}': {m: .1 + i * .01 + j * .1 for m in ('HOTA', 'AssA', 'DetA', 'IDF1')}
                           for s in range(21)})
            (m_cells if is_mht else p_cells).append(cell)
    return mc, pc, m_cells, p_cells


def test_all_twelve_cells_preserve_three_seeds_and_scope(cohort):
    mc, pc, ms, ps = cohort
    result = tool.assemble(mc, pc, ms, ps)
    assert result == tool.assemble(mc, pc, ms[::-1], ps[::-1])
    metric = result['three_seed_descriptive'][tool.MHT]['HOTA']
    # Independent hand-derived mean and sample SD of 0.50, 0.51, 0.52.
    assert metric['mean'] == pytest.approx(.51)
    assert metric['sample_sd'] == pytest.approx(.01)
    assert set(metric['by_seed']) == {'1337', '2027', '3407'}
    assert result['paired_mht_minus_baseline']['pkf']['2027']['primary']['HOTA'] == pytest.approx(.1)
    assert metric['mean'] != pytest.approx(.41)  # Not the mean of sequence HOTA values.
    assert len(result['per_run']) == result['completed_cells'] == 12
    assert result['sample_sd_is_not_confidence_interval'] is True
    assert result['actual_factors_identical_within_each_seed'] is True
    for flag in ('fair_resources_verified', 'paper_eligible', 'full_paper_comparison_completed',
                 'same_state_time_protocol_verified', 'recovery_only_causal_effect_verified',
                 'public_methods_reproduced', 'recoverable_method_included',
                 'strong_single_endpoint_controls_included', 'deployment_tail_latency_verified'):
        assert result[flag] is False
    assert result['metric_direction']['AMOTP_m'] == 'lower'
    assert result['metric_direction']['HOTA'] == 'higher'


@pytest.mark.parametrize('bad', ['mht_missing', 'mht_duplicate', 'jpda_missing', 'jpda_duplicate',
    'job_missing', 'job_duplicate', 'cp_missing', 'cp_duplicate', 'model', 'scorer', 'checkpoint',
    'configuration', 'width', 'job_width', 'argument_width', 'budget', 'destination', 'device',
    'source', 'shared_source', 'source_inventory', 'cache', 'schedule', 'runtime', 'runtime_missing',
    'metric_runtime', 'protocol', 'gt', 'factor', 'missing_metric', 'nan', 'sequence_missing',
    'sequence_metric_missing', 'sequence_nan', 'rss', 'latency', 'resource_claim', 'selection_claim'])
def test_noncomparable_or_incomplete_runs_cannot_generate_a_table(cohort, bad):
    mc, pc, ms, ps = cohort; r = ms[0]['report']; a = mc['jobs'][0]['arguments']
    if bad == 'mht_missing': ms.pop()
    elif bad == 'mht_duplicate': ms[0] = copy.deepcopy(ms[1])
    elif bad == 'jpda_missing': ps.pop()
    elif bad == 'jpda_duplicate': ps[0] = copy.deepcopy(ps[1])
    elif bad == 'job_missing': mc['jobs'].pop()
    elif bad == 'job_duplicate': mc['jobs'][0] = copy.deepcopy(mc['jobs'][1])
    elif bad == 'cp_missing': mc['checkpoints'].pop()
    elif bad == 'cp_duplicate': mc['checkpoints'][0] = copy.deepcopy(mc['checkpoints'][1])
    elif bad == 'model': mc['checkpoints'][0]['model_sha256'] = 'other'
    elif bad == 'scorer': r['scorer_signature'] = 'other'
    elif bad == 'checkpoint': a['checkpoint_sha256'] = 'other'
    elif bad == 'configuration': r['configuration']['state']['birth_score'] = .7
    elif bad == 'width': r['width'] = 8
    elif bad == 'job_width': mc['jobs'][0]['width'] = 8
    elif bad == 'argument_width': a['width'] = 8
    elif bad == 'budget': a['max_assignment_solves'] += 1
    elif bad == 'destination': a['output'] += '/other'
    elif bad == 'device': a['device'] = 'cuda'
    elif bad == 'source': r['inference_source_sha256'] = {}
    elif bad == 'shared_source': mc['source_sha256']['shared'] = 'other'
    elif bad == 'source_inventory': mc['source_sha256']['extra.py'] = 'other'
    elif bad == 'cache': r['cache_sha256'] = 'other'
    elif bad == 'schedule': a['schedule_sha256'] = 'other'
    elif bad == 'runtime': r['inference_runtime']['threads'] = 2
    elif bad == 'runtime_missing': r['inference_runtime'].pop('sqlite_version')
    elif bad == 'metric_runtime': ms[0]['evaluator_runtime']['native'] = 'other'
    elif bad == 'protocol': r['protocol']['roi'] = 'other'
    elif bad == 'gt': r['ground_truth_sha256'] = 'other'
    elif bad == 'factor': ms[0]['factor_stream_sha256'] = 'other'
    elif bad == 'missing_metric': ms[0]['primary'].pop('FN')
    elif bad == 'nan': ms[0]['primary']['HOTA'] = float('nan')
    elif bad == 'sequence_missing': ms[0]['sequences'].pop('0000')
    elif bad == 'sequence_metric_missing': ms[0]['sequences']['0000'].pop('AssA')
    elif bad == 'sequence_nan': ms[0]['sequences']['0000']['AssA'] = float('nan')
    elif bad == 'rss': ms[0]['resources']['process_peak_rss_bytes'] = 0
    elif bad == 'latency': ms[0]['resources']['latency_seconds_p50_p95_p99_max'] = [.2, .1, .3, .4]
    elif bad == 'resource_claim': r['fair_resources_verified'] = True
    elif bad == 'selection_claim': r['validation_parameter_search'] = True
    with pytest.raises(ValueError): tool.assemble(mc, pc, ms, ps)


def factor_fixture(tmp_path):
    rows = [dict(sequence_id=s, vehicle_frame=f) for s, f in [('a', '0'), ('a', '1'), ('b', '0')]]
    records = [dict(tracking=dict(sequence_id=r['sequence_id'], event_id=r['vehicle_frame'],
                                 factor_rows_sha256=str(i) * 64)) for i, r in enumerate(rows)]
    blocks = [tool.native.canonical([r['tracking']['sequence_id'], r['tracking']['event_id'],
                                     r['tracking']['factor_rows_sha256']]) + b'\n' for r in records]
    audits = {s: dict(frames=sum(r['sequence_id'] == s for r in rows),
        factor_stream_sha256=hashlib.sha256(b''.join(b for b, r in zip(blocks, rows) if r['sequence_id'] == s)).hexdigest())
              for s in ('a', 'b')}
    path = tmp_path / 'tracking.jsonl'
    path.write_bytes(b''.join(tool.native.canonical(r) + b'\n' for r in records))
    return path, rows, records, audits, hashlib.sha256(b''.join(blocks)).hexdigest()


def test_global_factor_fingerprint_matches_ordered_triplets_not_hash_of_hashes(tmp_path):
    path, rows, _, audit, expected = factor_fixture(tmp_path)
    assert tool.factor_fingerprint(path, rows, audit) == expected
    assert expected != hashlib.sha256(''.join(v['factor_stream_sha256'] for v in audit.values()).encode()).hexdigest()


@pytest.mark.parametrize('bad', ['missing', 'extra', 'reordered', 'duplicate', 'wrong_hash', 'invalid_hash',
                                'audit_missing', 'audit_digest', 'audit_count'])
def test_changed_factor_stream_is_rejected(tmp_path, bad):
    path, rows, records, audit, _ = factor_fixture(tmp_path)
    if bad == 'missing': records.pop()
    elif bad == 'extra': records.append(records[-1])
    elif bad == 'reordered': records.reverse()
    elif bad == 'duplicate': records[1] = copy.deepcopy(records[0])
    elif bad == 'wrong_hash': records[0]['tracking']['factor_rows_sha256'] = 'f' * 64
    elif bad == 'invalid_hash': records[0]['tracking']['factor_rows_sha256'] = 'not-a-hash'
    elif bad == 'audit_missing': audit.pop('b')
    elif bad == 'audit_digest': audit['a']['factor_stream_sha256'] = 'f' * 64
    elif bad == 'audit_count': audit['a']['frames'] += 1
    path.write_bytes(b''.join(tool.native.canonical(r) + b'\n' for r in records))
    with pytest.raises(ValueError): tool.factor_fingerprint(path, rows, audit)


@pytest.mark.parametrize('counts', [(0, 9), (2, 9), (3, 8), (4, 9), (3, 10)])
def test_incomplete_campaign_fails_before_any_file_or_gt_read(tmp_path, monkeypatch, counts):
    def forbidden(*args, **kwargs): pytest.fail('incomplete comparison read an evidence file')
    monkeypatch.setattr(tool.native, 'sha', forbidden)
    refs = lambda n, prefix: [(f'/fixture/{prefix}-{i}.json', '0' * 64) for i in range(n)]
    with pytest.raises(ValueError, match='three MHT and nine'):
        tool.compare('/missing-mht', '0' * 64, '/missing-jpda', '0' * 64,
                     refs(counts[0], 'm'), refs(counts[1], 'p'), tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_duplicate_report_path_fails_before_loading(tmp_path):
    with pytest.raises(ValueError, match='distinct'):
        tool.compare('missing', '0' * 64, 'missing', '0' * 64,
                     [('same', '0' * 64)] * 3, [('same', '0' * 64)] * 9, tmp_path / 'out')


@pytest.fixture
def stored_report(tmp_path, monkeypatch, cohort):
    # Small, explicitly fabricated bound object isolates the NEW report loader.
    # The full SQLite auditor and native metric engines are tested elsewhere.
    run = tmp_path / 'run'; run.mkdir()
    tracking, rows, _, seq_audits, factor = factor_fixture(run)
    receipt = dict(resource_record(), plan_sha256='plan', predictions_sha256='predictions')
    receipt_path = run / 'receipt.json'; tool.native.write_json(receipt_path, receipt)
    r = copy.deepcopy(cohort[2][0]['report']); r['run_directory'] = str(run)
    plan = dict(checkpoint_seed=r['seed'], configuration=r['configuration'], runtime=r['inference_runtime'],
                source_sha256=r['inference_source_sha256'],
                **{k: r[k] for k in ('cache_sha256', 'schedule_sha256', 'checkpoint_sha256', 'scorer_signature')})
    bound = dict(run=run, plan=plan, receipt=receipt, rows=rows, audit=dict(sequences=seq_audits),
                 evidence={str(p): tool.native.evidence(p) for p in (tracking, receipt_path)})
    monkeypatch.setattr(tool.contract, 'inspect_audited_run', lambda *a: copy.deepcopy(bound))
    monkeypatch.setattr(tool.native, 'load_adapter', lambda: None)
    protocol = dict(kind='fixture-native-mht', car=True)
    monkeypatch.setattr(tool.mht, 'protocol', lambda a: protocol)
    def check_runtime(runtime):
        if runtime != {'fixture': 'native'}: raise ValueError('runtime differs')
    monkeypatch.setattr(tool.native, 'validate_runtime', check_runtime)
    monkeypatch.setattr(tool.native, 'car_metrics', lambda data: data['primary'])
    monkeypatch.setattr(tool.native, 'sequence_vectors', lambda data: data['sequences'])
    gt = tmp_path / 'gt'; gt.mkdir()
    for name in ('manifest.json', 'ground-truth.jsonl'): (gt / name).write_text('fixture only')
    primary = {}
    for metric, (group, key) in tool.baseline.METRICS.items(): primary.setdefault(group, {})[key] = .2
    metrics = dict(primary=primary, sequences=cohort[2][0]['sequences'])
    output = tmp_path / 'evaluation'; output.mkdir()
    for name, value in (('metrics.json', metrics), ('runtime.json', {'fixture': 'native'}),
                        ('golden-cases.json', {'passed': True})):
        tool.native.write_json(output / name, value)
    expected = dict(bound['evidence'])
    for p in (Path(tool.mht.__file__), tool.native.ADAPTER_PATH, gt / 'manifest.json', gt / 'ground-truth.jsonl'):
        expected[str(p)] = tool.native.evidence(p)
    r.update(kind=tool.mht.KIND, status='complete', plan_sha256='plan', predictions_sha256='predictions',
        inference_receipt_sha256='final', inference_audit_path='fixture-audit', inference_audit_sha256='audit',
        ground_truth_directory=str(gt), ground_truth_manifest_sha256=tool.native.GT_MANIFEST_SHA256,
        ground_truth_sha256=tool.native.GT_SHA256, protocol=protocol,
        protocol_sha256=tool.contract.ledger.digest(protocol), coverage=dict(frames=3316, sequences=21),
        primary_car=primary, reporting_scope='car_only', validation_already_seen=True,
        test_payloads_read=False, parameter_training=False,
        files={p.name: tool.native.evidence(p) for p in output.iterdir()}, input_evidence=expected)
    report_path = output / 'report.json'; tool.native.write_json(report_path, r)
    return report_path, r, bound, factor


def test_loader_binds_metric_files_and_recomputes_factor_digest(stored_report):
    path, _, _, factor = stored_report
    cell = tool.load_mht_report(path, tool.native.sha(path))
    assert cell['factor_stream_sha256'] == factor
    assert set(cell['primary']) == set(tool.baseline.METRICS)
    assert cell['resources']['elapsed_seconds'] == 100.
    assert cell['resources']['deployment_tail_latency_verified'] is False
    assert str(path) in cell['evidence']


@pytest.mark.parametrize('bad', ['kind', 'status', 'seed', 'prediction', 'plan', 'width', 'configuration',
    'runtime', 'cache', 'scorer', 'protocol', 'protocol_digest', 'gt', 'frames', 'class', 'selection',
    'test', 'training', 'omitted_file', 'file_tamper', 'golden', 'metric_summary', 'input_inventory'])
def test_resealed_report_or_changed_payload_is_rejected(stored_report, bad):
    path, r, _, _ = stored_report
    if bad == 'kind': r['kind'] = 'subset'
    elif bad == 'status': r['status'] = 'failed'
    elif bad == 'seed': r['seed'] = 2027
    elif bad == 'prediction': r['predictions_sha256'] = 'other'
    elif bad == 'plan': r['plan_sha256'] = 'other'
    elif bad == 'width': r['width'] = 8
    elif bad == 'configuration': r['configuration'] = {}
    elif bad == 'runtime': r['inference_runtime'] = {}
    elif bad == 'cache': r['cache_sha256'] = 'other'
    elif bad == 'scorer': r['scorer_signature'] = 'other'
    elif bad == 'protocol': r['protocol'] = {'other': True}
    elif bad == 'protocol_digest': r['protocol_sha256'] = 'other'
    elif bad == 'gt': r['ground_truth_sha256'] = 'other'
    elif bad == 'frames': r['coverage']['frames'] = 8
    elif bad == 'class': r['reporting_scope'] = 'pedestrian'
    elif bad == 'selection': r['validation_already_seen'] = False
    elif bad == 'test': r['test_payloads_read'] = True
    elif bad == 'training': r['parameter_training'] = True
    elif bad == 'omitted_file': r['files'].pop('golden-cases.json')
    elif bad == 'file_tamper': (path.parent / 'metrics.json').write_text('{}')
    elif bad == 'golden':
        p = path.parent / 'golden-cases.json'; p.write_text('{"passed":false}')
        r['files'][p.name] = tool.native.evidence(p)
    elif bad == 'metric_summary': r['primary_car'] = {}
    elif bad == 'input_inventory': r['input_evidence'] = {}
    path.write_bytes(tool.native.canonical(r))
    with pytest.raises(ValueError): tool.load_mht_report(path, tool.native.sha(path))


def test_old_report_hash_is_rejected(stored_report):
    path, _, _, _ = stored_report
    with pytest.raises(ValueError, match='report identity'):
        tool.load_mht_report(path, '0' * 64)
