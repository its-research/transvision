"""Strict birth-only channel and three-seed replay-supplement checks."""
from __future__ import annotations

import ast
import copy
from dataclasses import asdict
import gzip
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import textwrap

import numpy as np
import pytest
import torch

from transvision.models.event_track_v2x import tracking_birth_score_v2 as birth
from transvision.models.event_track_v2x import tracking_mechanisms_v2 as base
from transvision.models.event_track_v2x.detection_cache_v2 import canonical
from transvision.models.event_track_v2x.tracking_v2 import TrackingConfigV2
from test_tracking_v2 import FixedModel, calibration, frame
from test_source_mask_v2 import short_sequence

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('birth_score_runner_test', ROOT/'tools/event_track_v2x/run_birth_score_diagnostic_v2.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def make(kind='M4', config=None):
    cls = birth.BirthScoreDiagnosticTrackerV2 if kind == 'M4' else base.MechanismDiagnosticTrackerV2
    return cls(FixedModel(), calibration(), 'seq01', 1_000_000, config=config, plan_sha256='a'*64)


def pair(time_us=1_000_000, number=1, vehicle_score=.4, road_score=.9):
    return frame(time_us=time_us, number=number, scores=[vehicle_score]), frame('infrastructure-side',
        time_us=time_us, number=number, scores=[road_score],
        states=[[3., 0., 1., 2., 4., 2., -np.pi/2, 0., 0.]])


def set_mass(monkeypatch, mass):
    hypotheses = []
    if mass < 1:
        hypotheses.append({'pairs': [[0, 0]], 'unmatched_left': [], 'unmatched_right': [], 'weight': 1-mass, 'energy': 0.})
    if mass:
        hypotheses.append({'pairs': [], 'unmatched_left': [0], 'unmatched_right': [0], 'weight': mass, 'energy': 0.})
    for module in (birth, base):
        monkeypatch.setattr(module, 'association_hypotheses', lambda *_, h=hypotheses: copy.deepcopy(h))


@pytest.mark.parametrize('mass,score,base_births,m4_births', [(.01, .9, 1, 2), (.1, .9, 1, 2),
    (.4, .9, 2, 2), (.9, .2, 1, 1), (1., .9, 2, 2), (0., .9, 1, 1)])
def test_only_birth_threshold_and_initial_score_change(monkeypatch, mass, score, base_births, m4_births):
    set_mass(monkeypatch, mass)
    original = make('M0').step(*pair(road_score=score))
    actual = make().step(*pair(road_score=score))
    assert canonical(original[1]) == canonical(actual[1])
    assert len(original[0]['predictions']) == base_births
    assert len(actual[0]['predictions']) == m4_births
    for old, new in zip(original[2]['nodes'], actual[2]['nodes']):
        assert {k: v for k, v in new.items() if k != 'birth_effective_score'} == old
        expected = score if new['kind'] == 'road_residual' else new['score']
        assert new['birth_effective_score'] == expected
    for event in actual[2]['events']:
        if event['event'] == 'birth':
            output = next(x for x in actual[0]['predictions'] if x['track_id'] == event['track_id'])
            assert output['score'] == event['birth_effective_score'] == event['score']
    if mass == 1 or mass == 0:
        assert canonical(original[0]) == canonical(actual[0])


def test_identical_entering_state_has_identical_temporal_costs_and_existing_updates():
    original, actual = make('M0'), make()
    first = (frame(), frame('infrastructure-side', states=[]))
    original.step(*first); actual.step(*first)
    second = pair(1_100_000, 2)
    old, old_assoc, old_diag = original.step(*second)
    new, new_assoc, new_diag = actual.step(*second)
    assert old_diag['temporal_assignments'] == new_diag['temporal_assignments']
    assert canonical(old_assoc) == canonical(new_assoc)
    assert all(p == next(q for q in new['predictions'] if q['track_id'] == p['track_id']) for p in old['predictions'])
    assert len(new['predictions']) > len(old['predictions'])


def test_low_common_score_temporally_matched_node_does_not_gain_birth_score(monkeypatch):
    set_mass(monkeypatch, .1)
    original, actual = make('M0'), make()
    for item in (original, actual):
        # A weak but live incumbent near the road residual. The vehicle-side
        # mixture is far away, leaving the residual as its only feasible node.
        tid = 'seq01:000001'
        item.tracks[tid] = {'track_id': tid, 'class_index': 0,
            'mean': np.array([3., 0., 1., 4., 2., 2., 0., 0., 0.]), 'cov': np.eye(9)*.001,
            'score': .06, 'last_update_us': 1_000_000,
            'identity_hypotheses': [{'track_id': tid, 'probability': 1.}], 'other_identity_probability': 0.}
        item.next_id = 2; item.last_reference_us = 1_000_000
    sources = (frame(time_us=1_100_000, number=2, states=[[300., 0., 1., 2., 4., 2., -np.pi/2, 0., 0.]], scores=[.4]),
               pair(1_100_000, 2)[1])
    old, _, old_diag = original.step(*sources)
    new, _, new_diag = actual.step(*sources)
    assert old_diag['temporal_assignments'] == new_diag['temporal_assignments']
    matched = next(x for x in new_diag['temporal_assignments'] if x['track_id'] == 'seq01:000001')
    assert matched['node_score'] == pytest.approx(.09)
    incumbent = next(x for x in new['predictions'] if x['track_id'] == 'seq01:000001')
    assert incumbent['score'] == pytest.approx(.09)
    assert incumbent == next(x for x in old['predictions'] if x['track_id'] == incumbent['track_id'])


@pytest.mark.parametrize('empty', ['left', 'right', 'both'])
def test_empty_side_no_intervention_is_exact_prediction_parity(empty):
    sources = (frame(states=[] if empty in ('left', 'both') else None),
               frame('infrastructure-side', states=[] if empty in ('right', 'both') else None))
    old, assoc, _ = make('M0').step(*sources)
    new, new_assoc, _ = make().step(*sources)
    assert canonical(old) == canonical(new) and canonical(assoc) == canonical(new_assoc)


def test_future_source_never_births_and_inputs_remain_frozen():
    item = make()
    sources = frame(), frame('infrastructure-side', image_us=1_100_001)
    before = [x.digest() for x in sources]
    weights = {k: v.clone() for k, v in item.model.state_dict().items()}
    result, _, diag = item.step(*sources)
    assert result['source_available'] == [True, False]
    assert all(n['source_side'] == 'vehicle-side' for n in diag['nodes'])
    assert [x.digest() for x in sources] == before
    for key, value in weights.items():
        torch.testing.assert_close(value, item.model.state_dict()[key], rtol=0, atol=0)


def test_copied_prebirth_arithmetic_is_ast_identical_after_removing_birth_sidecars():
    def prefix(func):
        node = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]
        statements = []
        for stmt in node.body:
            if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Tuple) and ast.unparse(stmt.target) == '(j, (mean, cov, score, cls))':
                break
            if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'birth_scores' for t in stmt.targets):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute) and isinstance(stmt.value.func.value, ast.Name) and stmt.value.func.value.id == 'birth_scores':
                continue
            statements.append(stmt)
        class StripBirth(ast.NodeTransformer):
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'birth_scores':
                    return None
                return self.generic_visit(node)
            def visit_Dict(self, node):
                kept = [(k, v) for k, v in zip(node.keys, node.values) if not isinstance(k, ast.Constant) or k.value != 'birth_effective_score']
                node.keys, node.values = [x[0] for x in kept], [x[1] for x in kept]
                return self.generic_visit(node)
        result = StripBirth().visit(ast.Module(body=statements, type_ignores=[]))
        return ast.dump(result, include_attributes=False)
    assert prefix(birth.BirthScoreDiagnosticTrackerV2.step) == prefix(base.MechanismDiagnosticTrackerV2.step)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


@pytest.fixture
def small_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'FRAME_COUNT', 5); monkeypatch.setattr(runner, 'SEQUENCE_COUNT', 1)
    reference = tmp_path/'reference'; put(reference/'frozen-plan.json', {'fixture': 'reference'})
    reference_receipts = {}
    for seed in runner.CHECKPOINTS:
        tracker = make('M0'); rows = [tracker.step(*pair) for pair in short_sequence()]
        name = f'M0-seed-{seed}'; folder = reference/name; folder.mkdir()
        (folder/'association.jsonl').write_bytes(b''.join(canonical(row[1])+b'\n' for row in rows))
        reference_receipts[name] = {'association_sha256': runner.sha_file(folder/'association.jsonl')}
    put(reference/'summary.json', {'runs': reference_receipts})
    plan = {'kind': 'birth_score_diagnostics_v2_plan', 'runs': runner.RUNS,
            'official_validation_frames': 5, 'sequences': 1,
            'cache_sha256': runner.CACHE_SHA, 'schedule_sha256': runner.SCHEDULE_SHA,
            'config': asdict(TrackingConfigV2()), 'reference_mechanism_root': str(reference),
            'reference_mechanism_plan_sha256': runner.sha_file(reference/'frozen-plan.json'),
            'reference_mechanism_summary_sha256': runner.sha_file(reference/'summary.json'),
            **{k: False for k in ('gt_model_inputs', 'test_payloads_read', 'val_parameter_fitting', 'seed_selection', 'paper_eligible')}}
    roots = [tmp_path/'first', tmp_path/'replay']
    for index, root in enumerate(roots):
        put(root/'frozen-plan.json', plan); plan_sha = runner.sha_file(root/'frozen-plan.json'); receipts = {}
        for spec in runner.RUNS:
            name = spec['run_id']; folder = root/name; folder.mkdir()
            item = birth.BirthScoreDiagnosticTrackerV2(FixedModel(), calibration(), 'seq01', 1_000_000, plan_sha256=plan_sha)
            rows = [item.step(*p) for p in short_sequence()]
            for stream, position in [('predictions.jsonl', 0), ('association.jsonl', 1)]:
                (folder/stream).write_bytes(b''.join(canonical(row[position])+b'\n' for row in rows))
            with (folder/'diagnostics.jsonl.gz').open('wb') as raw:
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as stream:
                    stream.write(b''.join(canonical(row[2])+b'\n' for row in rows))
            events = {}
            for _, _, diag in rows:
                for event in diag['events']:
                    events[event['event']] = events.get(event['event'], 0)+1
            receipt = {**spec, 'frames': 5, 'sequences': 1, 'weights_unchanged': True,
                       'sealed_association_parity': True, 'reference_M0_association_parity': True, 'sealed_prediction_parity': None,
                       'predictions': sum(len(r[0]['predictions']) for r in rows), 'nodes': sum(len(r[2]['nodes']) for r in rows),
                       'temporal_matches': sum(len(r[2]['temporal_assignments']) for r in rows), 'events': events,
                       'source_sensor_late': [sum(not r[0]['source_available'][i] for r in rows) for i in (0, 1)],
                       'selected_detections': [sum(r[0]['selected_detections'][i] for r in rows) for i in (0, 1)],
                       'sequence_commits': {'seq01': rows[-1][0]['commit_sha256']},
                       'diagnostic_sequence_commits': {'seq01': rows[-1][2]['diagnostic_commit_sha256']},
                       'predictions_sha256': runner.sha_file(folder/'predictions.jsonl'),
                       'association_sha256': runner.sha_file(folder/'association.jsonl'),
                       'diagnostics_gzip_sha256': runner.sha_file(folder/'diagnostics.jsonl.gz'),
                       'diagnostics_gzip_bytes': (folder/'diagnostics.jsonl.gz').stat().st_size}
            put(folder/'receipt.json', receipt); receipts[name] = receipt
        put(root/'summary.json', {'kind': 'birth_score_diagnostics_v2_complete', 'status': 'completed',
            'plan_sha256': plan_sha, 'reporting_scope': 'car_only', 'paper_eligible': False, 'weights_unchanged': True,
            'all_required_sealed_streams_match': True, 'elapsed_seconds': float(index+1), 'runs': receipts})
    return SimpleNamespace(first=roots[0], replay=roots[1], output=tmp_path/'audit.json', reference=reference)


def test_three_group_replay_audit_proves_common_and_birth_channels(small_replay):
    result = runner.verify_replay(small_replay)
    assert result['status'] == 'verified' and result['all_nine_streams_byte_identical'] is True
    assert len(result['runs']) == 3 and all(len(v) == 14 for v in result['file_manifests'].values())
    assert result['prediction_commits_verified'] == result['diagnostic_commits_verified'] == 30
    assert result['ground_truth_read'] is result['models_executed_during_audit'] is False
    assert all(r['birth_threshold_and_initial_score_verified'] and
               r['temporal_cost_and_existing_score_update_use_common_node_score'] for r in result['runs'].values())


@pytest.mark.parametrize('filename', ['predictions.jsonl', 'association.jsonl', 'diagnostics.jsonl.gz', 'receipt.json'])
def test_replay_tamper_is_rejected(small_replay, filename):
    with (small_replay.replay/'M4-seed-1337'/filename).open('ab') as stream:
        stream.write(b'bad')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        runner.verify_replay(small_replay)


def test_reference_sealed_association_tamper_is_rejected(small_replay):
    with (small_replay.reference/'M0-seed-2027'/'association.jsonl').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='association parity'):
        runner.verify_replay(small_replay)


def test_replay_output_must_be_outside_roots_and_create_once(small_replay):
    args = copy.copy(small_replay); args.output = args.first/'audit.json'
    with pytest.raises(ValueError, match='outside'):
        runner.verify_replay(args)
    small_replay.output.write_bytes(b'existing')
    with pytest.raises(ValueError, match='create-once'):
        runner.verify_replay(small_replay)


def test_supplement_matrix_is_exact_and_does_not_mutate_old_matrix():
    assert runner.RUNS == [{'run_id': f'M4-seed-{s}', 'mode': 'M4', 'seed': s, 'deterministic_control': False}
                           for s in (1337, 2027, 3407)]
    assert len(runner.load_independent_auditor().RUNS) == 10


@pytest.mark.parametrize('change', ['invalid_commit', 'wrong_birth_initial_score', 'birth_gate_score'])
def test_both_sides_resealed_do_not_bypass_chains_or_birth_semantics(small_replay, change):
    audit = runner.load_independent_auditor()
    for root in (small_replay.first, small_replay.replay):
        name = 'M4-seed-1337'; folder = root/name
        pred = [json.loads(line) for line in (folder/'predictions.jsonl').read_bytes().splitlines()]
        assoc = [json.loads(line) for line in (folder/'association.jsonl').read_bytes().splitlines()]
        with gzip.open(folder/'diagnostics.jsonl.gz', 'rb') as stream:
            diag = [json.loads(line) for line in stream]
        if change == 'invalid_commit':
            pred[0]['commit_sha256'] = 'f'*64
        elif change == 'wrong_birth_initial_score':
            event = next(e for e in diag[0]['events'] if e['event'] == 'birth')
            output = next(p for p in pred[0]['predictions'] if p['track_id'] == event['track_id'])
            output['score'] = .1
        else:
            diag[0]['nodes'][0]['birth_effective_score'] = .001
        if change != 'invalid_commit':
            pred_tip = diag_tip = '0'*64
            for p, a, d in zip(pred, assoc, diag):
                p['previous_commit_sha256'] = pred_tip
                p['commit_sha256'] = audit.digest({k: v for k, v in p.items() if k != 'commit_sha256'})
                pred_tip = p['commit_sha256']
                d['prediction_commit_sha256'] = pred_tip; d['previous_diagnostic_commit_sha256'] = diag_tip
                d['association_frame_sha256'] = audit.digest(a)
                d['diagnostic_commit_sha256'] = audit.digest({k: v for k, v in d.items() if k != 'diagnostic_commit_sha256'})
                diag_tip = d['diagnostic_commit_sha256']
        (folder/'predictions.jsonl').write_bytes(b''.join(canonical(p)+b'\n' for p in pred))
        with (folder/'diagnostics.jsonl.gz').open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as stream:
                stream.write(b''.join(canonical(d)+b'\n' for d in diag))
        receipt = audit.read_json(folder/'receipt.json')
        for filename, key in [('predictions.jsonl', 'predictions_sha256'), ('diagnostics.jsonl.gz', 'diagnostics_gzip_sha256')]:
            receipt[key] = audit.file_hash(folder/filename)
        receipt['diagnostics_gzip_bytes'] = (folder/'diagnostics.jsonl.gz').stat().st_size
        receipt['sequence_commits']['seq01'] = pred[-1]['commit_sha256']
        receipt['diagnostic_sequence_commits']['seq01'] = diag[-1]['diagnostic_commit_sha256']
        put(folder/'receipt.json', receipt)
        summary = audit.read_json(root/'summary.json'); summary['runs'][name] = receipt
        put(root/'summary.json', summary)
    with pytest.raises(ValueError):
        runner.verify_replay(small_replay)
    assert not small_replay.output.exists()
