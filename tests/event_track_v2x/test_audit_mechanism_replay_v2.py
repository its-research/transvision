"""Independent audit fail-closed tests using tiny actual diagnostic streams."""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_tracking_v2 import FixedModel, calibration
from test_source_mask_v2 import short_sequence
from transvision.models.event_track_v2x.tracking_mechanisms_v2 import MechanismDiagnosticTrackerV2

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('audit_mechanism_test', ROOT/'tools/event_track_v2x/audit_mechanism_replay_v2.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit.canonical(value))


def write_streams(folder, rows):
    for name, index in [('predictions.jsonl', 0), ('association.jsonl', 1)]:
        (folder/name).write_bytes(b''.join(audit.canonical(row[index])+b'\n' for row in rows))
    with (folder/'diagnostics.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=6, mtime=0) as stream:
            stream.write(b''.join(audit.canonical(row[2])+b'\n' for row in rows))


def reseal(root, name):
    folder = root/name
    receipt = audit.read_json(folder/'receipt.json')
    for filename, key in [('predictions.jsonl', 'predictions_sha256'), ('association.jsonl', 'association_sha256'),
                          ('diagnostics.jsonl.gz', 'diagnostics_gzip_sha256')]:
        receipt[key] = audit.file_hash(folder/filename)
    receipt['diagnostics_gzip_bytes'] = (folder/'diagnostics.jsonl.gz').stat().st_size
    put(folder/'receipt.json', receipt)
    summary = audit.read_json(root/'summary.json'); summary['runs'][name] = receipt
    put(root/'summary.json', summary)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'FRAME_COUNT', 5)
    monkeypatch.setattr(audit, 'SEQUENCE_COUNT', 1)
    plan = {'kind': 'mechanism_diagnostics_v2_plan', 'runs': audit.RUNS, 'official_validation_frames': 5,
            'sequences': 1, 'cache_sha256': audit.CACHE_SHA, 'schedule_sha256': audit.SCHEDULE_SHA,
            'device': 'cpu', 'cpu_threads': 1, 'reporting_scope': 'car_only', 'config': {'deadline_us': 100000},
            **{key: False for key in ('gt_model_inputs', 'test_payloads_read', 'val_parameter_fitting', 'seed_selection', 'paper_eligible')}}
    roots = [tmp_path/'first', tmp_path/'replay']
    for count, root in enumerate(roots):
        put(root/'frozen-plan.json', plan)
        plan_sha = audit.file_hash(root/'frozen-plan.json')
        receipts = {}
        for spec in audit.RUNS:
            name, mode = spec['run_id'], spec['mode']; folder = root/name; folder.mkdir()
            tracker = MechanismDiagnosticTrackerV2(FixedModel(), calibration(), 'seq01', 1_000_000,
                                                    mode=mode, plan_sha256=plan_sha)
            rows = [tracker.step(*pair) for pair in short_sequence()]
            write_streams(folder, rows)
            events = {}
            for _, _, diag in rows:
                for event in diag['events']:
                    events[event['event']] = events.get(event['event'], 0)+1
            receipt = {**spec, 'frames': 5, 'sequences': 1, 'weights_unchanged': True,
                       'predictions': sum(len(row[0]['predictions']) for row in rows),
                       'nodes': sum(len(row[2]['nodes']) for row in rows),
                       'temporal_matches': sum(len(row[2]['temporal_assignments']) for row in rows),
                       'events': events, 'source_sensor_late': [sum(not row[0]['source_available'][i] for row in rows) for i in (0, 1)],
                       'selected_detections': [sum(row[0]['selected_detections'][i] for row in rows) for i in (0, 1)],
                       'sequence_commits': {'seq01': rows[-1][0]['commit_sha256']},
                       'diagnostic_sequence_commits': {'seq01': rows[-1][2]['diagnostic_commit_sha256']},
                       'sealed_association_parity': True if mode != 'M1' else None,
                       'sealed_prediction_parity': True if mode == 'M0' else None,
                       'predictions_sha256': audit.file_hash(folder/'predictions.jsonl'),
                       'association_sha256': audit.file_hash(folder/'association.jsonl'),
                       'diagnostics_gzip_sha256': audit.file_hash(folder/'diagnostics.jsonl.gz'),
                       'diagnostics_gzip_bytes': (folder/'diagnostics.jsonl.gz').stat().st_size}
            put(folder/'receipt.json', receipt)
            receipts[name] = receipt
        put(root/'summary.json', {'kind': 'mechanism_diagnostics_v2_complete', 'status': 'completed',
            'plan_sha256': plan_sha, 'reporting_scope': 'car_only', 'paper_eligible': False,
            'weights_unchanged': True, 'all_required_sealed_streams_match': True,
            'elapsed_seconds': float(count+1), 'runs': receipts})
    return SimpleNamespace(first=roots[0], replay=roots[1], output=tmp_path/'audit.json')


def test_full_byte_replay_chains_and_manifest(runs):
    result = audit.audit(runs.first, runs.replay, runs.output)
    assert result['status'] == 'verified'
    assert result['all_30_streams_byte_identical'] is True
    assert result['all_10_receipts_byte_identical'] is True
    assert len(result['runs']) == 10
    assert all(len(manifest) == 42 for manifest in result['file_manifests'].values())
    assert result['prediction_commits_verified'] == result['diagnostic_commits_verified'] == 100
    assert result['summary_difference_allowed'] == ['elapsed_seconds']
    assert result['ground_truth_read'] is result['detector_or_model_executed'] is False
    assert result['paper_eligible'] is False
    assert runs.output.is_file()


@pytest.mark.parametrize('which', ['predictions.jsonl', 'association.jsonl', 'diagnostics.jsonl.gz', 'receipt.json'])
def test_one_replay_file_tamper_rejected(runs, which):
    with (runs.replay/'M1-all-unmatched'/which).open('ab') as stream:
        stream.write(b'x')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        audit.audit(runs.first, runs.replay, runs.output)
    assert not runs.output.exists()


@pytest.mark.parametrize('kind', ['extra_file', 'extra_dir', 'symlink'])
def test_inventory_is_exact_and_no_symlinks(runs, kind):
    path = runs.first/'unexpected'
    if kind == 'extra_file':
        path.write_bytes(b'x')
    elif kind == 'extra_dir':
        path.mkdir()
    else:
        path.symlink_to(runs.first/'summary.json')
    with pytest.raises(ValueError):
        audit.audit(runs.first, runs.replay, runs.output)


@pytest.mark.parametrize('field,value', [('status', 'running'), ('paper_eligible', True), ('weights_unchanged', False)])
def test_summary_changes_beyond_elapsed_rejected(runs, field, value):
    summary = audit.read_json(runs.replay/'summary.json'); summary[field] = value
    put(runs.replay/'summary.json', summary)
    with pytest.raises(ValueError):
        audit.audit(runs.first, runs.replay, runs.output)


def test_same_directory_is_not_independent_replay(runs):
    with pytest.raises(ValueError, match='distinct'):
        audit.audit(runs.first, runs.first, runs.output)


def test_output_is_create_once_and_outside_sealed_roots(runs):
    runs.output.write_bytes(b'preserved')
    with pytest.raises(ValueError, match='create-once'):
        audit.audit(runs.first, runs.replay, runs.output)
    assert runs.output.read_bytes() == b'preserved'
    with pytest.raises(ValueError, match='outside'):
        audit.audit(runs.first, runs.replay, runs.first/'new-audit.json')


def rewrite_first_record(root, mutation, *, recompute_commits):
    name = 'M1-all-unmatched'; folder = root/name
    pred = [json.loads(line) for line in (folder/'predictions.jsonl').read_bytes().splitlines()]
    assoc = [json.loads(line) for line in (folder/'association.jsonl').read_bytes().splitlines()]
    with gzip.open(folder/'diagnostics.jsonl.gz', 'rb') as stream:
        diag = [json.loads(line) for line in stream]
    mutation(pred[0], assoc[0], diag[0])
    if recompute_commits:
        previous_pred = previous_diag = audit.ZERO
        for p, a, d in zip(pred, assoc, diag):
            p['previous_commit_sha256'] = previous_pred
            p['commit_sha256'] = audit.digest({k: v for k, v in p.items() if k != 'commit_sha256'})
            previous_pred = p['commit_sha256']
            d['previous_diagnostic_commit_sha256'] = previous_diag
            d['prediction_commit_sha256'] = p['commit_sha256']
            d['association_frame_sha256'] = audit.digest(a)
            d['diagnostic_commit_sha256'] = audit.digest({k: v for k, v in d.items() if k != 'diagnostic_commit_sha256'})
            previous_diag = d['diagnostic_commit_sha256']
        receipt = audit.read_json(folder/'receipt.json')
        receipt['sequence_commits']['seq01'] = previous_pred
        receipt['diagnostic_sequence_commits']['seq01'] = previous_diag
        put(folder/'receipt.json', receipt)
    write_streams(folder, list(zip(pred, assoc, diag)))
    reseal(root, name)


@pytest.mark.parametrize('key', ['commit_sha256', 'previous_commit_sha256'])
def test_rehashing_both_files_cannot_hide_broken_prediction_chain(runs, key):
    def mutation(p, a, d):
        p[key] = 'a'*64
    for root in (runs.first, runs.replay):
        rewrite_first_record(root, mutation, recompute_commits=False)
    with pytest.raises(ValueError, match='prediction'):
        audit.audit(runs.first, runs.replay, runs.output)


@pytest.mark.parametrize('case', ['duplicate_id', 'future', 'source_index', 'score_mass', 'association_binding'])
def test_self_consistent_hashes_do_not_replace_semantic_checks(runs, case):
    def mutation(p, a, d):
        if case == 'duplicate_id':
            p['predictions'].append(copy.deepcopy(p['predictions'][0]))
            d['tracks_after'] += 1
        elif case == 'future':
            p['source_information_timestamp_us'][0] = p['decision_timestamp_us']+1
        elif case == 'source_index':
            d['nodes'][0]['source_raw_index'] = 999
        elif case == 'score_mass':
            d['nodes'][0]['score'] = .00001
        else:
            a['frame_id'] = 'different-frame'
    for root in (runs.first, runs.replay):
        rewrite_first_record(root, mutation, recompute_commits=True)
    with pytest.raises(ValueError):
        audit.audit(runs.first, runs.replay, runs.output)
    assert not runs.output.exists()


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{ "x": 1}', b'{"x":1}\n'])
def test_json_is_canonical_nonfinite_free_and_unique(raw):
    with pytest.raises(ValueError):
        audit.decode(raw)


def test_truncated_jsonl_or_different_stream_counts_rejected(runs):
    for root in (runs.first, runs.replay):
        folder = root/'M1-all-unmatched'
        lines = (folder/'predictions.jsonl').read_bytes().splitlines(keepends=True)
        (folder/'predictions.jsonl').write_bytes(b''.join(lines[:-1]))
        reseal(root, 'M1-all-unmatched')
    with pytest.raises(ValueError, match='line counts'):
        audit.audit(runs.first, runs.replay, runs.output)
