"""Synthetic-only train selection, real fresh processes and independent
evaluation."""
import copy
import datetime
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest
from test_paper_pipeline import native_frame

from tools.event_track_v2x.scan_paper_resources import freeze_scan, plan_scan, run_scan, run_stage_with_eta
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery
from transvision.models.event_track_v2x.paper_native_cache import NativePaperCache, write_native_cache
from transvision.models.event_track_v2x.paper_protocol import SEEDS, PaperProtocol
from transvision.models.event_track_v2x.paper_runtime import default_configuration
from transvision.models.event_track_v2x.tracking_v2 import physical_world


def bound(path):
    return dict(path=str(path), sha256=sha_file(path))


def asset(path, value):
    path.write_bytes(canonical(value))
    return bound(path)


@pytest.fixture
def spec(tmp_path):
    root = tmp_path / 'cache'
    frames = [native_frame(1, sequence_id=s, dataset_split='train') for s in ('0003', '0007')]
    digest = write_native_cache(root, frames, split='train', producer={'fit_split': 'train'}, fixture=True)
    cache = NativePaperCache(root, digest)
    events, gt = [], []
    for seq in ('0003', '0007'):
        entry, metadata = next(v for k, v in cache.index.items() if k[0] == seq)
        entry, metadata = json.loads(entry), json.loads(metadata)
        delivery = CacheDelivery(seq, 'vehicle-side', metadata['frame_id'], 1_100_000, entry['frame_sha256'])
        frame = cache.load_arrived(delivery, 1_100_000)
        means, _ = physical_world(frame, [0], 1_000_000)
        events.append(dict(sequence_id=seq, frame_id='out', reference_us=1_000_000, decision_us=1_100_000, event_id='e', deliveries=[asdict(delivery)]))
        gt.append(
            dict(
                sequence_id=seq,
                frame_id='out',
                box_reference_timestamp_us=1_000_000,
                ego_translation_world=metadata['lidar_to_world_translation'],
                objects=[dict(track_id='synthetic', class_label='car', mean=means[0].tolist())]))
    schedule, gt_file = tmp_path / 'schedule.jsonl', tmp_path / 'gt.jsonl'
    schedule.write_bytes(b''.join(canonical(e) + b'\n' for e in events))
    gt_file.write_bytes(b''.join(canonical(g) + b'\n' for g in gt))
    protocol = asdict(PaperProtocol('v2v4real', 'train'))
    gt_manifest = asset(
        tmp_path / 'gt-manifest.json',
        dict(
            kind='rbf_paper_evaluation_gt_v1',
            protocol=protocol,
            fixture=True,
            ground_truth=dict(path='gt.jsonl', sha256=sha_file(gt_file)),
            roi=dict(kind='strict_radial_xy', radius_m=50.),
            sequence_clusters={
                '0003': 'a',
                '0007': 'b'
            },
            frames=2))
    return dict(
        protocol=protocol,
        fixture=True,
        seeds=list(SEEDS),
        objective='HOTA',
        constraints=dict(peak_rss_bytes=1e12, state_file_bytes=1e10, p95_seconds=60.),
        cache_manifest=bound(root / 'manifest.json'),
        schedule=bound(schedule),
        gt_manifest=gt_manifest,
        environment_lock=asset(tmp_path / 'fixture-environment.json', dict(fixture=True, purpose='software test')),
        candidates=[dict(name='geometry', configuration=asset(tmp_path / 'geometry.json', default_configuration('geometry')))])


def test_scan_plan_freezes_configs_and_rejects_split_seed_or_asset_changes(tmp_path, spec):
    plan = plan_scan(spec, tmp_path / 'plan')
    assert len(plan['jobs']) == 3 and not plan['full_dataset_verified']
    assert plan['jobs'][0]['configuration']['method'] == 'geometry'
    with pytest.raises(ValueError, match='new directory'):
        plan_scan(spec, tmp_path / 'plan')
    for changed in (dict(spec, protocol=asdict(PaperProtocol('v2v4real', 'official_test'))), dict(spec, seeds=[1337]), dict(spec, constraints={'p95_seconds': float('nan')})):
        with pytest.raises(ValueError):
            plan_scan(changed, tmp_path / 'rejected')
        assert not (tmp_path / 'rejected').exists()
    Path(spec['schedule']['path']).write_text('changed')
    with pytest.raises(ValueError, match='changed asset'):
        plan_scan(spec, tmp_path / 'changed')


def test_learned_candidates_require_actual_checkpoints(tmp_path, spec):
    candidate = dict(
        name='rbf',
        configuration=asset(tmp_path / 'rbf.json', default_configuration('rbf')),
        checkpoints={str(s): dict(path=str(tmp_path / 'absent'), sha256='0' * 64)
                     for s in SEEDS})
    with pytest.raises(ValueError, match='changed asset'):
        plan_scan(dict(spec, candidates=[candidate]), tmp_path / 'plan')
    assert not (tmp_path / 'plan').exists()


def test_failed_worker_preserves_failure_without_completion_receipt(tmp_path, spec, monkeypatch):
    plan_scan(spec, tmp_path / 'plan')

    def fail(argv, env, stdout_path, stderr_path, timeout, *, job_id, stage):
        Path(stdout_path).write_bytes(b'')
        Path(stderr_path).write_bytes(b'synthetic failure')
        return 23

    monkeypatch.setattr('tools.event_track_v2x.scan_paper_resources.run_stage_with_eta', fail)
    with pytest.raises(RuntimeError, match='failed: 23'):
        run_scan(tmp_path / 'plan/plan.json', sha_file(tmp_path / 'plan/plan.json'), tmp_path / 'run', python=sys.executable, evaluator_python=sys.executable, timeout=60)
    assert not (tmp_path / 'run/receipt.json').exists()
    failure = json.loads((tmp_path / 'run/failure.json').read_bytes())
    assert failure['completed_jobs'] == []
    assert (tmp_path / 'run/geometry-1337/replay.stderr').read_bytes() == b'synthetic failure'


def test_child_eta_forwards_only_progress_and_keeps_original_logs(tmp_path, capsys):
    code = ('import json; '
            'print(json.dumps(dict(kind="rbf_experiment_progress_v1", stage="paper_replay_events", '
            'completed=2, total=4, eta_seconds=3, credential="must-not-forward")), flush=True); '
            'print("private child diagnostic", flush=True)')
    stdout, stderr = tmp_path / 'child.stdout', tmp_path / 'child.stderr'
    returncode = run_stage_with_eta([sys.executable, '-c', code], dict(os.environ), stdout,
                                    stderr, 10, job_id='fixture-1337', stage='replay')
    assert returncode == 0 and stderr.read_bytes() == b''
    assert b'must-not-forward' in stdout.read_bytes()
    public = capsys.readouterr().out
    rows = [json.loads(line) for line in public.splitlines()]
    assert len(rows) == 1
    finish = rows[0].pop('estimated_finish_utc')
    assert abs((datetime.datetime.fromisoformat(finish) -
                datetime.datetime.now(datetime.timezone.utc)).total_seconds() - 3) < 2
    assert rows == [dict(kind='rbf_child_experiment_progress_v1', job_id='fixture-1337',
                         subprocess_stage='replay', stage='paper_replay_events', completed=2,
                         total=4, eta_seconds=3, eta_status='estimated',
                         experiment_acceptance_proven=False)]
    assert 'must-not-forward' not in public and 'private child diagnostic' not in public


def test_child_eta_is_visible_before_child_exits(tmp_path, capsys):
    code = ('import json,time; '
            'print(json.dumps(dict(kind="rbf_experiment_progress_v1", stage="paper_replay_events", '
            'completed=1, total=4, eta_seconds=9)), flush=True); time.sleep(3)')
    result = []

    def run():
        result.append(run_stage_with_eta([sys.executable, '-c', code], dict(os.environ),
                                         tmp_path / 'live.stdout', tmp_path / 'live.stderr',
                                         10, job_id='fixture-2027', stage='replay'))

    worker = threading.Thread(target=run)
    worker.start()
    visible = ''
    deadline = time.monotonic() + 2.5
    while time.monotonic() < deadline and 'rbf_child_experiment_progress_v1' not in visible:
        visible += capsys.readouterr().out
        time.sleep(.05)
    assert 'rbf_child_experiment_progress_v1' in visible and worker.is_alive()
    worker.join(timeout=5)
    assert not worker.is_alive() and result == [0]


@pytest.mark.parametrize('max_rss', [1e12, 1.])
def test_fresh_process_scan_and_independent_train_freeze(tmp_path, spec, max_rss):
    spec['constraints']['peak_rss_bytes'] = max_rss
    evaluator = os.environ.get('RBF_EVALUATOR_PYTHON')
    if not evaluator:
        pytest.skip('independent evaluator environment not supplied')
    plan_scan(spec, tmp_path / 'plan')
    plan_path = tmp_path / 'plan/plan.json'
    receipt = run_scan(plan_path, sha_file(plan_path), tmp_path / 'run', python=sys.executable, evaluator_python=evaluator, timeout=60)
    assert receipt['fixture'] and len(receipt['jobs']) == 3
    assert not receipt['paper_results_verified']
    if max_rss == 1.:
        with pytest.raises(ValueError, match='no candidate meets'):
            freeze_scan(plan_path, sha_file(plan_path), tmp_path / 'run', sha_file(tmp_path / 'run/receipt.json'), tmp_path / 'freeze')
        assert not (tmp_path / 'freeze').exists()
        return
    frozen = freeze_scan(plan_path, sha_file(plan_path), tmp_path / 'run', sha_file(tmp_path / 'run/receipt.json'), tmp_path / 'freeze')
    assert frozen['selected_candidate'] == 'geometry' and frozen['selection_split'] == 'train'
    assert not frozen['equal_resources_claimed'] and not frozen['full_dataset_verified']
    assert len(frozen['candidates'][0]['trials']) == 3
    assert all(t['measured']['peak_rss_bytes'] > 0 for t in frozen['candidates'][0]['trials'])
    assert json.loads((tmp_path / 'freeze/configuration.json').read_bytes()) == default_configuration('geometry')
    with pytest.raises(ValueError, match='new directory'):
        freeze_scan(plan_path, sha_file(plan_path), tmp_path / 'run', sha_file(tmp_path / 'run/receipt.json'), tmp_path / 'freeze')
    damaged_receipt = copy.deepcopy(receipt)
    del damaged_receipt['jobs'][0]['files']['replay/resources.json']
    receipt_path = tmp_path / 'run/receipt.json'
    receipt_path.write_bytes(canonical(damaged_receipt))
    with pytest.raises(ValueError, match='incomplete artifact evidence'):
        freeze_scan(plan_path, sha_file(plan_path), tmp_path / 'run', sha_file(receipt_path), tmp_path / 'incomplete')
    receipt_path.write_bytes(canonical(receipt))
    report = tmp_path / 'run/geometry-1337/evaluation/report.json'
    data = json.loads(report.read_bytes())
    data['metrics']['HOTA'] = .987654321
    report.write_bytes(canonical(data))
    with pytest.raises(ValueError, match='artifact changed'):
        freeze_scan(plan_path, sha_file(plan_path), tmp_path / 'run', sha_file(tmp_path / 'run/receipt.json'), tmp_path / 'changed')
    assert not (tmp_path / 'changed').exists()
