"""Synthetic assets only: prepare -> fit -> freeze -> all internal backends."""
import json
from dataclasses import asdict

import pytest
from test_detection_cache_v2 import _build, _sources
from test_paper_pipeline import native_frame

from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery, VerifiedForestCache
from transvision.models.event_track_v2x.paper_native_cache import NativePaperCache, write_native_cache
from transvision.models.event_track_v2x.paper_protocol import PaperProtocol


@pytest.mark.parametrize('dataset', ['spd', 'v2v4real'])
def test_dual_format_preparation_training_and_internal_baselines(tmp_path, dataset):
    from tools.event_track_v2x.prepare_paper_rows import prepare
    from tools.event_track_v2x.train_forest_identity import FitConfig
    from tools.event_track_v2x.train_paper_identity import fit
    from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
    from transvision.models.event_track_v2x.forest_supervision import AnnotationIdentity
    from transvision.models.event_track_v2x.forest_tracking import PaperForestTrackingConfig
    from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
    from transvision.models.event_track_v2x.paper_runtime import default_configuration, replay
    if dataset == 'spd':
        sources = _sources(tmp_path, split='train', empty_vehicle=False)
        cache = VerifiedForestCache(sources[-1], _build(sources))
    else:
        frames = [
            native_frame(sequence_id=seq, side=side, agent_mask=index + 1, dataset_split='train') for seq in ('0003', '0007')
            for index, side in enumerate(('vehicle-side', 'infrastructure-side'))
        ]
        root = tmp_path / 'native-cache'
        sha = write_native_cache(root, frames, split='train', producer={'fit_split': 'train'}, fixture=True)
        cache = NativePaperCache(root, sha)
    frames, links, schedule, events = [], [], [], []
    for seq in ('0003', '0007'):
        pair = {}
        deliveries = []
        annotations = {}
        for (scene, side, fid), (entry, metadata) in cache.index.items():
            if scene != seq:
                continue
            entry, metadata = json.loads(entry), json.loads(metadata)
            d = CacheDelivery(scene, side, fid, metadata['box_reference_timestamp_us'] + 100_000, entry['frame_sha256'])
            frame = cache.load_arrived(d, d.arrival_us)
            source = 0 if side == 'vehicle-side' else 1
            ann = [AnnotationIdentity(seq, source, fid, str(i), seq + '-' + side + '-' + str(i)) for i in range(frame.count)]
            frames.append(
                dict(
                    sequence_id=seq,
                    source_id=source,
                    frame_id=fid,
                    timestamp_us=metadata['box_reference_timestamp_us'],
                    boxes=frame.states[:, :7].tolist(),
                    annotations=[asdict(a) for a in ann]))
            pair[side] = fid
            annotations[side] = ann
            deliveries.append(asdict(d))
        links.extend([a.reference, b.reference] for a, b in zip(annotations['vehicle-side'], annotations['infrastructure-side']))
        reference = min(d['arrival_us'] for d in deliveries) - 100_000
        schedule.append(dict(sequence_id=seq, vehicle_frame=pair['vehicle-side'], infrastructure_frame=pair['infrastructure-side'], box_reference_timestamp_us=reference))
        events.append(dict(sequence_id=seq, frame_id='out', reference_us=reference, decision_us=reference + 100_000, event_id='event', deliveries=deliveries))
    label_path = tmp_path / 'supervision.json'
    label_path.write_bytes(
        canonical(dict(kind='rbf_train_supervision_v1', dataset=dataset, split='train', fixture=True, frames=frames, links=links, schedule=schedule, provenance={})))
    rows = tmp_path / 'rows'
    manifest = prepare(cache.root, cache.manifest_sha256, label_path, sha_file(label_path), rows, dataset=dataset, fixture=True)
    assert manifest['row_protocol']['class_scope'] == ['car', 'bicycle', 'pedestrian']
    output = tmp_path / 'fit'
    fitted = fit(
        rows,
        sha_file(rows / 'manifest.json'),
        output,
        protocol=PaperProtocol(dataset, 'train'),
        groups={
            '0003': 'a',
            '0007': 'b'
        },
        config=FitConfig(epochs=1, batch_size=8, hidden=8, heads=2, dropout=0.),
        seeds=(1337, ),
        fixture=True)
    scorer, checkpoint = load_identity_checkpoint(output / 'seed-1337', fitted['seeds'][0]['sha256'], config=PaperForestTrackingConfig())
    binding = dict(
        candidate_protocol=PaperProtocol(dataset, 'train').candidates,
        dataset=dataset,
        fit_split='train',
        seed=1337,
        checkpoint_sha256=fitted['seeds'][0]['sha256'],
        frozen_cache_identity=checkpoint['frozen_cache_identity'])
    for method in ('rbf', 'geometry', 'single_vehicle', 'single_remote', 'topk', 'irreversible', 'mht', 'jpda', 'pkf'):
        run = tmp_path / ('run-' + method)
        result = replay(
            cache,
            events,
            run,
            protocol=PaperProtocol(dataset, 'train'),
            configuration=default_configuration(method),
            scorer=GeometryForestScorer() if method == 'geometry' else scorer,
            model_binding={} if method == 'geometry' else binding,
            fixture=True)
        assert result['completed_events'] == 2 and result['fixture']
        assert not result['paper_results_verified']
        resources = json.loads((run / 'resources.json').read_bytes())
        assert len(resources['databases']) == 2
        assert not resources['equal_resources_verified']
        if method == 'rbf':
            from transvision.models.event_track_v2x.covered_completion_tracking import CoveredCompletionTracker
            predictions = [json.loads(line) for line in (run / 'predictions.jsonl').read_bytes().splitlines()]
            for p in predictions:
                db = result['databases'][p['sequence_id']]
                restored = CoveredCompletionTracker.open(run / db['path'], expected_prediction_sha256=p['commit_sha256'], expected_database_sha256=db['sha256'])
                assert restored.close() == db['sha256']
    from tools.event_track_v2x.train_paper_priority import export
    from transvision.models.event_track_v2x.allocation_training import fit_priority, load_priority
    teacher_config = default_configuration('rbf', allocation='teacher')
    teacher_config['state']['expansion_budget'] = 1
    teacher = tmp_path / 'teacher'
    replay(cache, events, teacher, protocol=PaperProtocol(dataset, 'train'), configuration=teacher_config, scorer=scorer, model_binding=binding, fixture=True)
    priority_data = tmp_path / 'priority-data'
    priority_manifest = export(teacher, sha_file(teacher / 'receipt.json'), priority_data)
    priority_fit = tmp_path / 'priority-fit'
    priority_result = fit_priority(
        priority_data, sha_file(priority_data / 'manifest.json'), priority_fit, epochs=1, hidden=4, require_full_train=False, select_best_train_holdout=True)
    policy, _ = load_priority(priority_fit / '1337', priority_result['seeds'][0]['checkpoint_sha256'], binding=priority_manifest['binding'], require_full_train=False)
    factor_hashes = []
    for strategy in ('fixed', 'bound', 'learned'):
        configuration = dict(teacher_config, allocation=strategy)
        output = tmp_path / ('allocation-' + strategy)
        replay(
            cache,
            events,
            output,
            protocol=PaperProtocol(dataset, 'train'),
            configuration=configuration,
            scorer=scorer,
            model_binding=binding,
            allocation_policy=policy if strategy == 'learned' else None,
            fixture=True)
        factor_hashes.append([json.loads(line)['factor_rows_sha256'] for line in (output / 'audit.jsonl').read_bytes().splitlines()])
    assert factor_hashes[0] == factor_hashes[1] == factor_hashes[2]
    import os
    import subprocess
    evaluator_python = os.environ.get('RBF_EVALUATOR_PYTHON')
    if evaluator_python:
        from tools.event_track_v2x.train_paper_identity import ROOT
        from transvision.models.event_track_v2x.tracking_v2 import physical_world
        gt = []
        for event in events:
            d = next(CacheDelivery(**d) for d in event['deliveries'] if d['side'] == 'vehicle-side')
            frame = cache.load_arrived(d, d.arrival_us)
            means, _ = physical_world(frame, list(range(frame.count)), event['reference_us'])
            gt.append(
                dict(
                    sequence_id=event['sequence_id'],
                    frame_id=event['frame_id'],
                    box_reference_timestamp_us=event['reference_us'],
                    ego_translation_world=frame.metadata['lidar_to_world_translation'],
                    objects=[dict(track_id='fixture-' + str(i), class_label='car', mean=mean.tolist()) for i, mean in enumerate(means)]))
        gt_path = tmp_path / 'fixture-ground-truth.jsonl'
        gt_path.write_bytes(b''.join(canonical(g) + b'\n' for g in gt))
        gt_manifest = tmp_path / 'gt-manifest.json'
        gt_manifest.write_bytes(
            canonical(
                dict(
                    kind='rbf_paper_evaluation_gt_v1',
                    fixture=True,
                    protocol=asdict(PaperProtocol(dataset, 'train')),
                    frames=len(gt),
                    ground_truth=dict(path=gt_path.name, sha256=sha_file(gt_path)),
                    roi=dict(kind='strict_radial_xy', radius_m=50.),
                    sequence_clusters={
                        '0003': 'a',
                        '0007': 'b'
                    })))
        for method in ('rbf', 'geometry', 'mht'):
            run = tmp_path / ('run-' + method)
            evaluated = tmp_path / ('evaluated-' + method)
            completed = subprocess.run([
                evaluator_python,
                str(ROOT / 'tools/event_track_v2x/evaluate_paper.py'), '--gt-manifest',
                str(gt_manifest), '--gt-sha256',
                sha_file(gt_manifest), '--replay',
                str(run), '--receipt-sha256',
                sha_file(run / 'receipt.json'), '--output',
                str(evaluated)
            ],
                                       env=dict(os.environ, OMP_NUM_THREADS='1'),
                                       capture_output=True,
                                       text=True,
                                       timeout=60)
            assert completed.returncode == 0, completed.stdout + completed.stderr
            report = json.loads((evaluated / 'report.json').read_bytes())
            assert report['fixture'] and report['status'] == 'evaluated'
            assert set(report['per_sequence']) == {'0003', '0007'}
