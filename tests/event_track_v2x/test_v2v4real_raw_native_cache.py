import io
import json
import tarfile

import numpy as np
import pytest

from tools.event_track_v2x.build_v2v4real_raw_native_cache import (
    build, digest, RAW_LAYOUT, RAW_RULE, RAW_MANIFEST_FIELDS, MOTION_KIND,
)
from transvision.models.event_track_v2x.detection_cache_v2 import BOX_LAYOUT, canonical, sha_file
from transvision.models.event_track_v2x.paper_evaluation_policy import VEHICLE_PROTOCOL, vehicle_binding
from transvision.models.event_track_v2x.paper_native_cache import NATIVE_FEATURE, NativePaperCache, NativeDetectionFrame


def fixture(tmp_path, bad=None, *, split='train'):
    """Synthetic contract values only; no real estimator/model is being admitted."""
    raw_root = tmp_path/'raw'; raw_root.mkdir()
    motion_root = tmp_path/'motion'; motion_root.mkdir()
    records, arrays, poses, payloads = [], {}, [], {}
    transform = [[0., -1., 0., 10.], [1., 0., 0., 20.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
    def write(path, body):
        path.write_bytes(canonical(body))
        return dict(path=path.name, sha256=sha_file(path))
    def npz(body):
        stream = io.BytesIO(); np.savez_compressed(stream, **body)
        return stream.getvalue()
    for ordinal in range(2):
        for side in ('vehicle-side', 'infrastructure-side'):
            n = 0 if (ordinal, side) == (1, 'vehicle-side') else 2
            label = f'{ordinal}-{side}'
            # Identical overlapping candidates must both survive: no NMS.
            state = np.tile([1., 2., 3., 2., 4., 1., -.3], (n, 1)).astype(np.float64)
            feature = np.zeros((n, 128), np.float32); feature[:, 0] = 1.
            body = dict(states_source_legacy7=state, raw_scores=np.array([.8, .05][:n], np.float32),
                class_indices=np.zeros(n, np.int64), appearance=feature, appearance_valid=np.ones(n, bool))
            if bad == 'nonzero_class' and n: body['class_indices'][0] = 1
            if bad == 'raw_gt' and n: body['gt_id'] = np.arange(n)
            if bad == 'below_cutoff' and n: body['raw_scores'][1] = .049
            raw = npz(body)
            row = dict(sequence_id='scene', frame_id=f'{ordinal:06d}', frame_ordinal=ordinal, side=side,
                source_to_world=transform, pcd_sha256='b'*64, payload='payloads/'+label+'.npz',
                payload_sha256=digest(raw), candidates=n)
            pose = dict(sequence_id='scene', frame_key=row['frame_id'], frame_ordinal=ordinal,
                is_ego=side == 'vehicle-side', cav_id='0' if side == 'vehicle-side' else '1',
                source_to_world=transform, lidar_pose=transform, pcd_sha256='b'*64,
                pcd_path='pcd/'+label+'.pcd', pcd_size_bytes=10)
            records.append(row); poses.append(pose); payloads[row['payload']] = raw; arrays[label] = body
    if bad == 'missing_empty':
        removed = next(x for x in records if x['candidates'] == 0)
        records.remove(removed); del payloads[removed['payload']]
    if bad == 'pose_mismatch': records[0]['source_to_world'] = np.eye(4).tolist()
    if bad == 'pose_gt': poses[0]['vehicles'] = {'secret': 1}
    index = b''.join(canonical(r)+b'\n' for r in records)
    (raw_root/'native-feature-index').write_bytes(index)
    projection = tmp_path/'frames.jsonl'; projection.write_bytes(b''.join(canonical(p)+b'\n' for p in poses))
    anchor = npz({'unused': np.ones(1)})
    manifest = dict(anchor_array_sha256='d'*64, anchor_file_sha256=digest(anchor),
        anchor_generation='official_build_postprocessor_generate_anchor_box',
        appearance_valid_candidates=sum(r['candidates'] for r in records), candidate_protocol='rbf-all-class-top64-v1',
        checkpoint_seed=1337, checkpoint_sha256='a'*64, cohort='official_train_raw_native_features',
        covariance_exported=False, downstream_covariance_velocity_and_identity_admission_pending=True,
        export_groups=['scene'], feature_method=NATIVE_FEATURE, gt_read=False,
        kind='v2v4real_train_raw_native_features_v1', matching_candidate_rule=RAW_RULE, official_test_read=False,
        paper_metric=False, partition_sha256='e'*64, prepared_projection_sha256='f'*64,
        protocol_id='v2v4real-nominal-10hz-formal-v1', rows=len(records), rows_sha256=digest(index),
        sequence_ids=['scene'], split='train', split_admission_sha256='c'*64, state_layout=RAW_LAYOUT,
        total_candidates=sum(r['candidates'] for r in records), velocity_exported=False)
    assert set(manifest) == RAW_MANIFEST_FIELDS
    if split == 'official_test':
        projection_record = write(raw_root/'projection-manifest', dict(kind='v2v4real_pose_lidar_projection_v2',
            dataset_split='train' if bad == 'test_projection_is_train' else 'test', gt_in_projection=False,
            frames_sha256=sha_file(projection), source_frame_count=len(records)))
        admission_record = write(raw_root/'split-admission', dict(kind='v2v4real_official_split_admission_v1',
            split='train' if bad == 'test_admission_is_train' else 'official_test',
            projection_manifest_sha256=projection_record['sha256'], official_original_split_preserved=True,
            official_split_membership_verified=True, official_test_unchanged=True))
        manifest.update(kind='v2v4real_official_test_raw_native_features_v1', cohort='official_test_raw_native_features',
            split='official_test', official_test_read=True, projection_manifest=projection_record, split_admission=admission_record,
            prepared_projection_sha256=projection_record['sha256'], split_admission_sha256=admission_record['sha256'])
    if bad == 'test_as_train': manifest['split'] = 'official_test'
    if bad == 'nms': manifest['matching_candidate_rule'] = 'rotated_nms_0.15'
    write(raw_root/'native-feature-manifest', manifest)
    with tarfile.open(raw_root/'native-features', 'w:gz') as archive:
        for name, value in dict(payloads, **{'manifest.json': canonical(manifest), 'predictions.jsonl': index, 'anchors.npy': anchor}).items():
            member = tarfile.TarInfo(name); member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
        if bad == 'tar_duplicate':
            name = next(iter(payloads)); member = tarfile.TarInfo(name); member.size = len(payloads[name])
            archive.addfile(member, io.BytesIO(payloads[name]))
    clock = tmp_path/'clock.json'
    write(clock, dict(protocol_id=manifest['protocol_id'], dataset='v2v4real', nominal_rate_hz=10,
                     period_us=100000, measured=bad == 'measured_clock'))
    source = motion_root/'method.py'; source.write_text('# synthetic fixture; not a real admitted estimator\n')
    method = dict(kind='v2v4real_causal_motion_covariance_method_v1', velocity_method='fixture explicit supplied velocity',
        initial_observation_policy='fixture supplied prior', full_covariance_method='fixture supplied SPD matrix',
        implementation=dict(path=source.name, sha256=sha_file(source)))
    if bad == 'missing_initial_policy': del method['initial_observation_policy']
    method_record = write(motion_root/'method.json', method)
    model = motion_root/'model.bin'; model.write_bytes(b'fixture')
    model_record = dict(path=model.name, sha256=sha_file(model))
    fit = dict(kind='v2v4real_motion_covariance_train_fit_v1', fit_split='train', official_test_used=False,
        model_sha256=model_record['sha256'], method_sha256=method_record['sha256'], fit_groups=['train-fit'])
    if bad == 'motion_fit_test': fit['official_test_used'] = True
    fit_record = write(motion_root/'fit.json', fit)
    proof = dict(kind='v2v4real_motion_covariance_independent_admission_v1', causal_source_only_verified=True,
        initial_observation_policy_verified=True, full_nine_state_covariance_verified=True, independent_acceptance_verified=True,
        method_sha256=method_record['sha256'], model_sha256=model_record['sha256'], fit_receipt_sha256=fit_record['sha256'])
    if bad == 'unaccepted_motion': proof['independent_acceptance_verified'] = False
    proof_record = write(motion_root/'admission.json', proof)
    motions, motion_arrays = [], {}
    for i, row in enumerate(records):
        n = row['candidates']
        velocity = np.tile([3., -2.], (n, 1)).astype(np.float64)
        covariance = np.tile(np.eye(9), (n, 1, 1)); covariance[:, 0, 7] = covariance[:, 7, 0] = .03
        if bad == 'non_spd' and n: covariance[:, 8, 8] = -1
        if bad == 'velocity_nan' and n: velocity[0, 0] = np.nan
        if bad == 'velocity_cast': velocity = velocity.astype(np.float32)
        motion_body = dict(velocity_source=velocity, covariances=covariance)
        if bad == 'motion_gt': motion_body['gt_ids'] = np.arange(n)
        motion_payload = motion_root/f'motion-{i}.npz'; motion_payload.write_bytes(npz(motion_body))
        row_sha = digest(canonical(row))
        dependencies = [digest(canonical(prior)) for prior in records if prior['side'] == row['side']
                        and prior['frame_ordinal'] <= row['frame_ordinal']]
        if bad == 'future' and i == 0: dependencies.append(digest(canonical(records[-2])))
        if bad == 'cross_source' and i == 0: dependencies.append(digest(canonical(records[1])))
        if bad == 'no_current' and i == 0: dependencies = []
        motions.append(dict(raw_row_sha256=row_sha, arrays=dict(path=motion_payload.name, sha256=sha_file(motion_payload)), dependencies=dependencies))
        motion_arrays[row_sha] = motion_body
    if bad == 'missing_motion_empty': motions = [item for item, row in zip(motions, records) if row['candidates']]
    config_record = write(motion_root/'detector-config.json', dict(fixture=True))
    contract = dict(kind=MOTION_KIND, dataset='v2v4real', split=split, raw_manifest_sha256=sha_file(raw_root/'native-feature-manifest'),
        raw_index_sha256=manifest['rows_sha256'], raw_archive_sha256=sha_file(raw_root/'native-features'),
        projection_frames_sha256=sha_file(projection), clock_sha256=sha_file(clock), state_layout=BOX_LAYOUT,
        coordinate_system='source_lidar', velocity_units='metres_per_second', reference_time='source_capture',
        gt_read_online=False, cross_source_inputs=False, detector_config=config_record,
        method=method_record, model=model_record, fit_receipt=fit_record, independent_admission=proof_record,
        sequence_origin_us={'scene': 700000}, frames=motions)
    if bad == 'wrong_units': contract['velocity_units'] = 'kilometres_per_hour'
    if bad == 'wrong_motion_binding': contract['raw_archive_sha256'] = '0'*64
    if bad == 'missing_origin': contract['sequence_origin_us'] = {}
    if bad == 'test_motion_is_train': contract['split'] = 'train'
    contract_path = motion_root/'contract.json'; write(contract_path, contract)
    calibration = tmp_path/'calibration.json'
    write(calibration, dict(slope=1., intercept=.2, fit_split='train', fit_groups=['calibration-fit']))
    calibration_proof = tmp_path/'calibration-receipt.json'
    write(calibration_proof, dict(kind='v2v4real_native_vehicle_existence_calibration_v1', protocol_id=VEHICLE_PROTOCOL,
        fit_split='train', official_test_used=False, gt_written_to_calibration_artifact=False,
        calibration_sha256=sha_file(calibration), checkpoint_sha256='a'*64,
        evaluation_class_binding=vehicle_binding(), fit_groups=['calibration-fit']))
    args = dict(raw_root=raw_root, raw_manifest_sha256=sha_file(raw_root/'native-feature-manifest'),
        raw_archive_sha256=sha_file(raw_root/'native-features'), projection_frames=projection, projection_frames_sha256=sha_file(projection),
        motion_contract=contract_path, motion_contract_sha256=sha_file(contract_path), clock=clock, clock_sha256=sha_file(clock),
        calibration=calibration, calibration_sha256=sha_file(calibration), calibration_receipt=calibration_proof,
        calibration_receipt_sha256=sha_file(calibration_proof), output=tmp_path/'output', fixture=True)
    if split == 'official_test' and bad != 'missing_test_producer':
        receipt = tmp_path/'test-export-receipt.json'
        write(receipt, dict(kind='v2v4real_official_test_raw_export_v1', split='official_test',
            gt_read=False, official_test_read=True, paper_metric=False,
            raw_manifest_sha256=args['raw_manifest_sha256'], raw_archive_sha256=args['raw_archive_sha256'],
            checkpoint_sha256='a'*64, rows_sha256=manifest['rows_sha256'],
            projection_manifest_sha256=manifest['prepared_projection_sha256'], projection_frames_sha256=sha_file(projection),
            split_admission_sha256='0'*64 if bad == 'wrong_test_producer' else manifest['split_admission_sha256']))
        args.update(raw_export_receipt=receipt, raw_export_receipt_sha256=sha_file(receipt))
    return args, records, arrays, motion_arrays


def test_raw_bridge_keeps_top64_pose_nominal_time_and_all_empty_frames(tmp_path):
    args, rows, raw_arrays, motion_arrays = fixture(tmp_path)
    before = {p: sha_file(p) for p in tmp_path.rglob('*') if p.is_file()}
    progress = []
    receipt = build(**args, progress=progress.append)
    assert receipt['frames'] == 4 and receipt['detections'] == 6
    assert not receipt['gt_payload_opened'] and not receipt['detector_executed']
    assert not receipt['velocity_estimator_executed'] and not receipt['independent_acceptance_verified']
    assert receipt['motion_independent_admission_imported'] and not receipt['motion_semantics_independently_reverified']
    assert progress == [dict(completed_rows=i, total_rows=4) for i in range(1, 5)]
    assert all(sha_file(path) == sha for path, sha in before.items())
    cache = NativePaperCache(args['output'], receipt['cache_sha256'])
    manifest = json.loads(cache.manifest_json)
    assert manifest['producer']['evaluation_class'] == 'vehicle'
    assert manifest['producer']['timestamp_basis'].endswith('not_measured')
    assert manifest['producer']['source_sha256'] == receipt['source_sha256']
    assert manifest['producer']['source_modules'] == receipt['source_modules']
    required_modules = {'tools.event_track_v2x.build_v2v4real_raw_native_cache',
        'tools.event_track_v2x.fit_v2v4real_vehicle_calibration',
        *('transvision.models.event_track_v2x.'+name for name in (
          'detection_cache_v2', 'paper_native_cache', 'paper_calibration', 'paper_evaluation_policy',
          'experiment_progress', 'forest_cache_stream', 'paper_protocol', 'prediction_features',
          'v2v4real_ground_truth', 'v2v4real_vehicle_calibration'))}
    assert required_modules <= set(receipt['source_modules'])
    for module in required_modules:
        actual = __import__('sys').modules[module].__file__
        assert receipt['source_modules'][module] == actual
        assert receipt['source_sha256'][actual] == sha_file(actual)
    for row, entry in zip(rows, manifest['frames']):
        frame = NativeDetectionFrame.load(args['output'], entry)
        original = raw_arrays[f"{row['frame_ordinal']}-{row['side']}"]
        motion = motion_arrays[digest(canonical(row))]
        assert frame.count == row['candidates']
        assert frame.states[:, :7].tobytes() == original['states_source_legacy7'].tobytes()
        assert frame.states[:, 7:].tobytes() == motion['velocity_source'].tobytes()
        assert frame.covariances.tobytes() == motion['covariances'].tobytes()
        for name in ('raw_scores', 'class_indices', 'appearance', 'appearance_valid'):
            assert getattr(frame, name).tobytes() == original[name].tobytes()
        assert frame.metadata['box_reference_timestamp_us'] == 700000 + row['frame_ordinal']*100000
        np.testing.assert_array_equal(frame.metadata['lidar_to_world_row_rotation'], np.asarray(row['source_to_world'])[:3, :3].T)
    with pytest.raises(ValueError, match='fresh'):
        build(**args)


@pytest.mark.parametrize('bad', ['missing_empty', 'pose_mismatch', 'pose_gt', 'test_as_train', 'nms',
    'measured_clock', 'missing_initial_policy', 'motion_fit_test', 'unaccepted_motion', 'future', 'cross_source',
    'no_current', 'missing_motion_empty', 'wrong_units', 'wrong_motion_binding', 'missing_origin', 'nonzero_class',
    'raw_gt', 'below_cutoff', 'tar_duplicate', 'non_spd', 'velocity_nan', 'velocity_cast', 'motion_gt'])
def test_raw_bridge_rejects_missing_or_forged_contract_and_support(tmp_path, bad):
    args, *_ = fixture(tmp_path, bad)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        build(**args, progress=lambda value: None)
    assert not (tmp_path/'output-raw-bridge-receipt.json').exists()


def test_motion_contract_required_and_never_invented(tmp_path):
    args, *_ = fixture(tmp_path)
    args['motion_contract'].unlink()
    with pytest.raises(ValueError, match='SHA-256 bound input'):
        build(**args)
    assert not args['output'].exists()


def test_distinct_official_test_envelope_uses_train_fitted_models(tmp_path):
    args, *_ = fixture(tmp_path, split='official_test')
    receipt = build(**args, progress=lambda value: None)
    assert receipt['split'] == 'official_test'
    cache = NativePaperCache(args['output'], receipt['cache_sha256'])
    manifest = json.loads(cache.manifest_json)
    assert manifest['split'] == 'official_test' and manifest['producer']['fit_split'] == 'train'
    assert manifest['frame_count'] == 4 and manifest['detection_count'] == 6
    assert not receipt['independent_acceptance_verified']
    for entry in manifest['frames']:
        frame = NativeDetectionFrame.load(args['output'], entry)
        assert frame.metadata['dataset_split'] == 'official_test'
        assert frame.metadata['calibration_fit_split'] == 'train'


@pytest.mark.parametrize('bad', ['test_projection_is_train', 'test_admission_is_train', 'test_motion_is_train',
                               'missing_test_producer', 'wrong_test_producer'])
def test_official_test_needs_distinct_projection_split_and_producer(tmp_path, bad):
    args, *_ = fixture(tmp_path, bad, split='official_test')
    with pytest.raises(ValueError):
        build(**args, progress=lambda value: None)
    assert not args['output'].exists()


def test_production_cannot_use_synthetic_detector_config(tmp_path):
    args, *_ = fixture(tmp_path)
    args['fixture'] = False
    with pytest.raises(ValueError, match='production detector config'):
        build(**args)
    assert not args['output'].exists()


def test_runtime_source_change_fails_without_modifying_workspace_source(tmp_path, monkeypatch):
    import tools.event_track_v2x.build_v2v4real_raw_native_cache as tool
    args, *_ = fixture(tmp_path)
    temporary_copy = tmp_path/'loaded-source-fixture.py'
    temporary_copy.write_bytes(__import__('pathlib').Path(tool.__file__).read_bytes())
    original_paths = tool._loaded_source_paths()
    monkeypatch.setattr(tool, '_loaded_source_paths', lambda: dict(original_paths, temporary_source_fixture=temporary_copy))
    def mutate(value):
        if value['completed_rows'] == 1:
            with temporary_copy.open('ab') as stream:
                stream.write(b'\n# changed during the conversion\n')
    with pytest.raises(ValueError, match='source changed during conversion'):
        tool.build(**args, progress=mutate)
    assert not (tmp_path/'output-raw-bridge-receipt.json').exists()
