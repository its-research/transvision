"""Read-only independent raw-cache -> runtime 203-feature/causal-context audit.

Imports no tracker, Torch, SciPy or production feature adapter. Reuses the
accepted original cache/schedule and forward evidence; does not repeat their
full numeric/NN checks. A historical failed prefix is only a software fixture.
"""
import argparse
import collections
import datetime
import hashlib
import io
import json
import math
import sqlite3
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path('/Volumes/Data/test/recover-before-fuse')
ATOL = RTOL = 1e-8
FEATURE = 'recover-before-fuse-physical-world-cache203-v1'
CONTEXT = 'persistent-forest-arrival-row-parent-context-v1'
PROTOCOL = 'rbf-all-class-top64-v1'
SIDES = {'vehicle-side': 0, 'infrastructure-side': 1}
RAW_KEYS = {'sequence_id', 'node', 'detection_index', 'state_us', 'mean',
            'covariance', 'score', 'features', 'source_cache_sha256'}


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(v):
    return hashlib.sha256(canonical(v)).hexdigest()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def close(actual, expected, label):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all(), label + ' shape/finite'
    assert np.allclose(a, b, atol=ATOL, rtol=RTOL), label + ' numerical mismatch'
    return float(np.max(np.abs(a - b))) if a.size else 0.


def xyz_euler(matrix):
    """Independent normalized matrix-to-quaternion then extrinsic xyz angles.

    The input is a near-orthogonal column rotation. Normalize the quaternion
    from its largest diagonal/trace branch, matching the frozen pose contract's
    SO(3) normalization without calling its SciPy conversion.
    """
    m = np.asarray(matrix, dtype=np.float64)
    assert m.shape == (3, 3) and np.isfinite(m).all()
    diagonal = np.diag(m)
    tr = float(diagonal.sum())
    branch = int(np.argmax([*diagonal, tr]))
    q = np.zeros(4, dtype=np.float64)
    if branch == 3:
        q[:3] = [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]
        q[3] = 1 + tr
    else:
        i, j, k = branch, (branch + 1) % 3, (branch + 2) % 3
        q[i] = 1 - tr + 2 * m[i, i]
        q[j], q[k] = m[j, i] + m[i, j], m[k, i] + m[i, k]
        q[3] = m[k, j] - m[j, k]
    q /= np.linalg.norm(q)
    x, y, z, w = q
    normalized = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])
    pitch = math.asin(float(np.clip(-normalized[2, 0], -1., 1.)))
    if abs(math.cos(pitch)) < 1e-7:
        roll, yaw = math.atan2(-normalized[1, 2], normalized[1, 1]), 0.
    else:
        roll = math.atan2(normalized[2, 1], normalized[2, 2])
        yaw = math.atan2(normalized[1, 0], normalized[0, 0])
    return np.array([roll, pitch, yaw])


def physical_rows(meta, arrays, selected, arrival, origin, frame_sha):
    """Invert legacy dimensions/yaw and derive the world-state Jacobian."""
    rotation = np.asarray(meta['lidar_to_world_row_rotation'], dtype=np.float64)
    translation = np.asarray(meta['lidar_to_world_translation'], dtype=np.float64)
    legacy = arrays['states'][selected].astype(np.float64)
    covariance = arrays['covariances'][selected].astype(np.float64)
    n = len(selected)
    local = legacy[:, [0, 1, 2, 4, 3, 5, 6, 7, 8]].copy()
    local[:, 6] = (-local[:, 6] - math.pi/2 + math.pi) % (2*math.pi) - math.pi
    mean = local.copy()
    mean[:, :3] = local[:, :3] @ rotation + translation
    cosine, sine = np.cos(local[:, 6]), np.sin(local[:, 6])
    headings = np.column_stack((cosine, sine, np.zeros(n))) @ rotation
    tangent = np.column_stack((-sine, cosine, np.zeros(n))) @ rotation
    denominator = np.sum(headings[:, :2]**2, axis=1)
    assert (denominator >= 1e-8).all(), 'undefined world heading'
    mean[:, 6] = np.arctan2(headings[:, 1], headings[:, 0])
    mean[:, 7:9] = local[:, 7:9] @ rotation[:2, :2]
    # Combine legacy-axis permutation/sign inversion and world transformation
    # in a single Jacobian; do not consume the producer's covariance values.
    legacy_j = np.eye(9)[[0, 1, 2, 4, 3, 5, 6, 7, 8]]
    legacy_j[6, 6] = -1
    world_j = np.broadcast_to(np.eye(9), (n, 9, 9)).copy()
    world_j[:, :3, :3] = rotation.T
    world_j[:, 7:9, 7:9] = rotation[:2, :2].T
    world_j[:, 6, 6] = (headings[:, 0]*tangent[:, 1] - headings[:, 1]*tangent[:, 0]) / denominator
    jacobian = world_j @ legacy_j
    world_cov = jacobian @ covariance @ jacobian.transpose(0, 2, 1)
    state_us = meta['box_reference_timestamp_us']
    information = max(state_us, meta['source_image_timestamp_us'])
    assert information <= arrival
    source = SIDES[meta['side']]
    features = np.zeros((n, 203), dtype=np.float64)
    features[:, :6] = mean[:, :6] / [100, 100, 20, 20, 20, 10]
    features[:, 6:10] = np.column_stack((np.sin(mean[:, 6]), np.cos(mean[:, 6]), mean[:, 7:9]/30))
    features[:, 10:138] = arrays['appearance'][selected]
    features[:, 138:141] = np.eye(3)[arrays['class_indices'][selected]]
    features[:, 141:145] = [(state_us-origin)/1e8, (meta['source_image_timestamp_us']-state_us)/1e6,
                             (arrival-information)/1e6, source]
    features[:, 145:190] = world_cov[:, *np.triu_indices(9)] / 100
    features[:, 190:193] = translation / [100, 100, 20]
    features[:, 193:196] = xyz_euler(rotation.T) / math.pi
    features[:, 196:200] = [1, math.log(2)/8, 0, 0]
    features[:, 200:203] = np.column_stack((arrays['raw_scores'][selected], arrays['scores'][selected], arrays['appearance_valid'][selected]))
    rows = []
    for j, index in enumerate(selected):
        node = dict(source_id=source, frame_id=meta['frame_id'], information_us=information, arrival_us=arrival,
                    node_id=digest([meta['sequence_id'], source, meta['frame_id'], int(index), frame_sha]))
        rows.append(dict(sequence_id=meta['sequence_id'], node=node, detection_index=int(index), state_us=state_us,
                         mean=mean[j], covariance=world_cov[j], features=features[j],
                         score=float(arrays['scores'][index]), source_cache_sha256=frame_sha))
    return rows


class CacheAdmission:
    def __init__(self, seed, checkpoint):
        self.seed = seed
        b = ROOT/f'artifacts/rbf-original-nested-cache-replay-input-readback-v1-20261001/seed{seed}'
        m = ROOT/f'artifacts/rbf-original-cache-CPU-metadata-export-v1-20261001/seed{seed}'
        p = ROOT/f'artifacts/rbf-original-cache-full-numeric-causal-input-audit-v1-20261001/seed{seed}/acceptance.json'
        proof = json.loads(p.read_bytes())
        assert proof['kind'] == 'rbf_full_original_cache_numeric_causal_candidate_admission_v1' and proof['seed'] == seed
        for k in ('all_16338_frame_schema_numeric_pose_covariance_appearance_and_digest_checks_pass',
                  'all_7445_supplied_causal_events_and_first_arrivals_verified',
                  'all_class_raw_score_005_top64_candidate_ids_and_order_match_admitted_model_rows'):
            assert proof[k] is True
        assert proof['GT_read'] is False and proof['test_read'] is False
        assert proof['original_arrival_schedule_inferred_or_completed'] is False
        assert sha(b/'byte-readback-receipt.json') == proof['byte_readback_sha256']
        assert sha(m/'events.json') == proof['event_schedule_sha256']
        assert sha(m/'independent-acceptance.json') == proof['metadata_and_original_schedule_admission_sha256']
        metadata_proof = json.loads((m/'independent-acceptance.json').read_bytes())
        assert sha(m/'metadata.jsonl') == metadata_proof['artifacts']['metadata']['sha256']
        byte = json.loads((b/'byte-readback-receipt.json').read_bytes())
        assert byte['all_archive_and_manifest_member_bytes_verified'] is True
        self.root = Path(byte['cache_root'])
        assert sha(self.root/'manifest.json') == byte['manifest_sha256'] == proof['manifest_sha256']
        self.frames = {}
        for line in (m/'metadata.jsonl').open():
            v = json.loads(line); meta = v['metadata']; key = (meta['sequence_id'], meta['side'], meta['frame_id'])
            assert key not in self.frames
            self.frames[key] = v
        event_document = json.loads((m/'events.json').read_bytes())
        assert event_document['cache_manifest_sha256'] == proof['manifest_sha256']
        self.origins, self.events = event_document['origin_us_by_sequence'], collections.defaultdict(list)
        for e in event_document['events']:
            self.events[e['sequence_id']].append(e)
        assert sum(map(len, self.events.values())) == 7445 and len(self.events) == 46
        forward = ROOT/f'artifacts/rbf-joint-identity-all-row-GPU-admission-v1-20261001/seed{seed}'
        assert sha(forward/'independent-byte-coverage-receipt.json') == proof['forward_byte_admission_sha256']
        numeric = ROOT/f'artifacts/rbf-joint-identity-full-independent-numpy-v1-20261001/seed{seed}/numeric-v1/completion.json'
        assert sha(numeric) == proof['full_forward_numeric_admission_sha256']
        assert json.loads(numeric.read_bytes())['full_independent_numeric_pass'] is True
        forward_receipt = json.loads((forward/'receipt.json').read_bytes())
        forward_proof = json.loads((forward/'independent-byte-coverage-receipt.json').read_bytes())
        assert sha(forward/'receipt.json') == forward_proof['artifacts']['receipt']['sha256']
        assert sha(checkpoint) == forward_receipt['plan']['checkpoint']['sha256']
        self.checkpoint = json.loads(Path(checkpoint).read_bytes())
        assert self.checkpoint['seed'] == seed and self.checkpoint['labels_in_model_inputs'] is False
        source = ROOT/'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/source.bytes'
        source_sha = '038fa8118c9540d91073fbb8bf594fb6abefe8347f13bb69dfb006d27fcfda03'
        assert sha(source) == source_sha
        with tarfile.open(source,'r:gz') as archive:
            for filename in ('forest_tracking.py','forest_row_context.py','forest_potentials.py','prediction_features.py','tracking_v2.py'):
                name = 'transvision/models/event_track_v2x/'+filename
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == self.checkpoint['source_sha256'][name]
        self.proof, self.memo = proof, collections.OrderedDict()
        self.binding = dict(seed=seed, cache_admission_sha256=sha(p), cache_manifest_sha256=proof['manifest_sha256'],
                            checkpoint_sha256=sha(checkpoint), schedule_sha256=proof['event_schedule_sha256'],
                            independently_admitted_forward_sha256=proof['full_forward_numeric_admission_sha256'],
                            original_frozen_runtime_source_sha256=source_sha)

    def frame(self, delivery):
        key = (delivery['sequence_id'], delivery['side'], delivery['frame_id'])
        record = self.frames[key]; meta, entry = record['metadata'], record['manifest_entry']
        assert entry['frame_sha256'] == delivery['frame_sha256'], 'source frame identity'
        assert max(meta['box_reference_timestamp_us'], meta['source_image_timestamp_us']) <= delivery['arrival_us'], 'unavailable source payload'
        memo_key = key + (delivery['arrival_us'],)
        if memo_key in self.memo:
            return self.memo[memo_key]
        def member(field):
            spec = entry[field]; relative = Path(spec['path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            p = self.root/relative
            assert p.resolve().is_relative_to(self.root.resolve()) and not p.is_symlink()
            raw = p.read_bytes()
            assert len(raw) == spec['bytes'] and hashlib.sha256(raw).hexdigest() == spec['sha256'], 'consumed cache member byte binding'
            return raw
        assert member('metadata') == canonical(meta)
        with np.load(io.BytesIO(member('arrays')), allow_pickle=False) as z:
            arrays = {k: np.asarray(z[k]) for k in ('states','raw_scores','scores','class_indices','covariances','appearance','appearance_valid')}
        selected = sorted(np.flatnonzero(arrays['raw_scores'] >= .05).tolist(), key=lambda i: (-float(arrays['raw_scores'][i]), i))[:64]
        result = physical_rows(meta, arrays, selected, delivery['arrival_us'], self.origins[key[0]], entry['frame_sha256'])
        self.memo[memo_key] = result
        if len(self.memo) > 256:
            self.memo.popitem(last=False)
        return result


def expected_context(previous, current, config, decision):
    assert current['node']['arrival_us'] <= decision
    candidates = []
    # Scan complete committed/earlier-batch raw history, independently of the
    # producer SQL time-window and heap. No inferred events or GT query filter.
    for i, p in enumerate(previous):
        assert p['node']['arrival_us'] <= decision
        if (p['node']['source_id'], p['node']['frame_id']) == (current['node']['source_id'], current['node']['frame_id']):
            continue
        dt_us = current['state_us'] - p['state_us']
        if abs(dt_us) > config['max_parent_gap_us']:
            continue
        propagated = p['mean'][:2] + p['mean'][7:9] * (dt_us / 1e6)
        distance = float(np.linalg.norm(current['mean'][:2] - propagated))
        if distance <= config['gate_distance_m']:
            candidates.append((distance, i))
    chosen = sorted(i for _, i in sorted(candidates)[:config['parent_limit']])
    return chosen + [len(previous)]


def verify_database(path, expected_sha, admission, *, allow_prefix=False, progress=None):
    assert sha(path) == expected_sha
    connection = sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro', uri=True)
    connection.execute('PRAGMA query_only=ON')
    assert connection.execute('PRAGMA integrity_check').fetchone() == ('ok',)
    meta = {k: json.loads(v) for k,v in connection.execute('SELECT k,v FROM meta')}
    sequence, config = meta['sequence_id'], meta['config']; state = config['state']
    assert state['candidate_protocol'] == PROTOCOL
    assert state['parent_limit'] == 8 and state['gate_distance_m'] == 8. and state['max_parent_gap_us'] == 2000000
    ck = admission.checkpoint
    signature = digest(['learned_forest', FEATURE, ck['model_sha256'],
                        dict(max_nodes=9, max_pairs=81, geometry_weight=ck['geometry_weight'], process_noise=state['process_noise'])])
    scorer_binding = digest([CONTEXT, FEATURE, signature, config])
    assert meta['scorer_binding'] == scorer_binding, 'checkpoint scorer/config binding'
    cache_binding = digest([admission.proof['manifest_sha256'], sequence, admission.origins[sequence], .05, 64, 100000, scorer_binding, PROTOCOL])
    assert meta['cache_binding'] == cache_binding, 'raw cache/origin/selection binding'
    events = connection.execute('SELECT ordinal,event_id,prediction,audit FROM events ORDER BY ordinal').fetchall()
    original = admission.events[sequence]
    assert 0 < len(events) <= len(original)
    if not allow_prefix:
        assert len(events) == len(original), 'incomplete original schedule'
    previous, seen = [], {}
    classes = np.zeros(3, dtype=int)
    maximum = dict(mean=0., covariance=0., features=0., score=0.)
    contexts = duplicate_count = 0
    for ordinal, (stored_ordinal, event_id, prediction_raw, audit_raw) in enumerate(events):
        e = original[ordinal]; audit = json.loads(audit_raw); prediction = json.loads(prediction_raw)
        assert stored_ordinal == ordinal and event_id == e['event_id'] == audit['event_id']
        assert prediction['sequence_id'] == sequence and prediction['frame_id'] == e['frame_id']
        assert prediction['box_reference_timestamp_us'] == e['reference_us'] and prediction['decision_timestamp_us'] == e['decision_us']
        ing = audit['cache_ingestion']
        assert ing['request_sha256'] == digest([e['frame_id'],e['reference_us'],e['decision_us'],e['deliveries']])
        assert ing['cache_manifest_sha256'] == admission.proof['manifest_sha256'] and ing['configuration_sha256'] == cache_binding
        assert ing['scorer_context_recipe'] == CONTEXT and audit['scorer_binding'] == scorer_binding
        assert ing['candidate_protocol'] == PROTOCOL and ing['class_scope'] == ['car','bicycle','pedestrian'] and ing['evaluation_class'] == 'car'
        assert ing['first_arrival_preserved'] is True and ing['gt_model_inputs'] is False and ing['old_row_rescoring'] is False
        assert audit['rescored_rows'] == []
        deliveries, duplicates = [], []
        for d in sorted(e['deliveries'],key=lambda x:(x['arrival_us'],x['side'],x['frame_id'])):
            key = (d['side'],d['frame_id']); assert d['arrival_us'] <= e['decision_us']
            if key in seen:
                assert d['arrival_us'] >= seen[key][0] and d['frame_sha256'] == seen[key][1]
                duplicates.append(dict(receipt=d,first_arrival_us=seen[key][0])); continue
            if ordinal:
                assert d['arrival_us'] >= original[ordinal-1]['decision_us']
            seen[key] = d['arrival_us'],d['frame_sha256'];deliveries.append(d)
        assert ing['new_deliveries'] == deliveries and ing['duplicate_deliveries'] == duplicates
        duplicate_count += len(duplicates)
        new = [row for d in deliveries for row in admission.frame(d)]
        assert len(new) == ing['new_observations'] == audit['new_observations'] == len(ing['scorer_context_indices'])
        assert len(audit['appended_rows']) == len(new)
        for offset, expected in enumerate(new):
            index = len(previous)
            record = connection.execute('SELECT i,node_id,source,frame,detection_index,state_us,arrival_us,score,raw,sha FROM observations WHERE i=?',(index,)).fetchone()
            assert record is not None, 'missing committed raw observation'
            raw = json.loads(record[8]); assert set(raw) == RAW_KEYS and digest(raw) == record[9]
            assert raw['node'] == expected['node'] and raw['sequence_id'] == expected['sequence_id']
            assert raw['detection_index'] == expected['detection_index'] and raw['state_us'] == expected['state_us']
            assert raw['source_cache_sha256'] == expected['source_cache_sha256']
            assert record[:7] == (index,raw['node']['node_id'],raw['node']['source_id'],raw['node']['frame_id'],raw['detection_index'],raw['state_us'],raw['node']['arrival_us'])
            assert record[7] == raw['score']
            for name in maximum:
                maximum[name] = max(maximum[name], close(raw[name],expected[name],name))
            context = expected_context(previous,expected,state,e['decision_us'])
            assert ing['scorer_context_indices'][offset] == context, 'incomplete/illegal raw arrival context'
            parents = [-1] + context[:-1]
            assert [pair[0] for pair in audit['appended_rows'][offset]] == parents, 'raw context/factor support mismatch'
            assert [p for p, in connection.execute('SELECT p FROM potentials WHERE i=? ORDER BY p',(index,))] == parents
            classes[int(np.argmax(expected['features'][138:141]))] += 1
            previous.append(expected);contexts += 1
        assert audit['observation_count'] == len(previous)
        if progress and (ordinal+1)%16 == 0:
            progress(dict(stage='independent_raw_cache203_context',sequence_id=sequence,completed_events=ordinal+1,total_events=len(events),completed_rows=len(previous),ETA_seconds=None,ETA_reason='heterogeneous full historical raw candidate scan; no timing model'))
    assert connection.execute('SELECT count(*) FROM observations').fetchone()[0] == len(previous)
    assert {(side,frame):(arrival,h) for side,frame,arrival,h in connection.execute('SELECT * FROM cache_receipts')} == seen
    connection.close();assert sha(path) == expected_sha, 'database changed during read-only audit'
    if not allow_prefix:
        assert len(previous) == admission.proof['per_sequence_rows'][sequence]
    return dict(kind='rbf_independent_runtime_cache203_context_sequence_admission_v1',sequence_id=sequence,
                database_sha256=expected_sha,**admission.binding,events=len(events),original_sequence_events=len(original),
                rows=len(previous),delivered_frames=len(seen),duplicate_deliveries=duplicate_count,contexts=contexts,
                all_203_features_and_world_state_values_bound_to_raw_cache=True,all_contexts_bound_to_exact_first_arrival_history=True,
                class_counts=classes.tolist(),max_abs_error=maximum,atol=ATOL,rtol=RTOL,
                complete_original_sequence=not allow_prefix,scope='historical prefix software fixture only' if allow_prefix else 'original supplied paired train schedule',
                original_schedule_inferred=False,GT_read=False,test_read=False,NN_numeric_repeated=False,
                complete_online_method_accepted=False,full_stage_two_complete=False,paper_performance_complete=False)


def main():
    p = argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True,choices=(1337,2027,3407))
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--database',type=Path,required=True)
    p.add_argument('--database-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--historical-prefix-software-fixture',action='store_true');args=p.parse_args()
    assert not args.output.exists();args.output.mkdir(parents=True)
    started = time.monotonic();binding=dict(source_sha256=sha(__file__),seed=args.seed,database_sha256=args.database_sha256,
                                           fixture=args.historical_prefix_software_fixture,atol=ATOL,rtol=RTOL)
    (args.output/'binding.json').write_bytes(canonical(binding)+b'\n')
    try:
        admission = CacheAdmission(args.seed,args.checkpoint)
        result = verify_database(args.database,args.database_sha256,admission,allow_prefix=args.historical_prefix_software_fixture,
                                 progress=lambda v: print(json.dumps(v),flush=True))
        assert sha(__file__) == binding['source_sha256']
        result.update(source_sha256=binding['source_sha256'],elapsed_seconds=time.monotonic()-started,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (args.output/'acceptance.json').write_bytes(canonical(result)+b'\n');print(json.dumps(result),flush=True)
    except BaseException as e:
        (args.output/'failure.json').write_bytes(canonical(dict(binding,type=type(e).__name__,message=str(e),experiment_accepted=False))+b'\n');raise


if __name__ == '__main__':
    main()
