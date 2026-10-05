"""Independent full original seen-val physical-203 and causal-context audit.

Reuses the byte-pinned, already tested independent NumPy feature equations.
The new vectorized history scan is checked against the original scalar oracle,
and ambiguous floating-point cutoff ties use that original oracle directly.
No production tracker, Torch, SciPy, annotation, or model forward is imported.
"""
import argparse
import ast
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

R = Path('/Volumes/Data/test/recover-before-fuse')
ROWS = R / 'artifacts/rbf-matching-seen-val-target-free-original-row-contexts-v1-20261004'
V2 = R / 'artifacts/rbf-nested-seen-val-matching-V2-full-admission-v1-20261004'
CK = R / 'artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004'
REFERENCE = R / 'source-freezes/rbf-independent-runtime-cache203-context-v1-20261002/cache203.py'
SOFTWARE = R / 'artifacts/rbf-independent-runtime-cache203-context-software-v1-20261002/final-gate/software-gate.json'
PREPARED = R / 'source-freezes/rbf-matching-seen-val-target-free-row-input-candidate-v1-20261004'
FIELDS = {'features', 'mean', 'covariance', 'score', 'source', 'node_id', 'frame_id', 'cache_sha256',
          'information_us', 'arrival_us', 'state_us', 'detection_index', 'contexts', 'lengths', 'decision_us'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_reference():
    gate = json.loads(SOFTWARE.read_bytes())
    assert gate['kind'] == 'rbf_independent_runtime_cache203_context_verifier_software_gate_v1'
    assert len(gate['rejection_mutations']) == 9
    assert sha(REFERENCE) == gate['sources']['cache203.py']
    spec = importlib.util.spec_from_file_location('original_independent_cache203_reference', REFERENCE)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    assert reference.ATOL == reference.RTOL == 1e-8
    return reference


def context(previous, means, state_us, frames, current, config, decision, reference):
    """Scan all committed/earlier-batch raw observations, not producer heaps."""
    count = len(previous)
    assert current['node']['arrival_us'] <= decision
    if not count:
        return [0], False
    same = (current['node']['source_id'], current['node']['frame_id'])
    allowed = np.asarray([frame != same for frame in frames], dtype=bool)
    delta = current['state_us'] - state_us[:count]
    allowed &= np.abs(delta) <= config['max_parent_gap_us']
    indices = np.flatnonzero(allowed)
    if not len(indices):
        return [count], False
    projected = means[indices, :2] + means[indices, 7:9] * (delta[indices, None] / 1e6)
    difference = current['mean'][:2] - projected
    distances = np.sqrt(difference[:, 0]**2 + difference[:, 1]**2)
    # Preserve the original scalar norm's cutoff/tie behavior at uncertain
    # boundaries instead of relaxing tolerance or picking arbitrary parents.
    if np.any(np.abs(distances - config['gate_distance_m']) < 1e-12):
        return reference.expected_context(previous, current, config, decision), True
    eligible = distances <= config['gate_distance_m']
    indices, distances = indices[eligible], distances[eligible]
    order = np.lexsort((indices, distances))
    limit = min(config['parent_limit'], len(order))
    if len(order) > limit and abs(distances[order[limit-1]] - distances[order[limit]]) < 1e-12:
        return reference.expected_context(previous, current, config, decision), True
    chosen = sorted(map(int, indices[order[:limit]]))
    return chosen + [count], False


def software_check(reference):
    rng = np.random.default_rng(1337)
    config = dict(max_parent_gap_us=2000000, gate_distance_m=8., parent_limit=8)
    count = 0
    for size in (0, 1, 8, 9, 32, 128):
        for case in range(16):
            previous = [dict(mean=rng.normal(size=9), state_us=int(rng.integers(0, 5000000)),
                node=dict(source_id=int(rng.integers(0, 2)), frame_id=str(rng.integers(0, 8)), arrival_us=6000000))
                for _ in range(size)]
            current = dict(mean=rng.normal(size=9), state_us=2500000,
                node=dict(source_id=0, frame_id='0', arrival_us=6000000))
            if size > 8 and case % 2 == 0:
                for p in previous:
                    p['mean'][:] = 0
                    p['mean'][0] = 8. if case % 4 == 0 else 1.
                    p['state_us'] = current['state_us']
                    p['node']['frame_id'] = 'tie'
                current['mean'][:] = 0
            means = np.asarray([p['mean'] for p in previous], dtype=float).reshape(size, 9)
            states = np.asarray([p['state_us'] for p in previous], dtype=np.int64)
            frames = [(p['node']['source_id'], p['node']['frame_id']) for p in previous]
            actual, _ = context(previous, means, states, frames, current, config, 6000000, reference)
            assert actual == reference.expected_context(previous, current, config, 6000000)
            count += 1
    return count


def member(root, record):
    path = root / record['path']
    relative = Path(record['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
    assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(1337, 2027, 3407))
    parser.add_argument('--software-only', type=Path)
    parser.add_argument('--software-gate', type=Path)
    args = parser.parse_args()
    reference = load_reference()
    cases = software_check(reference)
    if args.software_only:
        new(args.software_only, dict(kind='rbf_seen_val_independent_history_vectorization_scalar_equivalence_software_gate_v1',
            software_cases=cases, includes_empty_singleton_time_gate_same_frame_radius_and_cutoff_ties=True,
            original_scalar_oracle_used_for_ambiguous_cutoffs=True, reference_sha256=sha(REFERENCE),
            source_sha256=sha(__file__), real_val_input_admitted=False))
        return
    assert args.seed in (1337, 2027, 3407)
    software_gate = json.loads(args.software_gate.read_bytes())
    assert software_gate['source_sha256'] == sha(__file__)
    assert software_gate['reference_sha256'] == sha(REFERENCE) and software_gate['software_cases'] == cases == 96
    directory = ROWS / f'seed{args.seed}'
    final = directory / 'independent-acceptance.json'
    assert not final.exists(), 'do not repeat an accepted cohort'
    manifest_path = directory / 'manifest.json'
    manifest_sha = sha(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    v2_proof_path = V2 / f'seed{args.seed}/independent-acceptance.json'
    v2 = json.loads(v2_proof_path.read_bytes())
    assert v2['full_raw_to_V2_numeric_admission'] is True and v2['seed'] == args.seed
    assert manifest['kind'] == 'rbf_prediction_only_original_arrival_row_contexts_v1'
    assert manifest['seed'] == args.seed and manifest['split'] == 'val' and manifest['no_GT_or_dummy_supervision_fields'] is True
    assert manifest['cache_manifest_sha256'] == v2['cache_manifest_sha256']
    assert manifest['candidate_protocol'] == 'rbf-all-class-top64-v1' and manifest['original_events'] == 3316
    preparation = json.loads((PREPARED/'preparation.json').read_bytes())
    producer_sha = preparation['sources']['prepare_rbf_seen_val_prediction_rows.py']['sha256']
    assert sha(PREPARED/'prepare_rbf_seen_val_prediction_rows.py') == producer_sha == manifest['source_sha256']
    input_gate_path = directory/'input-gate.json'
    input_gate = json.loads(input_gate_path.read_bytes())
    assert input_gate['source_sha256'] == producer_sha
    assert input_gate['V2_independent_acceptance_sha256'] == sha(v2_proof_path)
    assert input_gate['new_NN_numeric_acceptance_sha256'] == preparation['new_NN_full_independent_numeric_index_sha256']
    checkpoint_path = CK / f'seed{args.seed}/checkpoint'
    byte_proof = json.loads((CK / f'seed{args.seed}/acceptance.json').read_bytes())
    assert sha(checkpoint_path) == byte_proof['artifacts']['checkpoint']['sha256']
    checkpoint = json.loads(checkpoint_path.read_bytes())
    assert manifest['row_protocol'] == checkpoint['row_protocol']
    assert manifest['frozen_cache_identity'] == checkpoint['frozen_cache_identity']
    assert manifest['original_source_contract'] == input_gate['original_source_sha256'] == checkpoint['source_sha256']
    assert input_gate['checkpoint_sha256'] == sha(checkpoint_path)
    extraction = manifest['CPU_dependency_extraction']
    assert input_gate['CPU_dependency_extraction'] == extraction
    assert extraction['fake_Torch_module_used'] is False and extraction['model_NN_forward_executed'] is False
    original_root = R/'artifacts/rbf-original-joint-identity-source-byte-readback-20261001/source'
    original_tracking = original_root/'transvision/models/event_track_v2x/tracking_v2.py'
    assert sha(original_tracking) == extraction['original_file_sha256'] == checkpoint['source_sha256']['transvision/models/event_track_v2x/tracking_v2.py']
    definitions = {node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
        for node in ast.parse(original_tracking.read_text()).body
        if isinstance(node,ast.FunctionDef) and node.name in ('propagate','physical_world','ci')}
    assert definitions == extraction['selected_unchanged_function_AST_sha256']
    config = manifest['row_protocol']
    assert config['minimum_raw_score'] == .05 and config['maximum_detections'] == 64
    cache = Path(v2['cache'])
    assert sha(cache/'manifest.json') == v2['cache_manifest_sha256']
    cache_manifest = json.loads((cache/'manifest.json').read_bytes())
    frames = {}
    for entry in cache_manifest['frames']:
        meta = json.loads(member(cache, entry['metadata']).read_bytes())
        key = meta['sequence_id'], meta['side'], meta['frame_id']
        assert key not in frames
        frames[key] = entry, meta
    schedule_path = R/'artifacts/spd-mht-k4-input-recovery-20260929/schedule.json'
    assert sha(schedule_path) == manifest['original_schedule_sha256'] == v2['original_schedule_sha256']
    schedule = json.loads(schedule_path.read_bytes())
    assert schedule['contains_ground_truth'] is False and schedule['contains_system_error_offset'] is False
    assert len(schedule['frames']) == 3316 and len(manifest['shards']) == len(manifest['sequences']) == 21
    assert len(set(manifest['sequences'])) == 21
    assert [record['sequence_id'] for record in manifest['shards']] == manifest['sequences']
    assert manifest['all_empty_arrival_events_preserved'] is True and manifest['actual_measured_network_history'] is False
    count, maxima, accepted, started, last = Counter(), Counter(), [], time.monotonic(), 0
    for record in manifest['shards']:
        sequence = record['sequence_id']
        path = member(directory, {k:record[k] for k in ('path','bytes','sha256')})
        with np.load(path, allow_pickle=False) as payload:
            assert set(payload.files) == FIELDS
            arrays = {key:payload[key] for key in FIELDS}
        n = record['nodes']
        assert all(len(a) == n and not a.dtype.hasobject for a in arrays.values())
        shapes = dict(features=(n,203),mean=(n,9),covariance=(n,9,9),contexts=(n,config['parent_limit']+1))
        for key, values in arrays.items():
            assert values.shape == shapes.get(key,(n,))
            if key in ('features','mean','covariance','score'):
                assert values.dtype == np.float64 and np.isfinite(values).all()
            elif key in ('node_id','frame_id','cache_sha256'):
                assert values.dtype.kind == 'U'
            else:
                assert values.dtype == np.int64
        assert (arrays['state_us'] <= arrays['information_us']).all()
        assert (arrays['information_us'] <= arrays['arrival_us']).all()
        assert (arrays['arrival_us'] <= arrays['decision_us']).all()
        assert record['has_supervision_fields'] is False
        event_relative = Path(record['events_path'])
        assert not event_relative.is_absolute() and '..' not in event_relative.parts
        events_path = directory/event_relative
        assert events_path.resolve().is_relative_to(directory.resolve()) and not events_path.is_symlink()
        assert sha(events_path) == record['events_sha256']
        events = json.loads(events_path.read_bytes())
        original = [event for event in schedule['frames'] if event['sequence_id'] == sequence]
        assert len(events) == len(original) == record['original_events']
        origin = min(meta['box_reference_timestamp_us'] for key, (_,meta) in frames.items() if key[0] == sequence)
        previous, frame_ids, seen, classes = [], [], set(), Counter()
        expected_means = np.empty((n,9),dtype=np.float64)
        expected_states = np.empty(n,dtype=np.int64)
        empty = 0
        prior_reference = -1
        for event, actual_event in zip(original, events):
            assert type(event['box_reference_timestamp_us']) is int and event['box_reference_timestamp_us'] > prior_reference
            prior_reference = event['box_reference_timestamp_us']
            decision = event['box_reference_timestamp_us'] + 100000
            assert actual_event['original_schedule_row'] == event
            assert actual_event['decision_us'] == decision and actual_event['origin_us'] == origin
            assert actual_event['old_count'] == len(previous)
            deliveries, unavailable, new_rows = [], [], []
            for side, field in (('infrastructure-side','infrastructure_frame'),('vehicle-side','vehicle_frame')):
                entry, meta = frames[(sequence,side,event[field])]
                assert (side,event[field]) not in seen
                seen.add((side,event[field]))
                if max(meta['source_image_timestamp_us'],meta['box_reference_timestamp_us']) > decision:
                    unavailable.append(dict(side=side,frame_id=event[field],frame_sha256=entry['frame_sha256']))
                    continue
                with np.load(member(cache,entry['arrays']),allow_pickle=False) as payload:
                    values={key:payload[key] for key in payload.files}
                selected=sorted(np.flatnonzero(values['raw_scores']>=.05).tolist(),key=lambda i:(-float(values['raw_scores'][i]),i))[:64]
                deliveries.append(dict(side=side,frame_id=event[field],frame_sha256=entry['frame_sha256'],
                    declared_arrival_us=decision,selected_detection_indices=selected))
                new_rows.extend(reference.physical_rows(meta,values,selected,decision,origin,entry['frame_sha256']))
            assert actual_event['source_receipts'] == deliveries
            assert sorted(actual_event['unavailable_sources'],key=lambda d:d['side']) == sorted(unavailable,key=lambda d:d['side'])
            assert actual_event['new_count'] == len(new_rows)
            empty += not new_rows
            for expected in new_rows:
                row=len(previous);assert row<n
                node=expected['node']
                for key,value in dict(node_id=node['node_id'],frame_id=node['frame_id'],source=node['source_id'],
                    information_us=node['information_us'],arrival_us=node['arrival_us'],state_us=expected['state_us'],
                    detection_index=expected['detection_index'],cache_sha256=expected['source_cache_sha256'],decision_us=decision).items():
                    assert arrays[key][row] == value
                for key in ('mean','covariance','score','features'):
                    maxima[key]=max(maxima[key],reference.close(arrays[key][row],expected[key],key))
                indices,fallback=context(previous,expected_means,expected_states,frame_ids,expected,config,decision,reference)
                length=int(arrays['lengths'][row]);assert length==len(indices)
                assert arrays['contexts'][row,:length].tolist()==indices and (arrays['contexts'][row,length:]==-1).all()
                expected_means[row]=expected['mean'];expected_states[row]=expected['state_us']
                frame_ids.append((node['source_id'],node['frame_id']));previous.append(expected)
                count['scalar_ambiguous_context_fallbacks']+=fallback
                class_name=('car','bicycle','pedestrian')[int(np.argmax(expected['features'][138:141]))]
                count['rows']+=1;count[class_name]+=1;classes[class_name]+=1
            count['events']+=1
            now=time.monotonic()
            if now-last>20 or count['events']==3316:
                print(json.dumps(dict(stage='independent_complete_seen_val_physical203_causal_row_context',seed=args.seed,
                    completed_events=count['events'],total_events=3316,completed_rows=count['rows'],
                    ETA_seconds=(now-started)*(3316-count['events'])/count['events'],
                    ETA_scope='remaining heterogeneous original-event feature/context audit estimate',
                    max_absolute_errors=dict(maxima))),flush=True);last=now
        assert len(previous)==n and empty==record['empty_arrival_events']
        assert dict(classes) == record['candidate_class_counts']
        assert sha(path)==record['sha256'] and sha(events_path)==record['events_sha256']
        accepted.append(dict(sequence_id=sequence,nodes=n,events=len(events),source_frames=len(seen),empty_events=empty,
            shard_sha256=record['sha256'],events_sha256=record['events_sha256']))
    assert count['events']==3316 and count['rows']==manifest['rows']
    assert {r['sequence_id'] for r in accepted}==set(manifest['sequences'])==set(cache_manifest['sequences'])
    assert sha(manifest_path)==manifest_sha
    receipt=dict(kind='rbf_matching_seen_val_target_free_full_independent_feature_context_admission_v1',seed=args.seed,
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),row_manifest_sha256=manifest_sha,
        checkpoint_sha256=sha(checkpoint_path),cache_manifest_sha256=v2['cache_manifest_sha256'],
        V2_independent_admission_sha256=sha(v2_proof_path),original_schedule_sha256=sha(schedule_path),
        source_sha256=sha(__file__),independent_feature_reference_sha256=sha(REFERENCE),
        new_context_software_gate_sha256=sha(args.software_gate),row_input_gate_sha256=sha(input_gate_path),
        row_producer_sha256=producer_sha,row_producer_preparation_sha256=sha(PREPARED/'preparation.json'),
        original_reference_software_gate_sha256=sha(SOFTWARE),context_software_cases=cases,
        full_original_seen_val_features_and_contexts_independently_accepted=True,sequences=accepted,
        counts=dict(count),max_absolute_errors=dict(maxima),atol=1e-8,rtol=1e-8,
        complete_original_events_and_empty_events_preserved=True,all_class_raw_score_005_top64_preserved=True,
        GT_read=False,test_read=False,NN_forward_repeated=False,actual_measured_network_arrival_history=False,
        full_recoverable_forest_replay_complete=False,full_online_RBF_accepted=False,paper_performance_complete=False)
    new(final,receipt)
    print(json.dumps(dict(seed=args.seed,full_original_seen_val_features_and_contexts_independently_accepted=True,
        rows=count['rows'],events=count['events'],receipt=str(final))),flush=True)


if __name__=='__main__':
    main()
