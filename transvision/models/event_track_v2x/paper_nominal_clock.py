"""Explicit nominal clock envelope; never infer measured sensor timestamps."""
import hashlib
import json

from .detection_cache_v2 import canonical

NOMINAL_ID = 'v2v4real-nominal-10hz-formal-v1'
PERIOD_US = 100000
FORMAL_DELAYS_US = (0, 100000, 200000)


def validate_formal_variant(config):
    """Validate the publication protocol before creating any V2V4Real asset.

    This is deliberately stricter than the per-run schedule binding: it keeps
    the official split, candidate contract, and simulated-time claim attached
    to caches, checkpoints, predictions, and reports.
    """
    required = {
        'protocol_id', 'dataset', 'nominal_rate_hz', 'period_us', 'measured',
        'communication_delay_kind', 'remote_delay_conditions_us',
        'deadline_us', 'max_age_us', 'candidate_protocol', 'split_policy',
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('complete frozen V2V4Real nominal protocol required')
    if (config['protocol_id'] != NOMINAL_ID or config['dataset'] != 'v2v4real'
            or type(config['nominal_rate_hz']) is not int or config['nominal_rate_hz'] != 10
            or type(config['period_us']) is not int or config['period_us'] != PERIOD_US
            or config['measured'] is not False
            or config['communication_delay_kind'] != 'artificially_injected'
            or config['remote_delay_conditions_us'] != list(FORMAL_DELAYS_US)
            or config['deadline_us'] != PERIOD_US or config['max_age_us'] != PERIOD_US
            or config['candidate_protocol'] != 'rbf-all-class-top64-v1'
            or config['split_policy'] != {'development': 'official_train', 'final_evaluation': 'official_test'}):
        raise ValueError('V2V4Real formal variant differs from frozen protocol')
    return dict(config)


def validate_nominal_clock(clock, frames):
    """Validate shared frame mapping, including absent CAV observations/gaps.

    The caller supplies and hashes the mapping. Separate per-CAV compacted
    ordinals are not accepted. This validates nominal consistency, not physical
    synchronization, detector provenance, or train/test independence.
    """
    if not isinstance(clock, dict) or set(clock) != {'protocol_id', 'period_us', 'measured', 'mapping', 'mapping_sha256'}:
        raise ValueError('explicit nominal clock envelope required')
    if (clock['protocol_id'] != NOMINAL_ID or type(clock['period_us']) is not int
            or clock['period_us'] != PERIOD_US or clock['measured'] is not False):
        raise ValueError('unsupported nominal clock or measured-clock claim')
    mapping = clock['mapping']
    if not isinstance(mapping, list) or not mapping:
        raise ValueError('nonempty shared frame mapping required')
    if hashlib.sha256(canonical(mapping)).hexdigest() != clock['mapping_sha256']:
        raise ValueError('nominal frame mapping changed')
    lookup, indices = {}, {}
    for row in mapping:
        if not isinstance(row, dict) or set(row) != {'sequence_id', 'frame_id', 'frame_index'}:
            raise ValueError('unknown nominal mapping fields')
        seq, frame, index = row['sequence_id'], row['frame_id'], row['frame_index']
        if not all(type(v) is str and v for v in (seq, frame)) or type(index) is not int or index < 0:
            raise ValueError('invalid nominal frame identity/index')
        key = seq, frame
        if key in lookup or index in indices.setdefault(seq, set()):
            raise ValueError('duplicate frame identity or index')
        lookup[key] = index * PERIOD_US
        indices[seq].add(index)
    if any(min(values) != 0 for values in indices.values()):
        raise ValueError('each nominal sequence must have a zero origin')
    seen, keys = set(), set()
    for row in frames:
        key = row['sequence_id'], row['frame_id']
        sided = (*key, row['side'])
        if sided in seen or row['side'] not in {'vehicle-side', 'infrastructure-side'}:
            raise ValueError('duplicate/unknown nominal CAV frame')
        seen.add(sided); keys.add(key)
        if key not in lookup or type(row['timestamp_us']) is not int or row['timestamp_us'] != lookup[key]:
            raise ValueError('frame timestamp differs from shared nominal mapping')
    if keys != set(lookup):
        raise ValueError('nominal mapping coverage differs from source frames')
    return dict(protocol_id=NOMINAL_ID, period_us=PERIOD_US, measured=False)


def nominal_run_binding(cache, events, configuration):
    """Fail closed on missing schedule identity or changed causal availability."""
    clock = json.loads(cache.manifest_json).get('producer', {}).get('clock')
    schedule = configuration.get('nominal_schedule')
    if clock is None:
        if schedule is not None:
            raise ValueError('nominal schedule requires nominal cache')
        return None
    if (not isinstance(schedule, dict) or set(schedule) != {'remote_delay_us', 'deadline_us', 'max_age_us'}
            or any(type(v) is not int for v in schedule.values())
            or schedule['remote_delay_us'] not in FORMAL_DELAYS_US
            or schedule['deadline_us'] != PERIOD_US or schedule['max_age_us'] != PERIOD_US):
        raise ValueError('explicit frozen nominal schedule required')
    rows = []
    available = {}
    for entry_json, meta_json in cache.index.values():
        entry, meta = json.loads(entry_json), json.loads(meta_json)
        row = dict(sequence_id=meta['sequence_id'], frame_id=meta['frame_id'],
                   side=meta['side'], timestamp_us=meta['box_reference_timestamp_us'])
        rows.append(row)
        available.setdefault((row['sequence_id'], row['side']), []).append((row, entry['frame_sha256']))
    binding = validate_nominal_clock(clock, rows)
    references = {(r['sequence_id'], r['frame_id']): r['timestamp_us'] for r in rows if r['side'] == 'vehicle-side'}
    for event in events:
        now = references.get((event['sequence_id'], event['frame_id']))
        if now is None or event['reference_us'] != now or event['decision_us'] != now + PERIOD_US:
            raise ValueError('nominal reference/deadline differs')
        expected = {}
        for side in ('vehicle-side', 'infrastructure-side'):
            delay = 0 if side == 'vehicle-side' else schedule['remote_delay_us']
            candidates = [(row, digest) for row, digest in available.get((event['sequence_id'], side), [])
                          if now - PERIOD_US <= row['timestamp_us'] <= now
                          and row['timestamp_us'] + delay <= event['decision_us']]
            if candidates:
                row, digest = max(candidates, key=lambda item: item[0]['timestamp_us'])
                expected[side] = dict(sequence_id=row['sequence_id'], frame_id=row['frame_id'], side=side,
                                      arrival_us=row['timestamp_us'] + delay, frame_sha256=digest)
        actual = event['deliveries']
        if len(actual) != len(expected) or {d['side'] for d in actual} != set(expected):
            raise ValueError('nominal latest-available coverage differs')
        if any(d != expected[d['side']] for d in actual):
            raise ValueError('nominal source/arrival/latest-available selection differs')
    return dict(**binding, schedule=dict(schedule), mapping_sha256=clock['mapping_sha256'])
