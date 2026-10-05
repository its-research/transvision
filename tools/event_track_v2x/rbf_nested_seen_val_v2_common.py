"""Source and independently accepted input gates for the named val V2 cache."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path

R = Path('/Volumes/Data/test/recover-before-fuse')
RAW = R / 'artifacts/rbf-nested-detector-seen-val-independent-raw-readback-v1-20261004'
CAL = R / 'artifacts/rbf-nested-seen-val-frozen-calibration-byte-readback-20261004'
INPUT = R / 'artifacts/rbf-nested-detector-seen-val-full-input-admission-v1-20261004'
OUT = R / 'artifacts/rbf-nested-seen-val-matching-V2-full-admission-v1-20261004'
SIDES = ('vehicle-side', 'infrastructure-side')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def register(path, kind):
    ledger_path = R / 'receipts/20260928-execution-ledger.json'
    with open(str(ledger_path) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(ledger_path.read_bytes())
        if any(entry.get('receipt') == str(path) for entry in ledger['entries']):
            return
        ledger['entries'].append(dict(kind=kind, receipt=str(path), receipt_sha256=sha(path),
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), goal_status='active'))
        temporary = ledger_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + '\n')
        os.replace(temporary, ledger_path)


def source_gate():
    root = Path(__file__).resolve().parent
    configuration = json.loads((root / 'preparation.json').read_bytes())
    assert configuration['kind'] == 'rbf_nested_seen_val_four_shard_V2_source_preparation_v1'
    for name, record in configuration['sources'].items():
        path = root / name
        assert not path.is_symlink() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
    for record in configuration['original_source_receipts']:
        assert sha(record['path']) == record['sha256']
    return root, configuration


def admitted_seed(seed):
    """No cache construction until both side readers issued full acceptance."""
    root, configuration = source_gate()
    assert seed in (1337, 2027, 3407)
    dispatch = json.loads((R / 'receipts/rbf-nested-detector-matching-seen-val-raw-GPU-dispatch-20261004.json').read_bytes())
    proofs, raw_roots = [], []
    checkpoints = {}
    for side in SIDES:
        path = RAW / f'seed{seed}' / side / 'independent-acceptance.json'
        proof = json.loads(path.read_bytes())
        job = next(job for job in dispatch['jobs'] if job['seed'] == seed and job['side'] == side)
        assert proof['task_id'] == job['task_id'] and proof['seed'] == seed and proof['side'] == side
        assert proof['full_cloud_bytes_independently_read'] is True
        assert proof['full_raw_payload_and_frame_coverage_verified'] is True
        assert proof['GT_free_all_class_raw_queries_retained'] is True and proof['TF32_enabled'] is False
        assert proof['query_count_per_frame'] == 900 and len(proof['outcomes']) == 4
        assert {x['shard_index'] for x in proof['outcomes']} == set(range(4))
        assert proof['original_input_manifest_sha256'] == configuration['input_manifest_sha256']
        for outcome in proof['outcomes']:
            directory = Path(outcome['raw_root'])
            assert directory.resolve().is_relative_to((RAW / f'seed{seed}' / side).resolve())
            assert sha(directory / 'raw-cache-manifest.json') == outcome['manifest_sha256']
            raw = json.loads((directory / 'raw-cache-manifest.json').read_bytes())
            assert raw['checkpoint_sha256'] == job['recipe']['checkpoint_sha256']
            assert raw['shard_index'] == outcome['shard_index'] and raw['shard_count'] == 4
            assert raw['TF32_enabled'] is False and raw['legacy_placeholder_count'] == 0
            raw_roots.append(str(directory))
        checkpoints[side] = job['recipe']['checkpoint_sha256']
        proofs.append(dict(path=str(path), sha256=sha(path), task_id=proof['task_id']))
    calibration_acceptance = CAL / 'acceptance.json'
    assert sha(calibration_acceptance) == configuration['calibration_acceptance_sha256']
    entry = next(x for x in json.loads(calibration_acceptance.read_bytes())['seeds'] if x['seed'] == seed)
    calibration = Path(entry['path'])
    assert sha(calibration) == entry['sha256'] and calibration.stat().st_size == entry['bytes']
    assert entry['detector_checkpoint_sha256'] == checkpoints
    assert sha(INPUT / 'acceptance.json') == configuration['input_acceptance_sha256']
    inputs = INPUT / 'input-unpack/inputs'
    assert sha(inputs / 'input-manifest.json') == configuration['input_manifest_sha256']
    return dict(seed=seed, source_root=str(root), raw_roots=raw_roots,
        raw_independent_acceptances=proofs, calibration=str(calibration),
        calibration_sha256=entry['sha256'], inputs=str(inputs), detector_checkpoint_sha256=checkpoints,
        schedule=configuration['schedule'], schedule_sha256=configuration['schedule_sha256'])
