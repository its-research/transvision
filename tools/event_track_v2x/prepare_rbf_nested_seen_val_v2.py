"""Freeze a val-only four-shard transport adapter around the published builder."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil

from rbf_nested_seen_val_v2_common import INPUT, CAL, R, new, register, sha

S = R / 'source-freezes/rbf-nested-seen-val-matching-V2-full-admission-v1-20261004'
ORIGINAL = R / 'artifacts/rbf-original-nested-cache-builder-source-byte-readback-20261004'


def replace_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def main():
    if S.exists():
        raise ValueError('source freeze already exists; do not overwrite')
    original = ORIGINAL / 'code-unpack'
    builder_relative = 'tools/event_track_v2x/build_detection_cache_v2.py'
    original_builder = original / builder_relative
    assert sha(original_builder) == '9c415a4e323dc53faa29b3e4eeaa34772ed198341dcacfd2f40173714f735bf7'
    old = original_builder.read_text()
    source = replace_once(old, 'import sys\n', 'import sys\nimport time\n')
    source = replace_once(source, "raw['shard_count'] != 2 or raw['shard_index'] not in {0, 1}",
                          "raw['shard_count'] != 4 or raw['shard_index'] not in {0, 1, 2, 3}")
    source = replace_once(source, "sequences[raw['shard_index']::2]", "sequences[raw['shard_index']::4]")
    source = replace_once(source, 'shards != {(s, i) for s in SIDES for i in [0, 1]}',
                          'shards != {(s, i) for s in SIDES for i in range(4)}')
    source = replace_once(source,
        '    dataset_sha, split, sequences, expected_rows = input_cohort(inputs)\n',
        '    dataset_sha, split, sequences, expected_rows = input_cohort(inputs)\n'
        '    if split != "val":\n        raise ValueError("this separately named adapter is val-only")\n')
    source = replace_once(source,
        '    frames, identities, manifest_shas, shards = [], set(), [], set()\n',
        '    frames, identities, manifest_shas, shards = [], set(), [], set()\n'
        '    progress_started, progress_last = time.monotonic(), 0\n')
    source = replace_once(source,
        "            shard_count.update(frames=1, detections=n, appearance_valid=int(frame.appearance_valid.sum()))\n",
        "            shard_count.update(frames=1, detections=n, appearance_valid=int(frame.appearance_valid.sum()))\n"
        "            now = time.monotonic()\n"
        "            if now - progress_last > 20 or len(frames) == len(expected_rows):\n"
        "                print(json.dumps(dict(stage='matching_seen_val_four_shard_V2_build',\n"
        "                    completed_frames=len(frames), total_frames=len(expected_rows),\n"
        "                    ETA_seconds=(now-progress_started)*(len(expected_rows)-len(frames))/len(frames),\n"
        "                    ETA_scope='remaining V2 frame construction only')), flush=True)\n"
        "                progress_last = now\n")
    source = replace_once(source,
        '        proof = verify(args.output, sha, args.inputs)\n',
        "        print(json.dumps(dict(stage='producer_component_full_V2_rehash', ETA='unknown')), flush=True)\n"
        '        proof = verify(args.output, sha, args.inputs)\n')
    compile(source, builder_relative, 'exec')
    original_functions = {n.name: ast.dump(n, include_attributes=False)
                          for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: ast.dump(n, include_attributes=False)
                     for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    preserved = sorted(set(original_functions) - {'build_cache', 'main'})
    assert all(original_functions[name] == new_functions[name] for name in preserved)
    S.mkdir(parents=True)
    copied = []
    for relative in ('transvision/models/event_track_v2x/detection_cache_v2.py',
                     'transvision/models/event_track_v2x/prediction_features.py'):
        destination = S / 'code' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / relative, destination)
        assert sha(destination) == sha(original / relative)
        copied.append(dict(path=relative, original_sha256=sha(original / relative), unchanged=True))
    builder = S / 'code' / builder_relative
    builder.parent.mkdir(parents=True, exist_ok=True)
    builder.write_text(source)
    for relative in ('transvision/__init__.py', 'transvision/models/__init__.py',
                     'transvision/models/event_track_v2x/__init__.py'):
        # Isolate these two unchanged NumPy modules from the dirty workspace.
        (S / 'code' / relative).write_text('"""Isolated frozen NumPy cache runtime package."""\n')
    workspace = Path(__file__).resolve().parent
    for name in ('rbf_nested_seen_val_v2_common.py', 'seal_rbf_nested_seen_val_v2.py',
                 'accept_rbf_nested_seen_val_v2.py', 'continue_rbf_nested_seen_val_v2.py',
                 'prepare_rbf_nested_seen_val_v2.py'):
        compile((workspace / name).read_text(), name, 'exec')
        shutil.copyfile(workspace / name, S / name)
    schedule = R / 'artifacts/spd-mht-k4-input-recovery-20260929/schedule.json'
    assert sha(schedule) == '2c8999ecbe2ab98bedf13ba2da4b22bd6167eca07b368a99db148abf36de982a'
    inputs = INPUT / 'input-unpack/inputs'
    assert sha(inputs / 'input-manifest.json') == 'fb655debe6c857d062e26d125e873097b781419e6d962bd06166f92f32095cbe'
    receipts = [ORIGINAL / 'acceptance.json', INPUT / 'acceptance.json', CAL / 'acceptance.json']
    configuration = dict(kind='rbf_nested_seen_val_four_shard_V2_source_preparation_v1',
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_builder_task_id='ff89ad92e2b54b298ca4c40d9f483cf0',
        original_builder_sha256=sha(original_builder), adapter_builder_sha256=sha(builder),
        transport_changes=['shard_count_4', 'sequence_stride_4', 'two_side_eight_shard_complete_coverage'],
        val_only_guard=True, progress_ETA_added=True, unchanged_modules=copied,
        unchanged_AST_functions=preserved, independent_numeric_imports_producer=False,
        input_manifest_sha256=sha(inputs / 'input-manifest.json'),
        input_acceptance_sha256=sha(INPUT / 'acceptance.json'),
        calibration_acceptance_sha256=sha(CAL / 'acceptance.json'),
        schedule=str(schedule), schedule_sha256=sha(schedule),
        original_source_receipts=[dict(path=str(path), sha256=sha(path)) for path in receipts],
        sources={path.relative_to(S).as_posix(): dict(bytes=path.stat().st_size, sha256=sha(path))
                 for path in sorted(S.rglob('*')) if path.is_file()},
        expected_frames_per_seed=7189, raw_query_count_per_frame=900,
        expected_original_schedule_events=3316, numerical_atol=1e-8, numerical_rtol=1e-8,
        full_raw_independent_acceptance_required_per_side=True, no_calibration_refitting=True,
        cache_construction_started=False, GPU_task_created=False, independent_acceptance=False,
        measured_network_arrival_history_verified=False, full_online_RBF_accepted=False,
        paper_performance_complete=False)
    new(S / 'preparation.json', configuration)
    register(S / 'preparation.json', configuration['kind'])
    print(json.dumps(dict(source_freeze=str(S), sources=len(configuration['sources']),
        preparation_sha256=sha(S / 'preparation.json'), numerical_code_unchanged=True,
        actual_experiment_acceptance=False, no_GPU_task_created=True)), flush=True)


if __name__ == '__main__':
    main()
