"""Independent debugging-prefix bytes/factors/profile counters and fresh states.

This distinct reader never accepts a prefix as a full original sequence,
full Stage2, an isolated latency measurement, or a paper performance result.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import sys
import tarfile

from rbf_nested_seen_val_v2_common import R, new, register, sha

EXECUTOR = R/'source-freezes/rbf-original-real-prefix-GPU-function-profile-executor-v1-20261004'
OBSERVER = R/'source-freezes/rbf-original-frozen-replay-external-function-profile-v1-20261004'
CPU = R/'source-freezes/rbf-final-refit-full-forest-independent-CPU-v3-20261004'
FRESH = R/'source-freezes/rbf-independent-fresh-branch-state-v3-normalized-admission-20261001/oracle.py'
FRESH_SHA = '46d5f6b474202961077e43f9c572f6ecfb0a98e992e9f4b19fcd37cccfce1c58'
INDEX = R/'receipts/rbf-final-refit-three-seed-all-row-full-independent-numeric-acceptance-20261004.json'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def gate_plan(plan, preparation):
    assert plan['world_size'] in (4, 8) and type(plan['world_size']) is int
    expected = dict(preparation['plan'], world_size=plan['world_size'])
    assert plan == expected, 'profile plan differs from fixed debugging contract'
    assert plan['profile_prefix_events'] == 32 and plan['seed'] == 2027
    assert plan['method'] == 'rbf' and plan['configuration']['allocation'] == 'bound'
    assert plan['scoring_atol'] == plan['scoring_rtol'] == 1e-4
    assert plan['profiling_debug_contract'] is True and plan['full_dataset_or_latency_claim'] is False


def gate_report(report, plan, task_id):
    assert report['kind'] == 'rbf_original_frozen_real_prefix_GPU_function_profile_candidate_v1'
    actual = dict(report['plan'])
    actual.pop('cache_relative_root', None)
    assert actual == plan and report['task_id'] == task_id
    assert report['paper_performance_complete'] is False
    assert report['same_resource_baseline_comparison_accepted'] is False
    assert report['full_46_sequence_replay_completed'] is False
    assert report['paper_cost_admission'] is False and report['profiler_overhead_quantified'] is False


def expected_sources(plan):
    archive = R/'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/source.bytes'
    assert sha(archive) == plan['source']['sha256']
    sources = {}
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            if member.isfile() and member.name.startswith('transvision/models/event_track_v2x/') and member.name.endswith('.py'):
                sources[member.name] = hashlib.sha256(stream.extractfile(member).read()).hexdigest()
    sources.update(plan['exclusive_patches'])
    for name, record in plan['source_replacements'].items():
        assert sources[name] == record['before_sha256']
        sources[name] = record['sha256']
    return sources


def gate_function_profile(document, sources, events, nodes):
    assert type(events) is int and events > 0 and type(nodes) is int and nodes > 0
    assert set(document) == {'kind', 'functions', 'raw_call_arguments_persisted',
        'unallowlisted_functions_or_paths_persisted',
        'function_wall_times_include_blocking_and_are_not_GPU_kernel_times'}
    assert document['kind'] == 'rbf_allowlisted_original_replay_function_profile_v1'
    assert document['raw_call_arguments_persisted'] is False
    assert document['unallowlisted_functions_or_paths_persisted'] is False
    assert document['function_wall_times_include_blocking_and_are_not_GPU_kernel_times'] is True
    observer = load_module('frozen_original_profile_format', OBSERVER/'rbf_frozen_replay_cost_profile.py')
    functions = document['functions']
    seen = set()
    for row in functions:
        assert set(row) == {'source', 'line', 'function', 'source_sha256', 'primitive_calls',
            'total_calls', 'function_own_wall_seconds', 'function_inclusive_wall_seconds'}
        assert isinstance(row['source'], str) and isinstance(row['function'], str) and row['function']
        key = (row['source'], row['line'], row['function'])
        assert key not in seen
        seen.add(key)
        if row['source'] == 'allowed-native-function':
            assert row['source_sha256'] is None and row['function'] in observer.ALLOWED_NATIVE_NAMES
        else:
            assert row['source'] in sources and Path(row['source']).name in observer.ALLOWED_MODULES
            assert row['source_sha256'] == sources[row['source']]
        assert type(row['line']) is int and row['line'] >= 0
        assert all(type(row[k]) is int and row[k] >= 0 for k in ('primitive_calls','total_calls'))
        assert row['primitive_calls'] <= row['total_calls']
        assert all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k] >= 0
                   for k in ('function_own_wall_seconds','function_inclusive_wall_seconds'))
    def calls(file, function):
        rows = [r for r in functions if Path(r['source']).name == file and r['function'] == function]
        assert rows, ('missing profiled frozen function', file, function)
        return sum(r['total_calls'] for r in rows)
    assert calls('exclusive_paper_runtime.py', 'replay') == 1
    assert calls('persistent_cache_stream.py', 'step') == events
    assert calls('persistent_cache_stream.py', 'rows') == events
    assert calls('forest_potentials.py', '__call__') == nodes
    assert calls('forest_potentials.py', 'neural_parent_logits') == nodes
    assert calls('learned_identity.py', 'forward') == nodes
    assert calls('recoverable_identity.py', 'model_digest') == nodes
    tensor_calls = calls('recoverable_identity.py', '_tensor_digest')
    assert nodes > 0 and tensor_calls > 0 and tensor_calls % nodes == 0
    return dict(profiled_events=events, profiled_row_scorer_calls=nodes,
                model_digest_calls=nodes, parameter_tensor_digest_calls=tensor_calls,
                parameter_tensors_hashed_per_query=tensor_calls//nodes,
                profile_counters_match_real_replay_events_and_rows=True,
                independent_machine_clock_or_GPU_kernel_occupancy_remeasurement=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', required=True)
    args = parser.parse_args()
    assert len(args.task_id) == 32 and all(c in '0123456789abcdef' for c in args.task_id)
    own = Path(__file__).resolve().parent
    freeze = json.loads((own/'source-freeze.json').read_bytes())
    assert freeze['kind'] == 'rbf_original_real_prefix_function_profile_independent_reader_source_v1'
    for name, record in freeze['sources'].items():
        assert sha(own/name) == record['sha256']
    for record in freeze['references']:
        assert sha(record['path']) == record['sha256']
    reader = load_module('unchanged_full_forest_safe_byte_utilities', Path(freeze['safe_byte_helper']['path']))
    assert sha(freeze['safe_byte_helper']['path']) == freeze['safe_byte_helper']['sha256']
    preparation = json.loads((EXECUTOR/'preparation.json').read_bytes())
    for name, spec in preparation['sources'].items():
        assert sha(EXECUTOR/name) == spec['sha256']
    from clearml import Task
    task = Task.get_task(task_id=args.task_id)
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == preparation['plan']['bootstrap_sha256']
    parameters = task.get_parameters()
    plan = json.loads(parameters['General/plan'])
    gate_plan(plan, preparation)
    recipe_sha = hashlib.sha256(canonical(plan)).hexdigest()
    assert parameters['General/recipe_sha256'] == recipe_sha
    status = str(task.status)
    if status not in ('completed','failed'):
        print(json.dumps(dict(task_id=task.id,status=status,readback_started=False,ETA='unknown')))
        return
    root = R/'artifacts/rbf-original-real-prefix-GPU-function-profile-independent-v1-20261004'/task.id
    root.mkdir(parents=True, exist_ok=True)
    expected_keys = {'receipt','exclusive-source-manifest',*(f'replay-rank{i}' for i in range(plan['world_size']))}
    assert 'receipt' in task.artifacts and set(task.artifacts) <= expected_keys
    result_path = root/('independent-prefix-diagnostic-admission.json' if status=='completed' else 'independent-failure-byte-readback.json')
    if result_path.exists():
        existing = json.loads(result_path.read_bytes())
        assert existing['task_id'] == task.id and existing['recipe_sha256'] == recipe_sha
        assert set(existing['artifacts']) == set(task.artifacts)
        for key, record in existing['artifacts'].items():
            assert task.artifacts[key].hash == record['sha256'] and task.artifacts[key].size == record['bytes']
        print(json.dumps(dict(existing_receipt=str(result_path),not_repeated_or_overwritten=True)))
        return
    artifacts = {key:reader.read_artifact(task,key,root/(key+('.json' if key in ('receipt','exclusive-source-manifest') else '.tar.gz')))
                 for key in sorted(task.artifacts)}
    report = json.loads((root/'receipt.json').read_bytes())
    gate_report(report, plan, task.id)
    if status == 'failed':
        new(result_path,dict(kind='rbf_original_prefix_profile_failed_independent_bytes_v1',task_id=task.id,
            recipe_sha256=recipe_sha,artifacts=artifacts,producer_failure=report['failure'],
            rank_partial_reports=report['ranks'],automatic_retry_permitted=False,experiment_accepted=False))
        register(result_path,'rbf-original-prefix-profile-failure-independent-bytes')
        print(json.dumps(dict(failure_receipt=str(result_path),failure_preserved=True)))
        return
    assert set(artifacts) == expected_keys and report['failure'] is None
    assert report['all_selected_real_prefixes_profiled'] is True
    ranks = report['ranks']
    assert sorted(r['rank'] for r in ranks) == list(range(plan['world_size']))
    assert len({r['gpu_uuid'] for r in ranks}) == plan['world_size']
    source = json.loads((root/'exclusive-source-manifest.json').read_bytes())
    for left,right in (('patches','exclusive_patches'),('configuration','configuration'),('original_source','source'),
                       ('bootstrap_sha256','bootstrap_sha256'),('source_replacements','source_replacements')):
        assert source[left] == plan[right]
    index = json.loads(INDEX.read_bytes())
    assert sha(INDEX) == plan['numeric_reference_admission_sha256']
    entry = next(row for row in index['seeds'] if row['seed']==2027)
    for key in ('training_byte_proof','prediction_byte_proof','numeric_completion'):
        assert sha(entry[key]) == entry[key+'_sha256']
    checkpoint_path = Path(entry['training_byte_proof']).parent/'checkpoint'
    assert sha(checkpoint_path) == plan['checkpoint']['sha256']
    checkpoint = json.loads(checkpoint_path.read_bytes())
    assert checkpoint['model_sha256'] == plan['final_refit_model_sha256']
    forward_proof = json.loads(Path(entry['prediction_byte_proof']).read_bytes())
    schedule_path = R/'artifacts/rbf-original-cache-CPU-metadata-export-v1-20261001/seed2027/events.json'
    assert sha(schedule_path) == plan['events']['sha256']
    schedule = json.loads(schedule_path.read_bytes())
    sequences = sorted(schedule['origin_us_by_sequence'])[:plan['world_size']]
    assert len(schedule['events']) == 7445 and len(schedule['origin_us_by_sequence']) == 46
    expected = {s:[] for s in sequences}
    groups = {s:[e for e in schedule['events'] if e['sequence_id']==s][:32] for s in sequences}
    for asset in plan['forward_outputs']:
        assert asset == forward_proof['artifacts'][asset['key']]
        path = Path(entry['prediction_byte_proof']).parent/(asset['key']+'.jsonl')
        assert sha(path) == asset['sha256']
        with path.open('rb') as stream:
            for line in stream:
                row = json.loads(line)
                sid = row['sequence_id']
                if sid in expected and row['decision_us'] <= groups[sid][-1]['decision_us']:
                    assert row['row'] == len(expected[sid])
                    expected[sid].append(row)
    sources = expected_sources(plan)
    sys.path.insert(0,str(CPU))
    cache_oracle = load_module('unchanged_final_cache203_for_profile_prefix',CPU/'final_cache203.py')
    cache = cache_oracle.CacheAdmission(2027,checkpoint_path)
    assert sha(FRESH) == FRESH_SHA
    fresh = load_module('unchanged_fresh_state_for_profile_prefix',FRESH)
    outcomes = []
    for rank in range(plan['world_size']):
        rank_report = next(r for r in ranks if r['rank']==rank)
        assert rank_report['world_size'] == plan['world_size'] and rank_report['all_sequences_completed'] is True
        assert rank_report['TF32_matmul'] is False and rank_report['TF32_cudnn'] is False
        assert len(rank_report['sequences']) == 1 and rank_report['events_committed'] == 32
        sequence = sequences[rank]
        assert rank_report['sequences'][0]['sequence_id'] == sequence
        assert rank_report['sequences'][0]['expected_events'] == rank_report['sequences'][0]['committed_events'] == 32
        reader.unpack(root/f'replay-rank{rank}.tar.gz',root/f'rank{rank}-unpack')
        directory = root/f'rank{rank}-unpack/rank-{rank}'/sequence
        request = json.loads((directory/'request.json').read_bytes())
        measured = json.loads((directory/'measurement-receipt.json').read_bytes())
        functions = json.loads((directory/'function-profile.json').read_bytes())
        assert request['kind'] == 'rbf_original_replay_external_function_profile_request_v1'
        assert request['observer_sha256'] == plan['profile_observer_sha256']
        assert request['source_files'] == sources and request['selected_sequences'] == [sequence]
        assert request['per_sequence_prefix_count'] == request['event_count'] == 32
        assert request['admitted_full_schedule_sha256'] == plan['events']['sha256']
        assert request['event_bytes_sha256'] == hashlib.sha256(canonical(groups[sequence])).hexdigest()
        assert request['checkpoint_binding']['checkpoint_sha256'] == plan['checkpoint']['sha256']
        assert request['checkpoint_binding']['model_sha256'] == checkpoint['model_sha256']
        assert request['checkpoint_binding']['candidate_protocol'] == 'rbf-all-class-top64-v1'
        assert request['checkpoint_binding']['dataset'] == 'spd' and request['checkpoint_binding']['fit_split'] == 'train'
        assert request['checkpoint_binding']['seed'] == 2027
        runtime = request['runtime_identity']
        assert runtime['rank'] == rank and runtime['world_size'] == plan['world_size']
        assert runtime['GPU_uuid'] == rank_report['gpu_uuid'] and runtime['actual_device'] == 'cuda:'+str(rank)
        assert runtime['TF32_matmul'] is False and runtime['TF32_cudnn'] is False
        assert request['online_model_inputs_or_source_changed'] is False
        assert request['paper_latency_admission'] is False and request['full_Stage2_admission'] is False
        assert measured['kind'] == 'rbf_original_replay_external_function_profile_v1'
        assert measured['status'] == 'profile_produced_pending_independent_readback' and measured['failure'] is None
        assert measured['request_sha256'] == sha(directory/'request.json')
        assert measured['function_profile_sha256'] == sha(directory/'function-profile.json')
        for name in ('profiled_replay_wall_seconds','current_process_CPU_seconds','device_wait_before_seconds','device_wait_after_seconds'):
            assert type(measured[name]) in (int,float) and math.isfinite(measured[name]) and measured[name] >= 0
        assert measured['wall_seconds_include_profile_overhead_and_output_writing'] is True
        assert measured['function_profile_is_not_independent_physical_GPU_occupancy'] is True
        assert measured['profiler_overhead_quantified'] is False
        assert measured['full_schedule_coverage_claimed'] is False and measured['paper_cost_or_performance_accepted'] is False
        replay_root = directory/'original-replay'
        receipt = json.loads((replay_root/'receipt.json').read_bytes())
        assert receipt == measured['original_replay_receipt']
        original_plan = json.loads((replay_root/'plan.json').read_bytes())
        assert original_plan['configuration'] == plan['configuration'] and original_plan['fixture'] is False
        assert original_plan['model_binding'] == request['checkpoint_binding']
        assert original_plan['events_sha256'] == request['event_bytes_sha256']
        assert original_plan['source_sha256'] == sources
        assert receipt['status'] == 'software_replay_completed' and receipt['completed_events'] == 32
        assert receipt['completed_sequences'] == [sequence]
        for name, digest in receipt['files'].items():
            assert sha(replay_root/name) == digest
        database = receipt['databases'][sequence]
        path = (replay_root/database['path']).resolve()
        assert path.is_relative_to(replay_root.resolve()) and sha(path) == database['sha256']
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
            assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            commits = db.execute('SELECT event_id,prediction,audit FROM events ORDER BY ordinal').fetchall()
            assert [r[0] for r in commits] == [e['event_id'] for e in groups[sequence]]
            with (replay_root/'predictions.jsonl').open('rb') as predictions, (replay_root/'audit.jsonl').open('rb') as audits:
                for (_,p,a),p_line,a_line in zip(commits,predictions,audits,strict=True):
                    assert p == p_line.rstrip(b'\n') and a == a_line.rstrip(b'\n')
            maximum = 0.; count = 0
            for i,node,raw,digest in db.execute('SELECT i,node_id,raw,sha FROM observations ORDER BY i'):
                assert i == count and hashlib.sha256(raw).hexdigest() == digest
                ref = expected[sequence][i]
                assert ref['node_id'] == node and ref['row'] == i
                factors = db.execute('SELECT p,w FROM potentials WHERE i=? ORDER BY p',(i,)).fetchall()
                assert [p for p,w in factors] == [-1]+ref['context_indices'][:-1]
                normalized = reader.normalized_factors(ref['logits'])
                for (_,actual),value in zip(factors,normalized,strict=True):
                    assert math.isfinite(actual) and abs(actual-value) <= 1e-4+1e-4*abs(value)
                    maximum = max(maximum,abs(actual-value))
                count += 1
            assert count == len(expected[sequence])
        counter = gate_function_profile(functions,sources,32,count)
        # The original full-sequence gate remains unchanged. Its low-level
        # allow_prefix mode is invoked explicitly for this debugging prefix;
        # that oracle's historical fixture label confers no full acceptance.
        features = cache_oracle.verify_database(path,database['sha256'],cache,allow_prefix=True)
        state = fresh.verify_database(path,database['sha256'],progress=lambda _:None)
        assert features['events'] == state['events'] == 32 and features['rows'] == count
        assert features['atol'] == features['rtol'] == fresh.ATOL == fresh.RTOL == 1e-8
        outcome = dict(rank=rank,sequence_id=sequence,events=32,rows=count,database_sha256=database['sha256'],
            max_normalized_factor_difference=maximum,profile_counter_check=counter,
            raw_cache203_and_exact_causal_context_check=features,fresh_state_check=state,
            reported_profile_cost={k:measured[k] for k in ('profiled_replay_wall_seconds','current_process_CPU_seconds','device_wait_before_seconds','device_wait_after_seconds')})
        outcomes.append(outcome)
        print(json.dumps(dict(stage='independent_real_prefix_profile_readback',completed_ranks=len(outcomes),
            total_ranks=plan['world_size'],ETA_seconds=None,ETA_reason='heterogeneous fresh branch-state history')),flush=True)
    new(result_path,dict(kind='rbf_original_real_prefix_profile_independent_bytes_factors_context_states_diagnostic_v1',
        task_id=task.id,recipe_sha256=recipe_sha,artifacts=artifacts,outcomes=outcomes,
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),
        all_registered_bytes_verified=True,events_checked=32*plan['world_size'],NN_atol=1e-4,NN_rtol=1e-4,
        state_and_cache203_atol=1e-8,state_and_cache203_rtol=1e-8,
        measured_real_causal_prefixes_and_frozen_model_bound=True,
        profiling_counters_match_replay_events_and_model_queries=True,
        reported_profile_cost_is_debugging_only=True,independent_machine_clock_or_physical_GPU_cost_admission=False,
        profiler_overhead_quantified=False,full_46_sequence_replay_admitted=False,
        full_Stage2_or_same_resource_tail_latency_or_paper_performance_admitted=False))
    register(result_path,'rbf-original-real-prefix-profile-independent-diagnostic')
    print(json.dumps(dict(receipt=str(result_path),diagnostic_prefix_checked=True,paper_performance=False)))


if __name__ == '__main__':
    main()
