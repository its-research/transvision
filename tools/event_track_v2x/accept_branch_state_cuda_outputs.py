"""Independent complete-sequence CUDA audit using portable branch witnesses.

The producer's hashes bind every witness exactly. Persisted states and all
selected/unselected projections are checked independently at the original
1e-8 tolerance; the frozen raw-history oracle is still required. No production
tracker is imported. This grants no cohort, performance, or memory-target claim.
"""
import argparse
from collections import OrderedDict
import importlib.util
import json
from pathlib import Path
import sqlite3

from rbf_nested_seen_val_v2_common import R, new, register, sha

CHECKER = R/'source-freezes/rbf-optimized-state-single-sequence-independent-v3-request-bindings-20261004/accept_optimized_branch_state_sequence.py'
CHECKER_SHA = '2bdadf427b5488776b01d2b1563b0634c64e3ebaa0e3651768ee94ed34140206'
REFERENCE = R/'artifacts/rbf-batched-full-real-sequence-CPU-pair-v1-20261004/serial-replay/sequence-0.sqlite'
REFERENCE_SHA = 'c6fd79ebe28ba06050c68b2fc1694b55b516e59dfe4926e47468820496f22b80'
REFERENCE_PROOF = R/'artifacts/rbf-batched-full-sequence-independent-acceptance-v2-20261004/independent-sequence-receipt.json'
REFERENCE_PROOF_SHA = 'd104416c40b65e9e6082afc41fa74ad6f7f0d21f8a7e411ebd1420ac5bc0adbc'
PACKAGE = R/'artifacts/rbf-branch-state-CUDA-admitted-input-bundle-v2-20261004'
MANIFEST_SHA = 'ce8bf75c2b1c29aba37088aac6f286487097d8026a32ce3a5d718cc1d1313213'
CPU_ACCEPTANCE_SHA = 'a26c27c62f3eba4b76640d2ece2cbee5d5e1a151242da59e8271da79170069f1'
PRODUCER = R/'source-freezes/rbf-branch-state-CUDA-transport-producer-v2-witness-20261004/run_branch_state_cuda_candidate.py'
TRANSPORT_FREEZE_SHA = '4b87a785fe0f60d01c80fd27872295857eff3a45d40c9caaae223eb06288c3e9'


def load(path, checksum, name):
    assert sha(path) == checksum
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class WitnessEvent:
    """One exact event's values; absent, duplicated and surplus branches fail."""
    def __init__(self, record, ordinal, event_id, prediction, audit, checker):
        assert set(record) == {'ordinal','event_id','reference_us','decision_us','branches'}
        assert record['ordinal'] == ordinal and record['event_id'] == event_id
        assert record['reference_us'] == prediction['box_reference_timestamp_us']
        assert record['decision_us'] == prediction['decision_timestamp_us']
        self.values = {}
        self.used = set()
        for branch in record['branches']:
            assert set(branch) == {'component','handle','predictions'}
            key = (branch['component'], branch['handle'])
            assert all(type(v) is int and v >= 0 for v in key)
            assert key not in self.values, 'duplicate witness branch'
            self.values[key] = branch['predictions']
        expected = {}
        for component in audit['components']:
            for branch in component['branches']:
                key = (component['component'], branch['handle'])
                assert key not in expected
                expected[key] = branch['state_sha256']
                if component['expired_output_only']:
                    assert self.values.get(key) == [], 'expired witness has output'
                    self.used.add(key)
        assert set(self.values) == set(expected), 'witness branch coverage differs'
        for key, checksum in expected.items():
            assert checker.digest(self.values[key]) == checksum, 'witness byte commitment differs'
        self.projections = 0
        self.max_difference = 0.

    def finish(self):
        assert self.used == set(self.values), 'unverified witness branch'


class PortableProjection:
    def __init__(self, stored, component, witness, checker):
        self.stored, self.component, self.witness, self.checker = stored, component, witness, checker
        self.rows, self.prefixes = stored.rows, stored.prefixes

    def outputs(self, handle, reference, config, sequence, decision, nodes):
        values = self.witness.values[(self.component, handle)]
        fresh = self.stored.outputs(handle, reference, config, sequence, decision, nodes)
        count, error = self.checker.compare_predictions(fresh, values)
        self.witness.projections += count
        self.witness.max_difference = max(self.witness.max_difference, error)
        self.witness.used.add((self.component, handle))
        return values


def correspondence(left, right, execution, config, stream, checker):
    caps = dict(SQL_templates=256, ancestor_pairs_per_call=config['prefix_cache_entries'],
                raw_observations=config['prefix_cache_entries'])
    previous = [['0'*64, '0'*64] for _ in range(2)]
    summaries, caches, counts = [{}, {}], [OrderedDict(), OrderedDict()], [0, 0]
    events = predictions = branches = projections = 0
    max_output = max_branch = max_witness = 0.
    def stored(side, component):
        cache = caches[side]
        if component not in cache:
            cache[component] = checker.StoredProjection((left, right)[side], component)
            if len(cache) > 32: cache.popitem(last=False)
        cache.move_to_end(component)
        return cache[component]
    for rows in zip(left.execute('SELECT * FROM events ORDER BY ordinal'),
                    right.execute('SELECT * FROM events ORDER BY ordinal'), strict=True):
        assert rows[0][:2] == rows[1][:2] and rows[0][0] == events
        line = stream.readline(); assert line, 'missing witness event'
        values = []
        for side, row in enumerate(rows):
            p, a = json.loads(row[3]), json.loads(row[4])
            assert a['event_id'] == row[1] and a['sequence_id'] == p['sequence_id']
            counts[side] = checker.bind_request((left,right)[side],row[2],p,a,config,counts[side],explicit_scope=bool(side))
            assert [p['previous_commit_sha256'],a['previous_audit_sha256']] == previous[side]
            assert checker.digest({k:v for k,v in p.items() if k!='commit_sha256'}) == p['commit_sha256'] == a['prediction_sha256']
            previous[side] = [p['commit_sha256'], checker.digest(a)]
            for component in a['components']: summaries[side][component['component']] = component
            if side:
                witness = WitnessEvent(json.loads(line),row[0],row[1],p,a,checker)
                kernel = lambda c: PortableProjection(stored(1,c),c,witness,checker)
            else:
                kernel = lambda c: stored(0,c)
            audit, outputs, _ = checker.bind_branch_hashes(a,p,kernel,config['state'])
            if side:
                witness.finish()
                max_witness = max(max_witness,witness.max_difference)
                projections += witness.projections
            values.append((p,audit,outputs))
        a,b = (dict(v[1]) for v in values)
        a.pop('prediction_sha256'); a.pop('previous_audit_sha256')
        assert a == checker.semantic_audit(b,execution,caps), 'full semantic audit differs'
        a,b = (dict(v[0]) for v in values)
        x,y = a.pop('predictions'),b.pop('predictions')
        for p in (a,b): p.pop('commit_sha256'); p.pop('previous_commit_sha256')
        assert a == b, 'causal output identity differs'
        count,error = checker.compare_predictions(x,y)
        predictions += count; max_output = max(max_output,error)
        for (key_a,x),(key_b,y) in zip(values[0][2],values[1][2],strict=True):
            assert key_a == key_b
            _,error = checker.compare_predictions(x,y)
            branches += 1; max_branch = max(max_branch,error)
        events += 1
        if events % 10 == 0:
            print(json.dumps(dict(stage='CUDA_independent_portable_branch_correspondence',completed_events=events,
                total_events=195,branch_commitments=2*branches,ETA='unknown; heterogeneous branch sizes')),flush=True)
    assert not stream.read(1), 'surplus witness events'
    for side,db in enumerate((left,right)):
        state = json.loads(db.execute("SELECT v FROM meta WHERE k='state'").fetchone()[0])
        assert [state['prediction_sha256'],state['audit_sha256']] == previous[side]
        assert state['n'] == counts[side] and state['events'] == events
        assert {c:json.loads(v) for c,v in db.execute('SELECT component,payload FROM component_summaries')} == summaries[side]
    return dict(events=events,predictions=predictions,branch_commitments=2*branches,request_commitments=2*events,
        witness_projections=projections,max_witness_projection_error=max_witness,
        max_output_state_error=max_output,max_branch_projection_error=max_branch,
        complete_event_audits_and_commit_chains_bound=True)


def check_execution(root, receipt, plan, manifest):
    """Bind claimed device work and sources to independently read cloud bytes."""
    assert receipt['plan'] == plan and receipt['failure'] is None
    assert plan['recipe'] == 'rbf_one_complete_sequence_independent_root_state_CUDA_candidate_v1'
    assert plan['required_compute_GPUs'] == 1 and plan['CPU_acceptance_sha256'] == CPU_ACCEPTANCE_SHA
    assert plan['manifest']['sha256'] == MANIFEST_SHA
    assert receipt['candidate_checks_completed'] is True
    candidate = json.loads((root/'candidate-check.json').read_bytes())
    assert candidate['kind'] == 'real_factor_prefix_batched_state_candidate_v1'
    assert candidate['events'] == 195 and candidate['materialized_branch_states_checked'] == 56389
    assert candidate['numerical_atol'] == candidate['numerical_rtol'] == 1e-8
    assert candidate['complete_reference_sequence_checked'] is candidate['discrete_actions_factors_work_identical'] is True
    assert candidate['profiling_enabled'] is False and candidate['measurement_order'] == ['candidate']
    assert candidate['device']['GPU_executed'] is True and candidate['device']['device'] == 'cuda:0'
    assert candidate['recorded_reference']['sha256'] == REFERENCE_PROOF_SHA
    assert candidate['recorded_reference']['database_sha256'] == REFERENCE_SHA
    assert sha(root/'input-binding.json') == candidate['input_binding_sha256']
    binding = json.loads((root/'input-binding.json').read_bytes())
    assert binding['source_database_sha256'] == REFERENCE_SHA
    assert binding['events'] == 195 and binding['sequence'] == '0000' and binding['max_batch'] == 64
    assert binding['device'] == candidate['device'] and binding['recorded_reference'] == candidate['recorded_reference']
    source_binding = R/'artifacts/rbf-independent-root-state-full-sequence-CPU-v1-20261004/input-binding.json'
    assert sha(source_binding) == '79fd69cf77d4bb08aef36e0d2a6179943f40a2b7ab20c67e1d9c0b0e46a5598f'
    admitted_sources = json.loads(source_binding.read_bytes())
    assert binding['source_files'] == admitted_sources['source_files'], 'complete admitted implementation sources required'
    expected = manifest['files']
    for name,checksum in binding['source_files'].items():
        assert expected['execution/'+name]['sha256'] == checksum
    assert binding['qualifier_sha256'] == expected['execution/tools/event_track_v2x/qualify_batched_branch_states.py']['sha256']
    launch = json.loads((root/'GPU-launch-binding.json').read_bytes())
    for name,checksum in launch['execution_sources_sha256'].items():
        assert expected['execution/tools/event_track_v2x/'+name]['sha256'] == checksum
    assert launch['CPU_reference_acceptance_sha256'] == REFERENCE_PROOF_SHA
    assert launch['CPU_reference_reuse_requested'] is True
    metrics = candidate['metrics']['candidate']
    assert metrics['batch_calls'] > 0 and metrics['batch_rows'] > 0
    device = receipt['device']
    assert device['used_device'] == 'cuda:0' and device['used_GPU_count'] == 1
    assert device['visible_GPU_count'] >= 1
    assert device['uuid'] == candidate['device']['uuid'] and device['uuid'] not in ('','unknown','None')
    measurement = json.loads((root/'GPU-runtime-measurement.json').read_bytes())
    assert measurement['GPU_uuid'] == device['uuid'] and measurement['completed_events'] == 195
    assert measurement['TF32_matmul'] is measurement['TF32_cudnn'] is False
    assert measurement['replay_failure'] is None and measurement['sampler_finished'] is True
    assert not measurement['sampler_errors'] and measurement['peak_process_tensor_allocated_bytes'] > 0
    assert measurement['samples'] and measurement['elapsed_seconds_including_observer'] > 0
    for sample in measurement['samples']:
        assert 0 <= sample['device_free_bytes'] <= sample['device_total_bytes'] and sample['device_total_bytes'] > 0
    witness = json.loads((root/'branch-commitment-witness.receipt.json').read_bytes())
    assert witness['kind'] == 'rbf_complete_branch_projection_commitment_witness_v1'
    assert witness['database_sha256'] == candidate['database_sha256']['candidate']
    assert witness['events'] == witness['request_commitments'] == 195
    assert witness['all_branch_commitments_reproduced'] is True and witness['projection_checker_sha256'] == CHECKER_SHA
    assert witness['source_sha256'] == expected['execution/tools/event_track_v2x/export_branch_state_commitment_witness.py']['sha256']
    assert sha(root/'branch-commitment-witness.jsonl') == witness['witness_sha256']
    assert (root/'branch-commitment-witness.jsonl').stat().st_size == witness['witness_bytes']
    return candidate, binding, witness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--byte-admission',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    proof = json.loads(args.byte_admission.read_bytes())
    assert proof['kind'] == 'rbf_branch_state_CUDA_output_independent_cloud_bytes_v1'
    assert proof['completed_task'] is proof['full_cloud_bytes_independently_read'] is True
    base = args.byte_admission.parent
    assert base.resolve().is_relative_to(R/'artifacts')
    assert sha(base/'receipt.json') == proof['artifacts']['receipt']['sha256']
    receipt = json.loads((base/'receipt.json').read_bytes())
    assert receipt['task_id'] == proof['task_id'] and receipt['plan'] == proof['plan']
    assert sha(PRODUCER.parent/'source-freeze.json') == TRANSPORT_FREEZE_SHA
    assert receipt['plan']['bootstrap_sha256'] == sha(PRODUCER)
    assert sha(PACKAGE/'manifest.json') == MANIFEST_SHA
    manifest = json.loads((PACKAGE/'manifest.json').read_bytes())
    root = base/'unpack/CUDA-candidate'
    command = json.loads((root.parent/'command.json').read_bytes())
    assert command['argv'][1:] == manifest['command_relative_to_bundle_root'][1:]
    assert Path(command['argv'][0]).name.startswith('python')
    assert set(receipt['output_files']) == {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    for name,spec in receipt['output_files'].items():
        path = root/name
        assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
        assert sha(path) == spec['sha256'] and path.stat().st_size == spec['bytes']
    candidate,binding,witness = check_execution(root,receipt,proof['plan'],manifest)
    own = Path(__file__).resolve().parent
    frozen = json.loads((own/'source-freeze.json').read_bytes())
    for name,spec in frozen['sources'].items(): assert sha(own/name) == spec['sha256']
    for spec in frozen['references']: assert sha(spec['path']) == spec['sha256']
    expected = candidate['database_sha256']['candidate']; database = root/'candidate.sqlite'
    assert sha(database) == expected and sha(REFERENCE) == REFERENCE_SHA
    assert sha(REFERENCE_PROOF) == REFERENCE_PROOF_SHA
    checker = load(CHECKER,CHECKER_SHA,'CUDA_frozen_independent_correspondence')
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        left = sqlite3.connect(REFERENCE.resolve().as_uri()+'?mode=ro',uri=True)
        right = sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            assert right.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            a,b = ({k:json.loads(v) for k,v in db.execute('SELECT k,v FROM meta')} for db in (left,right))
            assert a.pop('schema') == 'persistent_exclusive_completion_component_identity_v1'
            assert b.pop('schema') == 'experimental_exclusive_batched_state_v1'
            execution = b.pop('state_execution')
            assert execution == dict(recipe='independent-root-wave-float64-state-candidate-v1',device='cuda:0',max_batch=64)
            sa,sb = a.pop('state'),b.pop('state'); assert a == b
            assert binding['configuration'] == a['config']
            for value in (sa,sb): value.pop('audit_sha256'); value.pop('prediction_sha256')
            assert sa == sb
            tables = checker.compare_tables(left,right)
            with (root/'branch-commitment-witness.jsonl').open() as stream:
                result = correspondence(left,right,execution,a['config'],stream,checker)
            assert result['events'] == 195 and result['branch_commitments'] == 2*witness['branches']
            # Expired branches have no projections and count zero on both sides.
            assert result['witness_projections'] == witness['branch_predictions']
            new(args.output/'correspondence.json',dict(tables=tables,**result))
        finally: left.close(); right.close()
        oracle = load(checker.FRESH,checker.FRESH_SHA,'CUDA_frozen_fresh_history_oracle')
        def progress(value):
            print(json.dumps(dict(stage='CUDA_independent_fresh_history',ETA='unknown',**value)),flush=True)
        numeric = oracle.verify_database(database,expected,progress)
        assert numeric['events'] == 195 and numeric['states'] == 56389
        assert numeric['stored_states_and_chosen_outputs_checked'] is True
        assert sha(database) == expected and sha(REFERENCE) == REFERENCE_SHA
        new(args.output/'fresh-state-independent.json',numeric)
        acceptance = args.output/'acceptance.json'
        new(acceptance,dict(kind='rbf_branch_state_CUDA_complete_sequence_independent_acceptance_v1',
            task_id=proof['task_id'],byte_admission_sha256=sha(args.byte_admission),source_sha256=sha(__file__),
            original_correspondence_source_sha256=CHECKER_SHA,fresh_oracle_sha256=checker.FRESH_SHA,
            correspondence_sha256=sha(args.output/'correspondence.json'),fresh_state_sha256=sha(args.output/'fresh-state-independent.json'),
            events=195,states=56389,atol=1e-8,rtol=1e-8,device=receipt['device'],actual_GPU_execution=True,
            max_fresh_state_error=numeric['max_abs_error'],all_branch_commitments_and_projections_checked=True,
            full_cohort_accepted=False,production_promotion_allowed=False,memory_target_accepted=False,
            isolated_speedup_accepted=False,paper_performance_complete=False))
        register(acceptance,'rbf-branch-state-CUDA-complete-sequence-independent')
        print(json.dumps(dict(receipt=str(acceptance))),flush=True)
    except BaseException as error:
        path = args.output/'failure.json'
        new(path,dict(exception_type=type(error).__name__,message=str(error),accepted=False,
            input_byte_admission_sha256=sha(args.byte_admission),automatic_retry=False))
        register(path,'rbf-branch-state-CUDA-independent-failure'); raise


if __name__ == '__main__': main()
