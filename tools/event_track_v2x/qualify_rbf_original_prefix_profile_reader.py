"""Software refusal controls only; never executes an original replay or a GPU."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def load_reader(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('prefix_profile_reader_qualification', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reader = load_reader(args.reader)
    preparation = json.loads((reader.EXECUTOR/'preparation.json').read_bytes())
    accepted = []

    def refuse(name, invoke):
        try:
            invoke()
        except (AssertionError, KeyError, ValueError, TypeError):
            accepted.append(name)
        else:
            raise AssertionError('software refusal was bypassed: '+name)

    for world in (4, 8):
        reader.gate_plan(dict(preparation['plan'], world_size=world), preparation)
    plan = dict(preparation['plan'], world_size=4)
    for key, value in (('world_size', True), ('world_size', 2), ('seed', 1337),
                       ('profile_prefix_events', 33), ('scoring_atol', 1e-3),
                       ('profiling_debug_contract', False), ('full_dataset_or_latency_claim', True)):
        wrong = dict(plan, **{key:value})
        refuse('refuse_plan_'+key+'_'+str(value), lambda wrong=wrong:reader.gate_plan(wrong, preparation))
    wrong = deepcopy(plan)
    wrong['configuration']['state']['candidate_protocol'] = 'car-only'
    refuse('refuse_car_only_competition', lambda:reader.gate_plan(wrong, preparation))
    wrong_limits = deepcopy(plan)
    wrong_limits['configuration']['limits']['max_prefix_nodes'] += 1
    refuse('refuse_changed_forest_work_cap', lambda:reader.gate_plan(wrong_limits, preparation))
    report = dict(kind='rbf_original_frozen_real_prefix_GPU_function_profile_candidate_v1',
        plan=plan, task_id='0'*32, paper_performance_complete=False,
        same_resource_baseline_comparison_accepted=False, full_46_sequence_replay_completed=False,
        paper_cost_admission=False, profiler_overhead_quantified=False)
    reader.gate_report(report, plan, '0'*32)
    for key in ('paper_performance_complete', 'same_resource_baseline_comparison_accepted',
                'full_46_sequence_replay_completed', 'paper_cost_admission', 'profiler_overhead_quantified'):
        bad = dict(report, **{key:True})
        refuse('refuse_report_claim_'+key, lambda bad=bad:reader.gate_report(bad, plan, '0'*32))
    refuse('refuse_other_task_identity', lambda:reader.gate_report(report, plan, '1'*32))
    # Deliberately manufactured counters exercise document acceptance, not
    # measured times, replay output, a checkpoint or any physical machine.
    sources = reader.expected_sources(plan)
    events, nodes = 32, 7
    rows = []
    entries = (('exclusive_paper_runtime.py','replay',1),
        ('persistent_cache_stream.py','step',events), ('persistent_cache_stream.py','rows',events),
        ('forest_potentials.py','__call__',nodes), ('forest_potentials.py','neural_parent_logits',nodes),
        ('learned_identity.py','forward',nodes), ('recoverable_identity.py','model_digest',nodes),
        ('recoverable_identity.py','_tensor_digest',nodes*10))
    for filename, function, count in entries:
        source = next(name for name in sources if Path(name).name == filename)
        rows.append(dict(source=source,source_sha256=sources[source],line=1,function=function,
            primitive_calls=count,total_calls=count,function_own_wall_seconds=0.,function_inclusive_wall_seconds=0.))
    document = dict(kind='rbf_allowlisted_original_replay_function_profile_v1',functions=rows,
        raw_call_arguments_persisted=False,unallowlisted_functions_or_paths_persisted=False,
        function_wall_times_include_blocking_and_are_not_GPU_kernel_times=True)
    counter = reader.gate_function_profile(document,sources,events,nodes)
    assert counter['parameter_tensors_hashed_per_query'] == 10
    for key, value in (('raw_call_arguments_persisted',True),
            ('unallowlisted_functions_or_paths_persisted',True),
            ('function_wall_times_include_blocking_and_are_not_GPU_kernel_times',False), ('raw_frames',[])):
        bad = deepcopy(document);bad[key] = value
        refuse('refuse_profile_'+key,lambda bad=bad:reader.gate_function_profile(bad,sources,events,nodes))
    for key, value in (('source_sha256','0'*64), ('source','/tmp/unallowlisted.py'),
                       ('total_calls',2), ('primitive_calls',2),
                       ('function_own_wall_seconds',float('nan')), ('function_inclusive_wall_seconds',-1),
                       ('call_arguments',{})):
        bad = deepcopy(document);bad['functions'][0][key] = value
        refuse('refuse_function_'+key,lambda bad=bad:reader.gate_function_profile(bad,sources,events,nodes))
    bad = deepcopy(document);bad['functions'].append(deepcopy(rows[0]))
    refuse('refuse_duplicate_function_key',lambda:reader.gate_function_profile(bad,sources,events,nodes))
    bad_missing = deepcopy(document);bad_missing['functions'] = rows[:-1]
    refuse('refuse_missing_tensor_digest_counter',lambda:reader.gate_function_profile(bad_missing,sources,events,nodes))
    bad_multiple = deepcopy(document);bad_multiple['functions'][-1]['total_calls'] -= 1
    bad_multiple['functions'][-1]['primitive_calls'] -= 1
    refuse('refuse_inconsistent_parameter_tensor_count',lambda:reader.gate_function_profile(bad_multiple,sources,events,nodes))
    refuse('refuse_zero_real_query_count',lambda:reader.gate_function_profile(document,sources,events,0))
    args.output.mkdir(parents=True,exist_ok=False)
    value = dict(kind='rbf_original_prefix_profile_reader_software_refusal_qualification_v1',
        reader_sha256=hashlib.sha256(args.reader.read_bytes()).hexdigest(),
        qualifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manufactured_document_cases_only=True,software_refusals=accepted,software_refusal_count=len(accepted),
        original_replay_body_executions=0,actual_clearml_queries=0,actual_GPU_profiles=0,
        full_stage2_or_latency_or_paper_admission=False)
    with (args.output/'qualification.json').open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(software_refusals=len(accepted),original_replay_executions=0,
        qualification=str(args.output/'qualification.json'))))


if __name__ == '__main__':
    main()
