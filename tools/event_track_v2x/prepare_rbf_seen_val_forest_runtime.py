"""Prepare the distinct SPD seen-val bound-forest producer, without dispatch.

The train kernel, weights, capacities, selection and numerical tolerances stay
fixed. Actual execution requires full train admission and independently read
cloud inputs. This preparation cannot supply those missing receipts.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import shutil

from rbf_nested_seen_val_v2_common import R, new, register, sha

PARENT = R/'source-freezes/rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004'
PARENT_SHA = 'e363579c19831420b9782eb7aef4d96e50f3ab303239942ed78923ee481929dd'
MAIN_FREEZE_SHA = 'f36f644f5f18fad3e5ab489edbc949e93bcab4efce22749dcb90211c8c8b27f7'
INPUT_INDEX = R/'receipts/rbf-seen-val-three-seed-forest-input-transport-review-20261005.json'
INPUT_INDEX_SHA = '2e140147735794d7126adcdb67e6e15efa6d6cd3ee53a5154c3b0328c9b58719'
NAME = 'rbf-seen-val-bound-forest-GPU-producer-v1-20261005'


def seen_val_admission(plan, base):
    """Injected remote gate; fetch and Task are source-pinned parent helpers."""
    publication = plan['seen_val_input_publication']
    assert publication['kind'] == 'rbf_seen_val_complete_forest_inputs_cloud_bytes_v1'
    assert publication['seed'] == plan['seed'] and publication['independent_cloud_bytes_verified'] is True
    inputs = publication['artifacts']
    assert set(inputs) == {'cache_archive', 'cache_manifest', 'events', 'binding', 'transport', 'review', 'main_acceptance'}
    assert len({v['task'] for v in inputs.values()}) == 1
    task = Task.get_task(task_id=inputs['binding']['task'])
    assert str(task.status) == 'completed'
    assert len(task.artifacts) == len(inputs) and set(task.artifacts) == {v['key'] for v in inputs.values()}
    for role, spec in inputs.items():
        item = task.artifacts[spec['key']]
        assert item.hash == spec['sha256'] and item.size == spec['bytes']
        if role in ('cache_archive', 'cache_manifest', 'events'):
            assert plan[role] == spec
        else:
            fetch(spec, base/('input-'+role+'.json'))
    b, transport, review, main = [json.loads((base/('input-'+role+'.json')).read_bytes())
                                 for role in ('binding','transport','review','main_acceptance')]
    assert inputs['review']['sha256'] == INPUT_INDEX_SHA
    assert b['kind'] == 'rbf_seen_val_forest_input_bridge_v1' and b['seed'] == plan['seed']
    assert b['full_original_schedule_to_forest_event_binding_passed'] is True
    assert b['events'] == 3316 and b['sequences'] == 21 and b['cache_frames'] == 7189
    assert b['checkpoint']['sha256'] == plan['checkpoint']['sha256']
    assert b['checkpoint']['model_sha256'] == plan['final_refit_model_sha256']
    assert b['forward_artifacts'] == plan['forward_outputs']
    assert b['events_sha256'] == plan['events']['sha256']
    assert transport['kind'] == 'rbf-seen-val-forest-cache-transport-v1-20261005'
    assert transport['seed'] == plan['seed'] and transport['all_member_bytes_independently_read'] is True
    assert transport['input_binding_sha256'] == inputs['binding']['sha256']
    assert transport['members'] == 14379 and transport['events_sha256'] == plan['events']['sha256']
    assert transport['cache_manifest_sha256'] == plan['cache_manifest']['sha256']
    for key in ('sha256','bytes'):
        assert transport['archive'][key] == plan['cache_archive'][key]
    assert review['all_three_local_forest_input_packages_bound'] is True
    chosen = [v for v in review['seeds'] if v['seed'] == plan['seed']]
    assert len(chosen) == 1 and chosen[0]['full_schedule_transport_linkage_review_passed'] is True
    assert chosen[0]['rows'] == b['rows']
    assert {v['sha256'] for v in chosen[0]['receipts']} == {inputs['binding']['sha256'], inputs['transport']['sha256']}
    gate = publication['full_train_main_local_gate']
    assert gate['kind'] == 'rbf_final_refit_main_prerequisite_for_capacity_teacher_v1'
    assert gate['main_prerequisite_verified'] is True and gate['seed'] == plan['seed']
    assert gate['main_acceptance_sha256'] == inputs['main_acceptance']['sha256']
    assert len(gate['sequence_proof_sha256']) == 46
    assert main['kind'] == 'rbf_final_refit_full_exclusive_cohort_independent_structure_causal_mass_recovery_state_action_scope_acceptance_v1'
    assert main['seed'] == plan['seed'] and main['task_id'] == gate['main_task_id']
    assert main['completed_sequences'] == 46 and main['completed_events'] == 7445
    assert main['driver_freeze_sha256'] == MAIN_FREEZE_SHA
    assert main['checkpoint_sha256'] == plan['checkpoint']['sha256']
    assert main['final_model_sha256'] == plan['final_refit_model_sha256'] == gate['final_model_sha256']
    assert main['byte_admission_sha256'] == gate['byte_admission_sha256']
    for flag in ('strict_declared_support_partition_coverage', 'padded_float64_mass_arithmetic_verified',
                 'output_class_recovery_provenance_verified', 'all_raw_factor_and_causal_commits_verified',
                 'all_decisions_search_or_capacity_undecided_semantics_verified',
                 'all_203_feature_recipe_values_independently_verified',
                 'all_scorer_contexts_bound_to_original_causal_raw_history'):
        assert main[flag] is True
    assert main['atol'] == main['rtol'] == 1e-8
    assert main['old_model_full_forest_acceptance_inherited'] is False
    assert main['paper_performance_complete'] is False
    upstream = publication['upstream_main']
    assert upstream['task_id'] == main['task_id']
    original_task = Task.get_task(task_id=main['task_id'])
    assert str(original_task.status) == 'completed'
    assert hashlib.sha256(original_task.data.script.diff.encode()).hexdigest() == PARENT_SHA
    params = original_task.get_parameters()
    original = json.loads(params['General/plan'])
    assert hashlib.sha256(canonical(original)).hexdigest() == params['General/recipe_sha256'] == main['recipe_sha256']
    assert original['world_size'] in (4,8)
    expected = {'receipt','exclusive-source-manifest'} | {f'replay-rank{i}' for i in range(original['world_size'])}
    assert set(original_task.artifacts) == set(upstream['artifacts']) == expected
    for key, spec in upstream['artifacts'].items():
        assert original_task.artifacts[key].hash == spec['sha256'] and original_task.artifacts[key].size == spec['bytes']
    for key in ('seed', 'method', 'configuration', 'checkpoint', 'weights_archive', 'source',
                'source_replacements', 'exclusive_patches', 'CPU_capacity_candidate_admission', 'final_refit_model_sha256'):
        assert original[key] == plan[key], ('seen-val changed frozen train contract', key)
    assert plan['method'] == 'rbf' and plan['configuration']['allocation'] == 'bound'
    assert plan['evaluation_scope'] == 'SPD seen-val exploratory scheduled snapshots; bound allocator baseline'
    write(base/'seen-val-input-binding.json', dict(seed=plan['seed'],
        input_publication=publication, full_train_interface_required=True,
        validation_or_test_selection=False, measured_network_arrival_history_verified=False,
        learned_Stage2_complete=False, same_resource_performance_accepted=False, paper_performance_complete=False))


def literals(source):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(source).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ('PATCHES','REPLACEMENTS')}


def build(parent):
    assert hashlib.sha256(parent.encode()).hexdigest() == PARENT_SHA
    candidate = parent
    edits = [
        ("protocol=PaperProtocol('spd','train')", "protocol=PaperProtocol('spd','val')"),
        ("task_name='final-refit exclusive full-train forest candidate'", "task_name='final-refit SPD seen-val bound forest candidate'"),
        ("base=Path('rbf-original-joint-exclusive-replay')", "base=Path('rbf-seen-val-joint-exclusive-replay')"),
        (" try:\n  for key in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest'):",
         " try:\n  seen_val_admission(plan,base)\n  for key in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest'):"),
        ("assert len(ev['events'])==7445 and len(ev['origin_us_by_sequence'])==46",
         "assert ev['kind']=='rbf_matching_seen_val_complete_forest_events_v1' and ev['split']=='val' and ev['seed']==plan['seed'];assert len(ev['events'])==3316 and len(ev['origin_us_by_sequence'])==21;assert ev['arrival_policy']=='scheduled_pair_snapshot_at_reference_plus_100ms' and ev['measured_network_arrival_history_verified'] is False"),
        ("'kind':'rbf_final_refit_exclusive_full_train_forest_candidate_v1'", "'kind':'rbf_final_refit_SPD_seen_val_bound_forest_candidate_v1'"),
        ("'all_46_sequences_7445_events_completed'", "'all_21_sequences_3316_events_completed'"),
        ("'scope':'full paired train, distinct frozen final-refit checkpoint, unchanged exclusive kernel and limits; bound allocator only; full independent forest/learned Stage2/MHT/metrics pending'",
         "'scope':'SPD seen-val exploratory scheduled snapshots, unchanged final checkpoint and bound allocator baseline; learned Stage2, resources and independent metrics pending'"),
        (" task.upload_artifact('receipt',artifact_object=base/'receipt.json',wait_on_upload=True)",
         " if (base/'seen-val-input-binding.json').exists():task.upload_artifact('seen-val-input-binding',artifact_object=base/'seen-val-input-binding.json',wait_on_upload=True)\n task.upload_artifact('receipt',artifact_object=base/'receipt.json',wait_on_upload=True)")
    ]
    for before, after in edits:
        assert candidate.count(before) == 1, ('parent contract changed', before)
        candidate = candidate.replace(before, after, 1)
    constants = f'\nPARENT_SHA={PARENT_SHA!r}\nMAIN_FREEZE_SHA={MAIN_FREEZE_SHA!r}\nINPUT_INDEX_SHA={INPUT_INDEX_SHA!r}\n'
    candidate = candidate.replace('\ndef main():', constants+'\n'+inspect.getsource(seen_val_admission)+'\ndef main():', 1)
    compile(candidate, 'bootstrap.py', 'exec')
    assert literals(candidate) == literals(parent)
    def functions(source):
        return {n.name: ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    left, right = functions(parent), functions(candidate)
    assert {k:v for k,v in left.items() if k not in ('work','main')} == {k:v for k,v in right.items() if k not in ('work','main','seen_val_admission')}
    work = next(n for n in ast.parse(parent).body if isinstance(n,ast.FunctionDef) and n.name == 'work')
    expected = ast.get_source_segment(parent,work).replace("PaperProtocol('spd','train')", "PaperProtocol('spd','val')")
    assert functions(expected)['work'] == right['work']
    return candidate, dict(parent_sha256=PARENT_SHA, edits=edits,
        forest_kernel_and_source_replacements_unchanged=True, worker_only_change='PaperProtocol spd train -> val',
        configuration_and_caps_changed=False, scoring_atol=1e-4, scoring_rtol=1e-4,
        full_train_main_admission_required=True, publication_byte_readback_required=True,
        dispatch_ready=False, actual_GPU_execution=False, paper_performance_complete=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=R/'source-freezes'/NAME)
    parser.add_argument('--software-log',type=Path,required=True)
    args = parser.parse_args()
    assert args.output == R/'source-freezes'/NAME and not args.output.exists()
    test_output = args.software_log.read_text()
    assert '19 passed' in test_output and 'FAILED' not in test_output and 'ERROR' not in test_output
    assert sha(INPUT_INDEX) == INPUT_INDEX_SHA
    index = json.loads(INPUT_INDEX.read_bytes())
    assert index['all_three_local_forest_input_packages_bound'] is True
    parent = (PARENT/'bootstrap.py').read_text()
    candidate, control = build(parent)
    prepared = json.loads((PARENT/'preparation.json').read_bytes())
    drafts = []
    for item in index['seeds']:
        seed = item['seed']
        directory = R/f'artifacts/rbf-seen-val-forest-input-bridge-v1-20261005/seed{seed}'
        for proof in item['receipts']:
            assert sha(proof['path']) == proof['sha256']
        binding = json.loads((directory/'input-binding.json').read_bytes())
        prior = next(p['plan'] for p in prepared['seeds'] if p['seed']==seed)
        assert prior['checkpoint']['sha256'] == binding['checkpoint']['sha256']
        drafts.append(dict(seed=seed, configuration=prior['configuration'],
            existing_cloud_inputs={k:prior[k] for k in ('checkpoint','weights_archive','source')},
            forward_outputs=binding['forward_artifacts'], local_cache_archive=item['archive'],
            local_events=dict(path=str(directory/'events.json'),sha256=sha(directory/'events.json')),
            input_binding_sha256=sha(directory/'input-binding.json'),
            required_full_train_main_admission=None, required_cloud_input_publication=None,
            dispatch_ready=False))
    args.output.mkdir()
    (args.output/'bootstrap.py').write_text(candidate)
    shutil.copyfile(__file__,args.output/Path(__file__).name)
    shutil.copyfile(Path(__file__).with_name('rbf_nested_seen_val_v2_common.py'),args.output/'rbf_nested_seen_val_v2_common.py')
    tests = Path(__file__).resolve().parents[2]/'tests/event_track_v2x/test_seen_val_forest_runtime.py'
    shutil.copyfile(tests,args.output/tests.name)
    shutil.copyfile(args.software_log,args.output/'software-tests.log')
    new(args.output/'source-control.json',control)
    new(args.output/'draft-inputs.json',drafts)
    proof = dict(kind=NAME, sources={p.name:sha(p) for p in args.output.iterdir() if p.is_file()},
        original_parent_source_sha256=PARENT_SHA, input_review_sha256=INPUT_INDEX_SHA,
        software_tests_passed=19, software_scope='AST preservation and software-only remote metadata gate fixtures; not actual forest acceptance',
        real_forest_function_bytes_unchanged=True, dispatch_ready=False, actual_GPU_execution=False,
        full_forest_independently_accepted=False, paper_performance_complete=False)
    new(args.output/'source-freeze.json',proof)
    register(args.output/'source-freeze.json',NAME+'_preparation')
    print(json.dumps(dict(source_freeze=str(args.output/'source-freeze.json'),
        sha256=sha(args.output/'source-freeze.json'),bootstrap_sha256=sha(args.output/'bootstrap.py'),dispatch_ready=False)),flush=True)


if __name__ == '__main__':
    main()
