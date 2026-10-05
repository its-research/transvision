"""Read-only lineage gates for actual final learned-priority replay outputs.

These checks bind an output to an admitted fitted model. They do not verify
the learned expansion ordering or continuous branch states themselves.
"""
import hashlib
import json
from pathlib import Path

from rbf_nested_seen_val_v2_common import R, sha
import submit_final_refit_learned_replay as dispatch
import publish_final_refit_priority_checkpoint as publication

DISPATCH = R/'source-freezes/rbf-final-refit-learned-priority-GPU-dispatch-v1-20261005'
DISPATCH_SHA = '287b3583ad9fc9a93ac72e46ad1462997e4aabfa8354738b039c9001f6acbf8c'
ORIGINAL_READER = R/'source-freezes/rbf-final-refit-main-topK-independent-output-reader-v1-20261004/read_rbf_final_refit_forest_outputs.py'
ORIGINAL_READER_SHA = '53747489a446c8990cfc6d9310eb686a69b39d99bde226c8945665a974d83f61'
KIND = 'rbf_final_refit_learned_priority_independent_bytes_events_factor_admission_v1'


def source_gate():
    own = Path(__file__).resolve().parent
    assert sha(DISPATCH/'source-freeze.json') == DISPATCH_SHA
    original = json.loads((DISPATCH/'source-freeze.json').read_bytes())
    for name, item in original['sources'].items():
        assert sha(DISPATCH/name) == item['sha256']
    assert sha(ORIGINAL_READER) == ORIGINAL_READER_SHA
    for module in (dispatch, publication):
        path = Path(module.__file__).resolve()
        assert path.parent == own, 'ambient dispatcher or publisher import forbidden'
        assert sha(path) == original['sources'][path.name]['sha256']
    return dispatch.source_gate()


def arguments(parser):
    parser.add_argument('--seed', type=int, choices=(1337, 2027, 3407), required=True)
    for name in ('run', 'numeric-audit', 'teacher-journal', 'published-prerequisite',
                 'runtime-admission', 'priority-publication'):
        parser.add_argument('--'+name, type=Path, required=True)


def qualify_job(args, Task, job, task, control):
    """Recheck the same real teacher/fit/publication prerequisites as dispatch."""
    assert job['seed'] == args.seed == job['plan']['seed']
    assert job['allocation_variant'] == dispatch.VARIANT
    assert job['priority_publication'] == str(args.priority_publication)
    assert sha(args.priority_publication) == job['priority_publication_sha256']
    paths, context = publication.qualify(args, Task)
    proof = publication.verify_publication(args.priority_publication, paths, context, Task)
    main = Task.get_task(task_id=context['upstream_tasks']['main']['task_id'])
    main_plan = json.loads(main.get_parameters()['General/plan'])
    assert hashlib.sha256(dispatch.canonical(main_plan)).hexdigest() == context['upstream_tasks']['main']['recipe_sha256']
    base = dispatch.expected_plan(main_plan, proof, control, sha(dispatch.PRODUCER/'bootstrap.py'), sha(dispatch.__file__))
    dispatch.verify_existing(task, job, base, dispatch.semantic(base))
    assert job['plan']['priority_publication'] == proof
    assert job['plan']['configuration']['allocation'] == 'learned'
    return proof


def artifact_keys(world):
    assert type(world) is int and world in (4, 8)
    return {'receipt', 'exclusive-source-manifest', 'priority-input-binding'} | {
        f'replay-rank{i}' for i in range(world)}


def contained(directory, relative):
    directory, relative = Path(directory), Path(relative)
    assert not relative.is_absolute() and '..' not in relative.parts
    path = directory/relative
    assert path.is_file() and path.resolve().is_relative_to(directory.resolve())
    assert not any(p.is_symlink() for p in (path, *path.parents))
    return path


def verify_priority_report(plan, report, binding, proof):
    assert plan['priority_publication'] == proof == binding['publication']
    assert plan['priority_inputs'] == proof['artifacts'] == binding['artifacts']
    assert binding['kind'] == 'rbf_final_refit_learned_replay_priority_input_binding_v1'
    assert binding['policy_signature'] == proof['policy_signature'] == plan['priority_policy_signature']
    assert binding['local_numeric_audit_sha256'] == plan['priority_inputs']['audit']['sha256']
    assert binding['learned_replay_accepted'] is binding['paper_performance_complete'] is False
    for role in ('main', 'teacher'):
        assert binding[role+'_task_id'] == proof['upstream_tasks'][role]['task_id']
    assert report['kind'] == 'rbf_final_refit_learned_priority_full_train_forest_candidate_v1'
    assert len(report['ranks']) == plan['world_size']
    assert sorted(r['rank'] for r in report['ranks']) == list(range(plan['world_size']))
    for rank in report['ranks']:
        assert rank['seed'] == plan['seed']
        assert rank['priority_policy_signature'] == plan['priority_policy_signature']
        assert rank['priority_checkpoint_sha256'] == plan['priority_inputs']['checkpoint']['sha256']


def verify_priority_sequence(plan, per_plan, receipt, sequence, event_count):
    assert per_plan['kind'] == 'rbf_paper_replay_v1' and receipt['kind'] == 'rbf_paper_replay_receipt_v1'
    assert per_plan['expected_sequences'] == receipt['completed_sequences'] == [sequence]
    assert per_plan['expected_events'] == receipt['completed_events'] == event_count
    assert per_plan['fixture'] is receipt['fixture'] is False
    assert per_plan['protocol']['split'] == per_plan['model_binding']['fit_split'] == 'train'
    assert per_plan['protocol']['dataset'] == per_plan['model_binding']['dataset'] == 'spd'
    assert per_plan['configuration'] == plan['configuration']
    assert per_plan['cache_sha256'] == plan['cache_manifest']['sha256']
    assert per_plan['model_binding']['seed'] == plan['seed']
    assert per_plan['model_binding']['priority_policy_signature'] == plan['priority_policy_signature']
    assert per_plan['model_binding']['priority_checkpoint_sha256'] == plan['priority_inputs']['checkpoint']['sha256']
    assert {'plan.json', 'predictions.jsonl', 'audit.jsonl', 'timings.json', 'resources.json'} <= set(receipt['files'])
    assert set(receipt['databases']) == {sequence}


def verify_task_unchanged(task, job, inventory, status):
    task.reload()
    assert str(task.status) == status
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == job['plan']['bootstrap_sha256']
    params = task.get_parameters()
    assert json.loads(params['General/plan']) == job['plan']
    assert params['General/recipe_sha256'] == job['recipe_sha256']
    assert set(task.artifacts) == set(inventory)
    for key, item in inventory.items():
        assert task.artifacts[key].hash == item['sha256'] and task.artifacts[key].size == item['bytes']
