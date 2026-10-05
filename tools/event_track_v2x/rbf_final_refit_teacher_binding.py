"""Exact final-model teacher lineage, shared by future dispatch and readers.

This metadata gate cannot admit a teacher trajectory. It reuses the final-main
acceptance and the finite Linux witness qualification without recomputation.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

from rbf_nested_seen_val_v2_common import R, sha

PRODUCER = R/'source-freezes/rbf-final-refit-capacity-witness-teacher-producer-v1-20261004'
PRODUCER_PREPARATION_SHA = 'a82ca538f414a29e471e9b43b6b01e77a4205bd05305dd5c146f13dfaa3e042d'
MAIN_JOURNAL = R/'receipts/rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json'
PREREQUISITE_KIND = 'rbf_final_refit_main_prerequisite_for_capacity_teacher_v1'
ALLOCATION_VARIANT = 'final_refit_exclusive_capacity_raw_witness_teacher_v1'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def source_gate():
    assert sha(PRODUCER/'preparation.json') == PRODUCER_PREPARATION_SHA
    preparation = json.loads((PRODUCER/'preparation.json').read_bytes())
    for name,checksum in preparation['sources'].items(): assert sha(PRODUCER/name) == checksum
    control = json.loads((PRODUCER/'source-control.json').read_bytes())
    assert control['original_replacements_unchanged'] is control['real_replay_function_AST_identical'] is True
    return preparation,control


def expected_plan(main_job, prerequisite, prerequisite_artifact, producer_sha, patches, world_size):
    assert world_size in (4,8) and type(world_size) is int
    base = main_job['plan']
    assert prerequisite['kind'] == PREREQUISITE_KIND
    assert prerequisite['main_prerequisite_verified'] is True
    assert prerequisite['teacher_runtime_or_targets_admitted'] is False
    assert prerequisite['seed'] == main_job['seed'] == base['seed'] in (1337,2027,3407)
    assert prerequisite['main_task_id'] == main_job['task_id']
    assert prerequisite['final_model_sha256'] == base['final_refit_model_sha256'] != base['original_nested_model_sha256']
    assert len(prerequisite['sequence_proof_sha256']) == 46
    assert prerequisite_artifact['bytes'] > 0 and prerequisite_artifact['key'] == 'final-main-prerequisite'
    assert base['configuration']['method'] == base['method'] == 'rbf'
    assert base['configuration']['allocation'] == 'bound'
    assert base['configuration']['backend'] == 'exclusive_root_partition_regions_v1'
    assert base['configuration']['state']['candidate_protocol'] == 'rbf-all-class-top64-v1'
    assert set(base['exclusive_patches']) < set(patches)
    assert all(patches[key] == value for key,value in base['exclusive_patches'].items())
    assert set(patches)-set(base['exclusive_patches']) == {
        'transvision/models/event_track_v2x/exclusive_teacher_probe_witness.py',
        'transvision/models/event_track_v2x/exclusive_witness_paper_runtime.py'}
    return dict(base,world_size=world_size,bootstrap_sha256=producer_sha,
        exclusive_patches=patches,configuration=dict(base['configuration'],allocation='teacher'),
        final_main_prerequisite_artifact=prerequisite_artifact,
        final_main_acceptance_sha256=prerequisite['main_acceptance_sha256'],
        final_main_byte_admission_sha256=prerequisite['byte_admission_sha256'])


def prerequisites(acceptance, byte, seed):
    preparation,control = source_gate()
    path = PRODUCER/'rbf_final_refit_teacher_prerequisites.py'
    spec = importlib.util.spec_from_file_location('final_teacher_prerequisite_gate',path)
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    proof = gate.gate(acceptance,byte,seed)
    main = next(j for j in json.loads(MAIN_JOURNAL.read_bytes())['jobs'] if j['task_id'] == proof['main_task_id'])
    assert main['seed'] == seed
    return proof,main,preparation,control


def validate_registered(job, acceptance, byte, published_prerequisite):
    proof,main,preparation,control = prerequisites(acceptance,byte,job['seed'])
    assert job['allocation_variant'] == ALLOCATION_VARIANT
    assert json.loads(published_prerequisite.read_bytes()) == proof
    plan = job['plan']; artifact = plan['final_main_prerequisite_artifact']
    assert artifact['sha256'] == sha(published_prerequisite) and artifact['bytes'] == published_prerequisite.stat().st_size
    assert plan == expected_plan(main,proof,artifact,preparation['bootstrap_sha256'],control['patches'],plan['world_size'])
    assert job['recipe_sha256'] == hashlib.sha256(canonical(plan)).hexdigest()
    from clearml import Task
    for spec in [artifact,*[plan[k] for k in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest')],*plan['forward_outputs']]:
        task = Task.get_task(task_id=spec['task'])
        assert str(task.status) == 'completed'
        assert task.artifacts[spec['key']].hash == spec['sha256'] and task.artifacts[spec['key']].size == spec['bytes']
    task = Task.get_task(task_id=job['task_id'])
    assert str(task.status) == 'completed', 'complete teacher required before collection'
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == preparation['bootstrap_sha256']
    assert task.get_parameters()['General/recipe_sha256'] == job['recipe_sha256']
    assert json.loads(task.get_parameters()['General/plan']) == plan
    return task,proof
