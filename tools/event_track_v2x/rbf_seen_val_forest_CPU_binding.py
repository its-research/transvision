"""Admission for a separate full seen-val forest audit; never inherit train results."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from rbf_nested_seen_val_v2_common import R, sha

ORIGINAL = R/'source-freezes/rbf-final-refit-full-forest-independent-CPU-v4-receipt-rows-20261004'
ORIGINAL_SHA = 'f36f644f5f18fad3e5ab489edbc949e93bcab4efce22749dcb90211c8c8b27f7'
READER = R/'source-freezes/rbf-seen-val-bound-forest-output-reader-v1-20261005'
READER_SHA = '39fb77dc9166f4c3125ad87ec66282da0561eb962901f1ce1bbd904e59d452b8'
SCOPE = 'SPD seen-val exploratory scheduled snapshots; bound allocator baseline'
KIND = 'rbf_seen_val_bound_forest_full_independent_structure_causal_mass_state_action_cache_acceptance_v1'


def frozen_modules():
    for directory, digest in ((ORIGINAL, ORIGINAL_SHA), (READER, READER_SHA)):
        assert sha(directory/'source-freeze.json') == digest
        freeze = json.loads((directory/'source-freeze.json').read_bytes())
        for name, item in freeze['sources'].items():
            assert sha(directory/name) == (item['sha256'] if isinstance(item, dict) else item)
    # Fail if another imported source has captured these module names.
    modules = []
    for directory, name in ((ORIGINAL, 'rbf_final_refit_forest_binding'),
                            (ORIGINAL, 'final_cache203'),
                            (READER, 'rbf_seen_val_forest_output_binding'),
                            (READER, 'read_rbf_seen_val_forest_outputs')):
        sys.path.insert(0, str(directory))
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve() == directory/(name+'.py')
        modules.append(module)
    modules[0].source_gate()
    modules[2].source_gate()
    return tuple(modules)


def source_gate():
    directory = Path(__file__).resolve().parent
    freeze = json.loads((directory/'source-freeze.json').read_bytes())
    assert freeze['kind'] == 'rbf_seen_val_bound_forest_full_independent_CPU_source_v1'
    for name, item in freeze['sources'].items(): assert sha(directory/name) == item['sha256']
    for item in freeze['references']: assert sha(item['path']) == item['sha256']
    frozen_modules()
    return dict(seen_val_CPU_source_freeze_sha256=sha(directory/'source-freeze.json'),
        original_train_CPU_source_freeze_sha256=ORIGINAL_SHA,
        seen_val_byte_reader_source_freeze_sha256=READER_SHA,
        scope=SCOPE, measured_network_arrival_history_verified=False,
        prior_train_or_other_seed_forest_acceptance_inherited=False)


def registered(path):
    path = Path(path)
    assert path.resolve().is_relative_to(R)
    assert not any(p.is_symlink() for p in (path,*path.parents))
    digest = sha(path)
    entries = json.loads((R/'receipts/20260928-execution-ledger.json').read_bytes())['entries']
    assert any(e.get('receipt') == str(path) and e.get('receipt_sha256') == digest for e in entries)
    return json.loads(path.read_bytes())


def validate_local(value, job, output_binding):
    plan = job['plan']
    assert value['kind'] == output_binding.KIND
    assert value['seed'] == job['seed'] == plan['seed'] in (1337,2027,3407)
    assert value['task_id'] == job['task_id']
    assert value['recipe_sha256'] == job['recipe_sha256'] == hashlib.sha256(output_binding.canonical(plan)).hexdigest()
    assert value['method'] == plan['method'] == 'rbf' and plan['configuration']['allocation'] == 'bound'
    assert value['world_size'] == plan['world_size'] in (4,8)
    assert value['NN_atol'] == value['NN_rtol'] == plan['scoring_atol'] == plan['scoring_rtol'] == 1e-4
    assert value['all_registered_bytes_verified'] is True
    assert value['all_21_sequences_3316_events_and_final_model_factor_nodes_verified'] is True
    for flag in ('measured_network_arrival_history_verified','full_forest_semantics_or_fresh_state_independently_accepted',
                 'learned_Stage2_complete','same_resource_performance_accepted','paper_performance_complete'):
        assert value[flag] is False
    assert value['scope'] == plan['scope'] == plan['evaluation_scope'] == SCOPE
    assert value['input_numeric_index_sha256'] == output_binding.INDEX_SHA
    assert value['source_sha256'] == sha(READER/'read_rbf_seen_val_forest_outputs.py')
    assert value['publication_sha256'] == job['publication_sha256']
    assert value['final_checkpoint_sha256'] == plan['checkpoint']['sha256']
    assert value['final_model_sha256'] == plan['final_refit_model_sha256']
    assert set(value['artifacts']) == output_binding.artifact_keys(plan['world_size'])
    sequences = value['sequences']
    assert len(sequences) == len({s['sequence_id'] for s in sequences}) == 21
    assert sum(s['events'] for s in sequences) == 3316
    assert sum(s['nodes'] for s in sequences) == value['total_nodes']
    for item in sequences:
        sid = item['sequence_id']
        assert isinstance(sid,str) and sid and '/' not in sid and '\\' not in sid and sid not in ('.','..')
        assert type(item['events']) is int and item['events'] > 0
        assert type(item['nodes']) is int and item['nodes'] >= 0
        digest = item['database_sha256']
        assert len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
    return plan


def validate(path, Task=None):
    source_gate()
    _, _, output_binding, reader = frozen_modules()
    value = registered(path)
    jobs = json.loads(output_binding.dispatch.JOURNAL.read_bytes())['jobs']
    matches = [j for j in jobs if j['seed'] == value['seed']]
    assert len(matches) == 1
    job = matches[0]; plan = validate_local(value,job,output_binding)
    if Task is None:
        from clearml import Task
    task = Task.get_task(task_id=job['task_id'])
    assert str(task.status) == 'completed'
    args = SimpleNamespace(seed=job['seed'], **{k:Path(job[k]) for k in ('main_admission','main_byte_admission','publication')})
    published = output_binding.qualify_job(args,Task,job,task)
    output_binding.verify_task_unchanged(task,job,value['artifacts'],'completed')
    root = Path(path).parent
    for key, item in value['artifacts'].items():
        local = reader.local_artifact(root,key)
        assert local.stat().st_size == item['bytes'] and sha(local) == item['sha256']
    report = json.loads((root/'receipt.json').read_bytes())
    binding = json.loads((root/'seen-val-input-binding.json').read_bytes())
    output_binding.verify_report(plan,report,binding,published)
    entry, _, expected, event_document, groups = output_binding.reference_inputs(job['seed'],plan)
    assert value['total_nodes'] == entry['rows']
    assert {s['sequence_id']:s['events'] for s in value['sequences']} == {s:len(e) for s,e in groups.items()}
    assert {s['sequence_id']:s['nodes'] for s in value['sequences']} == {s:len(e) for s,e in expected.items()}
    # Per-sequence serialized inputs must still agree after byte admission.
    for seq in value['sequences']:
        sid = seq['sequence_id']; found = list(root.glob(f'rank*-unpack/rank-*/{sid}/receipt.json'))
        assert len(found) == 1
        directory = found[0].parent
        receipt = json.loads(output_binding.contained(directory,'receipt.json').read_bytes())
        per_plan = json.loads(output_binding.contained(directory,'plan.json').read_bytes())
        output_binding.verify_sequence(plan,per_plan,receipt,sid,groups[sid])
        db = receipt['databases'][sid]
        assert db['sha256'] == seq['database_sha256']
        assert sha(output_binding.contained(directory,db['path'])) == db['sha256']
    return job
