"""Publish an already audited priority checkpoint and independently read it back.

Default mode only qualifies and describes the exact three-file payload. No
training data, SQLite, predictions, raw source or credentials are in that payload.
An interrupted publication is preserved, never automatically recreated.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

from rbf_nested_seen_val_v2_common import R, new, register, sha

PROJECT = 'Thesis/Recover-Before-Fuse/Inference'
DESTINATION = 'http://10.100.34.118:8081'
KIND = 'rbf_final_refit_priority_checkpoint_publication_independent_bytes_v1'
CONSUMER = R / 'source-freezes/rbf-final-refit-priority-export-consumer-v1-20261005'
CONSUMER_SHA = 'e358113abf6c23bab8cd2bb27f4a2df0d9586ff9ceda88e64679f6b87071b961'
AUDITOR = R / 'source-freezes/rbf-final-refit-priority-local-checkpoint-numeric-audit-v1-20261005'
AUDITOR_SHA = '250f88ee612c120c2618931414e89c4a68be5ac138cbfd0cca38be7b0e63caf7'
READER = R / 'source-freezes/rbf-final-refit-main-topK-independent-output-reader-v1-20261004/read_rbf_final_refit_forest_outputs.py'
READER_SHA = '53747489a446c8990cfc6d9310eb686a69b39d99bde226c8945665a974d83f61'
KEYS = {'audit':'priority-numeric-audit', 'checkpoint':'priority-checkpoint', 'weights':'priority-weights'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def source_gate():
    own = Path(__file__).resolve().parent
    for directory, expected in ((own, None), (CONSUMER, CONSUMER_SHA), (AUDITOR, AUDITOR_SHA)):
        path = directory / 'source-freeze.json'
        if expected is not None: assert sha(path) == expected
        freeze = json.loads(path.read_bytes())
        for name, entry in freeze['sources'].items():
            assert sha(directory/name) == entry['sha256']
        for reference in freeze.get('references', ()):
            assert sha(reference['path']) == reference['sha256']
    assert sha(READER) == READER_SHA


def checked_inputs(run, audit_path, audit):
    """Bind the precise completed run audited earlier; never accept arbitrary paths."""
    seed = audit['seed']
    assert type(seed) is int and seed in (1337, 2027, 3407)
    assert audit['kind'] == 'rbf_final_priority_local_full_export_and_selected_checkpoint_numeric_audit_v1'
    assert audit['source_freeze_sha256'] == AUDITOR_SHA and audit['consumer_freeze_sha256'] == CONSUMER_SHA
    assert audit['export']['events'] == 7445 and len(audit['export']['sequences']) == 46
    assert audit['export']['exact_export_features_and_targets_verified'] is True
    assert audit['export']['all_export_groups_independently_scored'] is True
    assert audit['producer_model_or_optimizer_imported'] is False
    for name in ('learned_replay_accepted', 'strict_pipeline_isolated_selection', 'paper_performance_complete'):
        assert audit[name] is False
    assert audit['selection']['atol'] == audit['selection']['rtol'] == 1e-8
    run = Path(run).absolute();fit = run / 'fit' / str(seed)
    expected = {run/'completion.json', run/'data/manifest.json', run/'fit/plan.json', run/'fit/receipt.json',
                fit/'checkpoint.json', fit/'weights.npz', fit/'epochs.jsonl'}
    assert set(audit['input_hashes']) == {str(p) for p in expected}
    for p in (*expected, Path(audit_path).absolute()):
        assert p.is_file() and not any(q.is_symlink() for q in (p, *p.parents))
        if p in expected: assert sha(p) == audit['input_hashes'][str(p)]
    return {'audit':Path(audit_path).absolute(), 'checkpoint':fit/'checkpoint.json', 'weights':fit/'weights.npz'}


def qualify(args, Task):
    audit_raw = args.numeric_audit.read_bytes()
    audit_sha = hashlib.sha256(audit_raw).hexdigest()
    audit = json.loads(audit_raw)
    paths = checked_inputs(args.run, args.numeric_audit, audit)
    assert audit['seed'] == args.seed
    ledger = json.loads((R/'receipts/20260928-execution-ledger.json').read_bytes())
    assert any(v.get('kind') == 'rbf-final-priority-local-checkpoint-numeric-audit'
               and v.get('receipt') == str(args.numeric_audit.absolute())
               and v.get('receipt_sha256') == audit_sha for v in ledger['entries'])
    bridge = module('priority_checkpoint_publication_provenance', CONSUMER/'run_final_refit_priority_after_targets.py')
    source, _ = bridge.source_gate(CONSUMER)
    gate = module('priority_checkpoint_publication_teacher_gate', source/'transvision/models/event_track_v2x/exclusive_priority_admission.py')
    run = args.run.absolute();data = run/'data'
    completion = json.loads((run/'completion.json').read_bytes())
    original = json.loads((run/'plan.json').read_bytes())
    assert all(completion[k] == v for k,v in original.items())
    assert completion['status'] == 'completed_pending_independent_checkpoint_admission'
    assert completion['bridge_sha256'] == sha(CONSUMER/'run_final_refit_priority_after_targets.py')
    assert completion['source_preparation_sha256'] == bridge.PREPARATION_SHA
    manifest = json.loads((data/'manifest.json').read_bytes());gate.verify_exported(data, manifest)
    assert sha(data/'manifest.json') == completion['training_manifest_sha256']
    target = json.loads((data/'teacher-target-admission.json').read_bytes())
    main = json.loads((data/'main-independent-admission.json').read_bytes())
    byte = json.loads((data/'main-byte-admission.json').read_bytes())
    binding = json.loads((data/'teacher-collection-binding.json').read_bytes())
    assert target['task_id'] == audit['teacher_task_id'] == completion['teacher_task_id']
    assert main['task_id'] == audit['main_task_id'] == completion['main_task_id']
    assert target['seed'] == audit['seed'] == completion['seed'] == manifest['upstream_identity_seed']
    assert sha(args.published_prerequisite) == target['published_prerequisite_sha256']
    assert sha(args.runtime_admission) == completion['runtime_admission_sha256']
    assert sorted(audit['export']['sequences']) == manifest['independent_admission_plan']['expected_sequences']
    jobs = [v for v in json.loads(args.teacher_journal.read_bytes())['jobs'] if v['task_id'] == target['task_id']]
    assert len(jobs) == 1
    lookup = lambda task_id:Task.get_task(task_id=task_id)
    bridge.live_provenance(lookup, jobs[0], main, byte, target, binding, args.published_prerequisite)
    bridge.runtime_gate(args.runtime_admission, lookup)
    checkpoint = json.loads(paths['checkpoint'].read_bytes())
    assert checkpoint['seed'] == args.seed and checkpoint['policy_signature'] == audit['policy_signature']
    assert checkpoint['weights_sha256'] == sha(paths['weights']) == completion['weights_sha256']
    assert sha(paths['checkpoint']) == completion['checkpoint_sha256']
    upstream = {}
    for role, task_id, artifacts in (('main', main['task_id'], byte['artifacts']),
                                     ('teacher', target['task_id'], target['registered_artifacts'])):
        task = lookup(task_id);task.reload();assert str(task.status) == 'completed'
        bridge.registered(task, artifacts)
        params = task.get_parameters();plan = json.loads(params['General/plan'])
        assert hashlib.sha256(canonical(plan)).hexdigest() == params['General/recipe_sha256']
        upstream[role] = dict(task_id=task_id, bootstrap_sha256=hashlib.sha256(task.data.script.diff.encode()).hexdigest(),
            recipe_sha256=params['General/recipe_sha256'], artifacts=artifacts)
    # Bind later publication to the exact inputs whose admission was checked,
    # including the numerical receipt itself; re-hashing later is not enough.
    expected = dict(audit=audit_sha, checkpoint=audit['input_hashes'][str(paths['checkpoint'])],
                    weights=audit['input_hashes'][str(paths['weights'])])
    qualified_payload = {role:dict(key=KEYS[role], sha256=sha(p), bytes=p.stat().st_size) for role,p in paths.items()}
    assert all(item['sha256'] == expected[role] for role,item in qualified_payload.items())
    context = dict(seed=args.seed, policy_signature=audit['policy_signature'], upstream_tasks=upstream,
                   qualified_payload=qualified_payload)
    return paths, context


def payload(paths, context):
    assert set(paths) == set(KEYS)
    assert set(context) == {'seed', 'policy_signature', 'upstream_tasks', 'qualified_payload'}
    assert set(context['upstream_tasks']) == {'main', 'teacher'}
    specs = {role:dict(key=KEYS[role], sha256=sha(p), bytes=Path(p).stat().st_size) for role,p in paths.items()}
    assert specs == context['qualified_payload'], 'payload changed after qualification'
    identity = hashlib.sha256(canonical(dict(context=context, payload=specs))).hexdigest()
    return specs, identity


def read_artifact(task, key, destination):
    assert sha(READER) == READER_SHA
    reader = module('priority_publication_frozen_byte_reader', READER)
    return reader.read_artifact(task, key, destination)


def verify_publication(path, paths, context, Task):
    specs, identity = payload(paths, context)
    value = json.loads(path.read_bytes())
    assert value['kind'] == KIND and value['identity'] == identity and value['seed'] == context['seed']
    assert value['independent_cloud_bytes_verified'] is True
    assert value['upstream_tasks'] == context['upstream_tasks']
    assert value['policy_signature'] == context['policy_signature']
    assert value['local_numeric_audit_sha256'] == specs['audit']['sha256']
    assert value['destination'] == DESTINATION and value['project'] == PROJECT
    assert value['learned_replay_accepted'] is value['paper_performance_complete'] is False
    task = Task.get_task(task_id=value['task_id']);task.reload()
    assert str(task.status) == 'completed' and set(task.artifacts) == set(KEYS.values())
    assert task.get_parameters()['General/publication_identity'] == identity
    assert value['artifacts'] == {role:dict(task=task.id, **item) for role,item in specs.items()}
    for role,item in specs.items():
        a = task.artifacts[item['key']]
        assert a.hash == item['sha256'] and a.size == item['bytes']
        local = path.parent/'independent-cloud-bytes'/item['key']
        assert not local.is_symlink() and local.stat().st_size == item['bytes'] and sha(local) == item['sha256']
    return value


def run(args, Task, paths, context):
    specs, identity = payload(paths, context)
    assert context['seed'] == args.seed
    if not args.execute:
        print(json.dumps(dict(seed=args.seed, destination=DESTINATION, project=PROJECT, payload=specs,
                             publication_requested=False, task_created=False)), flush=True)
        return
    name = f'RBF final-refit priority checkpoint seed{args.seed} ' + identity[:16]
    matches = Task.get_tasks(project_name=PROJECT, task_name='^'+re.escape(name)+'$')
    assert len(matches) <= 1
    receipt = args.output/'independent-publication.json'
    if matches or args.output.exists():
        assert len(matches) == 1 and receipt.is_file(), 'incomplete publication preserved; no automatic recreation'
        proof = verify_publication(receipt, paths, context, Task)
        assert proof['task_id'] == matches[0].id
        print(json.dumps(dict(task_id=proof['task_id'], receipt=str(receipt), not_repeated=True)), flush=True)
        return
    assert args.output.resolve().is_relative_to(R/'artifacts')
    assert not any(p.is_symlink() for p in (args.output, *args.output.parents))
    args.output.mkdir(parents=True, exist_ok=False)
    copies = args.output/'payload';copies.mkdir()
    for role,p in paths.items():
        with (copies/KEYS[role]).open('xb') as stream:stream.write(Path(p).read_bytes())
        assert sha(copies/KEYS[role]) == specs[role]['sha256']
    new(args.output/'publication-intent.json', dict(identity=identity, context=context, payload=specs,
        destination=DESTINATION, project=PROJECT, command=sys.argv, source_sha256=sha(__file__),
        Task_id_unknown_on_interrupted_create=True, automatic_retry=False))
    task = Task.create(project_name=PROJECT, task_name=name, task_type=Task.TaskTypes.data_processing)
    started = args.output/'task-created-before-upload.json'
    new(started, dict(task_id=task.id, identity=identity, seed=args.seed, payload=specs,
        destination=DESTINATION, project=PROJECT, command=sys.argv))
    register(started, 'rbf-final-priority-checkpoint-publication-started')
    try:
        task.output_uri = DESTINATION
        task.set_parameters(dict(publication_identity=identity, seed=args.seed, source_sha256=sha(__file__)))
        task.add_tags(['Recover-Before-Fuse', 'final-refit-priority-checkpoint', 'independently-audited-weights'])
        for role,item in specs.items():
            assert task.upload_artifact(item['key'], artifact_object=copies/item['key'], wait_on_upload=True)
        task.reload();assert set(task.artifacts) == set(KEYS.values())
        downloaded = args.output/'independent-cloud-bytes';downloaded.mkdir()
        for item in specs.values():
            assert read_artifact(task, item['key'], downloaded/item['key']) == dict(sha256=item['sha256'], bytes=item['bytes'])
        # Confirm original inputs did not change during transfer.
        assert payload(paths, context) == (specs, identity)
        task.mark_completed(force=True);task.reload();assert str(task.status) == 'completed'
        new(receipt, dict(kind=KIND, task_id=task.id, identity=identity, seed=args.seed,
            artifacts={role:dict(task=task.id, **item) for role,item in specs.items()},
            local_numeric_audit_sha256=specs['audit']['sha256'], policy_signature=context['policy_signature'],
            upstream_tasks=context['upstream_tasks'], destination=DESTINATION, project=PROJECT,
            source_freeze_sha256=sha(Path(__file__).resolve().parent/'source-freeze.json'),
            independent_cloud_bytes_verified=True, raw_source_or_database_uploaded=False,
            learned_replay_accepted=False, full_Stage2_complete=False, paper_performance_complete=False))
        verify_publication(receipt, paths, context, Task)
        register(receipt, 'rbf-final-priority-checkpoint-independent-publication')
        print(json.dumps(dict(task_id=task.id, receipt=str(receipt), learned_replay_accepted=False)), flush=True)
    except BaseException as error:
        failure = args.output/'failure.json'
        new(failure, dict(task_id=task.id, identity=identity, error_type=type(error).__name__,
                         original_task_and_partials_preserved=True, automatic_retry=False, accepted=False))
        register(failure, 'rbf-final-priority-checkpoint-publication-failure')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=(1337,2027,3407))
    for name in ('run', 'numeric-audit', 'teacher-journal', 'published-prerequisite', 'runtime-admission', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--execute', action='store_true');args = parser.parse_args();source_gate()
    if not args.numeric_audit.is_file() or not (args.run/'completion.json').is_file():
        print(json.dumps(dict(status='waiting_for_completed_priority_fit_and_numerical_audit', ETA='unknown', task_created=False)))
        return
    from clearml import Task
    paths, context = qualify(args, Task)
    if not args.execute:
        run(args, Task, paths, context);return
    os.environ['CLEARML_FILES_HOST'] = DESTINATION
    from clearml.storage.helper import _HttpDriver
    container = _HttpDriver._Container(name=DESTINATION)
    assert container._should_attach_auth_header() and 'Authorization' in container.get_headers(None)
    with (R/'receipts'/f'rbf-final-priority-checkpoint-seed{args.seed}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args, Task, paths, context)


if __name__ == '__main__':main()
