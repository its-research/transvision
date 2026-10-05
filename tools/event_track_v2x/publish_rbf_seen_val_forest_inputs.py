"""Create-once publication of admitted, complete SPD seen-val forest inputs.

Default execution is read-only. A completed full train forest and seven exact
local assets are required before any upload; partial publications are retained.
"""
import argparse
import ast
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile

from rbf_nested_seen_val_v2_common import R, new, register, sha
from rbf_final_refit_teacher_binding import canonical, prerequisites

PROJECT = 'Thesis/Recover-Before-Fuse/Inference'
DESTINATION = 'http://10.100.34.118:8081'
KIND = 'rbf_seen_val_complete_forest_inputs_cloud_bytes_v1'
PRODUCER = R/'source-freezes/rbf-seen-val-bound-forest-GPU-producer-v1-20261005'
PRODUCER_FREEZE_SHA = 'e3f42ad26a96620220def0c10593ba122295f04c8e3e5a57fbd04387dc1866ae'
BOOTSTRAP_SHA = 'd9cacf90832ca7e8ba6f25a3ce15d89bac970db857acccea8c4eb0345a60d98f'
REVIEW = R/'receipts/rbf-seen-val-three-seed-forest-input-transport-review-20261005.json'
REVIEW_SHA = '2e140147735794d7126adcdb67e6e15efa6d6cd3ee53a5154c3b0328c9b58719'
READER = R/'source-freezes/rbf-final-refit-Top1-independent-output-reader-v1-20261004/read_rbf_final_refit_top1_outputs.py'
READER_SHA = 'd8a9542fddc0f661c723d3f8661dbc317d106d7c34f0b66c11d64c7d65a672e7'
ROLES = {'cache_archive','cache_manifest','events','binding','transport','review','main_acceptance'}
SCOPE = 'SPD seen-val exploratory scheduled snapshots; bound allocator baseline'


def source_gate():
    own = Path(__file__).resolve().parent
    freeze = json.loads((own/'source-freeze.json').read_bytes())
    for name, record in freeze['sources'].items():
        assert sha(own/name) == record['sha256']
    for record in freeze['references']:
        assert sha(record['path']) == record['sha256']
    assert sha(PRODUCER/'source-freeze.json') == PRODUCER_FREEZE_SHA
    producer = json.loads((PRODUCER/'source-freeze.json').read_bytes())
    for name, checksum in producer['sources'].items():
        assert sha(PRODUCER/name) == checksum
    assert sha(PRODUCER/'bootstrap.py') == BOOTSTRAP_SHA
    assert sha(REVIEW) == REVIEW_SHA and sha(READER) == READER_SHA
    return own


def local_inputs(seed, acceptance, gate, main):
    """Bind previously reviewed assets, including every original archive byte."""
    assert seed in (1337,2027,3407) and gate['seed'] == main['seed'] == seed
    assert gate['main_prerequisite_verified'] is True
    assert sha(acceptance) == gate['main_acceptance_sha256']
    assert main['task_id'] == gate['main_task_id']
    assert sha(REVIEW) == REVIEW_SHA
    review = json.loads(REVIEW.read_bytes())
    assert review['all_three_local_forest_input_packages_bound'] is True
    chosen = [v for v in review['seeds'] if v['seed'] == seed]
    assert len(chosen) == 1
    item = chosen[0]
    assert item['full_schedule_transport_linkage_review_passed'] is True
    root = R/f'artifacts/rbf-seen-val-forest-input-bridge-v1-20261005/seed{seed}'
    transport = R/f'artifacts/rbf-seen-val-forest-cache-transport-v1-20261005/seed{seed}'
    cache = R/f'artifacts/rbf-nested-seen-val-matching-V2-full-admission-v1-20261004/seed{seed}/cache'
    paths = dict(cache_archive=transport/'cache.tar', cache_manifest=cache/'manifest.json',
        events=root/'events.json', binding=root/'input-binding.json',
        transport=transport/'local-transport-readback.json', review=REVIEW, main_acceptance=acceptance)
    records = {}
    for role, path in paths.items():
        assert path.resolve().is_relative_to(R) and not any(p.is_symlink() for p in (path,*path.parents))
        records[role] = dict(path=str(path),key='seen-val-'+role.replace('_','-'),sha256=sha(path),bytes=path.stat().st_size)
    entries = json.loads((R/'receipts/20260928-execution-ledger.json').read_bytes())['entries']
    for role in ('binding','transport','review','main_acceptance'):
        assert any(e.get('receipt') == str(paths[role]) and e.get('receipt_sha256') == records[role]['sha256'] for e in entries)
    assert {x['path']:x['sha256'] for x in item['receipts']} == {
        str(paths[k]):records[k]['sha256'] for k in ('binding','transport')}
    b, t = (json.loads(paths[k].read_bytes()) for k in ('binding','transport'))
    assert b['seed'] == t['seed'] == seed and b['rows'] == item['rows']
    assert b['full_original_schedule_to_forest_event_binding_passed'] is True
    assert b['events'] == 3316 and b['sequences'] == 21 and b['cache_frames'] == 7189
    assert b['checkpoint']['sha256'] == main['plan']['checkpoint']['sha256']
    assert b['checkpoint']['model_sha256'] == gate['final_model_sha256'] == main['plan']['final_refit_model_sha256']
    assert t['all_member_bytes_independently_read'] is True and t['members'] == 14379
    assert t['input_binding_sha256'] == records['binding']['sha256']
    assert t['events_sha256'] == b['events_sha256'] == records['events']['sha256']
    assert t['cache_manifest_sha256'] == records['cache_manifest']['sha256']
    assert t['archive'] == item['archive']
    assert t['archive'] == {k:records['cache_archive'][k] for k in ('path','bytes','sha256')}
    return records, b


def manifest(seed, records, gate, main, Task):
    assert set(records) == ROLES and seed == gate['seed'] == main['seed']
    assert gate['main_prerequisite_verified'] is True
    assert records['main_acceptance']['sha256'] == gate['main_acceptance_sha256']
    task = Task.get_task(task_id=main['task_id'])
    assert str(task.status) == 'completed'
    assert json.loads(task.get_parameters()['General/plan']) == main['plan']
    assert task.get_parameters()['General/recipe_sha256'] == main['recipe_sha256']
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == main['plan']['bootstrap_sha256']
    expected = {'receipt','exclusive-source-manifest'} | {f'replay-rank{i}' for i in range(main['plan']['world_size'])}
    assert set(task.artifacts) == expected
    upstream = dict(task_id=task.id, artifacts={k:dict(sha256=a.hash,bytes=a.size) for k,a in task.artifacts.items()})
    return dict(kind=KIND,seed=seed,artifacts={k:{f:v[f] for f in ('key','sha256','bytes')} for k,v in records.items()},
        full_train_main_local_gate=gate,upstream_main=upstream,destination=DESTINATION,project=PROJECT)


def read_artifact(task, key, destination):
    assert sha(READER) == READER_SHA
    spec = importlib.util.spec_from_file_location('seen_val_exact_cloud_reader',READER)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.read_artifact(task,key,destination)


def publication(path, expected, Task):
    value = json.loads(path.read_bytes())
    identity = hashlib.sha256(canonical(expected)).hexdigest()
    assert value['manifest'] == expected and value['identity'] == identity
    assert value['independent_cloud_bytes_verified'] is True
    assert value['kind'] == KIND and value['seed'] == expected['seed']
    task = Task.get_task(task_id=value['task_id'])
    assert str(task.status) == 'completed'
    assert task.get_parameters()['General/publication_identity'] == identity
    assert set(task.artifacts) == {v['key'] for v in expected['artifacts'].values()}
    inputs = {}
    for role, spec in expected['artifacts'].items():
        assert task.artifacts[spec['key']].hash == spec['sha256'] and task.artifacts[spec['key']].size == spec['bytes']
        local = path.parent/'independent-cloud-bytes'/spec['key']
        assert not local.is_symlink() and local.stat().st_size == spec['bytes'] and sha(local) == spec['sha256']
        inputs[role] = dict(spec,task=task.id)
    assert value['artifacts'] == inputs
    return dict(kind=KIND,seed=expected['seed'],independent_cloud_bytes_verified=True,artifacts=inputs,
        full_train_main_local_gate=expected['full_train_main_local_gate'],upstream_main=expected['upstream_main'])


def expected_plan(main, binding, published, world_size):
    assert world_size in (4,8) and type(world_size) is int
    assert main['seed'] == binding['seed'] == published['seed']
    base = main['plan']
    assert base['method'] == 'rbf' and base['configuration']['allocation'] == 'bound'
    assert binding['checkpoint']['sha256'] == base['checkpoint']['sha256']
    inputs = published['artifacts']
    # Train input admissions stay explicitly scoped to the inherited lineage.
    inherited = {k:v for k,v in base.items() if k.endswith('admission_sha256') or k.endswith('completion_sha256')}
    plan = {k:v for k,v in base.items() if k not in inherited and k != 'dispatcher_sha256'}
    plan.update(world_size=world_size,bootstrap_sha256=BOOTSTRAP_SHA,
        parent_bootstrap_sha256=base['bootstrap_sha256'],
        forward_outputs=binding['forward_artifacts'],seen_val_input_publication=published,
        inherited_train_admissions=inherited,evaluation_scope=SCOPE,scope=SCOPE,
        full_forest_independently_accepted=False,paper_performance_complete=False)
    for role in ('cache_archive','cache_manifest','events'):plan[role] = inputs[role]
    return plan


def remote_gate(plan, publication_path, Task):
    """Execute the unchanged producer admission against live Task metadata."""
    source = (PRODUCER/'bootstrap.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == BOOTSTRAP_SHA
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == 'seen_val_admission')
    def fetch(spec, destination):
        local = publication_path.parent/'independent-cloud-bytes'/spec['key']
        raw = local.read_bytes()
        assert len(raw) == spec['bytes'] and hashlib.sha256(raw).hexdigest() == spec['sha256']
        destination.write_bytes(raw)
    namespace = dict(Task=Task,json=json,hashlib=hashlib,canonical=canonical,fetch=fetch,
        INPUT_INDEX_SHA=REVIEW_SHA,MAIN_FREEZE_SHA='f36f644f5f18fad3e5ab489edbc949e93bcab4efce22749dcb90211c8c8b27f7',
        PARENT_SHA='e363579c19831420b9782eb7aef4d96e50f3ab303239942ed78923ee481929dd',write=new)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PRODUCER/'bootstrap.py'),'exec'),namespace)
    with tempfile.TemporaryDirectory(prefix='rbf-seen-val-admission-') as directory:
        namespace['seen_val_admission'](plan,Path(directory))


def run(args, Task, records, expected):
    assert set(records) == ROLES and args.seed == expected['seed']
    identity = hashlib.sha256(canonical(expected)).hexdigest()
    if not args.execute:
        print(json.dumps(dict(seed=args.seed,manifest=expected,identity=identity,uploaded=False,GPU_task_created=False)),flush=True)
        return
    name = f'RBF complete seen-val forest inputs seed{args.seed} '+identity[:16]
    matches = Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
    assert len(matches) <= 1
    if matches or args.output.exists():
        accepted = args.output/'independent-publication.json'
        assert len(matches) == 1 and accepted.is_file(), 'preserve partial publication; no automatic retry'
        result = publication(accepted,expected,Task)
        assert result['artifacts']['binding']['task'] == matches[0].id
        print(json.dumps(dict(task_id=matches[0].id,not_repeated=True)),flush=True);return
    assert args.output.resolve().is_relative_to(R/'artifacts')
    assert not any(p.is_symlink() for p in (args.output,*args.output.parents))
    args.output.mkdir(parents=True,exist_ok=False)
    task = Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.data_processing)
    started = args.output/'task-created-before-upload.json'
    new(started,dict(manifest=expected,task_id=task.id,identity=identity,source_sha256=sha(__file__),
        commands=['python',str(Path(__file__).resolve()),'--seed',str(args.seed),'--main-admission',str(args.main_admission),
            '--main-byte-admission',str(args.main_byte_admission),'--output',str(args.output),'--execute'],GPU_task_created=False))
    register(started,'rbf-seen-val-forest-publication-started')
    try:
        task.output_uri = DESTINATION
        task.set_parameters(dict(publication_identity=identity,seed=args.seed,source_sha256=sha(__file__),
            main_task_id=expected['upstream_main']['task_id'],scope=SCOPE,paper_performance_complete=False))
        task.add_tags(['Recover-Before-Fuse','seen-val-complete-cache','scheduled-snapshot','exploratory-only'])
        dest = args.output/'independent-cloud-bytes';dest.mkdir()
        inputs = {}
        for role in sorted(ROLES):
            spec = records[role];local = Path(spec['path'])
            assert sha(local) == spec['sha256'] and local.stat().st_size == spec['bytes']
            assert {k:spec[k] for k in ('key','sha256','bytes')} == expected['artifacts'][role]
            assert task.upload_artifact(spec['key'],artifact_object=local,wait_on_upload=True)
            task.reload()
            assert task.artifacts[spec['key']].hash == spec['sha256'] and task.artifacts[spec['key']].size == spec['bytes']
            assert read_artifact(task,spec['key'],dest/spec['key']) == {k:spec[k] for k in ('sha256','bytes')}
            inputs[role] = dict(expected['artifacts'][role],task=task.id)
        assert set(task.artifacts) == {v['key'] for v in inputs.values()}
        task.mark_completed(force=True);task.reload();assert str(task.status) == 'completed'
        receipt = args.output/'independent-publication.json'
        new(receipt,dict(kind=KIND,seed=args.seed,task_id=task.id,manifest=expected,identity=identity,artifacts=inputs,
            independent_cloud_bytes_verified=True,source_freeze_sha256=sha(Path(__file__).resolve().parent/'source-freeze.json'),
            GPU_task_created=False,full_forest_independently_accepted=False,paper_performance_complete=False))
        register(receipt,KIND);publication(receipt,expected,Task)
        print(json.dumps(dict(task_id=task.id,receipt=str(receipt),GPU_task_created=False)),flush=True)
    except BaseException as error:
        failed = args.output/'failure.json'
        new(failed,dict(task_id=task.id,identity=identity,exception_type=type(error).__name__,
            original_task_and_partials_preserved=True,automatic_retry=False,publication_accepted=False))
        register(failed,'rbf-seen-val-forest-publication-failure');raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(1337,2027,3407),required=True)
    for key in ('main-admission','main-byte-admission','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args = parser.parse_args()
    source_gate()
    if not args.main_admission.is_file() or not args.main_byte_admission.is_file():
        print(json.dumps(dict(seed=args.seed,status='waiting_for_full_independent_final_main_admission',ETA='unknown',task_created=False)),flush=True);return
    gate, job, _, _ = prerequisites(args.main_admission,args.main_byte_admission,args.seed)
    records, _ = local_inputs(args.seed,args.main_admission,gate,job)
    from clearml import Task
    expected = manifest(args.seed,records,gate,job,Task)
    if args.execute:
        os.environ['CLEARML_FILES_HOST'] = DESTINATION
        from clearml.storage.helper import _HttpDriver
        container = _HttpDriver._Container(name=DESTINATION)
        assert container._should_attach_auth_header() and 'Authorization' in container.get_headers(None)
        with (R/'receipts'/f'rbf-seen-val-forest-publication-seed{args.seed}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args,Task,records,expected)
    else:run(args,Task,records,expected)


if __name__ == '__main__':main()
