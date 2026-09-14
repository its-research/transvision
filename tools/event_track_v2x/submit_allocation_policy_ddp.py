#!/usr/bin/env python3
"""Read-only resource report; --execute submits independently seeded priority heads.

Current user policy: A100 ONLY, four GPUs per independently seeded job.
The old mixed-hardware matrix remains a library/reproduction capability, not
the current CLI policy. Queue snapshots are not reservations or readiness.
No identity-model retraining, worker/queue changes, V100, GT or prediction upload.
Seeds queue serially if only one four-A100 group is idle.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools.event_track_v2x.run_clearml_allocation_policy_ddp import check_package, extract_checked
from tools.event_track_v2x.submit_forest_identity_ddp import HOSTS, IMAGE, MATRIX, PROJECT, available, sha, slots

PREFIX = 'RBF priority-ddp '
CONTROLLER = 'tools/event_track_v2x/run_clearml_allocation_policy_ddp.py'
LIVE_OR_COMPLETE = {'queued', 'in_progress', 'completed'}


def ordinary(path):
    path = Path(path).absolute()
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('ordinary nonsymlink file required')
    return path


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def submission_lock(package):
    # Serializes this package on this filesystem, NOT submitters on other hosts.
    path = package / 'clearml-priority-submission.lock'
    if path.is_symlink():
        raise ValueError('submission lock cannot be a symlink')
    with path.open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('another local submitter owns this package') from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def assignments(seeds, *, a100_only=False):
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s not in (1337, 2027, 3407) for s in seeds)):
        raise ValueError('distinct predeclared priority seeds required')
    return tuple((s, 'A100', 'GPU4-A100') if a100_only else (s, f, q)
                 for s, f, q in MATRIX if s in seeds)


def capacity(ready, pending, *, a100_only=False):
    """Conservative disjoint snapshot; never count duplicate/overlapping agents."""
    unique = {}; selected = []
    for row in sorted(ready, key=lambda r: r['worker']):
        host, cards = slots(row['worker'])
        if (row['family'] not in HOSTS or host != HOSTS[row['family']]
                or len(cards) != 4 or row['queue'] != 'GPU4-' + row['family']):
            raise ValueError('only known four-A100/5090 queue assignments allowed')
        if any(host == h and cards & c for h, c in unique.values()):
            continue
        unique[row['worker']] = host, cards
        selected.append(row)
    need = Counter(f for _, f, _ in pending)
    if a100_only and need:
        if set(need) != {'A100'}:
            raise ValueError('A100 fallback cannot contain another family')
        need['A100'] = 1
    actual = Counter(r['family'] for r in selected)
    if any(actual[f] < n for f, n in need.items()):
        raise ValueError('insufficient disjoint idle four-GPU workers for pending seeds')
    return selected


def verify_package(package, expected_sha, controller, expected_controller_sha):
    """Revalidate the packager's pinned result before any external mutation."""
    package = Path(package).absolute()
    manifest = ordinary(package / 'package.json')
    controller = ordinary(controller)
    if sha(manifest) != expected_sha or sha(controller) != expected_controller_sha:
        raise ValueError('explicit package or controller SHA differs')
    meta = json.loads(manifest.read_bytes()); check_package(meta)
    if any(meta.get(k) is not False for k in
           ('strict_pipeline_isolated_selection', 'final_full_train_refit', 'paper_eligible')):
        raise ValueError('development priority head cannot claim final paper evidence')
    sources = {r['path']: r['sha256'] for r in meta['source_inventory']}
    if sources.get(CONTROLLER) != expected_controller_sha:
        raise ValueError('submitted controller differs from packaged controller')
    from tools.event_track_v2x.train_allocation_policy_ddp import audit_data, priority_ddp_sources
    if meta['source_sha256'] != priority_ddp_sources():
        raise ValueError('current priority training dependencies differ')
    uploads = [('package', manifest)]
    with tempfile.TemporaryDirectory(prefix='rbf-priority-submit-check-') as temp:
        for record in meta['artifacts']:
            path = ordinary(package / record['path'])
            if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
                raise ValueError('package archive changed')
            code = record['path'] == 'source.tar.gz'
            inventory = meta['source_inventory' if code else 'data_inventory']
            if code and any(not r['path'].endswith('.py') for r in inventory):
                raise ValueError('only source Python files permitted in code archive')
            if not code and any(re.fullmatch(r'manifest\.json|sequence-\d{4}\.jsonl', r['path']) is None
                                for r in inventory):
                raise ValueError('only numerical priority shards and manifest permitted')
            extract_checked(path, Path(temp) / ('source' if code else 'groups'), inventory,
                            16 * 1024**2 if code else 8 * 1024**3)
            uploads.append((record['path'], path))
        exported, fit, held, stats = audit_data(Path(temp) / 'groups', meta['training_manifest_sha256'])
        data_hashes = {r['path']: r['sha256'] for r in meta['data_inventory']}
        expected_data = {'manifest.json': meta['training_manifest_sha256'],
                         **{r['path']: r['sha256'] for r in exported['shards']}}
        if (exported['binding'] != meta['binding'] or stats != meta['statistics']
                or data_hashes != expected_data
                or exported['replay_receipt_sha256'] != meta['teacher_receipt_sha256']
                or sorted(r['sequence_id'] for r in fit) != meta['fit_sequences']
                or sorted(r['sequence_id'] for r in held) != meta['holdout_sequences']
                or any(sources.get(p) != h for p, h in meta['source_sha256'].items())):
            raise ValueError('derived priority groups differ from package provenance')
    hashes = {name: sha(path) for name, path in uploads}
    if hashes['package'] != expected_sha or any(hashes[r['path']] != r['sha256'] for r in meta['artifacts']):
        raise ValueError('package changed during submission validation')
    return meta, uploads, hashes


def task_name(package_sha, seed, family, attempt):
    return f'{PREFIX}{package_sha} seed-{seed} attempt-{attempt} four-{family}'


def cohort(Task, package_sha, seed):
    return Task.get_tasks(project_name=PROJECT, task_name='^' + re.escape(PREFIX + package_sha)
                          + f' seed-{seed}' + r' attempt-\d+ four-(?:A100|5090)$')


def saved_task(Task, path, expected, name):
    if not path.exists():
        return None
    old = json.loads(ordinary(path).read_bytes())
    if set(old) != set(expected) | {'task_id'} or any(old[k] != v for k, v in expected.items()):
        raise ValueError('saved priority submission identity differs')
    task = Task.get_task(task_id=old['task_id'])
    if task.name != name:
        raise ValueError('saved task no longer belongs to this priority campaign')
    return task


def reject_existing(Task, digest, seed, own=None):
    # A created task also blocks automatic duplication: it may be an interrupted
    # submission. Failed/stopped attempts require a new explicit attempt number.
    if any(t.id != own and str(t.status) not in ('failed', 'stopped') for t in cohort(Task, digest, seed)):
        raise ValueError('priority seed already has an unresolved, active or completed task')


def parameters(package_task_id, digest, controller_sha, seed, family, a100_only):
    return dict(package_task_id=package_task_id, package_sha256=digest, controller_sha256=controller_sha,
        seed=seed, gpu_family=family, required_world_size=4, global_batch_groups=64, epochs=10,
        class_scope='car', paper_eligible=False, a100_only=a100_only, V100_excluded=True,
        hardware_mixed_not_same_compute_benchmark=True, concurrent_execution_guaranteed=False,
        authorization='user-requested-ClearML-four-GPU-multi-machine-training-20260913')


def verify_running(task, expected, controller):
    actual = task.get_parameters(); normalized = {}
    for key, value in actual.items():
        key = key.removeprefix('General/')
        if key in normalized:
            raise ValueError('ambiguous existing task parameters')
        normalized[key] = value
    if (any(str(normalized.get(k)) != str(v) for k, v in expected.items())
            or task.data.script.diff != controller):
        raise ValueError('existing active/completed priority task configuration differs')


def execute(Task, snapshot, package, digest, controller_path, controller_sha, jobs, attempt, *, a100_only=False):
    if type(attempt) is not int or attempt < 1:
        raise ValueError('positive explicit attempt required')
    if tuple(jobs) != assignments(tuple(s for s, _, _ in jobs), a100_only=a100_only):
        raise ValueError('submission jobs differ from declared priority matrix')
    package = Path(package).absolute()
    _, uploads, hashes = verify_package(package, digest, controller_path, controller_sha)
    controller = ordinary(controller_path).read_text()
    if sha(controller_path) != controller_sha:
        raise ValueError('controller changed during submission validation')
    with submission_lock(package):
        planned = []
        for seed, family, queue in jobs:
            path = package / f'clearml-priority-seed-{seed}-attempt-{attempt}.json'
            expected = dict(package_sha256=digest, controller_sha256=controller_sha,
                            seed=seed, gpu_family=family, queue=queue, attempt=attempt, a100_only=a100_only)
            name = task_name(digest, seed, family, attempt)
            task = saved_task(Task, path, expected, name)
            reject_existing(Task, digest, seed, task.id if task else None)
            if task is None and Task.get_tasks(project_name=PROJECT, task_name='^' + re.escape(name) + '$'):
                raise ValueError('exact task exists without local receipt; inspect before retry')
            if task is not None and str(task.status) not in LIVE_OR_COMPLETE | {'created'}:
                raise ValueError('failed/stopped task requires a new explicit attempt')
            planned.append((seed, family, queue, path, expected, name, task))
        pending = [(s, f, q) for s, f, q, _, _, _, t in planned if t is None or str(t.status) == 'created']
        capacity(snapshot(), pending, a100_only=a100_only)
        pkg_name = 'RBF priority-derived package ' + digest
        pkg_path = package / 'clearml-priority-package.json'
        pkg = saved_task(Task, pkg_path, dict(package_sha256=digest), pkg_name)
        if pkg is None:
            if Task.get_tasks(project_name=PROJECT, task_name='^' + re.escape(pkg_name) + '$'):
                raise ValueError('package task exists without local receipt; inspect before retry')
            if not pending:
                raise ValueError('saved seed tasks have no owned package receipt')
            pkg = Task.create(project_name=PROJECT, task_name=pkg_name, task_type=Task.TaskTypes.data_processing)
            save(pkg_path, dict(task_id=pkg.id, package_sha256=digest))
        if str(pkg.status) not in ('created', 'completed'):
            raise ValueError('package task has unexpected status; inspect before retry')
        for name, path in uploads:
            current = Task.get_task(task_id=pkg.id)
            existing = current.artifacts.get(name)
            if sha(path) != hashes[name]:
                raise ValueError('local package changed before upload')
            if existing is not None:
                if existing.hash != hashes[name]:
                    raise ValueError('remote package artifact differs; never overwrite')
                continue
            if str(pkg.status) == 'completed':
                raise ValueError('completed package omitted an artifact')
            if not pkg.upload_artifact(name, artifact_object=path, wait_on_upload=True):
                raise RuntimeError('priority package upload outcome requires inspection')
            if Task.get_task(task_id=pkg.id).artifacts[name].hash != hashes[name] or sha(path) != hashes[name]:
                raise RuntimeError('priority package upload readback differs')
        if str(pkg.status) == 'created':
            pkg.mark_completed(force=True)
        capacity(snapshot(), pending, a100_only=a100_only)
        results = []
        for seed, family, queue, path, expected, name, task in planned:
            params = parameters(pkg.id, digest, controller_sha, seed, family, a100_only)
            if task is not None and str(task.status) in LIVE_OR_COMPLETE:
                verify_running(task, params, controller)
            else:
                reject_existing(Task, digest, seed, task.id if task else None)
                if task is None:
                    task = Task.create(project_name=PROJECT, task_name=name,
                                       task_type=Task.TaskTypes.training, binary='python3.12')
                    save(path, dict(expected, task_id=task.id))
                intent = path.with_name(path.stem + '-enqueue-intent.json')
                if intent.exists():
                    raise ValueError('previous enqueue outcome unresolved; inspect task/queue before any retry')
                task.set_script(repository='', branch='', commit='', working_dir='.',
                                entry_point=Path(CONTROLLER).name, diff=controller)
                task.set_packages(['clearml==2.1.2', 'numpy==1.26.4', 'scipy==1.14.1', 'cryptography==46.0.5'])
                task.set_base_docker(IMAGE, docker_arguments='-e CLEARML_AGENT_FORCE_TASK_INIT=0 --shm-size 16g '
                    '--env NCCL_P2P_DISABLE=1 --env NVIDIA_DRIVER_CAPABILITIES=compute '
                    '--env CLEARML_FILES_HOST=http://10.100.35.118:8081')
                task.set_parameters(params)
                task.add_tags(['Recover-Before-Fuse', 'priority-head', 'car-only', 'train-only',
                               'four-GPU-DDP', 'development-not-OOF', 'seed-' + str(seed)])
                save(intent, dict(task_id=task.id, queue=queue, package_sha256=digest))
                Task.enqueue(task, queue_name=queue)
                task = Task.get_task(task_id=task.id)
                if str(task.status) not in LIVE_OR_COMPLETE:
                    raise RuntimeError('enqueue not confirmed; inspect task and queue before retry')
            result = dict(seed=seed, task_id=task.id, queue=queue, status=str(task.status))
            results.append(result)
            print('RBF_PRIORITY_SUBMISSION ' + json.dumps(result, sort_keys=True), flush=True)
        result = dict(jobs=results, gpus_per_job=4, requested_physical_hosts=len({HOSTS[f] for _, f, _ in jobs}),
            a100_only=a100_only, concurrent_execution_guaranteed=False, training_results_audited=False,
            identity_model_retraining=False, paper_eligible=False)
        print('RBF_PRIORITY_CAMPAIGN ' + json.dumps(result, sort_keys=True), flush=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--package', type=Path)
    parser.add_argument('--package-sha256')
    parser.add_argument('--controller', type=Path, default=ROOT / CONTROLLER)
    parser.add_argument('--controller-sha256')
    parser.add_argument('--seeds', nargs='+', type=int, choices=(1337, 2027, 3407), default=(1337, 2027, 3407))
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--a100-only', action='store_true', default=True,
                        help='Compatibility flag: the current CLI is always A100-only by user request.')
    args = parser.parse_args(argv)
    jobs = assignments(args.seeds, a100_only=args.a100_only)
    if args.attempt < 1 or args.execute and (args.package is None or any(
            re.fullmatch(r'[0-9a-f]{64}', h or '') is None for h in (args.package_sha256, args.controller_sha256))):
        raise ValueError('execute requires an explicit package, two SHA pins and positive attempt')
    os.environ['CLEARML_FILES_HOST'] = 'http://10.100.35.118:8081'
    from clearml import Task
    from clearml.backend_api import Session
    from clearml.backend_api.session.client import APIClient
    if urlparse(Session.get_api_server_host()).hostname != '10.100.35.118':
        raise ValueError('only user-designated private ClearML server allowed')
    client = APIClient()
    def snapshot():
        workers = [w.to_dict() for w in client.workers.get_all(last_seen=120)]
        queues = {q.id: q.to_dict() for q in client.queues.get_all()}
        return [row for row in available(workers, queues) if row['family'] == 'A100']
    ready = snapshot()
    print(json.dumps(dict(available_four_gpu_workers=ready, selected_priority_seeds=jobs,
        excluded_families=['V100', '5090'], active_hardware_policy='A100-only',
        snapshot_is_network_readiness=False, snapshot_is_reservation=False,
        training_submitted_by_this_read=False), sort_keys=True), flush=True)
    if not args.execute:
        return
    return execute(Task, snapshot, args.package, args.package_sha256, args.controller,
                   args.controller_sha256, jobs, args.attempt, a100_only=args.a100_only)


if __name__ == '__main__':
    main()
