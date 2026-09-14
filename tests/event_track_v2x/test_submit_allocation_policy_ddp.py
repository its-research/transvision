"""No network or real ClearML mutations: exercise submission failure boundaries."""
import json
from pathlib import Path
import re
import sys
import tarfile
from types import SimpleNamespace

import pytest

from tools.event_track_v2x import submit_allocation_policy_ddp as tool


def ready():
    return [dict(worker=tool.HOSTS[f] + ':gpu' + cards, family=f, queue='GPU4-' + f)
            for f, cards in [('A100', '4,5,6,7'), ('5090', '0,1,2,3'), ('5090', '4,5,6,7')]]


@pytest.mark.parametrize('seeds', [(), (1337, 1337), (True,), (1,)])
def test_only_distinct_priority_seeds(seeds):
    with pytest.raises(ValueError, match='distinct'):
        tool.assignments(seeds)


def test_default_is_three_four_gpu_jobs_on_two_physical_hosts():
    jobs = tool.assignments((1337, 2027, 3407))
    assert jobs == tool.MATRIX
    assert len({tool.HOSTS[f] for _, f, _ in jobs}) == 2
    assert len(tool.capacity(ready(), jobs)) == 3
    for rows in [ready()[:2], [ready()[0], ready()[1], ready()[1]]]:
        with pytest.raises(ValueError, match='insufficient'):
            tool.capacity(rows, jobs)
    assert tool.capacity([], []) == []


@pytest.mark.parametrize('bad', ['V100', 'unknown-host', '8-cards', 'wrong-queue'])
def test_capacity_never_admits_other_hardware_or_queue(bad):
    row = ready()[0]
    if bad == 'V100': row['family'] = 'V100'
    elif bad == 'unknown-host': row['worker'] = 'unverified:gpu0,1,2,3'
    elif bad == '8-cards': row['worker'] = tool.HOSTS['A100'] + ':gpu0,1,2,3,4,5,6,7'
    else: row['queue'] = 'GPU8-A100'
    with pytest.raises(ValueError, match='known four'):
        tool.capacity([row], [])


def test_explicit_a100_fallback_can_train_all_new_priority_seeds_serially():
    jobs = tool.assignments((1337, 2027, 3407), a100_only=True)
    assert [s for s, _, _ in jobs] == [1337, 2027, 3407]
    assert all(f == 'A100' for _, f, _ in jobs)
    tool.capacity(ready()[:1], jobs, a100_only=True)
    with pytest.raises(ValueError, match='insufficient'):
        tool.capacity(ready()[:1], jobs)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    package = tmp_path / 'package'; package.mkdir()
    controller = tmp_path / 'controller.py'; controller.write_text('print("fixture only")\n')
    uploads = []
    for name in ('package', 'source.tar.gz', 'priority-groups.tar.gz'):
        path = package / name; path.write_text('fixture-' + name)
        uploads.append((name, path))
    # State-machine tests isolate the separately tested real package auditor.
    hashes = {name: tool.sha(path) for name, path in uploads}
    monkeypatch.setattr(tool, 'verify_package', lambda *args: ({}, uploads, hashes))

    class Task:
        TaskTypes = SimpleNamespace(data_processing='data_processing', training='training')
        records = {}; mutations = []; enqueue_error = None; status_on_enqueue_error = 'created'

        def __init__(self, name, task_type='training'):
            self.id = f'{len(self.records) + 1:032x}'; self.name = name; self.status = 'created'
            self.artifacts = {}; self.params = {}; self.kind = task_type
            self.data = SimpleNamespace(script=SimpleNamespace(diff=''))
            self.records[self.id] = self

        @classmethod
        def get_task(cls, task_id): return cls.records[task_id]

        @classmethod
        def get_tasks(cls, project_name, task_name):
            assert project_name == tool.PROJECT
            return [r for r in cls.records.values() if re.match(task_name, r.name)]

        @classmethod
        def create(cls, project_name, task_name, task_type, **kwargs):
            cls.mutations.append(('create', task_name))
            return cls(task_name, task_type)

        @classmethod
        def enqueue(cls, task, queue_name):
            cls.mutations.append(('enqueue', task.id, queue_name))
            if cls.enqueue_error:
                task.status = cls.status_on_enqueue_error
                raise cls.enqueue_error
            task.status = 'queued'

        def upload_artifact(self, name, artifact_object, wait_on_upload):
            self.mutations.append(('upload', name))
            self.artifacts[name] = SimpleNamespace(hash=tool.sha(artifact_object))
            return True

        def mark_completed(self, force):
            self.mutations.append(('complete', self.id)); self.status = 'completed'

        def set_script(self, **kwargs): self.data.script.diff = kwargs['diff']
        def set_packages(self, values): self.requirements = values
        def set_base_docker(self, image, docker_arguments): self.image = image
        def set_parameters(self, params): self.params = dict(params)
        def get_parameters(self): return dict(self.params)
        def add_tags(self, values): self.tags = values

    def run(seeds=(1337, 2027, 3407), *, attempt=1, available=None, a100_only=False):
        return tool.execute(Task, lambda: ready() if available is None else available,
            package, 'a' * 64, controller, tool.sha(controller),
            tool.assignments(seeds, a100_only=a100_only), attempt, a100_only=a100_only)
    return SimpleNamespace(Task=Task, run=run, package=package, controller=controller)


def test_priority_submission_keeps_completed_identity_campaign_untouched(campaign):
    c = campaign
    old = c.Task('RBF row-identity seed-1337 four-A100 unrelated'); old.status = 'completed'
    result = c.run()
    assert len(result['jobs']) == 3 and result['requested_physical_hosts'] == 2
    assert not result['training_results_audited'] and not result['concurrent_execution_guaranteed']
    assert old.status == 'completed' and old.params == {}
    training = [c.Task.records[r['task_id']] for r in result['jobs']]
    assert [r.params['gpu_family'] for r in training] == ['A100', '5090', '5090']
    assert all(r.params['required_world_size'] == 4 and r.params['global_batch_groups'] == 64 for r in training)
    assert all(r.params['epochs'] == 10 and r.params['class_scope'] == 'car' for r in training)
    assert [m[1] for m in c.Task.mutations if m[0] == 'upload'] == ['package', 'source.tar.gz', 'priority-groups.tar.gz']


def test_capacity_failure_precedes_upload_or_task_creation(campaign):
    with pytest.raises(ValueError, match='insufficient'):
        campaign.run(available=ready()[:1])
    assert not campaign.Task.mutations


@pytest.mark.parametrize('status', ['queued', 'in_progress', 'completed'])
def test_resume_never_retrains_or_reuploads_existing_tasks_even_when_all_workers_busy(campaign, status):
    first = campaign.run()
    for row in first['jobs']: campaign.Task.records[row['task_id']].status = status
    before = list(campaign.Task.mutations)
    resumed = campaign.run(available=[])
    assert [r['task_id'] for r in resumed['jobs']] == [r['task_id'] for r in first['jobs']]
    assert campaign.Task.mutations == before


@pytest.mark.parametrize('status', ['created', 'queued', 'in_progress', 'completed'])
def test_missing_receipt_does_not_authorize_duplicate_priority_seed(campaign, status):
    existing = campaign.Task(tool.task_name('a' * 64, 1337, '5090', 1)); existing.status = status
    with pytest.raises(ValueError, match='already has'):
        campaign.run()
    assert not campaign.Task.mutations


def test_failed_seed_needs_explicit_new_attempt_and_other_seeds_not_required(campaign):
    first = campaign.run((2027,))
    campaign.Task.records[first['jobs'][0]['task_id']].status = 'failed'
    before = list(campaign.Task.mutations)
    with pytest.raises(ValueError, match='new explicit attempt'):
        campaign.run((2027,))
    assert campaign.Task.mutations == before
    second = campaign.run((2027,), attempt=2, available=ready()[1:2])
    assert second['jobs'][0]['task_id'] != first['jobs'][0]['task_id']
    assert len([m for m in campaign.Task.mutations if m[0] == 'enqueue']) == 2


def test_uncertain_enqueue_does_not_retry_an_unresolved_created_task(campaign):
    campaign.Task.enqueue_error = TimeoutError('fixture uncertainty')
    with pytest.raises(TimeoutError): campaign.run((1337,))
    campaign.Task.enqueue_error = None
    with pytest.raises(ValueError, match='enqueue outcome unresolved'):
        campaign.run((1337,))
    assert len([m for m in campaign.Task.mutations if m[0] == 'enqueue']) == 1


def test_uncertain_enqueue_resumes_readonly_when_server_confirms_queued(campaign):
    campaign.Task.enqueue_error = TimeoutError('fixture uncertainty')
    campaign.Task.status_on_enqueue_error = 'queued'
    with pytest.raises(TimeoutError): campaign.run((1337,))
    campaign.Task.enqueue_error = None
    before = list(campaign.Task.mutations)
    assert campaign.run((1337,), available=[])['jobs'][0]['status'] == 'queued'
    assert campaign.Task.mutations == before


def test_changed_remote_artifact_never_overwritten(campaign):
    campaign.run((1337,))
    package_task = next(t for t in campaign.Task.records.values() if t.kind == 'data_processing')
    package_task.artifacts['source.tar.gz'].hash = 'changed'
    before = list(campaign.Task.mutations)
    with pytest.raises(ValueError, match='never overwrite'): campaign.run((1337,))
    assert campaign.Task.mutations == before


@pytest.mark.parametrize('bad', ['parameter', 'script', 'ledger'])
def test_resume_checks_ownership_and_executed_configuration(campaign, bad):
    result = campaign.run((1337,)); task = campaign.Task.records[result['jobs'][0]['task_id']]
    if bad == 'parameter': task.params['gpu_family'] = 'V100'
    elif bad == 'script': task.data.script.diff = 'another controller'
    else:
        path = campaign.package / 'clearml-priority-seed-1337-attempt-1.json'
        row = json.loads(path.read_bytes()); row['seed'] = 2027; path.write_text(json.dumps(row))
    before = list(campaign.Task.mutations)
    with pytest.raises(ValueError, match='differs'): campaign.run((1337,))
    assert campaign.Task.mutations == before


def test_default_cli_only_reads_workers_and_queues(monkeypatch, tmp_path):
    reads = []
    client = SimpleNamespace(
        workers=SimpleNamespace(get_all=lambda **kw: reads.append(('workers', kw)) or []),
        queues=SimpleNamespace(get_all=lambda: reads.append(('queues', {})) or []))
    monkeypatch.setitem(sys.modules, 'clearml', SimpleNamespace(Task=object()))
    monkeypatch.setitem(sys.modules, 'clearml.backend_api', SimpleNamespace(
        Session=SimpleNamespace(get_api_server_host=lambda: 'http://10.100.35.118:8008')))
    monkeypatch.setitem(sys.modules, 'clearml.backend_api.session.client', SimpleNamespace(APIClient=lambda: client))
    monkeypatch.setattr(tool, 'execute', lambda *a, **kw: pytest.fail('read-only CLI called execute'))
    tool.main(['--package', str(tmp_path / 'nonexistent')])
    assert reads == [('workers', {'last_seen': 120}), ('queues', {})]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('execute', [False, True])
def test_current_cli_policy_excludes_5090_and_assigns_only_four_a100(monkeypatch, tmp_path, capsys, execute):
    client = SimpleNamespace(workers=SimpleNamespace(get_all=lambda **kw: []),
                             queues=SimpleNamespace(get_all=lambda: []))
    monkeypatch.setitem(sys.modules, 'clearml', SimpleNamespace(Task=object()))
    monkeypatch.setitem(sys.modules, 'clearml.backend_api', SimpleNamespace(
        Session=SimpleNamespace(get_api_server_host=lambda: 'http://10.100.35.118:8008')))
    monkeypatch.setitem(sys.modules, 'clearml.backend_api.session.client', SimpleNamespace(APIClient=lambda: client))
    monkeypatch.setattr(tool, 'available', lambda *args: ready())
    calls = []
    def capture(Task, snapshot, package, digest, controller, controller_sha, jobs, attempt, **kwargs):
        assert kwargs['a100_only'] is True
        assert all(family == 'A100' and queue == 'GPU4-A100' for _, family, queue in jobs)
        assert snapshot() == ready()[:1]
        calls.append(jobs)
    monkeypatch.setattr(tool, 'execute', capture)
    args = ['--execute', '--package', str(tmp_path), '--package-sha256', 'a'*64,
            '--controller-sha256', 'b'*64] if execute else []
    tool.main(args)
    report = json.loads(capsys.readouterr().out)
    assert report['active_hardware_policy'] == 'A100-only'
    assert report['excluded_families'] == ['V100', '5090']
    assert report['available_four_gpu_workers'] == ready()[:1]
    assert report['selected_priority_seeds'] == [[s, 'A100', 'GPU4-A100'] for s in (1337, 2027, 3407)]
    assert len(calls) == int(execute)


def test_local_submission_lock_prevents_concurrent_package_mutation(tmp_path):
    with tool.submission_lock(tmp_path):
        with pytest.raises(ValueError, match='another local submitter'):
            with tool.submission_lock(tmp_path): pytest.fail('second lock acquired')
    with tool.submission_lock(tmp_path): pass


def test_real_package_validator_refuses_unpinned_or_partial_teacher_before_unpack(tmp_path):
    package = tmp_path / 'package'; package.mkdir()
    controller = tmp_path / 'controller.py'; controller.write_text('fixture')
    manifest = package / 'package.json'; manifest.write_text(json.dumps(dict(
        kind='component_priority_ddp_package_v1', split='train', class_scope=['car'],
        full_official_train_trace=False, teacher_frames=195, teacher_sequences=1)))
    with pytest.raises(ValueError, match='explicit package'):
        tool.verify_package(package, 'f' * 64, controller, tool.sha(controller))
    with pytest.raises(ValueError, match='complete car priority'):
        tool.verify_package(package, tool.sha(manifest), controller, tool.sha(controller))
    assert sorted(p.name for p in package.iterdir()) == ['package.json']


@pytest.fixture
def synthetic_package(tmp_path):
    """Synthetic 46-sequence HEADER/NUMERIC fixture, not a sealed real teacher.

    Deliberately bypasses the production packager's teacher receipt verification
    to test the submitter's archive/data boundary only. Never passed to ClearML.
    """
    from tools.event_track_v2x import package_allocation_policy_ddp as packager
    from tools.event_track_v2x.train_allocation_policy_ddp import audit_data, priority_ddp_sources
    from transvision.models.event_track_v2x import allocation_training as training
    from transvision.models.event_track_v2x.allocation_policy import FEATURES, RECIPE, TARGET
    data = tmp_path / 'groups'; data.mkdir()
    package = tmp_path / 'package'; package.mkdir()
    shards = []
    for i in range(46):
        path = data / f'sequence-{i:04d}.jsonl'
        path.write_text(json.dumps(dict(features=[[0.] * 18], targets=[.1])) + '\n')
        shards.append(dict(path=path.name, sequence_id=f'{i:04d}', sha256=tool.sha(path), groups=1, rows=1))
    exported = dict(kind=training.DATA_KIND, split='train', feature_recipe=RECIPE, target_recipe=TARGET,
        feature_names=list(FEATURES), shards=shards, source_sha256=training.allocation_sources(),
        full_official_train_trace=True, labels_are_model_not_true_risk=True, future_or_gt_inputs=False,
        binding={'synthetic_test_only': True}, replay_receipt_sha256='b' * 64)
    manifest = data / 'manifest.json'; manifest.write_text(json.dumps(exported))
    _, fit, held, stats = audit_data(data, tool.sha(manifest))
    sources = packager.source_inventory()
    records = [dict(path=p.name, sha256=tool.sha(p), bytes=p.stat().st_size) for p in sorted(data.iterdir())]
    packager.write_archive(tool.ROOT, sources, package / 'source.tar.gz', 16 * 1024**2)
    packager.write_archive(data, records, package / 'priority-groups.tar.gz', 8 * 1024**3)
    meta = dict(kind='component_priority_ddp_package_v1', split='train', class_scope=['car'],
        full_official_train_trace=True, teacher_frames=7445, teacher_sequences=46,
        raw_GT_included=False, teacher_traces_included=False, predictions_included=False,
        original_detection_stream_included=False, trained_models_included=False,
        strict_pipeline_isolated_selection=False, final_full_train_refit=False, paper_eligible=False,
        source_sha256=priority_ddp_sources(), source_inventory=sources, data_inventory=records,
        training_manifest_sha256=tool.sha(manifest), teacher_receipt_sha256='b' * 64,
        binding=exported['binding'], statistics=stats, fit_sequences=sorted(r['sequence_id'] for r in fit),
        holdout_sequences=sorted(r['sequence_id'] for r in held), artifacts=[dict(path=p.name,
            bytes=p.stat().st_size, sha256=tool.sha(p)) for p in sorted(package.iterdir())])
    def write_meta():
        path = package / 'package.json'; path.write_text(json.dumps(meta)); return tool.sha(path)
    return SimpleNamespace(root=package, data=data, meta=meta, write_meta=write_meta, records=records,
                           write_archive=packager.write_archive, controller=tool.ROOT / tool.CONTROLLER)


def test_real_submitter_auditor_validates_archive_members_numeric_data_and_binding(synthetic_package):
    p = synthetic_package
    metadata, uploads, hashes = tool.verify_package(p.root, p.write_meta(), p.controller, tool.sha(p.controller))
    assert metadata == p.meta and len(uploads) == 3
    assert hashes['package'] == tool.sha(p.root / 'package.json')
    assert not list(p.root.glob('clearml-*'))


@pytest.mark.parametrize('bad', ['controller', 'binding', 'source', 'stats', 'archive', 'paper', 'unreferenced-shard'])
def test_real_submitter_refuses_inconsistent_or_excess_package_before_external_access(synthetic_package, bad):
    p = synthetic_package
    if bad == 'controller':
        next(r for r in p.meta['source_inventory'] if r['path'] == tool.CONTROLLER)['sha256'] = '0' * 64
    elif bad == 'binding': p.meta['binding'] = {}
    elif bad == 'source': p.meta['source_sha256'] = {}
    elif bad == 'stats': p.meta['statistics'] = {}
    elif bad == 'archive': p.meta['artifacts'][0]['sha256'] = '0' * 64
    elif bad == 'paper': p.meta['paper_eligible'] = True
    else:
        extra = p.data / 'sequence-9999.jsonl'; extra.write_text('{"GT": "must not upload"}\n')
        p.records.append(dict(path=extra.name, bytes=extra.stat().st_size, sha256=tool.sha(extra)))
        # Rebuild only this synthetic archive in-place; not a production package.
        path = p.root / 'priority-groups.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            for r in p.records: archive.add(p.data / r['path'], arcname=r['path'], recursive=False)
        record = next(r for r in p.meta['artifacts'] if r['path'] == path.name)
        record.update(bytes=path.stat().st_size, sha256=tool.sha(path))
    with pytest.raises(ValueError):
        tool.verify_package(p.root, p.write_meta(), p.controller, tool.sha(p.controller))
    assert not list(p.root.glob('clearml-*'))
