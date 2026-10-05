"""Publish fully admitted target-free val inputs; independently read cloud bytes."""
import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import time

from rbf_nested_seen_val_v2_common import R, new, register, sha

ROWS = R/'artifacts/rbf-matching-seen-val-target-free-original-row-contexts-v1-20261004'
CONSUMER = R/'source-freezes/rbf-matching-seen-val-joint-identity-GPU-input-consumer-v1-20261004'
DEST = R/'artifacts/rbf-matching-seen-val-admitted-row-cloud-publication-v1-20261004'
PROJECT = 'Thesis/Recover-Before-Fuse/Inference'


def source_gate():
    p = json.loads((CONSUMER/'preparation.json').read_bytes())
    for name, record in p['sources'].items():
        assert sha(CONSUMER/name) == record['sha256'] and (CONSUMER/name).stat().st_size == record['bytes']
    assert sha(CONSUMER/'software-gate.json') == p['software_gate_sha256']
    return p


def archive_rows(path, directory, manifest):
    with path.open('xb') as target, gzip.GzipFile(fileobj=target, mode='wb', mtime=0, filename='') as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for record in manifest['shards']:
                for name, digest in ((record['path'],record['sha256']),(record['events_path'],record['events_sha256'])):
                    relative = Path(name)
                    assert not relative.is_absolute() and '..' not in relative.parts
                    item = directory/relative
                    assert item.resolve().is_relative_to(directory.resolve()) and not item.is_symlink() and sha(item) == digest
                    info = archive.gettarinfo(str(item), arcname=name)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    info.mode = 0o644
                    with item.open('rb') as stream:
                        archive.addfile(info, stream)


def readback(task, key, specification, path):
    import requests
    from clearml.backend_api.session import Session
    artifact = task.artifacts[key]
    assert artifact.hash == specification['sha256'] and artifact.size == specification['bytes']
    url = artifact.url.replace('10.100.35.118:8081','10.100.34.118:8081')
    completed, started = 0, time.monotonic()
    with requests.get(url, stream=True, timeout=(10,120), headers={'Authorization':'Bearer '+Session().token}) as response:
        assert response.status_code == 200
        with path.open('xb') as stream:
            for block in response.iter_content(1024**2):
                stream.write(block);completed += len(block)
                if completed % (32*1024**2) < len(block):
                    print(json.dumps(dict(stage='admitted_seen_val_input_independent_cloud_byte_readback',key=key,
                        completed_bytes=completed,total_bytes=specification['bytes'],
                        ETA_seconds=(time.monotonic()-started)*(specification['bytes']-completed)/completed,
                        ETA_scope='current cloud input byte readback only')),flush=True)
    assert completed == specification['bytes'] and sha(path) == specification['sha256']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True,choices=(1337,2027,3407))
    parser.add_argument('--resume-transport-failure',action='store_true')
    args = parser.parse_args()
    # The SDK only attaches the locally configured session token to known file
    # server hosts. Bind this process to the verified reachable file service;
    # no credential or persistent configuration is changed or serialized.
    os.environ['CLEARML_FILES_HOST'] = 'http://10.100.34.118:8081'
    from clearml import Task
    from clearml.storage.helper import _HttpDriver
    container = _HttpDriver._Container(name='http://10.100.34.118:8081')
    assert container._should_attach_auth_header() and 'Authorization' in container.get_headers(None)
    prepared = source_gate()
    directory = ROWS/f'seed{args.seed}'
    admission_path = directory/'independent-acceptance.json'
    admission = json.loads(admission_path.read_bytes())
    manifest_path = directory/'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    assert admission['kind'] == prepared['required_row_independent_admission_kind']
    assert admission['full_original_seen_val_features_and_contexts_independently_accepted'] is True
    assert admission['seed'] == manifest['seed'] == args.seed and admission['row_manifest_sha256'] == sha(manifest_path)
    assert admission['counts']['rows'] == manifest['rows'] and admission['counts']['events'] == manifest['original_events'] == 3316
    assert manifest['no_GT_or_dummy_supervision_fields'] is True and manifest['candidate_protocol'] == 'rbf-all-class-top64-v1'
    semantic = hashlib.sha256(json.dumps(dict(seed=args.seed,row_manifest_sha256=sha(manifest_path),
        independent_admission_sha256=sha(admission_path),consumer_sha256=sha(CONSUMER/'run_rbf_seen_val_identity_gpu.py')),
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    name = f'RBF fully admitted target-free matching seen-val inputs seed{args.seed} '+semantic[:16]
    matches = Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
    assert len(matches) <= 1
    output = DEST/f'seed{args.seed}'
    if (output/'independent-publication-acceptance.json').exists():
        proof = json.loads((output/'independent-publication-acceptance.json').read_bytes())
        assert proof['semantic_publication_identity'] == semantic and proof['cloud_input_bytes_independently_read'] is True
        assert len(matches) == 1 and matches[0].id == proof['task_id'] and str(matches[0].status) == 'completed'
        print(json.dumps(dict(seed=args.seed,existing_publication=proof['task_id'],duplicate_not_created=True)),flush=True)
        return
    recovery = bool(matches or output.exists())
    if recovery:
        assert args.resume_transport_failure and len(matches) == 1 and output.is_dir()
        assert (output/'publication-failure.json').exists() and not (output/'transport-recovery-v2.json').exists()
        original = json.loads((output/'task-created-before-upload.json').read_bytes())
        assert original['task_id'] == matches[0].id and original['semantic_publication_identity'] == semantic
        assert str(matches[0].status) == 'created' and not matches[0].artifacts
        assert json.loads((output/'publication-failure.json').read_bytes())['exception_type'] == 'ValueError'
        task = matches[0]
    else:
        assert not args.resume_transport_failure
        output.mkdir(parents=True,exist_ok=False)
    files = {'rows':output/'rows.tar.gz','row-manifest':manifest_path,'row-independent-admission':admission_path,
        'gpu-producer':CONSUMER/'run_rbf_seen_val_identity_gpu.py','gpu-source-preparation':CONSUMER/'preparation.json',
        'gpu-schema-software-gate':CONSUMER/'software-gate.json'}
    if recovery:
        assert set(original['files']) == set(files)
        for key,path in files.items():
            assert path.stat().st_size == original['files'][key]['bytes'] and sha(path) == original['files'][key]['sha256']
    else:
        archive_rows(files['rows'],directory,manifest)
        task = Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.data_processing)
    task.output_uri = 'http://10.100.34.118:8081'
    if not recovery:
        task.set_parameters(dict(seed=args.seed,semantic_publication_identity=semantic,
        source_sha256=sha(__file__),row_manifest_sha256=sha(manifest_path),
        full_input_independent_admission=True,GT_read=False,paper_performance_complete=False))
    task.add_tags(['Recover-Before-Fuse','seen-val-exploratory','all-class-top64','target-free-rows','admitted-inputs'])
    task_receipt = output/('transport-recovery-v2.json' if recovery else 'task-created-before-upload.json')
    new(task_receipt,dict(seed=args.seed,task_id=task.id,semantic_publication_identity=semantic,
        files={key:dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size) for key,path in files.items()},
        source_sha256=sha(__file__),upload_command=['python',str(Path(__file__).resolve()),'--seed',str(args.seed)],
        resumed_same_publication_task=recovery,SDK_file_service_auth_host_bound=True,
        credentials_read_only_from_local_SDK_session=True,credentials_written_or_logged=False,
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    register(task_receipt,'rbf_matching_seen_val_admitted_input_publication_task_created_v1')
    try:
        for key,path in files.items():
            assert task.upload_artifact(key,artifact_object=path,wait_on_upload=True)
        task.reload()
        specs = {key:dict(task=task.id,key=key,sha256=sha(path),bytes=path.stat().st_size) for key,path in files.items()}
        for key,spec in specs.items():
            assert task.artifacts[key].hash == spec['sha256'] and task.artifacts[key].size == spec['bytes']
        task.mark_completed(force=True);task.reload();assert str(task.status) == 'completed'
        read_directory = output/'independent-cloud-byte-readback'
        read_directory.mkdir()
        for key,spec in specs.items():
            readback(task,key,spec,read_directory/key)
        expected = {record[key]:record[digest] for record in manifest['shards']
            for key,digest in (('path','sha256'),('events_path','events_sha256'))}
        with tarfile.open(read_directory/'rows','r:gz') as archive:
            members = archive.getmembers()
            assert len(members) == len(expected) == 42 and {m.name for m in members} == set(expected)
            for member in members:
                assert member.isfile() and hashlib.sha256(archive.extractfile(member).read()).hexdigest() == expected[member.name]
        proof = output/'independent-publication-acceptance.json'
        new(proof,dict(kind='rbf_matching_seen_val_admitted_input_full_cloud_byte_readback_v1',seed=args.seed,
            task_id=task.id,semantic_publication_identity=semantic,artifacts=specs,
            cloud_input_bytes_independently_read=True,all_42_row_and_event_archive_members_match=True,
            row_manifest_sha256=sha(manifest_path),row_independent_admission_sha256=sha(admission_path),
            GPU_consumer_source_sha256=sha(files['gpu-producer']),source_sha256=sha(__file__),
            ready_for_generic_GPU_inference=True,GPU_inference_complete=False,paper_performance_complete=False))
        register(proof,'rbf_matching_seen_val_admitted_input_full_cloud_byte_readback_v1')
        print(json.dumps(dict(seed=args.seed,task_id=task.id,input_publication_independently_accepted=True,
            rows=manifest['rows'],proof=str(proof))),flush=True)
    except BaseException as error:
        failure = output/('publication-recovery-v2-failure.json' if recovery else 'publication-failure.json')
        new(failure,dict(seed=args.seed,task_id=task.id,exception_type=type(error).__name__,
            partials_preserved=True,no_automatic_retry=True,input_publication_accepted=False))
        register(failure,'rbf_matching_seen_val_admitted_input_publication_failure_preserved_v1')
        raise


if __name__ == '__main__':
    main()
