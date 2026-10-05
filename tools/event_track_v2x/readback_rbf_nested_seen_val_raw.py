"""Independent cloud bytes and complete raw prediction payload/coverage audit.

This CPU reader never imports Torch or the producer model. It is deliberately
not complete detector numerical acceptance or a full online RBF evaluation.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tarfile
import time

import requests
from clearml import Task
from clearml.backend_api.session import Session

R=Path('/Volumes/Data/test/recover-before-fuse')
S=R/'source-freezes/rbf-nested-detector-seen-val-raw-GPU4-v2-20261004'
OUT=R/'artifacts/rbf-nested-detector-seen-val-independent-raw-readback-v1-20261004'
INPUTS=R/'artifacts/rbf-nested-detector-seen-val-full-input-admission-v1-20261004/input-unpack/inputs'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def new(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False)
        stream.write('\n')


def register(path,kind,task_id):
    ledger_path=R/'receipts/20260928-execution-ledger.json'
    with open(str(ledger_path)+'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads(ledger_path.read_bytes())
        if any(e.get('receipt')==str(path) for e in ledger['entries']):
            return
        ledger['entries'].append(dict(kind=kind,receipt=str(path),receipt_sha256=sha(path),
            task_id=task_id,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_status='active'))
        temporary=ledger_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(ledger,indent=2,ensure_ascii=False)+'\n')
        os.replace(temporary,ledger_path)


def download(artifact,path,key):
    digest=hashlib.sha256()
    count=0
    started,last=time.monotonic(),0
    partial=path.with_suffix(path.suffix+'.partial')
    url=artifact.url.replace('10.100.35.118:8081','10.100.34.118:8081')
    with requests.get(url,headers={'Authorization':'Bearer '+Session().token},stream=True,timeout=(10,120)) as response:
        if response.status_code!=200:
            raise RuntimeError('artifact HTTP status '+str(response.status_code))
        with partial.open('xb') as stream:
            for block in response.iter_content(1024**2):
                stream.write(block)
                digest.update(block)
                count+=len(block)
                if count>artifact.size:
                    raise ValueError('registered byte count exceeded')
                if time.monotonic()-last>20:
                    print(json.dumps(dict(stage='independent_raw_cloud_byte_readback',artifact=key,
                        completed_bytes=count,total_bytes=artifact.size,
                        ETA_seconds=(time.monotonic()-started)*(artifact.size-count)/count,
                        ETA_scope='current artifact byte transfer only')),flush=True)
                    last=time.monotonic()
    if count!=artifact.size or digest.hexdigest()!=artifact.hash:
        raise ValueError('complete artifact hash/size differs')
    partial.rename(path)
    return dict(path=str(path),bytes=count,sha256=digest.hexdigest())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task-id',required=True)
    args=parser.parse_args()
    journal=json.loads((R/'receipts/rbf-nested-detector-matching-seen-val-raw-GPU-dispatch-20261004.json').read_bytes())
    job=next(j for j in journal['jobs'] if j['task_id']==args.task_id)
    output=OUT/('seed'+str(job['seed']))/job['side']
    if (output/'independent-acceptance.json').exists():
        print(json.dumps(dict(already_accepted=True,task_id=args.task_id,not_repeated=True)),flush=True)
        return
    output.mkdir(parents=True,exist_ok=False)
    stage='completed_task_and_source_binding'
    try:
        task=Task.get_task(task_id=args.task_id)
        assert str(task.status)=='completed'
        assert hashlib.sha256(task.data.script.diff.encode()).hexdigest()==job['bootstrap_sha256']
        assert task.get_parameters()['General/semantic_export_identity']==job['semantic_export_identity']
        keys=['runtime-admission','receipt']+[f'shard-{i}-{suffix}' for i in range(4) for suffix in ('manifest','raw-cache','log')]
        assert set(task.artifacts)==set(keys)
        artifacts={}
        for key in keys:
            artifacts[key]=download(task.artifacts[key],output/key,key)
        receipt=json.loads((output/'receipt').read_bytes())
        runtime=json.loads((output/'runtime-admission').read_bytes())
        assert receipt['recipe']==job['recipe'] and receipt['seed']==job['seed'] and receipt['side']==job['side']
        assert receipt['whole_side_coverage_verified'] and receipt['no_GT_in_producer'] and receipt['actual_CUDA_forward_completed']
        assert len(runtime['devices'])==4 and runtime['TF32_enabled'] is False and runtime['matmul_admitted'] is True
        assert sha(S/'spd_cache_primitives.py')=='311021ce12ce7ce1e1e8db8d031776c81a05215fcc91573d5639ad9672914926'
        module_spec=importlib.util.spec_from_file_location('frozen_raw_CPU_verifier',S/'spd_cache_primitives.py')
        verifier=importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(verifier)
        assert sha(INPUTS/'input-manifest.json')==job['recipe']['input_dataset_sha256']
        cohort=json.loads((INPUTS/'input-manifest.json').read_bytes())
        stage='complete_independent_raw_CPU_payload_coverage_audit'
        outcomes=[]
        sequences=[]
        for index in range(4):
            directory=output/('shard-'+str(index)+'-unpack')
            directory.mkdir()
            with tarfile.open(output/f'shard-{index}-raw-cache','r:gz') as archive:
                members=archive.getmembers()
                assert len({m.name for m in members})==len(members)
                assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                    and not m.issym() and not m.islnk() and (m.isfile() or m.isdir()) for m in members)
                archive.extractall(directory,filter='data')
            manifest_path=directory/'raw-cache-manifest.json'
            assert sha(manifest_path)==artifacts[f'shard-{index}-manifest']['sha256']
            manifest=json.loads(manifest_path.read_bytes())
            expected=dict(side=job['side'],shard_index=index,shard_count=4,
                checkpoint_sha256=job['recipe']['checkpoint_sha256'],input_manifest_sha256=job['recipe']['input_dataset_sha256'],
                producer_code_sha256=job['recipe']['source_inventory']['run_seen_val_cache.py']['sha256'],
                TF32_enabled=False,gt_inputs=False,test_payloads_read=False,val_payloads_read=True,
                optimizer_created=False,weights_unchanged=True,legacy_placeholder_count=0,
                detector_decode_source='raw-head-all-queries-no-roi-no-nms-v1',
                export_candidate_policy='all-queries; raw-score>=0.05/all-class-top64 downstream',
                preselection_roi=False,preselection_nms=False,preselection_topk=False)
            assert all(manifest.get(k)==v for k,v in expected.items())
            assert manifest['sequences']==cohort['validation_sequences'][index::4]
            assert manifest['classes']==['car','bicycle','pedestrian']
            assert manifest['raw_head_frames_verified']==manifest['frame_count']
            assert all(x['raw_query_count']==x['detections']==900 for x in manifest['frames'])
            assert sha(directory/'resolved-cache-config.py')==manifest['resolved_config_sha256']
            launch=json.loads((directory/'launch-receipt.json').read_bytes())
            assert all(manifest[k]==v for k,v in launch.items() if k not in ('kind','frames'))
            assert launch['frames']==manifest['frame_count']
            checked=verifier.verify_cache(directory,INPUTS)
            assert checked['all_payloads_read'] and checked['expected_frame_coverage_verified']
            assert checked['frames_verified']==manifest['frame_count']
            sequences.extend(manifest['sequences'])
            outcomes.append(dict(shard_index=index,manifest_sha256=sha(manifest_path),raw_root=str(directory),
                frames=manifest['frame_count'],detections=manifest['detection_count'],independent_CPU_full_payload_audit=checked))
            print(json.dumps(dict(stage=stage,completed_shards=index+1,total_shards=4,
                                 frames=manifest['frame_count'],ETA='unknown for remaining CPU audit')),flush=True)
        frames=sum(x['frames'] for x in outcomes)
        assert frames==cohort['frames'][job['side']]
        assert sorted(sequences)==cohort['validation_sequences']
        task.reload()
        assert str(task.status)=='completed'
        assert all(task.artifacts[k].hash==a['sha256'] and task.artifacts[k].size==a['bytes'] for k,a in artifacts.items())
        accepted=dict(kind='rbf_nested_seen_val_single_side_full_independent_raw_byte_payload_coverage_acceptance_v1',
            task_id=task.id,seed=job['seed'],side=job['side'],actual_worker=task.data.last_worker,
            artifacts=artifacts,runtime=runtime,outcomes=outcomes,frames=frames,query_count_per_frame=900,
            original_input_manifest_sha256=job['recipe']['input_dataset_sha256'],
            full_cloud_bytes_independently_read=True,full_raw_payload_and_frame_coverage_verified=True,
            GT_free_all_class_raw_queries_retained=True,TF32_enabled=False,
            independent_verifier_sha256=sha(S/'spd_cache_primitives.py'),reader_source_sha256=sha(__file__),
            full_detector_numerical_acceptance=False,full_online_RBF_accepted=False,paper_performance_complete=False)
        new(output/'independent-acceptance.json',accepted)
        register(output/'independent-acceptance.json',accepted['kind'],task.id)
        print(json.dumps(dict(task_id=task.id,raw_bytes_payload_coverage_accepted=True,frames=frames,
            receipt=str(output/'independent-acceptance.json'),paper_performance_complete=False)),flush=True)
    except BaseException as error:
        # Do not persist exception messages or raw agent configuration/URLs.
        failure=dict(stage=stage,task_id=args.task_id,exception_type=type(error).__name__,
            source_sha256=sha(__file__),independent_acceptance=False,partials_preserved=True)
        new(output/'execution-failure.json',failure)
        register(output/'execution-failure.json','rbf_seen_val_independent_raw_readback_failure_preserved',args.task_id)
        print(json.dumps(failure),flush=True)
        raise


if __name__=='__main__':
    main()
