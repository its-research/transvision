"""Audit the complete registered image/pose input before any val GPU dispatch."""
import argparse
import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import tarfile

import numpy as np

R = Path('/Volumes/Data/test/recover-before-fuse')
OUT = R/'artifacts/rbf-nested-detector-seen-val-full-input-admission-v1-20261004'
SMALL = R/'artifacts/rbf-final-refit-SPD-seen-val-matching-producer-input-recovery-v1-20261004'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


class RestrictedNumpyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name) not in {('numpy.core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype')}:
            raise ValueError('unexpected pickle global: '+module+'.'+name)
        return super().find_class(module,name)


def register(path,kind):
    ledger_path = R/'receipts/20260928-execution-ledger.json'
    with open(str(ledger_path)+'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger = json.loads(ledger_path.read_bytes())
        ledger['entries'].append(dict(kind=kind,receipt=str(path),receipt_sha256=sha(path),
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_status='active'))
        tmp = ledger_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(ledger,indent=2,ensure_ascii=False)+'\n')
        os.replace(tmp,ledger_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--byte-receipt',type=Path,required=True)
    parser.add_argument('--source-freeze',type=Path,required=True)
    parser.add_argument('--reuse-extracted-inputs', action='store_true')
    args = parser.parse_args()
    receipt = json.loads(args.byte_receipt.read_bytes())
    assert receipt['full_registered_archive_bytes_read'] is True
    archive_path = Path(receipt['archive'])
    assert archive_path.stat().st_size == receipt['bytes'] == 1796367315
    assert receipt['sha256']=='7a58ac98f96b7c0ea07429b4951744c34a6e2048bca4ad7bff6aa0952936b9d0'
    OUT.mkdir(exist_ok=args.reuse_extracted_inputs)
    target = OUT/'input-unpack'
    if not args.reuse_extracted_inputs:
        target.mkdir()
        with tarfile.open(archive_path,'r:gz') as archive:
            members = archive.getmembers()
            assert len({x.name for x in members}) == len(members)
            for item in members:
                p = Path(item.name)
                assert not p.is_absolute() and '..' not in p.parts and not item.issym() and not item.islnk()
                assert item.isdir() or item.isfile()
                assert p.parts[0] == 'inputs' or item.name in ('run_cache.py','cache_primitives.py')
            archive.extractall(target,filter='data')
    else:
        assert not (OUT/'acceptance.json').exists(), 'completed admission must not be repeated'
        original_inventory=json.loads((args.byte_receipt.parent/'archive-members.json').read_bytes())
        expected={x['path']:x['bytes'] for x in original_inventory if x['type']=='file'}
        actual={p.relative_to(target).as_posix():p.stat().st_size for p in target.rglob('*') if p.is_file()}
        assert expected==actual and not any(p.is_symlink() for p in target.rglob('*'))
    inputs = target/'inputs'
    input_path = inputs/'input-manifest.json'
    assert sha(input_path)=='fb655debe6c857d062e26d125e873097b781419e6d962bd06166f92f32095cbe'
    manifest = json.loads(input_path.read_bytes())
    assert manifest['kind']=='eventtrack_validation_image_pose_inputs_v1'
    assert manifest['gt_payloads_in_package'] is False and manifest['test_payloads_read'] is False
    assert manifest['val_payloads_read'] is True
    sequences = manifest['validation_sequences']
    assert sequences==sorted(set(sequences)) and len(sequences)==21
    assert manifest['frames']=={'vehicle-side':3748,'infrastructure-side':3441}
    records=[]
    identities=set()
    for side in ('vehicle-side','infrastructure-side'):
        directory=inputs/side
        expected=manifest[side]
        for name,key in (('image-pose-infos.pkl','infos_sha256'),('frame-index.json','frame_index_sha256'),
                         ('training-config.py','training_config_sha256')):
            assert sha(directory/name)==expected[key]
        rows=json.loads((directory/'frame-index.json').read_bytes())
        assert len(rows)==manifest['frames'][side]
        infos=RestrictedNumpyUnpickler(io.BytesIO((directory/'image-pose-infos.pkl').read_bytes())).load()['infos']
        assert len(infos)==len(rows)
        tokens=[]
        for info in infos:
            assert not any(k.startswith('gt_') or k in ('anno_tokens','next_anno_tokens','prev_anno_tokens','ann_info') for k in info)
            assert not info['next'] and not info['prev'] and not info['sweeps']
            assert info['scene_token'] in sequences
            tokens.append(str(info['token']))
        assert len(tokens)==len(set(tokens)) and set(tokens)=={r['frame_id'] for r in rows}
        info_by_token={str(x['token']):x for x in infos}
        for index,row in enumerate(rows):
            assert set(row)=={'sequence_id','frame_id','source_image_timestamp_us','box_reference_timestamp_us','image_sha256'}
            identity=(side,row['sequence_id'],row['frame_id'])
            assert identity not in identities and row['sequence_id'] in sequences
            assert info_by_token[row['frame_id']]['scene_token']==row['sequence_id']
            identities.add(identity)
            for key in ('source_image_timestamp_us','box_reference_timestamp_us'):
                assert type(row[key]) is int and row[key]>0
            image=directory/'images'/(row['frame_id']+'.jpg')
            assert image.is_file() and not image.is_symlink() and sha(image)==row['image_sha256']
            records.append(dict(side=side,frame_id=row['frame_id'],sequence_id=row['sequence_id'],image_sha256=row['image_sha256']))
            if index%500==0:
                print(json.dumps(dict(stage='independent_all_image_pose_coverage',side=side,
                    completed_frames=index+1,total_frames=len(rows),ETA='unknown until audit completes')),flush=True)
        assert len(list((directory/'images').glob('*.jpg')))==len(rows)
    assert len(identities)==7189
    split_path=inputs/'validation-split.json'
    assert split_path.is_file()
    split=json.loads(split_path.read_bytes())
    # Preserve the sealed input split as data; the inference adapter reads only val.
    assert split['batch_split']['val']==sequences
    assert all(split['batch_split'][key]==[] for key in ('train','test','test_A'))
    source=json.loads((args.source_freeze/'source-preparation.json').read_bytes())
    for name,item in source['inventory'].items():
        assert sha(args.source_freeze/name)==item['sha256']
    source_record=OUT/'full-image-pose-coverage.json'
    source_record.write_text(json.dumps(records,separators=(',',':'))+'\n')
    admission=dict(kind='rbf_nested_seen_val_complete_label_free_input_admission_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_clearml_task_id='5717bf5d1ead4fcba37f3d9101b11b20',
        archive_byte_readback_receipt=str(args.byte_receipt),archive_byte_readback_receipt_sha256=sha(args.byte_receipt),
        input_manifest_sha256=sha(input_path),validation_sequences=sequences,frames=manifest['frames'],
        image_pose_frames=7189,full_image_pose_input_coverage_verified=True,
        restricted_pickle_loading=True,GT_payloads_included=False,test_payloads_included=False,
        all_image_identity_receipt_sha256=sha(source_record),val_split_sha256=sha(split_path),
        inference_started=False,full_online_RBF_accepted=False,paper_performance_complete=False)
    path=OUT/'acceptance.json'
    path.write_text(json.dumps(admission,indent=2)+'\n')
    register(path,admission['kind'])
    collectors={1337:'bceb0a009246474dbaa176a823dbfec1',2027:'afcd45a8d7a54bf98b00888ee080e39c',3407:'c09211b2f04f438ca7edfe8e00d1272c'}
    jobs=[]
    for seed,task_id in collectors.items():
        collection_path=SMALL/task_id/'collection-receipt'
        collection=json.loads(collection_path.read_bytes())
        assert collection['seed']==seed and collection['cohort']=='nested-detector-fit'
        for side in ('vehicle-side','infrastructure-side'):
            jobs.append(dict(seed=seed,side=side,training_task_id=collection['training_task_id'],
                collection=dict(task=task_id,key='collection-receipt',sha256=sha(collection_path),bytes=collection_path.stat().st_size),
                checkpoint_sha256=collection['inventory'][side+'-final-checkpoint']['sha256']))
    prepared=dict(kind='rbf_nested_seen_val_complete_input_source_bound_GPU4_preparation_v1',
        full_input_payload_and_schema_admitted=True,input_admission_receipt=str(path),
        input_admission_receipt_sha256=sha(path),source_inventory=source['inventory'],jobs=jobs,
        inputs=dict(task='5717bf5d1ead4fcba37f3d9101b11b20',key='cache-inputs',bytes=1796367315,sha256=receipt['sha256']),
        input_manifest=dict(task='5717bf5d1ead4fcba37f3d9101b11b20',key='package-manifest',bytes=734,
            sha256='d7a7688321145ad4bc5ed5692858b2136ac2c6aa8376c200d45b8f908a3f4877'),
        runtime_manifest=dict(task='9a7e7a9213954b57a35403f777e74561',key='package-manifest',bytes=None,
            sha256='8d4c45019abae11fae1258c2d448ac744f5e316bc5e9348c62425e272484c39f'),
        libgl=dict(task='1e0730c280e846bebd92cc4de49893e3',key='libgl',bytes=1141794,
            sha256='5ac68c58a292e435a0ee55c98f0bd2720a9b088343afc813c262bdc552cc0e10'))
    # Query only registered metadata here; no credentials are stored in receipts.
    from clearml import Task
    runtime=Task.get_task(task_id=prepared['runtime_manifest']['task'])
    artifact=runtime.artifacts['package-manifest']
    assert str(runtime.status)=='completed' and artifact.hash==prepared['runtime_manifest']['sha256']
    prepared['runtime_manifest']['bytes']=artifact.size
    preparation=args.source_freeze/'GPU-preparation.json'
    with preparation.open('x') as stream:
        json.dump(prepared,stream,indent=2)
        stream.write('\n')
    register(preparation,prepared['kind'])
    print(json.dumps(dict(input_admitted=True,preparation=str(preparation),ready_GPU_side_jobs=len(jobs))),flush=True)


if __name__=='__main__':
    main()
