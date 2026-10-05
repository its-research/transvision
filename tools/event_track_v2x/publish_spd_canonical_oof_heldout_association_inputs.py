#!/usr/bin/env python3
"""Publish admitted GT-free car association input bytes; independent archive readback."""
import argparse,hashlib,json,re,tarfile,sys
from pathlib import Path
from clearml import Task


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def publish(fold):
    R=Path('/Volumes/Data/test/recover-before-fuse');data=R/f'artifacts/spd-canonical-oof-fold{fold}-full-heldout-car-association-input-v1-20261001';proof=data/'independent-full-input-acceptance.json';v=json.loads(proof.read_text());assert v['all_source_frames_and_vehicle_reference_examples_covered'] and v['all_feature_bytes_encoder_values_gate_distances_and_deadlines_verified'];assert sha(data/'manifest.json')==v['manifest_sha256'];B=R/f'artifacts/spd-canonical-oof-fold{fold}-heldout-car-association-cloud-input-v1-20261001';assert not B.exists();B.mkdir();inventory=[]
    for p in sorted(data.rglob('*')):
        if p.is_file():assert not p.is_symlink();inventory.append(dict(path=str(p.relative_to(data)),sha256=sha(p),bytes=p.stat().st_size))
    assert {r['path'] for r in inventory}=={'manifest.json','examples.json','source-events.json','independent-full-input-acceptance.json'}|{r['features']['path'] for r in json.loads((data/'examples.json').read_text())}
    archive=B/'input-archive.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for r in inventory:t.add(data/r['path'],arcname=r['path'],recursive=False)
    manifest=B/'input-manifest.json';manifest.write_text(json.dumps(dict(fold_id=fold,input_manifest_sha256=sha(data/'manifest.json'),input_independent_acceptance_sha256=sha(proof),source_frames=v['source_frames'],counts=v['counts'],inventory=inventory,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,class_scope='car',cache_role='held_out_association_inference_only',held_out_GT_read=False,paper_eligible=False),indent=2)+'\n');name=f'SPD canonical OOF fold-{fold} full held-out car association input bytes v1 '+sha(manifest)[:12];assert not Task.get_tasks(project_name='Thesis/EventTrack-V2X/Training',task_name='^'+re.escape(name)+'$');t=Task.create(project_name='Thesis/EventTrack-V2X/Training',task_name=name,task_type=Task.TaskTypes.data_processing);t.output_uri=True;t.set_parameters(dict(fold_id=fold,input_manifest_sha256=sha(manifest),role='held_out_association_inference_only',paper_eligible=False));(B/'publication-task.json').write_text(json.dumps(dict(task_id=t.id,source_sha256=sha(Path(__file__)),manifest_sha256=sha(manifest)),indent=2)+'\n');t.mark_started(force=True)
    for n,p in [('input-archive',archive),('input-manifest',manifest)]:assert t.upload_artifact(n,artifact_object=p,wait_on_upload=True)
    t.mark_completed(force=True);t.reload();assert t.status=='completed'
    sys.path.insert(0,str(R/'source-freezes/spd-canonical-association-compatible-GPU-source-publication-v2-20261001'));from spd_registered_artifact_chunked_candidate import download_registered_artifact
    O=B/'independent-cloud-readback';O.mkdir();registered={}
    for n,p in [('input-archive',archive),('input-manifest',manifest)]:
        q=O/p.name;download_registered_artifact(t.artifacts[n],sha(p),p.stat().st_size,q);registered[n]=dict(sha256=sha(q),bytes=q.stat().st_size)
    expected={r['path']:r for r in json.loads((O/'input-manifest.json').read_text())['inventory']};seen=set()
    with tarfile.open(O/'input-archive.tar.gz','r:gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in expected and member.name not in seen;f=tar.extractfile(member);d=hashlib.sha256();size=0
            for block in iter(lambda:f.read(1024**2),b''):d.update(block);size+=len(block)
            assert size==expected[member.name]['bytes'] and d.hexdigest()==expected[member.name]['sha256'];seen.add(member.name)
    assert seen==set(expected);accept=dict(fold_id=fold,task_id=t.id,asset_manifest_sha256=sha(manifest),registered_artifacts=registered,all_registered_bytes_and_archive_members_verified=True,input_independent_acceptance_sha256=sha(proof),held_out_GT_read=False,paper_eligible=False);(O/'acceptance-receipt.json').write_text(json.dumps(accept,indent=2)+'\n');print(json.dumps(dict(fold_id=fold,task_id=t.id,all_cloud_bytes_accepted=True,acceptance_sha256=sha(O/'acceptance-receipt.json'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);a=p.parse_args();publish(a.fold)
