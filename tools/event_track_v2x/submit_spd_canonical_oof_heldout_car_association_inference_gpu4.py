#!/usr/bin/env python3
"""Per-fold byte-admitted held-out inference on any collision-free idle GPU4."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,importlib.util,re
from clearml import Task
from clearml.backend_api.session.client import APIClient

R=Path('/Volumes/Data/test/recover-before-fuse');S=R/'source-freezes/spd-canonical-oof-full-heldout-car-association-GPU-inference-v1-20261001';source=S/'infer_spd_canonical_oof_car_association_gpu4.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    for r in json.loads((S/'source-inventory.json').read_text()):assert sha(S/r['path'])==r['sha256']
    spec=importlib.util.spec_from_file_location('slot_inspector',R/'source-freezes/spd-canonical-oof-association-car-only-GPU-v3-dispatch-20261001/submit_spd_canonical_oof_association_gpu4.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);api=APIClient();journal=R/'receipts/spd-canonical-oof-full-heldout-car-association-GPU-inference-v1-dispatch-20261001.json';jobs=json.loads(journal.read_text())['jobs'] if journal.exists() else [];accepted=json.loads((R/'receipts/spd-canonical-oof-car-association-fourfold-GPU-forward-acceptance-index-20261001.json').read_text())['jobs']+json.loads((R/'receipts/spd-canonical-oof-car-association-fold4-GPU-forward-acceptance-index-20261001.json').read_text())['jobs'];byfold={r['fold_id']:r for r in accepted}
    def save():
        tmp=journal.with_suffix('.json.tmp');tmp.write_text(json.dumps(dict(source_sha256=sha(source),checked_at_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,paper_eligible=False),indent=2)+'\n');tmp.replace(journal)
    for fold in range(5):
        B=R/f'artifacts/spd-canonical-oof-fold{fold}-heldout-car-association-cloud-input-v1-20261001';proof=B/'independent-cloud-readback/acceptance-receipt.json'
        if not proof.exists():print(json.dumps(dict(fold_id=fold,status='waiting_for_existing_publisher_cloud_readback',ETA='unknown')),flush=True);continue
        p=json.loads(proof.read_text());assert p['all_registered_bytes_and_archive_members_verified'] and p['fold_id']==fold;manifest=B/'input-manifest.json';assert sha(manifest)==p['asset_manifest_sha256'];owner=Task.get_task(task_id=p['task_id']);assert owner.status=='completed'
        for n,r in p['registered_artifacts'].items():assert owner.artifacts[n].hash==r['sha256'] and owner.artifacts[n].size==r['bytes']
        head=byfold[fold];assert head['registered_source_and_full_receipt_bytes_verified'] and head['all_rank_tensor_forward_checks_passed'];training=Task.get_task(task_id=head['training_task_id']);verifier=Task.get_task(task_id=head['task_id']);assert training.status==verifier.status=='completed' and training.artifacts['association-checkpoint'].hash==head['checkpoint_sha256'] and verifier.artifacts['tensor-forward-acceptance'].hash==head['artifact_sha256'];name=f'SPD canonical OOF fold-{fold} full held-out car association GPU inference v1 '+sha(source)[:12]+' '+head['checkpoint_sha256'][:12]+' '+sha(manifest)[:12];matches=Task.get_tasks(project_name=module.PROJECT,task_name='^'+re.escape(name)+'$');assert len(matches)<=1
        if matches:
            t=matches[0];assert t.get_parameters()['General/checkpoint_sha256']==head['checkpoint_sha256']
            if t.status!='created':continue
        else:
            choices,_=module.slots(api,{r['task_id'] for r in jobs})
            if not choices:print(json.dumps(dict(fold_id=fold,status='waiting_for_idle_collision_free_GPU4',ETA='unknown')),flush=True);continue
            t=Task.create(project_name=module.PROJECT,task_name=name,task_type=Task.TaskTypes.inference,binary='python3.12');t.output_uri='http://10.100.34.118:8081';t.set_script(repository='',branch='',commit='',working_dir='.',entry_point=source.name,diff=source.read_text());t.set_packages(['clearml==2.1.2']);t.set_base_docker(module.IMAGE,docker_arguments='-e CLEARML_AGENT_FORCE_TASK_INIT=0 --shm-size 16g --env NCCL_P2P_DISABLE=1 --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env CLEARML_FILES_HOST=http://10.100.34.118:8081 --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1');t.set_parameters(dict(fold_id=fold,seed=1337,input_task_id=owner.id,input_manifest_sha256=sha(manifest),input_cloud_readback_sha256=sha(proof),training_task_id=training.id,checkpoint_sha256=head['checkpoint_sha256'],checkpoint_acceptance_task_id=verifier.id,checkpoint_acceptance_artifact_sha256=head['artifact_sha256'],source_sha256=sha(source),class_scope='car',input_role='held_out_inference_only',held_out_GT_read=False,paper_eligible=False));t.add_tags(['Recover-Before-Fuse','canonical-OOF','car-only','held-out-GT-free','inference-not-paper-qualified']);t.reload();assert hashlib.sha256(t.data.script.diff.encode()).hexdigest()==sha(source);jobs.append(dict(task_id=t.id,fold_id=fold,training_task_id=training.id,checkpoint_sha256=head['checkpoint_sha256'],input_task_id=owner.id,input_manifest_sha256=sha(manifest),input_cloud_readback_sha256=sha(proof),status='created',ETA='unknown_until_progress'));save()
        choices,snapshot=module.slots(api,{r['task_id'] for r in jobs})
        if not choices:continue
        q=choices[0];fresh,_=module.slots(api,{r['task_id'] for r in jobs})
        if not any(x['queue_id']==q['queue_id'] for x in fresh):continue
        t.set_parameter('General/queue_name',q['queue_name']);Task.enqueue(t,queue_id=q['queue_id']);t.reload();r=next(r for r in jobs if r['task_id']==t.id);r.update(queue=q['queue_name'],status=str(t.status),worker_snapshot=snapshot);save();print(json.dumps({k:r[k] for k in ['task_id','fold_id','queue','status']}),flush=True)


if __name__=='__main__':main()
