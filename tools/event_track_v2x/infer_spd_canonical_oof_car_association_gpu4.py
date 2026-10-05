#!/usr/bin/env python3
"""Full vehicle-reference held-out logits and gate-aware retained joint alternatives.

All source events retained, no GT/optimizer/tracker evaluation or paper claim.
"""
import ast,hashlib,json,os,platform,socket,sys,tarfile,time
from pathlib import Path,PurePosixPath
from datetime import timedelta
from urllib.parse import urlparse,urlunparse
import numpy as np
import scipy
from scipy.stats import chi2
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from clearml import Task
from clearml.storage.helper import StorageHelper

MODEL_SOURCE=''
HYPOTHESES_SOURCE=''


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch(owner,name,expected,path):
    a=owner.artifacts[name];assert a.hash==expected;u=urlparse(a.url);assert u.scheme=='http' and u.netloc in ['10.100.34.118:8081','10.100.35.118:8081'] and not u.username and not u.query and not u.fragment
    url=urlunparse(u._replace(netloc='10.100.34.118:8081'));blocks=StorageHelper.get(url).download_as_stream(url,chunk_size=1024**2);assert blocks is not None;received=0;start=time.monotonic()
    with path.open('xb') as f:
        for block in blocks:
            f.write(block);received+=len(block);assert received<=a.size
            if received%(16*1024**2)<len(block):print(json.dumps(dict(stage='canonical_association_input_transfer',artifact=name,completed_bytes=received,total_bytes=a.size,eta_seconds=(a.size-received)*(time.monotonic()-start)/received)),flush=True)
    assert received==a.size and sha(path)==expected;return path


def worker(rank,port,root,fold):
    root=Path(root);data=root/'data';output=root/'predictions';torch.cuda.set_device(rank);torch.set_num_threads(1);assert os.environ['NCCL_P2P_DISABLE']=='1' and torch.__version__=='2.6.0+cu124' and platform.python_version()=='3.12.3' and np.__version__=='1.26.4' and scipy.__version__=='1.14.1'
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=4,timeout=timedelta(seconds=120))
    asset=json.loads((root/'input-manifest.json').read_text());manifest=json.loads((data/'manifest.json').read_text());proof=json.loads((data/'independent-full-input-acceptance.json').read_text());assert asset['fold_id']==manifest['fold_id']==proof['fold_id']==fold and sha(data/'manifest.json')==proof['manifest_sha256']==asset['input_manifest_sha256'] and sha(data/'independent-full-input-acceptance.json')==asset['input_independent_acceptance_sha256'];assert proof['all_feature_bytes_encoder_values_gate_distances_and_deadlines_verified'] and proof['all_source_frames_and_vehicle_reference_examples_covered']
    checkpoint=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=False);assert checkpoint['fold_id']==fold and checkpoint['epoch']==24 and checkpoint['seed']==1337 and checkpoint['feature_dimension']==203 and checkpoint['configuration']['class_scope']=='car' and checkpoint['configuration']['class_index']==0 and checkpoint['configuration']['fit_sequence_ids']==manifest['excluded_fit_sequence_ids'];assert set(manifest['held_out_sequence_ids']).isdisjoint(checkpoint['configuration']['fit_sequence_ids'])
    tree=ast.parse(MODEL_SOURCE);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PredictedAssociation');env={'torch':torch,'nn':nn};exec(compile(ast.Module(body=[cls],type_ignores=[]),'frozen_model','exec'),env);model=env['PredictedAssociation']();model.load_state_dict(checkpoint['model'],strict=True);model.requires_grad_(False).eval().cuda(rank);assert not any(p.requires_grad for p in model.parameters());assert all(torch.isfinite(p).all() for p in model.parameters());henv={};exec(compile(HYPOTHESES_SOURCE,'frozen_gate_aware_hypotheses','exec'),henv);hypotheses=henv['retained_hypotheses'];rows=json.loads((data/'examples.json').read_text());assert sha(data/'examples.json')==manifest['example_index_sha256'];selected=rows[rank::4];log=output/f'rank-{rank}-hypotheses.jsonl';records=[];start=time.monotonic()
    with log.open('x') as f:
        for i,row in enumerate(selected):
            p=data/row['features']['path'];assert sha(p)==row['features']['sha256'];assert row['sequence_id'] in manifest['held_out_sequence_ids']
            with np.load(p,allow_pickle=False) as z:x={k:z[k] for k in z.files}
            assert set(x)=={'left','right','left_classes','right_classes','left_query_indices','right_query_indices','geometry_gate','innovation_distance_squared'} and x['left'].shape[1:]==x['right'].shape[1:]==(203,) and (x['left_classes']==0).all() and (x['right_classes']==0).all()
            with torch.inference_mode():values=[v[0].cpu().numpy() for v in model(torch.from_numpy(x['left'][None]).cuda(rank),torch.from_numpy(x['right'][None]).cuda(rank))]
            pair,ld,rd=values;assert all(np.isfinite(v).all() for v in values);gates={str(confidence):x['innovation_distance_squared']<=chi2.ppf(confidence,3) for confidence in [.90,.95,.99]};assert np.array_equal(gates['0.99'],x['geometry_gate']);alternatives={str(confidence):{str(h):hypotheses(pair,ld,rd,gate,top_h=h) for h in [1,3,5]} for confidence,gate in gates.items()}
            name=row['sequence_id']+'-'+row['vehicle_frame_id']+'.npz';target=output/'logits'/name;np.savez_compressed(target,pair_logits=pair,left_dustbin_logits=ld,right_dustbin_logits=rd,left_query_indices=x['left_query_indices'],right_query_indices=x['right_query_indices'],left_classes=x['left_classes'],right_classes=x['right_classes'],innovation_distance_squared=x['innovation_distance_squared'])
            identity={k:row[k] for k in ['sequence_id','vehicle_frame_id','infrastructure_frame_id','decision_time_us','origin_us','available']};record=dict(**identity,input_features_sha256=row['features']['sha256'],logits=dict(path=str(target.relative_to(output)),sha256=sha(target),bytes=target.stat().st_size),gate_and_TopH_retained_hypotheses=alternatives);f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');records.append(dict(**identity,input_features_sha256=row['features']['sha256'],logits=record['logits']))
            if (i+1)%50==0 or i+1==len(selected):print(json.dumps(dict(kind='rbf_experiment_progress_v1',stage='canonical_heldout_car_association_inference',fold_id=fold,rank=rank,completed=i+1,total=len(selected),eta_seconds=(time.monotonic()-start)/(i+1)*(len(selected)-i-1))),flush=True)
    prop=torch.cuda.get_device_properties(rank);receipt=dict(rank=rank,host=socket.gethostname(),GPU_name=prop.name,GPU_uuid=str(prop.uuid),capability=list(torch.cuda.get_device_capability(rank)),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),rows=records,hypotheses_file=dict(path=log.name,sha256=sha(log),bytes=log.stat().st_size),optimizer_created=False,held_out_GT_read=False);(output/f'rank-{rank}-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');dist.barrier();dist.destroy_process_group()


def main():
    task=Task.init(project_name='Thesis/EventTrack-V2X/Training',task_name='canonical car association held-out inference',reuse_last_task_id=True,auto_connect_frameworks=False,auto_connect_arg_parser=False);task.output_uri='http://10.100.34.118:8081';params=task.get_parameters();get=lambda k:params['General/'+k];fold=int(get('fold_id'));root=Path('canonical-car-association-inference-'+task.id);root.mkdir();owner=Task.get_task(task_id=get('input_task_id'));assert owner.status=='completed';manifest=fetch(owner,'input-manifest',get('input_manifest_sha256'),root/'input-manifest.json');asset=json.loads(manifest.read_text());assert asset['fold_id']==fold and asset['class_scope']=='car' and asset['cache_role']=='held_out_association_inference_only';archive=fetch(owner,'input-archive',asset['archive_sha256'],root/'input-archive.tar.gz');expected={r['path']:r for r in asset['inventory']};data=root/'data';data.mkdir();seen=set()
    with tarfile.open(archive,'r:gz') as t:
        for member in t:
            rel=PurePosixPath(member.name);assert member.isfile() and member.name in expected and member.name not in seen and not rel.is_absolute() and '..' not in rel.parts;p=data/member.name;p.parent.mkdir(exist_ok=True,parents=True);entry=expected[member.name];assert member.size==entry['bytes']
            with p.open('xb') as f:
                stream=t.extractfile(member)
                for block in iter(lambda:stream.read(1024**2),b''):f.write(block)
            assert sha(p)==entry['sha256'];seen.add(member.name)
    assert seen==set(expected);training=Task.get_task(task_id=get('training_task_id'));assert training.status=='completed';fetch(training,'association-checkpoint',get('checkpoint_sha256'),root/'checkpoint.pt');acceptance=Task.get_task(task_id=get('checkpoint_acceptance_task_id'));assert acceptance.status=='completed';proof=fetch(acceptance,'tensor-forward-acceptance',get('checkpoint_acceptance_artifact_sha256'),root/'checkpoint-forward-acceptance.json');v=json.loads(proof.read_text());assert v['all_four_rank_tensor_and_real_fit_forward_checks_passed'] and v['checkpoint_sha256']==get('checkpoint_sha256') and v['training_task_id']==training.id
    output=root/'predictions';output.mkdir();(output/'logits').mkdir();assert torch.cuda.device_count()==4
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(port,str(root),fold),nprocs=4,join=True);ranks=[json.loads((output/f'rank-{rank}-receipt.json').read_text()) for rank in range(4)];assert len({r['GPU_uuid'] for r in ranks})==4 and len({r['host'] for r in ranks})==1;rows=[row for r in ranks for row in r['rows']];input_rows=json.loads((data/'examples.json').read_text());key=lambda r:(r['sequence_id'],r['vehicle_frame_id']);assert len(rows)==len(input_rows) and {key(r) for r in rows}=={key(r) for r in input_rows}
    archive=output/'all-logits.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for row in sorted(rows,key=key):t.add(output/row['logits']['path'],arcname=row['logits']['path'],recursive=False)
    combined=output/'all-hypotheses.jsonl'
    with combined.open('x') as f:
        for rank in range(4):f.write((output/f'rank-{rank}-hypotheses.jsonl').read_text())
    runtime=output/'runtime.json';runtime.write_text(json.dumps([dict((k,v) for k,v in r.items() if k not in ['rows','hypotheses_file']) for r in ranks],indent=2)+'\n');result=output/'inference-manifest.json';result.write_text(json.dumps(dict(fold_id=fold,training_task_id=training.id,checkpoint_sha256=sha(root/'checkpoint.pt'),checkpoint_acceptance_task_id=acceptance.id,input_task_id=owner.id,input_manifest_sha256=sha(manifest),input_independent_acceptance_sha256=asset['input_independent_acceptance_sha256'],input_vehicle_references=len(rows),source_frames=asset['source_frames'],class_scope='car',records=sorted(rows,key=key),logit_archive_sha256=sha(archive),hypotheses_sha256=sha(combined),runtime_sha256=sha(runtime),source_event_inventory_sha256=sha(data/'source-events.json'),gate_confidences=[.90,.95,.99],TopH=[1,3,5],explicit_all_unmatched_preserved=True,weights_are_retained_energy_not_calibrated_full_posterior=True,optimizer_created=False,held_out_GT_read=False,full_channel_tracking_replay=False,paper_eligible=False),indent=2)+'\n')
    for name,p in [('inference-manifest',result),('all-logits',archive),('all-retained-hypotheses',combined),('actual-GPU-runtime',runtime),('source-event-inventory',data/'source-events.json')]:assert task.upload_artifact(name,artifact_object=p,wait_on_upload=True)
    print(json.dumps(dict(stage='full_vehicle_reference_association_inference_uploaded_pending_independent_readback',fold_id=fold,completed=len(rows),eta_seconds=0,paper_eligible=False)),flush=True)


if __name__=='__main__':main()
