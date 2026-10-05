"""Verify a completed fold checkpoint on real held-out frames before full export.

Sampled forward acceptance does not prove full OOF prediction coverage or metrics.
"""
import argparse,copy,json,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
from run_spd_canonical_oof_raw_cache import (
    build_inference_dataset,check_model_inputs,state_hash,digest,
)
from spd_canonical_oof_export_binding import validate_binding
from spd_canonical_oof_cache_primitives import install_raw_detector_decode
from spd_canonical_oof_query_decode import install_uncropped_query_decode


RUNTIME_ARCHIVE_SHA256 = 'b6b39c66eec6921c0f0f51d2b9cd4e5571fcafdc44d3605112dfb349af8fab32'
CHECKPOINT_RUNTIME_SOURCES = {
    'runner/iter_based_runner.py': '2822bd2d628d6f182c378b34d96e333711edf755f37549b69b255740e3475625',
    'runner/hooks/checkpoint.py': '2b33e28b197e2326b7324ed86f6e7735b4b218ded77b7bbc4d8146aff127d39a',
    'runner/hooks/optimizer.py': 'e74e5878669645cc54be3235da60464dae345c71402756dbc6bba48a1cf9d2ed',
    'version.py': '0787bda6f81b6bc7848e5e22fdabec9109d960e26332da45a0ab75277a88b83f',
}


def verify_checkpoint_save_runtime():
    import torch
    root = Path(torch.__file__).resolve().parents[1] / 'mmcv'
    for name, expected in CHECKPOINT_RUNTIME_SOURCES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('checkpoint save runtime source differs from frozen archive')
    return {'runtime_archive_sha256': RUNTIME_ARCHIVE_SHA256,
            'component_sha256': dict(CHECKPOINT_RUNTIME_SOURCES),
            'contract': 'MMCV-1.4.0-after_train_iter-zero-based-meta-v1'}


def inspect_checkpoint_envelope(envelope, expected_iteration, checkpoint_filename):
    import torch
    if not isinstance(envelope,dict) or not isinstance(envelope.get('state_dict'),dict):
        raise ValueError('missing final state dictionary')
    runtime_binding = verify_checkpoint_save_runtime()
    if (type(expected_iteration) is not int or expected_iteration <= 0
            or checkpoint_filename != 'iter_%d.pth' % expected_iteration):
        raise ValueError('completed iteration or saved checkpoint filename differs')
    meta=envelope.get('meta')
    if not isinstance(meta,dict) or type(meta.get('iter')) is not int or meta['iter']!=expected_iteration-1:
        raise ValueError('checkpoint tensor envelope iteration differs from completed fit')
    weights=envelope['state_dict']
    if not weights or any(not isinstance(k,str) or not isinstance(v,torch.Tensor)
                          or not torch.isfinite(v).all() for k,v in weights.items()):
        raise ValueError('invalid or nonfinite checkpoint state')
    return {'checkpoint_iteration':meta['iter'],
            'completed_micro_iterations':expected_iteration,
            'checkpoint_filename':checkpoint_filename,
            'checkpoint_save_runtime':runtime_binding,'state_tensor_count':len(weights),
            'state_elements':sum(v.numel() for v in weights.values()),
            'all_state_tensors_finite':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('inputs','upstream','package-manifest','byte-freeze','checkpoint',
                'training-config','training-launch','training-startup','training-completion','output'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('checkpoint-sha256','package-manifest-sha256','byte-freeze-sha256','input-manifest-sha256'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--side',choices=('vehicle-side','infrastructure-side'),required=True)
    p.add_argument('--seed',type=int,choices=(1337,2027,3407),required=True)
    p.add_argument('--shard-index',type=int,choices=(0,1),required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('forward evidence is create-once')
    for path,expected in ((a.package_manifest,a.package_manifest_sha256),(a.byte_freeze,a.byte_freeze_sha256),
                          (a.inputs/'input-manifest.json',a.input_manifest_sha256),(a.checkpoint,a.checkpoint_sha256)):
        if digest(path)!=expected:raise ValueError('bound bytes differ before checkpoint deserialization')
    package=json.loads(a.package_manifest.read_bytes());frozen=json.loads(a.byte_freeze.read_bytes())
    inputs=json.loads((a.inputs/'input-manifest.json').read_bytes())
    launch=json.loads(a.training_launch.read_bytes());startup=json.loads(a.training_startup.read_bytes());completion=json.loads(a.training_completion.read_bytes())
    evidence={a.side+suffix:path.read_bytes() for suffix,path in (
        ('-launch-receipt.json',a.training_launch),('-optimizer-startup.json',a.training_startup),('-completion',a.training_completion))}
    binding=validate_binding(package,inputs,frozen,launch,startup,completion,side=a.side,seed=a.seed,
        checkpoint_sha256=a.checkpoint_sha256,config_sha256=digest(a.training_config),
        package_manifest_sha256=a.package_manifest_sha256,input_manifest_sha256=a.input_manifest_sha256,
        evidence_payloads=evidence)
    sys.path.insert(0,str(a.upstream))
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel,collate
    from mmcv.runner import load_checkpoint
    from mmdet.apis import set_random_seed
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    import projects.mmdet3d_plugin
    if torch.cuda.device_count()!=1:raise ValueError('one physically assigned CUDA device required')
    set_random_seed(a.seed,deterministic=True)
    if a.checkpoint.name != a.side+'-'+completion['checkpoint']:
        raise ValueError('frozen checkpoint filename differs from completed artifact')
    tensor_report=inspect_checkpoint_envelope(torch.load(str(a.checkpoint),map_location='cpu'),launch['max_micro_iterations'],completion['checkpoint'])
    dataset,cfg,rows,scenes,m=build_inference_dataset(a.inputs,a.side,a.upstream,a.shard_index,2,package,a.input_manifest_sha256)
    training=Config.fromfile(str(a.training_config));model_cfg=copy.deepcopy(training.model)
    model_cfg.pretrained=None;model_cfg.batch_size=1;model_cfg.train_cfg=None
    if model_cfg.train_det is not True or model_cfg.spatial_temporal_reason.history_reasoning or model_cfg.spatial_temporal_reason.future_reasoning:
        raise ValueError('not frozen single-frame detector config')
    model=build_model(model_cfg,test_cfg=training.get('test_cfg'))
    load_checkpoint(model,str(a.checkpoint),map_location='cpu',strict=True)
    model.requires_grad_(False).eval();model.CLASSES=dataset.CLASSES
    before=state_hash(model)
    uncropped=install_uncropped_query_decode(model,denormalize_bbox);snapshot=install_raw_detector_decode(model)
    model=MMDataParallel(model.cuda(),device_ids=[0])
    selected={0,len(dataset)//2,len(dataset)-1}
    for scene in scenes:selected.add(next(i for i,row in enumerate(dataset.data_infos) if row['scene_token']==scene))
    records=[];started=time.monotonic()
    for ordinal,index in enumerate(sorted(selected)):
        data=collate([dataset[index]],samples_per_gpu=1);check_model_inputs(data)
        with torch.no_grad():out=model(return_loss=False,rescale=True,**data)
        info=dataset.data_infos[index]
        if len(out)!=1 or out[0]['token']!=info['token'] or snapshot['frames']!=ordinal+1 or uncropped['frames']!=ordinal+1:
            raise ValueError('forward identity or raw-head snapshot differs')
        if snapshot['last_audit']['frame_id']!=str(info['token']):raise ValueError('snapshot token differs')
        result=out[0];boxes=result['boxes_3d_det'].tensor;scores=result['scores_3d_det'];labels=result['labels_3d_det']
        count=snapshot['last_audit']['head_queries']
        if (count<=0 or tuple(boxes.shape)!=(count,9) or tuple(scores.shape)!=(count,) or tuple(labels.shape)!=(count,)
                or not torch.isfinite(boxes).all() or not torch.isfinite(scores).all()
                or not (boxes[:,3:6]>0).all() or not ((scores>=0)&(scores<=1)).all()
                or not ((labels>=0)&(labels<3)).all()):
            raise ValueError('raw-head forward dropped queries or has invalid geometry/scores/classes')
        records.append({'sequence_id':info['scene_token'],'frame_id':info['token'],'queries':int(count),
                        'frame_index_row':rows[info['token']],'arrays_finite':True})
        print('OOF_CHECKPOINT_FORWARD_PROGRESS '+json.dumps({'frames':len(records),'total':len(selected),
              'eta_seconds':(time.monotonic()-started)*(len(selected)-len(records))/len(records)}),flush=True)
    if state_hash(model.module)!=before:raise ValueError('checkpoint state mutated during sampled inference')
    if {r['sequence_id'] for r in records}!=set(scenes):raise ValueError('sample does not cover each shard scene')
    receipt={'kind':'spd_canonical_oof_final_checkpoint_sampled_forward_acceptance_v1','binding':binding,
             'byte_freeze_sha256':a.byte_freeze_sha256,'tensor_report':tensor_report,'records':records,
             'strict_state_dict_load_verified':True,'raw_head_forward_verified':True,'weights_unchanged':True,
             'actual_device':torch.cuda.get_device_name(0),'side':a.side,'fold_id':package['fold_id'],'seed':a.seed,
             'shard_index':a.shard_index,'shard_count':2,'sample_scope':'first/middle/last plus first frame of each held-out shard scene',
             'complete_prediction_coverage_verified':False,'formal_paper_eligible':False,
             'verifier_sha256':digest(Path(__file__)),'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    with a.output.open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    print('OOF_CHECKPOINT_SAMPLED_FORWARD_ACCEPTED '+digest(a.output),flush=True)
if __name__=='__main__':main()
