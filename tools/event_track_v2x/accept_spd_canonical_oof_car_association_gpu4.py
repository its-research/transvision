"""Independent saved-tensor and equivalent functional forward check, fit only."""
import ast
from datetime import timedelta
import hashlib,json,os,platform,socket
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from clearml import Task
from clearml.storage.helper import StorageHelper
from urllib.parse import urlparse,urlunparse

# Fixed bindings and genuine car-only fit inputs are appended during source freeze.
BINDINGS = {}
MODEL_SOURCE = ''


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def worker(rank,port,checkpoint,fold,output):
    assert os.environ['NCCL_P2P_DISABLE']=='1'
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    assert platform.python_version()=='3.12.3' and torch.__version__=='2.6.0+cu124'
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=4,timeout=timedelta(seconds=120))
    data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert data['fold_id']==fold and data['epoch']==24 and data['seed']==1337 and data['feature_dimension']==203
    assert data['configuration']['class_scope']=='car' and data['configuration']['class_index']==0
    assert data['configuration']['data_manifest_sha256']==BINDINGS[str(fold)]['data_manifest_sha256']
    tree=ast.parse(MODEL_SOURCE);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PredictedAssociation');env={'nn':nn,'torch':torch};exec(compile(ast.Module(body=[cls],type_ignores=[]),'frozen_model_architecture','exec'),env)
    model=env['PredictedAssociation']();expected=model.state_dict();state=data['model'];assert set(state)==set(expected)
    for name,tensor in state.items():assert tensor.shape==expected[name].shape and tensor.dtype==torch.float32 and torch.isfinite(tensor).all()
    model.load_state_dict(state,strict=True);model.cuda(rank).eval();w={k:v.cuda(rank) for k,v in state.items()}
    fixture=BINDINGS[str(fold)]['fit_features'];left=torch.tensor(fixture['left'],device=rank,dtype=torch.float32)[None];right=torch.tensor(fixture['right'],device=rank,dtype=torch.float32)[None]
    def encode(x):
        x=F.linear(x,w['encoder.0.weight'],w['encoder.0.bias']);x=F.layer_norm(x,(128,),w['encoder.1.weight'],w['encoder.1.bias'],1e-5);return F.gelu(F.linear(F.gelu(x),w['encoder.4.weight'],w['encoder.4.bias']))
    with torch.no_grad():
        outputs=model(left,right);l,r=encode(left),encode(right);a=l[:,:,None,:].expand(-1,-1,r.shape[1],-1);b=r[:,None,:,:].expand(-1,l.shape[1],-1,-1)
        pair=F.linear(F.gelu(F.linear(torch.cat([a,b,abs(a-b),a*b],-1),w['pair.0.weight'],w['pair.0.bias'])),w['pair.3.weight'],w['pair.3.bias']).squeeze(-1)
        reference=[pair,F.linear(l,w['left_dustbin.weight'],w['left_dustbin.bias']).squeeze(-1),F.linear(r,w['right_dustbin.weight'],w['right_dustbin.bias']).squeeze(-1)]
        for actual,ref in zip(outputs,reference):assert torch.isfinite(actual).all() and torch.allclose(actual,ref,atol=1e-5,rtol=1e-5)
        empty=model(left[:,:0],right[:,:0]);assert empty[0].shape==(1,0,0)
    prop=torch.cuda.get_device_properties(rank);receipt=dict(rank=rank,GPU_name=prop.name,GPU_uuid=str(prop.uuid),host=socket.gethostname(),fold_id=fold,checkpoint_sha256=sha(Path(checkpoint)),all_tensor_keys_shapes_dtype_finiteness_verified=True,real_fit_feature_manual_functional_forward_agrees=True,empty_side_forward_verified=True,output_values=[x.cpu().tolist() for x in outputs],paper_eligible=False)
    receipts=[None]*4;dist.all_gather_object(receipts,receipt)
    if rank==0:
        assert len({r['GPU_uuid'] for r in receipts})==4 and len({r['host'] for r in receipts})==1
        for r in receipts[1:]:
            for a,b in zip(r['output_values'],receipts[0]['output_values']):assert torch.allclose(torch.tensor(a),torch.tensor(b),atol=1e-4,rtol=1e-4)
        Path(output).write_text(json.dumps(dict(fold_id=fold,training_task_id=BINDINGS[str(fold)]['training_task_id'],source_binding=BINDINGS[str(fold)],checkpoint_sha256=sha(Path(checkpoint)),GPU_rank_receipts=receipts,all_four_rank_tensor_and_real_fit_forward_checks_passed=True,held_out_GT_used=False,paper_eligible=False),indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


def main():
    task=Task.init(project_name='Thesis/EventTrack-V2X/Training',task_name='canonical car association independent acceptance',reuse_last_task_id=True,auto_connect_frameworks=False,auto_connect_arg_parser=False);task.output_uri='http://10.100.34.118:8081';fold=int(task.get_parameters()['General/fold_id']);binding=BINDINGS[str(fold)]
    parent=Task.get_task(task_id=binding['training_task_id']);assert parent.status=='completed'
    artifact=parent.artifacts['association-checkpoint'];assert artifact.hash==binding['checkpoint_sha256'] and artifact.size==binding['checkpoint_bytes']
    route=urlparse(artifact.url);assert route.scheme=='http' and route.netloc in ['10.100.34.118:8081','10.100.35.118:8081'] and not route.username and not route.query and not route.fragment
    url=urlunparse(route._replace(netloc='10.100.34.118:8081'));root=Path('car-association-accept-'+task.id);root.mkdir();p=root/'checkpoint.pt'
    with p.open('xb') as stream:
        for block in StorageHelper.get(url).download_as_stream(url,chunk_size=1024**2):stream.write(block)
    assert sha(p)==binding['checkpoint_sha256'] and p.stat().st_size==binding['checkpoint_bytes'];assert torch.cuda.device_count()==4
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    output=root/'tensor-forward-acceptance.json';mp.spawn(worker,args=(port,str(p),fold,str(output)),nprocs=4,join=True)
    assert task.upload_artifact('tensor-forward-acceptance',artifact_object=output,wait_on_upload=True)
    print(json.dumps(dict(stage='car_association_independent_GPU_forward_complete_pending_byte_readback',fold_id=fold,eta_seconds=0)),flush=True)


if __name__=='__main__':main()
