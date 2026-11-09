"""Build source-bound numeric GPU candidate code, with existing raw proof pins."""
import base64
import hashlib
import json
from pathlib import Path

FILES=('run_v2v4real_numeric_capture_gpu_shards.py','run_v2v4real_native_numeric_capture.py',
    'capture_v2v4real_native_numeric_rows.py','capture_v2v4real_native_heads.py',
    'v2v4real_native_numeric_reference.py','v2v4real_compact_bev_trace.py',
    'verify_v2v4real_native_numeric_traces.py','merge_v2v4real_numeric_trace_shards.py',
    'v2v4real_numeric_trace_archive.py','admit_v2v4real_numeric_raw_inputs.py',
    'run_v2v4real_official_test_raw_features.py','run_v2v4real_official_test_gpu_shards.py',
    'merge_v2v4real_official_test_raw_shards.py','v2v4real_official_test_runtime.py')
RAW_TASKS={1337:'c0f709ae40f4409ca71fa01cda5f7eeb',2027:'fc76e021981e4650afd35c24fedb35a8',3407:'1912710926bd4352b928217f8685d38f'}


def build(directory, proofs):
    if set(proofs)!=set(RAW_TASKS):raise ValueError('all three frozen seed proof pins required')
    files={};proof_pins={}
    for name in FILES:
        path=Path(directory)/name
        if not path.is_file() or path.is_symlink():raise ValueError('ordinary source file required')
        raw=path.read_bytes();compile(raw,str(path),'exec')
        files['tools/event_track_v2x/'+name]=raw
    for seed,(path,pin) in proofs.items():
        path=Path(path)
        if path.is_symlink():raise ValueError('ordinary accepted proof required')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('frozen accepted proof SHA differs')
        proof=json.loads(raw)
        if (proof.get('task_id')!=RAW_TASKS[seed] or proof.get('checkpoint_seed')!=seed
                or proof.get('independent_byte_schema_coverage_accepted') is not True
                or not all(name in proof['input_sha256'] for name in ('native-features','native-feature-manifest','native-feature-index'))):
            raise ValueError('accepted original raw seed proof required')
        relative=f'accepted-proofs/seed{seed}.json';files[relative]=raw
        proof_pins[str(seed)]=dict(path=relative,sha256=pin,task_id=RAW_TASKS[seed])
    inventory={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in files.items()}
    identity=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    embedded={name:dict(inventory[name],base64=base64.b64encode(raw).decode()) for name,raw in files.items()}
    script='''"""Frozen original post-head numeric capture candidate; no acceptance claims."""
import base64,hashlib,json,os,sys
from pathlib import Path
FILES = '''+repr(embedded)+'''
BUNDLE_SHA256 = '''+repr(identity)+'''
PROOFS = '''+repr(proof_pins)+'''
def materialize(root):
    root=Path(root)
    if root.exists() or any(p.is_symlink() for p in root.parents):raise ValueError('fresh execution code directory required')
    decoded={}
    for name,item in FILES.items():
        raw=base64.b64decode(item['base64'],validate=True)
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('embedded source bytes differ')
        decoded[name]=raw
    root.mkdir(parents=True,exist_ok=False)
    for name,raw in decoded.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(raw)
    return root

def main():
    from clearml import Task
    task=Task.init(project_name='Thesis/Recover-Before-Fuse/Inference',task_name='V2V4Real original post-head numeric capture',
        reuse_last_task_id=False,auto_connect_frameworks=False,auto_connect_arg_parser=False)
    params=task.get_parameters()
    if params.get('General/execution_bundle_sha256')!=BUNDLE_SHA256:raise ValueError('execution bundle differs')
    seed=str(params['General/seed'])
    if seed not in PROOFS:raise ValueError('registered seed proof required')
    if any(n=='tools' or n.startswith('tools.') for n in sys.modules):raise ValueError('ambient execution namespace forbidden')
    root=materialize(Path('/tmp')/('rbf-numeric-code-'+task.id))
    sys.path.insert(0,str(root))
    os.environ['PYTHONPATH']=str(root)
    from tools.event_track_v2x.v2v4real_official_test_runtime import prepare_runtime
    prepare_runtime(task,root)
    from tools.event_track_v2x.run_v2v4real_numeric_capture_gpu_shards import run
    pin=PROOFS[seed]
    run(task,accepted_proof=root/pin['path'],proof_sha=pin['sha256'])
    task.close()
if __name__=='__main__':main()
'''
    compile(script,'numeric_bootstrap.py','exec')
    return script,dict(kind='v2v4real_numeric_GPU_candidate_source_bundle_v1',execution_bundle_sha256=identity,
        bootstrap_sha256=hashlib.sha256(script.encode()).hexdigest(),sources=inventory,accepted_raw_proofs=proof_pins,
        actual_GPU_execution_verified=False,independent_numeric_acceptance=False,network_forward_verified=False)
