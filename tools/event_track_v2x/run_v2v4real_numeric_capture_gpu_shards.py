"""GPU task wrapper for the missing original post-head mathematical evidence.

Runtime preparation and physical collision admission belong to the frozen
bootstrap/dispatcher. This does not rerun raw export or certify network forward.
"""
import json
from pathlib import Path
import subprocess
import sys
import shutil
import time

from tools.event_track_v2x import run_v2v4real_official_test_gpu_shards as transport
from tools.event_track_v2x import run_v2v4real_native_numeric_capture as capture
from tools.event_track_v2x.admit_v2v4real_numeric_raw_inputs import admit,sha
from tools.event_track_v2x.merge_v2v4real_numeric_trace_shards import merge
from tools.event_track_v2x.v2v4real_numeric_trace_archive import pack



def storage_admission(root, *, world, per_rank_bytes, archive_budget, raw_budget):
    if world not in (4,8) or any(type(v) is not int or v<=0 for v in (per_rank_bytes,archive_budget,raw_budget)):
        raise ValueError('explicit positive storage budgets required')
    # Original projection/input bytes, isolated extension builds/package scratch,
    # two trace copies during merging, archive, and conservative NPZ/TAR overhead.
    required=20*1024**3+raw_budget+2*world*per_rank_bytes+archive_budget+3986*1024**2
    free=shutil.disk_usage(root).free
    receipt=dict(stage='numeric_capture_disk_admission',actual_free_bytes=free,
        required_free_bytes=required,within_budget=free>=required,
        scope='current container filesystem free bytes; not a host reservation')
    if free<required:raise ValueError('measured worker storage below declared capture budget')
    return receipt


def rank_progress(log_paths, offsets, states, started):
    for rank,path in enumerate(log_paths):
        if not path.exists():continue
        with path.open('rb') as f:
            f.seek(offsets.get(rank,0))
            while True:
                before=f.tell();line=f.readline()
                if not line or not line.endswith(b'\n'):
                    offsets[rank]=before;break
                offsets[rank]=f.tell()
                try:value=json.loads(line)
                except (ValueError,UnicodeError):continue
                if not isinstance(value,dict) or value.get('stage')!='native_numeric_trace_capture':continue
                done=value.get('completed_rows');total=value.get('total_rows')
                if type(done) is not int or type(total) is not int or not 0<=done<=total<=3986:
                    raise ValueError('invalid real rank progress')
                old=states.get(rank)
                if old and (old['total']!=total or old['done']>done):raise ValueError('rank progress regressed')
                states[rank]=dict(done=done,total=total)
    completed=sum(v['done'] for v in states.values())
    all_observed=len(states)==len(log_paths)
    if all_observed and sum(v['total'] for v in states.values())!=3986:
        raise ValueError('rank progress cohort coverage differs')
    return dict(stage='native_numeric_trace_task_progress',completed_rows=completed,total_rows=3986,
        observed_ranks=len(states),world_size=len(log_paths),
        ETA_seconds=(time.monotonic()-started)/completed*(3986-completed) if all_observed and completed else None,
        ETA_scope='this trace capture only; excludes packaging/readback/independent acceptance')

def launch(root, *, source, prepared, admission, checkpoint, seed, world,
           raw_index, raw_sha, raw_root, per_rank_bytes):
    if type(world) is not int or world not in (4,8) or type(per_rank_bytes) is not int or per_rank_bytes<=0:
        raise ValueError('four/eight cards and explicit positive trace budget required')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=world:
        raise ValueError('assigned physical CUDA count differs')
    children=[];logs=[];directories=[];offsets={};states={};started=time.monotonic();last=started
    try:
        for rank in range(world):
            directory=root/f'rank{rank}';directories.append(directory)
            log=(root/f'rank{rank}.log').open('xb');logs.append(log)
            command=[sys.executable,str(Path(capture.__file__).absolute()),
                '--source-archive',str(source),'--inputs-root',str(prepared),'--admission',str(admission),
                '--checkpoint',str(checkpoint),'--seed',str(seed),'--output',str(directory),
                '--accepted-raw-index',str(raw_index),'--accepted-raw-sha',raw_sha,
                '--accepted-raw-root',str(raw_root),'--max-trace-bytes',str(per_rank_bytes),
                '--device',f'cuda:{rank}','--rank',str(rank),'--world-size',str(world),
                '--trace-layout','compact-cells-v1']
            print(json.dumps(dict(stage='original_numeric_capture_rank_launch',rank=rank,ETA_seconds=None)),flush=True)
            children.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
        while True:
            statuses=[p.poll() for p in children]
            now=time.monotonic()
            if now-last>=15 or all(v==0 for v in statuses):
                print(json.dumps(rank_progress([root/f'rank{r}.log' for r in range(world)],offsets,states,started)),flush=True)
                last=now
            if any(s is not None and s!=0 for s in statuses):raise RuntimeError('numeric capture rank failed; original logs retained')
            if all(s==0 for s in statuses):break
            time.sleep(1)
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for log in logs:log.close()
    return merge([d/'trace' for d in directories],full_frames=prepared/'frames.jsonl',
        frames_sha=transport.export.FRAME_SHA,raw_index=raw_index,raw_sha=raw_sha,output=root/'merged')


def run(task, *, accepted_proof, proof_sha):
    from clearml import Task
    params=task.get_parameters();seed=int(params['General/seed']);world=int(params['General/world_size'])
    per_rank_bytes=int(params['General/max_trace_bytes_per_rank']);archive_budget=int(params['General/max_archive_bytes'])
    input_budget=int(params['General/max_raw_unpacked_bytes'])
    if seed not in transport.CHECKPOINT_TASKS or world not in (4,8) or min(per_rank_bytes,archive_budget,input_budget)<=0:
        raise ValueError('qualified native seed, card count and explicit storage budgets required')
    if sha(accepted_proof)!=proof_sha:raise ValueError('frozen accepted proof differs')
    proof=json.loads(Path(accepted_proof).read_bytes())
    if proof.get('checkpoint_seed')!=seed or proof.get('independent_byte_schema_coverage_accepted') is not True:
        raise ValueError('accepted native seed proof required')
    root=Path('/tmp')/('rbf-original-numeric-capture-'+task.id);root.mkdir(exist_ok=False)
    try:
        storage=storage_admission(root,world=world,per_rank_bytes=per_rank_bytes,archive_budget=archive_budget,raw_budget=input_budget)
        print(json.dumps(storage),flush=True)
        source=root/'original-source.tar.gz';projection=root/'projection.tar.gz';checkpoint=root/'checkpoint'
        publication=root/'publication.json';evidence=[]
        for owner,name,digest,path in [
            (transport.SOURCE_TASK,'source',transport.export.SOURCE_ARCHIVE_SHA,source),
            (transport.export.PROJECTION_TASK,'publication-receipt',transport.PUBLICATION_SHA,publication),
            (transport.export.PROJECTION_TASK,'projection',transport.PROJECTION_ARCHIVE_SHA,projection),
            (transport.CHECKPOINT_TASKS[seed],'best-checkpoint',transport.export.CHECKPOINTS[seed],checkpoint)]:
            evidence.append(transport.download(Task.get_task(task_id=owner),name,digest,path))
        raw_task=Task.get_task(task_id=proof['task_id']);pins=proof['input_sha256']
        for name in ('native-features','native-feature-manifest','native-feature-index'):
            evidence.append(transport.download(raw_task,name,pins[name],root/name))
        input_receipt=admit(proof=accepted_proof,proof_sha=proof_sha,task_id=raw_task.id,seed=seed,
            archive=root/'native-features',manifest=root/'native-feature-manifest',index=root/'native-feature-index',
            output=root/'accepted-raw',max_unpacked_bytes=input_budget)
        prepared,admission=transport.unpack_projection(projection,root/'projection',publication)
        result=launch(root,source=source,prepared=prepared,admission=admission,checkpoint=checkpoint,
            seed=seed,world=world,raw_index=root/'native-feature-index',raw_sha=pins['native-feature-index'],
            raw_root=root/'accepted-raw',per_rank_bytes=per_rank_bytes)
        envelope=pack(root/'merged',root/'transport',expected_rows=3986,
            raw_index_sha=pins['native-feature-index'],max_archive_bytes=archive_budget)
        result.update(kind='v2v4real_original_numeric_capture_GPU_candidate_v1',seed=seed,world_size=world,
            registered_inputs=evidence,raw_input_transport=input_receipt,trace_transport=envelope,
            accepted_proof_sha256=proof_sha,worker_storage_admission=storage,gt_read=False,independent_numeric_acceptance=False,
            network_forward_verified=False,nine_state_cache_admitted=False,tracking_accepted=False,paper_metric_complete=False)
        (root/'execution-receipt.json').write_text(json.dumps(result,sort_keys=True,allow_nan=False))
        for name,path in [('numeric-traces',root/'transport/numeric-traces.tar'),
            ('trace-manifest',root/'transport/trace-manifest.json'),('execution-receipt',root/'execution-receipt.json')]:
            task.upload_artifact(name,artifact_object=str(path),wait_on_upload=True)
        for rank in range(world):
            task.upload_artifact(f'rank{rank}-runtime',artifact_object=str(root/f'rank{rank}/numeric-capture-receipt.json'),wait_on_upload=True)
            task.upload_artifact(f'rank{rank}-extension',artifact_object=str(root/f'rank{rank}/extension-build-receipt.json'),wait_on_upload=True)
            extension=json.loads((root/f'rank{rank}/extension-build-receipt.json').read_bytes())
            binary=root/f'rank{rank}/frozen-runtime'/extension['binaries'][0]['path']
            task.upload_artifact(f'rank{rank}-extension-binary',artifact_object=str(binary),wait_on_upload=True)
        return result
    except Exception as exc:
        failure=root/'failure.json';failure.write_text(json.dumps(dict(exception_type=type(exc).__name__,seed=seed,
            independent_numeric_acceptance=False,whole_experiment_ETA='unknown')))
        task.upload_artifact('failure-receipt',artifact_object=str(failure),wait_on_upload=True)
        for log in root.glob('rank*.log'):task.upload_artifact(log.stem,artifact_object=str(log),wait_on_upload=True)
        raise
