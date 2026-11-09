"""Independent completed-candidate trace transport and full-row post-head math.

Uses previously accepted raw bytes; no new raw production, test GT or fit.
Does not certify the network forward, 9D, tracking or benchmark metrics.
"""
import argparse
import json
from pathlib import Path
from tools.event_track_v2x import run_v2v4real_official_test_raw_features as original
from tools.event_track_v2x.run_v2v4real_official_test_gpu_shards import download
from tools.event_track_v2x.v2v4real_numeric_trace_archive import readback
from tools.event_track_v2x.admit_v2v4real_numeric_raw_inputs import admit
from tools.event_track_v2x.verify_v2v4real_native_numeric_traces import digest,verify
from tools.event_track_v2x.readback_v2v4real_official_test_raw_gpu import verify_runtime_dependency_receipt
from tools.event_track_v2x import v2v4real_official_test_runtime as runtime


def qualify(task,source):
    params=task.get_parameters()
    if str(task.status)!='completed':raise ValueError('completed candidate required; retain live tasks')
    if (digest_text(task.data.script.diff)!=source['bootstrap_sha256']
            or params.get('General/execution_bundle_sha256')!=source['execution_bundle_sha256']):
        raise ValueError('cloud candidate execution bytes differ')
    seed=int(params['General/seed']);world=int(params['General/world_size'])
    if seed not in original.CHECKPOINTS or world not in (4,8):raise ValueError('registered seed and ranks required')
    return seed,world


def digest_text(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def verify_rank(value, *, seed, rank, world, source, raw_sha, interpreter):
    if (value.get('kind')!='v2v4real_native_numeric_trace_producer_v1' or value.get('seed')!=seed
            or value.get('projection_frames_sha256')!=original.FRAME_SHA or value.get('gt_read') is not False
            or value.get('TF32_enabled') is not False or value.get('trace_layout')!='compact-cells-v1'
            or value.get('producer_sha256')!=source['sources']['tools/event_track_v2x/run_v2v4real_native_numeric_capture.py']['sha256']
            or value.get('accepted_raw_index_sha256')!=raw_sha
            or value.get('shard')!=dict(rank=rank,world_size=world,full_source_rows=3986)
            or value.get('actual_device',{}).get('device')!=f'cuda:{rank}'
            or value.get('rank_runtime',{}).get('interpreter')!=interpreter):
        raise ValueError('rank source/runtime/original-row admission differs')
    device=value['actual_device']
    if device.get('device_type')!='cuda' or device.get('logical_index')!=rank or not device.get('name'):
        raise ValueError('actual CUDA rank identity missing')
    expected_rows=2*(1993//world+(rank<1993%world))
    if type(value.get('rows')) is not int or value['rows']!=expected_rows:
        raise ValueError('original frame-pair rank row count differs')
    runtime.require_runtime(value['rank_runtime'])
    for field in ('independent_numeric_acceptance','network_forward_verified','nine_state_cache_admitted','tracking_accepted','paper_metric_complete'):
        if value.get(field) is not False:raise ValueError('producer scope exceeded')
    return value['rows']


def verify_execution(execution, envelope, *, params, seed, world, proof_sha, raw_sha):
    if (execution.get('kind')!='v2v4real_original_numeric_capture_GPU_candidate_v1'
            or execution.get('seed')!=seed or execution.get('world_size')!=world
            or execution.get('rows')!=3986 or execution.get('accepted_proof_sha256')!=proof_sha
            or execution.get('gt_read') is not False or execution.get('trace_transport')!=envelope
            or envelope.get('rows')!=3986 or envelope.get('raw_index_sha256')!=raw_sha):
        raise ValueError('complete original execution envelope required')
    for value in (execution,envelope):
        for field in ('independent_numeric_acceptance','network_forward_verified','nine_state_cache_admitted','tracking_accepted','paper_metric_complete'):
            if value.get(field) is not False:raise ValueError('candidate acceptance scope exceeded')
    budgets=[int(params['General/'+name]) for name in ('max_trace_bytes_per_rank','max_archive_bytes','max_raw_unpacked_bytes')]
    if min(budgets)<=0:raise ValueError('positive registered budgets required')
    per_rank,archive,raw=budgets
    storage=execution.get('worker_storage_admission',{})
    required=20*1024**3+raw+2*world*per_rank+archive+3986*1024**2
    if (storage.get('required_free_bytes')!=required or storage.get('within_budget') is not True
            or type(storage.get('actual_free_bytes')) is not int or storage['actual_free_bytes']<required
            or type(envelope.get('archive_bytes')) is not int or not 0<envelope['archive_bytes']<=archive):
        raise ValueError('registered storage budget or measured admission differs')


def verify_original_rank(value, extension, *, files, builder_ast, interpreter):
    import re
    bindings={name:dict(path=path,sha256=files[path]) for name,path in (
        ('tools.event_track_v2x.prepare_v2v4real_inputs','project/tools/event_track_v2x/prepare_v2v4real_inputs.py'),
        ('transvision.models.event_track_v2x.v2v4real_inputs','project/transvision/models/event_track_v2x/v2v4real_inputs.py'))}
    if value.get('point_reader_binding')!=bindings:raise ValueError('original point reader binding differs')
    expected=dict(kind='v2v4real_frozen_official_extension_build_v1',status='completed',
        source_archive_sha256=original.SOURCE_ARCHIVE_SHA,builder_path=original.EXTENSION_BUILDER,
        builder_sha256=original.EXTENSION_BUILDER_SHA,builder_function_ast_sha256=builder_ast,
        python_executable=interpreter,original_source_count=len(files),original_sources_unchanged=True,
        subprocess_import_verified=True,independent_runtime_acceptance=False)
    if any(extension.get(k)!=v for k,v in expected.items()):raise ValueError('original extension provenance differs')
    commands=extension.get('commands',[])
    if len(commands)!=2:raise ValueError('exact original extension commands required')
    root=Path(commands[0]['cwd'])
    if not root.is_absolute() or root.name!='official' or commands[1].get('cwd')!=str(root):
        raise ValueError('original extension working directory differs')
    argv=[[interpreter,str(root/'opencood/utils/setup.py'),'build_ext','--inplace'],
        [interpreter,'-c','from opencood.utils.box_overlaps import bbox_overlaps; assert callable(bbox_overlaps)']]
    for c,a,t in zip(commands,argv,(900,180)):
        if c.get('argv')!=a or c.get('timeout_seconds')!=t or c.get('check') is not True or c.get('status')!='completed' or c.get('returncode')!=0:
            raise ValueError('original extension command differs')
    binaries=extension.get('binaries',[])
    if len(binaries)!=1:raise ValueError('one original built extension required')
    b=binaries[0]
    if (not re.fullmatch(r'official/opencood/utils/box_overlaps[^/]*\.so',b.get('path',''))
            or not re.fullmatch('[0-9a-f]{64}',b.get('sha256','')) or type(b.get('bytes')) is not int or b['bytes']<=0):
        raise ValueError('original extension binary inventory differs')


def verify_registered_inputs(rows, expected, owners):
    from urllib.parse import urlparse
    if len(rows)!=len(expected):raise ValueError('exact registered input inventory required')
    seen=set()
    for row in rows:
        key=(row.get('task_id'),row.get('artifact'))
        if key in seen or key not in expected:raise ValueError('duplicate or substituted registered input')
        seen.add(key);item=owners[key[0]].artifacts[key[1]];url=urlparse(item.url)
        if (item.hash!=expected[key] or row.get('sha256')!=expected[key] or row.get('bytes')!=item.size
                or type(row.get('bytes')) is not int or row['bytes']<=0
                or url.scheme!='http' or url.netloc not in {'10.100.34.118:8081','10.100.35.118:8081'}
                or url.username or url.password or url.query or url.fragment
                or row.get('registered_file_host')!=url.netloc or row.get('transport_file_host')!='10.100.34.118:8081'):
            raise ValueError('registered execution input bytes or transport differs')


def run(task, *, source_dir, accepted_raw_dir, output, max_trace_bytes, max_raw_bytes, original_source_archive):
    source_dir=Path(source_dir);raw_dir=Path(accepted_raw_dir);out=Path(output)
    original.asset(original_source_archive,original.SOURCE_ARCHIVE_SHA)
    import tarfile,yaml,ast,hashlib
    with tarfile.open(original_source_archive) as archive:
        config=yaml.safe_load(archive.extractfile('official/opencood/hypes_yaml/point_pillar_late_fusion.yaml').read())
    with tarfile.open(original_source_archive) as archive:
        files={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive if m.isfile()}
        tree=ast.parse(archive.extractfile(original.EXTENSION_BUILDER).read())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_official_extensions')
    builder_ast=hashlib.sha256(ast.dump(function,include_attributes=False).encode()).hexdigest()
    lidar_range=config['preprocess']['cav_lidar_range']
    source=json.loads((source_dir/'source-manifest.json').read_bytes())
    # Check exact frozen execution closure before using its runtime policy.
    if digest(source_dir/'bootstrap.py')!=source['bootstrap_sha256']:raise ValueError('frozen bootstrap differs')
    for name,row in source['sources'].items():
        p=source_dir/'execution-code'/name
        if p.is_symlink() or p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:
            raise ValueError('frozen execution closure differs')
    seed,world=qualify(task,source);pin=source['accepted_raw_proofs'][str(seed)]
    proof_path=raw_dir/'independent-byte-schema-coverage-receipt.json'
    if digest(proof_path)!=pin['sha256']:raise ValueError('existing accepted raw proof differs')
    proof=json.loads(proof_path.read_bytes());raw_sha=proof['input_sha256']['native-feature-index']
    if proof['task_id']!=pin['task_id'] or proof['checkpoint_seed']!=seed:raise ValueError('raw proof seed/task differs')
    if digest(raw_dir/'native-feature-index')!=raw_sha:raise ValueError('existing raw index differs')
    out.mkdir(parents=True,exist_ok=False);inputs=[]
    names=['numeric-traces','trace-manifest','execution-receipt','runtime-dependencies',*[f'rank{r}-{kind}' for r in range(world) for kind in ('runtime','extension','extension-binary')]]
    for name in names:inputs.append(download(task,name,task.artifacts[name].hash,out/name))
    offline=None;runtime_value=json.loads((out/'runtime-dependencies').read_bytes())
    if runtime_value.get('offline_libraries'):
        from clearml import Task
        offline=out/'original-library-manifest'
        inputs.append(download(Task.get_task(task_id=runtime.OFFLINE_TASK),'manifest',runtime.OFFLINE_MANIFEST_SHA,offline))
    runtime_receipt=verify_runtime_dependency_receipt(out/'runtime-dependencies',source,offline)
    execution=json.loads((out/'execution-receipt').read_bytes());envelope=json.loads((out/'trace-manifest').read_bytes())
    verify_execution(execution,envelope,params=task.get_parameters(),seed=seed,world=world,proof_sha=pin['sha256'],raw_sha=raw_sha)
    from clearml import Task
    from tools.event_track_v2x import run_v2v4real_official_test_gpu_shards as transport_policy
    expected={(transport_policy.SOURCE_TASK,'source'):original.SOURCE_ARCHIVE_SHA,
        (original.PROJECTION_TASK,'publication-receipt'):transport_policy.PUBLICATION_SHA,
        (original.PROJECTION_TASK,'projection'):transport_policy.PROJECTION_ARCHIVE_SHA,
        (transport_policy.CHECKPOINT_TASKS[seed],'best-checkpoint'):original.CHECKPOINTS[seed]}
    expected.update({(pin['task_id'],name):proof['input_sha256'][name] for name in ('native-features','native-feature-manifest','native-feature-index')})
    owners={owner:Task.get_task(task_id=owner) for owner,_ in expected}
    verify_registered_inputs(execution.get('registered_inputs',[]),expected,owners)
    count=0
    for rank in range(world):
        value=json.loads((out/f'rank{rank}-runtime').read_bytes())
        extension_path=out/f'rank{rank}-extension'
        if digest(extension_path)!=value.get('extension_build_sha256'):raise ValueError('rank extension receipt binding differs')
        extension=json.loads(extension_path.read_bytes())
        verify_original_rank(value,extension,files=files,builder_ast=builder_ast,interpreter=runtime_value['after']['interpreter'])
        binary=extension['binaries'][0];binary_path=out/f'rank{rank}-extension-binary'
        if binary_path.stat().st_size!=binary['bytes'] or digest(binary_path)!=binary['sha256']:
            raise ValueError('original built extension binary bytes differ')
        count+=verify_rank(value,seed=seed,rank=rank,world=world,source=source,raw_sha=raw_sha,
            interpreter=runtime_value['after']['interpreter'])
    if count!=3986:raise ValueError('full original rank coverage required')
    admit(proof=proof_path,proof_sha=pin['sha256'],task_id=proof['task_id'],seed=seed,
        archive=raw_dir/'native-features',manifest=raw_dir/'native-feature-manifest',index=raw_dir/'native-feature-index',
        output=out/'accepted-raw',max_unpacked_bytes=max_raw_bytes)
    transport=readback(out/'numeric-traces',out/'trace-manifest',out/'traces',
        manifest_sha=task.artifacts['trace-manifest'].hash,max_total_bytes=max_trace_bytes)
    result=verify(trace_index=out/'traces/traces.jsonl',trace_sha=envelope['trace_index_sha256'],
        raw_index=raw_dir/'native-feature-index',raw_sha=raw_sha,trace_root=out/'traces',
        raw_root=out/'accepted-raw',checkpoint_sha=original.CHECKPOINTS[seed],
        source_archive_sha=original.SOURCE_ARCHIVE_SHA,lidar_range=lidar_range)
    result.update(kind='v2v4real_original_full_row_independent_post_head_numeric_readback_v1',
        task_id=task.id,seed=seed,cloud_inputs=inputs,transport=transport,runtime=runtime_receipt,
        bootstrap_sha256=source['bootstrap_sha256'],existing_raw_proof_sha256=pin['sha256'])
    (out/'independent-post-head-numeric-receipt.json').write_text(json.dumps(result,sort_keys=True,allow_nan=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('task-id','source-dir','accepted-raw-dir','output','original-source-archive'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-trace-bytes',type=int,required=True);p.add_argument('--max-raw-bytes',type=int,required=True)
    args=vars(p.parse_args());from clearml import Task
    task=Task.get_task(task_id=args.pop('task_id'));print(json.dumps(run(task,**args),sort_keys=True))

if __name__=='__main__':main()
