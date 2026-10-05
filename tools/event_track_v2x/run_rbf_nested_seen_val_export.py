"""Frozen nested detector seen-val cache producer, four disjoint shards/side."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import requests
from clearml import Task
from clearml.backend_api.session import Session
from run_cooptrack_a100 import validate_archive


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def fetch(spec, output):
    task = Task.get_task(task_id=spec['task'])
    if str(task.status) != 'completed':
        raise ValueError('required input task is not completed: ' + spec['task'])
    artifact = task.artifacts[spec['key']]
    if artifact.hash != spec['sha256'] or artifact.size != spec['bytes']:
        raise ValueError('registered artifact identity differs: ' + spec['key'])
    url = artifact.url.replace('10.100.35.118:8081', '10.100.34.118:8081')
    started, count, last = time.monotonic(), 0, 0
    h = hashlib.sha256()
    with requests.get(url, headers={'Authorization': 'Bearer ' + Session().token},
                      timeout=(10, 120), stream=True) as response:
        if response.status_code != 200:
            raise RuntimeError('input HTTP status ' + str(response.status_code))
        with Path(output).open('xb') as stream:
            for block in response.iter_content(1024**2):
                stream.write(block)
                h.update(block)
                count += len(block)
                if count > spec['bytes']:
                    raise ValueError('input byte count exceeded registered size')
                if time.monotonic() - last > 30:
                    print(json.dumps(dict(stage='input_byte_transfer', key=spec['key'],
                        completed_bytes=count, total_bytes=spec['bytes'],
                        ETA_seconds=(time.monotonic()-started)*(spec['bytes']-count)/count,
                        ETA_scope='current input transfer only')), flush=True)
                    last = time.monotonic()
    if count != spec['bytes'] or h.hexdigest() != spec['sha256']:
        raise ValueError('complete input bytes differ: ' + spec['key'])
    return Path(output)


def spec(task_id, key, digest):
    task = Task.get_task(task_id=task_id)
    artifact = task.artifacts[key]
    if str(task.status) != 'completed' or artifact.hash != digest:
        raise ValueError('registered runtime identity differs')
    return dict(task=task_id, key=key, sha256=digest, bytes=artifact.size)


def publish(task, key, path):
    if not task.upload_artifact(key, artifact_object=Path(path), wait_on_upload=True):
        raise RuntimeError('upload failed: ' + key)
    task.reload()
    artifact = task.artifacts[key]
    if artifact.hash != sha(path) or artifact.size != Path(path).stat().st_size:
        raise ValueError('uploaded artifact metadata differs')


def main():
    task = Task.init(project_name='Thesis/Recover-Before-Fuse/Training',
        task_name='RBF nested detector seen-val raw cache', reuse_last_task_id=False,
        auto_connect_frameworks=False, auto_connect_arg_parser=False,
        output_uri='http://10.100.34.118:8081')
    parameters = task.get_parameters()
    recipe = json.loads(parameters['General/recipe'])
    if hashlib.sha256(canonical(recipe)).hexdigest() != parameters['General/recipe_sha256']:
        raise ValueError('frozen execution recipe differs')
    side = recipe['side']
    if recipe['world_size'] != 4 or side not in ('vehicle-side', 'infrastructure-side'):
        raise ValueError('four CUDA shards of one side required')
    root = Path('/rbf-nested-seen-val-cache')
    if any(p.exists() for p in (root, Path('/opt/cooptrack'), Path('/workspace/CoopTrack'), Path('/entrypoints'))):
        raise FileExistsError('fresh frozen runtime required')
    root.mkdir()
    code = Path(__file__).resolve().parent
    for name, item in recipe['source_inventory'].items():
        if sha(code/name) != item['sha256'] or (code/name).stat().st_size != item['bytes']:
            raise ValueError('execution source differs: ' + name)
    try:
        runtime_manifest = fetch(recipe['runtime_manifest'], root/'runtime-manifest.json')
        inventory = {v['path']: v for v in json.loads(runtime_manifest.read_bytes())['inventory']}
        for key, prefixes, links in (
            ('runtime', ['opt/cooptrack', 'usr/local/cuda-11.8/targets/x86_64-linux/lib'], True),
            ('source', ['workspace/CoopTrack', 'entrypoints'], False)):
            path = fetch(spec(recipe['runtime_manifest']['task'], key,
                              inventory[key+'.tar.gz']['sha256']), root/(key+'.tar.gz'))
            validate_archive(path, prefixes, allow_links=links)
            subprocess.run(['tar', '--no-same-owner', '-xzf', str(path), '-C', '/'], check=True)
        gl = fetch(recipe['libgl'], root/'libgl.tar.gz')
        validate_archive(gl, ['libgl', 'manifest.json'], allow_links=True)
        gl_root = root/'gl-runtime'
        gl_root.mkdir()
        subprocess.run(['tar', '--no-same-owner', '-xzf', str(gl), '-C', str(gl_root)], check=True)
        package_manifest = json.loads(fetch(recipe['input_manifest'], root/'package-manifest.json').read_bytes())
        if (package_manifest['kind'] != 'eventtrack_validation_raw_cache_package_v1'
            or package_manifest['gt_payloads_included'] or package_manifest['test_payloads_included']
            or not package_manifest['val_payloads_included']):
            raise ValueError('val input boundary differs')
        archive = fetch(recipe['inputs'], root/'inputs.tar.gz')
        validate_archive(archive, ['inputs', 'run_cache.py', 'cache_primitives.py'])
        subprocess.run(['tar', '--no-same-owner', '-xzf', str(archive), '-C', str(root)], check=True)
        if sha(root/'inputs/input-manifest.json') != recipe['input_dataset_sha256']:
            raise ValueError('input manifest bytes differ')
        inputs = json.loads((root/'inputs/input-manifest.json').read_bytes())
        if (inputs['kind'] != 'eventtrack_validation_image_pose_inputs_v1'
            or inputs['gt_payloads_in_package'] or inputs['test_payloads_read']
            or not inputs['val_payloads_read'] or len(inputs['validation_sequences']) != 21
            or inputs['frames'] != {'vehicle-side':3748, 'infrastructure-side':3441}):
            raise ValueError('input schema/scope differs')
        for name, digest in package_manifest['scripts'].items():
            if sha(root/name) != digest:
                raise ValueError('registered package source differs')
        weights = root/'weights'
        weights.mkdir()
        collection = json.loads(fetch(recipe['collection'], weights/'collection-receipt.json').read_bytes())
        if (collection['training_task_id'] != recipe['training_task_id']
            or collection['seed'] != recipe['seed'] or collection['cohort'] != 'nested-detector-fit'):
            raise ValueError('nested training collection differs')
        for suffix in ('completion', 'final-checkpoint', 'launch-receipt.json', 'optimizer-startup.json', 'detector.py'):
            key = side+'-'+suffix
            item = collection['inventory'][key]
            fetch(dict(task=recipe['training_task_id'], key=key, **item), weights/key)
        binding = collection['bindings'][side]
        train_sequences = sorted(binding['fit_sequence_ids'] + binding['calibration_sequence_ids']
                                 + binding['held_out_sequence_ids'])
        if len(train_sequences) != 46 or len(set(train_sequences)) != 46:
            raise ValueError('frozen original train sequence binding differs')
        if set(train_sequences) & set(inputs['validation_sequences']):
            raise ValueError('train/val sequence membership overlaps')
        (weights/'training-sequences.json').write_bytes(canonical(train_sequences))
        appearance = fetch(spec(recipe['runtime_manifest']['task'], 'pretrained',
            '0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a'), root/'appearance.pth')
        env = dict(os.environ, PATH='/opt/cooptrack/bin:'+os.environ.get('PATH',''),
            PYTHONPATH='/workspace/CoopTrack', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', NVIDIA_TF32_OVERRIDE='0',
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',
            LD_LIBRARY_PATH=str(gl_root/'libgl')+':/opt/cooptrack/lib:/opt/cooptrack/lib/python3.8/site-packages/torch/lib:'
                            '/usr/local/cuda-11.8/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64')
        python = '/opt/cooptrack/bin/python'
        probe = """import json, torch
from mmcv import Config
import mmdet3d, projects.mmdet3d_plugin
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
assert torch.cuda.device_count()==4
torch.manual_seed(1337)
a=torch.randn(64,64); b=torch.randn(64,64)
assert torch.allclose(a@b,(a.cuda()@b.cuda()).cpu(),atol=1e-4,rtol=1e-4)
print(json.dumps(dict(torch_version=torch.__version__,devices=[torch.cuda.get_device_name(i) for i in range(4)],
compute_capabilities=[list(torch.cuda.get_device_capability(i)) for i in range(4)],
compiled_architectures=torch.cuda.get_arch_list(),TF32_enabled=False,matmul_admitted=True,
complete_detector_numerical_acceptance=False)))
"""
        admitted = json.loads(subprocess.check_output([python,'-c',probe], env=env, cwd='/workspace/CoopTrack', text=True).splitlines()[-1])
        (root/'runtime-admission.json').write_text(json.dumps(admitted,indent=2)+'\n')
        publish(task,'runtime-admission',root/'runtime-admission.json')
        print(json.dumps(dict(stage='runtime_and_frozen_input_admitted',seed=recipe['seed'],side=side,
            actual_devices=admitted['devices'],total_frames=inputs['frames'][side],ETA_seconds=None,
            ETA_scope='unknown until actual shard frame progress')),flush=True)

        def shard(index):
            output = root/('shard-'+str(index))
            command = [python,str(code/'run_seen_val_cache.py')]
            options = dict(inputs=root/'inputs',upstream='/workspace/CoopTrack',
                checkpoint=weights/(side+'-final-checkpoint'),
                **{'checkpoint-sha256':collection['inventory'][side+'-final-checkpoint']['sha256'],
                'appearance-checkpoint':appearance,'output':output,'training-config':weights/(side+'-detector.py'),
                'training-launch':weights/(side+'-launch-receipt.json'),
                'training-startup':weights/(side+'-optimizer-startup.json'),
                'training-completion':weights/(side+'-completion'),'training-cohort':'nested-detector-fit',
                'training-sequences-json':weights/'training-sequences.json',
                'side':side,'shard-index':index,'shard-count':4})
            for key,value in options.items():
                command.extend(['--'+key,str(value)])
            log = root/('shard-'+str(index)+'.log')
            child_env = dict(env,CUDA_VISIBLE_DEVICES=str(index))
            with log.open('x') as stream:
                child = subprocess.Popen(command,cwd='/workspace/CoopTrack',env=child_env,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in child.stdout:
                    stream.write(line)
                    stream.flush()
                    print('shard'+str(index)+' '+line.rstrip(),flush=True)
                returncode = child.wait()
            publish(task,'shard-'+str(index)+'-log',log)
            if returncode:
                raise RuntimeError('actual raw forward failed in shard '+str(index)+': '+str(returncode))
            manifest_path = output/'raw-cache-manifest.json'
            result = json.loads(manifest_path.read_bytes())
            if (result['TF32_enabled'] or not result['weights_unchanged']
                or result['raw_head_frames_verified'] != result['frame_count']
                or result['legacy_placeholder_count'] != 0 or result['training_binding'] != binding
                or result['checkpoint_sha256'] != recipe['checkpoint_sha256']
                or result['shard_count'] != 4 or result['shard_index'] != index):
                raise ValueError('completed shard binding differs')
            verification = json.loads(subprocess.check_output([python,str(code/'spd_cache_primitives.py'),
                str(output),'--inputs',str(root/'inputs')],env=child_env,text=True))
            if (not verification['all_payloads_read'] or not verification['expected_frame_coverage_verified']
                or verification['frames_verified'] != result['frame_count']
                or verification['manifest_sha256'] != sha(manifest_path)):
                raise ValueError('independent CPU payload/coverage verification differs')
            archive_path = root/('shard-'+str(index)+'.tar.gz')
            subprocess.run(['tar','-czf',str(archive_path),'-C',str(output),'.'],check=True)
            publish(task,'shard-'+str(index)+'-manifest',manifest_path)
            publish(task,'shard-'+str(index)+'-raw-cache',archive_path)
            return dict(shard_index=index,frames=result['frame_count'],sequences=result['sequences'],
                        archive_sha256=sha(archive_path),CPU_payload_readback=verification,
                        actual_cuda_device=result['actual_cuda_device'],TF32_enabled=False)
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(shard,range(4)))
        scenes = [seq for row in outcomes for seq in row['sequences']]
        if sorted(scenes) != sorted(inputs['validation_sequences']) or sum(row['frames'] for row in outcomes) != inputs['frames'][side]:
            raise ValueError('four-shard complete side coverage differs')
        receipt = dict(kind='rbf_nested_detector_seen_val_raw_export_GPU4_candidate_v1',
            seed=recipe['seed'],side=side,recipe=recipe,outcomes=outcomes,
            total_frames=inputs['frames'][side],whole_side_coverage_verified=True,
            no_GT_in_producer=True,actual_CUDA_forward_completed=True,
            local_independent_cloud_byte_readback_accepted=False,
            complete_detector_numerical_acceptance=False,full_online_RBF_accepted=False,
            paper_performance_complete=False,ETA_seconds=None,ETA_scope='independent acceptance pending')
        (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        publish(task,'receipt',root/'receipt.json')
        task.close()
    except BaseException as error:
        failure = dict(seed=recipe['seed'],side=side,exception_type=type(error).__name__,message=str(error),
            actual_forward_complete=False,independent_acceptance=False,recipe=recipe)
        (root/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        publish(task,'failure',root/'failure.json')
        raise


if __name__ == '__main__':
    main()
