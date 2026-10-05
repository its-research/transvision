"""ClearML full-train refit candidate, using registered original input bytes."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import requests
from clearml import Task
from clearml.backend_api.session import Session

RUNNER_BASE64 = '__RUNNER_BASE64__'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(spec, path):
    task = Task.get_task(task_id=spec['task'])
    assert str(task.status) == 'completed'
    artifact = task.artifacts[spec['key']]
    assert artifact.hash == spec['sha256'] and artifact.size == spec['bytes']
    url = artifact.url.replace('10.100.35.118:8081', '10.100.34.118:8081')
    completed = 0
    started = time.monotonic()
    headers = {'Authorization': 'Bearer ' + Session().token}
    with requests.get(url, headers=headers, timeout=(10, 120), stream=True) as response:
        if response.status_code != 200:
            raise RuntimeError('input HTTP status ' + str(response.status_code))
        with path.open('xb') as stream:
            for chunk in response.iter_content(1024**2):
                stream.write(chunk)
                completed += len(chunk)
                if completed % (32 * 1024**2) < len(chunk):
                    print(json.dumps(dict(stage='registered_input_transfer', artifact=spec['key'],
                        completed_bytes=completed,total_bytes=spec['bytes'],
                        ETA_seconds=(time.monotonic()-started)*(spec['bytes']-completed)/completed,
                        ETA_scope='current input transfer only')),flush=True)
    assert path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256']


def unpack(path, output):
    output.mkdir()
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        assert len({m.name for m in members}) == len(members)
        assert all(not m.issym() and not m.islnk() and not m.name.startswith('/')
                   and '..' not in Path(m.name).parts and (m.isfile() or m.isdir()) for m in members)
        archive.extractall(output, filter='data')


def main():
    task = Task.init(project_name='Thesis/Recover-Before-Fuse/Training',
        task_name='RBF nested-selected all-class full-train refit',reuse_last_task_id=False,
        auto_connect_frameworks=False,auto_connect_arg_parser=False,output_uri='http://10.100.34.118:8081')
    params = task.get_parameters()
    recipe = json.loads(params['General/recipe'])
    assert hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest() == params['General/recipe_sha256']
    import torch
    assert torch.cuda.device_count() == recipe['world_size']
    assert recipe['world_size'] in (4,8) and recipe['seed'] in (1337,2027,3407)
    base = Path('rbf-all-class-final-refit')
    base.mkdir()
    (base/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    runner = base/'runner.py'
    runner.write_bytes(base64.b64decode(RUNNER_BASE64))
    assert sha(runner) == recipe['runner_sha256']
    for key in ('source','rows','manifest','checkpoint','weights_archive'):
        fetch(recipe[key],base/key)
    unpack(base/'source',base/'source-unpack')
    (base/'source').unlink()
    (base/'source-unpack').rename(base/'source')
    unpack(base/'rows',base/'rows-unpack')
    (base/'rows').unlink()
    (base/'rows-unpack').rename(base/'rows')
    unpack(base/'weights_archive',base/'selected-training')
    (base/'checkpoint').rename(base/'selected-checkpoint.json')
    selected = json.loads((base/'selected-checkpoint.json').read_bytes())
    plans = list((base/'selected-training').rglob('plan.json'))
    assert len(plans) == 1 and sha(plans[0]) == selected['plan_sha256']
    plan = json.loads(plans[0].read_bytes())
    assert plan['selection'] == 'minimum_train_holdout_macro_row_surrogate'
    assert plan['config'] == dict(recipe['fit_config'],epochs=10)
    epochs = list((base/'selected-training').rglob('epochs.json'))
    assert len(epochs) == 1
    history = json.loads(epochs[0].read_bytes())
    assert [x['epoch'] for x in history] == list(range(1,11))
    assert min(history,key=lambda x:x['holdout']['macro_row_surrogate'])['epoch'] == recipe['fit_config']['epochs']
    assert sha(base/'rows/rows/manifest.json') == recipe['manifest']['sha256']
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',NVIDIA_TF32_OVERRIDE='0',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0')
    command = [sys.executable,'-m','torch.distributed.run','--standalone',
               '--nproc_per_node',str(recipe['world_size']),str(runner),
               '--base',str(base),'--recipe',str(base/'recipe.json')]
    print(json.dumps(dict(stage='full_train_refit_input_admitted',seed=recipe['seed'],
        world_size=recipe['world_size'],epochs=recipe['fit_config']['epochs'],
        ETA_seconds=None,ETA_scope='runtime admission and input audit pending')),flush=True)
    result = subprocess.run(command)
    if result.returncode:
        failure = base/'failure.json'
        failure.write_text(json.dumps(dict(seed=recipe['seed'],returncode=result.returncode,
            recipe=recipe,completed=False,independent_acceptance=False),indent=2)+'\n')
        task.upload_artifact('failure',artifact_object=failure,wait_on_upload=True)
        raise RuntimeError('refit process failed: '+str(result.returncode))
    training = base/'training'
    output_archive = base/'identity-training.tar.gz'
    with tarfile.open(output_archive,'w:gz') as archive:
        archive.add(training,arcname='training')
    files = {'identity-training':output_archive,'recipe':base/'recipe.json',
        'runtime-admission':training/'runtime-admission.json','plan':training/'plan.json',
        'receipt':training/'receipt.json',
        'checkpoint':training/f"seed-{recipe['seed']}/checkpoint.json"}
    for name,path in files.items():
        assert task.upload_artifact(name,artifact_object=path,wait_on_upload=True)
    print(json.dumps(dict(stage='refit_execution_and_artifact_upload_completed',seed=recipe['seed'],
        independent_checkpoint_acceptance=False,paper_performance_complete=False,
        ETA_seconds=None,ETA_scope='independent acceptance pending')),flush=True)
    task.close()


if __name__ == '__main__':
    main()
