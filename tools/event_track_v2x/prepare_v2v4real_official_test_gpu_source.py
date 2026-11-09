#!/usr/bin/env python3
"""Build a self-contained ClearML script with exact, minimal execution sources."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

FILES=('run_v2v4real_official_test_raw_features.py','merge_v2v4real_official_test_raw_shards.py',
       'run_v2v4real_official_test_gpu_shards.py','v2v4real_official_test_runtime.py')


def build_bootstrap(directory):
    directory=Path(directory);files={}
    for name in FILES:
        p=directory/name
        if not p.is_file() or p.is_symlink():raise ValueError('ordinary execution sources required')
        raw=p.read_bytes();compile(raw,str(p),'exec')
        files['tools/event_track_v2x/'+name]=dict(sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw),base64=base64.b64encode(raw).decode())
    identity=hashlib.sha256(json.dumps({k:{x:v[x] for x in ('sha256','bytes')} for k,v in files.items()},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    script='''"""Exact source-bound official-test raw GPU candidate; no metric claims."""
import base64, hashlib, json, sys
from pathlib import Path
from clearml import Task
FILES = '''+repr(files)+'''
BUNDLE_SHA256 = '''+repr(identity)+'''
def main():
    task=Task.init(project_name='Thesis/Recover-Before-Fuse/Inference',task_name='V2V4Real official-test raw shards',
        reuse_last_task_id=False,auto_connect_frameworks=False,auto_connect_arg_parser=False)
    p=task.get_parameters()
    if p.get('General/execution_bundle_sha256')!=BUNDLE_SHA256:raise ValueError('execution bundle identity differs')
    root=Path('/tmp')/('rbf-official-test-code-'+task.id);root.mkdir()
    for name,row in FILES.items():
        raw=base64.b64decode(row['base64'],validate=True)
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('embedded source bytes differ')
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    sys.path.insert(0,str(root))
    from tools.event_track_v2x.v2v4real_official_test_runtime import prepare_runtime
    prepare_runtime(task,root)
    from tools.event_track_v2x.run_v2v4real_official_test_gpu_shards import main as execute
    execute(task=task)
if __name__=='__main__':main()
'''
    compile(script,'official_test_bootstrap.py','exec')
    return script,dict(kind='v2v4real_official_test_minimal_GPU_source_bundle_v1',execution_bundle_sha256=identity,
        bootstrap_sha256=hashlib.sha256(script.encode()).hexdigest(),sources={k:{x:v[x] for x in ('sha256','bytes')} for k,v in files.items()},
        original_inference_core_changed=False,actual_GPU_execution_verified=False,independent_raw_acceptance=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();script,manifest=build_bootstrap(Path(__file__).parent)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'bootstrap.py').write_text(script)
    (a.output/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':main()
