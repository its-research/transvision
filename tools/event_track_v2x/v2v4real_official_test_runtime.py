"""Verify/install the declared native runtime before downloading experiment data."""
import json
import hashlib
import tarfile
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

DEPENDENCIES=(('clearml','2.1.2','clearml'),('numpy','1.26.4','numpy'),
 ('scipy','1.14.1','scipy'),('PyYAML','6.0.2','yaml'),
 ('spconv-cu126','2.3.8','spconv.pytorch'),('cumm-cu126','0.7.11','cumm'),
 ('open3d','0.19.0','open3d'),('shapely','2.0.7','shapely.geometry'),
 ('einops','0.8.1','einops'),('Cython','3.0.12','Cython'),
 ('pybind11','2.13.6','pybind11'),('ninja','1.11.1.4','ninja'))
PACKAGES=[name+'=='+version for name,version,module in DEPENDENCIES]
PROBE='''import importlib, importlib.metadata, json, sys, re
rows=[]
for name,expected,module in json.loads(sys.argv[1]):
    row=dict(distribution=name,expected=expected,module=module)
    try:
        row['actual']=importlib.metadata.version(name)
        importlib.import_module(module)
        row['imported']=True
    except Exception as exc:
        row['imported']=False;row['exception_type']=type(exc).__name__
        # Persist only allowlisted library basenames, never exception text.
        if 'cannot open shared object file' in str(exc):
            row['missing_shared_libraries']=sorted(set(re.findall(r'lib[A-Za-z0-9_-]+[.]so(?:[.][0-9]+)*',str(exc))))
    rows.append(row)
print(json.dumps(dict(interpreter=sys.executable,dependencies=rows)))
'''


def inspect_runtime():
    result=subprocess.run([sys.executable,'-c',PROBE,json.dumps(DEPENDENCIES)],
        capture_output=True,text=True,timeout=300)
    if result.returncode:raise RuntimeError('native runtime probe process failed')
    # Library import notices can precede the JSON. Never persist raw output.
    value=json.loads(result.stdout.splitlines()[-1])
    if value['interpreter']!=sys.executable:raise ValueError('runtime interpreter differs')
    return value


def require_runtime(value):
    rows=value['dependencies']
    if len(rows)!=len(DEPENDENCIES):raise ValueError('incomplete native runtime attestation')
    for row,(name,version,module) in zip(rows,DEPENDENCIES):
        if (row.get('distribution'),row.get('expected'),row.get('module'))!=(name,version,module):
            raise ValueError('native runtime dependency identity differs')
        if row.get('actual')!=version or row.get('imported') is not True:
            raise ValueError('native runtime version or import differs: '+name)


# Only a missing library actually reported by the fresh import probe can
# trigger an OS package install. Other errors remain failures with evidence.
SYSTEM_PACKAGES={'libX11.so.6':'libx11-6','libGL.so.1':'libgl1',
    'libXext.so.6':'libxext6','libXrender.so.1':'libxrender1',
    'libgomp.so.1':'libgomp1','libSM.so.6':'libsm6','libICE.so.6':'libice6'}


OFFLINE_TASK='1e0730c280e846bebd92cc4de49893e3'
OFFLINE_ARCHIVE_SHA='5ac68c58a292e435a0ee55c98f0bd2720a9b088343afc813c262bdc552cc0e10'
OFFLINE_MANIFEST_SHA='2905061e79777c2858471fa2ee3ccbfaf3cf064fa62429dfa4e1424d34530f82'
OFFLINE_LIBRARIES={'libGL.so.1','libGLdispatch.so.0','libGLX.so.0','libX11.so.6',
                   'libxcb.so.1','libXau.so.6','libXdmcp.so.6','libbsd.so.0','libmd.so.0'}


def unpack_offline_libraries(archive,manifest_path,directory):
    raw=Path(manifest_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=OFFLINE_MANIFEST_SHA:
        raise ValueError('offline library manifest differs')
    if hashlib.sha256(Path(archive).read_bytes()).hexdigest()!=OFFLINE_ARCHIVE_SHA:
        raise ValueError('offline library archive differs')
    manifest=json.loads(raw);rows=manifest['inventory']
    expected={row['path']:dict(bytes=row['bytes'],sha256=row['sha256']) for row in rows}
    if (len(expected)!=len(rows) or set(expected)!={'libgl/'+x for x in OFFLINE_LIBRARIES}
            or manifest.get('libc_and_loader_excluded') is not True):
        raise ValueError('offline library inventory differs')
    expected['manifest.json']=dict(bytes=len(raw),sha256=OFFLINE_MANIFEST_SHA)
    directory=Path(directory);directory.mkdir();seen=set()
    with tarfile.open(archive) as stream:
        for member in stream:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError('unsafe offline library member')
            row=expected[member.name]
            if member.size!=row['bytes']:raise ValueError('offline library size differs')
            payload=stream.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest()!=row['sha256']:
                raise ValueError('offline library bytes differ')
            target=directory/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(payload);seen.add(member.name)
    if seen!=set(expected):raise ValueError('incomplete offline library inventory')
    return dict(task_id=OFFLINE_TASK,archive_sha256=OFFLINE_ARCHIVE_SHA,
        manifest_sha256=OFFLINE_MANIFEST_SHA,inventory=rows,libc_and_loader_excluded=True,
        prior_SPD_runtime_acceptance_inherited=False)


def materialize_offline_libraries(root):
    from clearml import Task
    # Packages were installed before this import; the wrapper provides the
    # same strict registered-host/authentication/size/hash transport boundary.
    from tools.event_track_v2x.run_v2v4real_official_test_gpu_shards import download
    owner=Task.get_task(task_id=OFFLINE_TASK)
    archive=Path(root)/'offline-libraries.tar.gz';manifest=Path(root)/'offline-libraries-manifest.json'
    download(owner,'libgl',OFFLINE_ARCHIVE_SHA,archive)
    download(owner,'manifest',OFFLINE_MANIFEST_SHA,manifest)
    directory=Path(root)/'offline-libraries'
    evidence=unpack_offline_libraries(archive,manifest,directory)
    os.environ['LD_LIBRARY_PATH']=str(directory/'libgl')+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
    return evidence


def repair_system_dependencies(value,root):
    missing={lib for row in value['dependencies'] if row.get('imported') is False
             for lib in row.get('missing_shared_libraries',[])}
    if not missing or not missing.issubset(OFFLINE_LIBRARIES):return value,[]
    if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):
        return value,[]
    evidence=materialize_offline_libraries(root)
    return inspect_runtime(),[evidence]


def prepare_runtime(task,root):
    start=time.monotonic();before=inspect_runtime();installed=False
    try:require_runtime(before)
    except ValueError:
        installed=True
        print(json.dumps(dict(stage='official_test_runtime_install',ETA_seconds=None,
            eta_scope='dependency installation only; completion time unknown')),flush=True)
        # Captured pip output may contain private index credentials. Do not
        # publish or print it, including on failure; report only return code.
        result=subprocess.run([sys.executable,'-m','pip','install','--no-input',
            '--disable-pip-version-check',*PACKAGES],capture_output=True,timeout=1800)
        if result.returncode:raise RuntimeError('native runtime dependency installation failed, exit '+str(result.returncode))
    after=inspect_runtime();python_after=after
    after,offline_libraries=repair_system_dependencies(after,root)
    receipt=dict(kind='v2v4real_native_runtime_dependency_attestation_v1',before=before,
        after=after,after_Python_install=python_after,system_packages=[],offline_libraries=offline_libraries,explicit_install_executed=installed,elapsed_seconds=time.monotonic()-start,
        same_interpreter_as_rank_children=True,model_or_inference_source_changed=False,
        raw_package_manager_output_persisted=False,experiment_accepted=False)
    path=Path(root)/'runtime-dependencies.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    task.upload_artifact('runtime-dependencies',artifact_object=str(path),wait_on_upload=True)
    require_runtime(after)
    return receipt
