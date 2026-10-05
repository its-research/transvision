#!/usr/bin/env python3
"""Execute unchanged canonical fit and independent reconstruction on admitted Linux CPU.

This runner is not a bootstrap or an experiment acceptance receipt. Its inputs
must first be staged from independently admitted ClearML artifacts.
"""
import argparse,hashlib,json,os,platform,subprocess,sys
from pathlib import Path
from importlib.metadata import version,distribution
SOURCE_MANIFEST_SHA='2d01f5d0bfc94c4837eedd574ef6b2176f8211dc428352b49d27573a8533c517'
METADATA_MANIFEST_SHA='907e5a0986daa4c46a747d0f7344d5a5f3edf0f7dd1ed718fb37c4710ae28492'
FITTER_SHA='9911baf2b64e76ce1af98d8ca4c0648012f3924fc81192d59491da48056db241'
VERIFIER_SHA='72177e04f3d737d68ab4ea61584d2743f39568f0bf7a9977b17a7bf3a7d001b7'
RUNTIME_TASK='969e56c034714dfab10e16471d68d7e1'
PACKAGE_RECORDS={'numpy':'84c76366753383a6e63fff7d93c58ff964d7a2c4a894b73dd12d8f011c8ab45d','scipy':'4c78b1b9fce1b1577a1d8d9adf2e8e0b139a5708085409ec7168403046fea4b3','clearml':'a65926454fcc1c95e5a5d669a8d4894882dd54d0660ae15504a30d5c5de1632c'}
RAW_SOURCE='5d218685995548948ccb37b92dd337de'
RAW_ACCEPTOR='fefd651a33f6dabf56ea0612eebc80673af5e5d50b7466d0e928644c7569f878'
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def inventory(root,rows):
 seen=set()
 for row in rows:
  name=Path(row['path']);need(not name.is_absolute() and '..' not in name.parts and row['path'] not in seen,'unsafe inventory')
  p=root/name;need(not p.is_symlink() and p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'inventory file changed')
  seen.add(row['path'])
 need(not root.is_symlink() and not any(p.is_symlink() for p in root.rglob('*')),'symlinked inventory root')
 actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
 need(actual==seen,'extra or missing inventory file')
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('source-root','source-manifest','metadata-root','metadata-manifest','supervision-root','fit-inputs','raw-readback','output'):parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--raw-readback-sha256',required=True);parser.add_argument('--fold',type=int,choices=(0,1),required=True)
 a=parser.parse_args();need(not a.output.exists(),'output exists; preserve prior attempt')
 need(platform.system()=='Linux' and platform.machine()=='x86_64' and sys.version_info[:2]==(3,12),'CPU interpreter/platform differs')
 need(platform.python_version()=='3.12.3','admitted Linux interpreter version differs')
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU runtime must hide CUDA')
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):need(os.environ.get(key)=='1','CPU thread policy differs')
 versions={key:version(key) for key in ('numpy','scipy','clearml')};need(versions=={'numpy':'1.26.4','scipy':'1.14.1','clearml':'2.1.5'},'Linux126 package versions differ')
 records={}
 for key,expected in PACKAGE_RECORDS.items():
  record=distribution(key).read_text('RECORD');need(record is not None,'missing package installation record')
  records[key]=hashlib.sha256(record.encode()).hexdigest();need(records[key]==expected,'package installation record differs')
 need(sha(a.source_manifest)==SOURCE_MANIFEST_SHA and sha(a.metadata_manifest)==METADATA_MANIFEST_SHA,'cloud source/metadata manifest differs')
 s=json.loads(a.source_manifest.read_bytes());m=json.loads(a.metadata_manifest.read_bytes());inventory(a.source_root,s['inventory']);inventory(a.metadata_root,m['inventory'])
 tool=a.source_root/'tools/event_track_v2x';fitter=tool/'fit_spd_canonical_oof_calibration_runtime_v3_candidate.py';verifier=tool/'verify_spd_canonical_calibration_examples_runtime_v3_candidate.py';need(sha(fitter)==FITTER_SHA and sha(verifier)==VERIFIER_SHA,'fitter/oracle source differs')
 need(not a.raw_readback.is_symlink() and sha(a.raw_readback)==a.raw_readback_sha256,'fit raw admission bytes differ')
 r=json.loads(a.raw_readback.read_bytes());need(r.get('kind')=='spd_canonical_oof_fit_feature_raw_pose_v2_cloud_independent_content_readback' and r.get('status')=='all_cloud_bytes_frames_arrays_raw_poses_verified' and r.get('fold_id')==a.fold and r.get('source_task_id')==RAW_SOURCE and r.get('acceptor_sha256')==RAW_ACCEPTOR and r.get('held_out_selection_scoring_eligible') is False,'complete fit raw acceptance missing')
 from clearml import Task
 probe=Task.get_task(task_id=RUNTIME_TASK);need(probe.status=='completed' and probe.artifacts['runtime-probe'].hash=='2fc885c76f46de4ba7756aec537daae960dc2ca043802ac237e1bce77395b62a','admitted Linux CPU probe registration differs')
 # Preserve original audit ROOT unchanged. The isolated bootstrap must stage
 # the exact five admitted manifests there, without prediction/GT payloads.
 absolute=Path('/Volumes/Data/test/recover-before-fuse/artifacts/spd-canonical-oof-heldout-inference-inputs-20260930')
 for fold in range(5):need(sha(absolute/f'fold-{fold}/input-manifest.json')==sha(a.metadata_root/f'heldout-manifests/fold-{fold}/input-manifest.json'),'original audit manifest mount differs')
 package=a.supervision_root/'package';converted=a.supervision_root/'converted';byte_freeze=a.metadata_root/f'byte-freezes/fold-{a.fold}.json';poses=a.metadata_root/'raw-poses';held=a.metadata_root/f'heldout-manifests/fold-{a.fold}'
 common=['--package',str(package),'--converted',str(converted),'--fit-inputs',str(a.fit_inputs),'--heldout-inputs',str(held),'--raw-readback',str(a.raw_readback),'--raw-readback-sha256',a.raw_readback_sha256,'--byte-freeze',str(byte_freeze),'--raw-poses',str(poses)]
 # Child import path contains the admitted source only; no ambient repository.
 env=dict(os.environ);env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
 runner="import runpy,sys;sys.path[:0]=[sys.argv.pop(1),sys.argv.pop(1)];runpy.run_path(sys.argv.pop(1),run_name='__main__')"
 def execute(script,args):subprocess.run([sys.executable,'-I','-c',runner,str(a.source_root),str(tool),str(script),*args],env=env,check=True)
 a.output.mkdir(parents=True)
 runtime=dict(kind='canonical_calibration_linux126_actual_cpu_execution_identity',interpreter=sys.version,executable=sys.executable,platform=platform.platform(),versions=versions,package_record_sha256=records,cuda_visible_devices='',runtime_probe_task_id=RUNTIME_TASK,runner_sha256=sha(Path(__file__)),source_manifest_sha256=sha(a.source_manifest),metadata_manifest_sha256=sha(a.metadata_manifest),raw_fit_readback_sha256=sha(a.raw_readback),fold_id=a.fold,formal_v2_ready=False,paper_eligible=False)
 (a.output/'runtime-identity.json').write_text(json.dumps(runtime,indent=2)+'\n')
 print('CALIBRATION_PHASE fit_collection_and_optimization overall_eta=unknown',flush=True)
 candidate=a.output/'candidate';execute(fitter,common+['--output',str(candidate)])
 c=candidate/'calibration.json';receipt=a.output/'independent-examples-and-parameters.json'
 print('CALIBRATION_PHASE independent_full_example_and_parameter_readback overall_eta=unknown',flush=True)
 execute(verifier,common+['--calibration',str(c),'--calibration-sha256',sha(c),'--receipt',str(receipt)])
 proof=json.loads(receipt.read_bytes());need(proof['fold_id']==a.fold and proof['raw_GT_examples_independently_reconstructed'] is True and proof['independent_parameters_recomputed'] is True and proof['held_out_GT_used_for_fitting'] is False,'independent calibration result incomplete')
 print('CALIBRATION_CPU_CANDIDATE_INDEPENDENT_READBACK_COMPLETE '+sha(receipt),flush=True)
if __name__=='__main__':main()
