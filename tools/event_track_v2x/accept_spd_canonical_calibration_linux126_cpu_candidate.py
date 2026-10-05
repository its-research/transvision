#!/usr/bin/env python3
"""Independent cloud byte/full raw-GT/numerical readback of CPU calibration.

Uses the unchanged frozen example oracle against already accepted local fit raw,
and preserves its cross-version numeric tolerance. No V2/paper gate is opened.
"""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from spd_registered_artifact_chunked_candidate import download_registered_artifact
R=Path('/Volumes/Data/test/recover-before-fuse')
SOURCE=R/'source-freezes/spd-canonical-calibration-runtime-v3-source-bound-closure-20261001'
BOOTSTRAP_SHA='8d8b5f98a0fcba9e576dabf80dca31fa4752458cce87a7b79972f7bf7cf9546d'
RUNNER_SHA='f8b9fb992d0830a1a890c70f2a05ea80c7044a93ac2a4a13d41a76b8b1424455'
EXPECTED_VERSIONS={'numpy':'1.26.4','scipy':'1.14.1','clearml':'2.1.5'}
EXPECTED_RECORDS={'numpy':'84c76366753383a6e63fff7d93c58ff964d7a2c4a894b73dd12d8f011c8ab45d','scipy':'4c78b1b9fce1b1577a1d8d9adf2e8e0b139a5708085409ec7168403046fea4b3','clearml':'a65926454fcc1c95e5a5d669a8d4894882dd54d0660ae15504a30d5c5de1632c'}
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def validate_runtime(runtime,fold,raw_sha):
 need(runtime.get('kind')=='canonical_calibration_linux126_actual_cpu_execution_identity' and runtime.get('fold_id')==fold and runtime.get('raw_fit_readback_sha256')==raw_sha and runtime.get('runner_sha256')==RUNNER_SHA and runtime.get('runtime_probe_task_id')=='969e56c034714dfab10e16471d68d7e1','runtime source/fold binding differs')
 need(runtime.get('versions')==EXPECTED_VERSIONS and runtime.get('package_record_sha256')==EXPECTED_RECORDS and runtime.get('cuda_visible_devices')=='' and runtime.get('interpreter','').startswith('3.12.3 ') and runtime.get('platform','').startswith('Linux-'),'actual CPU interpreter/package identity differs')
 need(runtime.get('source_manifest_sha256')=='2d01f5d0bfc94c4837eedd574ef6b2176f8211dc428352b49d27573a8533c517' and runtime.get('metadata_manifest_sha256')=='907e5a0986daa4c46a747d0f7344d5a5f3edf0f7dd1ed718fb37c4710ae28492','runtime assets differ')
 need(runtime.get('formal_v2_ready') is False and runtime.get('paper_eligible') is False,'runtime overclaims eligibility')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('dispatch','raw-readback','fit-inputs','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--task-id',required=True);a=p.parse_args();need(not a.output.exists(),'preserve existing partial/result readback; fresh output required')
 dispatch=json.loads(a.dispatch.read_bytes());fold=dispatch['fold_id'];need(type(fold) is int and fold in (0,1) and dispatch['task_id']==a.task_id and dispatch['bootstrap_sha256']==BOOTSTRAP_SHA and dispatch['raw_fit_readback_sha256']==sha(a.raw_readback),'local dispatch/source binding differs')
 raw=json.loads(a.raw_readback.read_bytes());need(raw['status']=='all_cloud_bytes_frames_arrays_raw_poses_verified' and raw['fold_id']==fold and raw['task_id']==dispatch['raw_fit_task_id'],'fit raw acceptance differs')
 from clearml import Task
 task=Task.get_task(task_id=a.task_id);need(task.status=='completed','calibration task not completed')
 need(hashlib.sha256(task.data.script.diff.encode()).hexdigest()==BOOTSTRAP_SHA and task.data.container.image=='gitlab.zhht.ai.com:5000/aitech/ultralytics_rfdetr@sha256:e5b249d993f9675971b328152aab4e1023930f3480a80dd9cfd6e9620b208bfb','executed bootstrap/image differs')
 need('L40S' in (task.data.last_worker or ''),'CPU job did not run on admitted L40S listener')
 params=task.get_parameters();need(params['General/device']=='cpu' and params['General/fold_id']==fold and params['General/seed']==1337 and params['General/runner_sha256']==RUNNER_SHA and params['General/raw_fit_task_id']==raw['task_id'] and params['General/raw_fit_readback_sha256']==sha(a.raw_readback) and params['General/raw_fit_readback_text']==a.raw_readback.read_text(),'executed CPU configuration differs')
 expected={'calibration','candidate-receipt','runtime-identity','independent-examples-and-parameters','execution-summary'}|{side+'-'+name+'-examples' for side in ('vehicle-side','infrastructure-side') for name in ('car','bicycle','pedestrian')}
 need(set(task.artifacts)==expected,'missing/extra CPU artifacts')
 a.output.mkdir(parents=True);records={}
 for name in sorted(expected):
  item=task.artifacts[name];suffix='.npz' if name.endswith('-examples') else '.json';target=a.output/(name+suffix);download_registered_artifact(item,item.hash,item.size,target);records[name]=dict(path=target.name,sha256=sha(target),bytes=target.stat().st_size)
 runtime=json.loads((a.output/'runtime-identity.json').read_bytes());validate_runtime(runtime,fold,sha(a.raw_readback))
 summary=json.loads((a.output/'execution-summary.json').read_bytes())
 need(summary['kind']=='canonical_calibration_linux126_cpu_fitted_and_job_local_reconstructed_summary' and summary['fold_id']==fold and summary['raw_fit_task_id']==raw['task_id'] and summary['raw_fit_readback_sha256']==sha(a.raw_readback) and summary['runner_sha256']==RUNNER_SHA and summary['source_task_id']=='641448cbbb83421ab6c63b8537c3bf99' and summary['metadata_task_id']=='c5ca5855aed84f6e83ceac022578f5ed' and summary['supervision_task_id']=={0:'9459a23705a24c1c89a2fa7613268815',1:'e554bf277e58439382e30900889af509'}[fold],'CPU summary source/fold differs')
 need(summary['artifacts']=={name:row for name,row in records.items() if name!='execution-summary'} and all(summary[k] is False for k in ('independent_cloud_readback_verified','formal_v2_ready','paper_eligible')),'CPU summary output inventory/eligibility differs')
 materialized=summary['fit_materialization'];need(materialized['status']=='passed' and materialized['fold_id']==fold and materialized['all_reconstructed_payload_bytes_verified'] is True and materialized['predictions_generated'] is False,'CPU fit input reconstruction scope differs')
 package=R/f'artifacts/spd-official-oof-fivefold-20260930/fold-{fold}-package';converted=R/f'artifacts/spd-official-oof-fivefold-20260930/remote-conversion/fold-{fold}-converted';freeze=R/f'artifacts/spd-oof-completed-byte-freeze-watch-20261001/fold-{fold}-byte-freeze/acceptance-receipt.json';held=R/f'artifacts/spd-canonical-oof-heldout-inference-inputs-20260930/fold-{fold}';poses=R/'artifacts/spd-canonical-oof-heldout-raw-poses-20260930'
 need(materialized['folds'][0]['input_manifest_sha256']==sha(a.fit_inputs/'input-manifest.json'),'CPU fit input identity differs')
 source_manifest=json.loads((SOURCE/'source-freeze-receipt.json').read_bytes())
 for row in source_manifest['inventory']:
  path=SOURCE/row['path'];need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'independent oracle closure changed')
 oracle=SOURCE/'tools/event_track_v2x/verify_spd_canonical_calibration_examples_runtime_v3_candidate.py';need(sha(oracle)=='72177e04f3d737d68ab4ea61584d2743f39568f0bf7a9977b17a7bf3a7d001b7','independent oracle source differs')
 receipt=a.output/'full-local-raw-GT-reconstruction.json';runner="import runpy,sys;sys.path[:0]=[sys.argv.pop(1),sys.argv.pop(1)];runpy.run_path(sys.argv.pop(1),run_name='__main__')"
 command=[sys.executable,'-I','-c',runner,str(SOURCE),str(oracle.parent),str(oracle)]
 for key,value in dict(package=package,converted=converted,fit_inputs=a.fit_inputs,heldout_inputs=held,raw_readback=a.raw_readback,byte_freeze=freeze,raw_poses=poses,calibration=a.output/'calibration.json',receipt=receipt).items():command+=['--'+key.replace('_','-'),str(value)]
 command+=['--raw-readback-sha256',sha(a.raw_readback),'--calibration-sha256',records['calibration']['sha256']]
 subprocess.run(command,check=True)
 proof=json.loads(receipt.read_bytes());job_proof=json.loads((a.output/'independent-examples-and-parameters.json').read_bytes());need(proof==job_proof,'independent cloud/full local reconstruction disagrees with job-local report')
 need(proof['raw_GT_examples_independently_reconstructed'] is True and proof['independent_parameters_recomputed'] is True and proof['fit_frames']==sum(raw['frame_totals'].values()),'full calibration coverage/numerics incomplete')
 accepted=dict(proof,cloud_calibration_task_id=task.id,cloud_artifacts=records,actual_cpu_runtime_independently_verified=True,full_cloud_bytes_verified=True,cloud_execution_bootstrap_sha256=BOOTSTRAP_SHA,acceptor_sha256=sha(Path(__file__)),fit_raw_acceptance_sha256=sha(a.raw_readback),oracle_sha256=sha(oracle),local_interpreter=sys.version,shared_helpers_disclosed=proof['shared_helpers'],checked_at_utc=datetime.now(timezone.utc).isoformat())
 with (a.output/'acceptance-receipt.json').open('x') as f:json.dump(accepted,f,indent=2,allow_nan=False);f.write('\n')
 print('CALIBRATION_CLOUD_FULL_EXAMPLES_PARAMETERS_ACCEPTED '+sha(a.output/'acceptance-receipt.json'),flush=True)
if __name__=='__main__':main()
