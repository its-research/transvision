#!/usr/bin/env python3
"""Verify and freeze three learned K=4 MHT full-SPD-val jobs without launching.

All detection payloads and the three frozen trained checkpoints are checked.
The plan is not a training, inference, evaluation or resource-reservation receipt.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x import audit_full_mht_validation as contract
from tools.event_track_v2x import run_mht_tracking_v2 as producer
from tools.event_track_v2x.run_tracking_v2 import schedule_rows
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity

native = contract.native


def make_jobs(cache,schedule,checkpoints,output):
    """Fixed seed coverage and explicit budgets; independent of observed metrics."""
    if len(checkpoints) != 3 or {c['seed'] for c in checkpoints} != set(contract.CHECKPOINTS):
        raise ValueError('all three distinct declared identity seeds required')
    config = asdict(producer.PersistentMHTConfig())
    if contract.ledger.digest(config) != contract.CONFIG_SHA:
        raise ValueError('current MHT defaults differ from frozen K=4 contract')
    jobs = []
    for c in sorted(checkpoints,key=lambda c:c['seed']):
        if c['sha256'] != contract.CHECKPOINTS[c['seed']]:
            raise ValueError('undeclared checkpoint identity')
        destination = Path(output)/f"seed-{c['seed']}-mht-k4-v1"
        if destination.exists(): raise ValueError('MHT output already exists; inspect it before any retry')
        arguments = dict(cache=str(Path(cache).absolute()),cache_sha256=contract.CACHE_SHA,
            schedule=str(Path(schedule).absolute()),schedule_sha256=contract.SCHEDULE_SHA,
            checkpoint=c['path'],checkpoint_sha256=c['sha256'],output=str(destination.absolute()),device='cpu',width=4)
        arguments.update({k:config[k] for k in config if k.startswith('max_assignment_') or k=='max_generated_candidates'})
        jobs.append(dict(seed=c['seed'],width=4,output=str(destination.absolute()),arguments=arguments))
    return jobs


def prepare(cache_path,cache_sha,schedule_path,schedule_sha,checkpoint_refs,output):
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('new non-symlink campaign output required')
    if cache_sha != contract.CACHE_SHA or schedule_sha != contract.SCHEDULE_SHA:
        raise ValueError('frozen full official SPD val inputs required')
    expected = set(contract.CHECKPOINTS.values())
    if len(checkpoint_refs) != 3 or {sha for _,sha in checkpoint_refs} != expected:
        raise ValueError('all three pinned A100-trained identity checkpoints required')
    sources = producer.mht_sources()
    tool_sources = {str(p.relative_to(ROOT)):native.sha(p) for p in (
        Path(__file__),Path(contract.__file__),Path(contract.ledger.__file__),
        ROOT/'tools/event_track_v2x/evaluate_mht_validation.py',Path(contract.common.__file__),Path(native.__file__))}
    rows = schedule_rows(schedule_path,schedule_sha)
    cache = VerifiedForestCache(cache_path,cache_sha)
    manifest = json.loads(cache.manifest_json)
    if (manifest['split'] != 'val' or manifest['frame_count'] != 7189
            or set(manifest['sequences']) != {r['sequence_id'] for r in rows}):
        raise ValueError('complete official validation cache required')
    runtime = producer.mht_runtime_evidence('cpu')
    if (any(runtime.get(k) != v for k,v in contract.RUNTIME.items())
            or runtime['thread_environment'] != contract.THREAD_ENV):
        raise ValueError('frozen MHT inference runtime required')
    config = producer.PersistentMHTConfig(); checkpoints = []; inputs = {
        str(Path(cache_path).absolute()/'manifest.json'):cache_sha,str(Path(schedule_path).absolute()):schedule_sha}
    for root,sha in checkpoint_refs:
        root = Path(root).absolute()
        scorer,cp = load_identity_checkpoint(root,sha,config=config.state,device='cpu')
        if (cp.get('full_official_train') is not True or cp['seed'] not in contract.CHECKPOINTS
                or contract.CHECKPOINTS[cp['seed']] != sha or frozen_cache_identity(cache) != cp['frozen_cache_identity']):
            raise ValueError('declared full-train checkpoint and matching frozen detector required')
        checkpoints.append(dict(seed=cp['seed'],path=str(root),sha256=sha,model_sha256=cp['model_sha256'],
                                scorer_signature=scorer.signature,full_official_train=True))
        inputs[str(root/'checkpoint.json')] = sha
        inputs[str(root/cp['weights']['path'])] = cp['weights']['sha256']
    jobs = make_jobs(cache_path,schedule_path,checkpoints,output)
    if producer.mht_sources() != sources or any(native.sha(p) != h for p,h in inputs.items()):
        raise ValueError('MHT inputs or sources changed during preflight')
    if any(native.sha(ROOT/p) != h for p,h in tool_sources.items()):
        raise ValueError('MHT audit/evaluation contract changed during preflight')
    result = dict(kind='fixed_scan_mht_three_seed_full_val_campaign_v1',status='inputs_verified_plan_frozen',
        jobs=jobs,planned_jobs=3,configuration=asdict(config),checkpoints=checkpoints,runtime=runtime,
        source_sha256=sources,contract_source_sha256=tool_sources,input_file_sha256=inputs,
        schedule_rows_sha256=contract.ledger.digest(rows),scheduled_frames_per_job=3316,sequences=21,
        sealed_source_frames=7189,max_parallel_full_validation_cpu_jobs=3,
        existing_full_validation_jobs_share_this_limit=True,class_scope=['car'],
        frozen_identity_training_hardware='four_A100',new_parameter_training=False,
        inference_started=False,metrics_computed=False,test_payloads_read=False,ground_truth_read=False,
        validation_already_seen=True,validation_parameter_search=False,validation_checkpoint_selection=False,
        public_method_reproduction=False,fair_resources_verified=False,resource_reservation=False,paper_eligible=False)
    output.mkdir();native.write_json(output/'campaign.json',result)
    print(json.dumps({k:result[k] for k in ('status','planned_jobs','scheduled_frames_per_job','inference_started')}))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache','schedule','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('cache-sha256','schedule-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--checkpoint',nargs=2,action='append',required=True,metavar=('DIRECTORY','SHA256'))
    args = p.parse_args();prepare(args.cache,args.cache_sha256,args.schedule,args.schedule_sha256,args.checkpoint,args.output)
