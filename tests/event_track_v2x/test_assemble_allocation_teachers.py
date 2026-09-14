from dataclasses import asdict
import copy
import json

import pytest

from tools.event_track_v2x import assemble_allocation_teachers as tool
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows
from tools.event_track_v2x.train_forest_identity import FitConfig,fit_dataset,training_sources
from transvision.models.event_track_v2x.allocation_training import allocation_sources,export_training,training_binding
from transvision.models.event_track_v2x.beam_recovery_tracking import BeamRecoveryConfig
from transvision.models.event_track_v2x.detection_cache_v2 import canonical,sha_file
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
from test_forest_training_data import prepared_rows


@pytest.fixture
def leaves(prepared_rows,tmp_path):
    data,_,cache,rows=prepared_rows
    trained=tmp_path/'trained'
    fit=fit_dataset(data,sha_file(data/'manifest.json'),trained,
        config=FitConfig(epochs=1,batch_size=4,hidden=8,heads=2,dropout=0.),require_full_train=False)
    cp=fit['seeds'][0];config=BeamRecoveryConfig(recovery_budget=3)
    scorer,_=load_identity_checkpoint((trained/cp['checkpoint_manifest']).parent,cp['checkpoint_sha256'],config=config.state)
    metadata=tmp_path/'pairs.json'
    metadata.write_bytes(canonical([dict(vehicle_sequence=r['sequence_id'],infrastructure_sequence=r['sequence_id'],
        vehicle_frame=r['vehicle_frame'],infrastructure_frame=r['infrastructure_frame']) for r in rows]))
    sources=dict(training_sources(),**allocation_sources())
    for name in ('collect_allocation_training.py','run_persistent_forest_v2.py','prepare_forest_training.py','train_forest_identity.py'):
        p='tools/event_track_v2x/'+name;sources[p]=sha_file(tool.ROOT/p)
    replays=[]
    for i,scene in enumerate(sorted({r['sequence_id'] for r in rows})):
        subset=[r for r in rows if r['sequence_id']==scene];out=tmp_path/f'leaf-{i}'
        plan=dict(kind='component_allocation_teacher_full_train_plan_v1',class_scope=['car'],
            cache_sha256=cache.manifest_sha256,cooperative_metadata_sha256=sha_file(metadata),
            full_official_train_verified=False,input_full_official_train_verified=False,
            development_sequence=scene,scheduled_frames=len(subset),geometry_development=False,
            source_sha256=sources,configuration=asdict(config),identity_checkpoint_sha256=cp['checkpoint_sha256'],
            allocation_training_binding=training_binding(config,scorer.signature,frozen_cache_identity(cache)))
        receipt=replay_rows(cache,subset,out,config,allocation_teacher=True,learned_scorer=scorer,plan=plan)
        (out/'development-teacher-receipt.json').write_bytes(canonical(dict(receipt,full_official_train_trace_completed=False)))
        replays.append((out,sha_file(out/'receipt.json')))
    return replays,metadata


def reseal(path):
    receipt=json.loads((path/'receipt.json').read_bytes())
    receipt.update({key:sha_file(path/name) for key,name in tool.STREAMS.items()})
    receipt['plan_sha256']=sha_file(path/'plan.json')
    (path/'receipt.json').write_bytes(canonical(receipt))
    (path/'development-teacher-receipt.json').write_bytes(canonical(dict(receipt,full_official_train_trace_completed=False)))
    return path,sha_file(path/'receipt.json')


def test_joined_teacher_reexports_exact_groups_and_keeps_originals(leaves,tmp_path):
    replays,metadata=leaves
    before={p:sha_file(p) for path,_ in replays for p in path.iterdir() if p.is_file()}
    out=tmp_path/'joined'
    receipt=tool.assemble(list(reversed(replays)),metadata,out,allow_fixture=True)
    plan=json.loads((out/'plan.json').read_bytes())
    assert plan['kind']==tool.KIND and not plan['full_official_train_verified']
    assert receipt['completed_frames']==receipt['scheduled_frames']==2
    assert receipt['execution_topology']=='independent_complete_sequence_teachers'
    assert receipt['source_sequence_run_count']==2
    assert 'source_teacher_process_count' not in receipt
    assert receipt['teacher_latency_is_not_deployment_latency']
    assert not (out/'full-train-teacher-receipt.json').exists()
    tool.verify_assembly(out,plan,receipt,metadata,allow_fixture=True)
    joined=export_training(out,sha_file(out/'receipt.json'),tmp_path/'joined-data')
    for i,(path,digest) in enumerate(replays):
        source=export_training(path,digest,tmp_path/f'leaf-data-{i}')
        item=next(r for r in joined['shards'] if r['sequence_id']==source['shards'][0]['sequence_id'])
        for key in ('sha256','groups','rows'):assert item[key]==source['shards'][0][key]
        source_receipt=json.loads((path/'receipt.json').read_bytes())
        database=next(iter(source_receipt['sequence_heads'].values()))['database']
        assert (out/f'sequence-{i:04d}.sqlite').stat().st_ino != (path/database).stat().st_ino
    assert all(sha_file(p)==h for p,h in before.items())
    with pytest.raises(ValueError,match='official full train'):
        tool.assemble(replays,metadata,tmp_path/'not-real')
    assert not (tmp_path/'not-real').exists()


@pytest.mark.parametrize('bad',['missing','duplicate','hash','unfinished','model','configuration','source','pair_sha','mode','scope'])
def test_bad_sequence_set_or_mixed_provenance_is_rejected_before_output(leaves,tmp_path,bad):
    replays,metadata=leaves;replays=list(replays)
    path=replays[-1][0]
    if bad=='missing':replays.pop()
    elif bad=='duplicate':replays[-1]=replays[0]
    elif bad=='hash':replays[-1]=(path,'0'*64)
    elif bad=='unfinished':(path/'development-teacher-receipt.json').unlink()
    else:
        plan=json.loads((path/'plan.json').read_bytes())
        if bad=='model':plan['identity_checkpoint_sha256']='0'*64
        elif bad=='configuration':
            plan['configuration']['recovery_budget']+=1
            plan['allocation_training_binding']['configuration']=copy.deepcopy(plan['configuration'])
        elif bad=='source':plan['source_sha256'].pop(next(iter(plan['source_sha256'])))
        elif bad=='pair_sha':plan['cooperative_metadata_sha256']='0'*64
        elif bad=='mode':plan['geometry_development']=True
        else:plan['class_scope']=['pedestrian']
        (path/'plan.json').write_bytes(canonical(plan));replays[-1]=reseal(path)
    with pytest.raises((ValueError,FileNotFoundError)):
        tool.assemble(replays,metadata,tmp_path/'bad',allow_fixture=True)
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('bad',['stream','database','input_reference','tool','missing_original',
    'mode','elapsed_scope','rss_scope','timing','elapsed_value','database_bytes','unavailable'])
def test_package_reverification_does_not_trust_assembly_flags(leaves,tmp_path,bad):
    replays,metadata=leaves;out=tmp_path/'joined'
    receipt=tool.assemble(replays,metadata,out,allow_fixture=True)
    plan=json.loads((out/'plan.json').read_bytes())
    if bad=='stream':
        f=out/'predictions.jsonl';f.write_bytes(f.read_bytes()+b'{}\n')
        receipt['predictions_sha256']=sha_file(f)
    elif bad=='database':(out/'sequence-0000.sqlite').write_bytes(b'not a database')
    elif bad=='input_reference':plan['assembly_inputs'][0]['final_sha256']='0'*64
    elif bad=='tool':plan['assembly_tool_sha256']='0'*64
    elif bad=='missing_original':(replays[0][0]/'development-teacher-receipt.json').unlink()
    elif bad=='mode':receipt['additional_beam_recovery_enabled']=False
    elif bad=='elapsed_scope':receipt['elapsed_scope']='campaign wall'
    elif bad=='rss_scope':receipt['rss_scope']='concurrent peak'
    elif bad=='timing':receipt['latency_seconds_p50_p95_p99_max'][0]+=1
    elif bad=='elapsed_value':receipt['elapsed_seconds']+=1
    elif bad=='database_bytes':receipt['database_bytes']+=1
    else:receipt['source_unavailable']={}
    (out/'plan.json').write_bytes(canonical(plan));receipt['plan_sha256']=sha_file(out/'plan.json')
    with pytest.raises((ValueError,FileNotFoundError)):
        tool.verify_assembly(out,plan,receipt,metadata,allow_fixture=True)


def test_write_failure_keeps_inputs_and_has_no_success_receipt(leaves,tmp_path,monkeypatch):
    replays,metadata=leaves;out=tmp_path/'failed'
    before={path:sha_file(path/'receipt.json') for path,_ in replays}
    def fail(*args,**kwargs):raise OSError('injected disk error')
    monkeypatch.setattr(tool.shutil,'copyfileobj',fail)
    with pytest.raises(OSError,match='disk error'):tool.assemble(replays,metadata,out,allow_fixture=True)
    assert json.loads((out/'failure.json').read_bytes())['partial_outputs_not_final_results']
    assert not (out/'receipt.json').exists()
    assert all(sha_file(path/'receipt.json')==h for path,h in before.items())


@pytest.mark.parametrize('bad',['frame_timing','head_frames','frame_summary'])
def test_resealed_invalid_leaf_measurements_are_rejected(leaves,tmp_path,bad):
    replays,metadata=leaves;replays=list(replays);path=replays[0][0]
    if bad=='frame_timing':
        f=path/'frame-timings.jsonl';lines=[json.loads(line) for line in f.read_bytes().splitlines()]
        lines[0]['frame_seconds']=-1
        f.write_bytes(b''.join(canonical(row)+b'\n' for row in lines))
    else:
        f=path/'receipt.json';receipt=json.loads(f.read_bytes())
        if bad=='head_frames':next(iter(receipt['sequence_heads'].values()))['frames']+=1
        else:receipt['frame_latency_seconds_p50_p95_p99_max'][0]+=1
        f.write_bytes(canonical(receipt))
    replays[0]=reseal(path)
    with pytest.raises(ValueError):tool.assemble(replays,metadata,tmp_path/'bad',allow_fixture=True)
    assert not (tmp_path/'bad').exists()
