"""Independent reader interface checks, with no new GPU or NN execution."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sqlite3
import sys
import tarfile
from types import SimpleNamespace as N

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
reader=importlib.import_module('read_rbf_seen_val_forest_outputs')
b=importlib.import_module('rbf_seen_val_forest_output_binding')


def dump(path,value):path.write_bytes(b.canonical(value)+b'\n')


@pytest.mark.parametrize('seed',[1337,2027,3407])
def test_existing_admitted_val_references_resolve_without_repeating_NN(seed):
    root=b.R/f'artifacts/rbf-seen-val-forest-input-bridge-v1-20261005/seed{seed}'
    bound=json.loads((root/'input-binding.json').read_bytes())
    transport=json.loads((b.R/f'artifacts/rbf-seen-val-forest-cache-transport-v1-20261005/seed{seed}/local-transport-readback.json').read_bytes())
    plan=dict(checkpoint=dict(sha256=bound['checkpoint']['sha256']),final_refit_model_sha256=bound['checkpoint']['model_sha256'],
        forward_outputs=bound['forward_artifacts'],events=dict(sha256=bound['events_sha256']),cache_manifest=dict(sha256=transport['cache_manifest_sha256']))
    entry,checkpoint,expected,events,groups=b.reference_inputs(seed,plan)
    assert len(groups)==21 and sum(map(len,groups.values()))==3316
    assert sum(map(len,expected.values()))==bound['rows']==entry['rows']
    assert events['measured_network_arrival_history_verified'] is False
    plan['checkpoint']['sha256']='changed'
    with pytest.raises(AssertionError):b.reference_inputs(seed,plan)


def sequence_fixture(tmp_path,mutation=None):
    seq='fixture-sequence';directory=tmp_path/seq;directory.mkdir()
    events=[dict(event_id='event-with-row',deliveries=[]),dict(event_id='event-with-no-new-row',deliveries=[])]
    plan=dict(configuration=dict(allocation='bound',method='rbf'),checkpoint=dict(sha256='checkpoint'),
        final_refit_model_sha256='model',cache_manifest=dict(sha256='val-cache'),
        exclusive_patches={'kernel':'kernel-bytes'},source_replacements={'source':dict(sha256='source-bytes')})
    pp=dict(kind='rbf_paper_replay_v1',expected_sequences=[seq],expected_events=2,
        protocol=dict(dataset='spd',split='val',candidates='rbf-all-class-top64-v1',evaluation_class='car',
            maximum_detections=64,minimum_raw_score=.05),configuration=plan['configuration'],fixture=False,
        model_binding=dict(checkpoint_sha256='checkpoint',model_sha256='model'),
        events_sha256=hashlib.sha256(b.canonical(events)).hexdigest(),cache_sha256='val-cache',
        source_sha256={'kernel':'kernel-bytes','source':'source-bytes'})
    if mutation=='train-protocol':pp['protocol']['split']='train'
    if mutation=='car-only':pp['protocol']['candidates']='car-only'
    if mutation=='changed-model':pp['model_binding']['model_sha256']='other-model'
    if mutation=='changed-source':pp['source_sha256']['kernel']='other-source'
    if mutation=='changed-schedule':pp['events_sha256']='changed'
    if mutation=='changed-cache':pp['cache_sha256']='train-cache'
    if mutation=='extra-sequence':pp['expected_sequences'].append('unrelated')
    dump(directory/'plan.json',pp)
    db_path=directory/'forest.sqlite';db=sqlite3.connect(db_path)
    db.execute('CREATE TABLE events (ordinal INTEGER,event_id TEXT,prediction BLOB,audit BLOB)')
    db.execute('CREATE TABLE observations (i INTEGER,node_id TEXT,raw BLOB,sha TEXT)')
    db.execute('CREATE TABLE potentials (i INTEGER,p INTEGER,w REAL)')
    predictions=[];audits=[]
    for i,e in enumerate(events):
        audit=dict(cache_ingestion=dict(new_deliveries=[],old_rows_rescored=False),explicit_residual_partition=True,
            residual_partition_version=1,components=[dict(representation='exclusive_root_partition_regions_v1')])
        if mutation=='changed-arrival' and i==0:audit['cache_ingestion']['new_deliveries']=[{'invented':True}]
        if mutation=='rescore' and i==0:audit['cache_ingestion']['old_rows_rescored']=True
        if mutation=='representation' and i==0:audit['components'][0]['representation']='legacy'
        prediction=b.canonical(dict(event=i,predictions=[]));raw_audit=b.canonical(audit)
        if mutation=='missing-empty-event' and i==1:continue
        db.execute('INSERT INTO events VALUES (?,?,?,?)',(i,e['event_id'],prediction,raw_audit))
        predictions.append(prediction);audits.append(raw_audit)
    observation=dict(features=[0.]*203,state_us=10000000,node=dict(information_us=5,arrival_us=10))
    observation['features'][141]=.1
    if mutation=='wrong-clock':observation['features'][141]=.2
    if mutation=='late-information':observation['node']['information_us']=11
    if mutation=='future-arrival':observation['node']['arrival_us']=30
    raw=b.canonical(observation);digest=hashlib.sha256(raw).hexdigest()
    if mutation=='wrong-raw-digest':digest='changed'
    db.execute('INSERT INTO observations VALUES (?,?,?,?)',(0,'node',raw,digest))
    db.execute('INSERT INTO potentials VALUES (?,?,?)',(0,0 if mutation=='wrong-context' else -1,1. if mutation=='wrong-factor' else 0.))
    db.commit();db.close()
    if mutation=='extra-prediction':predictions.append(b'{}')
    (directory/'predictions.jsonl').write_bytes(b'\n'.join(predictions)+b'\n')
    (directory/'audit.jsonl').write_bytes(b'\n'.join(audits)+b'\n')
    receipt=dict(kind='rbf_paper_replay_receipt_v1',completed_sequences=[seq],completed_events=2,status='software_replay_completed',
        databases={seq:dict(path='forest.sqlite',sha256=b.sha(db_path))},
        files={p.name:b.sha(p) for p in directory.iterdir() if p.is_file()})
    dump(directory/'receipt.json',receipt)
    refs=[dict(row=0,node_id='wrong' if mutation=='wrong-node' else 'node',context_indices=[0],logits=[-7.],decision_us=20)]
    if mutation=='extra-row':refs.append(dict(refs[0],row=1))
    return directory,seq,events,0,refs,plan,reader.helpers().normalized_factors


def test_complete_sequence_includes_event_with_no_new_queries(tmp_path):
    result=reader.check_sequence(*sequence_fixture(tmp_path))
    assert result['events']==2 and result['nodes']==1 and result['max_factor_difference_to_admitted_seen_val_forward']==0.


@pytest.mark.parametrize('mutation',['train-protocol','car-only','changed-model','changed-source','changed-schedule','changed-cache',
    'extra-sequence','changed-arrival','rescore','representation','missing-empty-event','wrong-clock','late-information',
    'future-arrival','wrong-raw-digest','wrong-context','wrong-factor','extra-prediction','wrong-node','extra-row'])
def test_rehashed_invalid_outputs_do_not_pass(tmp_path,mutation):
    with pytest.raises(AssertionError):reader.check_sequence(*sequence_fixture(tmp_path,mutation))


@pytest.mark.parametrize('mutation',['unknown-device','duplicate-device','TF32','non-native','partial-ranks','wrong-input','formal-claim'])
def test_report_gate_rejects_runtime_or_scope_changes(mutation):
    plan=dict(world_size=4,seed=2027,seen_val_input_publication={'accepted-input':'hash'})
    bound=dict(seed=2027,input_publication=plan['seen_val_input_publication'],full_train_interface_required=True,
        validation_or_test_selection=False,measured_network_arrival_history_verified=False,learned_Stage2_complete=False,
        same_resource_performance_accepted=False,paper_performance_complete=False)
    report=dict(kind='rbf_final_refit_SPD_seen_val_bound_forest_candidate_v1',all_21_sequences_3316_events_completed=True,
        ranks=[dict(rank=i,seed=2027,method='rbf',world_size=4,all_sequences_completed=True,TF32_matmul=False,TF32_cudnn=False,
            gpu_uuid=str(i),capability=[8,0],native_architectures=['sm_80']) for i in range(4)])
    b.verify_report(plan,report,bound,plan['seen_val_input_publication'])
    if mutation=='unknown-device':report['ranks'][0]['gpu_uuid']='unavailable'
    if mutation=='duplicate-device':report['ranks'][0]['gpu_uuid']='1'
    if mutation=='TF32':report['ranks'][0]['TF32_matmul']=True
    if mutation=='non-native':report['ranks'][0]['native_architectures']=['sm_70']
    if mutation=='partial-ranks':report['ranks'].pop()
    if mutation=='wrong-input':bound['input_publication']={}
    if mutation=='formal-claim':bound['measured_network_arrival_history_verified']=True
    with pytest.raises(AssertionError):b.verify_report(plan,report,bound,plan['seen_val_input_publication'])


def test_running_task_never_downloads_or_creates_root(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(reader,'R',tmp_path)
    monkeypatch.setattr(b,'qualify_job',lambda *a:pytest.fail('running task must remain untouched'))
    plan=dict(world_size=4,method='rbf',configuration=dict(allocation='bound'))
    job=dict(plan=plan,recipe_sha256=hashlib.sha256(b.canonical(plan)).hexdigest())
    reader.read_job(N(seed=2027),job,N(status='in_progress',id='running'),None)
    assert not list(tmp_path.iterdir()) and 'no_readback_started' in capsys.readouterr().out


def test_only_declared_paths_and_artifacts_are_accepted(tmp_path):
    (tmp_path/'valid').write_text('x')
    assert b.contained(tmp_path,'valid')==tmp_path/'valid'
    for path in ('../escape','/absolute'):
        with pytest.raises(AssertionError):b.contained(tmp_path,path)
    (tmp_path/'link').symlink_to(tmp_path/'valid')
    with pytest.raises(AssertionError):b.contained(tmp_path,'link')
    assert len(b.artifact_keys(4))==7 and len(b.artifact_keys(8))==11
    with pytest.raises(AssertionError):b.artifact_keys(3)


@pytest.mark.parametrize('mutation',['coherent-receipt-change','extra-member','symlink','directory-symlink'])
def test_extraction_marker_cannot_hide_changed_member_bytes(tmp_path,mutation):
    raw=tmp_path/'input';raw.mkdir();(raw/'receipt.json').write_text('{"original":true}')
    archive=tmp_path/'rank.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(raw,arcname='rank-0')
    out=tmp_path/'unpack';reader.helpers().unpack(archive,out)
    reader.verify_extracted_archive(archive,out)
    if mutation=='coherent-receipt-change':(out/'rank-0/receipt.json').write_text('{"modified":true}')
    if mutation=='extra-member':(out/'not-in-archive').write_text('extra')
    if mutation=='symlink':
        p=out/'rank-0/receipt.json';p.unlink();p.symlink_to(raw/'receipt.json')
    if mutation=='directory-symlink':(out/'external-directory').symlink_to(raw,target_is_directory=True)
    with pytest.raises(AssertionError):reader.verify_extracted_archive(archive,out)
