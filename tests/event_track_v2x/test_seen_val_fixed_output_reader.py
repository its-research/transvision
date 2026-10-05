"""Fixed K1/K4 output gates using rehashed corruptions and full mock streams."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tarfile
from types import SimpleNamespace as N

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
b=importlib.import_module('rbf_seen_val_fixed_output_binding')
reader=importlib.import_module('read_rbf_seen_val_fixed_outputs')
prepare=importlib.import_module('prepare_seen_val_fixed_output_reader')
from test_seen_val_forest_output_reader import sequence_fixture as shared_fixture,dump

COMMON=['train-protocol','car-only','changed-model','changed-source','changed-schedule','changed-cache',
    'extra-sequence','changed-arrival','rescore','missing-empty-event','wrong-clock','late-information',
    'future-arrival','wrong-raw-digest','wrong-context','wrong-factor','extra-prediction','wrong-node','extra-row']


def sequence_fixture(tmp_path,width,mutation=None):
    values=list(shared_fixture(tmp_path,mutation if mutation in COMMON else None))
    directory,seq,events,origin,refs,plan,normal=values
    plan.update(seed=2027,method='topk',baseline_K=width,configuration=dict(method='topk',allocation='bound',
        state=dict(active_limit=width,decision_mode='retained')))
    pp=json.loads((directory/'plan.json').read_bytes())
    pp['configuration']=copy.deepcopy(plan['configuration'])
    pp['model_binding'].update(seed=2027,fit_split='train',dataset='spd',candidate_protocol='rbf-all-class-top64-v1')
    if mutation=='wrong-K':pp['configuration']['state']['active_limit']=1 if width==4 else 4
    if mutation=='wrong-seed':pp['model_binding']['seed']=1337
    if mutation=='val-fit':pp['model_binding']['fit_split']='val'
    if mutation=='main-backend':pp['configuration']['method']='rbf'
    dump(directory/'plan.json',pp)
    db=sqlite3.connect(directory/'forest.sqlite');audits=[]
    for ordinal,raw in db.execute('SELECT ordinal,audit FROM events ORDER BY ordinal').fetchall():
        audit=json.loads(raw)
        for key in ('explicit_residual_partition','residual_partition_version','components'):audit.pop(key,None)
        audit.update(kind='persistent_irreversible_identity_beam_v1',beam_width=width,recovery_enabled=False,
            archived_prefixes_used_for_recovery=False,reproduced_classical_mht=False,
            pruning_policy='fixed_top_k_root_classes_after_each_arrival_ordered_node',
            output_policy='conditional_bayes_over_retained_classes_no_regret_fallback',allocation_policy='fixed_width_no_adaptive_search')
        if mutation=='audit-width':audit['beam_width']=8
        if mutation=='recovery':audit['recovery_enabled']=True
        if mutation=='archived-recovery':audit['archived_prefixes_used_for_recovery']=True
        if mutation=='MHT':audit['reproduced_classical_mht']=True
        if mutation=='adaptive':audit['allocation_policy']='learned'
        if mutation=='pruning':audit['pruning_policy']='bound'
        if mutation=='regret-fallback':audit['output_policy']='other'
        raw=b.canonical(audit);audits.append(raw)
        db.execute('UPDATE events SET audit=? WHERE ordinal=?',(raw,ordinal))
    db.commit();db.close();(directory/'audit.jsonl').write_bytes(b'\n'.join(audits)+b'\n')
    rehash(directory)
    return (*values,dict(kernel='kernel-bytes',source='source-bytes'))


def rehash(directory):
    receipt=json.loads((directory/'receipt.json').read_bytes())
    for item in receipt['databases'].values():item['sha256']=b.sha(directory/item['path'])
    receipt['files']={p.name:b.sha(p) for p in directory.iterdir() if p.is_file() and p.name!='receipt.json'}
    dump(directory/'receipt.json',receipt)


@pytest.mark.parametrize('width',[1,4])
def test_fixed_backend_retains_empty_events_and_numeric_factors(tmp_path,width):
    result=reader.check_sequence(*sequence_fixture(tmp_path,width))
    assert result['events']==2 and result['nodes']==1 and result['max_factor_difference_to_admitted_seen_val_forward']==0


@pytest.mark.parametrize('mutation',COMMON+['wrong-K','wrong-seed','val-fit','main-backend','audit-width','recovery',
    'archived-recovery','MHT','adaptive','pruning','regret-fallback'])
def test_coherently_rehashed_bad_outputs_are_rejected(tmp_path,mutation):
    with pytest.raises(AssertionError):reader.check_sequence(*sequence_fixture(tmp_path,4,mutation))


def report_fixture(width,world=4):
    plan=dict(seed=2027,baseline_K=width,method='topk',configuration=dict(method='topk',state=dict(active_limit=width,decision_mode='retained')),
        world_size=world,main_seen_val_input_plan={'seen_val_input_publication':{'input':'frozen'}},
        main_seen_val_input_plan_sha256='main-plan',baseline_train_CPU_prerequisite={'CPU':'frozen'})
    inputs=dict(seed=2027,input_publication={'input':'frozen'},full_train_interface_required=True,validation_or_test_selection=False,
        measured_network_arrival_history_verified=False,learned_Stage2_complete=False,same_resource_performance_accepted=False,paper_performance_complete=False)
    baseline=dict(seed=2027,K=width,baseline_configuration_unchanged=True,baseline_train_CPU_prerequisite={'CPU':'frozen'},
        main_input_plan_sha256='main-plan',full_train_baseline_acceptance_inherited=False,
        measured_network_arrival_history_verified=False,same_resource_performance_accepted=False,paper_performance_complete=False)
    report=dict(kind=f'rbf_final_refit_seen_val_fixed_K{width}_candidate_v1',all_21_sequences_3316_events_completed=True,
        ranks=[dict(rank=i,seed=2027,method='topk',world_size=world,all_sequences_completed=True,TF32_matmul=False,TF32_cudnn=False,
            gpu_uuid=str(i),capability=[8,0],native_architectures=['sm_80']) for i in range(world)])
    return plan,report,inputs,baseline,{'input':'frozen'}


@pytest.mark.parametrize('mutation',['K','CPU','main-plan','inherit','formal','TF32','duplicate-device','non-native','partial','main-method'])
def test_fixed_report_rejects_wrong_lineage_device_or_claim(mutation):
    plan,report,inputs,baseline,published=report_fixture(4);b.verify_report(plan,report,inputs,baseline,published)
    if mutation=='K':baseline['K']=1
    if mutation=='CPU':baseline['baseline_train_CPU_prerequisite']={}
    if mutation=='main-plan':baseline['main_input_plan_sha256']='other'
    if mutation=='inherit':baseline['full_train_baseline_acceptance_inherited']=True
    if mutation=='formal':baseline['measured_network_arrival_history_verified']=True
    if mutation=='TF32':report['ranks'][0]['TF32_matmul']=True
    if mutation=='duplicate-device':report['ranks'][0]['gpu_uuid']='1'
    if mutation=='non-native':report['ranks'][0]['native_architectures']=[]
    if mutation=='partial':report['ranks'].pop()
    if mutation=='main-method':report['ranks'][0]['method']='rbf'
    with pytest.raises(AssertionError):b.verify_report(plan,report,inputs,baseline,published)


def test_fixed_reader_derivation_and_original_source_map():
    source,control=prepare.derive()
    assert source==Path(reader.__file__).read_text() and control['exact_reversible_edits']==21
    assert len(b.artifact_keys(4))==7 and 'exclusive-source-manifest' not in b.artifact_keys(8)
    sources=b.source_map({'source':{'sha256':b.SOURCE_SHA}})
    reference=b.R/'artifacts/rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/topk/seed3407/rank0-unpack/rank-0/0000/plan.json'
    assert sources==json.loads(reference.read_bytes())['source_sha256']


@pytest.mark.parametrize('width',[1,4])
def test_running_task_never_reads_outputs(tmp_path,monkeypatch,width):
    monkeypatch.setattr(reader,'R',tmp_path)
    monkeypatch.setattr(b,'qualify_job',lambda *a:pytest.fail('running task is not a candidate for readback'))
    plan=dict(world_size=4,method='topk',baseline_K=width,seed=2027,configuration=dict(allocation='bound'))
    job=dict(plan=plan,K=width,seed=2027,recipe_sha256=hashlib.sha256(b.canonical(plan)).hexdigest())
    reader.read_job(N(seed=2027,K=width),job,N(status='in_progress',id='running'),None)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('width',[1,4])
@pytest.mark.parametrize('status',['completed','failed'])
def test_full_mock_21_sequence_3316_event_readback_retains_scope_and_deduplicates(tmp_path,monkeypatch,width,status):
    cloud=tmp_path/'cloud';cloud.mkdir();work=tmp_path/'work';work.mkdir()
    plan,report,inputs,baseline,published=report_fixture(width)
    expected={};groups={};sources=None
    for i in range(21):
        sid=f'{i:04d}';parent=work/f'rank-{i%4}';parent.mkdir(exist_ok=True)
        values=sequence_fixture(parent,width)
        directory,_,events,_,refs,seqplan,_,sources=values
        destination=parent/sid;directory.rename(destination)
        pp=json.loads((destination/'plan.json').read_bytes());pp['expected_sequences']=[sid]
        count=158 if i<20 else 156
        group=[dict(event_id=f'{sid}-{j}',deliveries=[]) for j in range(count)]
        pp['expected_events']=count;pp['events_sha256']=hashlib.sha256(b.canonical(group)).hexdigest();dump(destination/'plan.json',pp)
        receipt=json.loads((destination/'receipt.json').read_bytes());receipt['completed_sequences']=[sid];receipt['completed_events']=count
        receipt['databases']={sid:next(iter(receipt['databases'].values()))};dump(destination/'receipt.json',receipt)
        db=sqlite3.connect(destination/'forest.sqlite');pred,audit=db.execute('SELECT prediction,audit FROM events LIMIT 1').fetchone()
        db.execute('DELETE FROM events')
        db.executemany('INSERT INTO events VALUES (?,?,?,?)',[(j,event['event_id'],pred,audit) for j,event in enumerate(group)])
        db.commit();db.close()
        (destination/'predictions.jsonl').write_bytes((pred+b'\n')*count);(destination/'audit.jsonl').write_bytes((audit+b'\n')*count)
        rehash(destination);groups[sid]=group;expected[sid]=refs
    plan.update(seqplan);plan.update(world_size=4,bootstrap_sha256='bootstrap',evaluation_scope=f'SPD seen-val exploratory scheduled snapshots; fixed K{width} baseline')
    report.update(task_id='fixture-task',plan=plan,failure=None if status=='completed' else {'type':'fixture_failure'},
        paper_performance_complete=False,same_resource_baseline_comparison_accepted=False)
    for rank in report['ranks']:
        ids=sorted(groups)[rank['rank']::4];rank['sequences']=[dict(sequence_id=sid) for sid in ids]
        rank['events_committed']=sum(len(groups[sid]) for sid in ids)
        with tarfile.open(cloud/f'replay-rank{rank["rank"]}.tar.gz','w:gz') as archive:
            archive.add(work/f'rank-{rank["rank"]}',arcname=f'rank-{rank["rank"]}')
    for key,value in [('receipt',report),('seen-val-input-binding',inputs),('seen-val-baseline-binding',baseline)]:dump(cloud/(key+'.json'),value)
    artifacts={key:N(hash=b.sha(reader.local_artifact(cloud,key)),size=reader.local_artifact(cloud,key).stat().st_size) for key in b.artifact_keys(4)}
    job=dict(K=width,seed=2027,task_id='fixture-task',plan=plan,recipe_sha256=hashlib.sha256(b.canonical(plan)).hexdigest())
    pub=tmp_path/'publication';pub.write_text('fixture')
    args=N(K=width,seed=2027,publication=pub)
    task=N(id='fixture-task',status=status,artifacts=artifacts)
    helper=reader.helpers()
    def download(task,key,path):shutil.copyfile(reader.local_artifact(cloud,key),path);return dict(sha256=b.sha(path),bytes=path.stat().st_size)
    monkeypatch.setattr(reader,'helpers',lambda:N(read_artifact=download,unpack=helper.unpack,normalized_factors=helper.normalized_factors))
    monkeypatch.setattr(reader,'R',tmp_path)
    monkeypatch.setattr(reader,'register',lambda *a:None)
    monkeypatch.setattr(b,'qualify_job',lambda *a:published)
    monkeypatch.setattr(b,'source_gate',lambda:None)
    monkeypatch.setattr(b,'source_map',lambda *a:sources)
    monkeypatch.setattr(b,'verify_task_unchanged',lambda *a:None)
    monkeypatch.setattr(b,'reference_inputs',lambda *a:(dict(rows=21),dict(model_sha256='model'),expected,dict(origin_us_by_sequence={s:0 for s in groups}),groups))
    reader.read_job(args,job,task,None)
    root=tmp_path/f'artifacts/rbf-seen-val-fixed-K{width}-independent-byte-factor-v1-20261005/seed2027'
    receipt=root/('independent-byte-coverage-factor-admission.json' if status=='completed' else 'independent-failure-byte-readback.json')
    proof=json.loads(receipt.read_bytes());checksum=b.sha(receipt)
    assert proof['K']==width
    if status=='completed':
        assert proof['method']=='topk' and len(proof['sequences'])==21
        assert sum(v['events'] for v in proof['sequences'])==3316 and proof['total_nodes']==21
        assert proof['full_baseline_semantics_or_fresh_state_independently_accepted'] is False
        assert proof['main_or_train_full_forest_acceptance_inherited'] is proof['paper_performance_complete'] is False
    else:
        assert proof['experiment_accepted'] is proof['automatic_retry_permitted'] is False
        assert proof['producer_failure']=={'type':'fixture_failure'}
        assert not (root/'independent-byte-coverage-factor-admission.json').exists()
    reader.read_job(args,job,task,None);assert b.sha(receipt)==checksum
    def forbidden(*a):pytest.fail('existing output must not be redownloaded')
    monkeypatch.setattr(reader,'helpers',forbidden)
    file=root/'seen-val-baseline-binding.json';file.write_text('{}')
    with pytest.raises(AssertionError):reader.read_job(args,job,task,None)


@pytest.mark.parametrize('changed',['K','seed',*b.dispatch.PATH_ARGUMENTS])
def test_changed_input_receipt_or_identity_rejected_before_any_upstream_read(tmp_path,monkeypatch,changed):
    args=N(K=4,seed=2027)
    job=dict(K=4,seed=2027,plan=dict(seed=2027,baseline_K=4),input_paths={},input_receipt_sha256={})
    for key in b.dispatch.PATH_ARGUMENTS:
        path=tmp_path/key;path.write_text('original');setattr(args,key,path)
        job['input_paths'][key]=str(path);job['input_receipt_sha256'][key]=b.sha(path)
    if changed in ('K','seed'):job[changed]=1
    else:getattr(args,changed).write_text('replacement receipt')
    monkeypatch.setattr(b.producer,'local_cpu_prerequisite',lambda *a:pytest.fail('mutated input must stop before upstream checks'))
    with pytest.raises(AssertionError):b.qualify_job(args,None,job,None)
