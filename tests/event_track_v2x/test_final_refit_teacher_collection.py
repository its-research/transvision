import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest


@pytest.fixture
def collector(monkeypatch):
    root=Path(__file__).resolve().parents[2]/'tools/event_track_v2x'
    monkeypatch.syspath_prepend(str(root))
    spec=importlib.util.spec_from_file_location('final_teacher_collection_tests',root/'collect_final_refit_capacity_teacher.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def two_sequence_envelope(module, root):
    """Synthetic byte-consumer fixture, never registered as real experimental data."""
    root=root.resolve()
    cfg=dict(method='rbf',allocation='teacher',backend='exclusive_root_partition_regions_v1',
        state=dict(candidate_protocol='rbf-all-class-top64-v1'),limits=dict(residual_partition_version=1))
    origin=dict(events={'fixture':'synthetic-two-sequence'},checkpoint=dict(sha256='b'*64),configuration=cfg,cache_manifest=dict(sha256='c'*64))
    binding=dict(original_plan=origin,GT_read=False,test_read=False,seed=1337,original_event_asset=origin['events'],
        checkpoint_sha256='b'*64,final_model_sha256='d'*64,expected_runtime_sources={'fixture.py':'a'*64})
    natives={};events=[]
    digest=lambda value:hashlib.sha256(module.canonical(value)).hexdigest()
    for sid in ('s0','s1'):
        native=root/sid;native.mkdir()
        event=dict(sequence_id=sid,event_id=sid+':event',frame_id='f',reference_us=100,decision_us=200,deliveries=[])
        events.append(event)
        p=dict(sequence_id=sid,frame_id='f',decision_timestamp_us=200,box_reference_timestamp_us=100,
            previous_commit_sha256='0'*64,predictions=[])
        p['commit_sha256']=digest(p)
        config=dict(state=cfg['state'],**cfg['limits'])
        a=dict(sequence_id=sid,event_id=event['event_id'],prediction_sha256=p['commit_sha256'],previous_audit_sha256='0'*64,
            kind=module.SCHEMA,training_trace_only=True,raw_probe_witness_recipe=module.RECIPE,
            initial_search_witnesses=dict(recipe=module.INITIAL),explicit_residual_partition=True,residual_partition_version=1,
            configuration_sha256=digest(config),cache_ingestion=dict(request_sha256=digest(['f',100,200,[]]),
                new_deliveries=[],duplicate_deliveries=[],gt_model_inputs=False,candidate_protocol='rbf-all-class-top64-v1',cache_manifest_sha256='c'*64),
            allocation_trace=[dict(allocation_training=dict(candidates=[{'software-fixture':True}]))])
        database=native/'trace.sqlite';db=sqlite3.connect(database)
        db.executescript('CREATE TABLE meta(k TEXT PRIMARY KEY,v BLOB);'
            'CREATE TABLE events(ordinal INTEGER PRIMARY KEY,event_id TEXT,prediction BLOB,audit BLOB);')
        meta=dict(schema=module.SCHEMA,sequence_id=sid,config=config,
            state=dict(events=1,prediction_sha256=p['commit_sha256'],audit_sha256=digest(a)))
        db.executemany('INSERT INTO meta VALUES(?,?)',[(k,module.canonical(v)) for k,v in meta.items()])
        db.execute('INSERT INTO events VALUES(0,?,?,?)',(event['event_id'],module.canonical(p),module.canonical(a)))
        db.commit();db.close()
        plan=dict(kind='rbf_paper_replay_v1',fixture=False,expected_sequences=[sid],expected_events=1,events_sha256=digest([event]),
            protocol=dict(dataset='spd',split='train'),model_binding=dict(dataset='spd',fit_split='train',seed=1337,
            checkpoint_sha256='b'*64,model_sha256='d'*64),configuration=cfg,cache_sha256='c'*64,source_sha256={'fixture.py':'a'*64})
        module.write(native/'plan.json',plan)
        (native/'predictions.jsonl').write_bytes(module.canonical(p)+b'\n')
        (native/'audit.jsonl').write_bytes(module.canonical(a)+b'\n')
        module.write(native/'timings.json',[dict(event=0,seconds=0.)]);module.write(native/'resources.json',{'scope':'synthetic'})
        receipt=dict(kind='rbf_paper_replay_receipt_v1',status='software_replay_completed',fixture=False,
            completed_sequences=[sid],completed_events=1,databases={sid:dict(path='trace.sqlite',sha256=module.sha(database))},
            files={n:module.sha(native/n) for n in ('plan.json','predictions.jsonl','audit.jsonl','timings.json','resources.json')},
            peak_rss_native_units=1024,peak_rss_scope='synthetic-control')
        module.write(native/'receipt.json',receipt);natives[sid]=native
    return events,natives,binding


def test_two_sequences_keep_original_plan_and_preserve_old_failure(collector,tmp_path):
    events,natives,binding=two_sequence_envelope(collector,tmp_path)
    frozen=collector.load(collector.PARENT,collector.PARENT_SHA,'preserved_original_teacher_collector')
    with pytest.raises(TypeError):
        frozen.assemble(events,natives,tmp_path.resolve()/'old-attempt',expected_events=2,
            expected_sequences=['s0','s1'],fixture=False,binding=binding)
    result=collector.assemble(events,natives,tmp_path.resolve()/'corrected',expected_events=2,
        expected_sequences=['s0','s1'],fixture=False,binding=binding)
    assert result['completed_events']==2 and result['labels']==2
    assert result['full_real_teacher_target_admission'] is False
    assert result['total_teacher_resource_cost_admitted'] is False
    assert json.loads((tmp_path/'corrected/collection-binding.json').read_bytes())['original_plan']==binding['original_plan']


def test_wrong_final_model_on_later_sequence_refused(collector,tmp_path):
    events,natives,binding=two_sequence_envelope(collector,tmp_path)
    path=natives['s1']/'plan.json';plan=json.loads(path.read_bytes());plan['model_binding']['model_sha256']='e'*64
    path.write_bytes(collector.canonical(plan)+b'\n')
    rp=natives['s1']/'receipt.json';receipt=json.loads(rp.read_bytes());receipt['files']['plan.json']=collector.sha(path)
    rp.write_bytes(collector.canonical(receipt)+b'\n')
    with pytest.raises(AssertionError):
        collector.assemble(events,natives,tmp_path.resolve()/'rejected',expected_events=2,
            expected_sequences=['s0','s1'],fixture=False,binding=binding)
    assert not (tmp_path/'rejected/receipt.json').exists()


@pytest.mark.parametrize('world',[4,8])
def test_full_rank_coverage_and_no_TF32(collector,world):
    sequences=[f'{i:04d}' for i in range(46)];groups={s:[{}] for s in sequences}
    plan=dict(world_size=world,seed=1337)
    report=dict(kind='rbf_final_refit_capacity_witness_full_train_teacher_candidate_v1',plan=plan,task_id='teacher',
        failure=None,all_46_sequences_7445_events_completed=True,ranks=[dict(rank=i,seed=1337,method='rbf',world_size=world,
            all_sequences_completed=True,TF32_matmul=False,TF32_cudnn=False,gpu_uuid=f'uuid{i}',events_committed=len(sequences[i::world]),
            sequences=[dict(sequence_id=s,completed=True,failure=None,committed_events=1,expected_events=1) for s in sequences[i::world]]) for i in range(world)])
    collector.validate_rank_report(report,'teacher',plan,sequences,groups)
    bad=copy.deepcopy(report);bad['ranks'][0]['sequences'].pop()
    with pytest.raises(AssertionError):collector.validate_rank_report(bad,'teacher',plan,sequences,groups)
    bad=copy.deepcopy(report);bad['ranks'][0]['TF32_matmul']=True
    with pytest.raises(AssertionError):collector.validate_rank_report(bad,'teacher',plan,sequences,groups)
    assert len(collector.artifact_keys(world))==world+2


def test_expected_plan_only_changes_teacher_allocation_and_witness_sources(collector):
    import rbf_final_refit_teacher_binding as gate
    prep,control=gate.source_gate()
    main=json.loads(gate.MAIN_JOURNAL.read_bytes())['jobs'][0]
    proof=dict(kind=gate.PREREQUISITE_KIND,main_prerequisite_verified=True,teacher_runtime_or_targets_admitted=False,
        seed=main['seed'],main_task_id=main['task_id'],final_model_sha256=main['plan']['final_refit_model_sha256'],
        sequence_proof_sha256={str(i):'a'*64 for i in range(46)},main_acceptance_sha256='b'*64,byte_admission_sha256='c'*64)
    artifact=dict(task='not-an-actual-task',key='final-main-prerequisite',bytes=123,sha256='d'*64)
    plan=gate.expected_plan(main,proof,artifact,prep['bootstrap_sha256'],control['patches'],8)
    assert plan['configuration']==dict(main['plan']['configuration'],allocation='teacher')
    assert plan['source_replacements']==main['plan']['source_replacements'] and plan['checkpoint']==main['plan']['checkpoint']
    assert plan['world_size']==8
    for mutation in ('model','partial','claim'):
        bad=copy.deepcopy(proof)
        if mutation=='model':bad['final_model_sha256']=main['plan']['original_nested_model_sha256']
        elif mutation=='partial':bad['sequence_proof_sha256'].pop('0')
        else:bad['teacher_runtime_or_targets_admitted']=True
        with pytest.raises(AssertionError):gate.expected_plan(main,bad,artifact,prep['bootstrap_sha256'],control['patches'],8)


def test_independent_factor_check_rejects_missing_or_changed_rows(collector,tmp_path):
    dbpath=tmp_path/'factors.sqlite';db=sqlite3.connect(dbpath)
    db.executescript('CREATE TABLE observations(i INTEGER PRIMARY KEY,node_id TEXT,raw BLOB,sha TEXT);'
        'CREATE TABLE potentials(i INTEGER,p INTEGER,w REAL);')
    observation=dict(features=[0.]*142,state_us=0);raw=collector.canonical(observation)
    db.execute('INSERT INTO observations VALUES(0,?,?,?)',('node',raw,hashlib.sha256(raw).hexdigest()))
    db.execute('INSERT INTO potentials VALUES(0,-1,0.)');db.commit()
    expected=[dict(row=0,node_id='node',context_indices=[0],logits=[0.])]
    assert collector.verify_factors(dbpath,expected,0)['rows']==1
    db.execute('UPDATE potentials SET w=0.1');db.commit()
    with pytest.raises(AssertionError):collector.verify_factors(dbpath,expected,0)
    db.close()
