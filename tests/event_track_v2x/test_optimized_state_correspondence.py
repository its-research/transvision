import importlib.util
import copy
import json
from pathlib import Path
import sqlite3

import pytest
import numpy as np


@pytest.fixture
def checker(monkeypatch):
    tools=Path(__file__).resolve().parents[2]/'tools/event_track_v2x'
    monkeypatch.syspath_prepend(str(tools))
    spec=importlib.util.spec_from_file_location('optimized_correspondence',tools/'accept_optimized_branch_state_sequence.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_raw_tables_compare_by_primary_key_and_reject_branch_change(checker):
    a,b=sqlite3.connect(':memory:'),sqlite3.connect(':memory:')
    try:
        for db in (a,b):
            db.execute('CREATE TABLE pc1_prefixes(h INTEGER PRIMARY KEY, parent INTEGER, payload BLOB)')
        a.executemany('INSERT INTO pc1_prefixes VALUES(?,?,?)',[(1,0,b'a'),(2,1,b'b')])
        b.executemany('INSERT INTO pc1_prefixes VALUES(?,?,?)',[(2,1,b'b'),(1,0,b'a')])
        assert checker.compare_tables(a,b)=={'pc1_prefixes':2}
        b.execute('UPDATE pc1_prefixes SET parent=0 WHERE h=2')
        with pytest.raises(AssertionError,match='table differs'):
            checker.compare_tables(a,b)
    finally:a.close();b.close()


@pytest.mark.parametrize('mutation',['execution','caps','claim'])
def test_only_declared_execution_metadata_can_be_removed(checker,mutation):
    execution=dict(device='cpu',max_batch=64,recipe='frozen')
    caps=dict(SQL_templates=256,raw_observations=4096,ancestor_pairs_per_call=4096)
    value=dict(state_execution=dict(execution),execution_extra_cache_caps=dict(caps),
        state_execution_independently_accepted=False,prediction_sha256='a',previous_audit_sha256='b',components=[])
    assert checker.semantic_audit(value,execution,caps)=={'components':[]}
    if mutation=='execution':value['state_execution']['max_batch']=128
    elif mutation=='caps':value['execution_extra_cache_caps']['SQL_templates']=1024
    else:value['state_execution_independently_accepted']=True
    with pytest.raises(AssertionError):checker.semantic_audit(value,execution,caps)


@pytest.fixture
def branch_fixture(checker):
    db=sqlite3.connect(':memory:')
    db.executescript('''CREATE TABLE pc1_observations(i INTEGER PRIMARY KEY,raw BLOB);
        CREATE TABLE pc1_prefixes(h INTEGER PRIMARY KEY,parent INTEGER,depth INTEGER,choice INTEGER,root INTEGER,previous_root INTEGER,sha TEXT);
        CREATE TABLE pc1_states(h INTEGER PRIMARY KEY,payload BLOB);''')
    mean=[0.,0.,0.,1.,1.,1.,0.,2.,0.]
    features=[0.]*141;features[138]=1.
    row=dict(node=dict(node_id='a',source_id=0,frame_id='0',information_us=0,arrival_us=0),
        state_us=0,score=.9,mean=mean,covariance=np.eye(9).tolist(),features=features)
    state=dict(mean=mean,covariance=np.eye(9).tolist(),first_us=0,last_us=0,last_order=[0,0,'a'],max_score=.9)
    db.execute('INSERT INTO pc1_observations VALUES(0,?)',(json.dumps(row),))
    db.execute('INSERT INTO pc1_prefixes VALUES(1,0,1,-1,0,NULL,?)',('prefix-hash',))
    db.execute('INSERT INTO pc1_states VALUES(1,?)',(json.dumps(state),))
    config=dict(survival_per_second=.99,birth_score=.05,prune_score=.01,max_age_us=2_000_000,process_noise=.1)
    k=checker.StoredProjection(db,1)
    outputs=k.outputs(1,1_000_000,config,'s',1_000_000,1)
    component=dict(component=1,nodes=1,expired_output_only=False,active=[dict(handle=1)],output_handle=1,
        branches=[dict(handle=1,sha256='prefix-hash',state_sha256=checker.digest(outputs))])
    audit=dict(components=[component])
    prediction=dict(sequence_id='s',box_reference_timestamp_us=1_000_000,decision_timestamp_us=1_000_000,predictions=outputs)
    yield db,k,config,audit,prediction
    db.close()


def test_projection_hash_is_bound_before_comparing_float_states(checker,branch_fixture):
    db,k,config,audit,prediction=branch_fixture
    changed=copy.deepcopy(audit)
    result,branches,chosen=checker.bind_branch_hashes(audit,prediction,lambda c:k,config)
    assert 'state_sha256' not in result['components'][0]['branches'][0]
    assert audit==changed  # The committed audit remains untouched.
    assert branches==[((1,1),prediction['predictions'])] and chosen==prediction['predictions']
    assert chosen[0]['mean'][0]==2.
    changed['components'][0]['branches'][0]['state_sha256']='0'*64
    with pytest.raises(AssertionError,match='state commitment'):
        checker.bind_branch_hashes(changed,prediction,lambda c:k,config)


@pytest.mark.parametrize('mutation',['state','prefix','expiry','chosen','future','missing_branch'])
def test_projection_rejects_corruption_even_when_tiny_state_differences_are_allowed(checker,branch_fixture,mutation):
    db,k,config,audit,prediction=branch_fixture
    if mutation=='state':
        state=json.loads(db.execute('SELECT payload FROM pc1_states').fetchone()[0])
        state['mean'][0]+=1e-12
        db.execute('UPDATE pc1_states SET payload=?',(json.dumps(state),))
    elif mutation=='prefix':audit['components'][0]['branches'][0]['sha256']='bad'
    elif mutation=='expiry':audit['components'][0]['expired_output_only']=True
    elif mutation=='chosen':prediction['predictions'][0]['score']=.5
    elif mutation=='future':prediction['decision_timestamp_us']=-1
    else:audit['components'][0]['branches']=[]
    with pytest.raises(AssertionError):checker.bind_branch_hashes(audit,prediction,lambda c:k,config)


def test_fixed_numeric_tolerance_and_exact_identity(checker,branch_fixture):
    *_,prediction=branch_fixture
    first=prediction['predictions'];second=copy.deepcopy(first)
    second[0]['mean'][0]+=1e-12
    count,error=checker.compare_predictions(first,second)
    assert count==1 and 0<error<1e-8
    second[0]['mean'][0]+=1e-3
    with pytest.raises(AssertionError,match='numerical mismatch'):checker.compare_predictions(first,second)
    second=copy.deepcopy(first);second[0]['track_id']='other'
    with pytest.raises(AssertionError,match='identity'):checker.compare_predictions(first,second)


def test_request_digest_verifies_default_scope_equivalence_and_explicit_input(checker):
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE observations(i INTEGER PRIMARY KEY,state_us INTEGER,raw BLOB)')
    db.executemany('INSERT INTO observations VALUES(?,?,?)',[(0,0,b'{}'),(1,90,b'{}'),(2,200,b'{}')])
    p=dict(frame_id='frame',box_reference_timestamp_us=100,decision_timestamp_us=100)
    a=dict(observation_count=2,new_observations=2,decision_indices=[1],appended_rows=[[],[]],rescored_rows=[],cache_ingestion=None,scorer_binding=None)
    config=dict(state=dict(window_us=20))
    arguments=['frame',100,100,[{},{}],[[],[]],[],None,None,None]
    implicit=checker.digest(arguments)
    arguments[6]=[1];explicit=checker.digest(arguments)
    assert implicit!=explicit
    assert checker.bind_request(db,implicit,p,a,config,0,explicit_scope=False)==2
    assert checker.bind_request(db,explicit,p,a,config,0,explicit_scope=True)==2
    with pytest.raises(AssertionError,match='request commitment'):
        checker.bind_request(db,implicit,p,a,config,0,explicit_scope=True)
    changed=dict(a,decision_indices=[0,1])
    with pytest.raises(AssertionError,match='decision scope'):
        checker.bind_request(db,implicit,p,changed,config,0,explicit_scope=False)
    with pytest.raises(AssertionError,match='request commitment'):
        checker.bind_request(db,'0'*64,p,a,config,0,explicit_scope=False)
    db.close()


def test_portable_commitment_witness_exports_all_values_and_rejects_corruption(checker,branch_fixture,tmp_path):
    import export_branch_state_commitment_witness as exporter
    db,k,cfg,a,p=branch_fixture
    raw=k.rows[0]
    db.execute('CREATE TABLE observations(i INTEGER PRIMARY KEY,state_us INTEGER,raw BLOB)')
    db.execute('INSERT INTO observations VALUES(0,0,?)',(json.dumps(raw),))
    db.execute('CREATE TABLE meta(k TEXT PRIMARY KEY,v BLOB)')
    db.execute('CREATE TABLE events(ordinal INTEGER PRIMARY KEY,event_id TEXT,request TEXT,prediction BLOB,audit BLOB)')
    p.update(frame_id='f',previous_commit_sha256='0'*64)
    p['commit_sha256']=checker.digest(p)
    a.update(event_id='e',sequence_id='s',observation_count=1,new_observations=1,appended_rows=[[[-1,0.]]],
        rescored_rows=[],decision_indices=[0],cache_ingestion=None,scorer_binding=None,
        prediction_sha256=p['commit_sha256'],previous_audit_sha256='0'*64)
    request=checker.digest(['f',1_000_000,1_000_000,[raw],a['appended_rows'],[],[0],None,None])
    meta=dict(schema='experimental_exclusive_batched_state_v1',sequence_id='s',config=dict(state=cfg),
        state=dict(events=1,n=1,prediction_sha256=p['commit_sha256'],audit_sha256=checker.digest(a)))
    db.executemany('INSERT INTO meta VALUES(?,?)',[(key,json.dumps(v)) for key,v in meta.items()])
    db.execute('INSERT INTO events VALUES(0,?,?,?,?)',('e',request,json.dumps(p),json.dumps(a)))
    db.commit()
    path=tmp_path/'case.sqlite';saved=sqlite3.connect(path);db.backup(saved);saved.close()
    output=tmp_path/'witness.jsonl'
    receipt=exporter.export(path,exporter.sha(path),output)
    proof=json.loads(receipt.read_bytes());row=json.loads(output.read_text())
    assert proof['events']==proof['branches']==proof['branch_predictions']==1
    assert proof['all_branch_commitments_reproduced'] is True
    assert proof['fresh_history_numeric_acceptance'] is False
    assert checker.digest(row['branches'][0]['predictions'])==a['components'][0]['branches'][0]['state_sha256']
    bad=sqlite3.connect(path);bad.execute("UPDATE events SET request='changed'");bad.commit();bad.close()
    with pytest.raises(AssertionError,match='request commitment'):
        exporter.export(path,exporter.sha(path),tmp_path/'changed.jsonl')
