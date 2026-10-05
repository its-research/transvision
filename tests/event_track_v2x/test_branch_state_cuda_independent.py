import copy
import importlib.util
import io
import json
from pathlib import Path

import pytest

from test_optimized_state_correspondence import checker, branch_fixture


@pytest.fixture
def receiver(checker):
    path = Path(__file__).resolve().parents[2]/'tools/event_track_v2x/accept_branch_state_cuda_outputs.py'
    spec = importlib.util.spec_from_file_location('CUDA_independent_receiver_test',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def record(prediction, values):
    return dict(ordinal=0,event_id='e',reference_us=prediction['box_reference_timestamp_us'],
        decision_us=prediction['decision_timestamp_us'],branches=[dict(component=1,handle=1,predictions=copy.deepcopy(values))])


def test_portable_hash_and_independent_projection_together(receiver,checker,branch_fixture):
    db,k,config,audit,prediction = branch_fixture
    values = copy.deepcopy(prediction['predictions'])
    values[0]['mean'][0] += 1e-12
    audit['components'][0]['branches'][0]['state_sha256'] = checker.digest(values)
    prediction['predictions'] = values
    witness = receiver.WitnessEvent(record(prediction,values),0,'e',prediction,audit,checker)
    kernel = receiver.PortableProjection(k,1,witness,checker)
    checker.bind_branch_hashes(audit,prediction,lambda c:kernel,config)
    witness.finish()
    assert 0 < witness.max_difference < 1e-8 and witness.projections == 1


@pytest.mark.parametrize('mutation',['event','time','duplicate','missing','extra','hash','value','identity'])
def test_rejects_unbound_or_numerically_wrong_witness(receiver,checker,branch_fixture,mutation):
    db,k,config,audit,prediction = branch_fixture
    row = record(prediction,prediction['predictions'])
    if mutation == 'event': row['event_id'] = 'different'
    elif mutation == 'time': row['decision_us'] += 1
    elif mutation == 'duplicate': row['branches'].append(copy.deepcopy(row['branches'][0]))
    elif mutation == 'missing': row['branches'] = []
    elif mutation == 'extra': row['branches'].append(dict(component=1,handle=2,predictions=[]))
    elif mutation == 'hash': row['branches'][0]['predictions'][0]['mean'][0] += 1e-12
    else:
        values = row['branches'][0]['predictions']
        if mutation == 'value': values[0]['mean'][0] += 1e-3
        else: values[0]['track_id'] = 'wrong-identity'
        # Even consistent producer hashes do not prove numerical correctness.
        audit['components'][0]['branches'][0]['state_sha256'] = checker.digest(values)
        prediction['predictions'] = values
    with pytest.raises(AssertionError):
        witness = receiver.WitnessEvent(row,0,'e',prediction,audit,checker)
        kernel = receiver.PortableProjection(k,1,witness,checker)
        checker.bind_branch_hashes(audit,prediction,lambda c:kernel,config)
        witness.finish()


def test_unchosen_branch_is_not_skipped(receiver,checker,branch_fixture):
    db,k,config,audit,prediction = branch_fixture
    db.execute("INSERT INTO pc1_prefixes SELECT 2,parent,depth,choice,root,previous_root,'second-prefix' FROM pc1_prefixes WHERE h=1")
    db.execute('INSERT INTO pc1_states SELECT 2,payload FROM pc1_states WHERE h=1')
    k = checker.StoredProjection(db,1)
    values = copy.deepcopy(prediction['predictions'])
    values[0]['mean'][0] += .01
    c = audit['components'][0]
    c['active'].append(dict(handle=2))
    c['branches'].append(dict(handle=2,sha256='second-prefix',state_sha256=checker.digest(values)))
    row = record(prediction,prediction['predictions'])
    row['branches'].append(dict(component=1,handle=2,predictions=values))
    witness = receiver.WitnessEvent(row,0,'e',prediction,audit,checker)
    with pytest.raises(AssertionError,match='numerical mismatch'):
        checker.bind_branch_hashes(audit,prediction,lambda c:receiver.PortableProjection(k,1,witness,checker),config)


@pytest.mark.parametrize('tail',['','{}\n'])
def test_complete_correspondence_binds_requests_audits_and_final_state(receiver,checker,branch_fixture,tail):
    import sqlite3
    source,k,cfg,a,p = branch_fixture
    raw = k.rows[0]
    config = dict(prefix_cache_entries=4096,state=dict(cfg,window_us=2_000_000))
    execution = dict(recipe='independent-root-wave-float64-state-candidate-v1',device='cuda:0',max_batch=64)
    caps = dict(SQL_templates=256,ancestor_pairs_per_call=4096,raw_observations=4096)
    p.update(frame_id='f',previous_commit_sha256='0'*64); p['commit_sha256'] = checker.digest(p)
    a.update(event_id='e',sequence_id='s',observation_count=1,new_observations=1,appended_rows=[[[-1,0.]]],
        rescored_rows=[],decision_indices=[0],cache_ingestion=None,scorer_binding=None,
        prediction_sha256=p['commit_sha256'],previous_audit_sha256='0'*64)
    databases = []
    for side in (0,1):
        db = sqlite3.connect(':memory:')
        source.commit(); source.backup(db)
        db.executescript('CREATE TABLE observations(i INTEGER PRIMARY KEY,state_us INTEGER,raw BLOB);'
            'CREATE TABLE events(ordinal INTEGER PRIMARY KEY,event_id TEXT,request TEXT,prediction BLOB,audit BLOB);'
            'CREATE TABLE component_summaries(component INTEGER PRIMARY KEY,payload BLOB);'
            'CREATE TABLE meta(k TEXT PRIMARY KEY,v BLOB);')
        db.execute('INSERT INTO observations VALUES(0,0,?)',(json.dumps(raw),))
        audit = copy.deepcopy(a)
        if side: audit.update(state_execution=execution,execution_extra_cache_caps=caps,state_execution_independently_accepted=False)
        request = checker.digest(['f',1_000_000,1_000_000,[raw],a['appended_rows'],[],[0] if side else None,None,None])
        db.execute('INSERT INTO events VALUES(0,?,?,?,?)',('e',request,json.dumps(p),json.dumps(audit)))
        db.execute('INSERT INTO component_summaries VALUES(1,?)',(json.dumps(audit['components'][0]),))
        db.execute('INSERT INTO meta VALUES(?,?)',('state',json.dumps(dict(n=1,events=1,prediction_sha256=p['commit_sha256'],audit_sha256=checker.digest(audit)))))
        databases.append(db)
    try:
        stream = io.StringIO(json.dumps(record(p,p['predictions']))+'\n'+tail)
        if tail:
            with pytest.raises(AssertionError,match='surplus witness'):
                receiver.correspondence(*databases,execution,config,stream,checker)
        else:
            result = receiver.correspondence(*databases,execution,config,stream,checker)
            assert result['events'] == result['witness_projections'] == 1
            assert result['branch_commitments'] == result['request_commitments'] == 2
    finally:
        for db in databases: db.close()
