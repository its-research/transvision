import copy
import json
import sqlite3

import pytest

from tools.event_track_v2x import audit_probabilistic_validation_identity as tool


@pytest.fixture
def history(tmp_path):
    """Small explicit test data, not an official sequence or evaluation result."""
    anchors = [(0, 0), (1, 0), (2, 2)]
    observations = [(0, 'n0', 0, 'f0', 0, 50, .9), (1, 'n1', 1, 'f1', 100, 150, .8),
                    (2, 'n2', 0, 'f2', 200, 250, .7)]
    config = dict(state=dict(birth_score=.5, max_age_us=1000, survival_per_second=1., prune_score=.1))
    pairs = []; h = tool.hashlib.sha256()
    for i, (_, root) in enumerate(anchors):
        h.update(tool.native.canonical(anchors[i]) + b'\n')
        def box(r, nodes, first, last, score):
            return dict(track_id='s:' + tool.digest(['forest-birth', 'n' + str(r)])[:24],
                class_label='car', mean=[float(i)], covariance=[[1.]], score=score,
                observation_ids=nodes, birth_state_us=first, last_update_us=last)
        boxes = [box(0, ['n0'] if i == 0 else ['n0', 'n1'], 0, 0 if i == 0 else 100, .9)]
        if i == 2: boxes.append(box(2, ['n2'], 200, 200, .7))
        p = dict(sequence_id='s', frame_id=str(i), predictions=boxes,
            box_reference_timestamp_us=i*100, decision_timestamp_us=i*100+100, commit_sha256=str(i)*64)
        roots = [0] if i == 1 else []
        a = dict(sequence_id='s', event_id=str(i), observation_count=i+1, new_observations=1,
            identity_anchor_sha256=h.hexdigest(), conditional_scans=[dict(indices=[i],
                conditioned_track_roots=roots, source_slot=list(observations[i][2:4]),
                reference_us=i*100, decision_us=i*100+100, allowed=[[True]] if roots else [],
                anchors=[dict(index=i, root=root, representative_parent=0 if i == 1 else -1)])])
        pairs.append((a, p))
    path = tmp_path/'history.sqlite'
    db = sqlite3.connect(path)
    db.executescript('''CREATE TABLE events(ordinal INTEGER,event_id TEXT,prediction BLOB,audit BLOB);
        CREATE TABLE potentials(i INTEGER,p INTEGER); CREATE TABLE meta(k TEXT,v BLOB);
        CREATE TABLE identity_anchors(i INTEGER,root INTEGER);
        CREATE TABLE observations(i INTEGER,node_id TEXT,source INTEGER,frame TEXT,
            state_us INTEGER,arrival_us INTEGER,score REAL);''')
    db.executemany('INSERT INTO potentials VALUES(?,?)', [(0,-1),(1,-1),(1,0),(2,-1)])
    db.executemany('INSERT INTO identity_anchors VALUES(?,?)', anchors)
    db.executemany('INSERT INTO observations VALUES(?,?,?,?,?,?,?)', observations)
    for i, (a, p) in enumerate(pairs):
        db.execute('INSERT INTO events VALUES(?,?,?,?)',(i,str(i),tool.native.canonical(p),tool.native.canonical(a)))
    meta = dict(schema='persistent_single_history_probabilistic_tracker_v1',sequence_id='s',config=config,
        state=dict(n=3,events=3,prediction_sha256=pairs[-1][1]['commit_sha256'],audit_sha256=tool.digest(pairs[-1][0])))
    db.executemany('INSERT INTO meta VALUES(?,?)', [(k,tool.native.canonical(v)) for k,v in meta.items()])
    db.commit(); db.close()
    head = dict(frames=3,prediction_sha256=meta['state']['prediction_sha256'],database=path.name,
                database_sha256=tool.native.sha(path))
    return dict(anchors=anchors,observations=observations,pairs=pairs,path=path,config=config,head=head)


def read(h):
    return tool.inspect_sequence(h['path'], 's', h['head'], iter(h['pairs']), h['config'])


def sync_events(h):
    """Coherent test mutation: checks must reject relations, not only file seals."""
    with sqlite3.connect(h['path']) as db:
        for i,(a,p) in enumerate(h['pairs']):
            db.execute('UPDATE events SET event_id=?,prediction=?,audit=? WHERE ordinal=?',
                       (a['event_id'],tool.native.canonical(p),tool.native.canonical(a),i))


def test_readonly_database_prefix_and_raw_membership(history):
    before=tool.native.sha(history['path']); value=read(history)
    assert value['frames']==3 and value['observations']==3
    assert value['every_historical_anchor_hash_reconstructed']
    assert tool.native.sha(history['path'])==before


@pytest.mark.parametrize('error',['wrong_index','duplicate_index','missing_scan','wrong_root','wrong_parent',
    'birth_parent','future_parent','unknown_conditioned_root','duplicate_roots','forbidden_edge','wrong_source',
    'wrong_reference','wrong_decision','wrong_count','wrong_new_count','wrong_hash','wrong_id',
    'wrong_member','wrong_score','wrong_birth','wrong_last','missing_box','extra_box'])
def test_coherently_changed_event_relations_rejected(history,error):
    a,p=history['pairs'][1];scan=a['conditional_scans'][0]
    if error=='wrong_index':scan['indices']=[2]
    elif error=='duplicate_index':scan['indices']=[1,1]
    elif error=='missing_scan':a['conditional_scans']=[]
    elif error=='wrong_root':scan['anchors'][0]['root']=1
    elif error=='wrong_parent':scan['anchors'][0]['representative_parent']=-1
    elif error=='birth_parent':history['pairs'][0][0]['conditional_scans'][0]['anchors'][0]['representative_parent']=0
    elif error=='future_parent':scan['anchors'][0]['representative_parent']=2
    elif error=='unknown_conditioned_root':scan['conditioned_track_roots']=[2]
    elif error=='duplicate_roots':scan['conditioned_track_roots']=[0,0]
    elif error=='forbidden_edge':scan['allowed']=[[False]]
    elif error=='wrong_source':scan['source_slot']=[0,'other']
    elif error=='wrong_reference':scan['reference_us']+=1
    elif error=='wrong_decision':scan['decision_us']+=1
    elif error=='wrong_count':a['observation_count']=0
    elif error=='wrong_new_count':a['new_observations']=0
    elif error=='wrong_hash':a['identity_anchor_sha256']='f'*64
    elif error=='wrong_id':p['predictions'][0]['track_id']='s:wrong'
    elif error=='wrong_member':p['predictions'][0]['observation_ids']=['n0']
    elif error=='wrong_score':p['predictions'][0]['score']=.8
    elif error=='wrong_birth':p['predictions'][0]['birth_state_us']=1
    elif error=='wrong_last':p['predictions'][0]['last_update_us']=0
    elif error=='missing_box':p['predictions']=[]
    elif error=='extra_box':p['predictions'].append(copy.deepcopy(p['predictions'][0]))
    sync_events(history)
    with pytest.raises(ValueError):read(history)


@pytest.mark.parametrize('query',[
    'DELETE FROM potentials WHERE i=1 AND p=0',
    'UPDATE observations SET arrival_us=999 WHERE i=1',
    "UPDATE observations SET source=0,frame='f0' WHERE i=1",
    'DELETE FROM observations WHERE i=1',
    'UPDATE identity_anchors SET root=2 WHERE i=1',
    'DELETE FROM events WHERE ordinal=2',
    "UPDATE events SET event_id='wrong' WHERE ordinal=1",
    'UPDATE events SET ordinal=10 WHERE ordinal=2',
])
def test_database_tampering_rejected(history,query):
    with sqlite3.connect(history['path']) as db: db.execute(query)
    with pytest.raises(ValueError):read(history)


def test_payload_state_changes_are_reported_without_identity_change(history):
    base=read(history)
    history['pairs'][1][1]['predictions'][0]['mean']=[123.]
    sync_events(history); changed=read(history)
    runs={'jpda-ci':{'s':base},'jpda-kalman':{'s':changed},'pkf':{'s':base}}
    result=tool.summarize(runs)
    assert result['all_historical_identity_maps_identical'] and result['all_non_state_outputs_identical']
    assert result['sequences']['s']['changed_prediction_payload_frames_relative_to_jpda_ci']['jpda-kalman']==1


def test_valid_identity_difference_is_not_hidden_or_rejected(history):
    base=read(history); changed=copy.deepcopy(base)
    changed['identity_anchor_stream_sha256']='different'
    result=tool.summarize({'jpda-ci':{'s':base},'jpda-kalman':{'s':changed},'pkf':{'s':base}})
    assert result['all_historical_identity_maps_identical'] is False


@pytest.mark.parametrize('error',['missing_method','missing_sequence','different_event','different_observations'])
def test_summary_requires_all_paired_cells(history,error):
    base=read(history);runs={r:{'s':copy.deepcopy(base)} for r in tool.evaluation.RULES}
    if error=='missing_method':runs.pop('pkf')
    elif error=='missing_sequence':runs['pkf']={}
    elif error=='different_event':runs['pkf']['s']['events'][0]['frame_id']='other'
    else:runs['pkf']['s']['observations']+=1
    with pytest.raises(ValueError):tool.summarize(runs)


def test_streaming_run_checks_complete_published_sequence(history,tmp_path):
    for name,rows in [('tracking.jsonl',[{'tracking':a} for a,_ in history['pairs']]),
                      ('predictions.jsonl',[p for _,p in history['pairs']])]:
        (tmp_path/name).write_bytes(b''.join(tool.native.canonical(v)+b'\n' for v in rows))
    receipt=dict(sequence_heads={'s':history['head']},completed_frames=3)
    assert tool.fingerprint_run(tmp_path,receipt,history['config'])['s']['frames']==3
    receipt['completed_frames']=4
    with pytest.raises(ValueError,match='coverage'):tool.fingerprint_run(tmp_path,receipt,history['config'])


def test_partial_validation_rejected_before_database_or_gt_access(tmp_path):
    with pytest.raises(ValueError):tool.audit([],tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('error',['missing_method','different_seed','different_checkpoint','different_factors',
    'different_runtime','different_state','different_sources'])
def test_triplet_difference_rejected_before_reading_database(history,tmp_path,monkeypatch,error):
    bounds=[]
    for rule in tool.evaluation.RULES:
        bounds.append(dict(plan=dict(configuration=dict(update_rule=rule,state=copy.deepcopy(history['config']['state'])),
            checkpoint_seed=1337,checkpoint_sha256='same',source_sha256={'same':'source'},runtime=dict(pid=len(bounds),device='cpu')),
            audit=dict(factor_stream_sha256='same')))
    target=bounds[-1]
    if error=='missing_method':target['plan']['configuration']['update_rule']='jpda-ci'
    elif error=='different_seed':target['plan']['checkpoint_seed']=2027
    elif error=='different_checkpoint':target['plan']['checkpoint_sha256']='different'
    elif error=='different_factors':target['audit']['factor_stream_sha256']='different'
    elif error=='different_runtime':target['plan']['runtime']['device']='cuda'
    elif error=='different_state':target['plan']['configuration']['state']['birth_score']=.6
    else:target['plan']['source_sha256']['same']='different'
    it=iter(bounds)
    monkeypatch.setattr(tool.evaluation,'inspect_run',lambda *args:next(it))
    with pytest.raises(ValueError):tool.audit([('fixture',)*4]*3,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_multi_sequence_database_stream_pairing(history,tmp_path):
    second_path=tmp_path/'second.sqlite'
    with sqlite3.connect(history['path']) as src,sqlite3.connect(second_path) as dst:src.backup(dst)
    second_pairs=copy.deepcopy(history['pairs'])
    for a,p in second_pairs:
        a['sequence_id']=p['sequence_id']='t'
        for b in p['predictions']:b['track_id']='t:'+b['track_id'].split(':',1)[1]
    with sqlite3.connect(second_path) as db:
        db.execute('UPDATE meta SET v=? WHERE k=?',(tool.native.canonical('t'),'sequence_id'))
        state=json.loads(db.execute("SELECT v FROM meta WHERE k='state'").fetchone()[0])
        state['audit_sha256']=tool.digest(second_pairs[-1][0])
        db.execute('UPDATE meta SET v=? WHERE k=?',(tool.native.canonical(state),'state'))
        for i,(a,p) in enumerate(second_pairs):
            db.execute('UPDATE events SET prediction=?,audit=? WHERE ordinal=?',
                       (tool.native.canonical(p),tool.native.canonical(a),i))
    head=dict(history['head'],database=second_path.name,database_sha256=tool.native.sha(second_path))
    receipt=dict(sequence_heads={'s':history['head'],'t':head},completed_frames=6)
    for name,rows in [('tracking.jsonl',[{'tracking':a} for a,_ in history['pairs']+second_pairs]),
                      ('predictions.jsonl',[p for _,p in history['pairs']+second_pairs])]:
        (tmp_path/name).write_bytes(b''.join(tool.native.canonical(v)+b'\n' for v in rows))
    result=tool.fingerprint_run(tmp_path,receipt,history['config'])
    assert set(result)=={'s','t'} and sum(r['frames'] for r in result.values())==6
    assert result['s']['identity_anchor_stream_sha256']!=result['t']['identity_anchor_stream_sha256']
