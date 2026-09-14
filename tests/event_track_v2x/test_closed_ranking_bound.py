import json
import sqlite3

import pytest

from tools.event_track_v2x.audit_closed_ranking_bound import audit,sha


def fixture(tmp_path):
    path=tmp_path/'closed.sqlite'
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE component_catalog(component,live);
            INSERT INTO component_catalog VALUES(3,1);
            CREATE TABLE component_members(component,local_i,global_i);
            INSERT INTO component_members VALUES(3,0,0),(3,1,1);
            CREATE TABLE pc3_meta(k,v);
            CREATE TABLE pc3_observations(i,source,frame);
            INSERT INTO pc3_observations VALUES(0,0,'a'),(1,0,'b');
            CREATE TABLE pc3_potentials(i,p,w);
            INSERT INTO pc3_potentials VALUES(0,-1,0.),(1,-1,0.),(1,0,0.);
            CREATE TABLE pc3_prefixes(h,parent,depth,root);
            INSERT INTO pc3_prefixes VALUES(0,NULL,0,NULL),(1,0,1,0),(2,1,2,0);
        ''')
        db.execute('INSERT INTO pc3_meta VALUES(?,?)',('state',json.dumps(dict(n=2,active=[2],decision_us=1))))
    return path


def test_closed_snapshot_is_unchanged_and_does_not_claim_an_online_result(tmp_path):
    path=fixture(tmp_path);before=sha(path);result=audit(path,before)
    assert sha(path)==before
    assert result['components'][0]['node_count']==2
    assert result['components'][0]['samples'][-1]['log_bound_reduction']==pytest.approx(0.)
    for key in ('online_replay','failed_event_reconstructed','top_k_certified','partition_mass_bound',
                'formal_numeric_certificate','ground_truth_used','parameter_training','paper_eligible'):
        assert result[key] is False


@pytest.mark.parametrize('fault',['hash','wal','journal','ancestry','duplicate_slot','budget'])
def test_snapshot_corruption_or_unclosed_state_rejected(tmp_path,fault):
    path=fixture(tmp_path);kwargs={}
    if fault in ('wal','journal'): (tmp_path/('closed.sqlite-'+fault)).touch()
    elif fault in ('ancestry','duplicate_slot'):
        with sqlite3.connect(path) as db:
            db.execute('UPDATE pc3_prefixes SET depth=0 WHERE h=2' if fault=='ancestry'
                       else "UPDATE pc3_observations SET frame='a'")
    elif fault=='budget': kwargs['largest_components']=0
    with pytest.raises(ValueError): audit(path,'0'*64 if fault=='hash' else sha(path),**kwargs)
