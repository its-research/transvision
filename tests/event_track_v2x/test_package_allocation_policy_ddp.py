"""Packaging checks never authorize synthetic data as a real training run."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from tools.event_track_v2x import package_allocation_policy_ddp as tool
from transvision.models.event_track_v2x.detection_cache_v2 import canonical,sha_file
from test_allocation_training import priority_data
from test_forest_training_data import prepared_rows


@pytest.fixture
def teacher_header():
    scenes=[f'{i:04d}' for i in range(46)]
    manifest=dict(binding={'known':'binding'},shards=[dict(sequence_id=s) for s in scenes])
    receipt=dict(status='complete',allocation_teacher=True,cache_split='train',completed_frames=7445,
        scheduled_frames=7445,sequence_heads={s:{} for s in scenes},learned_identity_enabled=True)
    final=dict(receipt,full_official_train_trace_completed=True)
    plan=dict(kind='component_allocation_teacher_full_train_plan_v1',full_official_train_verified=True,
        input_full_official_train_verified=True,development_sequence=None,scheduled_frames=7445,
        class_scope=['car'],cache_sha256=tool.TRAIN_CACHE_SHA256,cooperative_metadata_sha256=tool.PAIR_SHA256,
        geometry_development=False,allocation_training_binding=manifest['binding'],source_sha256=tool.allocation_sources())
    return manifest,receipt,final,plan


def test_full_teacher_header_contract(teacher_header):
    tool.check_teacher_contract(*teacher_header)


@pytest.mark.parametrize('bad',['incomplete','development','one_sequence','geometry','car','cache','pairs','binding',
                              'source','false_final','final_changed','exported_sequences'])
def test_full_teacher_claim_cannot_override_actual_receipts_or_cohort(teacher_header,bad):
    manifest,receipt,final,plan=copy.deepcopy(teacher_header)
    if bad=='incomplete':receipt['completed_frames']=195
    elif bad=='development':plan['development_sequence']='0000'
    elif bad=='one_sequence':receipt['sequence_heads']={'0000':{}}
    elif bad=='geometry':plan['geometry_development']=True
    elif bad=='car':plan['class_scope']=['pedestrian']
    elif bad=='cache':plan['cache_sha256']='different'
    elif bad=='pairs':plan['cooperative_metadata_sha256']='different'
    elif bad=='binding':plan['allocation_training_binding']={}
    elif bad=='source':plan['source_sha256']={}
    elif bad=='false_final':final['full_official_train_trace_completed']=False
    elif bad=='final_changed':final['completed_frames']=7444
    else:manifest['shards'][0]['sequence_id']='outside'
    with pytest.raises(ValueError):tool.check_teacher_contract(manifest,receipt,final,plan)


@pytest.mark.parametrize('bad',[None,'missing','duplicate','outside','nonteacher','cross_source'])
def test_every_teacher_event_must_match_frozen_full_train_pairs(tmp_path,bad):
    pairs=[dict(vehicle_sequence=f'{i%46:04d}',infrastructure_sequence=f'{i%46:04d}',vehicle_frame=f'{i:06d}')
           for i in range(7445)]
    events=[dict(tracking=dict(sequence_id=p['vehicle_sequence'],event_id=p['vehicle_frame'],
        training_trace_only=True,offline_counterfactual_probes=True)) for p in pairs]
    if bad=='missing':events.pop()
    elif bad=='duplicate':events[0]=events[1]
    elif bad=='outside':events[0]['tracking']['event_id']='outside'
    elif bad=='nonteacher':events[0]['tracking']['offline_counterfactual_probes']=False
    elif bad=='cross_source':pairs[0]['infrastructure_sequence']='outside'
    trace=tmp_path/'tracking.jsonl';trace.write_bytes(b''.join(canonical(r)+b'\n' for r in events))
    if bad:
        with pytest.raises(ValueError):tool.check_event_coverage(trace,pairs)
    else:tool.check_event_coverage(trace,pairs)


def test_production_pack_refuses_real_fixture_before_creating_output(priority_data,tmp_path):
    data,*_=priority_data
    with pytest.raises(ValueError,match='complete 46-sequence'):
        tool.pack(data,sha_file(data/'manifest.json'),tmp_path/'teacher','a'*64,tmp_path/'pairs',tmp_path/'package')
    assert not (tmp_path/'package').exists()


def test_packaged_source_dependency_closure_imports_in_a_fresh_process(tmp_path):
    records=tool.source_inventory();archive=tmp_path/'source.tar.gz'
    tool.write_archive(tool.ROOT,records,archive,16*1024**2)
    with tarfile.open(archive,'r:gz') as packed:
        assert set(packed.getnames())=={r['path'] for r in records}
        assert all(m.isfile() and m.name.endswith('.py') for m in packed.getmembers())
        packed.extractall(tmp_path/'isolated',filter='data')
    command='''import json
from tools.event_track_v2x.train_allocation_policy_ddp import priority_ddp_sources,priority_model
assert tuple(priority_model(32)[0].weight.shape)==(32,18)
print(json.dumps(priority_ddp_sources(),sort_keys=True))
'''
    result=subprocess.run([sys.executable,'-c',command],cwd=tmp_path/'isolated',
        env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)==tool.priority_ddp_sources()


def test_archive_caps_and_changed_input_rejected(tmp_path):
    path=tmp_path/'numeric.jsonl';path.write_bytes(b'{}\n')
    records=[dict(path=path.name,bytes=path.stat().st_size,sha256=sha_file(path))]
    with pytest.raises(ValueError,match='size cap'):tool.write_archive(tmp_path,records,tmp_path/'too-big.tar.gz',1)
    assert not (tmp_path/'too-big.tar.gz').exists()
    path.write_bytes(b'{"changed":true}\n')
    with pytest.raises(ValueError,match='input changed'):
        tool.write_archive(tmp_path,records,tmp_path/'changed.tar.gz',100)


def test_bound_header_refuses_changed_digest(tmp_path):
    (tmp_path/'receipt.json').write_bytes(b'{}')
    assert tool.read_bound(tmp_path,'receipt.json',sha_file(tmp_path/'receipt.json'))=={}
    with pytest.raises(ValueError,match='evidence changed'):tool.read_bound(tmp_path,'receipt.json','0'*64)
