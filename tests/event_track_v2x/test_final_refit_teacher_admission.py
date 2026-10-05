"""Software-only scope controls; these fixtures never produce experiment receipts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def target(monkeypatch):
    root=Path(__file__).resolve().parents[2]/'tools/event_track_v2x'
    monkeypatch.syspath_prepend(str(root))
    spec=importlib.util.spec_from_file_location('final_teacher_target_tests',root/'accept_final_refit_capacity_teacher.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def scope_fixture(module):
    sequences=[f'{i:04d}' for i in range(46)]
    events=[dict(sequence_id=sequences[i%46],event_id=f'software-{i}') for i in range(7445)]
    config=dict(method='rbf',allocation='teacher',backend='exclusive_root_partition_regions_v1',
        state=dict(candidate_protocol='rbf-all-class-top64-v1'))
    sources={'software-fixture.py':'1'*64};model='2'*64;checkpoint='3'*64;main='4'*64
    job=dict(task_id='software-only-not-a-task',seed=1337,recipe_sha256='5'*64,
        plan=dict(configuration=config,final_refit_model_sha256=model,checkpoint=dict(sha256=checkpoint),
            original_nested_model_sha256='6'*64,final_main_acceptance_sha256=main,world_size=4,
            cache_manifest=dict(sha256='7'*64)))
    inventory={k:dict(sha256='8'*64,bytes=1) for k in module.artifact_keys(4)}
    byte=dict(kind=module.BYTE_KIND,task_id=job['task_id'],seed=1337,recipe_sha256=job['recipe_sha256'],
        world_size=4,registered_artifacts=inventory,all_registered_bytes_verified=True,
        all_7445_events_and_46_native_sequences_byte_bound=True,all_final_model_factor_rows_independently_bound=True,
        full_teacher_factor_state_action_target_admission=False,original_events_inferred_or_completed=False,
        final_model_sha256=model,final_checkpoint_sha256=checkpoint,main_replay_admission_sha256=main,
        completed_events=7445,completed_sequences=sequences,labels=1,NN_atol=1e-4,NN_rtol=1e-4,
        final_model_factor_reports={s:dict(rows=1,atol=1e-4,rtol=1e-4,max_abs_error=0.) for s in sequences})
    receipt=dict(collection_kind=module.KIND,fixture=False,full_real_teacher_target_admission=False,
        completed_events=7445,completed_sequences=sequences,databases={s:dict(path=f'{s}.sqlite') for s in sequences})
    plan=dict(configuration=config,source_sha256=sources,cache_sha256='7'*64,fixture=False,
        protocol=dict(dataset='spd',split='train'),model_binding=dict(dataset='spd',fit_split='train',seed=1337,
            model_sha256=model,checkpoint_sha256=checkpoint),expected_events=7445,expected_sequences=sequences,
        events_sha256=hashlib.sha256(module.canonical(events)).hexdigest())
    binding=dict(fixture=False,GT_read=False,test_read=False,original_plan=job['plan'],
        registered_artifacts=inventory,recipe_sha256=job['recipe_sha256'],task_id=job['task_id'],seed=1337,
        expected_runtime_sources=sources,final_model_sha256=model,main_replay_admission_sha256=main,labels=1)
    return byte,receipt,plan,binding,events,job,main


def test_full_scope_metadata_control_does_not_claim_numeric_acceptance(target):
    values=scope_fixture(target)
    assert len(target.scope(*values))==46
    assert values[0]['full_teacher_factor_state_action_target_admission'] is False
    assert values[1]['full_real_teacher_target_admission'] is False


@pytest.mark.parametrize('mutation',[
    'old-model','test-input','fixture','missing-event','duplicate-event','rank-count',
    'missing-rank','cache','source','checkpoint','missing-factors','relaxed-tolerance','NaN-factor'])
def test_wrong_lineage_or_partial_scope_refused(target,mutation):
    byte,receipt,plan,binding,events,job,main=copy.deepcopy(scope_fixture(target))
    if mutation=='old-model':plan['model_binding']['model_sha256']=job['plan']['original_nested_model_sha256']
    elif mutation=='test-input':binding['test_read']=True
    elif mutation=='fixture':receipt['fixture']=True
    elif mutation=='missing-event':events.pop()
    elif mutation=='duplicate-event':events[-1]=copy.deepcopy(events[0])
    elif mutation=='rank-count':byte['world_size']=8
    elif mutation=='missing-rank':byte['registered_artifacts'].pop('replay-rank3')
    elif mutation=='cache':plan['cache_sha256']='9'*64
    elif mutation=='source':plan['source_sha256']={'changed.py':'0'*64}
    elif mutation=='checkpoint':byte['final_checkpoint_sha256']='9'*64
    elif mutation=='missing-factors':byte['final_model_factor_reports'].pop('0000')
    elif mutation=='relaxed-tolerance':byte['final_model_factor_reports']['0000']['rtol']=1e-3
    elif mutation=='NaN-factor':byte['final_model_factor_reports']['0000']['max_abs_error']=float('nan')
    with pytest.raises(AssertionError):target.scope(byte,receipt,plan,binding,events,job,main)


def test_frozen_oracles_load_without_producer_or_real_data_replay(target):
    root=Path(target.__file__).parent
    code='''
import json,sys
sys.path.insert(0,sys.argv[1])
import accept_final_refit_capacity_teacher as m
driver,refs=m.modules()
assert set(refs)=={'fresh','trajectory','action','cache203'}
assert driver.verify_sequence.__module__=='final_teacher_original_independent_oracles'
assert m.sha(refs['cache203'].__file__)==m.FINAL_CACHE_SHA
assert 'torch' not in sys.modules and 'scipy' not in sys.modules
assert not any(n=='transvision' or n.startswith('transvision.') for n in sys.modules)
assert refs['trajectory'].numeric.ATOL == refs['trajectory'].numeric.RTOL == 1e-8
print(json.dumps({'sources_loaded':True,'real_sequence_executed':False}))
'''
    result=subprocess.run([sys.executable,'-c',code,str(root)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['real_sequence_executed'] is False


def test_changed_ambient_oracle_refused(target,monkeypatch):
    import types
    monkeypatch.setitem(sys.modules,'teacher_causal',types.SimpleNamespace(__file__='/not-the-frozen-oracle.py'))
    with pytest.raises(AssertionError,match='ambient teacher oracle'):
        target.modules()
