"""Independent arithmetic/coverage fixtures, not real checkpoint acceptance."""
import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def code(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
    return importlib.import_module('audit_final_refit_priority_checkpoint')


def data_fixture(code,tmp_path):
    data=tmp_path/'data';cohort=tmp_path/'cohort';data.mkdir();cohort.mkdir()
    groups={'a':[dict(features=[[0.]*18,[1.]*18],targets=[-.5,.5]),dict(features=[[.2]*18],targets=[.9])],
        'b':[dict(features=[[.3]*18],targets=[-.1])]}
    rows=[];events=[];shards=[]
    for scene,gs in groups.items():
        name=scene+'.jsonl';(data/name).write_bytes(b''.join(code.canonical(g)+b'\n' for g in gs))
        shards.append(dict(sequence_id=scene,path=name,sha256=code.sha(data/name),groups=len(gs),rows=sum(len(g['targets']) for g in gs)))
        operations=[]
        for g in gs:
            record=dict(feature_recipe=code.RECIPE,target_recipe=code.TARGET,future_or_gt_inputs=False,labels_are_model_not_true_risk=True,
                candidates=[dict(features=x,target=y) for x,y in zip(g['features'],g['targets'])])
            operations.append(dict(allocation_training=record))
        rows.append(dict(sequence_id=scene,event_id=scene,kind='persistent_exclusive_completion_teacher_raw_probe_witness_v1',
            training_trace_only=True,offline_counterfactual_probes=True,allocation_trace=operations))
        events.append(dict(sequence_id=scene,event_id=scene))
    (cohort/'audit.jsonl').write_bytes(b''.join(code.canonical(r)+b'\n' for r in rows))
    (cohort/'events.json').write_bytes(code.canonical(events))
    receipt=dict(files={n:code.sha(cohort/n) for n in ('audit.jsonl','events.json')})
    arrays=(np.zeros((32,18)),np.zeros(32),np.zeros((1,32)),np.array([np.arctanh(.2)]))
    return data,dict(shards=shards),cohort,receipt,arrays,{'b'}


def test_all_groups_match_and_objective_is_equal_group_not_equal_row(code,tmp_path):
    args=data_fixture(code,tmp_path)
    r=code.verify_export(*args,expected_events=2,expected_sequences=['a','b'],expected_labels=4)
    assert r['events']==2 and r['fit_groups']==2 and r['holdout_groups']==1
    assert r['final_checkpoint_fit_mse']==pytest.approx((.29+.49)/2)
    assert r['final_checkpoint_holdout_mse']==pytest.approx(.09)
    assert r['final_checkpoint_fit_mse']!=pytest.approx((.49+.09+.49)/3)


@pytest.mark.parametrize('mutation',('alter_target','alter_feature','reverse_groups','extra_group','missing_group','input_identity_field','extra_shard','repeat_path'))
def test_data_corruption_rejected_even_with_updated_shard_hash(code,tmp_path,mutation):
    args=list(data_fixture(code,tmp_path));data,manifest=args[:2]
    path=data/'a.jsonl';groups=[json.loads(x) for x in path.read_bytes().splitlines()]
    if mutation=='alter_target':groups[0]['targets'][0]+=.01
    elif mutation=='alter_feature':groups[0]['features'][0][0]+=.01
    elif mutation=='reverse_groups':groups.reverse()
    elif mutation=='extra_group':groups.append(groups[0])
    elif mutation=='missing_group':groups.pop()
    elif mutation=='input_identity_field':groups[0]['identity']='forbidden'
    elif mutation=='extra_shard':manifest['shards'].append(copy.deepcopy(manifest['shards'][0]))
    elif mutation=='repeat_path':manifest['shards'][1]['path']='a.jsonl'
    path.write_bytes(b''.join(code.canonical(x)+b'\n' for x in groups));manifest['shards'][0]['sha256']=code.sha(path)
    with pytest.raises(ValueError):code.verify_export(*args,expected_events=2,expected_sequences=['a','b'],expected_labels=4)


def test_signed_targets_remain_signed_and_infinite_values_rejected(code):
    x,y=code.group_arrays(dict(features=[[0.]*18],targets=[-.5]));assert y[0]==-.5
    with pytest.raises(ValueError):code.group_arrays(dict(features=[[float('nan')]*18],targets=[0.]))
    with pytest.raises(ValueError):code.group_arrays(dict(features=[[0.]*18],targets=[1.01]))


def test_weight_shape_dtype_signature_and_known_forward(code,tmp_path):
    path=tmp_path/'w.npz';a=np.zeros((32,18));a[0,0]=1
    c=np.zeros((1,32));c[0,0]=2
    values=[a,np.zeros(32),c,np.zeros(1)]
    np.savez(path,**{'w'+str(i):v for i,v in enumerate(values)})
    arrays,signature=code.weights(path);assert len(signature)==64
    x=np.zeros((2,18));x[:,0]=[0.,1.]
    assert code.scores(arrays,x)==pytest.approx([0.,np.tanh(2*np.tanh(1))],abs=1e-14)
    values[0]=values[0].astype('float32');np.savez(path,**{'w'+str(i):v for i,v in enumerate(values)})
    with pytest.raises(ValueError):code.weights(path)


def selection_fixture():
    losses=[1.,.09,.09,.2,.3,.4,.5,.6,.7,.8]
    epochs=[dict(epoch=i,training_mse=.2,train_sequence_holdout_mse=loss,fit_groups=2,holdout_groups=1,official_validation_or_test_read=False) for i,loss in enumerate(losses,1)]
    checkpoint=dict(selection='minimum_train_sequence_holdout_mse_earliest_tie',selected_epoch=2,selected_train_holdout_mse=.09)
    summary=dict(fit_groups=2,holdout_groups=1,final_checkpoint_holdout_mse=.09)
    return epochs,checkpoint,summary


def test_selected_checkpoint_independent_mse_and_earliest_tie(code):
    args=selection_fixture();proof=code.verify_selection(*args);assert proof['selected_epoch']==2
    args[1]['selected_epoch']=3
    with pytest.raises(ValueError):code.verify_selection(*args)


@pytest.mark.parametrize('mutation',('missing_epoch','test_selection','nonfinite','group_count','wrong_weights','stored_loss','epoch_order'))
def test_invalid_selection_or_wrong_fixed_weights_rejected(code,mutation):
    epochs,checkpoint,summary=selection_fixture()
    if mutation=='missing_epoch':epochs.pop()
    elif mutation=='test_selection':epochs[0]['official_validation_or_test_read']=True
    elif mutation=='nonfinite':epochs[0]['training_mse']=float('nan')
    elif mutation=='group_count':epochs[0]['fit_groups']=3
    elif mutation=='wrong_weights':summary['final_checkpoint_holdout_mse']=.0901
    elif mutation=='stored_loss':checkpoint['selected_train_holdout_mse']=.0901
    elif mutation=='epoch_order':epochs[0]['epoch']=2
    with pytest.raises(ValueError):code.verify_selection(epochs,checkpoint,summary)


def test_holdout_is_sequence_deterministic_and_disjoint(code):
    sequences=[f'{i:04d}' for i in range(46)]
    fit,held=code.holdout_split(sequences)
    assert len(fit)==37 and len(held)==9 and not set(fit)&set(held)
    assert code.holdout_split(list(reversed(sequences)))==(fit,held)
    with pytest.raises(ValueError):code.holdout_split(['same','same'])


def test_unsafe_input_path_is_rejected(code,tmp_path):
    outside=tmp_path/'file';outside.write_text('payload');root=tmp_path/'root';root.mkdir()
    with pytest.raises(ValueError):code.contained(root,'../file')
    (root/'link').symlink_to(outside)
    with pytest.raises(ValueError):code.contained(root,'link')
