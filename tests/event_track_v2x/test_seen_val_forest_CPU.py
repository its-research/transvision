"""New scope/lineage rejection checks and real input adapter compatibility."""
import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as N

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
b = importlib.import_module('rbf_seen_val_forest_CPU_binding')
c = importlib.import_module('rbf_seen_val_forest_cache203')
p = importlib.import_module('prepare_seen_val_forest_CPU')


def fixture():
    _, _, output, _ = b.frozen_modules()
    plan = dict(seed=2027,method='rbf',world_size=4,configuration=dict(allocation='bound'),
        scoring_atol=1e-4,scoring_rtol=1e-4,scope=b.SCOPE,evaluation_scope=b.SCOPE,
        checkpoint=dict(sha256='ck'),final_refit_model_sha256='model')
    job = dict(seed=2027,task_id='task',plan=plan,publication_sha256='publication',
        recipe_sha256=hashlib.sha256(output.canonical(plan)).hexdigest())
    value = dict(kind=output.KIND,seed=2027,task_id='task',recipe_sha256=job['recipe_sha256'],method='rbf',
        world_size=4,NN_atol=1e-4,NN_rtol=1e-4,all_registered_bytes_verified=True,
        all_21_sequences_3316_events_and_final_model_factor_nodes_verified=True,
        measured_network_arrival_history_verified=False,full_forest_semantics_or_fresh_state_independently_accepted=False,
        learned_Stage2_complete=False,same_resource_performance_accepted=False,paper_performance_complete=False,
        scope=b.SCOPE,input_numeric_index_sha256=output.INDEX_SHA,
        source_sha256=b.sha(b.READER/'read_rbf_seen_val_forest_outputs.py'),publication_sha256='publication',
        final_checkpoint_sha256='ck',final_model_sha256='model',
        artifacts={k:{} for k in output.artifact_keys(4)},total_nodes=21,
        sequences=[dict(sequence_id=f'{i:04d}',events=3316-20 if i==0 else 1,nodes=1,database_sha256='a'*64) for i in range(21)])
    return value,job,output


def test_complete_local_scope_gate():
    value,job,output = fixture()
    assert b.validate_local(value,job,output) == job['plan']


@pytest.mark.parametrize('mutation',[
    'train-receipt','wrong-seed','wrong-task','recipe','topk','world','atol','rtol','partial-bytes',
    'partial-events','measured-arrivals','inherited-forest','stage2','resources','paper','scope','NN-index',
    'reader-source','publication','checkpoint','model','missing-rank','missing-sequence','duplicate-sequence',
    'missing-event','missing-row','unsafe-sequence','negative-rows','database-hash',
])
def test_reject_invalid_scope_before_remote_calls(mutation):
    v,j,o = fixture()
    modifications = {'train-receipt':('kind','train'), 'wrong-seed':('seed',1337), 'wrong-task':('task_id','other'),
        'recipe':('recipe_sha256','other'),'topk':('method','topk'),'world':('world_size',8),
        'atol':('NN_atol',1e-3),'rtol':('NN_rtol',1e-3),'partial-bytes':('all_registered_bytes_verified',False),
        'partial-events':('all_21_sequences_3316_events_and_final_model_factor_nodes_verified',False),
        'measured-arrivals':('measured_network_arrival_history_verified',True),
        'inherited-forest':('full_forest_semantics_or_fresh_state_independently_accepted',True),
        'stage2':('learned_Stage2_complete',True),'resources':('same_resource_performance_accepted',True),
        'paper':('paper_performance_complete',True),'scope':('scope','train'),
        'NN-index':('input_numeric_index_sha256','other'),'reader-source':('source_sha256','other'),
        'publication':('publication_sha256','other'),'checkpoint':('final_checkpoint_sha256','other'),
        'model':('final_model_sha256','other')}
    if mutation in modifications:
        key,value=modifications[mutation];v[key]=value
    elif mutation=='missing-rank':v['artifacts'].pop('replay-rank0')
    elif mutation=='missing-sequence':v['sequences'].pop()
    elif mutation=='duplicate-sequence':v['sequences'][1]['sequence_id']=v['sequences'][0]['sequence_id']
    elif mutation=='missing-event':v['sequences'][0]['events']-=1
    elif mutation=='missing-row':v['total_nodes']+=1
    elif mutation=='unsafe-sequence':v['sequences'][0]['sequence_id']='../other'
    elif mutation=='negative-rows':v['sequences'][0]['nodes']=-1;v['total_nodes']-=2
    elif mutation=='database-hash':v['sequences'][0]['database_sha256']='changed'
    with pytest.raises(AssertionError): b.validate_local(v,j,o)


def input_plan(seed):
    bridge = b.R/f'artifacts/rbf-seen-val-forest-input-bridge-v1-20261005/seed{seed}'
    bound = json.loads((bridge/'input-binding.json').read_bytes())
    inventory = json.loads((bridge/'cache-inventory.json').read_bytes())
    plan = dict(seed=seed,checkpoint=dict(sha256=bound['checkpoint']['sha256']),
        final_refit_model_sha256=bound['checkpoint']['model_sha256'],forward_outputs=bound['forward_artifacts'],
        events=dict(sha256=bound['events_sha256']),cache_manifest=dict(sha256=inventory['cache_manifest_sha256']))
    return bound,dict(seed=seed,plan=plan)


@pytest.mark.parametrize('seed',[1337,2027,3407])
def test_real_cache_adapter_binds_full_schedule_and_original_math(seed):
    bound,job = input_plan(seed)
    admission = c.CacheAdmission(seed,Path(bound['checkpoint']['path']),job)
    assert len(admission.frames)==7189 and len(admission.events)==21
    assert sum(map(len,admission.events.values()))==3316
    assert sum(admission.proof['per_sequence_rows'].values())==bound['rows']
    assert admission.final_model_binding['inherited_train_NN_admission']['rows'] != bound['rows']
    sid=sorted(admission.events)[0];event=admission.events[sid][0]
    rows=[r for d in event['deliveries'] for r in admission.frame(d)]
    reference = b.R/f'artifacts/rbf-matching-seen-val-target-free-original-row-contexts-v1-20261004/seed{seed}'
    manifest=json.loads((reference/'manifest.json').read_bytes())
    shard=next(s for s in manifest['shards'] if s['sequence_id']==sid)
    assert b.sha(reference/shard['path'])==shard['sha256']
    with np.load(reference/shard['path'],allow_pickle=False) as data:
        for name in ('features','mean','covariance','score'):
            # The admitted row transport stores float64 values; unchanged tolerance.
            np.testing.assert_allclose(np.asarray([r[name] for r in rows]),data[name][:len(rows)],atol=1e-8,rtol=1e-8)
        assert [r['node']['node_id'] for r in rows]==data['node_id'][:len(rows)].tolist()
    bad=copy.deepcopy(event['deliveries'][0]);bad['arrival_us']=0
    with pytest.raises(AssertionError):admission.frame(bad)
    bad=copy.deepcopy(event['deliveries'][0]);bad['frame_sha256']='other'
    with pytest.raises(AssertionError):admission.frame(bad)


@pytest.mark.parametrize('seed',[1337,2027,3407])
def test_wrong_checkpoint_rejected(seed):
    bound,job=input_plan(seed)
    job['plan']['checkpoint']['sha256']='other'
    with pytest.raises(AssertionError): c.CacheAdmission(seed,Path(bound['checkpoint']['path']),job)


def test_frozen_driver_all_original_oracles_kept_and_no_short_cohort():
    source,changes=p.derive_driver()
    assert len(changes)==10
    assert 'total_sequences=46' not in source and 'completed_events=7445' not in source
    assert 'completed_events=3316' in source and 'completed_sequences=21' in source
    assert 'formal_independent_evaluation=False' in source
    assert 'sequence_receipts=' in source and 'assert validate(args.byte_admission)==job' in source
    tree=ast.parse(source)
    original=ast.parse((b.ORIGINAL/'accept_final_refit_cohort.py').read_text())
    # Every oracle call and its argument list is unchanged.
    names={'feature_database','audit_database','causal_database','action_database'}
    def calls(t):
        return [ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.Call)
            and ((isinstance(n.func,ast.Name) and n.func.id in names) or
                 (isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='fresh'))]
    assert calls(tree)==calls(original)


def test_no_prefix_admission():
    with pytest.raises(AssertionError):c.verify_database(None,None,None,allow_prefix=True)


def test_generated_entry_imports_without_starting_experiment(tmp_path):
    source,_=p.derive_driver()
    for name in ('rbf_seen_val_forest_CPU_binding.py','rbf_seen_val_forest_cache203.py','rbf_nested_seen_val_v2_common.py'):
        shutil.copyfile(Path(b.__file__).parent/name,tmp_path/name)
    shutil.copyfile(b.ORIGINAL/'decoder_capacity.py',tmp_path/'decoder_capacity.py')
    entry=tmp_path/'accept_seen_val_cohort.py';entry.write_text(source)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,str(entry),'--help'],env=env,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert '--byte-admission' in result.stdout and '--checkpoint' in result.stdout
    assert not (tmp_path/'acceptance.json').exists()


@pytest.mark.parametrize('change',['train','GT','count','origin'])
def test_metadata_scope_and_all_frame_origins_rejected(change):
    bound,job=input_plan(2027)
    root=b.R/'artifacts/rbf-nested-seen-val-matching-V2-full-admission-v1-20261004/seed2027/cache'
    manifest=json.loads((root/'manifest.json').read_bytes())
    checkpoint=json.loads(Path(bound['checkpoint']['path']).read_bytes())
    events=json.loads((b.R/'artifacts/rbf-seen-val-forest-input-bridge-v1-20261005/seed2027/events.json').read_bytes())
    origins=events['origin_us_by_sequence']
    if change=='train':manifest['split']='train'
    elif change=='GT':manifest['gt_in_cache']=True
    elif change=='count':manifest['frames'].pop()
    else:origins[next(iter(origins))]+=1
    with pytest.raises(AssertionError):c.load_metadata(root,manifest,checkpoint,origins)
