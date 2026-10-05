"""Complete fixed-baseline scope gates and unchanged causal/state/cache oracles."""
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
b=importlib.import_module('rbf_seen_val_fixed_CPU_binding')
c=importlib.import_module('rbf_seen_val_fixed_cache203')
p=importlib.import_module('prepare_seen_val_fixed_CPU')


def fixture(width=4):
    _,_,output,_=b.frozen_modules(width)
    plan=dict(seed=2027,baseline_K=width,method='topk',world_size=4,
        configuration=dict(method='topk',allocation='bound',history_features=True,state=dict(active_limit=width,decision_mode='retained')),
        scoring_atol=1e-4,scoring_rtol=1e-4,scope=b.scope(width),evaluation_scope=b.scope(width),
        checkpoint=dict(sha256='ck'),final_refit_model_sha256='model',source=dict(sha256=output.SOURCE_SHA))
    job=dict(seed=2027,K=width,task_id='task',plan=plan,input_receipt_sha256={'publication':'publication'},
        recipe_sha256=hashlib.sha256(output.canonical(plan)).hexdigest())
    value=dict(kind=output.KIND,K=width,seed=2027,task_id='task',recipe_sha256=job['recipe_sha256'],method='topk',
        world_size=4,NN_atol=1e-4,NN_rtol=1e-4,all_registered_bytes_verified=True,
        all_21_sequences_3316_events_and_final_model_factor_nodes_verified=True,
        measured_network_arrival_history_verified=False,full_baseline_semantics_or_fresh_state_independently_accepted=False,
        main_or_train_full_forest_acceptance_inherited=False,learned_Stage2_complete=False,same_resource_performance_accepted=False,paper_performance_complete=False,
        scope=b.scope(width),source_package_sha256=output.SOURCE_SHA,input_numeric_index_sha256=output.shared.INDEX_SHA,
        source_sha256=b.sha(b.READER/'read_rbf_seen_val_fixed_outputs.py'),publication_sha256='publication',
        final_checkpoint_sha256='ck',final_model_sha256='model',artifacts={k:{} for k in output.artifact_keys(4)},total_nodes=21,
        sequences=[dict(sequence_id=f'{i:04d}',events=3296 if i==0 else 1,nodes=1,database_sha256='a'*64) for i in range(21)])
    return value,job,output


@pytest.mark.parametrize('width',[1,4])
def test_complete_local_baseline_scope(width):
    value,job,output=fixture(width);assert b.validate_local(value,job,output,width)==job['plan']


@pytest.mark.parametrize('mutation',['train','K','seed','task','recipe','main','world','atol','rtol','bytes','events',
    'formal','inherited','main-inherited','Stage2','resource','paper','scope','NN-index','reader-source','package-source',
    'publication','checkpoint','model','rank','sequence','duplicate','event-count','row-count','unsafe-sequence','negative-row','database-hash',
    'configuration-K','history','decision-mode'])
def test_incomplete_or_mixed_proofs_rejected_before_remote_calls(mutation):
    v,j,o=fixture()
    changes={'train':('kind','train'),'K':('K',1),'seed':('seed',1337),'task':('task_id','other'),'recipe':('recipe_sha256','other'),
        'main':('method','rbf'),'world':('world_size',8),'atol':('NN_atol',1e-3),'rtol':('NN_rtol',1e-3),
        'bytes':('all_registered_bytes_verified',False),'events':('all_21_sequences_3316_events_and_final_model_factor_nodes_verified',False),
        'formal':('measured_network_arrival_history_verified',True),'inherited':('full_baseline_semantics_or_fresh_state_independently_accepted',True),
        'main-inherited':('main_or_train_full_forest_acceptance_inherited',True),'Stage2':('learned_Stage2_complete',True),
        'resource':('same_resource_performance_accepted',True),'paper':('paper_performance_complete',True),'scope':('scope','train'),
        'NN-index':('input_numeric_index_sha256','other'),'reader-source':('source_sha256','other'),
        'package-source':('source_package_sha256','other'),'publication':('publication_sha256','other'),
        'checkpoint':('final_checkpoint_sha256','other'),'model':('final_model_sha256','other')}
    if mutation in changes:
        key,value=changes[mutation];v[key]=value
    elif mutation=='rank':v['artifacts'].pop('replay-rank0')
    elif mutation=='sequence':v['sequences'].pop()
    elif mutation=='duplicate':v['sequences'][1]['sequence_id']=v['sequences'][0]['sequence_id']
    elif mutation=='event-count':v['sequences'][0]['events']-=1
    elif mutation=='row-count':v['total_nodes']+=1
    elif mutation=='unsafe-sequence':v['sequences'][0]['sequence_id']='../other'
    elif mutation=='negative-row':v['sequences'][0]['nodes']=-1;v['total_nodes']-=2
    elif mutation=='database-hash':v['sequences'][0]['database_sha256']='other'
    else:
        if mutation=='configuration-K':j['plan']['configuration']['state']['active_limit']=1
        if mutation=='history':j['plan']['configuration']['history_features']=False
        if mutation=='decision-mode':j['plan']['configuration']['state']['decision_mode']='all-legal-hamming'
        j['recipe_sha256']=v['recipe_sha256']=hashlib.sha256(o.canonical(j['plan'])).hexdigest()
    with pytest.raises(AssertionError):b.validate_local(v,j,o,4)


def input_plan(seed,width):
    bridge=b.R/f'artifacts/rbf-seen-val-forest-input-bridge-v1-20261005/seed{seed}'
    bound=json.loads((bridge/'input-binding.json').read_bytes());inventory=json.loads((bridge/'cache-inventory.json').read_bytes())
    return bound,dict(seed=seed,K=width,plan=dict(seed=seed,checkpoint=dict(sha256=bound['checkpoint']['sha256']),
        final_refit_model_sha256=bound['checkpoint']['model_sha256'],forward_outputs=bound['forward_artifacts'],
        events=dict(sha256=bound['events_sha256']),cache_manifest=dict(sha256=inventory['cache_manifest_sha256'])))


@pytest.mark.parametrize('width',[1,4])
@pytest.mark.parametrize('seed',[1337,2027,3407])
def test_admitted_val_adapter_binds_full_schedule_and_matches_original_math(seed,width):
    bound,job=input_plan(seed,width);checkpoint=Path(bound['checkpoint']['path'])
    admission=c.CacheAdmission(seed,checkpoint,job)
    model,original,output,_=b.frozen_modules(width)
    assert Path(admission._original.__file__)==b.ORIGINALS[width][0]/'final_cache203_receipt_v2.py'
    assert admission.final_model_binding==b.model_binding(seed,checkpoint,job,model,output)
    assert len(admission.frames)==7189 and len(admission.events)==21
    assert sum(map(len,admission.events.values()))==3316 and sum(admission.proof['per_sequence_rows'].values())==bound['rows']
    assert admission.scope==b.scope(width)
    sid=sorted(admission.events)[0];event=admission.events[sid][0]
    rows=[r for d in event['deliveries'] for r in admission.frame(d)]
    reference=b.R/f'artifacts/rbf-matching-seen-val-target-free-original-row-contexts-v1-20261004/seed{seed}'
    manifest=json.loads((reference/'manifest.json').read_bytes());shard=next(s for s in manifest['shards'] if s['sequence_id']==sid)
    assert b.sha(reference/shard['path'])==shard['sha256']
    with np.load(reference/shard['path'],allow_pickle=False) as data:
        for key in ('features','mean','covariance','score'):
            np.testing.assert_allclose(np.asarray([r[key] for r in rows]),data[key][:len(rows)],atol=1e-8,rtol=1e-8)
        assert [r['node']['node_id'] for r in rows]==data['node_id'][:len(rows)].tolist()
    bad=copy.deepcopy(event['deliveries'][0]);bad['arrival_us']=0
    with pytest.raises(AssertionError):admission.frame(bad)
    bad=copy.deepcopy(event['deliveries'][0]);bad['frame_sha256']='wrong'
    with pytest.raises(AssertionError):admission.frame(bad)


@pytest.mark.parametrize('width',[1,4])
def test_driver_oracles_are_byte_preserved_and_scope_is_complete(width):
    source,changes=p.derive_driver(width);assert len(changes)==15
    path=Path(p.__file__).with_name(f'accept_seen_val_fixed_K{width}_cohort.py');assert path.read_text()==source
    original=(b.ORIGINALS[width][0]/f'accept_final_refit_{"top1" if width==1 else "topk"}_cohort.py').read_text()
    def calls(text):
        return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Call) and
                ((isinstance(n.func,ast.Name) and n.func.id in ('feature_database','causal_database')) or
                 (isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='fresh'))]
    assert calls(source)==calls(original)
    assert 'completed_events=3316' in source and 'completed_sequences=21' in source
    assert 'total_sequences=46' not in source and 'completed_events=7445' not in source
    assert f'assert validate_output(args.byte_admission,width={width})' in source
    assert 'train_or_main_baseline_full_acceptance_inherited=False' in source and 'sequence_receipts=' in source


def test_cache_adapter_edits_do_not_modify_feature_or_context_math():
    source,changes=p.derive_cache();assert len(changes)==4 and source==Path(c.__file__).read_text()
    parent=(b.VAL_CPU/'rbf_seen_val_forest_cache203.py').read_text()
    def function(text,name):
        n=next(v for v in ast.parse(text).body if isinstance(v,ast.FunctionDef) and v.name==name)
        return ast.get_source_segment(text,n)
    assert function(source,'load_metadata')==function(parent,'load_metadata')
    with pytest.raises(AssertionError):c.verify_database(None,None,None,allow_prefix=True)
    for width in (1,4):
        original=(b.ORIGINALS[width][0]/'final_cache203_receipt_v2.py').read_text()
        shared=(b.MODEL_CPU/'final_cache203.py').read_text()
        for name in ('physical_rows','expected_context','verify_database'):assert function(original,name)==function(shared,name)


@pytest.mark.parametrize('change',['train','GT','count','origin'])
def test_metadata_rejects_changed_scope_or_partial_origins(change):
    bound,job=input_plan(2027,4)
    root=b.R/'artifacts/rbf-nested-seen-val-matching-V2-full-admission-v1-20261004/seed2027/cache'
    manifest=json.loads((root/'manifest.json').read_bytes());checkpoint=json.loads(Path(bound['checkpoint']['path']).read_bytes())
    origins=json.loads((b.R/'artifacts/rbf-seen-val-forest-input-bridge-v1-20261005/seed2027/events.json').read_bytes())['origin_us_by_sequence']
    if change=='train':manifest['split']='train'
    elif change=='GT':manifest['gt_in_cache']=True
    elif change=='count':manifest['frames'].pop()
    else:origins[next(iter(origins))]+=1
    with pytest.raises(AssertionError):c.load_metadata(root,manifest,checkpoint,origins)
