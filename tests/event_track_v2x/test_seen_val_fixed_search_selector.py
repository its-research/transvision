"""Scope/lineage and real file orchestration controls, not experiment acceptance."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
import accept_seen_val_fixed_search_selector as d


def sequence(width, entry):
    tag = 'Top1' if width == 1 else 'topK'
    result = dict(kind=f'rbf_independent_fixed_{tag}_raw_graph_search_pruning_audit_v1',
        database_sha256=entry['database_sha256'], sequence_id=entry['sequence_id'],
        events=entry['events'], observations=entry['nodes'], beam_width=width, atol=1e-8, rtol=1e-8,
        raw_edge_components_and_all_member_remaps_verified=True,
        retained_predecessor_products_independently_enumerated=True,
        all_node_local_legal_root_class_topK_choices_verified=True,
        equivalent_parent_path_mass_and_irreversible_pruning_verified=True,
        all_pruning_scope_and_support_completeness_flags_verified=True,
        archived_branches_never_resurrected=True, all_materialized_prefixes_accounted_for=True,
        search_work_counters_and_fixed_caps_verified=True,
        component_loss_scope_weights_and_stored_prefix_weights_verified=True,
        conditional_output_selector_independently_accepted=False, paper_performance_complete=False,
        node_pruning_steps=entry['nodes'], merged_components=0, max_abs_error={'fixture':0.})
    if width == 1:
        result['all_node_local_legal_root_class_Top1_choices_verified'] = result.pop('all_node_local_legal_root_class_topK_choices_verified')
    return result


def selector_result(width, entry):
    tag = 'Top1' if width == 1 else 'topK'
    return dict(kind=f'rbf_independent_fixed_{tag}_conditional_selector_audit_v1',
        database_sha256=entry['database_sha256'], sequence_id=entry['sequence_id'],
        events=entry['events'], observations=entry['nodes'], atol=1e-8, rtol=1e-8, decimal_precision=70,
        upstream_same_byte_search_proof_required=True, pairwise_conditional_risks_independently_verified=True,
        all_outputs_match_frozen_float64_minimum_and_SHA_tie_rule=True,
        complete_and_truncated_conditional_lower_semantics_verified=True, no_MAP_substitution_or_regret_fallback=True,
        conditional_output_selector_independently_accepted=True,
        search_pruning_rerun=False, continuous_states_independently_accepted=False, full_online_method_accepted=False,
        global_full_history_Bayes_accepted=False, true_posterior_certificate=False,
        same_resource_performance_accepted=False, paper_performance_complete=False, max_abs_error={'fixture':0.})


def fixture(width=4):
    value=dict(K=width,method='topk',scope=d.scope(width),seed=2027,task_id='fixture-task',recipe_sha256='a'*64,total_nodes=20,
        sequences=[dict(sequence_id=f'{i:04d}',events=3296 if i==0 else 1,nodes=0 if i==20 else 1,database_sha256='b'*64) for i in range(21)])
    model={'checkpoint':'fixture-checkpoint'}; sources={'input_gate_source_binding':{'fixture':'gate'}}
    search=dict(kind=d.kind(width,'search'),stage='search',K=width,method='topk',scope=d.scope(width),
        task_id=value['task_id'],seed=value['seed'],recipe_sha256=value['recipe_sha256'],final_model_binding=model,source_binding=sources,
        completed_sequences=21,completed_events=3316,observations=20,node_pruning_steps=20,
        all_seen_val_search_pruning_independently_verified=True,conditional_output_selector_independently_accepted=False,
        atol=1e-8,rtol=1e-8,max_abs_error=0.,**{k:False for k in d.FALSE_FLAGS},
        sequence_receipts=[dict(path=f'sequence-{i:02d}.json',sha256='c'*64,sequence_id=e['sequence_id'],database_sha256=e['database_sha256']) for i,e in enumerate(value['sequences'])])
    return value,model,sources,search


@pytest.mark.parametrize('width',[1,4])
def test_complete_scope_including_zero_query_sequence(width):
    value,model,sources,search=fixture(width)
    d.validate_search(search,value,model,sources,width)
    oracle,_=d.oracle_gate(width,'selector')
    for entry in value['sequences']:
        oracle.validate_search_sequence(sequence(width,entry),entry['database_sha256'],entry['sequence_id'],entry['events'],entry['nodes'])
        d.validate_selector_result(selector_result(width,entry),entry,width)


@pytest.mark.parametrize('mutation',['kind','stage','K','method','scope','task_id','seed','recipe_sha256','model','source',
    'count','events','nodes','pruning','accepted','selector','atol','rtol','nan','negative-error','proof-count','proof-order','proof-database',*d.FALSE_FLAGS])
def test_mixed_or_partial_search_cannot_feed_selector(mutation):
    value,model,sources,search=fixture()
    changes={'kind':('kind','train-result'),'stage':('stage','selector'),'K':('K',1),'method':('method','rbf'),
        'scope':('scope','train'),'task_id':('task_id','other-task'),'seed':('seed',1337),'recipe_sha256':('recipe_sha256','other'),
        'model':('final_model_binding',{}),'source':('source_binding',{}),'count':('completed_sequences',20),'events':('completed_events',7445),
        'nodes':('observations',19),'pruning':('node_pruning_steps',19),'accepted':('all_seen_val_search_pruning_independently_verified',False),
        'selector':('conditional_output_selector_independently_accepted',True),'atol':('atol',1e-6),'rtol':('rtol',1e-6),
        'nan':('max_abs_error',float('nan')),'negative-error':('max_abs_error',-1.)}
    if mutation in changes:
        key,value2=changes[mutation];search[key]=value2
    elif mutation=='proof-count':search['sequence_receipts'].pop()
    elif mutation=='proof-order':search['sequence_receipts'].reverse()
    elif mutation=='proof-database':search['sequence_receipts'][0]['database_sha256']='d'*64
    else:search[mutation]=True
    with pytest.raises(AssertionError):d.validate_search(search,value,model,sources,4)


@pytest.mark.parametrize('width',[1,4])
@pytest.mark.parametrize('stage',['search','selector'])
def test_oracles_resolve_to_unchanged_original_freezes(width,stage):
    oracle,binding=d.oracle_gate(width,stage)
    name,digest=d.ORIGINALS[width,stage]
    assert Path(oracle.__file__).parent==d.R/'source-freezes'/name
    assert d.sha(oracle.__file__)==binding['sha256']
    assert binding['original_source_freeze_sha256']==digest
    assert oracle.ATOL==oracle.RTOL==1e-8


def setup_files(tmp_path,monkeypatch,width=4):
    value,model,sources,search=fixture(width)
    # Only dataflow is emulated. No fake receipt is registered outside tmp_path.
    monkeypatch.setattr(d,'R',tmp_path)
    monkeypatch.setattr(d,'source_gate',lambda _:sources)
    byte_path=tmp_path/'artifacts/bytes/proof.json';byte_path.parent.mkdir(parents=True)
    for i,e in enumerate(value['sequences']):
        root=byte_path.parent/'rank0-unpack/rank-0'/e['sequence_id'];root.mkdir(parents=True)
        db=root/'sequence.sqlite';db.write_bytes(f'fixture database {i}'.encode());e['database_sha256']=d.sha(db)
        d.new(root/'receipt.json',dict(databases={e['sequence_id']:dict(path=db.name,sha256=d.sha(db))}))
        search['sequence_receipts'][i]['database_sha256']=d.sha(db)
    d.new(byte_path,value)
    calls=[];registered={str(byte_path):d.sha(byte_path)};gate_calls=[]
    def read_registered(path):
        assert registered[str(path)]==d.sha(path)
        return json.loads(Path(path).read_bytes())
    def input_gate(path,**kw):
        assert path==byte_path and kw=={'width':width};gate_calls.append(True)
        read_registered(path)
        return {'recipe_sha256':value['recipe_sha256']},'checkpoint',model,sources['input_gate_source_binding']
    module=N(registered=read_registered,validate_output=input_gate)
    monkeypatch.setattr(d,'input_module',lambda:module)
    monkeypatch.setattr(d,'register',lambda path,kind:registered.update({str(path):d.sha(path)}))
    # Same frozen per-sequence validator; numerical execution itself is stubbed.
    frozen_name,_=d.ORIGINALS[width,'selector'];base=Path('/Volumes/Data/test/recover-before-fuse/source-freezes')/frozen_name
    stem='top1' if width==1 else 'topk'
    spec=importlib.util.spec_from_file_location('fixture_validation_only',base/f'independent_fixed_{stem}_selector.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    def verify_search(db,expected,progress):
        entry=next(e for e in value['sequences'] if e['sequence_id']==db.parent.name)
        assert expected==entry['database_sha256']==d.sha(db)
        progress(dict(completed_events=entry['events'],total_events=entry['events']))
        calls.append(('search',entry['sequence_id']));return sequence(width,entry)
    def verify_selector(db,expected,proof,progress):
        entry=next(e for e in value['sequences'] if e['sequence_id']==db.parent.name)
        assert expected==entry['database_sha256']==d.sha(db)
        original.validate_search_sequence(proof,expected,entry['sequence_id'],entry['events'],entry['nodes'])
        calls.append(('selector',entry['sequence_id']));return selector_result(width,entry)
    search_oracle=N(verify_database=verify_search)
    selector=N(verify_database=verify_selector,validate_search_sequence=original.validate_search_sequence)
    monkeypatch.setattr(d,'oracle_gate',lambda _,stage:({'search':search_oracle,'selector':selector}[stage],{}))
    args=N(K=width,stage='search',byte_admission=byte_path,search_admission=None,output=tmp_path/'artifacts/search')
    return args,value,calls,gate_calls,registered,module,search_oracle


@pytest.mark.parametrize('width',[1,4])
def test_search_then_selector_dataflow_visits_all_sequences_without_repeating_search(tmp_path,monkeypatch,width,capsys):
    args,value,calls,gate_calls,_,_,_=setup_files(tmp_path,monkeypatch,width)
    search=d.run(args);assert len(calls)==21 and len(gate_calls)==2
    args.stage='selector';args.search_admission=args.output/'acceptance.json';args.output=tmp_path/'artifacts/selector'
    result=d.run(args)
    assert len(calls)==42 and len(gate_calls)==4
    assert calls==[(stage,e['sequence_id']) for stage in ('search','selector') for e in value['sequences']]
    assert result['search_pruning_rerun'] is False and result['completed_events']==3316
    assert result['all_seen_val_conditional_selectors_independently_verified'] is True
    assert search['all_seen_val_search_pruning_independently_verified'] is True
    assert all(result[k] is False for k in d.FALSE_FLAGS)
    assert '"whole_cohort_ETA_seconds": null' in capsys.readouterr().out


@pytest.mark.parametrize('mutation',['unregistered','foreign-byte','changed-sequence','foreign-sequence-path'])
def test_selector_refuses_changed_completed_search_before_output_creation(tmp_path,monkeypatch,mutation):
    args,_,calls,_,registered,_,_=setup_files(tmp_path,monkeypatch)
    d.run(args);args.stage='selector';args.search_admission=args.output/'acceptance.json';args.output=tmp_path/'artifacts/selector'
    path=args.search_admission;search=json.loads(path.read_bytes())
    if mutation=='unregistered':registered.pop(str(path))
    elif mutation=='changed-sequence':Path(search['sequence_receipts'][0]['path']).write_text('{}')
    else:
        if mutation=='foreign-byte':search['byte_admission_sha256']='f'*64
        else:search['sequence_receipts'][0]['path']=str(tmp_path/'foreign.json')
        path.write_text(json.dumps(search));registered[str(path)]=d.sha(path)
    with pytest.raises((AssertionError,KeyError)):d.run(args)
    assert len(calls)==21 and not args.output.exists()


@pytest.mark.parametrize('mutation',['remote-change','byte-change','sequence-proof-change','oracle-failure'])
def test_failed_admission_preserves_partial_evidence_without_acceptance(tmp_path,monkeypatch,mutation):
    args,_,calls,_,_,module,search_oracle=setup_files(tmp_path,monkeypatch)
    original=search_oracle.verify_database
    def verify(*params):
        result=original(*params)
        if len(calls)==2:
            if mutation=='oracle-failure':raise AssertionError('deliberate numerical control')
            if mutation=='sequence-proof-change':(args.output/'sequence-00.json').write_text('{}')
            if mutation=='byte-change':args.byte_admission.write_text('{}')
            if mutation=='remote-change':module.validate_output=lambda *a,**kw:('changed',)*4
        return result
    search_oracle.verify_database=verify
    with pytest.raises(AssertionError):d.run(args)
    assert (args.output/'binding.json').exists() and (args.output/'sequence-00.json').exists()
    assert not (args.output/'acceptance.json').exists()
    failure=json.loads((args.output/'failure.json').read_bytes())
    assert failure['automatic_retry'] is failure['experiment_accepted'] is False


@pytest.mark.parametrize('width',[1,4])
@pytest.mark.parametrize('flag',['pairwise_conditional_risks_independently_verified','no_MAP_substitution_or_regret_fallback',
    'conditional_output_selector_independently_accepted','global_full_history_Bayes_accepted','true_posterior_certificate','search_pruning_rerun'])
def test_selector_refuses_missing_semantics_or_inflated_claims(width,flag):
    value,_,_,_=fixture(width);entry=value['sequences'][0];result=selector_result(width,entry)
    result[flag]=not result[flag]
    with pytest.raises(AssertionError):d.validate_selector_result(result,entry,width)


@pytest.mark.parametrize('mutation',['traversal','symlink'])
def test_database_cannot_escape_bound_sequence(tmp_path,monkeypatch,mutation):
    args,value,_,_,_,_,_=setup_files(tmp_path,monkeypatch)
    entry=value['sequences'][0];root=args.byte_admission.parent/'rank0-unpack/rank-0'/entry['sequence_id']
    if mutation=='traversal':
        p=root/'receipt.json';v=json.loads(p.read_bytes());v['databases'][entry['sequence_id']]['path']='../foreign.sqlite';p.write_text(json.dumps(v))
    else:
        db=root/'sequence.sqlite';db.rename(root/'original.sqlite');db.symlink_to(root/'original.sqlite')
    with pytest.raises(AssertionError):d.database_for(args.byte_admission,entry)
