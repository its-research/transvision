"""Frozen learned-replay interface qualification; no real GPU or fit claim."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as N

import pytest


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


@pytest.fixture(scope='module')
def built():
    import sys
    directory = Path(__file__).resolve().parents[2] / 'tools/event_track_v2x'
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('learned_builder', directory / 'prepare_rbf_final_refit_learned_replay.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    parent = (builder.PARENT / 'bootstrap.py').read_text()
    candidate, control = builder.build(parent, builder.CONSUMER / 'source')
    return builder, parent, candidate, control


def function(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_only_frozen_policy_integration_changes_the_GPU_worker(built):
    builder, parent, candidate, control = built
    expected = ast.get_source_segment(parent, function(parent, 'work'))
    expected = expected.replace("allocation='bound'", "allocation='learned'")
    expected = expected.replace(" event_asset=json.loads((base/'events.json').read_text());", builder.POLICY_LOAD + " event_asset=json.loads((base/'events.json').read_text());")
    expected = expected.replace('model_binding=bound,fixture=False)', 'model_binding=bound,allocation_policy=policy,fixture=False)')
    expected = expected.replace("'paper_performance_complete':False};start=time.monotonic()", "'paper_performance_complete':False,'priority_policy_signature':policy.signature,'priority_checkpoint_sha256':plan['priority_inputs']['checkpoint']['sha256']};start=time.monotonic()")
    assert ast.dump(function(expected, 'work'), include_attributes=False) == ast.dump(function(candidate, 'work'), include_attributes=False)
    assert builder.literals(candidate)['REPLACEMENTS'] == builder.literals(parent)['REPLACEMENTS']
    assert set(builder.literals(candidate)['PATCHES']) - set(builder.literals(parent)['PATCHES']) == {builder.PREFIX+n for n in builder.ADDED}
    assert control['original_factor_comparison_tolerance'] == dict(atol=1e-4, rtol=1e-4)
    with pytest.raises(AssertionError): builder.build(parent+'\n', builder.CONSUMER/'source')


def test_actual_frozen_policy_loader_and_exclusive_binding(built, tmp_path, monkeypatch):
    """Synthetic checkpoint tests the real loader, not a teacher/fit admission."""
    import sys
    import types
    import numpy as np
    builder = built[0]
    source = builder.CONSUMER / 'source'
    monkeypatch.syspath_prepend(str(source))
    for name in ('transvision', 'transvision.models', 'transvision.models.event_track_v2x'):
        module = types.ModuleType(name);module.__path__ = [str(source.joinpath(*name.split('.')))]
        monkeypatch.setitem(sys.modules, name, module)
    from transvision.models.event_track_v2x import exclusive_allocation_training as training
    from transvision.models.event_track_v2x.exclusive_completion_tracking import PersistentExclusiveCompletionConfig
    from transvision.models.event_track_v2x.allocation_policy import FrozenPriorityPolicy, FEATURES, RECIPE, TARGET
    weights = [np.full(shape, 0.01, dtype=np.float64) for shape in ((32,18),(32,),(1,32),(1,))]
    policy = FrozenPriorityPolicy(weights)
    with (tmp_path/'weights.npz').open('xb') as stream:np.savez(stream, **{f'w{i}':v for i,v in enumerate(weights)})
    binding = training.training_binding(PersistentExclusiveCompletionConfig(), 'fixture-scorer', {'fixture-cache':'software-only'})
    checkpoint = dict(kind=training.KIND,split='train',binding=binding,source_sha256=training.allocation_sources(),
        feature_recipe=RECIPE,target_recipe=TARGET,feature_names=list(FEATURES),full_official_train_trace=True,
        weights_sha256=builder.sha(tmp_path/'weights.npz'),policy_signature=policy.signature,
        fixture=True,software_only=True)
    (tmp_path/'checkpoint.json').write_bytes(canonical(checkpoint))
    digest = builder.sha(tmp_path/'checkpoint.json')
    loaded, metadata = training.load_priority(tmp_path,digest,binding=binding)
    assert loaded.signature == policy.signature and metadata['fixture'] is True
    wrong = copy.deepcopy(binding);wrong['factor_scorer_signature'] = 'other-model'
    with pytest.raises(ValueError):training.load_priority(tmp_path,digest,binding=wrong)
    with (tmp_path/'weights.npz').open('ab') as stream:stream.write(b'corruption')
    with pytest.raises(ValueError):training.load_priority(tmp_path,digest,binding=binding)


@pytest.mark.parametrize('corrupt', [None, 'unread_publication', 'extra_artifact', 'partial_export',
    'numeric_driver', 'consumer', 'audit_seed', 'policy', 'weight_binding', 'checkpoint_bytes',
    'validation_selection', 'pipeline_claim', 'main_running', 'teacher_failed', 'main_source',
    'teacher_recipe', 'changed_caps', 'changed_model', 'changed_upstream_artifact', 'missing_rank'])
def test_remote_policy_gate_rejects_unqualified_inputs(built, tmp_path, corrupt):
    builder, parent, candidate, _ = built
    main_plan = copy.deepcopy(json.loads((builder.PARENT/'preparation.json').read_bytes())['seeds'][0]['plan'])
    main_plan['world_size'] = 4
    seed = main_plan['seed']
    teacher_plan = copy.deepcopy(main_plan);teacher_plan['configuration']['allocation'] = 'teacher';teacher_plan['world_size'] = 8
    teacher_source = (builder.R/'source-freezes/rbf-final-refit-capacity-witness-teacher-producer-v1-20261004/bootstrap.py').read_text()
    tasks = {}
    upstream = {}
    for role, config, source in (('main', main_plan, parent), ('teacher', teacher_plan, teacher_source)):
        params = {'General/plan': canonical(config).decode(), 'General/recipe_sha256': hashlib.sha256(canonical(config)).hexdigest()}
        art = {k:N(hash='b'*64,size=20) for k in {'receipt','exclusive-source-manifest'} | {f'replay-rank{i}' for i in range(config['world_size'])}}
        tasks[role] = N(status='completed', artifacts=art, data=N(script=N(diff=source)), get_parameters=lambda p=params:p)
        upstream[role] = dict(task_id=role,bootstrap_sha256=hashlib.sha256(source.encode()).hexdigest(),recipe_sha256=params['General/recipe_sha256'],artifacts={k:dict(sha256=v.hash,bytes=v.size) for k,v in art.items()})
    checkpoint = dict(seed=seed,kind='exclusive_component_priority_model_progress_checkpoint_v1',split='train',
        full_official_train_trace=True, official_validation_or_test_used_for_selection=False,
        strict_pipeline_isolated_selection=False,paper_eligible=False,policy_signature='policy',weights_sha256='w'*64)
    checkpoint_bytes = canonical(checkpoint)
    audit = dict(kind='rbf_final_priority_local_full_export_and_selected_checkpoint_numeric_audit_v1',
        seed=seed, source_freeze_sha256=builder.NUMERIC_FREEZE,consumer_freeze_sha256=builder.CONSUMER_FREEZE,
        policy_signature='policy',main_task_id='main',teacher_task_id='teacher',
        producer_model_or_optimizer_imported=False,learned_replay_accepted=False,
        strict_pipeline_isolated_selection=False,paper_performance_complete=False,
        export=dict(events=7445,sequences={str(i):{} for i in range(46)},
            exact_export_features_and_targets_verified=True,all_export_groups_independently_scored=True),
        input_hashes={f'/fixture/fit/{seed}/checkpoint.json':hashlib.sha256(checkpoint_bytes).hexdigest(),f'/fixture/fit/{seed}/weights.npz':'w'*64})
    plan = copy.deepcopy(main_plan);plan['configuration']['allocation'] = 'learned'
    inputs = {key:dict(task='published',key=key,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
              for key,data in (('audit',canonical(audit)),('checkpoint',checkpoint_bytes),('weights',b'fixture-not-real-npz'))}
    inputs['weights']['sha256'] = 'w'*64  # fetch is a byte-stage stub; real fetch is separately frozen/hash-gated.
    plan.update(priority_inputs=inputs,priority_policy_signature='policy',priority_publication=dict(
        kind='rbf_final_refit_priority_checkpoint_publication_independent_bytes_v1',
        independent_cloud_bytes_verified=True,seed=seed,artifacts=copy.deepcopy(inputs),local_numeric_audit_sha256=inputs['audit']['sha256'],
        upstream_tasks=upstream,learned_replay_accepted=False,paper_performance_complete=False))
    tasks['published'] = N(status='completed',artifacts={k:N(hash=v['sha256'],size=v['bytes']) for k,v in inputs.items()})
    if corrupt == 'unread_publication':plan['priority_publication']['independent_cloud_bytes_verified']=False
    elif corrupt == 'extra_artifact':tasks['published'].artifacts['extra']=N(hash='x',size=1)
    elif corrupt == 'partial_export':audit['export']['events']-=1
    elif corrupt == 'numeric_driver':audit['source_freeze_sha256']='changed'
    elif corrupt == 'consumer':audit['consumer_freeze_sha256']='changed'
    elif corrupt == 'audit_seed':audit['seed']+=1
    elif corrupt == 'policy':plan['priority_policy_signature']='changed'
    elif corrupt == 'weight_binding':checkpoint['weights_sha256']='changed'
    elif corrupt == 'checkpoint_bytes':audit['input_hashes'][f'/fixture/fit/{seed}/checkpoint.json']='changed'
    elif corrupt == 'validation_selection':checkpoint['official_validation_or_test_used_for_selection']=True
    elif corrupt == 'pipeline_claim':checkpoint['strict_pipeline_isolated_selection']=True
    elif corrupt == 'main_running':tasks['main'].status='in_progress'
    elif corrupt == 'teacher_failed':tasks['teacher'].status='failed'
    elif corrupt == 'main_source':tasks['main'].data.script.diff+='\n'
    elif corrupt == 'teacher_recipe':tasks['teacher'].get_parameters()['General/recipe_sha256']='changed'
    elif corrupt == 'changed_caps':plan['configuration']['limits']['max_decision_nodes']+=1
    elif corrupt == 'changed_model':plan['final_refit_model_sha256']='changed'
    elif corrupt == 'changed_upstream_artifact':tasks['main'].artifacts['receipt'].hash='changed'
    elif corrupt == 'missing_rank':tasks['teacher'].artifacts.pop('replay-rank0')
    def fetch(item,path):
        path.write_bytes(canonical(audit) if item['key']=='audit' else canonical(checkpoint) if item['key']=='checkpoint' else b'fixture-not-real-npz')
    namespace=dict(Path=Path,json=json,hashlib=hashlib,canonical=canonical,fetch=fetch,
        write=lambda p,v:p.write_bytes(canonical(v)),Task=N(get_task=lambda task_id:tasks[task_id]),
        **{k:getattr(builder,k) for k in ('PARENT_SHA','TEACHER_SHA','NUMERIC_FREEZE','CONSUMER_FREEZE','LINEAGE')})
    exec(compile(ast.Module(body=[function(candidate,'learned_admission')],type_ignores=[]),'<frozen-policy-gate>', 'exec'),namespace)
    if corrupt:
        with pytest.raises(AssertionError):namespace['learned_admission'](plan,tmp_path)
    else:
        assert namespace['learned_admission'](plan,tmp_path)['seed']==seed
        assert json.loads((tmp_path/'priority-input-binding.json').read_bytes())['learned_replay_accepted'] is False
