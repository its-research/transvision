"""Generated producer contract checks, not a real teacher experiment."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def built(monkeypatch):
    directory = Path(__file__).resolve().parents[2]/'tools/event_track_v2x'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('teacher_builder', directory/'prepare_rbf_final_refit_capacity_teacher.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    if not builder.LINUX_PROOF.exists():
        pytest.skip('frozen Linux witness assets unavailable; not an admission')
    parent, linux = ((p/'bootstrap.py').read_text() for p in (builder.PARENT, builder.LINUX))
    proof = json.loads(builder.LINUX_PROOF.read_bytes())
    candidate, control = builder.build(parent, linux, proof)
    return builder, parent, linux, proof, candidate, control


def test_source_change_rejected_and_only_named_factory_changed(built):
    builder, parent, linux, proof, candidate, control = built
    with pytest.raises(AssertionError):
        builder.build(parent+'\n', linux, proof)
    assert control['original_replacements_unchanged']
    assert control['real_replay_function_AST_identical']
    tree = ast.parse(candidate)
    # Every arithmetic, factor comparison, sequence loop and failure/archive
    # statement in the GPU worker remains the original implementation.
    expected = ast.get_source_segment(parent, builder.function(parent, 'work')).replace(
        '.exclusive_paper_runtime import', '.exclusive_witness_paper_runtime import').replace(
        "allocation='bound'", "allocation='teacher'")
    assert ast.dump(builder.function(expected, 'work'), include_attributes=False) == ast.dump(
        builder.function(candidate, 'work'), include_attributes=False)
    assert builder.literals(candidate)['REPLACEMENTS'] == builder.literals(parent)['REPLACEMENTS']
    assert set(builder.literals(candidate)['PATCHES']) - set(builder.literals(parent)['PATCHES']) == {
        builder.PREFIX+n for n in builder.ADDED}


@pytest.mark.parametrize('corrupt', [None, 'main_status', 'main_code', 'main_recipe', 'model',
                                    'budget', 'inputs', 'partial', 'proof_hash', 'witness_status',
                                    'witness_code', 'witness_bytes'])
def test_remote_gate_rejects_unqualified_teacher_before_data_replay(built, tmp_path, corrupt):
    builder, parent, linux, witness_proof, candidate, _ = built
    main_plan = copy.deepcopy(json.loads((builder.PARENT/'preparation.json').read_bytes())['seeds'][0]['plan'])
    plan = copy.deepcopy(main_plan)
    plan['configuration']['allocation'] = 'teacher'
    proof = dict(kind='rbf_final_refit_main_prerequisite_for_capacity_teacher_v1',
        main_prerequisite_verified=True, teacher_runtime_or_targets_admitted=False,
        seed=plan['seed'], final_model_sha256=plan['final_refit_model_sha256'],
        sequence_proof_sha256={f'sequence-{i:02d}.json': 'a'*64 for i in range(46)},
        main_acceptance_sha256='b'*64, byte_admission_sha256='c'*64, main_task_id='main')
    plan.update(final_main_prerequisite_artifact={'fixture': True},
        final_main_acceptance_sha256='b'*64, final_main_byte_admission_sha256='c'*64)
    import hashlib
    canonical = lambda v: json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    params = {'General/plan': canonical(main_plan).decode(),
              'General/recipe_sha256': hashlib.sha256(canonical(main_plan)).hexdigest()}
    main_task = SimpleNamespace(status='completed', data=SimpleNamespace(script=SimpleNamespace(diff=parent)),
                                get_parameters=lambda: params)
    witness_task = SimpleNamespace(status='completed', data=SimpleNamespace(script=SimpleNamespace(diff=linux)),
        artifacts={key: SimpleNamespace(hash=a['sha256'], size=a['bytes'])
                   for key, a in witness_proof['registered_artifacts'].items()})
    if corrupt == 'main_status': main_task.status = 'in_progress'
    elif corrupt == 'main_code': main_task.data.script.diff += '\n'
    elif corrupt == 'main_recipe': params['General/recipe_sha256'] = '0'*64
    elif corrupt == 'model': plan['final_refit_model_sha256'] = '0'*64
    elif corrupt == 'budget': plan['configuration']['limits']['max_decision_nodes'] += 1
    elif corrupt == 'inputs': plan['cache_manifest']['sha256'] = '0'*64
    elif corrupt == 'partial': proof['sequence_proof_sha256'].pop('sequence-00.json')
    elif corrupt == 'proof_hash': plan['final_main_acceptance_sha256'] = '0'*64
    elif corrupt == 'witness_status': witness_task.status = 'failed'
    elif corrupt == 'witness_code': witness_task.data.script.diff += '\n'
    elif corrupt == 'witness_bytes': next(iter(witness_task.artifacts.values())).size += 1
    tasks = {'main': main_task, witness_proof['task_id']: witness_task}
    def fetch(spec, destination):
        assert spec == {'fixture': True}
        destination.write_bytes(canonical(proof))
    namespace = dict(json=json, hashlib=hashlib, canonical=canonical, fetch=fetch,
        Task=SimpleNamespace(get_task=lambda task_id: tasks[task_id]),
        PARENT_MAIN_SHA=builder.PARENT_SHA, WITNESS_BOOTSTRAP_SHA=builder.LINUX_SHA,
        WITNESS_ADMISSION=dict(task_id=witness_proof['task_id'], registered_artifacts=witness_proof['registered_artifacts']))
    node = builder.function(candidate, 'teacher_admission')
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<generated-teacher-admission>', 'exec'), namespace)
    if corrupt:
        with pytest.raises(AssertionError):
            namespace['teacher_admission'](plan, tmp_path)
    else:
        namespace['teacher_admission'](plan, tmp_path)
