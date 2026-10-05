"""Prepare final-model learned replay without fitting, uploading or dispatching.

The original replay/forest bytes stay fixed. Real use still requires complete
teacher/priority admission and a separately published, independently read-back
checkpoint. This file does not manufacture either of those receipts.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import tarfile
import zlib

from rbf_nested_seen_val_v2_common import R, new, register, sha

PARENT = R / 'source-freezes/rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004'
PARENT_SHA = 'e363579c19831420b9782eb7aef4d96e50f3ab303239942ed78923ee481929dd'
CONSUMER = R / 'source-freezes/rbf-final-refit-priority-export-consumer-v1-20261005'
CONSUMER_FREEZE = 'e358113abf6c23bab8cd2bb27f4a2df0d9586ff9ceda88e64679f6b87071b961'
NUMERIC_FREEZE = '250f88ee612c120c2618931414e89c4a68be5ac138cbfd0cca38be7b0e63caf7'
TEACHER_SHA = 'c847e3bef79402d7f8cf755530091f3a25601e07e7295c910f8aaefab8c01359'
PREFIX = 'transvision/models/event_track_v2x/'
ADDED = ('exclusive_teacher_probe_witness.py', 'exclusive_witness_paper_runtime.py',
         'exclusive_priority_admission.py', 'exclusive_allocation_training.py')
LINEAGE = ('source', 'events', 'checkpoint', 'weights_archive', 'cache_archive', 'cache_manifest',
           'forward_outputs', 'source_replacements', 'CPU_capacity_candidate_admission',
           'final_refit_model_sha256', 'original_nested_model_sha256',
           'numeric_reference_admission_sha256', 'final_refit_training_byte_admission_sha256',
           'forward_full_byte_and_coverage_admission_sha256', 'final_refit_NN_numeric_completion_sha256')


def literals(source):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(source).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ('PATCHES', 'REPLACEMENTS')}


def replace_once(source, before, after):
    assert source.count(before) == 1, 'frozen source contract changed: ' + before
    return source.replace(before, after, 1)


def learned_admission(plan, base):
    """Injected into the producer; dependencies are its existing cloud helpers."""
    publication = plan['priority_publication']
    inputs = plan['priority_inputs']
    assert set(inputs) == {'audit', 'checkpoint', 'weights'}
    assert publication['kind'] == 'rbf_final_refit_priority_checkpoint_publication_independent_bytes_v1'
    assert publication['independent_cloud_bytes_verified'] is True
    assert publication['seed'] == plan['seed'] and publication['artifacts'] == inputs
    assert publication['local_numeric_audit_sha256'] == inputs['audit']['sha256']
    assert publication['learned_replay_accepted'] is publication['paper_performance_complete'] is False
    assert len({v['task'] for v in inputs.values()}) == 1
    prior = Task.get_task(task_id=inputs['audit']['task'])
    assert str(prior.status) == 'completed'
    assert set(prior.artifacts) == {v['key'] for v in inputs.values()} and len(prior.artifacts) == 3
    for role, item in inputs.items():
        artifact = prior.artifacts[item['key']]
        assert artifact.hash == item['sha256'] and artifact.size == item['bytes']
    root = base / 'priority'
    root.mkdir()
    for role, filename in (('audit', 'numeric-audit.json'), ('checkpoint', 'checkpoint.json'), ('weights', 'weights.npz')):
        fetch(inputs[role], root / filename)
    audit = json.loads((root / 'numeric-audit.json').read_bytes())
    checkpoint = json.loads((root / 'checkpoint.json').read_bytes())
    assert audit['kind'] == 'rbf_final_priority_local_full_export_and_selected_checkpoint_numeric_audit_v1'
    assert audit['source_freeze_sha256'] == NUMERIC_FREEZE
    assert audit['consumer_freeze_sha256'] == CONSUMER_FREEZE
    assert audit['seed'] == checkpoint['seed'] == plan['seed']
    assert audit['export']['events'] == 7445 and len(audit['export']['sequences']) == 46
    assert audit['export']['exact_export_features_and_targets_verified'] is True
    assert audit['export']['all_export_groups_independently_scored'] is True
    assert audit['producer_model_or_optimizer_imported'] is False
    assert audit['learned_replay_accepted'] is audit['strict_pipeline_isolated_selection'] is audit['paper_performance_complete'] is False
    assert checkpoint['kind'] == 'exclusive_component_priority_model_progress_checkpoint_v1'
    assert checkpoint['split'] == 'train' and checkpoint['full_official_train_trace'] is True
    assert checkpoint['official_validation_or_test_used_for_selection'] is False
    assert checkpoint['strict_pipeline_isolated_selection'] is checkpoint['paper_eligible'] is False
    assert checkpoint['policy_signature'] == audit['policy_signature'] == plan['priority_policy_signature']
    assert checkpoint['weights_sha256'] == inputs['weights']['sha256']
    for role, filename in (('checkpoint', 'checkpoint.json'), ('weights', 'weights.npz')):
        matches = [v for k, v in audit['input_hashes'].items()
                   if Path(k).parts[-2:] == (str(plan['seed']), filename)]
        assert matches == [inputs[role]['sha256']]
    main = Task.get_task(task_id=audit['main_task_id'])
    teacher = Task.get_task(task_id=audit['teacher_task_id'])
    assert set(publication['upstream_tasks']) == {'main', 'teacher'}
    for role, task, expected_source, allocation in (( 'main', main, PARENT_SHA, 'bound'), ('teacher', teacher, TEACHER_SHA, 'teacher')):
        upstream = publication['upstream_tasks'][role]
        assert upstream['task_id'] == audit[role + '_task_id']
        assert upstream['bootstrap_sha256'] == expected_source
        assert str(task.status) == 'completed'
        assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == expected_source
        params = task.get_parameters()
        original = json.loads(params['General/plan'])
        assert hashlib.sha256(canonical(original)).hexdigest() == params['General/recipe_sha256'] == upstream['recipe_sha256']
        assert original['world_size'] in (4, 8)
        expected_artifacts = {'receipt', 'exclusive-source-manifest'} | {f'replay-rank{i}' for i in range(original['world_size'])}
        assert set(task.artifacts) == set(upstream['artifacts']) == expected_artifacts
        for key, item in upstream['artifacts'].items():
            assert task.artifacts[key].hash == item['sha256'] and task.artifacts[key].size == item['bytes']
        assert original['seed'] == plan['seed'] and original['method'] == plan['method'] == 'rbf'
        assert original['configuration']['allocation'] == allocation
        assert plan['configuration'] == dict(original['configuration'], allocation='learned')
        for key in LINEAGE:
            assert original[key] == plan[key], ('priority/main/teacher lineage differs', key)
    write(base / 'priority-input-binding.json', dict(
        kind='rbf_final_refit_learned_replay_priority_input_binding_v1',
        artifacts=inputs, policy_signature=audit['policy_signature'], publication=publication,
        main_task_id=audit['main_task_id'], teacher_task_id=audit['teacher_task_id'],
        local_numeric_audit_sha256=inputs['audit']['sha256'],
        whole_experiment_ETA='unknown', learned_replay_accepted=False, paper_performance_complete=False))
    return audit


POLICY_LOAD = """ from transvision.models.event_track_v2x.exclusive_allocation_training import load_priority,training_binding
 from transvision.models.event_track_v2x.exclusive_completion_tracking import PersistentExclusiveCompletionConfig
 priority_config=PersistentExclusiveCompletionConfig(state=PaperForestTrackingConfig(**config['state']),**config['limits'])
 policy,priority_checkpoint=load_priority(base/'priority',plan['priority_inputs']['checkpoint']['sha256'],binding=training_binding(priority_config,scorer.signature,ck['frozen_cache_identity']),require_full_train=True)
 assert priority_checkpoint['seed']==plan['seed'] and policy.signature==plan['priority_policy_signature']
 bound=dict(bound,priority_checkpoint_sha256=plan['priority_inputs']['checkpoint']['sha256'],priority_policy_signature=policy.signature)
"""


def build(parent, source):
    assert hashlib.sha256(parent.encode()).hexdigest() == PARENT_SHA
    old = literals(parent)
    patches = dict(old['PATCHES'])
    for name, entry in {**old['PATCHES'], **old['REPLACEMENTS']}.items():
        assert sha(source / name) == entry['sha256']
    for name in ADDED:
        relative = PREFIX + name
        assert relative not in patches and relative not in old['REPLACEMENTS']
        raw = (source / relative).read_bytes()
        patches[relative] = dict(sha256=hashlib.sha256(raw).hexdigest(),
                                data=base64.b64encode(zlib.compress(raw, 9)).decode())
    node = next(n for n in ast.parse(parent).body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'PATCHES')
    lines = parent.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = ['PATCHES=' + repr(patches) + '\n']
    candidate = ''.join(lines)
    edits = [
        ("default_configuration(plan['method'],allocation='bound',width=4)",
         "default_configuration(plan['method'],allocation='learned',width=4)"),
        (" event_asset=json.loads((base/'events.json').read_text());", POLICY_LOAD + " event_asset=json.loads((base/'events.json').read_text());"),
        ("model_binding=bound,fixture=False)", "model_binding=bound,allocation_policy=policy,fixture=False)"),
        ("'paper_performance_complete':False};start=time.monotonic()", "'paper_performance_complete':False,'priority_policy_signature':policy.signature,'priority_checkpoint_sha256':plan['priority_inputs']['checkpoint']['sha256']};start=time.monotonic()"),
        ("task_name='final-refit exclusive full-train forest candidate'", "task_name='final-refit learned-priority full-train forest candidate'"),
        ("Path('rbf-original-joint-exclusive-replay')", "Path('rbf-final-refit-learned-priority-replay')"),
        ("  for key in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest'):fetch(plan[key],base/key)",
         "  priority_audit=learned_admission(plan,base)\n  for key in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest'):fetch(plan[key],base/key)"),
        ("assert len(ev['events'])==7445 and len(ev['origin_us_by_sequence'])==46", "assert len(ev['events'])==7445 and sorted(ev['origin_us_by_sequence'])==sorted(priority_audit['export']['sequences'])"),
        ("'kind':'rbf_final_refit_exclusive_full_train_forest_candidate_v1'", "'kind':'rbf_final_refit_learned_priority_full_train_forest_candidate_v1'"),
        ("'scope':'same exclusive factory with capacity-undecided action handling; unchanged model, limits and full real inputs'", "'scope':'same exclusive forest and limits; independently admitted frozen learned priority; full original inputs'"),
        ("'scope':'full paired train, distinct frozen final-refit checkpoint, unchanged exclusive kernel and limits; bound allocator only; full independent forest/learned Stage2/MHT/metrics pending'", "'scope':'full paired train final model with learned priority; full independent forest/priority ordering/cost/metrics pending; not pipeline-isolated evaluation'"),
        (" task.upload_artifact('receipt',artifact_object=base/'receipt.json',wait_on_upload=True)", " if (base/'priority-input-binding.json').exists():task.upload_artifact('priority-input-binding',artifact_object=base/'priority-input-binding.json',wait_on_upload=True)\n task.upload_artifact('receipt',artifact_object=base/'receipt.json',wait_on_upload=True)"),
    ]
    for before, after in edits:
        candidate = replace_once(candidate, before, after)
    own = Path(__file__).read_text()
    gate = next(n for n in ast.parse(own).body if isinstance(n, ast.FunctionDef) and n.name == 'learned_admission')
    constants = ''.join(name + '=' + repr(value) + '\n' for name, value in (
        ('PARENT_SHA', PARENT_SHA), ('TEACHER_SHA', TEACHER_SHA), ('CONSUMER_FREEZE', CONSUMER_FREEZE),
        ('NUMERIC_FREEZE', NUMERIC_FREEZE), ('LINEAGE', LINEAGE)))
    candidate = replace_once(candidate, 'def main():\n', constants + ast.get_source_segment(own, gate) + '\n\ndef main():\n')
    compile(candidate, '<final-refit-learned-replay>', 'exec')
    return candidate, dict(original_main_bootstrap_sha256=PARENT_SHA,
                          unchanged_replacements=True, patches={k:v['sha256'] for k,v in patches.items()},
                          priority_runtime_bytes_from_consumer_freeze=CONSUMER_FREEZE,
                          original_factor_comparison_tolerance=dict(atol=1e-4, rtol=1e-4))


def prepare(output):
    assert output.resolve().is_relative_to(R / 'source-freezes') and not output.exists()
    assert sha(CONSUMER / 'source-freeze.json') == CONSUMER_FREEZE
    preparation = json.loads((CONSUMER / 'preparation.json').read_bytes())
    source = CONSUMER / 'source'
    for name, item in preparation['sources'].items():
        assert sha(source / name) == item['sha256'] and (source / name).stat().st_size == item['bytes']
    parent = (PARENT / 'bootstrap.py').read_text()
    candidate, control = build(parent, source)
    # All checkpoint-bound pre-existing modules must match, not just the three core replacements.
    archive = R / 'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/source.bytes'
    plans = json.loads((PARENT / 'preparation.json').read_bytes())['seeds']
    assert sha(archive) == plans[0]['plan']['source']['sha256']
    with tarfile.open(archive, 'r:*') as bundle:
        existing = {m.name:bundle.extractfile(m).read() for m in bundle.getmembers() if m.isfile()}
    decoded = literals(candidate)
    for name, item in {**decoded['PATCHES'], **decoded['REPLACEMENTS']}.items():
        existing[name] = zlib.decompress(base64.b64decode(item['data']))
    for name in preparation['sources']:
        if name.startswith(PREFIX):
            assert name in existing and hashlib.sha256(existing[name]).hexdigest() == preparation['sources'][name]['sha256']
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'bootstrap.py').open('x') as stream: stream.write(candidate)
    for p in (Path(__file__).resolve(), Path(__file__).with_name('rbf_nested_seen_val_v2_common.py')):
        with (output / p.name).open('xb') as stream: stream.write(p.read_bytes())
    test = Path(__file__).resolve().parents[2] / 'tests/event_track_v2x/test_final_refit_learned_producer.py'
    logs = [Path('/private/tmp') / name for name in (
        'rbf-final-refit-learned-producer-tests-20261005.log',
        'rbf-final-refit-learned-producer-tests-20261005-attempt2.log',
        'rbf-final-refit-learned-producer-tests-20261005-numpy126.log')]
    assert '22 passed' in logs[-1].read_text()
    for p in (test, *logs):
        with (output / p.name).open('xb') as stream: stream.write(p.read_bytes())
    new(output / 'source-control.json', control)
    new(output / 'source-freeze.json', dict(kind='rbf_final_refit_learned_replay_source_preparation_v1',
        sources={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir() if p.is_file()},
        consumer_source_freeze_sha256=CONSUMER_FREEZE, numerical_checkpoint_driver_freeze_sha256=NUMERIC_FREEZE,
        full_real_replay_loop_unchanged=True, complete_existing_model_sources_identical=True,
        local_numpy126_software_checks_passed=22, original_fixture_setup_failure_preserved=True,
        Linux_runtime_independently_qualified=False, GPU_runtime_qualified=False,
        actual_priority_fit=False, GPU_task_created=False, learned_replay_accepted=False, full_Stage2_complete=False,
        remaining=['actual full teacher and three-seed fitted checkpoint numerical admission',
                   'checkpoint publication with independent cloud bytes and collision-safe dispatcher',
                   'learned replay runtime qualification and full independent ordering/forest/cost acceptance']))
    register(output / 'source-freeze.json', 'rbf-final-refit-learned-replay-source-preparation')
    return output / 'source-freeze.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = prepare(parser.parse_args().output)
    print(json.dumps(dict(source_freeze=str(result), sha256=sha(result), actual_experiment_started=False)))
