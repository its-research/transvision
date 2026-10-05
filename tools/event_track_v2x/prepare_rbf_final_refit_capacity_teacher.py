"""Prepare the real final-refit witness teacher, without upload or dispatch."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import tarfile
import zlib

from rbf_nested_seen_val_v2_common import R, new, register, sha

PARENT = R/'source-freezes/rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004'
LINUX = R/'source-freezes/rbf-exclusive-capacity-teacher-Linux-CPU-witness-controls-v1-20261002'
LINUX_PROOF = R/'artifacts/rbf-exclusive-capacity-teacher-Linux-CPU-witness-controls-v1-20261002/acceptance.json'
PARENT_SHA = 'e363579c19831420b9782eb7aef4d96e50f3ab303239942ed78923ee481929dd'
LINUX_SHA = 'f534af41526bde04524ef256b5e0c8c63ca7893ebc9b8a26ef5f905eaec6a241'
PROOF_SHA = '25744960d42c432a6ec349ca2cfa7ddda1cb9080facc0cc102d90338e7f01c40'
PREFIX = 'transvision/models/event_track_v2x/'
ADDED = ('exclusive_teacher_probe_witness.py', 'exclusive_witness_paper_runtime.py')


def literals(source):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(source).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ('PATCHES', 'REPLACEMENTS', 'SOURCE')}


def unpack(value):
    return zlib.decompress(base64.b64decode(value['data'] if isinstance(value, dict) else value))


def function(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)


def build(parent, linux, proof):
    assert hashlib.sha256(parent.encode()).hexdigest() == PARENT_SHA
    assert hashlib.sha256(linux.encode()).hexdigest() == LINUX_SHA
    original, witness = literals(parent), literals(linux)
    assert proof['task_id'] == 'e547e2d97df94f1caf0e26f252689afc'
    assert proof['actual_Linux_CPU_producer_qualified'] is True
    assert proof['full_real_teacher_target_admission'] is False
    assert proof['trajectory']['all_18_features_and_signed_targets_independently_recomputed'] is True
    assert proof['action']['capacity_undecided_counted_as_optimal'] is False
    for name, entry in original['PATCHES'].items():
        assert hashlib.sha256(unpack(witness['PATCHES'][name])).hexdigest() == entry['sha256']
    for name, entry in original['REPLACEMENTS'].items():
        other = witness['REPLACEMENTS'][name]
        assert entry['before_sha256'] == other['before_sha256']
        # The older Linux probe stores replacement bytes as plain base64;
        # its new modules and the final-main producer use zlib+base64.
        raw = base64.b64decode(other['data'])
        assert hashlib.sha256(raw).hexdigest() == other['sha256'] == entry['sha256']
        assert unpack(entry) == raw
    patches = dict(original['PATCHES'])
    for name in ADDED:
        relative = PREFIX+name
        raw = unpack(witness['PATCHES'][relative])
        assert relative not in patches
        patches[relative] = dict(sha256=hashlib.sha256(raw).hexdigest(),
                                 data=base64.b64encode(zlib.compress(raw, 9)).decode())
    # Same causal replay loop; only the factory returns the witness teacher.
    a = function(unpack(patches[PREFIX+'exclusive_paper_runtime.py']), 'replay')
    b = function(unpack(patches[PREFIX+'exclusive_witness_paper_runtime.py']), 'replay')
    assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    node = next(n for n in ast.parse(parent).body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'PATCHES')
    lines = parent.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = ['PATCHES='+repr(patches)+'\n']
    candidate = ''.join(lines)
    edits = [
        ('from transvision.models.event_track_v2x.exclusive_paper_runtime import replay,default_configuration',
         'from transvision.models.event_track_v2x.exclusive_witness_paper_runtime import replay,default_configuration'),
        ("default_configuration(plan['method'],allocation='bound',width=4)",
         "default_configuration(plan['method'],allocation='teacher',width=4)"),
        ("task_name='final-refit exclusive full-train forest candidate'",
         "task_name='final-refit capacity raw-witness full-train teacher candidate'"),
        ("Path('rbf-original-joint-exclusive-replay')", "Path('rbf-final-refit-capacity-witness-teacher')"),
        ("'kind':'rbf_final_refit_exclusive_full_train_forest_candidate_v1'",
         "'kind':'rbf_final_refit_capacity_witness_full_train_teacher_candidate_v1'"),
        ("'scope':'same exclusive factory with capacity-undecided action handling; unchanged model, limits and full real inputs'",
         "'scope':'same exclusive capacity core and real replay loop; separate offline raw-witness teacher allocation; final model and deployment limits unchanged'"),
        ("'scope':'full paired train, distinct frozen final-refit checkpoint, unchanged exclusive kernel and limits; bound allocator only; full independent forest/learned Stage2/MHT/metrics pending'",
         "'scope':'full paired train final-refit capacity witness teacher; offline probes retained and charged separately; full target admission, learned policy and metrics pending'"),
        ("  for key in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest'):fetch(plan[key],base/key)",
         "  teacher_admission(plan,base)\n  for key in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest'):fetch(plan[key],base/key)"),
    ]
    for before, after in edits:
        assert candidate.count(before) == 1, before
        candidate = candidate.replace(before, after, 1)
    # A future dispatcher must first run the independent final-main gate and
    # independently read back its published proof. Missing proofs fail here.
    helper = '''
def teacher_admission(plan,base):
 spec=plan['final_main_prerequisite_artifact'];fetch(spec,base/'final-main-prerequisite.json')
 proof=json.loads((base/'final-main-prerequisite.json').read_text())
 assert proof['kind']=='rbf_final_refit_main_prerequisite_for_capacity_teacher_v1'
 assert proof['main_prerequisite_verified'] is True and proof['teacher_runtime_or_targets_admitted'] is False
 assert proof['seed']==plan['seed'] and proof['final_model_sha256']==plan['final_refit_model_sha256']
 assert len(proof['sequence_proof_sha256'])==46
 assert proof['main_acceptance_sha256']==plan['final_main_acceptance_sha256']
 assert proof['byte_admission_sha256']==plan['final_main_byte_admission_sha256']
 prior=Task.get_task(task_id=proof['main_task_id']);assert str(prior.status)=='completed'
 assert hashlib.sha256(prior.data.script.diff.encode()).hexdigest()==PARENT_MAIN_SHA
 params=prior.get_parameters();main_plan=json.loads(params['General/plan'])
 assert hashlib.sha256(canonical(main_plan)).hexdigest()==params['General/recipe_sha256']
 assert main_plan['seed']==plan['seed'] and main_plan['method']==plan['method']=='rbf'
 for key in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest','forward_outputs','source_replacements','CPU_capacity_candidate_admission','final_refit_model_sha256','original_nested_model_sha256','numeric_reference_admission_sha256','final_refit_training_byte_admission_sha256','forward_full_byte_and_coverage_admission_sha256','final_refit_NN_numeric_completion_sha256'):
  assert main_plan[key]==plan[key], ('changed final-main teacher lineage',key)
 assert main_plan['configuration']['allocation']=='bound'
 assert plan['configuration']==dict(main_plan['configuration'],allocation='teacher')
 witness=Task.get_task(task_id=WITNESS_ADMISSION['task_id']);assert str(witness.status)=='completed'
 assert hashlib.sha256(witness.data.script.diff.encode()).hexdigest()==WITNESS_BOOTSTRAP_SHA
 assert set(witness.artifacts)==set(WITNESS_ADMISSION['registered_artifacts'])
 for key,record in WITNESS_ADMISSION['registered_artifacts'].items():
  assert witness.artifacts[key].hash==record['sha256'] and witness.artifacts[key].size==record['bytes']

'''
    constants = ('PARENT_MAIN_SHA='+repr(PARENT_SHA)+'\nWITNESS_BOOTSTRAP_SHA='+repr(LINUX_SHA)+
                 '\nWITNESS_ADMISSION='+repr(dict(task_id=proof['task_id'], registered_artifacts=proof['registered_artifacts']))+'\n')
    candidate = candidate.replace('def main():\n', constants+helper+'def main():\n', 1)
    compile(candidate, '<final-refit-capacity-teacher>', 'exec')
    return candidate, dict(patches={k: v['sha256'] for k, v in patches.items()},
                          original_replacements_unchanged=True, real_replay_function_AST_identical=True,
                          original_main_bootstrap_sha256=PARENT_SHA, Linux_witness_bootstrap_sha256=LINUX_SHA)


def main():
    assert sha(LINUX_PROOF) == PROOF_SHA
    proof = json.loads(LINUX_PROOF.read_bytes())
    assert sha(LINUX/'source-freeze.json') == proof['producer_source_freeze_sha256']
    candidate, control = build((PARENT/'bootstrap.py').read_text(), (LINUX/'bootstrap.py').read_text(), proof)
    preparation = json.loads((PARENT/'preparation.json').read_bytes())
    for entry in preparation['seeds']:
        assert entry['plan']['source'] == literals((LINUX/'bootstrap.py').read_text())['SOURCE']
        assert entry['plan']['configuration']['allocation'] == 'bound'
    original_archive = R/'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/source.bytes'
    assert sha(original_archive) == preparation['seeds'][0]['plan']['source']['sha256']
    with tarfile.open(original_archive, 'r:*') as archive:
        names = {member.name for member in archive.getmembers()}
        for name in control['patches']:
            assert name not in names and not any(p.endswith('/'+name) for p in names)
    root = R/'source-freezes/rbf-final-refit-capacity-witness-teacher-producer-v1-20261004'
    root.mkdir(exist_ok=False)
    (root/'bootstrap.py').write_text(candidate)
    for p in (Path(__file__).resolve(), Path(__file__).with_name('rbf_final_refit_teacher_prerequisites.py')):
        (root/p.name).write_bytes(p.read_bytes())
    test = Path(__file__).resolve().parents[2]/'tests/event_track_v2x/test_final_refit_capacity_teacher_producer.py'
    (root/test.name).write_bytes(test.read_bytes())
    test_log = Path('/private/tmp/rbf-final-capacity-teacher-source-tests.log')
    assert '36 passed' in test_log.read_text()
    (root/'software-tests.log').write_bytes(test_log.read_bytes())
    new(root/'source-control.json', control)
    new(root/'preparation.json', dict(kind='rbf_final_refit_capacity_witness_teacher_source_candidate_v1',
        bootstrap_sha256=sha(root/'bootstrap.py'), sources={p.name:sha(p) for p in root.iterdir() if p.is_file()},
        parent_preparation_sha256=sha(PARENT/'preparation.json'), Linux_witness_acceptance_sha256=PROOF_SHA,
        final_main_prerequisite_required=True, final_main_proof_cloud_publication_required=True,
        upload_started=False, GPU_task_created=False, full_teacher_target_admitted=False,
        dispatcher_prepared=False, target_reader_prepared=False, production_training_ready=False,
        remaining=['matching final main full independent acceptance and proof publication',
                   'collision-safe deduplicating dispatcher bound to this producer',
                   'capacity-aware final-model full target readback/trajectory admission',
                   'three-seed teacher export, priority fitting, learned replay and independent evaluation']))
    receipt = R/'receipts/rbf-final-refit-capacity-witness-teacher-producer-preparation-20261004.json'
    new(receipt, dict(kind='rbf_final_refit_capacity_witness_teacher_source_preparation_v1',
        preparation=str(root/'preparation.json'), sha256=sha(root/'preparation.json'),
        actual_teacher_experiment_started=False, full_Stage2_complete=False))
    register(receipt, 'rbf-final-refit-capacity-witness-teacher-source-preparation')
    print(json.dumps(dict(receipt=str(receipt), sha256=sha(receipt))))


if __name__ == '__main__':
    main()
