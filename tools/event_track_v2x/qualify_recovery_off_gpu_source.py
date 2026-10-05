"""Independently reconstruct the code that the remote ablation will execute.

This gate reuses the already qualified recovery runtime and actual checkpoint
load receipt. It does not replay data, validate full forests, or dispatch jobs.
"""
import ast
import base64
import datetime
import hashlib
import json
from pathlib import Path
import tarfile
import zlib

from rbf_nested_seen_val_v2_common import R, new, register, sha

ROOT = R/'source-freezes/rbf-final-refit-recovery-off-bound-GPU-v1-20261005'
CANDIDATE = R/'source-freezes/rbf-recovery-off-original-source-bound-candidate-v3-20261005'
PARENT = R/'source-freezes/rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004'
INPUT = R/'artifacts/rbf-recovery-off-final-checkpoint-input-contract-v1-20261005/acceptance.json'
SOFTWARE = R/'receipts/rbf-Top1-three-seed-and-recovery-source-review-20261005T002847276541Z.json'


def assignments(text):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(text).body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('PATCHES', 'REPLACEMENTS')}


def functions(text):
    return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


def main():
    assert not (ROOT/'source-qualification.json').exists()
    prepared = json.loads((ROOT/'preparation.json').read_bytes())
    evidence = list(prepared['prerequisite_receipts'])
    for name, spec in prepared['execution_sources'].items():
        assert sha(ROOT/name) == spec['sha256'] and (ROOT/name).stat().st_size == spec['bytes']
    for spec in evidence:
        assert sha(spec['path']) == spec['sha256']
    assert sha(INPUT) == '760e966dbeb9b4652c69d6641985ae5be806c179cc6d3dbf6a08261102a1481f'
    assert sha(CANDIDATE/'source-freeze.json') == 'b778aee162f177066da207b70ecc807578be687373583a6b9748fa63b3aa9b70'
    assert sha(SOFTWARE) == '3d1c01b0c4e9d5c4899c8829ec7992a482614a98bdea9f8d833385dd347bceba'
    software = json.loads(SOFTWARE.read_bytes())
    assert software['recovery_candidate_frozen_software_tests_passed'] == 38
    evidence.append(dict(path=str(SOFTWARE), sha256=sha(SOFTWARE)))
    log = next(x for x in software['archived_sources_and_logs']
               if Path(x['path']).name == 'rbf-recovery-off-original-source-frozen-tests-v3-20261005.log')
    assert sha(log['path']) == log['sha256'] and '38 passed' in Path(log['path']).read_text()
    evidence.append(dict(path=log['path'], sha256=log['sha256']))
    source = json.loads((CANDIDATE/'source-freeze.json').read_bytes())['sources']
    origin = R/'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/source.bytes'
    parent_prepared = json.loads((PARENT/'preparation.json').read_bytes())
    plan = prepared['seeds'][0]['plan']
    assert sha(origin) == plan['source']['sha256']
    with tarfile.open(origin) as archive:
        files = {}
        for member in archive.getmembers():
            assert member.isfile() and member.name not in files
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            files[member.name] = archive.extractfile(member).read()
    manifest = json.loads(files.pop('source-manifest.json'))
    assert {x['path'] for x in manifest['files']} == set(files)
    for item in manifest['files']:
        assert hashlib.sha256(files[item['path']]).hexdigest() == item['sha256']
    bootstrap = (ROOT/'bootstrap.py').read_text()
    for kind, table in assignments(bootstrap).items():
        for name, record in table.items():
            if kind == 'PATCHES':
                assert name not in files
            else:
                assert hashlib.sha256(files[name]).hexdigest() == record['before_sha256']
            data = zlib.decompress(base64.b64decode(record['data']))
            assert hashlib.sha256(data).hexdigest() == record['sha256']
            files[name] = data
    for name, data in files.items():
        assert data == (CANDIDATE/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == source[name]['sha256']
    old_functions = functions((PARENT/'bootstrap.py').read_text())
    new_functions = functions(bootstrap)
    assert old_functions.keys() == new_functions.keys()
    for name, node in new_functions.items():
        if name == 'main':
            continue  # Explicit metadata changes are checked below.
        if name == 'work':
            imports = [n for n in ast.walk(node) if isinstance(n, ast.ImportFrom)
                       and n.module == 'transvision.models.event_track_v2x.recovery_off_paper_runtime']
            assert len(imports) == 1
            imports[0].module = 'transvision.models.event_track_v2x.exclusive_paper_runtime'
        assert ast.dump(node) == ast.dump(old_functions[name])
    control = json.loads((ROOT/'source-control.json').read_bytes())
    # The bootstrap outside the embedded source tables must differ only by
    # the explicitly recorded runtime import and descriptive metadata.
    normalized = bootstrap
    for before, after in control['literal_bootstrap_edits']:
        assert normalized.count(after) == 1
        normalized = normalized.replace(after, before)
    a, b = ast.parse(normalized), ast.parse((PARENT/'bootstrap.py').read_text())
    for tree in (a, b):
        for node in tree.body:
            if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ('PATCHES', 'REPLACEMENTS')):
                node.value = ast.Constant(None)
    assert ast.dump(a) == ast.dump(b)
    real = json.loads(INPUT.read_bytes())
    excluded = {'configuration','bootstrap_sha256','dispatcher_sha256','exclusive_patches',
                'source_replacements','scope','recovery_off_source_freeze_sha256',
                'recovery_off_real_checkpoint_input_contract_sha256',
                'original_exclusive_acceptance_inherited','full_forest_independently_accepted',
                'recovery_off_independent_semantics_accepted','learned_Stage2_complete'}
    for item in prepared['seeds']:
        before = next(x['plan'] for x in parent_prepared['seeds'] if x['seed'] == item['seed'])
        after = item['plan']
        assert {k:v for k,v in before.items() if k not in excluded} == {k:v for k,v in after.items() if k not in excluded}
        assert after['configuration'] == real['configuration']
        assert after['bootstrap_sha256'] == sha(ROOT/'bootstrap.py')
        assert after['dispatcher_sha256'] == sha(ROOT/'dispatch.py')
    receipt = dict(kind='rbf_recovery_off_GPU_source_qualification_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        preparation_sha256=sha(ROOT/'preparation.json'), execution_sources=prepared['execution_sources'],
        evidence=evidence, deployed_source_files=len(files),
        actual_deployed_sources_match_qualified_candidate=True,
        producer_math_AST_unchanged_except_runtime_import=True,
        all_three_exact_input_and_configuration_bindings=True,
        original_experiment_acceptance_inherited=False, full_forest_independently_accepted=False,
        only_bound_allocation_supported=True, GPU_dispatched=False, paper_performance_complete=False)
    new(ROOT/'source-qualification.json', receipt)
    register(ROOT/'source-qualification.json', receipt['kind'])
    from recovery_off_dispatch_gate import require_qualified_source
    require_qualified_source(ROOT, prepared)
    print(json.dumps(dict(qualification=str(ROOT/'source-qualification.json'),
                         sha256=sha(ROOT/'source-qualification.json'), files=len(files))))


if __name__ == '__main__': main()
