"""Rebase the bound recovery ablation onto the exact final-forest source.

This creates a local, create-once source candidate. It neither uploads nor
inherits acceptance of a complete forest, a learned policy, or paper metrics.
"""
import ast
import base64
import copy
import datetime
import hashlib
import json
from pathlib import Path
import tarfile
import zlib

from rbf_nested_seen_val_v2_common import R, new, register, sha

NAME = 'rbf-recovery-off-original-source-bound-candidate-v3-20261005'
PREFIX = 'transvision/models/event_track_v2x/'
CANDIDATE = R/'source-freezes/rbf-recovery-off-bound-candidate-v1-20261005'
DISPATCH = R/'receipts/rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json'
ORIGIN = R/'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback'
BOOTSTRAP = R/'source-freezes/rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004/bootstrap.py'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def function(data, name):
    return next(n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef) and n.name == name)


def remove_progress_metadata(node):
    """Only normalize the one progress-constructor metadata extension."""
    node = copy.deepcopy(node)
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'ExperimentProgress']
    assert len(calls) == 1
    call = calls[0]
    assert {k.arg for k in call.keywords} <= {'context', 'eta_scope'}
    call.keywords = []
    return ast.dump(node)


def main():
    assert sha(DISPATCH) == '82aaebf996a6d9809158a6cd4257cabadf904e255ace5a5c6e43158e32383efa'
    assert sha(CANDIDATE/'source-freeze.json') == '4cf820e62d372604be1feb01521f6b64ecb0ab6f806298cce12141a5a2f5f218'
    frozen = json.loads((CANDIDATE/'source-freeze.json').read_bytes())['sources']
    jobs = json.loads(DISPATCH.read_bytes())['jobs']
    assert {j['seed'] for j in jobs} == {1337, 2027, 3407}
    plans = [j['plan'] for j in jobs]
    for key in ('source', 'exclusive_patches', 'source_replacements', 'configuration', 'bootstrap_sha256'):
        assert all(p[key] == plans[0][key] for p in plans)
    plan = plans[0]
    assert sha(BOOTSTRAP) == plan['bootstrap_sha256']
    proof = json.loads((ORIGIN/'acceptance.json').read_bytes())
    assert proof['task_id'] == plan['source']['task']
    assert proof['all_registered_bytes_independently_verified'] is True
    assert sha(ORIGIN/'acceptance.json') == plan['source_and_events_cloud_admission_sha256']
    archive = ORIGIN/'source.bytes'
    assert sha(archive) == plan['source']['sha256'] == proof['source_sha256']
    assert archive.stat().st_size == plan['source']['bytes']
    original = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            assert member.name not in original
            original[member.name] = tar.extractfile(member).read()
    manifest = json.loads(original.pop('source-manifest.json'))
    assert manifest['bootstrap_sha256'] == plan['parent_bootstrap_sha256']
    assert {r['path'] for r in manifest['files']} == set(original)
    for row in manifest['files']:
        assert digest(original[row['path']]) == row['sha256']
        assert len(original[row['path']]) == row['bytes']
    embedded = {}
    for node in ast.parse(BOOTSTRAP.read_bytes()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('PATCHES', 'REPLACEMENTS'):
                embedded[node.targets[0].id] = ast.literal_eval(node.value)
    assert {k: v['sha256'] for k, v in embedded['PATCHES'].items()} == plan['exclusive_patches']
    assert {k: {f: v[f] for f in ('before_sha256', 'sha256')}
            for k, v in embedded['REPLACEMENTS'].items()} == plan['source_replacements']
    for kind, files in embedded.items():
        for name, row in files.items():
            if kind == 'PATCHES':
                assert name not in original
            else:
                assert digest(original[name]) == row['before_sha256']
            data = zlib.decompress(base64.b64decode(row['data']))
            assert digest(data) == row['sha256']
            original[name] = data

    def candidate(name):
        data = (CANDIDATE/name).read_bytes()
        assert not (CANDIDATE/name).is_symlink()
        assert digest(data) == frozen[name]['sha256'] and len(data) == frozen[name]['bytes']
        return data

    differences = [name for name, data in original.items() if candidate(name) != data]
    assert set(differences) == {PREFIX+n for n in (
        'allocation_training.py', 'experiment_progress.py', 'paper_runtime.py', 'exclusive_paper_runtime.py')}
    original_replay = function(original[PREFIX+'exclusive_paper_runtime.py'], 'replay')
    candidate_replay = function(candidate(PREFIX+'exclusive_paper_runtime.py'), 'replay')
    off_replay = function(candidate(PREFIX+'recovery_off_paper_runtime.py'), 'replay')
    assert ast.dump(candidate_replay) == ast.dump(off_replay)
    assert remove_progress_metadata(original_replay) == remove_progress_metadata(off_replay)

    result = dict(original)
    provenance = {n: 'exact_final_forest_source' for n in original}
    # Only progress instrumentation is inherited from the newer workspace;
    # priority training and legacy replay retain the remote original bytes.
    for name in ('exclusive_paper_runtime.py', 'experiment_progress.py'):
        result[PREFIX+name] = candidate(PREFIX+name)
        provenance[PREFIX+name] = 'explicit_progress_metadata_extension'
    for name in ('recovery_off_tracking.py', 'recovery_off_paper_runtime.py', 'paper_runtime_selection.py'):
        assert PREFIX+name not in result
        result[PREFIX+name] = candidate(PREFIX+name)
        provenance[PREFIX+name] = 'explicit_bound_recovery_ablation_addition'
    for name in ('transvision/__init__.py', 'transvision/register.py', 'transvision/version.py',
                 'transvision/models/__init__.py', 'tools/__init__.py'):
        if name in frozen:
            assert name not in result
            result[name] = candidate(name); provenance[name] = 'package_import_support'
    # Test dependencies are copied from the already frozen candidate, not from
    # the mutable checkout. They are qualification inputs, never model code.
    for name in frozen:
        if name.startswith('tests/event_track_v2x/') or name in (
                'tools/event_track_v2x/build_detection_cache_v2.py',
                'tools/event_track_v2x/probe_frontier_completion.py'):
            assert name not in result
            result[name] = candidate(name); provenance[name] = 'frozen_software_qualification_support'
    config = json.loads((CANDIDATE/'bound-configuration.json').read_bytes())
    assert digest((CANDIDATE/'bound-configuration.json').read_bytes()) == frozen['bound-configuration.json']['sha256']
    normalized = copy.deepcopy(config)
    assert normalized.pop('backend') == 'exclusive_event_boundary_recovery_off_v1'
    assert normalized['limits'].pop('recovery_off_version') == 1
    control = copy.deepcopy(plan['configuration']); control.pop('backend')
    assert normalized == control, 'only declared recovery switch may change the final configuration'
    out = R/'source-freezes'/NAME
    assert not out.exists()
    out.mkdir()
    for name, data in result.items():
        p = out/name; p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as stream: stream.write(data)
        assert sha(p) == digest(data)
    new(out/'bound-configuration.json', config)
    with (out/'prepare_recovery_off_original_source_candidate.py').open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    sources = {n: dict(bytes=len(b), sha256=digest(b), provenance=provenance[n]) for n, b in result.items()}
    for name in ('bound-configuration.json', 'prepare_recovery_off_original_source_candidate.py'):
        sources[name] = dict(bytes=(out/name).stat().st_size, sha256=sha(out/name), provenance='preparation')
    record = dict(kind='rbf_recovery_off_original_final_source_bound_candidate_v3',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), sources=sources,
        final_dispatch=dict(path=str(DISPATCH), sha256=sha(DISPATCH)),
        base_source=dict(task_id=proof['task_id'], archive_sha256=sha(archive), acceptance_sha256=sha(ORIGIN/'acceptance.json')),
        final_bootstrap_sha256=sha(BOOTSTRAP), final_source_files=len(original),
        retained_exact_final_source_files=sum(v=='exact_final_forest_source' for v in provenance.values()),
        excluded_workspace_changes=[PREFIX+'allocation_training.py', PREFIX+'paper_runtime.py'],
        newer_progress_instrumentation=[PREFIX+'exclusive_paper_runtime.py', PREFIX+'experiment_progress.py'],
        replay_AST_equal_except_progress_metadata=True, final_configuration_equal_except_recovery_switch=True,
        candidate_parent_sha256=sha(CANDIDATE/'source-freeze.json'),
        allocation='bound', learned_priority_binding_implemented=False,
        qualification_pending=True, uploaded=False, GPU_experiment_dispatched=False,
        real_data_forest_accepted=False, paper_performance_complete=False)
    new(out/'source-freeze.json', record); register(out/'source-freeze.json', record['kind'])
    print(json.dumps(dict(source_freeze=str(out/'source-freeze.json'), sha256=sha(out/'source-freeze.json'),
                          sources=len(sources), retained_exact_final_source_files=record['retained_exact_final_source_files'])))


if __name__ == '__main__': main()
