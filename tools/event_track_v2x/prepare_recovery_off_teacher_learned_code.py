#!/usr/bin/env python3
"""Freeze tested recovery-off teacher/learned software; never dispatch experiments.

The snapshot preserves existing qualification evidence and verifies immutable
copies, independent references and the exact checker CLI source gate offline.
It is not a deployment/runtime or real-cohort acceptance.
"""
import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
TOOLS=ROOT/'tools/event_track_v2x'
ARTIFACT_ROOT=Path('/Volumes/Data/test/recover-before-fuse')
RUNTIME=ARTIFACT_ROOT/'runtimes/batched-row-CPU-parity-torch260-v1/bin/python'
MODEL_DIR='transvision/models/event_track_v2x/'
TEST_DIR='tests/event_track_v2x/'
MODELS=('recovery_off_allocation.py','recovery_off_tracking.py','recovery_off_paper_runtime.py',
        'paper_runtime_selection.py','allocation_training.py')
TESTS=('test_recovery_off_learned_trajectory.py','test_recovery_off_allocation.py',
       'test_recovery_off_tracking.py','test_recovery_off_paper_runtime.py','test_allocation_training.py')
CHECKERS=('check_recovery_off_learned_search_trajectory.py','rbf_independent_recovery_off_learned_trajectory.py')
EVIDENCE={
 'checks.log':('/private/tmp/rbf-recovery-off-learned-oracle-tests-v4-20261005.log','52624bf22d9607c6a976142166cdd9da8b3ba2a8e0546c500b0e65a2292e8b8c'),
 'original-source-hashes.sha256':('/private/tmp/rbf-recovery-off-priority-original-sources-20261005.sha256',None),
 'bound-byte-parity.json':('/private/tmp/rbf-recovery-off-bound-original-byte-parity-v2-20261005.json','8b17f167e08d183605af796507a4aedc53ec163b6a258637313a2085a364cdd1'),
 'bound-byte-parity-script.py':('/private/tmp/rbf_check_recovery_off_bound_parity_20261005.py',None),
 'runtime-tests-41-passed.log':('/private/tmp/rbf-recovery-off-priority-tests-v4-20261005.log','906544652987b429198fb909988c5b6387d34d45ec8125ffea308835e4d98c3c'),
}
TESTED={
 MODEL_DIR+'recovery_off_allocation.py':'bde183fd70594b276488367526d2bb3bbb3e1dab927f25776dbc1c8b9eaf8f4f',
 MODEL_DIR+'recovery_off_tracking.py':'064c06d70974b53ab72e62ac0b5426858b97618ce9d113e7755d1905f70b6279',
 MODEL_DIR+'recovery_off_paper_runtime.py':'6a9832f5328e74ff72048b6d96c05a67f9c590ff3d4180f3bd7c32ede033294e',
 MODEL_DIR+'paper_runtime_selection.py':'26293085243772db1a88304a407864a1ebbbb7a145f6259baab350f6745f172f',
 MODEL_DIR+'allocation_training.py':'e5d05460cf4a2998dd22d043667dd9fa500c177e3db2390c6ad1e90325f81a97',
 TEST_DIR+'test_recovery_off_allocation.py':'dbf8e219f0bb5b19658003abe8bca7964ff94e38a2d4e8f79d82912c64beb727',
 TEST_DIR+'test_recovery_off_paper_runtime.py':'bb4fc3ec0fbbbb89487194f5bc8c2c20b9250ca18bfc4139af90cddacd22236b',
 TEST_DIR+'test_recovery_off_learned_trajectory.py':'98cb12e28a48faf409c72b06956ad5ee9bba9a8d6a9ddc424eff08c6ff3d1968',
 'tools/event_track_v2x/'+CHECKERS[0]:'63438fe8edd503221e61f166eae0cdaed77703b6302e83ba8db79b74013d219c',
 'tools/event_track_v2x/'+CHECKERS[1]:'4825c43116af8e4f4e5dbcfddb5814aa8e1d799a0f9fb9eb2a501d10730e9a14',
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def local_dependencies(paths):
    """Record local Python source dependencies without executing model imports."""
    pending=list(paths); seen=set()
    while pending:
        path=Path(pending.pop()).resolve()
        if path in seen or not path.is_file() or not path.is_relative_to(ROOT):continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_bytes(),filename=str(path))):
            candidates=[]
            if isinstance(node,ast.ImportFrom):
                if node.level:
                    directory=path.parent
                    for _ in range(node.level-1):directory=directory.parent
                    base=directory.joinpath(*(node.module or '').split('.'))
                    candidates=[base.with_suffix('.py'),*(base/a.name for a in node.names)]
                    candidates += [p.with_suffix('.py') for p in candidates if p.suffix!='.py']
                elif node.module:
                    base=ROOT.joinpath(*node.module.split('.'))
                    candidates=[base.with_suffix('.py'),base/'__init__.py']
                    candidates += [(ROOT/TEST_DIR/(node.module+'.py'))]
            elif isinstance(node,ast.Import):
                candidates=[ROOT.joinpath(*a.name.split('.')).with_suffix('.py') for a in node.names]
            pending.extend(p for p in candidates if p.is_file())
    return sorted(seen)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.absolute()
    assert out.is_relative_to(ARTIFACT_ROOT/'source-freezes')
    assert not out.exists() and not any(p.is_symlink() for p in (out,*out.parents))
    for name,digest in TESTED.items():assert sha(ROOT/name)==digest,'tested source changed: '+name
    for path,digest in EVIDENCE.values():
        assert Path(path).is_file()
        if digest:assert sha(path)==digest,'qualification evidence changed: '+path
    assert re.search(r'\b69 passed\b',Path(EVIDENCE['checks.log'][0]).read_text())
    parity=json.loads(Path(EVIDENCE['bound-byte-parity.json'][0]).read_bytes())
    assert sha(parity['original_path'])==parity['original_sha256']
    assert parity['current_sha256']==TESTED[MODEL_DIR+'recovery_off_tracking.py']
    assert len(parity['checks'])==8 and all(c['events']==3 and c['prediction_byte_parity'] and c['audit_byte_parity'] for c in parity['checks'])
    primary=[*(ROOT/MODEL_DIR/n for n in MODELS),*(ROOT/TEST_DIR/n for n in TESTS),
             *(TOOLS/n for n in CHECKERS),TOOLS/'rbf_nested_seen_val_v2_common.py',Path(__file__).resolve()]
    sources=local_dependencies(primary)
    pins={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sources}
    out.mkdir(parents=True);(out/'evidence').mkdir();(out/'checker').mkdir()
    for p in sources:
        target=out/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    for name,(path,_) in EVIDENCE.items():(out/'evidence'/name).write_bytes(Path(path).read_bytes())
    # Preserve earlier failed software checks as failures, never erase/relabel.
    for path in sorted(Path('/private/tmp').glob('rbf-recovery-off-learned-oracle-tests-v[123]-20261005.log')):
        (out/'evidence'/path.name).write_bytes(path.read_bytes())
    command=[str(RUNTIME),'-m','pytest','-q','-p','no:cacheprovider',*(str(ROOT/TEST_DIR/n) for n in TESTS)]
    new(out/'evidence/command.json',dict(argv=command,cwd=str(ROOT),
        environment_overrides=dict(PYTHONPATH='/private/tmp/rbf-code-preparation-test-deps',
            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'),
        invocation='preserved completed 69-test qualification, not rerun during source preparation',
        stdout_and_stderr=EVIDENCE['checks.log'][0]))
    for name in CHECKERS:(out/'checker'/name).write_bytes((TOOLS/name).read_bytes())
    sys.path.insert(0,str(out/'checker'))
    spec=importlib.util.spec_from_file_location('frozen_recovery_off_checker_cli',out/'checker'/CHECKERS[0])
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    reference=dict(path=str(cli.oracle.REFERENCE),sha256=cli.oracle.REFERENCE_SHA)
    new(out/'checker/source-freeze.json',dict(kind='rbf_independent_recovery_off_learned_search_trajectory_source_v1',
        sources={n:dict(sha256=sha(out/'checker'/n),bytes=(out/'checker'/n).stat().st_size) for n in CHECKERS},references=[reference]))
    proof=cli.source_gate(out/'checker')
    for name,record in pins.items():
        assert sha(ROOT/name)==record['sha256'],'workspace changed while snapshotting: '+name
        assert sha(out/'source'/name)==record['sha256']
    new(out/'source-freeze.json',dict(kind='rbf_recovery_off_teacher_learned_software_source_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=pins,
        tested_primary_sources=TESTED,checker_source_freeze_sha256=proof,
        independent_reference=reference,original_bound_reference=dict(path=parity['original_path'],sha256=parity['original_sha256']),
        scope='software preparation; restricted support and separately bound priority training, no unrestricted weight transfer',
        same_priority_weights_one_variable_ablation_established=False,runtime_dependencies_frozen=False))
    evidence=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((out/'evidence').iterdir())]
    receipt=out/'software-preparation.json'
    new(receipt,dict(kind='rbf_recovery_off_teacher_learned_software_preparation_v1',
        source_freeze_sha256=sha(out/'source-freeze.json'),checker_source_freeze_sha256=proof,
        software_tests_passed=69,evidence=evidence,offline_checker_source_gate_passed=True,
        bound_prediction_audit_byte_parity_events=24,
        checkpoint_requires_recovery_off_source_config_and_support_binding=True,
        learned_order_reconstructs_correlated_historical_support=True,
        independent_trajectory_atol=1e-8,independent_trajectory_rtol=1e-8,
        experiment_complete=False,real_teacher_complete=False,real_priority_fit_complete=False,
        real_learned_replay_complete=False,real_cohort_accepted=False,
        source_uploaded=False,task_created=False,paper_performance_complete=False,
        qualification='finite software fixtures only; final forest/action/state and measured cost remain separate'))
    # Freeze the new package only. No old source-freeze or experiment is touched.
    for p in out.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted((p for p in out.rglob('*') if p.is_dir()),reverse=True):p.chmod(0o555)
    out.chmod(0o555)
    sys.path.insert(0,str(TOOLS));import rbf_nested_seen_val_v2_common as common
    common.register(receipt,'rbf_recovery_off_teacher_learned_software_preparation_v1')
    assert cli.source_gate(out/'checker')==proof
    print(json.dumps(dict(output=str(out),source_files=len(sources),software_tests_passed=69,
        receipt=str(receipt),receipt_sha256=sha(receipt),source_freeze_sha256=sha(out/'source-freeze.json'),
        checker_source_freeze_sha256=proof,experiment_complete=False,real_cohort_accepted=False)))


if __name__=='__main__':main()
