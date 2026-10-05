#!/usr/bin/env python3
"""Qualify and freeze the bound recovery-off independent CPU acceptance entry.

Tests use a pinned producer plus independently reconstructed tiny raw caches.
This does not run a real cohort, publish anything or inherit main acceptance.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
TOOLS=ROOT/'tools/event_track_v2x'
SOURCES=('recovery_off_final_binding.py','accept_recovery_off_final_cohort.py',
         'recovery_off_structure_oracle.py','recovery_off_action_oracle.py','rbf_nested_seen_val_v2_common.py')
TESTS=('test_recovery_off_final_binding.py','test_recovery_off_structure_oracle.py',
       'test_recovery_off_action_oracle.py','test_recovery_off_full_oracle_integration.py')
RUNTIME=Path('/Volumes/Data/test/recover-before-fuse/runtimes/batched-row-CPU-parity-torch260-v1/bin/python')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new(path,value):
    with Path(path).open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); out=args.output.absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('fresh source-freeze directory required')
    out.mkdir(parents=True)
    test_paths=[ROOT/'tests/event_track_v2x'/name for name in TESTS]
    originals=[*(TOOLS/name for name in SOURCES),*test_paths,Path(__file__).resolve()]
    pins={str(p):sha(p) for p in originals}
    for name in SOURCES:
        raw=(TOOLS/name).read_bytes();ast.parse(raw)
        (out/name).write_bytes(raw)
    (out/'software-tests').mkdir()
    for p in test_paths: (out/'software-tests'/p.name).write_bytes(p.read_bytes())
    (out/'prepare-source.py').write_bytes(Path(__file__).read_bytes())
    spec=importlib.util.spec_from_file_location('recovery_qualification_binding',out/'recovery_off_final_binding.py')
    module=importlib.util.module_from_spec(spec);sys.path.insert(0,str(out));spec.loader.exec_module(module)
    for path,digest in module.UNCHANGED.items():
        if sha(path)!=digest: raise ValueError('unchanged oracle differs: '+str(path))
    command=[str(RUNTIME),'-m','pytest','-q','-p','no:cacheprovider','--junitxml='+str(out/'checks.xml'),*(str(p) for p in test_paths)]
    env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/private/tmp/rbf-code-preparation-test-deps')
    new(out/'command.json',dict(argv=command,cwd=str(ROOT),environment_overrides={k:env[k] for k in ('OMP_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTHONPATH')}))
    with (out/'checks.log').open('x') as log:
        code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    if code:
        new(out/'qualification-failed.json',dict(returncode=code,real_cohort_accepted=False));raise SystemExit(code)
    cases=list(ET.parse(out/'checks.xml').iter('testcase'))
    if not cases or any(list(c) for c in cases): raise ValueError('test failure, skip or unexpected test result')
    required=('test_original_raw_cache_causal_fresh_and_restricted_oracles_integrate',
              'test_rehashed_numeric_mutation_rejected_by_original_cache_checker',
              'test_frozen_database_action_scope','test_rehashed_semantic_mutations_rejected',
              'test_unrestricted_oracle_would_admit_a_better_but_discarded_history')
    if not all(any(c.attrib['name'].split('[')[0]==name for c in cases) for name in required):
        raise ValueError('required real-schema or negative-control test missing')
    if any(sha(p)!=h for p,h in pins.items()): raise ValueError('source changed during software qualification')
    sources={name:sha(out/name) for name in SOURCES}
    if any(sha(TOOLS/name)!=h for name,h in sources.items()): raise ValueError('frozen source differs from tested source')
    new(out/'source-freeze.json',dict(kind='rbf_recovery_off_full_independent_CPU_source_v1',sources=sources,
        unchanged_independent_sources={str(p):h for p,h in module.UNCHANGED.items()},tested_workspace_sources=pins,
        scope='frozen bound-allocation event-boundary recovery-off only; no learned cohort inheritance'))
    evidence=[dict(path=str(out/name),sha256=sha(out/name)) for name in ('checks.xml','checks.log','command.json')]
    new(out/'qualification.json',dict(kind='rbf_recovery_off_full_independent_CPU_software_gate_v1',
        source_freeze_sha256=sha(out/'source-freeze.json'),sources=sources,evidence=evidence,
        software_tests_passed=len(cases),integrated_pass=True,binding_negative_controls_passed=True,
        restricted_structure_negative_controls_passed=True,restricted_action_negative_controls_passed=True,
        unchanged_cache_causal_fresh_schema_integration_passed=True,original_experiment_acceptance_inherited=False,
        real_cohort_accepted=False,paper_performance_complete=False))
    proof=module.source_gate(out)
    common=__import__('rbf_nested_seen_val_v2_common')
    common.register(out/'qualification.json','rbf_recovery_off_full_independent_CPU_software_gate_v1')
    print(json.dumps(dict(output=str(out),tests=len(cases),**proof,real_cohort_accepted=False)))


if __name__=='__main__': main()
