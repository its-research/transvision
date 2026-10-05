"""Finite no-data probe construction and readback controls, no Linux admission."""
import ast
import copy
import datetime
import importlib
import io
import json
from pathlib import Path
import tarfile

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
    return importlib.import_module('prepare_final_priority_runtime_probe'),importlib.import_module('control_final_priority_runtime_probe')


def test_archive_has_only_exact_declared_regular_source_members(modules):
    build,_=modules;files={'candidate/example.py':b'x=1\n','preparation.json':b'{}'}
    a=build.archive_bytes(files);assert a==build.archive_bytes(files)
    with tarfile.open(fileobj=io.BytesIO(a)) as t:
        assert {m.name for m in t.getmembers()}==set(files)
        for m in t.getmembers():assert m.isfile() and t.extractfile(m).read()==files[m.name]
    with pytest.raises(AssertionError):build.archive_bytes({'../outside':b'x'})


def test_generated_probe_uses_new_complete_consumer_and_keeps_offline_install(modules):
    build,_=modules
    if not build.OLD.exists():pytest.skip('qualified original source unavailable on this host')
    original=(build.OLD/'bootstrap.py').read_text()
    candidate=build.render(original,b'synthetic archive',b'{}');tree=ast.parse(candidate)
    assert "bridge.source_gate(consumer)" in candidate
    assert "gate.verify_teacher_sources" in candidate
    assert "gate.ACTUAL_CPU_WITNESS" not in candidate and "bridge.apply_source_overlay" not in candidate
    assert "np.__version__ == '1.26.4'" in candidate and "sys.dont_write_bytecode = True" in candidate
    def functions(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    before,after=functions(original),functions(candidate)
    for name in ('sha','extract','load'):assert before[name]==after[name]
    # The offline wheel download/byte verification/installation segment is intact.
    start="        wheels_archive = read(plan['runtime_wheels'], 'runtime-wheels.tar.gz')"
    end="        import torch"
    expected=original[original.index(start):original.index(end)]
    actual=candidate[candidate.index(start):candidate.index('        import numpy as np')]
    assert actual==expected


def report_example(control):
    plan={'consumer_member_count':17,'full_source_member_count':135,
        'offline_runtime_versions':{'cryptography':'46.0.3','cffi':'2.0.0','pycparser':'2.23','pip':'24.3.1'}}
    report=dict(kind='rbf_final_refit_priority_Linux_CPU_import_CLI_probe_v1',task_id='software-control',plan=plan,
        platform='Linux',python_version='3.12.9',Torch_version='2.6.0+cu124',numpy_version='1.26.4',
        CPU_threads=1,CPU_interop_threads=1,CUDA_visible_devices='',CUDA_initialized=False,cryptography_version='46.0.3',
        offline_runtime_versions={k:v for k,v in plan['offline_runtime_versions'].items() if k!='pip'},
        exact_consumer_source_members_verified=17,full_source_member_count_verified=135)
    # Required flags are explicit independent observations, not default success.
    true=('training_only_source_roles_verified all_registered_runtime_wheels_and_members_verified isolated_offline_runtime_install_succeeded '
          'original_Torch_and_fit_model_bytes_unchanged capacity_core_exactly_replaced capacity_target_gate_actual_imported '
          'exact_original_core_overlay_actually_verified original_init_and_source_namespace_used_without_model_mock '
          'all_registered_original_and_new_consumer_source_bytes_verified actual_new_exporter_full_dependency_import_succeeded '
          'original_event_envelope_CLI_present mutually_exclusive_arrival_modes_rejected absent_local_teacher_inputs_rejected_before_export_output '
          'absent_training_inputs_and_invalid_seed_rejected_before_fit_output portable_bridge_actual_Linux_import_and_help_passed')
    false=('actual_Linux_full_export_or_fit_executed dataset_read checkpoint_or_weights_read optimizer_created neural_forward_executed '
           'teacher_replay_executed full_teacher_target_admission priority_training_executed full_Stage2_complete paper_performance_complete')
    report.update({k:True for k in true.split()});report.update({k:False for k in false.split()})
    return report,plan


def test_complete_observations_admit_only_runtime(modules):
    _,control=modules;r,p=report_example(control)
    control.validate_report(r,p,'software-control','L40S:cpu')


@pytest.mark.parametrize('field,value',[
    ('platform','Darwin'),('Torch_version','2.5.0'),('numpy_version','2.5.3'),('CUDA_initialized',True),
    ('full_source_member_count_verified',134),('training_only_source_roles_verified',False),
    ('dataset_read',True),('priority_training_executed',True),('full_Stage2_complete',True),
    ('all_registered_runtime_wheels_and_members_verified',False),('actual_new_exporter_full_dependency_import_succeeded',False),
    ('kind','old-v7'),('CPU_threads',8)])
def test_incomplete_or_wrong_runtime_not_accepted(modules,field,value):
    _,control=modules;r,p=report_example(control);r[field]=value
    with pytest.raises(AssertionError):control.validate_report(r,p,'software-control','L40S:cpu')


def test_stale_busy_wrong_queue_or_GPU_worker_not_used(modules):
    _,control=modules;now=datetime.datetime.now(datetime.timezone.utc)
    q={'name':'GPU3-L40S','entries':[]}
    w={'id':'host-L40S','last_activity_time':now.isoformat(),'task':None,'queues':[{'id':control.Q}]}
    assert control.idle_workers(q,[w],now)==[w]
    for change in ({'task':{'id':'running'}},{'id':'host-A100'},
                   {'last_activity_time':(now-datetime.timedelta(seconds=95)).isoformat()},{'queues':[]}):
        assert control.idle_workers(q,[dict(w,**change)],now)==[]
    assert control.idle_workers(dict(q,entries=[{'task':'pending'}]),[w],now)==[]
