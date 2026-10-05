"""Freeze a read-only memory-target observer and a separate user target record."""
import ast
import datetime
import importlib.util
import json
from pathlib import Path

from rbf_nested_seen_val_v2_common import R,new,register,sha


def main():
    directory = Path(__file__).resolve().parent
    path = directory/'observe_rbf_task_GPU_memory_target.py'
    spec = importlib.util.spec_from_file_location('memory_observer_software_control',path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    controls = []
    for value,expected in ((0,'below_75_percent'),(74.9,'below_75_percent'),
                           (75,'within_75_to_80_percent'),(80,'within_75_to_80_percent'),
                           (80.1,'above_80_percent')):
        assert module.summarize_percentage([(1000,value)],1001)['status'] == expected
        controls.append('boundary_'+str(value))
    assert module.summarize_percentage([],1001)['status'] == 'unknown'
    assert module.summarize_percentage([(1000,77)],1241)['status'] == 'unknown_stale'
    controls.extend(('missing_is_unknown','stale_is_unknown'))
    for name,call in (('out_of_range_percentage',lambda:module.summarize_percentage([(1000,101)],1001)),
        ('seconds_masquerading_as_milliseconds',lambda:module.series_samples(dict(x=[1790000000],y=[77]),0,2000000000)),
        ('duplicate_timestamps',lambda:module.series_samples(dict(x=[1790000000000]*2,y=[77,78]),0,2000000000))):
        try:
            call()
        except AssertionError:
            controls.append('refused_'+name)
        else:
            raise AssertionError('software refusal bypassed: '+name)
    monitor = Path(__import__('clearml.utilities.resource_monitor',fromlist=['x']).__file__)
    text = monitor.read_text()
    assert 'gpu_%d_mem_usage' in text and 'g["memory.used"]' in text and 'g["memory.total"]' in text
    assert 'gpu_mem[i] if gpu_mem and i in gpu_mem else g["memory.used"]' in text
    assert module.bound_logical_count([dict(id='test:gpu0,1,2,3')]) == 4
    controls.append('binding_cardinality_limits_logical_monitor_count')
    root = R/'source-freezes/rbf-allowlisted-Task-GPU-memory-target-observer-v2-binding-cardinality-20261004'
    assert not root.exists(), 'preserve source freezes'
    root.mkdir()
    sources = {}
    for name in ('observe_rbf_task_GPU_memory_target.py','prepare_rbf_task_GPU_memory_target.py','rbf_nested_seen_val_v2_common.py'):
        content = (directory/name).read_bytes();ast.parse(content);compile(content,str(root/name),'exec')
        with (root/name).open('xb') as stream:stream.write(content)
        sources[name] = dict(bytes=len(content),sha256=sha(root/name))
    ref = dict(path=str(monitor),sha256=sha(monitor))
    new(root/'source-freeze.json',dict(kind='rbf_allowlisted_Task_GPU_memory_target_observer_source_v2_binding_cardinality',
        sources=sources,references=[ref],monitor_reference=dict(**ref,SDK_version=__import__('clearml').__version__),
        software_boundary_and_refusal_controls=controls,actual_API_queries_in_qualification=0,
        mem_used_plus_free_ratio_forbidden_due_to_mixed_scope=True,actual_GPU_tasks_created=0))
    receipt = R/'receipts/rbf-user-GPU-memory-75-to-80-target-and-safe-observer-v2-readiness-20261004.json'
    new(receipt,dict(kind='rbf_user_GPU_memory_target_observer_readiness_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        user_instruction='显存利用率需要提升到75%-80%',GPU_memory_target_percent=[75,80],GPU_model_restriction=False,
        CPU_only_L40S_exception_preserved=True,source_freeze=str(root/'source-freeze.json'),source_freeze_sha256=sha(root/'source-freeze.json'),
        reference_metric='gpu_N_mem_usage: device memory percentage, not process mem_used_gb mixed with global mem_free_gb',
        software_controls=len(controls),real_batching_or_independent_sequences_for_future_distinct_jobs=True,
        no_memory_padding_or_unrelated_jobs_to_hit_target=True,
        unchanged_running_tasks_and_frozen_experiment_batch_size=True,
        scientific_protocol_or_paper_acceptance_criteria_not_changed_by_operational_memory_target=True,
        target_reached=False,actual_GPU_experiment_launched=False,full_paper_accepted=False))
    register(receipt,'rbf-user-GPU-memory-target-safe-observer-readiness')
    print(json.dumps(dict(receipt=str(receipt),source_freeze_sha256=sha(root/'source-freeze.json'),software_controls=len(controls))))


if __name__=='__main__':main()
