"""Persist software-only checks; fixtures never confer GPU acceptance."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

from rbf_nested_seen_val_v2_common import R,new,register,sha


def main():
    root=R/'source-freezes/rbf-batched-complete-sequence-GPU-independent-reader-v1-20261004'
    freeze=json.loads((root/'source-freeze.json').read_bytes())
    assert sha(root/'source-freeze.json')=='f096b08ea2dc95aeae62756c476b721353b2872a92bcc90352ba99a9971915c6'
    reader=root/'read_rbf_batched_gpu_measurement.py';assert sha(reader)==freeze['sources'][reader.name]['sha256']
    spec=importlib.util.spec_from_file_location('frozen_batched_memory_gate',reader)
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    document=dict(kind='rbf_real_replay_per_device_memory_measurement_v1',GPU_uuid='software-fixture',device='cuda:0',
        sampler_finished=True,sampler_errors=[],replay_failure=None,completed_events=195,TF32_matmul=False,TF32_cudnn=False,
        target_independently_accepted=False,isolated_performance_or_paper_acceptance=False,target_percent=[75,80],
        device_usage_includes_all_processes=True,elapsed_seconds_including_observer=10.,events_per_second_with_observer=19.5,
        peak_process_tensor_allocated_bytes=50,peak_process_allocator_reserved_bytes=70,
        samples=[dict(elapsed_seconds=t,device_free_bytes=25,device_total_bytes=100,device_used_percent=75.,
            process_tensor_allocated_bytes=50,process_allocator_reserved_bytes=70) for t in (0.,10.)])
    rank=dict(rank=0,gpu_uuid='software-fixture')
    positive=gate.measurement_gate(document,rank,195)
    assert positive['all_observed_samples_in_target'] and not positive['continuous_occupancy_between_samples_proven']
    assert positive['maximum_observation_gap_seconds']==10.
    rejected=[]
    changes=[('GPU_uuid','wrong'),('completed_events',194),('sampler_finished',False),('sampler_errors',[dict(type='test')]),
        ('TF32_matmul',True),('TF32_cudnn',True),('target_independently_accepted',True),('events_per_second_with_observer',20.),
        ('elapsed_seconds_including_observer',float('nan')),('peak_process_tensor_allocated_bytes',1),('replay_failure',dict(type='test'))]
    cases=[]
    for key,value in changes:
        candidate=copy.deepcopy(document);candidate[key]=value;cases.append((key,candidate))
    for key,value in [('device_used_percent',80.),('device_free_bytes',101),('process_tensor_allocated_bytes',71),('elapsed_seconds',11.)]:
        candidate=copy.deepcopy(document);candidate['samples'][0][key]=value;cases.append(('sample_'+key,candidate))
    for name,candidate in cases:
        try:gate.measurement_gate(candidate,rank,195)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('corrupted evidence accepted: '+name)
    low=copy.deepcopy(document)
    for sample in low['samples']:sample.update(device_free_bytes=98,device_used_percent=2.)
    result=gate.measurement_gate(low,rank,195)
    assert not result['all_observed_samples_in_target']
    output=R/'receipts/rbf-batched-GPU-memory-reader-software-controls-20261004.json'
    value=dict(kind='rbf_batched_GPU_memory_reader_software_controls_v1',reader=str(reader),reader_sha256=sha(reader),
        reader_source_freeze_sha256=sha(root/'source-freeze.json'),qualifier_sha256=sha(__file__),
        command=[sys.executable,str(Path(__file__).resolve())],positive_fixture_count=2,
        rejected_corruptions=rejected,software_refusal_count=len(rejected),
        sparse_samples_do_not_prove_continuous_occupancy=True,low_usage_reports_target_unmet=True,
        actual_GPU_execution_count=0,actual_ClearML_queries=0,GPU_memory_target_admitted=False)
    new(output,value);register(output,'rbf-batched-GPU-memory-reader-software-controls')
    print(json.dumps(dict(receipt=str(output),sha256=sha(output),software_refusals=len(rejected),actual_GPU_execution_count=0)))


if __name__=='__main__':main()
