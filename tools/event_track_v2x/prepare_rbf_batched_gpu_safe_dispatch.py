"""Prepare collision-safe priority dispatch; no API or GPU execution."""
import ast
import datetime
import importlib.util
import json
from pathlib import Path
import tempfile

from rbf_nested_seen_val_v2_common import R,new,register,sha


def main():
    workspace=Path(__file__).resolve().parent
    collision=R/'source-freezes/rbf-matching-seen-val-joint-GPU-publish-dispatch-v3-live-reservations-20261004'
    root=R/'source-freezes/rbf-batched-complete-sequence-GPU-safe-dispatch-v1-20261004'
    assert not root.exists(), 'preserve frozen dispatchers'
    references={}

    def ref(path,expected=None):
        digest=sha(path);assert expected is None or digest==expected;references[str(path)]=digest

    collision_prep=json.loads((collision/'preparation.json').read_bytes())
    ref(collision/'preparation.json')
    for name,record in collision_prep['sources'].items():ref(collision/name,record['sha256'])
    executor=R/'source-freezes/rbf-batched-complete-sequence-GPU-measurement-v1-20261004'
    ref(executor/'preparation.json','566993724b7d3509905fbc390ceff74725b4ecdbe035183ad25f1bf7529586a0')
    contract=json.loads((executor/'preparation.json').read_bytes())
    for name,record in contract['sources'].items():ref(executor/name,record['sha256'])
    reader=R/'source-freezes/rbf-batched-complete-sequence-GPU-independent-reader-v1-20261004'
    ref(reader/'source-freeze.json','f096b08ea2dc95aeae62756c476b721353b2872a92bcc90352ba99a9971915c6')
    rf=json.loads((reader/'source-freeze.json').read_bytes())
    for name,record in rf['sources'].items():ref(reader/name,record['sha256'])
    for record in rf['references']:ref(Path(record['path']),record['sha256'])
    root.mkdir()
    sources={}
    for name,parent in (('submit_rbf_batched_sequence_gpu_measurement.py',workspace),
        ('prepare_rbf_batched_gpu_safe_dispatch.py',workspace),('rbf_nested_seen_val_v2_common.py',workspace),
        ('submit_rbf_final_identity.py',collision),('submit_rbf_seen_val_joint_identity.py',collision)):
        data=(parent/name).read_bytes();ast.parse(data);compile(data,str(root/name),'exec')
        with (root/name).open('xb') as stream:stream.write(data)
        sources[name]=dict(bytes=len(data),sha256=sha(root/name))
    path=root/'submit_rbf_batched_sequence_gpu_measurement.py'
    spec=importlib.util.spec_from_file_location('frozen_diagnostic_dispatch_priority_controls',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    controls=[]
    with tempfile.TemporaryDirectory(prefix='rbf-profile-priority-') as temporary:
        journals=[Path(temporary)/str(i) for i in range(3)]
        assert len(module.core_priority(journals)[1])==9;controls.append('empty_core_blocks_profile')
        cohorts=[[dict(seed=seed,task_id=f'{group}{seed}') for seed in (2027,1337,3407)] for group in range(3)]
        for path,cohort in zip(journals,cohorts):path.write_text(json.dumps(dict(jobs=cohort)))
        assert len(module.core_priority(journals)[0])==9 and not module.core_priority(journals)[1]
        controls.append('nine_distinct_core_scientific_identities_required')
        journals[1].write_text(json.dumps(dict(jobs=cohorts[1][:-1])))
        assert module.core_priority(journals)[1]==[dict(journal=str(journals[1]),seed=3407)]
        controls.append('missing_K4_seed3407_blocks_profile')
        journals[1].write_text(json.dumps(dict(jobs=cohorts[1])))
        journals[2].write_text(json.dumps(dict(jobs=[])))
        assert len(module.core_priority(journals)[1])==3;controls.append('three_Top1_candidates_take_priority')
        for case,bad in (('duplicate_seed',cohorts[0]+[cohorts[0][0]]),
                         ('unknown_seed',[dict(seed=999,task_id='unknown')])):
            journals[0].write_text(json.dumps(dict(jobs=bad)))
            try:module.core_priority(journals)
            except AssertionError:controls.append('refuse_'+case)
            else:raise AssertionError('priority refusal bypassed')
        journals[0].write_text(json.dumps(dict(jobs=cohorts[0])))
        journals[2].write_text(json.dumps(dict(jobs=[dict(seed=2027,task_id=cohorts[0][0]['task_id'])])))
        try:module.core_priority(journals)
        except AssertionError:controls.append('refuse_same_Task_in_two_methods')
        else:raise AssertionError('cross-method Task duplication allowed')
    new(root/'software-priority-qualification.json',dict(kind='rbf_profile_dispatch_software_priority_controls_v1',
        controls=controls,actual_API_queries=0,actual_Task_creations=0,actual_GPU_runs=0,
        physical_collision_helper_unchanged=True,actual_experiment_or_paper_accepted=False))
    sources['software-priority-qualification.json']=dict(bytes=(root/'software-priority-qualification.json').stat().st_size,
        sha256=sha(root/'software-priority-qualification.json'))
    new(root/'preparation.json',dict(kind='rbf_batched_complete_sequence_GPU_priority_safe_dispatch_source_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,
        references=[dict(path=p,sha256=h) for p,h in sorted(references.items())],
        executor_preparation=str(executor/'preparation.json'),executor_preparation_sha256=sha(executor/'preparation.json'),
        reader_path=str(reader/'read_rbf_batched_gpu_measurement.py'),reader_source_freeze_sha256=sha(reader/'source-freeze.json'),
        remaining_K4_and_Top1_have_priority=True,GPU_model_restriction=False,L40S_CPU_only=True,
        operational_GPU_memory_target_percent=[75,80],actual_GPU_memory_target_reached=False,
        commands_after_core_priority=[str(path),'--execute'],actual_upload_or_Task_execution_started=False,
        original_weights_configuration_caps_and_tolerances_unchanged=True,full_forest_or_paper_accepted=False))
    receipt=R/'receipts/rbf-batched-complete-sequence-GPU-safe-dispatch-readiness-20261004.json'
    new(receipt,dict(kind='rbf_batched_complete_sequence_GPU_safe_dispatch_readiness_v1',
        preparation=str(root/'preparation.json'),preparation_sha256=sha(root/'preparation.json'),
        software_priority_controls=len(controls),actual_API_queries=0,actual_Task_creations=0,actual_GPU_profile_runs=0,
        GPU_memory_target_percent=[75,80],target_reached=False,full_forest_or_paper_accepted=False))
    register(receipt,'rbf-batched-complete-sequence-GPU-safe-dispatch-readiness')
    print(json.dumps(dict(receipt=str(receipt),preparation_sha256=sha(root/'preparation.json'),
        software_priority_controls=len(controls),actual_GPU_task_created=False)))


if __name__=='__main__':main()
