"""Full final-refit teacher admission after original native byte collection.

Reuses unchanged independent structural, causal, state, action and raw probe
oracles, with the separately frozen final-model cache203 binding. No tracker,
Torch, producer feature generator or target generator is imported.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

from rbf_nested_seen_val_v2_common import R,new,register,sha
from rbf_final_refit_teacher_binding import canonical,validate_registered
from collect_final_refit_capacity_teacher import BYTE_KIND,KIND,artifact_keys,contained,runtime_sources

OLD = R/'source-freezes/rbf-capacity-undecided-teacher-full-target-admission-v2-20261002'
OLD_FREEZE_SHA = '026a6693e032906fea9daf3cc6c384297d02ee2c685ad6eff0a7a920e7e540d2'
FINAL = R/'source-freezes/rbf-final-refit-full-forest-independent-CPU-v4-receipt-rows-20261004'
FINAL_FREEZE_SHA = 'f36f644f5f18fad3e5ab489edbc949e93bcab4efce22749dcb90211c8c8b27f7'
FINAL_CACHE_SHA = 'f1c2c1d5eb2b575b0029740deefb2afb37573da5302c9f1fe5ac67c792375801'


def modules():
    assert sha(OLD/'source-freeze.json') == OLD_FREEZE_SHA
    assert sha(FINAL/'source-freeze.json') == FINAL_FREEZE_SHA
    old = json.loads((OLD/'source-freeze.json').read_bytes())
    final = json.loads((FINAL/'source-freeze.json').read_bytes())
    for name,digest in old['sources'].items(): assert sha(OLD/name) == digest
    for name,digest in final['sources'].items(): assert sha(FINAL/name) == digest
    for dependency in ('teacher_structure','teacher_causal'):
        assert dependency not in sys.modules, 'ambient teacher oracle binding is forbidden'
    for dependency in ('rbf_final_refit_forest_binding',):
        if dependency in sys.modules:
            assert Path(sys.modules[dependency].__file__).resolve() == FINAL/(dependency+'.py'), 'ambient final-model gate'
    original_path = sys.path[:]
    try:
        sys.path.insert(0,str(OLD))
        spec=importlib.util.spec_from_file_location('final_teacher_original_independent_oracles',OLD/'accept.py')
        driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        references={name:driver.load_frozen('final_teacher_'+name,R/v['path'],v['sha256'])
            for name,v in old['independent_references'].items() if name!='cache203'}
        sys.path.insert(0,str(FINAL))
        references['cache203']=driver.load_frozen('final_teacher_final_cache203',FINAL/'final_cache203.py',FINAL_CACHE_SHA)
        assert references['cache203'].ATOL == references['cache203'].RTOL == 1e-8
        assert references['fresh'].ATOL == references['fresh'].RTOL == 1e-8
        return driver,references
    finally: sys.path[:]=original_path


def scope(byte, receipt, plan, binding, events, job, main_sha):
    assert byte['kind'] == BYTE_KIND and receipt['collection_kind'] == KIND
    assert byte['all_registered_bytes_verified'] is byte['all_7445_events_and_46_native_sequences_byte_bound'] is True
    assert byte['all_final_model_factor_rows_independently_bound'] is True
    assert byte['full_teacher_factor_state_action_target_admission'] is False
    assert byte['original_events_inferred_or_completed'] is False
    assert byte['world_size'] == job['plan']['world_size']
    assert set(byte['registered_artifacts']) == artifact_keys(byte['world_size'])
    assert receipt['fixture'] is plan['fixture'] is binding['fixture'] is False
    assert receipt['full_real_teacher_target_admission'] is False
    assert binding['GT_read'] is binding['test_read'] is False
    assert binding['original_plan'] == job['plan']
    assert binding['registered_artifacts'] == byte['registered_artifacts']
    assert binding['recipe_sha256'] == byte['recipe_sha256'] == job['recipe_sha256']
    assert binding['task_id'] == byte['task_id'] == job['task_id']
    assert plan['configuration'] == job['plan']['configuration']
    assert plan['source_sha256'] == binding['expected_runtime_sources']
    assert plan['cache_sha256'] == job['plan']['cache_manifest']['sha256']
    assert plan['configuration']['method'] == 'rbf' and plan['configuration']['allocation'] == 'teacher'
    assert plan['configuration']['backend'] == 'exclusive_root_partition_regions_v1'
    assert plan['configuration']['state']['candidate_protocol'] == 'rbf-all-class-top64-v1'
    assert plan['protocol']['dataset'] == plan['model_binding']['dataset'] == 'spd'
    assert plan['protocol']['split'] == plan['model_binding']['fit_split'] == 'train'
    assert plan['model_binding']['seed'] == binding['seed'] == byte['seed'] == job['seed']
    assert plan['model_binding']['model_sha256'] == binding['final_model_sha256'] == byte['final_model_sha256'] == job['plan']['final_refit_model_sha256']
    assert plan['model_binding']['model_sha256'] != job['plan']['original_nested_model_sha256']
    assert plan['model_binding']['checkpoint_sha256'] == byte['final_checkpoint_sha256'] == job['plan']['checkpoint']['sha256']
    assert binding['main_replay_admission_sha256'] == byte['main_replay_admission_sha256'] == job['plan']['final_main_acceptance_sha256'] == main_sha
    assert len(events) == plan['expected_events'] == receipt['completed_events'] == byte['completed_events'] == 7445
    sequences=sorted({e['sequence_id'] for e in events})
    assert len(sequences)==46 and sequences==plan['expected_sequences']==receipt['completed_sequences']==byte['completed_sequences']
    assert set(receipt['databases']) == set(sequences)
    assert len({(e['sequence_id'],e['event_id']) for e in events}) == 7445
    assert hashlib.sha256(canonical(events)).hexdigest() == plan['events_sha256']
    assert type(binding['labels']) is int and binding['labels'] == byte['labels'] > 0
    assert byte['NN_atol'] == byte['NN_rtol'] == 1e-4
    assert set(byte['final_model_factor_reports']) == set(sequences)
    for report in byte['final_model_factor_reports'].values():
        assert type(report['rows']) is int and report['rows'] >= 0
        assert report['atol'] == report['rtol'] == 1e-4
        assert math.isfinite(report['max_abs_error']) and report['max_abs_error'] >= 0
    return sequences


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','main-admission','main-byte-admission','published-prerequisite','teacher-journal','checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); own=Path(__file__).resolve().parent
    frozen=json.loads((own/'source-freeze.json').read_bytes())
    for name,spec in frozen['sources'].items(): assert sha(own/name)==spec['sha256']
    for spec in frozen['references']: assert sha(spec['path'])==spec['sha256']
    byte_path=contained(args.cohort,'independent-byte-cohort-collection.json')
    byte=json.loads(byte_path.read_bytes())
    assert byte['collector_sha256']==sha(own/'collect_final_refit_capacity_teacher.py')
    jobs=[j for j in json.loads(args.teacher_journal.read_bytes())['jobs'] if j['task_id']==byte['task_id']]
    assert len(jobs)==1;job=jobs[0]
    task,prerequisite=validate_registered(job,args.main_admission,args.main_byte_admission,args.published_prerequisite)
    assert set(task.artifacts)==set(byte['registered_artifacts'])
    for key,spec in byte['registered_artifacts'].items():
        assert task.artifacts[key].hash==spec['sha256'] and task.artifacts[key].size==spec['bytes']
    receipt_path=contained(args.cohort,'receipt.json')
    assert sha(receipt_path)==byte['collection_receipt_sha256']
    receipt=json.loads(receipt_path.read_bytes())
    assert {'plan.json','events.json','audit.jsonl','predictions.jsonl','timings.json',
        'resources.json','collection-binding.json'} <= set(receipt['files'])
    for name,checksum in receipt['files'].items(): assert sha(contained(args.cohort,name))==checksum
    plan=json.loads(contained(args.cohort,'plan.json').read_bytes())
    binding=json.loads(contained(args.cohort,'collection-binding.json').read_bytes())
    events=json.loads(contained(args.cohort,'events.json').read_bytes())
    sequences=scope(byte,receipt,plan,binding,events,job,sha(args.main_admission))
    assert plan['source_sha256']==runtime_sources(job['plan'])
    assert sha(args.checkpoint)==plan['model_binding']['checkpoint_sha256']
    assert binding['main_byte_admission_sha256']==sha(args.main_byte_admission)
    assert binding['published_prerequisite_sha256']==sha(args.published_prerequisite)
    event_path=R/f"artifacts/rbf-original-cache-CPU-metadata-export-v1-20261001/seed{byte['seed']}/events.json"
    assert sha(event_path)==binding['original_event_asset_sha256']==byte['original_event_asset_sha256']==job['plan']['events']['sha256']
    assert json.loads(event_path.read_bytes())['events']==events
    driver,refs=modules()
    admission=refs['cache203'].CacheAdmission(byte['seed'],args.checkpoint)
    assert admission.binding['final_model_sha256']==byte['final_model_sha256']
    output=args.output.resolve(); assert output.is_relative_to(R/'artifacts') and not args.output.is_symlink()
    output.mkdir(parents=True,exist_ok=False)
    base=dict(task_id=task.id,seed=byte['seed'],recipe_sha256=job['recipe_sha256'],configuration=plan['configuration'],
        checkpoint_sha256=sha(args.checkpoint),final_model_sha256=byte['final_model_sha256'],
        source_sha256=plan['source_sha256'],scorer_signature=plan['scorer_signature'],
        frozen_cache_identity=plan['model_binding']['frozen_cache_identity'],
        registered_artifacts=byte['registered_artifacts'],teacher_replay_receipt_sha256=sha(receipt_path),
        old_model_acceptance_inherited=False,main_prerequisite=prerequisite,
        byte_cohort_sha256=sha(byte_path),source_freeze_sha256=sha(own/'source-freeze.json'),
        main_replay_admission_sha256=sha(args.main_admission),main_byte_admission_sha256=sha(args.main_byte_admission),
        published_prerequisite_sha256=sha(args.published_prerequisite),strict_pipeline_isolated_selection=False,
        unchanged_target_driver_freeze_sha256=OLD_FREEZE_SHA,final_cache203_sha256=FINAL_CACHE_SHA,
        GT_read=False,test_read=False,total_teacher_resource_cost_admitted=False,
        same_resource_performance_accepted=False,learned_Stage2_complete=False,full_Stage2_complete=False,paper_performance_complete=False)
    new(output/'binding.json',base);results=[];started=time.monotonic()
    try:
        for index,sequence in enumerate(sequences):
            dbinfo=receipt['databases'][sequence];database=contained(args.cohort,dbinfo['path'])
            assert sha(database)==dbinfo['sha256']
            def progress(value):
                print(json.dumps(dict(value,sequence_id=sequence,completed_sequences=index,total_sequences=46,
                    ETA_seconds=None,ETA_reason='heterogeneous branch/probe work; whole cohort unknown')),flush=True)
            result=driver.verify_sequence(database,dbinfo['sha256'],refs,admission,progress)
            assert result['targets']['events']==sum(e['sequence_id']==sequence for e in events)
            assert result['runtime_raw_cache203_context']['final_model_sha256']==byte['final_model_sha256']
            assert result['runtime_raw_cache203_context']['rows']==byte['final_model_factor_reports'][sequence]['rows']
            assert result['action']['capacity_undecided_counted_as_optimal'] is False
            result['sequence_id']=sequence;results.append(result)
            new(output/f'sequence-{index:02d}.json',result)
        assert len(results)==46 and sum(r['targets']['events'] for r in results)==7445
        labels=sum(r['targets']['labels'] for r in results)
        assert labels==binding['labels']==byte['labels']
        assert sha(byte_path)==base['byte_cohort_sha256'] and sha(receipt_path)==base['teacher_replay_receipt_sha256']
        assert sha(args.checkpoint)==base['checkpoint_sha256']
        assert sha(args.main_admission)==base['main_replay_admission_sha256']
        assert sha(args.main_byte_admission)==base['main_byte_admission_sha256']
        assert sha(args.published_prerequisite)==base['published_prerequisite_sha256']
        for name,spec in frozen['sources'].items(): assert sha(own/name)==spec['sha256']
        for spec in frozen['references']: assert sha(spec['path'])==spec['sha256']
        acceptance=output/'acceptance.json'
        new(acceptance,dict(base,kind='rbf_final_refit_full_teacher_independent_model_bound_targets_v1',
            completed_events=7445,completed_sequences=sequences,labels=labels,
            full_real_teacher_target_admission=True,all_causal_features_and_counterfactual_targets_verified=True,
            all_18_features_and_signed_targets_independently_recomputed=True,
            all_probe_operations_and_charged_steps_reconstructed=True,
            all_203_feature_values_and_first_arrival_histories_verified=True,
            full_fresh_conditional_states_verified=True,capacity_undecided_counted_as_optimal=False,
            capacity_undecided_component_decisions=sum(r['action']['capacity_undecided_component_decisions'] for r in results),
            exact_identity_loss_or_posterior_optimum_target_claimed=False,
            max_target_error=max(r['targets']['max_abs_error'] for r in results),
            max_fresh_state_error=max(r['fresh_state']['max_abs_error'] for r in results),
            target_atol=refs['trajectory'].numeric.ATOL,target_rtol=refs['trajectory'].numeric.RTOL,
            fresh_atol=refs['fresh'].ATOL,fresh_rtol=refs['fresh'].RTOL,elapsed_seconds=time.monotonic()-started))
        register(acceptance,'rbf-final-refit-full-teacher-independent-targets')
        print(json.dumps(dict(receipt=str(acceptance))),flush=True)
    except BaseException as error:
        failure=output/'failure.json'
        new(failure,dict(base,exception_type=type(error).__name__,message=str(error),
            completed_sequences=len(results),accepted=False,automatic_producer_retry=False))
        register(failure,'rbf-final-refit-teacher-target-failure');raise


if __name__=='__main__':main()
