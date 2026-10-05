"""Exact fixed-width lineage gates for complete SPD seen-val baseline outputs."""
import hashlib
import json
from pathlib import Path
import tarfile

from rbf_nested_seen_val_v2_common import R,sha
import prepare_seen_val_fixed_baselines as producer
import submit_seen_val_fixed_baselines as dispatch
import rbf_seen_val_forest_output_binding as shared
import publish_rbf_seen_val_forest_inputs as publication

DISPATCH=R/'source-freezes/rbf-seen-val-fixed-K1-K4-GPU-dispatch-v1-20261005'
DISPATCH_SHA='bd076e1b35f76598fc7be735809ebc1933b3915f9b8e9e97bc62d92f864a25da'
SHARED=R/'source-freezes/rbf-seen-val-bound-forest-output-reader-v1-20261005'
SHARED_SHA='39fb77dc9166f4c3125ad87ec66282da0561eb962901f1ce1bbd904e59d452b8'
SOURCE=R/'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/source.bytes'
SOURCE_SHA='038fa8118c9540d91073fbb8bf594fb6abefe8347f13bb69dfb006d27fcfda03'
KIND='rbf_seen_val_fixed_baseline_independent_bytes_events_factors_v1'
FAILED_KIND='rbf_seen_val_fixed_baseline_failed_candidate_independent_bytes_v1'
canonical=publication.canonical
contained=shared.contained
reference_inputs=shared.reference_inputs
verify_task_unchanged=shared.verify_task_unchanged


def source_gate():
    own=Path(__file__).resolve().parent
    freeze=json.loads((own/'source-freeze.json').read_bytes())
    assert freeze['kind']=='rbf_seen_val_fixed_baseline_output_reader_source_v1'
    for name,item in freeze['sources'].items():assert sha(own/name)==item['sha256']
    for item in freeze['references']:assert sha(item['path'])==item['sha256']
    assert sha(DISPATCH/'source-freeze.json')==DISPATCH_SHA
    assert sha(SHARED/'source-freeze.json')==SHARED_SHA
    for root in (DISPATCH,SHARED):
        prior=json.loads((root/'source-freeze.json').read_bytes())
        for name,item in prior['sources'].items():assert sha(root/name)==item['sha256']
    for module,root in ((producer,DISPATCH),(dispatch,DISPATCH),(shared,SHARED),(publication,DISPATCH)):
        path=Path(module.__file__).resolve()
        assert path.parent==own and sha(path)==sha(root/path.name)
    assert sha(SOURCE)==SOURCE_SHA
    shared.source_gate()
    return sha(own/'source-freeze.json')


def arguments(parser):
    parser.add_argument('--K',type=int,choices=(1,4),required=True)
    parser.add_argument('--seed',type=int,choices=(1337,2027,3407),required=True)
    for key in dispatch.PATH_ARGUMENTS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)


def qualify_job(args,Task,job,task):
    assert job['K']==args.K==job['plan']['baseline_K'] and job['seed']==args.seed==job['plan']['seed']
    assert set(job['input_paths'])==set(job['input_receipt_sha256'])==set(dispatch.PATH_ARGUMENTS)
    for key in dispatch.PATH_ARGUMENTS:
        assert job['input_paths'][key]==str(getattr(args,key))
        assert sha(getattr(args,key))==job['input_receipt_sha256'][key]
    payload=producer.local_cpu_prerequisite(args.K,args.seed,args.baseline_admission,args.baseline_byte_admission)
    proof,main,_,_=publication.prerequisites(args.main_admission,args.main_byte_admission,args.seed)
    records,bound=publication.local_inputs(args.seed,args.main_admission,proof,main)
    manifest=publication.manifest(args.seed,records,proof,main,Task)
    published=publication.publication(args.publication,manifest,Task)
    main_plan=publication.expected_plan(main,bound,published,4)
    base=producer.make_plan(args.K,main_plan,payload,sha(dispatch.PRODUCERS/f'K{args.K}'/'bootstrap.py'),4)
    dispatch.verify_existing(task,job,base)
    dispatch.remote_gate(job['plan'],args.publication,Task)
    return published


def artifact_keys(world):
    assert type(world) is int and world in (4,8)
    return {'receipt','seen-val-input-binding','seen-val-baseline-binding'}|{f'replay-rank{i}' for i in range(world)}


def source_map(plan):
    """The fixed backend records original package bytes, without exclusive patches."""
    assert sha(SOURCE)==plan['source']['sha256']==SOURCE_SHA
    result={}
    with tarfile.open(SOURCE) as archive:
        for member in archive:
            path=Path(member.name)
            if (path.parent==Path('transvision/models/event_track_v2x') and path.suffix=='.py') or member.name=='tools/event_track_v2x/persistent_mht_tracking.py':
                assert member.isfile() and member.name not in result
                result[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    assert 'transvision/models/event_track_v2x/paper_runtime.py' in result
    assert 'tools/event_track_v2x/persistent_mht_tracking.py' in result
    return result


def verify_report(plan,report,input_binding,baseline_binding,published):
    width=plan['baseline_K'];assert width in (1,4)
    assert plan['method']==plan['configuration']['method']=='topk'
    assert plan['configuration']['state']['active_limit']==width
    assert plan['configuration']['state']['decision_mode']=='retained'
    assert report['kind']==f'rbf_final_refit_seen_val_fixed_K{width}_candidate_v1'
    assert report['all_21_sequences_3316_events_completed'] is True
    assert input_binding['seed']==baseline_binding['seed']==plan['seed']
    assert input_binding['input_publication']==published==plan['main_seen_val_input_plan']['seen_val_input_publication']
    assert input_binding['full_train_interface_required'] is True
    assert baseline_binding['K']==width and baseline_binding['baseline_configuration_unchanged'] is True
    assert baseline_binding['baseline_train_CPU_prerequisite']==plan['baseline_train_CPU_prerequisite']
    assert baseline_binding['main_input_plan_sha256']==plan['main_seen_val_input_plan_sha256']
    assert baseline_binding['full_train_baseline_acceptance_inherited'] is False
    for key in ('measured_network_arrival_history_verified','same_resource_performance_accepted','paper_performance_complete'):
        assert input_binding[key] is baseline_binding[key] is False
    assert input_binding['validation_or_test_selection'] is input_binding['learned_Stage2_complete'] is False
    assert len(report['ranks'])==plan['world_size']
    assert sorted(v['rank'] for v in report['ranks'])==list(range(plan['world_size']))
    for rank in report['ranks']:
        assert rank['seed']==plan['seed'] and rank['method']=='topk'
        assert rank['world_size']==plan['world_size'] and rank['all_sequences_completed'] is True
        assert rank['TF32_matmul'] is rank['TF32_cudnn'] is False
        assert rank['gpu_uuid'] not in ('unavailable','None',None,'')
        assert 'sm_'+''.join(map(str,rank['capability'])) in rank['native_architectures']
    assert len({v['gpu_uuid'] for v in report['ranks']})==plan['world_size']


def verify_sequence(plan,per_plan,receipt,sequence,events,sources):
    assert plan['method']==plan['configuration']['method']=='topk'
    assert plan['configuration']['state']['active_limit']==plan['baseline_K']
    assert per_plan['kind']=='rbf_paper_replay_v1' and receipt['kind']=='rbf_paper_replay_receipt_v1'
    assert per_plan['expected_sequences']==receipt['completed_sequences']==[sequence]
    assert per_plan['expected_events']==receipt['completed_events']==len(events)
    assert per_plan['protocol']==dict(dataset='spd',split='val',candidates='rbf-all-class-top64-v1',
        evaluation_class='car',maximum_detections=64,minimum_raw_score=.05)
    assert per_plan['configuration']==plan['configuration'] and per_plan['fixture'] is False
    model=per_plan['model_binding']
    assert model['checkpoint_sha256']==plan['checkpoint']['sha256']
    assert model['model_sha256']==plan['final_refit_model_sha256'] and model['seed']==plan['seed']
    assert model['fit_split']=='train' and model['dataset']=='spd' and model['candidate_protocol']=='rbf-all-class-top64-v1'
    assert per_plan['events_sha256']==hashlib.sha256(canonical(events)).hexdigest()
    assert per_plan['cache_sha256']==plan['cache_manifest']['sha256']
    assert receipt['status']=='software_replay_completed' and per_plan['source_sha256']==sources


def verify_audit(audit,width):
    assert audit['kind']=='persistent_irreversible_identity_beam_v1' and audit['beam_width']==width
    assert audit['recovery_enabled'] is audit['archived_prefixes_used_for_recovery'] is False
    assert audit['reproduced_classical_mht'] is False
    assert audit['pruning_policy']=='fixed_top_k_root_classes_after_each_arrival_ordered_node'
    assert audit['output_policy']=='conditional_bayes_over_retained_classes_no_regret_fallback'
    assert audit['allocation_policy']=='fixed_width_no_adaptive_search'
