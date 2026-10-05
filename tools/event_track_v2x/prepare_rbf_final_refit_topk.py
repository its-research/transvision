"""Bind the frozen fixed-Top-K control to the distinct final-refit weights."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil

from prepare_rbf_final_refit_full_forest import sha, new, R

D = R/'source-freezes/rbf-final-refit-full-train-fixed-topK-GPU-v1-20261004'
MAIN = R/'source-freezes/rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004'
OLD = R/'source-freezes/rbf-original-joint-coupled-topk-GPU4-v5-native34-20261001'


def main():
    assert not D.exists(), 'create-once freeze; inspect existing preparation'
    parent = json.loads((MAIN/'preparation.json').read_bytes())
    for name,record in parent['execution_sources'].items():
        assert sha(MAIN/name)==record['sha256']
    old=(OLD/'bootstrap.py').read_text()
    assert sha(OLD/'bootstrap.py')=='a6013028f385e6c5db1725fce9710b812ad74bb5132b6351a0b592c529cc8efb'
    edits=[
        ("seqs=sorted(event_asset['origin_us_by_sequence'])[rank::4]", "seqs=sorted(event_asset['origin_us_by_sequence'])[rank::plan['world_size']]"),
        ('for i in range(4):', "for i in range(len(plan['forward_outputs'])):"),
        ('assert torch.cuda.device_count()==4', "assert plan['world_size'] in (4,8) and torch.cuda.device_count()==plan['world_size']"),
        ('nprocs=4,join=True', "nprocs=plan['world_size'],join=True"),
        ('len(reports)==4', "len(reports)==plan['world_size']"),
        ('for rank in range(4):', "for rank in range(plan['world_size']):"),
        ("'kind':'rbf_original_train_joint_coupled_forest_replay_candidate_v1'", "'kind':'rbf_final_refit_full_train_fixed_topK_candidate_v1'"),
        ("task_name='original joint coupled forest replay'", "task_name='final-refit fixed Top-K full-train control'"),
        ("'scope':'original paired train development only, nested checkpoint, deterministic allocation or fixed Top-K, no full Stage2/MHT/official evaluation claim'", "'scope':'full paired train, distinct frozen final-refit checkpoint, unchanged fixed Top-K K4 backend and limits; independent replay and same-resource comparison pending'"),
        ("from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache", "capability=torch.cuda.get_device_capability(rank);assert 'sm_'+str(capability[0])+str(capability[1]) in torch.cuda.get_arch_list(), 'native GPU architecture required';props=torch.cuda.get_device_properties(rank);assert str(getattr(props,'uuid','unavailable')) not in ('unavailable','None')\n from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache"),
        ("'device':torch.cuda.get_device_name(rank)", "'world_size':plan['world_size'],'gpu_uuid':str(props.uuid),'capability':list(capability),'native_architectures':torch.cuda.get_arch_list(),'device':torch.cuda.get_device_name(rank)"),
    ]
    candidate=old
    for before,after in edits:
        assert candidate.count(before)==1,before
        candidate=candidate.replace(before,after,1)
    compile(candidate,'bootstrap.py','exec')
    def functions(source):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(old),functions(candidate)
    assert {k:v for k,v in a.items() if k not in ('main','work')}=={k:v for k,v in b.items() if k not in ('main','work')}
    old_jobs=json.loads((R/'receipts/rbf-original-joint-coupled-topk-GPU4-v5-native34-dispatch-20261001.json').read_bytes())['jobs']
    seeds=[]
    for seed in (1337,2027,3407):
        legacy=next(v for v in old_jobs if v['seed']==seed)['plan']
        refit=next(v for v in parent['seeds'] if v['seed']==seed)['plan']
        for key in ('source','events','cache_archive','cache_manifest'):
            assert legacy[key]==refit[key]
        assert legacy['configuration']['method']=='topk' and legacy['configuration']['state']['active_limit']==4
        plan=dict(legacy)
        for key in ('checkpoint','weights_archive','forward_outputs','numeric_reference_admission_sha256',
            'forward_full_byte_and_coverage_admission_sha256','final_refit_training_byte_admission_sha256',
            'final_refit_NN_numeric_completion_sha256','final_refit_model_sha256','original_nested_model_sha256'):
            plan[key]=refit[key]
        plan.update(world_size=None,bootstrap_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
            scope='new final-refit fixed Top-K K4 on full original train; no independent/same-resource performance claim',
            full_forest_independently_accepted=False,learned_Stage2_complete=False)
        seeds.append(dict(seed=seed,plan=plan))
    dispatcher=(MAIN/'submit_rbf_final_refit_full_forest.py').read_text()
    for before,after in (
        ('rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004','rbf-final-refit-full-train-fixed-topK-GPU-v1-20261004'),
        ('rbf-final-refit-full-train-exclusive-forest-GPU-dispatch','rbf-final-refit-full-train-fixed-topK-GPU-dispatch'),
        ('rbf_final_refit_full_train_exclusive_forest_GPU_dispatch_v1','rbf_final_refit_full_train_fixed_topK_GPU_dispatch_v1'),
        ('exclusive_kernel_and_configuration_unchanged','fixed_topK_backend_and_configuration_unchanged'),
        ('RBF final-refit full-train exclusive forest seed','RBF final-refit full-train fixed Top-K K4 seed'),
        ("'exclusive-recoverable-forest','bound-allocation'", "'fixed-topK-K4-control'")):
        assert before in dispatcher
        dispatcher=dispatcher.replace(before,after)
    compile(dispatcher,'submit_rbf_final_refit_topk.py','exec')
    D.mkdir()
    (D/'bootstrap.py').write_text(candidate)
    (D/'submit_rbf_final_refit_topk.py').write_text(dispatcher)
    shutil.copyfile(__file__,D/Path(__file__).name)
    for name in ('rbf_nested_seen_val_v2_common.py','submit_rbf_final_identity.py','submit_rbf_seen_val_joint_identity.py'):
        shutil.copyfile(MAIN/name,D/name)
    new(D/'source-control.json',dict(kind='rbf_final_refit_fixed_topK_source_control_v1',
        parent_bootstrap_sha256=sha(OLD/'bootstrap.py'),candidate_bootstrap_sha256=sha(D/'bootstrap.py'),
        literal_edits=[dict(before=a,after=b) for a,b in edits],frozen_backend_source_package_unchanged=True,
        configuration_and_K4_unchanged=True,scoring_tolerances_unchanged=True,
        function_bodies_except_partition_runtime_metadata_and_receipt_unchanged=True,
        sequence_partition_controls=json.loads((MAIN/'source-control.json').read_bytes())['sequence_partition_controls'],
        same_resource_comparison_accepted=False))
    for item in seeds:
        item['plan']['dispatcher_sha256']=sha(D/'submit_rbf_final_refit_topk.py')
    prep=dict(kind='rbf_final_refit_full_train_fixed_topK_GPU_preparation_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seeds=seeds,
        bootstrap_sha256=sha(D/'bootstrap.py'),
        execution_sources={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(D.iterdir())},
        prerequisite_receipts=parent['prerequisite_receipts']+[dict(path=str(MAIN/'preparation.json'),sha256=sha(MAIN/'preparation.json'))],
        all_three_distinct_final_refit_weights_and_full_NN_rows_admitted=True,
        fixed_topK_backend_and_configuration_unchanged=True,GPU_model_restriction=False,L40S_CPU_only=True,
        same_resource_comparison_accepted=False,paper_performance_complete=False)
    new(D/'preparation.json',prep)
    print(json.dumps(dict(preparation=str(D/'preparation.json'),sha256=sha(D/'preparation.json'),method='topk',K=4)))


if __name__=='__main__':
    main()
