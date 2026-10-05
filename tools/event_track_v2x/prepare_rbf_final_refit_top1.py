"""Bind the existing fixed Top-1 control to admitted final-refit weights.

This prepares a distinct experiment. It preserves the original baseline kernel,
configuration and caps, and does not inherit old-model replay acceptance.
"""
import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil

from prepare_rbf_final_refit_full_forest import R, new, sha

D = R/'source-freezes/rbf-final-refit-full-train-fixed-Top1-GPU-v1-20261004'
TOPK = R/'source-freezes/rbf-final-refit-full-train-fixed-topK-GPU-v1-20261004'
OLD = R/'source-freezes/rbf-original-joint-fixed-top1-GPU4-v1-20261002'
OLD_INDEX = R/'receipts/rbf-Top1-three-seed-full-cache-causal-final-independent-index-20261003.json'


def main():
    assert not D.exists(), 'create-once freeze; inspect existing preparation'
    parent = json.loads((TOPK/'preparation.json').read_bytes())
    for name, record in parent['execution_sources'].items():
        assert sha(TOPK/name) == record['sha256']
    for proof in parent['prerequisite_receipts']:
        assert sha(proof['path']) == proof['sha256']
    old_freeze = json.loads((OLD/'source-freeze.json').read_bytes())
    assert sha(OLD/'bootstrap.py') == old_freeze['bootstrap_sha256'] == 'c05abff174a834017fa925064f4ec43aca2b11df885876d48ae6deee016e5ecf'
    assert sha(OLD/'plans.json') == old_freeze['plans_sha256']
    old_plans = json.loads((OLD/'plans.json').read_bytes())
    old_index = json.loads(OLD_INDEX.read_bytes())
    assert old_index['all_203_feature_and_causal_scope_accepted'] is True
    assert len(old_index['cohort']) == 3
    old = (OLD/'bootstrap.py').read_text()
    edits = [
        ("seqs=sorted(event_asset['origin_us_by_sequence'])[rank::4]", "seqs=sorted(event_asset['origin_us_by_sequence'])[rank::plan['world_size']]"),
        ('for i in range(4):', "for i in range(len(plan['forward_outputs'])):"),
        ('assert torch.cuda.device_count()==4', "assert plan['world_size'] in (4,8) and torch.cuda.device_count()==plan['world_size']"),
        ('nprocs=4,join=True', "nprocs=plan['world_size'],join=True"),
        ('len(reports)==4', "len(reports)==plan['world_size']"),
        ('for rank in range(4):', "for rank in range(plan['world_size']):"),
        ("'kind':'rbf_original_train_joint_coupled_forest_replay_candidate_v1'", "'kind':'rbf_final_refit_full_train_fixed_Top1_candidate_v1'"),
        ("task_name='original joint coupled forest replay'", "task_name='final-refit fixed Top-1 full-train control'"),
        ("'scope':'original paired train development only, nested checkpoint, deterministic allocation or fixed Top-K, no full Stage2/MHT/official evaluation claim'", "'scope':'full paired train, distinct frozen final-refit checkpoint, unchanged fixed Top-1 backend and limits; independent replay and same-resource comparison pending'"),
        ("from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache", "capability=torch.cuda.get_device_capability(rank);assert 'sm_'+str(capability[0])+str(capability[1]) in torch.cuda.get_arch_list(), 'native GPU architecture required';props=torch.cuda.get_device_properties(rank);assert str(getattr(props,'uuid','unavailable')) not in ('unavailable','None')\n from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache"),
        ("'device':torch.cuda.get_device_name(rank)", "'world_size':plan['world_size'],'gpu_uuid':str(props.uuid),'capability':list(capability),'native_architectures':torch.cuda.get_arch_list(),'device':torch.cuda.get_device_name(rank)"),
    ]
    candidate = old
    for before, after in edits:
        assert candidate.count(before) == 1, before
        candidate = candidate.replace(before, after, 1)
    compile(candidate, 'bootstrap.py', 'exec')

    def functions(source):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef)}

    left, right = functions(old), functions(candidate)
    assert {k: v for k, v in left.items() if k not in ('main', 'work')} == {
        k: v for k, v in right.items() if k not in ('main', 'work')}
    assert "default_configuration(plan['method'],allocation='bound',width=1)" in candidate
    seeds = []
    controls = []
    for seed in (1337, 2027, 3407):
        legacy = next(v for v in old_plans if v['seed'] == seed)['plan']
        refit = next(v for v in parent['seeds'] if v['seed'] == seed)['plan']
        for key in ('source', 'events', 'cache_archive', 'cache_manifest'):
            assert legacy[key] == refit[key]
        expected_config = copy.deepcopy(refit['configuration'])
        assert expected_config['method'] == 'topk' and expected_config['state']['active_limit'] == 4
        expected_config['state']['active_limit'] = 1
        assert expected_config == legacy['configuration']
        legacy_result = next(v for v in old_index['cohort'] if v['seed'] == seed)
        assert legacy_result['checkpoint_sha256'] == legacy['checkpoint']['sha256']
        assert refit['checkpoint']['sha256'] != legacy_result['checkpoint_sha256']
        plan = copy.deepcopy(refit)
        plan.update(configuration=expected_config, world_size=None,
            bootstrap_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
            baseline_variant='fixed_Top1_final_refit_full_train_v1',
            scope='new final-refit fixed Top-1 on full original train; full independent and same-resource performance admission pending',
            original_Top1_plan_sha256=hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            comparison_scope='same final-refit model, source cache, actual event schedule and caps as K4; hardware resources and performance still require independent comparison',
            old_nested_Top1_acceptance_inherited=False,
            full_forest_independently_accepted=False, learned_Stage2_complete=False)
        seeds.append(dict(seed=seed, plan=plan))
        controls.append(dict(seed=seed, original_Top1_task_id=legacy_result['task_id'],
            original_Top1_checkpoint_sha256=legacy_result['checkpoint_sha256'],
            final_refit_checkpoint_sha256=refit['checkpoint']['sha256'],
            final_refit_model_sha256=refit['final_refit_model_sha256'],
            original_nested_model_sha256=refit['original_nested_model_sha256'],
            only_configuration_difference_from_final_refit_K4='state.active_limit: 4 -> 1',
            original_Top1_configuration_unchanged=True, old_acceptance_not_inherited=True))
    dispatcher = (TOPK/'submit_rbf_final_refit_topk.py').read_text()
    for before, after in (
        ('rbf-final-refit-full-train-fixed-topK-GPU-v1-20261004', 'rbf-final-refit-full-train-fixed-Top1-GPU-v1-20261004'),
        ('rbf-final-refit-full-train-fixed-topK-GPU-dispatch', 'rbf-final-refit-full-train-fixed-Top1-GPU-dispatch'),
        ('rbf_final_refit_full_train_fixed_topK_GPU_dispatch_v1', 'rbf_final_refit_full_train_fixed_Top1_GPU_dispatch_v1'),
        ('fixed_topK_backend_and_configuration_unchanged', 'fixed_Top1_backend_and_configuration_unchanged'),
        ('RBF final-refit full-train fixed Top-K K4 seed', 'RBF final-refit full-train fixed Top-1 seed'),
        ("'fixed-topK-K4-control'", "'fixed-Top1-control'")):
        assert before in dispatcher
        dispatcher = dispatcher.replace(before, after)
    compile(dispatcher, 'submit_rbf_final_refit_top1.py', 'exec')
    D.mkdir()
    (D/'bootstrap.py').write_text(candidate)
    (D/'submit_rbf_final_refit_top1.py').write_text(dispatcher)
    shutil.copyfile(__file__, D/Path(__file__).name)
    for name in ('rbf_nested_seen_val_v2_common.py', 'submit_rbf_final_identity.py', 'submit_rbf_seen_val_joint_identity.py'):
        shutil.copyfile(TOPK/name, D/name)
    new(D/'source-control.json', dict(kind='rbf_final_refit_fixed_Top1_source_control_v1',
        parent_bootstrap_sha256=sha(OLD/'bootstrap.py'), candidate_bootstrap_sha256=sha(D/'bootstrap.py'),
        literal_edits=[dict(before=a, after=b) for a, b in edits], seed_configuration_controls=controls,
        frozen_backend_source_package_unchanged=True, original_Top1_configuration_and_caps_unchanged=True,
        scoring_tolerances_unchanged=True, TF32_enabled=False,
        function_bodies_except_partition_runtime_metadata_and_receipt_unchanged=True,
        sequence_partition_controls=json.loads((TOPK/'source-control.json').read_bytes())['sequence_partition_controls'],
        same_resource_comparison_accepted=False, old_model_acceptance_inherited=False))
    for item in seeds:
        item['plan']['dispatcher_sha256'] = sha(D/'submit_rbf_final_refit_top1.py')
    prerequisites = parent['prerequisite_receipts'] + [dict(path=str(p), sha256=sha(p)) for p in
        (TOPK/'preparation.json', OLD/'source-freeze.json', OLD/'plans.json', OLD_INDEX)]
    prep = dict(kind='rbf_final_refit_full_train_fixed_Top1_GPU_preparation_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), seeds=seeds,
        bootstrap_sha256=sha(D/'bootstrap.py'),
        execution_sources={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(D.iterdir())},
        prerequisite_receipts=prerequisites,
        all_three_distinct_final_refit_weights_and_full_NN_rows_admitted=True,
        fixed_Top1_backend_and_configuration_unchanged=True, GPU_model_restriction=False, L40S_CPU_only=True,
        independent_K1_full_output_reader_and_CPU_gates_prepared=False,
        independent_K4_gates_cannot_be_inherited=True,
        full_forest_independently_accepted=False, same_resource_comparison_accepted=False, paper_performance_complete=False)
    new(D/'preparation.json', prep)
    print(json.dumps(dict(preparation=str(D/'preparation.json'), sha256=sha(D/'preparation.json'), method='topk', K=1)))


if __name__ == '__main__':
    main()
