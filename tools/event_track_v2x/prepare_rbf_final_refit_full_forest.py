"""Bind final-refit models to unchanged exclusive real-train replay inputs."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil

R = Path('/Volumes/Data/test/recover-before-fuse')
D = R/'source-freezes/rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004'
OLD = R/'source-freezes/rbf-exclusive-action-capacity-undecided-GPU4-v2-20261002'
HELPERS = R/'source-freezes/rbf-matching-seen-val-joint-GPU-publish-dispatch-v3-live-reservations-20261004'
INDEX = R/'receipts/rbf-final-refit-three-seed-all-row-full-independent-numeric-acceptance-20261004.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def main():
    assert not D.exists(), 'create-once freeze; inspect existing preparation instead'
    old = (OLD/'bootstrap.py').read_text()
    assert sha(OLD/'bootstrap.py') == 'de4ca7d57636209741e008e84712dd889cccd06fa107129422759a9c1b112d4e'
    frozen = json.loads((HELPERS/'preparation.json').read_bytes())
    for name,spec in frozen['sources'].items():
        assert sha(HELPERS/name) == spec['sha256']
    edits = [
        ("seqs=sorted(event_asset['origin_us_by_sequence'])[rank::4]", "seqs=sorted(event_asset['origin_us_by_sequence'])[rank::plan['world_size']]"),
        ('for i in range(4):', "for i in range(len(plan['forward_outputs'])):"),
        ('assert torch.cuda.device_count()==4', "assert plan['world_size'] in (4,8) and torch.cuda.device_count()==plan['world_size']"),
        ('nprocs=4,join=True', "nprocs=plan['world_size'],join=True"),
        ('len(reports)==4', "len(reports)==plan['world_size']"),
        ('for rank in range(4):', "for rank in range(plan['world_size']):"),
        ("'kind':'rbf_exclusive_action_capacity_undecided_full_replay_candidate_v1'", "'kind':'rbf_final_refit_exclusive_full_train_forest_candidate_v1'"),
        ("task_name='exclusive action capacity undecided full replay candidate'", "task_name='final-refit exclusive full-train forest candidate'"),
        ("'scope':'original paired train development only, nested checkpoint, explicit residual disjoint regions, deterministic bound allocation, no full Stage2/MHT/official evaluation claim'", "'scope':'full paired train, distinct frozen final-refit checkpoint, unchanged exclusive kernel and limits; bound allocator only; full independent forest/learned Stage2/MHT/metrics pending'"),
        ("from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache", "capability=torch.cuda.get_device_capability(rank);assert 'sm_'+str(capability[0])+str(capability[1]) in torch.cuda.get_arch_list(), 'native GPU architecture required';props=torch.cuda.get_device_properties(rank);assert str(getattr(props,'uuid','unavailable')) not in ('unavailable','None')\n from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache"),
        ("'device':torch.cuda.get_device_name(rank)", "'world_size':plan['world_size'],'gpu_uuid':str(props.uuid),'capability':list(capability),'native_architectures':torch.cuda.get_arch_list(),'device':torch.cuda.get_device_name(rank)"),
    ]
    candidate = old
    for before,after in edits:
        assert candidate.count(before) == 1, ('unexpected source edit',before)
        candidate = candidate.replace(before,after,1)
    compile(candidate,'bootstrap.py','exec')
    old_ast,new_ast = ast.parse(old),ast.parse(candidate)
    def values(tree):
        return {n.targets[0].id:ast.dump(n.value,include_attributes=False) for n in tree.body
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id in ('PATCHES','REPLACEMENTS')}
    assert values(old_ast) == values(new_ast)
    def functions(tree):
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    left,right = functions(old_ast),functions(new_ast)
    assert {k:v for k,v in left.items() if k not in ('work','main')} == {k:v for k,v in right.items() if k not in ('work','main')}
    partitions = []
    for world in (4,8):
        groups = [list(range(46))[rank::world] for rank in range(world)]
        assert all(groups) and sorted(v for group in groups for v in group) == list(range(46))
        assert len({v for group in groups for v in group}) == 46
        partitions.append(dict(world_size=world,sequences_per_rank=list(map(len,groups)),full_coverage_no_duplicates=True))
    index = json.loads(INDEX.read_bytes())
    assert index['full_independent_joint_attention_cross_source_temporal_motion_numerical_pass'] is True
    assert index['rows'] == 677744 and len(index['seeds']) == 3
    old_jobs = json.loads((R/'receipts/rbf-exclusive-action-capacity-undecided-GPU4-v2-dispatch-20261002.json').read_bytes())['jobs']
    proof_paths = [INDEX,OLD/'source-freeze.json',HELPERS/'preparation.json']
    seeds = []
    digest = hashlib.sha256(candidate.encode()).hexdigest()
    for seed in (1337,2027,3407):
        accepted = next(v for v in index['seeds'] if v['seed']==seed)
        for key in ('training_byte_proof','prediction_byte_proof','numeric_completion'):
            p=Path(accepted[key]);assert sha(p)==accepted[key+'_sha256'];proof_paths.append(p)
        assert accepted['numeric_failures']==0
        train = json.loads(Path(accepted['training_byte_proof']).read_bytes())
        forward = json.loads(Path(accepted['prediction_byte_proof']).read_bytes())
        ck_path = Path(accepted['training_byte_proof']).parent/'checkpoint'
        ck = json.loads(ck_path.read_bytes())
        assert sha(ck_path)==train['artifacts']['checkpoint']['sha256']
        old_ck = json.loads((R/f'artifacts/rbf-joint-identity-full-independent-numpy-v1-20261001/seed{seed}/weights_archive-unpack/training/seed-{seed}/checkpoint.json').read_bytes())
        assert ck['row_protocol']==old_ck['row_protocol'] and ck['frozen_cache_identity']==old_ck['frozen_cache_identity']
        assert ck['model_sha256']!=old_ck['model_sha256'], 'same weights would duplicate the original experiment'
        old_job = next(j for j in old_jobs if j['seed']==seed)
        plan = dict(old_job['plan'])
        original_inputs = {k:plan[k] for k in ('source','events','cache_archive','cache_manifest')}
        plan.update(checkpoint=train['artifacts']['checkpoint'],weights_archive=train['artifacts']['identity-training'],
            forward_outputs=[v for k,v in sorted(forward['artifacts'].items()) if k.startswith('predictions-rank')],
            bootstrap_sha256=digest,dispatcher_sha256=sha(Path(__file__).with_name('submit_rbf_final_refit_full_forest.py')),
            world_size=None,numeric_reference_admission_sha256=sha(INDEX),
            forward_full_byte_and_coverage_admission_sha256=sha(accepted['prediction_byte_proof']),
            final_refit_training_byte_admission_sha256=sha(accepted['training_byte_proof']),
            final_refit_NN_numeric_completion_sha256=sha(accepted['numeric_completion']),
            scope='distinct final-refit weights, full original paired train, same exclusive kernel/configuration; raw candidate before full independent forest admission',
            final_refit_model_sha256=ck['model_sha256'],original_nested_model_sha256=old_ck['model_sha256'],
            full_forest_independently_accepted=False,learned_Stage2_complete=False)
        assert original_inputs == {k:plan[k] for k in original_inputs}
        assert plan['configuration'] == old_job['plan']['configuration']
        seeds.append(dict(seed=seed,plan=plan))
    D.mkdir()
    (D/'bootstrap.py').write_text(candidate)
    workspace = Path(__file__).resolve().parent
    for name in ('prepare_rbf_final_refit_full_forest.py','submit_rbf_final_refit_full_forest.py'):
        shutil.copyfile(workspace/name,D/name)
    for name in ('rbf_nested_seen_val_v2_common.py','submit_rbf_final_identity.py','submit_rbf_seen_val_joint_identity.py'):
        shutil.copyfile(HELPERS/name,D/name)
    control = dict(kind='rbf_final_refit_full_train_exclusive_forest_source_control_v1',
        parent_bootstrap_sha256=sha(OLD/'bootstrap.py'),candidate_bootstrap_sha256=sha(D/'bootstrap.py'),
        literal_edits=[dict(before=a,after=b) for a,b in edits],
        exclusive_kernel_patch_AST_identical=True,capacity_repair_AST_identical=True,
        unchanged_fetch_and_unpack_functions=True,sequence_partition_controls=partitions,
        numerical_tolerances_unchanged=True,resource_limits_unchanged=True,
        fixture_or_partition_test_is_not_real_forest_acceptance=True)
    new(D/'source-control.json',control)
    preparation = dict(kind='rbf_final_refit_full_train_exclusive_forest_GPU_preparation_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        bootstrap_sha256=sha(D/'bootstrap.py'),seeds=seeds,
        execution_sources={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(D.iterdir())},
        prerequisite_receipts=[dict(path=str(p),sha256=sha(p)) for p in dict.fromkeys(proof_paths)],
        all_three_distinct_final_refit_weights_and_full_NN_rows_admitted=True,
        exclusive_kernel_and_configuration_unchanged=True,old_nested_results_and_auditors_preserved=True,
        GPU_model_restriction=False,L40S_CPU_only=True,full_forest_independently_accepted=False,
        teacher_prerequisite_gate_bypassed=False,paper_performance_complete=False)
    new(D/'preparation.json',preparation)
    print(json.dumps(dict(preparation=str(D/'preparation.json'),sha256=sha(D/'preparation.json'),
        seeds=[v['seed'] for v in seeds],distinct_models=True,real_forest_accepted=False)))


if __name__=='__main__':
    main()
