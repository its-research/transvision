"""Freeze the named three-seed recovery-off producer and byte reader.

No dispatch or upload occurs here. The original real inputs and tolerances stay
fixed; this separate candidate changes historical support, so no parent forest
or learned-priority acceptance transfers to it.
"""
import ast
import base64
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zlib

from rbf_nested_seen_val_v2_common import R, new, register, sha

PARENT = R/'source-freezes/rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004'
READER = R/'source-freezes/rbf-final-refit-main-topK-independent-output-reader-v1-20261004'
CANDIDATE = R/'source-freezes/rbf-recovery-off-original-source-bound-candidate-v3-20261005'
INPUT = R/'artifacts/rbf-recovery-off-final-checkpoint-input-contract-v1-20261005/acceptance.json'
NAME = 'rbf-final-refit-recovery-off-bound-GPU-v1-20261005'
JOURNAL = 'rbf-final-refit-recovery-off-bound-GPU-dispatch-20261005.json'
OUT = R/'source-freezes'/NAME
PREFIX = 'transvision/models/event_track_v2x/'


def replace_one(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def embedded(text):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(text).body
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ('PATCHES','REPLACEMENTS')}


def replace_assignment(text, name, value):
    node = next(n for n in ast.parse(text).body if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
    lines=text.splitlines(True)
    return ''.join(lines[:node.lineno-1])+name+'='+repr(value)+'\n'+''.join(lines[node.end_lineno:])


def main():
    assert not OUT.exists()
    assert sha(CANDIDATE/'source-freeze.json') == 'b778aee162f177066da207b70ecc807578be687373583a6b9748fa63b3aa9b70'
    assert sha(INPUT) == '760e966dbeb9b4652c69d6641985ae5be806c179cc6d3dbf6a08261102a1481f'
    proof=json.loads(INPUT.read_bytes());assert proof['all_three_actual_final_checkpoints_loaded_and_bound']
    freeze=json.loads((CANDIDATE/'source-freeze.json').read_bytes())
    for name,spec in freeze['sources'].items():assert sha(CANDIDATE/name)==spec['sha256']
    original=json.loads((PARENT/'preparation.json').read_bytes())
    for name,spec in original['execution_sources'].items():assert sha(PARENT/name)==spec['sha256']
    for item in original['prerequisite_receipts']:assert sha(item['path'])==item['sha256']
    old=(PARENT/'bootstrap.py').read_text();tables=embedded(old)
    # These are additive modules absent from the original archived source.
    for name in ('exclusive_paper_runtime.py','recovery_off_tracking.py','recovery_off_paper_runtime.py'):
        p=CANDIDATE/PREFIX/name;data=p.read_bytes()
        tables['PATCHES'][PREFIX+name]=dict(sha256=sha(p),data=base64.b64encode(zlib.compress(data)).decode())
    progress=CANDIDATE/PREFIX/'experiment_progress.py'
    source_manifest=json.loads((R/'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/manifest.bytes').read_bytes())
    old_progress=next(v['sha256'] for v in source_manifest['files'] if v['path']==PREFIX+'experiment_progress.py')
    tables['REPLACEMENTS'][PREFIX+'experiment_progress.py']=dict(before_sha256=old_progress,
        sha256=sha(progress),data=base64.b64encode(zlib.compress(progress.read_bytes())).decode())
    candidate=old
    for name,table in tables.items():candidate=replace_assignment(candidate,name,table)
    edits=[
        ('from transvision.models.event_track_v2x.exclusive_paper_runtime import replay,default_configuration',
         'from transvision.models.event_track_v2x.recovery_off_paper_runtime import replay,default_configuration'),
        ("task_name='final-refit exclusive full-train forest candidate'", "task_name='final-refit recovery-off bound full-train candidate'"),
        ("'kind':'rbf_final_refit_exclusive_full_train_forest_candidate_v1'", "'kind':'rbf_final_refit_recovery_off_bound_full_train_candidate_v1'"),
        ("'scope':'same exclusive factory with capacity-undecided action handling; unchanged model, limits and full real inputs'",
         "'scope':'explicit event-boundary recovery-off candidate; same real inputs, model, limits and bound allocator; full independent ablation acceptance pending'"),
        ("'scope':'full paired train, distinct frozen final-refit checkpoint, unchanged exclusive kernel and limits; bound allocator only; full independent forest/learned Stage2/MHT/metrics pending'",
         "'scope':'full paired train, same final checkpoint and limits; event-boundary irreversible historical support; bound allocator only, independent recovery-off forest/state/metrics pending'")]
    for before,after in edits:candidate=replace_one(candidate,before,after)
    old_work=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name=='work')
    new_work=next(n for n in ast.parse(candidate).body if isinstance(n,ast.FunctionDef) and n.name=='work')
    for node in ast.walk(new_work):
        if isinstance(node,ast.ImportFrom) and node.module=='transvision.models.event_track_v2x.recovery_off_paper_runtime':
            node.module='transvision.models.event_track_v2x.exclusive_paper_runtime'
    assert ast.dump(old_work)==ast.dump(new_work), 'actual GPU producer math must remain unchanged'
    compile(candidate,'bootstrap.py','exec')
    dispatcher=(PARENT/'submit_rbf_final_refit_full_forest.py').read_text()
    substitutions=[
        ('rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004',NAME),
        ('rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json',JOURNAL),
        ('rbf_final_refit_full_train_exclusive_forest_GPU_dispatch_v1','rbf_final_refit_recovery_off_bound_GPU_dispatch_v1'),
        ('rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-', 'rbf-final-refit-recovery-off-bound-GPU-dispatch-'),
        ('exclusive_kernel_and_configuration_unchanged','only_declared_recovery_and_progress_changes'),
        ('RBF final-refit full-train exclusive forest seed','RBF final-refit recovery-off bound full-train seed'),
        ('exclusive-recoverable-forest','exclusive-recovery-off-forest')]
    for before,after in substitutions:dispatcher=replace_one(dispatcher,before,after)
    dispatcher=replace_one(dispatcher,"    jobs = json.loads(JOURNAL.read_bytes())['jobs'] if JOURNAL.exists() else []",
        "    from recovery_off_dispatch_gate import require_qualified_source\n"
        "    require_qualified_source(ROOT, prepared)\n"
        "    jobs = json.loads(JOURNAL.read_bytes())['jobs'] if JOURNAL.exists() else []")
    compile(dispatcher,'dispatch.py','exec')
    # A separate reader retains byte, database, input and normalized NN checks.
    # It accepts only this intervention's explicit audit representation.
    reader=(READER/'read_rbf_final_refit_forest_outputs.py').read_text()
    assert sha(READER/'read_rbf_final_refit_forest_outputs.py')=='53747489a446c8990cfc6d9310eb686a69b39d99bde226c8945665a974d83f61'
    reader=replace_one(reader,"choices=('rbf','topk')", "choices=('rbf',)")
    reader=replace_one(reader,"journal=R/f'receipts/rbf-final-refit-full-train-{variant}-GPU-dispatch-20261004.json'", "journal=R/'receipts/"+JOURNAL+"'")
    reader=replace_one(reader,'rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/{args.method}/seed{args.seed}',
        'rbf-final-refit-recovery-off-bound-independent-byte-factor-v1-20261005/seed{args.seed}')
    reader=reader.replace('rbf_final_refit_exclusive_full_train_forest_candidate_v1','rbf_final_refit_recovery_off_bound_full_train_candidate_v1')
    reader=reader.replace('rbf_final_refit_forest_failed_candidate_independent_bytes_v1','rbf_final_refit_recovery_off_failed_candidate_independent_bytes_v1')
    reader=reader.replace('rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1','rbf_final_refit_recovery_off_bound_independent_bytes_events_factor_admission_v1')
    reader=replace_one(reader,"assert all(c['representation']=='exclusive_root_partition_regions_v1' for c in value['components'])",
        "assert value['recovery_enabled'] is False and value['kind']=='persistent_exclusive_event_boundary_recovery_off_v1'\n                            assert value['original_model_risk_certified'] is False\n                            assert all(c['representation']=='exclusive_recovery_off_prefix_regions_v1' and c['complete_raw_support_retained'] is False for c in value['components'])")
    reader=replace_one(reader,"assert plan['method']==args.method and world in (4,8)",
        "assert plan['method']==args.method and world in (4,8)\n    assert plan['configuration']['backend']=='exclusive_event_boundary_recovery_off_v1'\n    assert plan['configuration']['limits']['recovery_off_version']==1\n    assert plan['recovery_off_source_freeze_sha256']=='"+sha(CANDIDATE/'source-freeze.json')+"'")
    compile(reader,'read_outputs.py','exec')
    digest=lambda value:hashlib.sha256(value.encode()).hexdigest()
    seeds=[];configuration=proof['configuration']
    for item in original['seeds']:
        plan=dict(item['plan']);seed=item['seed']
        checked=next(v for v in proof['seeds'] if v['seed']==seed)
        assert plan['final_refit_model_sha256']==checked['model_sha256']
        for key,spec in checked['input_cloud_assets'].items():assert plan[key]==spec
        plan.update(configuration=configuration, bootstrap_sha256=digest(candidate),dispatcher_sha256=digest(dispatcher),
            exclusive_patches={k:v['sha256'] for k,v in tables['PATCHES'].items()},
            source_replacements={k:{s:v[s] for s in ('before_sha256','sha256')} for k,v in tables['REPLACEMENTS'].items()},
            scope='distinct recovery-off bound ablation; same final model, all-class real train inputs and limits',
            recovery_off_source_freeze_sha256=sha(CANDIDATE/'source-freeze.json'),
            recovery_off_real_checkpoint_input_contract_sha256=sha(INPUT),
            original_exclusive_acceptance_inherited=False, full_forest_independently_accepted=False,
            recovery_off_independent_semantics_accepted=False, learned_Stage2_complete=False)
        seeds.append(dict(seed=seed,plan=plan))
    OUT.mkdir()
    for name,text in (('bootstrap.py',candidate),('dispatch.py',dispatcher),('read_outputs.py',reader)):
        (OUT/name).write_text(text)
    for name in ('rbf_nested_seen_val_v2_common.py','submit_rbf_final_identity.py','submit_rbf_seen_val_joint_identity.py'):
        shutil.copyfile(PARENT/name,OUT/name)
    for name in ('recovery_off_dispatch_gate.py','qualify_recovery_off_gpu_source.py'):
        shutil.copyfile(Path(__file__).parent/name,OUT/name)
    shutil.copyfile(__file__,OUT/Path(__file__).name)
    control=dict(kind='rbf_recovery_off_GPU_source_control_v1',parent_bootstrap_sha256=sha(PARENT/'bootstrap.py'),
        runtime_source_candidate_sha256=sha(CANDIDATE/'source-freeze.json'),literal_bootstrap_edits=edits,
        work_function_AST_identical_except_runtime_import=True,NN_tolerance_unchanged=True,
        original_model_cache_schedule_unchanged=True,physical_collision_dispatch_logic_unchanged=True,
        full_forest_or_recovery_semantics_independently_accepted=False)
    new(OUT/'source-control.json',control)
    prepared=dict(kind='rbf_recovery_off_bound_real_train_GPU_preparation_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seeds=seeds,
        bootstrap_sha256=sha(OUT/'bootstrap.py'),
        execution_sources={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in OUT.iterdir()},
        prerequisite_receipts=original['prerequisite_receipts']+[
            dict(path=str(INPUT),sha256=sha(INPUT)),dict(path=str(CANDIDATE/'source-freeze.json'),sha256=sha(CANDIDATE/'source-freeze.json'))],
        all_three_distinct_final_refit_weights_and_full_NN_rows_admitted=True,
        only_declared_recovery_and_progress_changes=True,
        original_exclusive_acceptance_inherited=False,full_forest_independently_accepted=False,
        learned_Stage2_complete=False,paper_performance_complete=False,
        source_control_qualification_pending=True)
    new(OUT/'preparation.json',prepared);register(OUT/'preparation.json',prepared['kind'])
    print(json.dumps(dict(preparation=str(OUT/'preparation.json'),sha256=sha(OUT/'preparation.json'),
        GPU_dispatched=False,full_forest_accepted=False)),flush=True)


if __name__=='__main__':main()
