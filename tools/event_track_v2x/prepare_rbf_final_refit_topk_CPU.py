"""Create-once fixed TopK full-state driver; no old acceptance is inherited."""
import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil

from rbf_final_refit_topk_binding import INDEX, MAIN_CPU, OLD_JOURNAL, PRODUCER, canonical, validate_local_output
from rbf_final_refit_forest_binding import validate_final_model
from rbf_nested_seen_val_v2_common import R, new, register, sha

D = R/'source-freezes/rbf-final-refit-fixed-topK-full-independent-CPU-v1-20261004'


def main():
    assert not D.exists(), 'immutable source preparation already exists'
    main_freeze = json.loads((MAIN_CPU/'source-freeze.json').read_bytes())
    for name, digest in main_freeze['sources'].items():
        assert sha(MAIN_CPU/name) == digest
    preparation = json.loads((PRODUCER/'preparation.json').read_bytes())
    for name, spec in preparation['execution_sources'].items():
        assert sha(PRODUCER/name) == spec['sha256']
    old_jobs = json.loads(OLD_JOURNAL.read_bytes())['jobs']
    entries = json.loads(INDEX.read_bytes())['seeds']
    sources = [Path(__file__), Path(__file__).with_name('rbf_final_refit_topk_binding.py'),
        Path(__file__).with_name('accept_final_refit_topk_cohort.py')]
    for path in sources:
        ast.parse(path.read_bytes())
    references = [MAIN_CPU/'source-freeze.json', MAIN_CPU/'source-control.json',
        MAIN_CPU/'final_cache203.py', MAIN_CPU/'rbf_final_refit_forest_binding.py',
        R/'source-freezes/rbf-independent-fresh-branch-state-v3-normalized-admission-20261001/oracle.py',
        R/'source-freezes/rbf-independent-exclusive-forest-structure-recovery-v1-20261002/causal.py',
        R/'source-freezes/rbf-independent-exclusive-forest-structure-recovery-v1-20261002/oracle.py',
        INDEX, PRODUCER/'preparation.json', OLD_JOURNAL]
    reference_inventory = [dict(path=str(p), sha256=sha(p)) for p in references]
    # The pure output fixtures deliberately have no asserted real database or
    # cloud byte identity. They test input-gate refusal, not output acceptance.
    fixtures = []
    metadata_bindings = []
    for item in preparation['seeds']:
        seed = item['seed']; plan = dict(item['plan'], world_size=4)
        job = dict(seed=seed, task_id='d'*32, plan=plan, recipe_sha256=hashlib.sha256(canonical(plan)).hexdigest())
        old = next(v for v in old_jobs if v['seed'] == seed)
        entry = next(v for v in entries if v['seed'] == seed)
        checkpoint = Path(entry['training_byte_proof']).parent/'checkpoint'
        metadata_bindings.append(validate_final_model(seed, checkpoint, plan=plan))
        forward = json.loads(Path(entry['prediction_byte_proof']).read_bytes())
        events = json.loads((R/f'artifacts/rbf-original-cache-CPU-metadata-export-v1-20261001/seed{seed}/events.json').read_bytes())
        counts = {sid:sum(v['sequence_id']==sid for v in events['events']) for sid in events['origin_us_by_sequence']}
        value = dict(kind='rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1',
            all_registered_bytes_verified=True, all_46_sequences_7445_events_and_final_model_factor_nodes_verified=True,
            full_forest_semantics_or_fresh_state_independently_accepted=False,
            learned_Stage2_complete=False, same_resource_performance_accepted=False, paper_performance_complete=False,
            seed=seed, method='topk', task_id=job['task_id'], recipe_sha256=job['recipe_sha256'], world_size=4,
            NN_atol=1e-4, NN_rtol=1e-4, final_checkpoint_sha256=plan['checkpoint']['sha256'],
            final_model_sha256=plan['final_refit_model_sha256'], input_numeric_index_sha256=plan['numeric_reference_admission_sha256'],
            artifacts={k:{} for k in ('receipt', *(f'replay-rank{i}' for i in range(4)))},
            sequences=[dict(sequence_id=s, events=counts[s], nodes=forward['sequence_counts'][s], database_sha256='a'*64) for s in sorted(counts)],
            total_nodes=entry['rows'])
        for world in (4,8):
            variant, candidate = copy.deepcopy(value), copy.deepcopy(job)
            candidate['plan']['world_size'] = variant['world_size'] = world
            candidate['recipe_sha256'] = variant['recipe_sha256'] = hashlib.sha256(canonical(candidate['plan'])).hexdigest()
            variant['artifacts'] = {k:{} for k in ('receipt', *(f'replay-rank{i}' for i in range(world)))}
            validate_local_output(variant, candidate, item['plan'], old)
            fixtures.append(dict(seed=seed, world_size=world, local_fixture_gate_passed=True, real_output_admission=False))
    item = preparation['seeds'][0]; seed = item['seed']
    # Reconstruct the first seed fixture once for the refusal matrix.
    old = next(v for v in old_jobs if v['seed'] == seed)
    entry = next(v for v in entries if v['seed'] == seed)
    plan = dict(item['plan'],world_size=4)
    job = dict(seed=seed, task_id='d'*32, plan=plan, recipe_sha256=hashlib.sha256(canonical(plan)).hexdigest())
    forward = json.loads(Path(entry['prediction_byte_proof']).read_bytes())
    events = json.loads((R/f'artifacts/rbf-original-cache-CPU-metadata-export-v1-20261001/seed{seed}/events.json').read_bytes())
    counts = {sid:sum(v['sequence_id']==sid for v in events['events']) for sid in events['origin_us_by_sequence']}
    value = dict(kind='rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1',
        all_registered_bytes_verified=True, all_46_sequences_7445_events_and_final_model_factor_nodes_verified=True,
        full_forest_semantics_or_fresh_state_independently_accepted=False,
        learned_Stage2_complete=False, same_resource_performance_accepted=False, paper_performance_complete=False,
        seed=seed, method='topk', task_id=job['task_id'], recipe_sha256=job['recipe_sha256'], world_size=4,
        NN_atol=1e-4, NN_rtol=1e-4, final_checkpoint_sha256=plan['checkpoint']['sha256'],
        final_model_sha256=plan['final_refit_model_sha256'], input_numeric_index_sha256=plan['numeric_reference_admission_sha256'],
        artifacts={k:{} for k in ('receipt', *(f'replay-rank{i}' for i in range(4)))},
        sequences=[dict(sequence_id=s,events=counts[s],nodes=forward['sequence_counts'][s],database_sha256='a'*64) for s in sorted(counts)],
        total_nodes=entry['rows'])
    mutations = {
        'old_nested_admission': lambda v,j:v.update(kind='rbf_coupled_train_replay_output_byte_coverage_logsoftmax_factor_admission_v2'),
        'main_method_input': lambda v,j:v.update(method='rbf'),
        'wrong_seed': lambda v,j:v.update(seed=2027 if seed!=2027 else 1337),
        'wrong_task': lambda v,j:v.update(task_id='e'*32),
        'partial_factor_admission': lambda v,j:v.update(all_46_sequences_7445_events_and_final_model_factor_nodes_verified=False),
        'partial_byte_admission': lambda v,j:v.update(all_registered_bytes_verified=False),
        'wrong_checkpoint': lambda v,j:v.update(final_checkpoint_sha256='b'*64),
        'old_model': lambda v,j:v.update(final_model_sha256=plan['original_nested_model_sha256']),
        'missing_rank_archive': lambda v,j:v['artifacts'].pop('replay-rank0'),
        'foreign_exclusive_manifest': lambda v,j:v['artifacts'].update({'exclusive-source-manifest':{}}),
        'partial_sequence_coverage': lambda v,j:v['sequences'].pop(),
        'wrong_node_total': lambda v,j:v.update(total_nodes=v['total_nodes']+1),
        'unsafe_database_sequence': lambda v,j:v['sequences'][0].update(sequence_id='../outside'),
        'premature_fresh_state_acceptance': lambda v,j:v.update(full_forest_semantics_or_fresh_state_independently_accepted=True),
        'premature_resource_comparison': lambda v,j:v.update(same_resource_performance_accepted=True),
        'premature_paper_metric': lambda v,j:v.update(paper_performance_complete=True),
        'widened_tolerance': lambda v,j:v.update(NN_atol=1e-3),
        'changed_K': lambda v,j:j['plan']['configuration']['state'].update(active_limit=8),
        'changed_work_cap': lambda v,j:j['plan']['configuration']['limits'].update(max_decision_nodes=8192),
        'changed_candidate_history': lambda v,j:j['plan']['configuration'].update(history_features=False),
    }
    rejected = []
    for name, edit in mutations.items():
        bad_value, bad_job = copy.deepcopy(value), copy.deepcopy(job); edit(bad_value,bad_job)
        if bad_job['plan'] != job['plan']:
            bad_job['recipe_sha256'] = bad_value['recipe_sha256'] = hashlib.sha256(canonical(bad_job['plan'])).hexdigest()
        try:
            validate_local_output(bad_value,bad_job,item['plan'],old)
        except (AssertionError,KeyError):
            rejected.append(name)
        else:
            raise AssertionError('mutation admitted: '+name)
    D.mkdir()
    for source in sources:
        shutil.copyfile(source,D/source.name)
    control = dict(kind='rbf_final_refit_fixed_topK_CPU_source_control_v1',
        raw_cache203_and_parent_context_oracle_source_unchanged=True,
        causal_raw_commit_oracle_source_unchanged=True, fresh_branch_state_oracle_source_unchanged=True,
        reference_state_atol=1e-8, reference_state_rtol=1e-8,
        upstream_NN_atol=1e-4, upstream_NN_rtol=1e-4,
        final_model_binding_uses_separate_admitted_refit=True,
        old_nested_forests_or_main_exclusive_acceptance_inherited=False,
        same_resource_comparison_accepted=False, paper_performance_complete=False)
    new(D/'source-control.json',control)
    freeze = dict(kind='rbf_final_refit_fixed_topK_independent_CPU_source_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in D.iterdir()},
        unchanged_references=reference_inventory,
        reference_final_cache_sha256=sha(MAIN_CPU/'final_cache203.py'),
        method='topk', K=4, full_cohort_CPU_admission_started=False,
        full_forest_accepted=False, same_resource_performance_accepted=False,paper_performance_complete=False)
    new(D/'source-freeze.json',freeze)
    receipt = R/'receipts/rbf-final-refit-fixed-topK-full-independent-CPU-preparation-20261004.json'
    new(receipt,dict(kind='rbf_final_refit_fixed_topK_CPU_source_and_metadata_gate_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_freeze_sha256=sha(D/'source-freeze.json'), three_seed_final_model_bindings=metadata_bindings,
        rank_partition_metadata_fixtures=fixtures, rejected_mutations=rejected,
        fixtures_do_not_assert_real_output_bytes_or_numerics=True,
        old_running_TopK_auditors_preserved=True, real_CPU_output_admission_started=False,
        full_forest_accepted=False, same_resource_performance_accepted=False,paper_performance_complete=False,
        next_dependency='completed new TopK task plus independent full-byte/event/factor readback'))
    register(receipt,'rbf_final_refit_fixed_topK_CPU_source_and_metadata_gate_v1')
    print(json.dumps(dict(source_freeze=str(D/'source-freeze.json'),receipt=str(receipt),
        rejected_mutations=len(rejected),fixtures=len(fixtures),real_output_accepted=False)))


if __name__ == '__main__':
    main()
