"""Independently verify every actual recovery-off output after byte admission.

Fresh/raw/causal routines are original frozen functions. Restricted support and
restricted action semantics require separately qualified independent oracles.
No acceptance of the original main cohort is inherited or required.
"""
import argparse
import datetime
import json
import sqlite3
import sys
import time
from pathlib import Path

from rbf_nested_seen_val_v2_common import new, register, sha
from recovery_off_final_binding import load_unchanged_oracles, source_gate, validate


def assert_database_binding(path, expected_sha, sequence, configuration):
    assert sha(path) == expected_sha
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        assert db.execute('PRAGMA integrity_check').fetchone() == ('ok',)
        metadata = {k: json.loads(v) for k, v in db.execute('SELECT k,v FROM meta')}
        # SQLite config is dataclass state+limits, unlike outer runtime plan.
        assert metadata['sequence_id'] == sequence
        config = metadata['config']
        assert config['state'] == configuration['state']
        assert {k: v for k, v in config.items() if k != 'state'} == configuration['limits']
        for payload, in db.execute('SELECT audit FROM events ORDER BY ordinal'):
            audit = json.loads(payload)
            assert audit['kind'] == 'persistent_exclusive_event_boundary_recovery_off_v1'
            assert audit['recovery_enabled'] is False
            assert audit['original_model_risk_certified'] is False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--byte-admission', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    software = source_gate()
    value = json.loads(args.byte_admission.read_bytes())
    job, model_binding = validate(value, args.checkpoint)
    model, cache, causal, fresh = load_unchanged_oracles()
    # Local modules have already been hash-checked by source_gate; they must not
    # be replaced by producer methods or an earlier unrestricted oracle.
    from recovery_off_structure_oracle import verify_database as structure_database
    from recovery_off_action_oracle import verify_database as action_database
    for name in ('recovery_off_structure_oracle', 'recovery_off_action_oracle'):
        assert Path(sys.modules[name].__file__).resolve() == Path(__file__).resolve().parent/(name+'.py')
    for key, record in value['artifacts'].items():
        suffix = '.json' if key in ('receipt', 'exclusive-source-manifest') else '.tar.gz'
        path = args.byte_admission.parent/(key+suffix)
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
    admission = cache.CacheAdmission(value['seed'], args.checkpoint)
    assert admission.final_model_binding == {k: v for k, v in model_binding.items() if k in admission.final_model_binding}
    binding = dict(kind='rbf_recovery_off_full_independent_binding_v1', seed=value['seed'], task_id=value['task_id'],
                   recipe_sha256=value['recipe_sha256'], byte_admission_sha256=sha(args.byte_admission),
                   checkpoint_sha256=sha(args.checkpoint), software=software, model_binding=model_binding,
                   original_experiment_acceptance_inherited=False, learned_Stage2_complete=False,
                   same_resource_performance_accepted=False, paper_performance_complete=False,
                   complete_online_method_accepted=False, formal_interval_certificate=False,
                   full_original_model_posterior_guarantee=False, atol=1e-8, rtol=1e-8)
    assert not args.output.exists() and not any(p.is_symlink() for p in (args.output, *args.output.parents))
    args.output.mkdir(parents=True)
    new(args.output/'binding.json', binding)
    started, results = time.monotonic(), []
    try:
        for index, row in enumerate(value['sequences']):
            files = list(args.byte_admission.parent.glob(f"rank*-unpack/rank-*/{row['sequence_id']}/receipt.json"))
            assert len(files) == 1
            receipt = json.loads(files[0].read_bytes())
            item = receipt['databases'][row['sequence_id']]
            assert item['sha256'] == row['database_sha256']
            relative = Path(item['path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            database = files[0].parent/relative
            assert not database.is_symlink() and database.resolve().is_relative_to(files[0].parent.resolve())
            assert_database_binding(database, row['database_sha256'], row['sequence_id'], job['plan']['configuration'])
            def progress(data):
                payload = dict(data)
                payload.update(scope_stage='full_independent_recovery_off_scope', completed_sequences=len(results), total_sequences=46,
                               ETA_seconds=None, ETA_reason='heterogeneous component and raw-history cost; no whole-cohort timing model')
                print(json.dumps(payload), flush=True)
            features = cache.verify_database(database, row['database_sha256'], admission, progress=progress)
            commits = causal.verify_database(database, row['database_sha256'])
            structure = structure_database(database, row['database_sha256'], progress=progress)
            state = fresh.verify_database(database, row['database_sha256'], progress=progress)
            action = action_database(database, row['database_sha256'], progress=progress)
            assert structure['independent_restricted_support_coverage_verified'] is True
            assert action['restricted_action_search_or_capacity_fallback_verified'] is True
            assert features['events'] == commits['events'] == structure['events'] == state['events'] == action['events'] == row['events']
            assert features['rows'] == commits['observations'] == row['nodes']
            assert features['complete_original_sequence'] is True
            assert sha(database) == row['database_sha256']
            result = dict(sequence_id=row['sequence_id'], database_sha256=row['database_sha256'],
                          runtime_raw_cache203_context=features, causal=commits, restricted_structure=structure,
                          fresh_state=state, restricted_action_search=action)
            results.append(result)
            new(args.output/f'sequence-{index:02d}.json', result)
            progress(dict(stage='full_independent_recovery_off_sequence_complete', completed_sequences=index+1))
        assert len(results) == 46 and sum(v['causal']['events'] for v in results) == 7445
        assert sum(v['causal']['observations'] for v in results) == value['total_nodes']
        assert source_gate() == software
        final = dict(binding, kind='rbf_recovery_off_full_train_independent_structure_causal_state_action_scope_acceptance_v1',
                     completed_sequences=46, completed_events=7445, completed_observations=value['total_nodes'],
                     independent_restricted_support_coverage_verified=True,
                     restricted_action_search_or_capacity_fallback_verified=True,
                     all_203_feature_recipe_values_independently_verified=True,
                     all_scorer_contexts_bound_to_original_causal_raw_history=True,
                     fresh_conditional_states=sum(v['fresh_state']['states'] for v in results),
                     fresh_predictions=sum(v['fresh_state']['predictions'] for v in results),
                     max_fresh_state_abs_error=max(v['fresh_state']['max_abs_error'] for v in results),
                     elapsed_seconds=time.monotonic()-started,
                     checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        new(args.output/'acceptance.json', final)
        register(args.output/'acceptance.json', final['kind'])
        print(json.dumps(final), flush=True)
    except BaseException as error:
        new(args.output/'failure.json', dict(binding, type=type(error).__name__, message=str(error),
                                           completed_sequences=len(results), experiment_accepted=False))
        raise


if __name__ == '__main__':
    main()
