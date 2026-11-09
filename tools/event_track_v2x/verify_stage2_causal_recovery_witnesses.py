#!/usr/bin/env python3
"""Verify current-output recovery against immutable SQL history, independently."""
import argparse
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--admission', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    assert sha(Path(__file__)) == plan['witness_auditor_sha256']
    helper = Path(plan['full_auditor'])
    assert sha(helper) == plan['full_auditor_sha256']
    spec = importlib.util.spec_from_file_location('independent_sql_helpers', helper)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert sha(args.admission) == plan['full_admission_sha256']
    admission = json.loads(args.admission.read_text())
    assert admission['all_new_runs_full_events_independently_verified']
    records = []
    for result in admission['checks']:
        candidate = Path(plan['candidate_paths'][result['run']])
        assert sha(candidate) == result['candidate_sha256']
        data = json.loads(candidate.read_text())
        dbpath = Path(data['database_path'])
        assert sha(dbpath) == result['database_sha256']
        names = [r['node']['node_id'] for r in data['raw']]
        witnesses = []
        with sqlite3.connect(dbpath.as_uri() + '?mode=ro', uri=True) as db:
            for i, event in enumerate(data['events']):
                expected_count = event['available_observations']
                before_count = 0 if i == 0 else data['events'][i - 1]['available_observations']
                # Ingestion uses only arrived observations and their fixed rows;
                # the final closed DB cannot make future rows part of an old event.
                assert event['audit']['appended_rows'] == data['rows'][before_count:expected_count]
                stamp = event['prediction']['decision_timestamp_us']
                assert all(r['node']['arrival_us'] <= stamp for r in data['raw'][:expected_count])
                if i:
                    old_stamp = data['events'][i - 1]['prediction']['decision_timestamp_us']
                    assert stamp >= old_stamp
                    assert all(r['node']['arrival_us'] >= old_stamp for r in data['raw'][before_count:expected_count])
                if not i or result['events'][i]['chosen_roots'][:3] != ['a', 'b', 'b']:
                    continue
                if result['events'][i - 1]['chosen_roots'][2] == 'b':
                    continue  # only a change in the old ambiguous identity qualifies
                status = result['initial_target_status']
                if status['initially_explicit'] or not status['initially_frontier_covered']:
                    continue
                for comp in event['audit']['components']:
                    cid = comp['component']
                    members = [v[0] for v in db.execute('SELECT global_i FROM component_members WHERE component=? ORDER BY local_i', (cid,))][:comp['nodes']]
                    current_roots = mod.roots(mod.prefix(db, cid, comp['output_handle']), [names[k] for k in members])
                    assert comp['output_handle'] in {a['handle'] for a in comp['active']}
                    for rec in comp['recovery_events']:
                        if not rec['previously_unexpanded'] or rec['projection_nodes'] < 3:
                            continue
                        assert rec['current_component'] == cid and rec['output_handle'] == comp['output_handle']
                        # These predeclared graphs grow a single connected
                        # component without merges; check that fact, don't guess.
                        assert rec['previous_component'] == cid and not comp['merge_restart']
                        old = next(c for c in data['events'][i - 1]['audit']['components'] if c['component'] == cid)
                        h = rec['predecessor_handle']
                        assert h is not None and rec['projection_nodes'] == old['nodes']
                        assert h not in {a['handle'] for a in old['active']} and h != old['output_handle']
                        old_roots = mod.roots(mod.prefix(db, cid, h), [names[k] for k in members[:old['nodes']]])
                        assert old_roots == current_roots[:old['nodes']]
                        assert old_roots[:3] == ('a', 'b', 'b')
                        assert db.execute(f'SELECT sha FROM pc{int(cid)}_prefixes WHERE h=?', (h,)).fetchone()[0] == rec['projection_sha256']
                        discovered = db.execute(f'SELECT decision_us FROM pc{int(cid)}_discovered WHERE h=?', (h,)).fetchone()
                        assert discovered is None or discovered[0] > old_stamp
                        assert not rec['previously_discovered']
                        witnesses.append(dict(event_id=event['event_id'], previous_event_id=data['events'][i - 1]['event_id'],
                            initial_target_only_frontier_covered=True, predecessor_not_active_or_output=True,
                            predecessor_discovery_after_previous_decision=True, selected_current_class_is_active=True,
                            previous_c_root=result['events'][i - 1]['chosen_roots'][2], current_c_root='b',
                            exact_regret_before=result['events'][i - 1]['exact_same_action_model_regret'],
                            exact_regret_after=result['events'][i]['exact_same_action_model_regret'],
                            source_commit_and_past_prediction_immutable=True, metadata=rec))
        assert sha(dbpath) == result['database_sha256']
        assert bool(witnesses) == result['recovery_claimed']
        records.append(dict(run=result['run'], witnesses=witnesses, accepted_recovery=bool(witnesses),
            last_model_regret=result['events'][-1]['exact_same_action_model_regret'],
            last_truth_loss=result['events'][-1]['evaluator_only_identity_loss'],
            outcome='current decision recovery' if witnesses else 'no recovery within fixed predeclared budget',
            budget_changed_after_failure=False, finite_synthetic_case_only=True))
    receipt = dict(kind='rbf_stage2_causal_recovery_SQL_witness_readback_v1', runs=records,
        all_witness_metadata_independently_verified=True, accepted_recovery_runs=sum(r['accepted_recovery'] for r in records),
        no_recovery_runs=sum(not r['accepted_recovery'] for r in records), admission_sha256=sha(args.admission),
        source_sha256=sha(Path(__file__)), production_modules_imported=False, learned_priority_executed=False,
        same_resource_comparison_complete=False, full_Stage2_complete=False, goal_status='active')
    with args.output.open('xb') as f:
        f.write(mod.canonical(receipt) + b'\n')
    print(json.dumps({k: receipt[k] for k in ['accepted_recovery_runs', 'no_recovery_runs', 'full_Stage2_complete']}))


if __name__ == '__main__':
    main()
