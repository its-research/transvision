"""Derive a distinct complete learned-cohort driver without running a replay."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

PARENT = Path('/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-final-refit-full-forest-independent-CPU-v4-receipt-rows-20261004/accept_final_refit_cohort.py')
PARENT_SHA = '06fc3b18417c54169a327e69e2b6d5fafdd800445696d786c0d67b152adb0b2f'


def build(parent):
    assert hashlib.sha256(parent.encode()).hexdigest() == PARENT_SHA
    source = parent
    replacements = [
        ('"""Full new exclusive cohort audit after registered-byte/coverage/factor admission.',
         '"""Full learned exclusive cohort: forest and checkpoint-bound allocation trajectory.'),
        ('from rbf_final_refit_forest_binding import validate, source_gate',
         'from rbf_final_refit_learned_forest_binding import (validate, source_gate, arguments, check_sequence, ordering, ordering_summary, KIND)'),
        (" p=argparse.ArgumentParser();p.add_argument('--byte-admission',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);args=p.parse_args()",
         " p=argparse.ArgumentParser();arguments(p);args=p.parse_args()\n assert args.output.resolve().is_relative_to(ROOT/'artifacts')"),
        ('job=validate(v)', 'job=validate(v,args,Task)'),
        ("key in ('receipt','exclusive-source-manifest')", "key in ('receipt','exclusive-source-manifest','priority-input-binding')"),
        ("   receipt=json.loads(files[0].read_text());info=receipt['databases'][s['sequence_id']];assert info['sha256']==s['database_sha256'];db=files[0].parent/info['path'];assert db.resolve().is_relative_to(files[0].parent.resolve())",
         "   db=check_sequence(files[0].parent,s,job)"),
        (";action=action_database(db,s['database_sha256'],progress)",
         ";action=action_database(db,s['database_sha256'],progress)\n   learned_order=ordering(db,s,args,job,progress)"),
        ("features['events']==s['events'] and causal['observations']==features['rows']==s['nodes']",
         "features['events']==learned_order['events']==s['events'] and causal['observations']==features['rows']==learned_order['observations']==s['nodes']"),
        ('runtime_raw_cache203_context=features)',
         'runtime_raw_cache203_context=features,learned_search_trajectory=learned_order)'),
        ("  final=dict(binding,kind='rbf_final_refit_full_exclusive_cohort_independent_structure_causal_mass_recovery_state_action_scope_acceptance_v1',",
         "  assert source_gate()==final_driver_binding\n  assert sha(args.byte_admission)==binding['byte_admission_sha256']\n  assert validate(v,args,Task)==job\n  priority_summary=ordering_summary(results,job)\n  final=dict(binding,kind=KIND,"),
        ("  (args.output/'acceptance.json').write_text(json.dumps(final,indent=2)+'\\n');",
         "  final.update(priority_summary)\n  (args.output/'acceptance.json').write_text(json.dumps(final,indent=2)+'\\n');"),
    ]
    for before, after in replacements:
        assert source.count(before) == 1, 'full CPU source changed: ' + before
        source = source.replace(before, after, 1)
    ast.parse(source)
    checks = {}
    for name, start, end in [
        ('original_software_gates', " gate=ROOT/'artifacts/", ' final_driver_binding=source_gate();'),
        ('original_independent_oracle_calls', '   features=feature_database(', "\n   assert structure['events']"),
    ]:
        old = parent[parent.index(start):parent.index(end)]
        if name == 'original_independent_oracle_calls':
            # The five existing calls are unchanged; the new call is appended.
            assert old in source
        else:
            assert old in source
        checks[name] = hashlib.sha256(old.encode()).hexdigest()
    assert 'atol=' not in ''.join(after for _, after in replacements)
    return source, dict(parent_driver_sha256=PARENT_SHA, replacements=len(replacements),
        preserved_code_blocks_sha256=checks, original_five_numeric_oracles_unchanged=True,
        NN_atol=1e-4, NN_rtol=1e-4, priority_atol=1e-8, priority_rtol=1e-8,
        actual_cohort_executed=False, paper_performance_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, control = build(PARENT.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'accept_final_refit_learned_cohort.py').write_text(source)
    (args.output / 'source-control.json').write_text(json.dumps(control, indent=2) + '\n')
    print(json.dumps(dict(directory=str(args.output), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                         actual_experiment_accepted=False)))


if __name__ == '__main__':
    main()
