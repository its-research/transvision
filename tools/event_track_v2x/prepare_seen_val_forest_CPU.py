"""Derive a separately scoped seen-val driver while retaining all full oracles."""
import argparse
import ast
import datetime
import json
from pathlib import Path
import re
import shutil

from rbf_nested_seen_val_v2_common import R, new, register, sha
from rbf_seen_val_forest_CPU_binding import ORIGINAL, ORIGINAL_SHA, READER, READER_SHA, KIND

NAME = 'rbf-seen-val-bound-forest-full-independent-CPU-v1-20261005'


def derive_driver():
    assert sha(ORIGINAL/'source-freeze.json') == ORIGINAL_SHA
    freeze = json.loads((ORIGINAL/'source-freeze.json').read_bytes())
    path = ORIGINAL/'accept_final_refit_cohort.py'
    assert sha(path) == freeze['sources'][path.name]
    source = path.read_text()
    edits = [
        ('Whole original supplied train schedule only.', 'Whole original SPD seen-val schedule, exploratory scheduled snapshots only.'),
        ('from final_cache203 import CacheAdmission,verify_database as feature_database',
         'from rbf_seen_val_forest_cache203 import CacheAdmission,verify_database as feature_database'),
        ('from rbf_final_refit_forest_binding import validate, source_gate',
         'from rbf_seen_val_forest_CPU_binding import validate, source_gate'),
        ('job=validate(v)', 'job=validate(args.byte_admission)'),
        ("CacheAdmission(v['seed'],args.checkpoint)", "CacheAdmission(v['seed'],args.checkpoint,job)"),
        ("key in ('receipt','exclusive-source-manifest')", "key in ('receipt','exclusive-source-manifest','seen-val-input-binding')"),
        ('total_sequences=46', 'total_sequences=21'),
        ('assert len(results)==46 and sum(s[\'structure\'][\'events\'] for s in results)==7445',
         'assert len(results)==21 and sum(s[\'structure\'][\'events\'] for s in results)==3316'),
        ("kind='rbf_final_refit_full_exclusive_cohort_independent_structure_causal_mass_recovery_state_action_scope_acceptance_v1',completed_sequences=46,completed_events=7445",
         "kind='"+KIND+"',completed_sequences=21,completed_events=3316"),
        ("  (args.output/'acceptance.json').write_text",
         "  assert validate(args.byte_admission)==job\n  assert source_gate()==final_driver_binding\n  final.update(sequence_receipts=[dict(path=str(args.output/f'sequence-{i:02d}.json'),sha256=sha(args.output/f'sequence-{i:02d}.json')) for i in range(len(results))],formal_independent_evaluation=False,measured_network_arrival_history_verified=False)\n  (args.output/'acceptance.json').write_text"),
    ]
    changes = []
    for before, after in edits:
        count = source.count(before)
        assert count == (2 if before == 'total_sequences=46' else 1), (before,count)
        source = source.replace(before,after)
        changes.append(dict(before=before,after=after,count=count))
    ast.parse(source)
    # Stronger than a hand-maintained assurance: every other byte is identical.
    reversed_source = source
    for change in reversed(changes):
        assert reversed_source.count(change['after']) == change['count']
        reversed_source = reversed_source.replace(change['after'],change['before'])
    assert reversed_source == path.read_text()
    return source, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--software-log',type=Path,required=True)
    args = parser.parse_args()
    out = R/'source-freezes'/NAME
    assert not out.exists()
    source, changes = derive_driver()
    log = args.software_log.read_text()
    matches = re.findall(r'(\d+) passed',log)
    assert matches and int(matches[-1]) == 43 and 'FAILED' not in log and 'ERROR' not in log
    assert sha(READER/'source-freeze.json') == READER_SHA
    original = json.loads((ORIGINAL/'source-freeze.json').read_bytes())
    reader = json.loads((READER/'source-freeze.json').read_bytes())
    refs = {str(ORIGINAL/name):digest for name,digest in original['sources'].items()}
    refs.update({item['path']:item['sha256'] for item in original['unchanged_independent_sources']})
    refs.update({str(READER/name):item['sha256'] for name,item in reader['sources'].items()})
    refs.update({item['path']:item['sha256'] for item in reader['references']})
    refs.update({str(ORIGINAL/'source-freeze.json'):ORIGINAL_SHA,str(READER/'source-freeze.json'):READER_SHA})
    for path,digest in refs.items(): assert sha(path) == digest
    own = Path(__file__).resolve().parent
    files = [own/name for name in ('rbf_seen_val_forest_CPU_binding.py','rbf_seen_val_forest_cache203.py',
        'prepare_seen_val_forest_CPU.py','rbf_nested_seen_val_v2_common.py')]
    files += [own.parents[1]/'tests/event_track_v2x/test_seen_val_forest_CPU.py',ORIGINAL/'decoder_capacity.py']
    assert sha(own/'rbf_nested_seen_val_v2_common.py') == original['sources']['rbf_nested_seen_val_v2_common.py']
    for path in files: ast.parse(path.read_text())
    out.mkdir()
    for path in files: shutil.copyfile(path,out/path.name)
    (out/'accept_seen_val_cohort.py').write_text(source)
    new(out/'source-control.json',dict(original_driver=str(ORIGINAL/'accept_final_refit_cohort.py'),
        original_driver_sha256=original['sources']['accept_final_refit_cohort.py'],exact_reversible_edits=changes,
        all_other_driver_bytes_identical=True,raw_feature_and_context_math_called_from_original_frozen_module=True,
        structure_causality_fresh_state_mass_action_oracles_unchanged=True,atol=1e-8,rtol=1e-8))
    shutil.copyfile(args.software_log,out/'software-tests.log')
    result = dict(kind='rbf_seen_val_bound_forest_full_independent_CPU_source_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.iterdir())},
        references=[dict(path=p,sha256=h) for p,h in sorted(refs.items())],
        software_tests=int(matches[-1]),scope='software and admitted-input interface only; no real seen-val forest executed',
        all_21_sequences_3316_events_required=True,atol=1e-8,rtol=1e-8,
        actual_seen_val_forest_admitted=False,complete_online_method_accepted=False,
        measured_network_arrival_history_verified=False,learned_Stage2_complete=False,
        same_resource_performance_accepted=False,paper_performance_complete=False)
    new(out/'source-freeze.json',result); register(out/'source-freeze.json',result['kind'])
    print(json.dumps(dict(source_freeze=str(out/'source-freeze.json'),sha256=sha(out/'source-freeze.json'),software_tests=result['software_tests'])),flush=True)


if __name__ == '__main__': main()
