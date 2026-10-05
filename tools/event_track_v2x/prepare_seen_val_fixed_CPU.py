"""Derive separately scoped K1/K4 val CPU drivers with unchanged numerical oracles."""
import argparse
import ast
import datetime
import json
from pathlib import Path
import re
import shutil

from rbf_nested_seen_val_v2_common import R,new,register,sha
from rbf_seen_val_fixed_CPU_binding import ORIGINALS,READER,READER_SHA,VAL_CPU,VAL_CPU_SHA,MODEL_CPU,MODEL_CPU_SHA

NAME='rbf-seen-val-fixed-K1-K4-full-independent-CPU-v1-20261005'


def replace_exact(parent,edits):
    value=parent
    for before,after in edits:
        assert value.count(before)==1,(before,value.count(before))
        value=value.replace(before,after,1)
    ast.parse(value)
    reversed_value=value
    for before,after in reversed(edits):
        assert reversed_value.count(after)==1,after
        reversed_value=reversed_value.replace(after,before,1)
    assert reversed_value==parent
    return value,[dict(before=before,after=after) for before,after in edits]


def derive_cache():
    assert sha(VAL_CPU/'source-freeze.json')==VAL_CPU_SHA
    path=VAL_CPU/'rbf_seen_val_forest_cache203.py'
    freeze=json.loads((VAL_CPU/'source-freeze.json').read_bytes())
    assert sha(path)==freeze['sources'][path.name]['sha256']
    return replace_exact(path.read_text(),[
        ('from rbf_seen_val_forest_CPU_binding import frozen_modules, registered, SCOPE',
         'from rbf_seen_val_fixed_CPU_binding import frozen_modules, registered, scope'),
        ('model_binding, original, output_binding, _ = frozen_modules()',
         'model_binding, original, output_binding, _ = frozen_modules(job[\'K\'])'),
        ('        self._original = original','        self._original = original\n        self.scope = scope(job[\'K\'])'),
        ('result.update(scope=SCOPE,kind=\'rbf_seen_val_full_runtime_cache203_context_sequence_admission_v1\'',
         'result.update(scope=admission.scope,kind=\'rbf_seen_val_fixed_full_runtime_cache203_context_sequence_admission_v1\''),
    ])


def derive_driver(width):
    root,digest=ORIGINALS[width];assert sha(root/'source-freeze.json')==digest
    lower='top1' if width==1 else 'topk';name='Top1' if width==1 else 'topK'
    path=root/f'accept_final_refit_{lower}_cohort.py'
    freeze=json.loads((root/'source-freeze.json').read_bytes());assert sha(path)==freeze['sources'][path.name]['sha256']
    return replace_exact(path.read_text(),[
        (f'Read-only full fixed {"Top1" if width==1 else "TopK"} causal/cache203/fresh-state output admission.',
         f'Read-only complete SPD seen-val fixed K{width} causal/cache203/fresh-state admission.'),
        (f'from rbf_final_refit_{lower}_binding import R, validate_output','from rbf_seen_val_fixed_CPU_binding import R, validate_output'),
        ('from final_cache203_receipt_v2 import CacheAdmission, verify_database as feature_database',
         'from rbf_seen_val_fixed_cache203 import CacheAdmission, verify_database as feature_database'),
        ('validate_output(value)',f'validate_output(args.byte_admission,width={width})'),
        ("key == 'receipt'","key in ('receipt','seen-val-input-binding','seen-val-baseline-binding')"),
        ("CacheAdmission(value['seed'], checkpoint)","CacheAdmission(value['seed'], checkpoint, job)"),
        (f"scope_stage='final_refit_full_{name}_causal_cache_fresh_state'",f"scope_stage='seen_val_full_fixed_K{width}_causal_cache_fresh_state'"),
        ('total_sequences=46','total_sequences=21'),
        ("assert len(results) == 46 and sum(v['causal']['events'] for v in results) == 7445",
         "assert len(results) == 21 and sum(v['causal']['events'] for v in results) == 3316"),
        (f'from rbf_final_refit_{lower}_binding import source_gate','from rbf_seen_val_fixed_CPU_binding import source_gate'),
        ('assert source_gate() == source_binding and sha(FRESH) == FRESH_SHA',
         f'assert source_gate({width}) == source_binding and sha(FRESH) == FRESH_SHA\n        assert validate_output(args.byte_admission,width={width}) == (job,checkpoint,model_binding,source_binding)'),
        (f"kind='rbf_final_refit_fixed_{name}_full_causal_cache203_fresh_state_admission_v1'",
         f"kind='rbf_seen_val_fixed_K{width}_full_causal_cache203_fresh_state_admission_v1'"),
        ('completed_sequences=46, completed_events=7445','completed_sequences=21, completed_events=3316'),
        ("        new(args.output/'acceptance.json', final);", 
         "        final.update(scope=job['plan']['evaluation_scope'],formal_independent_evaluation=False,measured_network_arrival_history_verified=False,\n            train_or_main_baseline_full_acceptance_inherited=False,sequence_receipts=[dict(path=str(args.output/f'sequence-{i:02d}.json'),sha256=sha(args.output/f'sequence-{i:02d}.json')) for i in range(len(results))])\n        new(args.output/'acceptance.json', final);"),
        (f"kind='rbf_final_refit_fixed_{name}_CPU_admission_failure_v1'",f"kind='rbf_seen_val_fixed_K{width}_CPU_admission_failure_v1'"),
    ])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--software-log',type=Path,required=True);args=parser.parse_args()
    log=args.software_log.read_text();matches=re.findall(r'(\d+) passed',log)
    assert matches and int(matches[-1])==50 and 'FAILED' not in log and 'ERROR' not in log
    out=R/'source-freezes'/NAME;assert not out.exists()
    cache,cache_changes=derive_cache();drivers={w:derive_driver(w) for w in (1,4)}
    own=Path(__file__).resolve().parent
    assert cache==(own/'rbf_seen_val_fixed_cache203.py').read_text()
    for width,(source,_) in drivers.items():assert source==(own/f'accept_seen_val_fixed_K{width}_cohort.py').read_text()
    refs={}
    for root,digest in [*ORIGINALS.values(),(READER,READER_SHA),(VAL_CPU,VAL_CPU_SHA),(MODEL_CPU,MODEL_CPU_SHA)]:
        assert sha(root/'source-freeze.json')==digest;refs[str(root/'source-freeze.json')]=digest
        freeze=json.loads((root/'source-freeze.json').read_bytes())
        refs.update({str(root/name):v['sha256'] if isinstance(v,dict) else v for name,v in freeze['sources'].items()})
        for field in ('references','unchanged_references','unchanged_independent_sources'):
            refs.update({v['path']:v['sha256'] for v in freeze.get(field,[])})
    for path,digest in refs.items():assert sha(path)==digest
    files=[own/name for name in ('rbf_seen_val_fixed_CPU_binding.py','rbf_seen_val_fixed_cache203.py','prepare_seen_val_fixed_CPU.py',
        'accept_seen_val_fixed_K1_cohort.py','accept_seen_val_fixed_K4_cohort.py','rbf_nested_seen_val_v2_common.py')]
    for path in files:ast.parse(path.read_text())
    out.mkdir()
    for path in files:shutil.copyfile(path,out/path.name)
    test=own.parents[1]/'tests/event_track_v2x/test_seen_val_fixed_CPU.py'
    destination=out/'tests/event_track_v2x'/test.name;destination.parent.mkdir(parents=True);shutil.copyfile(test,destination)
    shutil.copyfile(args.software_log,out/'software-tests.log')
    shutil.copyfile(own.parents[1]/'docs/recover-before-fuse/remaining-experiment-code-preparation-20261004.md',out/'remaining-experiments.md')
    new(out/'source-control.json',dict(cache_reversible_edits=cache_changes,driver_reversible_edits={str(w):v[1] for w,v in drivers.items()},
        all_other_driver_and_cache_bytes_unchanged=True,three_numerical_oracles_unchanged=True,atol=1e-8,rtol=1e-8))
    result=dict(kind='rbf_seen_val_fixed_K1_K4_full_CPU_source_v1',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()},
        references=[dict(path=p,sha256=h) for p,h in sorted(refs.items())],software_tests=int(matches[-1]),
        software_scope='scope rejection, unchanged algorithm checks and admitted input adapter compatibility; no real val forest admission',
        all_21_sequences_3316_events_required=True,atol=1e-8,rtol=1e-8,actual_val_baseline_CPU_admitted=False,
        complete_online_method_accepted=False,full_Stage2_complete=False,same_resource_performance_accepted=False,paper_performance_complete=False)
    new(out/'source-freeze.json',result);register(out/'source-freeze.json',result['kind'])
    print(json.dumps(dict(source_freeze=str(out/'source-freeze.json'),sha256=sha(out/'source-freeze.json'),software_tests=int(matches[-1]))),flush=True)


if __name__=='__main__':main()
