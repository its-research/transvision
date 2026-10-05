"""Freeze the tested K1/K4 dispatcher without creating or enqueuing tasks."""
import argparse
import ast
import datetime
import json
from pathlib import Path
import shutil

from rbf_nested_seen_val_v2_common import R,new,register,sha
from submit_seen_val_fixed_baselines import PRODUCERS,PRODUCERS_SHA,PUB,PUB_SHA

NAME = 'rbf-seen-val-fixed-K1-K4-GPU-dispatch-v1-20261005'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--software-log',type=Path,required=True)
    args=parser.parse_args()
    out=R/'source-freezes'/NAME
    assert not out.exists(), 'retain existing freeze; never replace it'
    log=args.software_log.read_text()
    assert '23 passed' in log and 'FAILED' not in log and 'ERROR' not in log
    assert sha(PRODUCERS/'source-freeze.json')==PRODUCERS_SHA
    assert sha(PUB/'source-freeze.json')==PUB_SHA
    producers=json.loads((PRODUCERS/'source-freeze.json').read_bytes())
    prior=json.loads((PUB/'source-freeze.json').read_bytes())
    own=Path(__file__).resolve().parent
    repo=own.parents[1]
    dependencies=('publish_rbf_seen_val_forest_inputs.py','submit_rbf_seen_val_bound_forest.py',
        'rbf_nested_seen_val_v2_common.py','rbf_final_refit_teacher_binding.py',
        'submit_rbf_final_identity.py','submit_rbf_seen_val_joint_identity.py')
    paths={name:PUB/name for name in dependencies}
    for name,path in paths.items():
        assert sha(path)==sha(own/name)==prior['sources'][name]['sha256']
    name='prepare_seen_val_fixed_baselines.py'
    assert sha(own/name)==sha(PRODUCERS/name)==producers['sources'][name]['sha256']
    paths[name]=PRODUCERS/name
    for name in ('submit_seen_val_fixed_baselines.py','prepare_seen_val_fixed_dispatch.py'):
        paths[name]=own/name
    # Preserve the fixture layout so the packaged dispatch tests can be rerun.
    for name in ('test_seen_val_fixed_dispatch.py','test_seen_val_fixed_baselines.py',
                 'test_seen_val_forest_runtime.py','test_final_refit_teacher_dispatch.py'):
        relative='tests/event_track_v2x/'+name
        paths[relative]=repo/relative
    relative='tools/event_track_v2x/prepare_rbf_seen_val_forest_runtime.py'
    paths[relative]=repo/relative
    for path in paths.values():ast.parse(path.read_text())
    refs={str(PRODUCERS/'source-freeze.json'):PRODUCERS_SHA,str(PUB/'source-freeze.json'):PUB_SHA}
    for root,freeze in ((PRODUCERS,producers),(PUB,prior)):
        refs.update({str(root/name):record['sha256'] for name,record in freeze['sources'].items()})
        refs.update({item['path']:item['sha256'] for item in freeze['references']})
    for path,checksum in refs.items():assert sha(path)==checksum
    out.mkdir()
    for relative,path in paths.items():
        destination=out/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
    shutil.copyfile(args.software_log,out/'software-tests.log')
    shutil.copyfile(repo/'docs/recover-before-fuse/remaining-experiment-code-preparation-20261004.md',
                    out/'remaining-experiments.md')
    result=dict(kind='rbf_seen_val_fixed_baseline_GPU_dispatch_source_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={str(path.relative_to(out)):dict(sha256=sha(path),bytes=path.stat().st_size)
                 for path in sorted(out.rglob('*')) if path.is_file()},
        references=[dict(path=path,sha256=checksum) for path,checksum in sorted(refs.items())],
        command=['python',str(Path(__file__).resolve()),'--software-log',str(args.software_log.resolve())],
        software_tests=23,software_scope='mock task lifecycle, exact producer gates and physical GPU reservations',
        frozen_producers_unchanged=True,default_read_only=True,
        creates_durable_intent_before_task_creation=True,unknown_creation_outcome_blocks_retry=True,
        deduplicates_K_and_seed_across_GPU_card_count=True,L40S_CPU_only=True,
        actual_GPU_tasks_created=0,actual_memory_utilization_admitted=False,
        full_seen_val_baseline_admission=False,same_resource_performance_accepted=False,
        full_Stage2_complete=False,paper_performance_complete=False,
        remaining_dependencies=['complete main and matching baseline train CPU admission',
            'independently read published seen-val inputs','baseline-specific val byte and full CPU admission',
            'same-resource measurements and independent performance evaluation'])
    new(out/'source-freeze.json',result)
    register(out/'source-freeze.json',result['kind'])
    print(json.dumps(dict(source_freeze=str(out/'source-freeze.json'),sha256=sha(out/'source-freeze.json'),
                          software_tests=23,GPU_tasks_created=0)),flush=True)


if __name__=='__main__':main()
