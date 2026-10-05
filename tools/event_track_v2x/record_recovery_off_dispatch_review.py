"""Archive this execution step without rewriting any experiment evidence."""
import datetime
import json
from pathlib import Path
import shutil
import subprocess

from rbf_nested_seen_val_v2_common import R, new, register, sha


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = R/'artifacts'/('rbf-recovery-off-dispatch-and-main-readback-review-'+stamp)
    out.mkdir()
    archived = []
    sources = [Path(__file__), *[Path('tools/event_track_v2x')/name for name in (
        'qualify_recovery_off_final_inputs.py','prepare_recovery_off_gpu_replay.py',
        'qualify_recovery_off_gpu_source.py','recovery_off_dispatch_gate.py','observe_recovery_off_gpu.py',
        'continue_final_forest_seed1337.py')],
        Path('tests/event_track_v2x/test_recovery_off_dispatch_gate.py'),
        Path('tests/event_track_v2x/test_final_forest_seed1337_continuation.py')]
    names = ['rbf-recovery-off-real-final-input-contract-v1-20261005.log',
        'rbf-recovery-off-dispatch-gate-tests-20261005.log','rbf-recovery-off-GPU-prepare-v1-20261005.log',
        'rbf-recovery-off-GPU-source-qualification-v1-20261005.log',
        'rbf-recovery-off-GPU-dry-dispatch-v1-20261005.log','rbf-recovery-off-GPU-dispatch-v1-20261005.log',
        'rbf-recovery-off-GPU-live-20261005.log','rbf-final-forest-seed1337-controller-tests-20261005.log',
        'rbf-final-forest-seed1337-frozen-controller-tests-20261005.log',
        'rbf-final-forest-seed3407-controller-tests-20261005.log',
        'rbf-final-forest-seed3407-controller-tests-v2-20261005.log',
        'rbf-final-forest-seed3407-controller-tests-v3-20261005.log',
        'rbf-final-forest-seed3407-frozen-controller-tests-20261005.log',
        'rbf-final-forest-seed3407-controller-literal-edits-20261005.json',
        'rbf-final-forest-seed1337-controller-literal-edits-20261005.json',
        'rbf-allowlisted-live-recovery-dispatch-20261005.log']
    sources.extend(Path('/private/tmp')/name for name in names)
    for p in sources:
        assert p.is_file(), str(p)
        target = out/p.name
        assert not target.exists()
        shutil.copyfile(p,target)
        archived.append(dict(original=str(p.resolve()),path=str(target),sha256=sha(target),bytes=target.stat().st_size))
    process = subprocess.run(['ps','-p','2867,4246,13369,14229,3256,69201,92290,24126,42941,73876',
                              '-o','pid,ppid,etime,%cpu,command'],capture_output=True,text=True,timeout=10,check=True)
    (out/'processes.txt').write_text(process.stdout)
    status = subprocess.run(['git','status','--short'],capture_output=True,text=True,check=True)
    (out/'git-status.txt').write_text(status.stdout)
    references = [R/'source-freezes/rbf-final-refit-recovery-off-bound-GPU-v1-20261005/preparation.json',
        R/'source-freezes/rbf-final-refit-recovery-off-bound-GPU-v1-20261005/source-qualification.json',
        R/'artifacts/rbf-recovery-off-final-checkpoint-input-contract-v1-20261005/acceptance.json',
        R/'source-freezes/rbf-final-forest-seed1337-after-prefetch-continuation-v1-20261005/source-freeze.json',
        R/'source-freezes/rbf-final-forest-seed3407-after-prefetch-continuation-v1-20261005/source-freeze.json',
        R/'receipts/rbf-final-refit-all-baselines-allowlisted-live-20261005T005300239118Z.json',
        R/'receipts/rbf-final-refit-recovery-off-bound-GPU-dispatch-20261005.json']
    live_log = Path('/private/tmp/rbf-recovery-off-GPU-live-20261005.log').read_text().splitlines()
    live = Path(json.loads(live_log[-1])['receipt'])
    references.append(live)
    tasks = json.loads(live.read_bytes())['tasks']
    assert {t['seed'] for t in tasks} == {1337,2027,3407}
    cpu_paths = [('Top1',2027,'rbf-final-refit-Top1-seed2027-full-independent-CPU-v2-20261004'),
        ('Top1',1337,'rbf-final-top1-seed1337-after-readback-continuation-v1-20261005/independent-acceptance'),
        ('Top1',3407,'rbf-final-top1-seed3407-after-readback-continuation-v1-20261005/independent-acceptance'),
        ('K4',1337,'rbf-final-refit-K4-seed1337-CPU-receipt-rows-v2-20261004'),
        ('K4',2027,'rbf-final-topk-after-readback-continuation-v3-20261004/seed2027/independent-acceptance'),
        ('K4',3407,'rbf-final-topk-seed3407-after-readback-continuation-v1-20261005/independent-acceptance')]
    cpu = []
    for method,seed,name in cpu_paths:
        directory=R/'artifacts'/name; rows=[]
        for path in sorted(directory.glob('sequence-*.json')):
            value=json.loads(path.read_bytes())
            assert value['runtime_raw_cache203_context']['complete_original_sequence'] is True
            assert value['runtime_raw_cache203_context']['all_203_features_and_world_state_values_bound_to_raw_cache'] is True
            assert value['causal']['all_admitted_information_and_arrivals_before_decision'] is True
            assert value['fresh_state']['stored_states_and_chosen_outputs_checked'] is True
            rows.append(dict(path=str(path),sha256=sha(path),sequence_id=value['sequence_id']))
        assert len({x['sequence_id'] for x in rows}) == len(rows)
        cpu.append(dict(method=method,seed=seed,sequence_receipts=len(rows),total_sequences=46,
                        evidence=rows,whole_CPU_ETA='unknown',full_cohort_acceptance_not_inferred=True))
    downloads=[]
    for seed in (1337,3407):
        paths=list((R/'receipts').glob(f'rbf-final-refit-rank-byte-prefetch-seed{seed}-*-started.json'))
        assert len(paths)==1
        start=json.loads(paths[0].read_bytes()); references.append(paths[0])
        current={p.name:p.stat().st_size for p in Path(start['cache_directory']).glob('replay-rank*.tar.gz*')}
        downloads.append(dict(seed=seed,pid=start['pid'],task_id=start['task_id'],
            current_bytes=sum(current.values()),total_registered_bytes=sum(a['bytes'] for a in start['registered_snapshot'].values()),
            files=current,downloaded_bytes_not_yet_accepted=True,whole_experiment_ETA='unknown'))
    value=dict(kind='rbf_recovery_off_dispatch_and_main_readback_progress_v1',checked_at_utc=stamp,
        previous_goal_turn_classification='progress: main3407 readback chain and actual final checkpoint input qualification',
        current_goal_turn_classification='progress',new_recovery_off_tasks=tasks,
        main_GPU_tasks_completed=3,main_GPU_tasks_full_independent_acceptance_complete=0,
        main2027_rank5_corruption_unresolved=True,original_reads=downloads,ongoing_CPU=cpu,
        references=[dict(path=str(p),sha256=sha(p)) for p in references],archived_sources_and_logs=archived,
        process_snapshot=dict(path=str(out/'processes.txt'),sha256=sha(out/'processes.txt')),
        git_snapshot=dict(path=str(out/'git-status.txt'),sha256=sha(out/'git-status.txt')),
        original_running_jobs_preserved=True,software_checks_are_not_full_experiments=True,
        numerical_tolerances_unchanged=True,GPU_memory_75_to_80_percent_measured=False,
        V2V4Real_formal_independence_gate_open=False,whole_experiment_ETA='unknown',
        no_git_commit_or_push=True,goal_status='active',paper_performance_complete=False)
    path=R/'receipts'/('rbf-recovery-off-dispatch-and-main-readback-review-'+stamp+'.json')
    new(path,value);register(path,value['kind']);print(json.dumps(dict(receipt=str(path),sha256=sha(path),
        cpu=[{k:v for k,v in x.items() if k!='evidence'} for x in cpu],downloads=downloads)))


if __name__ == '__main__': main()
