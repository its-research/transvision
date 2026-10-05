"""Read bounded progress without restarting or revalidating a live experiment."""
import datetime
import json
from pathlib import Path
import subprocess

from rbf_nested_seen_val_v2_common import R, new, register, sha


def main():
    previous = R / 'receipts/rbf-preserved-legacy-CPU-process-and-proof-progress-20261004T000258176076Z.json'
    prior = json.loads(previous.read_bytes())
    pids = [entry['pid'] for entry in prior['records']]
    command = ['ps', '-p', ','.join(map(str, pids)), '-o', 'pid=,stat=,%cpu=,etime=']
    observed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert observed.returncode in (0, 1), observed.stderr
    processes = []
    for line in observed.stdout.splitlines():
        pid, stat, cpu, elapsed = line.split()
        processes.append(dict(pid=int(pid), stat=stat, cpu_percent=float(cpu), elapsed=elapsed))
    alive = {entry['pid'] for entry in processes}
    records = []
    for entry in prior['records']:
        directory = Path(entry['directory'])
        assert directory.is_relative_to(R / 'artifacts')
        completed = []
        events = 0
        for path in sorted(directory.glob('sequence-*.json')):
            raw = path.read_bytes()
            proof = json.loads(raw)
            if 'causal' in proof:
                events += int(proof['causal']['events'])
            else:
                assert proof['stored_states_and_chosen_outputs_checked'] is True
                events += int(proof['events'])
            completed.append(dict(path=str(path), sha256=sha(path)))
        records.append(dict(seed=entry['seed'], pid=entry['pid'], pid_observed_alive=entry['pid'] in alive,
            directory=str(directory), completed_sequences=len(completed), completed_events=events,
            prior_completed_sequences=entry['completed_sequences'], prior_completed_events=entry['completed_events'],
            completed_sequence_proofs=completed, whole_cohort_acceptance_exists=(directory / 'acceptance.json').exists(),
            failure_exists=(directory / 'failure.json').exists(), ETA='unknown'))
    stamp = datetime.datetime.now(datetime.timezone.utc)
    root = R / 'source-freezes/rbf-preserved-legacy-CPU-bounded-observer-v1-20261004'
    root.mkdir(exist_ok=True)
    source = Path(__file__).resolve()
    for original in (source, source.parent / 'rbf_nested_seen_val_v2_common.py'):
        target = root / original.name
        if target.exists():
            assert sha(target) == sha(original), 'source freeze changed'
        else:
            with target.open('xb') as output:
                output.write(original.read_bytes())
    freeze = root / 'source-freeze.json'
    if not freeze.exists():
        new(freeze, dict(kind='rbf_preserved_legacy_CPU_bounded_observer_source_v1',
            sources={path.name: dict(bytes=path.stat().st_size, sha256=sha(path))
                     for path in root.glob('*.py')}, experiment_restarted=False, numerical_validation_repeated=False))
        register(freeze, 'preserved-legacy-CPU-observer-source')
    receipt = R / 'receipts' / ('rbf-preserved-legacy-CPU-process-and-proof-progress-' + stamp.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    new(receipt, dict(kind='rbf_preserved_legacy_CPU_process_and_completed_proof_progress_v1',
        checked_at_utc=stamp.isoformat(), observation_command=command, process_observation_exit_code=observed.returncode,
        observer_source=str(root / source.name), observer_source_sha256=sha(root / source.name),
        previous_progress_receipt=str(previous), previous_progress_receipt_sha256=sha(previous),
        processes=processes, records=records, original_nested_weights_only=True,
        experiment_restarted=False, numerical_validation_repeated=False, full_experiment_accepted=False,
        full_Stage2_complete=False, paper_performance_complete=False, goal_status='active'))
    register(receipt, 'preserved-legacy-CPU-bounded-progress')
    print(json.dumps(dict(receipt=str(receipt), sha256=sha(receipt),
        records=[{key: value for key, value in entry.items() if key != 'completed_sequence_proofs'} for entry in records]), ensure_ascii=False))


if __name__ == '__main__':
    main()
