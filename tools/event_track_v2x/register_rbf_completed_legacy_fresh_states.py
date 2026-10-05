"""Index completed legacy-model state gates without rerunning numerical work."""
import datetime
import hashlib
import json
import os
from pathlib import Path

from rbf_nested_seen_val_v2_common import R, new, register, sha

INDEX = R/'receipts/rbf-legacy-Top1-K4-completed-fresh-state-independent-index-20261004.json'
FINAL = R/'source-freezes/rbf-final-refit-full-train-fixed-Top1-GPU-v1-20261004/preparation.json'


def main():
    assert not INDEX.exists(), 'preserve completed index; inspect matching hashes instead'
    os.environ['CLEARML_FILES_HOST'] = 'http://10.100.34.118:8081'
    from clearml import Task
    records = []
    for width in (1, 4):
        journal = R/('receipts/rbf-original-joint-fixed-top1-GPU4-v1-dispatch-20261002.json' if width == 1
            else 'receipts/rbf-original-joint-coupled-topk-GPU4-v5-native34-dispatch-20261001.json')
        jobs = json.loads(journal.read_bytes())['jobs']
        for seed in (1337, 2027):
            name = (f'rbf-fixed-top1-seed{seed}-full-independent-fresh-branch-state-v1-20261002' if width == 1
                else f'rbf-topk-seed{seed}-full-independent-fresh-branch-state-resume-v2-20261002')
            directory = R/'artifacts'/name
            acceptance_path = directory/'acceptance.json'
            value = json.loads(acceptance_path.read_bytes())
            binding_path = directory/'binding.json'
            binding = json.loads(binding_path.read_bytes())
            assert all(value[k] == v for k, v in binding.items())
            assert value['kind'] == 'rbf_independent_fresh_branch_state_full_cohort_acceptance'
            assert value['stored_component_states_and_all_chosen_outputs_numerically_accepted'] is True
            assert value['atol'] == value['rtol'] == 1e-8
            assert value['completed_sequences'] == 46 and value['completed_events'] == 7445
            assert value['identity_search_and_partition_bounds_accepted'] is value['same_resource_performance_accepted'] is value['paper_performance_complete'] is False
            assert not (directory/'failure.json').exists()
            source = R/('source-freezes/rbf-independent-fresh-branch-state-v3-normalized-admission-20261001' if width == 1
                else 'source-freezes/rbf-independent-fresh-branch-state-v4-interruption-resume-20261002')
            for filename, digest in value['sources'].items():
                assert sha(source/filename) == digest
            job = next(v for v in jobs if v['task_id'] == value['task_id'])
            assert job['seed'] == value['seed'] == seed
            assert job['plan']['configuration']['state']['active_limit'] == width
            final = next(v['plan'] for v in json.loads(FINAL.read_bytes())['seeds'] if v['seed'] == seed)
            assert job['plan']['checkpoint']['sha256'] != final['checkpoint']['sha256']
            byte_root = R/('artifacts/rbf-original-joint-fixed-top1-GPU4-v1-readback-20261002/topk' if width == 1
                else 'artifacts/rbf-original-joint-coupled-topk-GPU4-v5-native34-readback-20261001/topk')/f'seed{seed}'
            byte_path = byte_root/'independent-byte-coverage-factor-admission.json'
            assert sha(byte_path) == value['byte_admission_sha256']
            byte = json.loads(byte_path.read_bytes())
            assert byte['task_id'] == value['task_id'] and byte['seed'] == seed and byte['method'] == 'topk'
            assert all(byte[k] is True for k in ('all_registered_bytes_verified',
                'all_7445_event_commits_and_46_sequences_verified',
                'all_normalized_factor_nodes_verified_against_original_independently_admitted_forward'))
            task = Task.get_task(task_id=value['task_id'])
            assert str(task.status) == 'completed'
            assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == job['plan']['bootstrap_sha256']
            assert set(task.artifacts) == set(byte['artifacts'])
            for key, spec in byte['artifacts'].items():
                assert task.artifacts[key].hash == spec['sha256'] and task.artifacts[key].size == spec['bytes']
            paths = sorted(directory.glob('sequence-*.json'))
            assert [p.name for p in paths] == [f'sequence-{i:02d}.json' for i in range(46)]
            proofs = []
            for path, entry in zip(paths, byte['sequences']):
                proof = json.loads(path.read_bytes())
                assert proof['sequence'] == entry['sequence_id'] and proof['database_sha256'] == entry['database_sha256']
                assert proof['events'] == entry['events'] and proof['stored_states_and_chosen_outputs_checked'] is True
                assert proof['atol'] == proof['rtol'] == 1e-8
                proofs.append(dict(path=str(path), sha256=sha(path), **proof))
            assert sum(p['events'] for p in proofs) == 7445
            assert sum(p['states'] for p in proofs) == value['states']
            assert sum(p['predictions'] for p in proofs) == value['predictions']
            assert max(p['max_abs_error'] for p in proofs) == value['max_abs_error']
            resume = None
            if width == 4:
                path = R/f'artifacts/rbf-topk-seed{seed}-fresh-resume-launch-v1-20261002/resume-manifest.json'
                assert sha(path) == value['resume_manifest_sha256']
                for filename, digest in value['reused_complete_sequence_receipts'].items():
                    assert sha(directory/filename) == digest
                resume = dict(path=str(path),sha256=sha(path))
            register(acceptance_path, value['kind'])
            records.append(dict(K=width, seed=seed, task_id=value['task_id'], task_status='completed',
                acceptance=str(acceptance_path), acceptance_sha256=sha(acceptance_path),
                binding_sha256=sha(binding_path), byte_admission=str(byte_path), byte_admission_sha256=sha(byte_path),
                checkpoint=job['plan']['checkpoint'], final_refit_checkpoint_not_inherited=final['checkpoint'],
                source_bindings=value['sources'], completed_sequences=46, completed_events=7445,
                states=value['states'], predictions=value['predictions'], max_abs_error=value['max_abs_error'],
                full_sequence_proofs=proofs, resume_binding=resume,
                numerical_validator_rerun=False, final_refit_experiment_accepted=False))
    receipt = dict(kind='rbf_legacy_Top1_K4_completed_fresh_state_independent_index_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), records=records,
        source_sha256=sha(__file__), final_refit_preparation_sha256=sha(FINAL),
        old_nested_checkpoint_only=True, completed_existing_numeric_work_not_rerun=True,
        identity_search_or_partition_bounds_accepted=False, full_Stage2_complete=False,
        same_resource_performance_accepted=False, paper_performance_complete=False, goal_complete=False)
    new(INDEX, receipt); register(INDEX, receipt['kind'])
    print(json.dumps(dict(receipt=str(INDEX), completed_existing_gates=len(records), final_refit_experiment_accepted=False)))


if __name__ == '__main__':
    main()
