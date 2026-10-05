"""Reconcile previously sampled logical streams with their current binding.

No fresh API queries; preserve all originally sampled evidence. Extra
historical logical streams cannot inflate the number of physical cards.
"""
import argparse
import datetime
import importlib.util
import json
from pathlib import Path

from rbf_nested_seen_val_v2_common import R,new,register,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--observation',type=Path,required=True)
    args=parser.parse_args()
    observation=json.loads(args.observation.read_bytes())
    binding_path=Path(observation['binding_receipt'])
    assert sha(binding_path)==observation['binding_receipt_sha256']
    binding=json.loads(binding_path.read_bytes())
    assert binding['Agent_configuration_and_nonallowlisted_logs_excluded'] is True
    module_path=R/'source-freezes/rbf-allowlisted-Task-GPU-memory-target-observer-v2-binding-cardinality-20261004/observe_rbf_task_GPU_memory_target.py'
    freeze_path=module_path.parent/'source-freeze.json'
    freeze=json.loads(freeze_path.read_bytes())
    assert sha(module_path)==freeze['sources'][module_path.name]['sha256']
    spec=importlib.util.spec_from_file_location('binding_scoped_memory_observer',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    outcomes=[]
    for task in observation['tasks']:
        workers=[w for w in binding['fleet']['workers'] if w.get('task_id')==task['task_id'] and 'L40' not in w['id']]
        count=module.bound_logical_count(workers)
        retained=[row for row in task['logical_GPUs'] if 0<=row['logical_GPU']<count]
        excluded=[row['logical_GPU'] for row in task['logical_GPUs'] if row['logical_GPU']>=count]
        assert [row['logical_GPU'] for row in retained]==list(range(count))
        outcomes.append(dict(task_id=task['task_id'],current_snapshot_Worker=workers[0]['id'],
            original_RBF_source_identity_verified=task['original_RBF_source_identity_verified'],
            declared_binding_cardinality=count,retained_logical_observations=retained,
            excluded_out_of_binding_reported_logical_ids=excluded,
            device_UUID_mapping_still_requires_independent_readback=True))
    count=sum(row['declared_binding_cardinality'] for row in outcomes)
    assert count==binding['fleet']['physical_worker_task_bound_GPUs']==40
    cards=[card for row in outcomes for card in row['retained_logical_observations']]
    statuses=[card['reported_device_memory_percentage']['status'] for card in cards]
    now=datetime.datetime.now(datetime.timezone.utc)
    receipt=R/'receipts'/('rbf-GPU-memory-target-bound-cardinality-scoped-audit-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    new(receipt,dict(kind='rbf_GPU_memory_target_immutable_observation_binding_scope_audit_v1',
        checked_at_utc=now.isoformat(),sampled_at_utc=observation['checked_at_utc'],
        command=['python',str(Path(__file__).resolve()),'--observation',str(args.observation)],source_sha256=sha(__file__),
        observation=str(args.observation),observation_sha256=sha(args.observation),
        binding_receipt=str(binding_path),binding_receipt_sha256=sha(binding_path),
        v2_observer_source_freeze_sha256=sha(freeze_path),GPU_memory_target_percent=[75,80],
        original_logical_series_count=sum(len(row['logical_GPUs']) for row in observation['tasks']),
        declared_current_binding_cardinality=count,outcomes=outcomes,
        below_target_cards=statuses.count('below_75_percent'),in_target_cards=statuses.count('within_75_to_80_percent'),
        above_target_cards=statuses.count('above_80_percent'),unknown_cards=sum(status.startswith('unknown') for status in statuses),
        target_reached=False,independent_physical_UUID_NVML_admission=False,
        original_evidence_overwritten=False,actual_API_queries=0,actual_GPU_tasks_launched_or_restarted=0,
        paper_or_Stage2_cost_admission=False,whole_experiment_ETA='unknown',goal_status='active'))
    register(receipt,'rbf-GPU-memory-target-binding-scoped-audit')
    print(json.dumps(dict(receipt=str(receipt),sha256=sha(receipt),cards=count,
        below_target=statuses.count('below_75_percent'),in_target=statuses.count('within_75_to_80_percent'),
        unknown=sum(status.startswith('unknown') for status in statuses),actual_API_queries=0)))


if __name__=='__main__':main()
