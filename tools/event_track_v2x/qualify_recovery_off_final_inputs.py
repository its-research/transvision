"""Load real final checkpoints with the frozen recovery-off source candidate.

This qualifies source/model/input identities, not forest outputs or GPU numeric
parity. Existing full cache and NN audits are reused by exact identity; no data,
training, forward-output or forest experiment is regenerated here.
"""
import datetime
import json
from pathlib import Path
import sys

from rbf_nested_seen_val_v2_common import R, new, register, sha

SOURCE = R/'source-freezes/rbf-recovery-off-original-source-bound-candidate-v3-20261005'
SOURCE_SHA = 'b778aee162f177066da207b70ecc807578be687373583a6b9748fa63b3aa9b70'
DISPATCH = R/'receipts/rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json'
INDEX = R/'receipts/rbf-final-refit-three-seed-all-row-full-independent-numeric-acceptance-20261004.json'
OUTPUT = R/'artifacts/rbf-recovery-off-final-checkpoint-input-contract-v1-20261005'


def main():
    assert not OUTPUT.exists(), 'preserve the original qualification attempt'
    assert sha(SOURCE/'source-freeze.json') == SOURCE_SHA
    frozen = json.loads((SOURCE/'source-freeze.json').read_bytes())
    for name, spec in frozen['sources'].items():
        assert sha(SOURCE/name) == spec['sha256'] and (SOURCE/name).stat().st_size == spec['bytes']
    assert sha(DISPATCH) == frozen['final_dispatch']['sha256']
    assert sha(INDEX) == '93e7e3849dfdce2814c9a22510575dade16efd7a2fd5c6569f5074f0e3550816'
    sys.path.insert(0, str(SOURCE))
    from transvision.models.event_track_v2x import forest_training_checkpoint as loader
    from transvision.models.event_track_v2x import recovery_off_paper_runtime as runtime
    from transvision.models.event_track_v2x.forest_tracking import PaperForestTrackingConfig
    from transvision.models.event_track_v2x.recoverable_identity import model_digest
    assert Path(loader.__file__).resolve().is_relative_to(SOURCE)
    assert Path(runtime.__file__).resolve().is_relative_to(SOURCE)
    config = runtime.default_configuration()
    assert config == json.loads((SOURCE/'bound-configuration.json').read_bytes())
    index = json.loads(INDEX.read_bytes())
    assert index['full_independent_joint_attention_cross_source_temporal_motion_numerical_pass'] is True
    assert index['rows'] == 677744 and len(index['seeds']) == 3
    jobs = json.loads(DISPATCH.read_bytes())['jobs']
    OUTPUT.mkdir()
    results = []
    binding = dict(kind='rbf_recovery_off_real_final_checkpoint_input_contract_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        candidate_source_freeze_sha256=SOURCE_SHA, final_NN_index_sha256=sha(INDEX),
        final_dispatch_sha256=sha(DISPATCH), qualifier_sha256=sha(__file__), configuration=config,
        device='cpu', existing_full_cache_and_NN_results_reused_not_regenerated=True,
        full_cache_constructor_rerun=False, forest_replayed=False, GPU_forward_tested=False,
        full_forest_semantics_accepted=False, paper_performance_complete=False)
    new(OUTPUT/'binding.json', binding)
    try:
        for seed in (1337, 2027, 3407):
            job = next(j for j in jobs if j['seed'] == seed)
            plan = job['plan']
            entry = next(v for v in index['seeds'] if v['seed'] == seed)
            for key in ('training_byte_proof', 'prediction_byte_proof', 'numeric_completion'):
                assert sha(entry[key]) == entry[key+'_sha256']
            assert entry['numeric_failures'] == 0
            base = Path(entry['training_byte_proof']).parent
            train = json.loads(Path(entry['training_byte_proof']).read_bytes())
            assert train['artifacts']['checkpoint'] == plan['checkpoint']
            assert train['artifacts']['identity-training'] == plan['weights_archive']
            checkpoint_root = base/f'archive-unpack/training/seed-{seed}'
            assert sha(checkpoint_root/'checkpoint.json') == sha(base/'checkpoint') == plan['checkpoint']['sha256']
            scorer, checkpoint = loader.load_identity_checkpoint(checkpoint_root,
                plan['checkpoint']['sha256'], config=PaperForestTrackingConfig(**config['state']), device='cpu')
            assert checkpoint['seed'] == seed and checkpoint['fixture'] is False
            assert checkpoint['model_sha256'] == model_digest(scorer.model) == plan['final_refit_model_sha256']
            assert checkpoint['row_protocol']['candidate_protocol'] == 'rbf-all-class-top64-v1'
            assert checkpoint['row_protocol']['class_scope'] == ['car', 'bicycle', 'pedestrian']
            assert checkpoint['row_protocol']['minimum_raw_score'] == .05
            assert checkpoint['row_protocol']['maximum_detections'] == 64
            assert not scorer.model.training and all(not p.requires_grad for p in scorer.model.parameters())
            assert config['state'] == plan['configuration']['state']
            before = dict(config['limits']); assert before.pop('recovery_off_version') == 1
            assert before == plan['configuration']['limits']
            cache_proof = R/f'artifacts/rbf-original-nested-cache-replay-input-readback-v1-20261001/seed{seed}/byte-readback-receipt.json'
            cache = json.loads(cache_proof.read_bytes())
            assert cache['all_archive_and_manifest_member_bytes_verified'] is True
            assert cache['task_id'] == plan['cache_archive']['task'] == plan['cache_manifest']['task']
            assert cache['manifest_sha256'] == plan['cache_manifest']['sha256']
            assert cache['artifacts']['cache-v2']['sha256'] == plan['cache_archive']['sha256']
            root = Path(cache['cache_root'])
            assert sha(root/'manifest.json') == cache['manifest_sha256']
            manifest = json.loads((root/'manifest.json').read_bytes())
            assert manifest['split'] == 'train' and manifest['gt_in_cache'] is False
            events_path = R/f'artifacts/rbf-original-cache-CPU-metadata-export-v1-20261001/seed{seed}/events.json'
            assert sha(events_path) == plan['events']['sha256']
            events = json.loads(events_path.read_bytes())
            assert events['cache_manifest_sha256'] == cache['manifest_sha256']
            assert len(events['events']) == 7445 and len(events['origin_us_by_sequence']) == 46
            assert sorted(events['origin_us_by_sequence']) == checkpoint['partition']['fit']
            # Validation of every delivery/payload remains the unchanged actual
            # replay constructor's responsibility; this receipt cannot replace it.
            result = dict(seed=seed, checkpoint_task_id=train['task_id'],
                checkpoint_sha256=plan['checkpoint']['sha256'], model_sha256=checkpoint['model_sha256'],
                finite_tensor_state_loaded=True, strict_state_dict_loaded=True,
                actual_frozen_scoring_source_hashes_verified=True,
                frozen_cache_identity=checkpoint['frozen_cache_identity'],
                exact_cache_manifest_sha256=cache['manifest_sha256'],cache_byte_proof_sha256=sha(cache_proof),
                exact_events_sha256=sha(events_path),events=7445,sequences=46,
                scoring_sources={n:sha(SOURCE/'transvision/models/event_track_v2x'/n) for n in loader.SCORING_SOURCES},
                input_cloud_assets={k:plan[k] for k in ('events','cache_archive','cache_manifest','checkpoint','weights_archive')},
                configuration_diff=['backend', 'limits.recovery_off_version'],
                actual_GPU_forward_or_full_replay_accepted=False)
            new(OUTPUT/f'seed{seed}.json', result); results.append(result)
            print(json.dumps(dict(seed=seed, real_checkpoint_loaded=True, events_bound=7445,
                                  sequences_bound=46, ETA='unknown for future replay')), flush=True)
        assert len({x['model_sha256'] for x in results}) == 3
        final = dict(binding, seeds=results, all_three_actual_final_checkpoints_loaded_and_bound=True)
        new(OUTPUT/'acceptance.json', final); register(OUTPUT/'acceptance.json', final['kind'])
        print(json.dumps(dict(acceptance=str(OUTPUT/'acceptance.json'),sha256=sha(OUTPUT/'acceptance.json'))),flush=True)
    except BaseException as error:
        failure = dict(binding,error_type=type(error).__name__,completed_seeds=len(results),input_contract_accepted=False)
        new(OUTPUT/'failure.json',failure);register(OUTPUT/'failure.json','rbf_recovery_off_final_input_contract_failure_v1')
        raise


if __name__ == '__main__': main()
