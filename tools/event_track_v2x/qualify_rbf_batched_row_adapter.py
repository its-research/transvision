"""Real-history SQL/context integration check, not a full forest acceptance."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import types

from rbf_nested_seen_val_v2_common import R, new, register, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    output = parser.parse_args().output.resolve()
    assert output.is_relative_to(R/'artifacts')
    output.mkdir(exist_ok=False)
    previous = R/'artifacts/rbf-same-arrival-batched-scorer-CPU-prefix-parity-v1-20261004'
    proof = json.loads((previous/'CPU-prefix-parity.json').read_bytes())
    binding = json.loads((previous/'source-binding.json').read_bytes())
    assert sha(previous/'source-binding.json') == proof['source_binding_sha256']
    for name, checksum in binding['source_files'].items():
        assert sha(previous/'source'/name) == checksum
    source = output/'source'
    shutil.copytree(previous/'source', source)
    module_path = 'transvision/models/event_track_v2x/batched_persistent_cache_stream.py'
    candidate = Path(__file__).resolve().parents[2]/module_path
    destination = source/module_path
    with destination.open('xb') as stream:
        stream.write(candidate.read_bytes())
    new(output/'source-binding.json', dict(original_qualified_source_binding_sha256=sha(previous/'source-binding.json'),
        adapter_sha256=sha(destination), qualifier_sha256=sha(__file__), production_integration_enabled=False))
    for name in ('transvision', 'transvision.models', 'transvision.models.event_track_v2x'):
        module = types.ModuleType(name)
        module.__path__ = [str(source.joinpath(*name.split('.')))]
        sys.modules[name] = module
    import numpy as np
    import torch
    from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
    from transvision.models.event_track_v2x.forest_potentials import LearnedForestScorer
    from transvision.models.event_track_v2x.batched_row_context_scoring import BatchedLearnedRowScorer
    from transvision.models.event_track_v2x.batched_persistent_cache_stream import BatchedRowContextScorer, create_batched_cache_stream
    from transvision.models.event_track_v2x.persistent_cache_stream import RowContextScorer
    from transvision.models.event_track_v2x.persistent_forest import PersistentForestTracker
    from transvision.models.event_track_v2x.forest_training_data import TrainingShard
    from transvision.models.event_track_v2x.detection_cache_v2 import canonical
    from transvision.models.event_track_v2x.recoverable_identity import model_digest
    import hashlib
    torch.set_num_threads(1)
    train = R/'artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004/seed2027'
    assert sha(train/'checkpoint') == proof['checkpoint_sha256']
    checkpoint = json.loads((train/'checkpoint').read_bytes())
    weights = train/'archive-unpack/training/seed-2027/weights.pt'
    assert sha(weights) == proof['weights_sha256']
    model = RecoverableIdentityModel(**checkpoint['architecture'])
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    assert model_digest(model) == proof['model_sha256']
    original = LearnedForestScorer(model, max_nodes=9, max_pairs=81, geometry_weight=1., process_noise=.1)
    tracker = PersistentForestTracker(output/'history-only.sqlite', sequence_id='0000')
    serial = RowContextScorer(tracker, original)
    batch = BatchedLearnedRowScorer(original, max_batch=64)
    adapter = BatchedRowContextScorer(tracker, batch)
    assert adapter.binding != serial.binding
    rows_root = R/'artifacts/rbf-joint-identity-full-independent-numpy-v1-20261001/seed2027'
    assert sha(rows_root/'manifest') == checkpoint['dataset_sha256']
    manifest = json.loads((rows_root/'manifest').read_bytes())
    record = next(r for r in manifest['shards'] if r['sequence_id'] == '0000')
    hits = list((rows_root/'rows-unpack').rglob(record['path']))
    assert len(hits) == 1 and sha(hits[0]) == record['sha256']
    shard = TrainingShard(hits[0], record, manifest['row_protocol']['parent_limit'])
    schedule_path = R/'artifacts/rbf-original-cache-CPU-metadata-export-v1-20261001/seed2027/events.json'
    assert sha(schedule_path) == '3440542d0fb6b7c52a6b97a886a888ab5acca73ab1b73a37139e1c49b37c6c25'
    events = [e for e in json.loads(schedule_path.read_bytes())['events'] if e['sequence_id'] == '0000'][:4]
    checked = []
    maximum = 0.
    for event in events:
        indices = np.flatnonzero(shard.arrays['decision_us'] == event['decision_us']).tolist()
        assert indices == list(range(tracker.n, tracker.n+len(indices)))
        observations = tuple(shard._observation(i) for i in indices)
        got, contexts = adapter.rows(observations, event['decision_us'])
        wanted, serial_contexts = serial.rows(observations, event['decision_us'])
        assert contexts == serial_contexts
        for i, context, row, reference in zip(indices, contexts, got, wanted, strict=True):
            length = int(shard.arrays['lengths'][i])
            assert context == tuple(map(int, shard.arrays['contexts'][i, :length]))
            assert [p for p, _ in row] == [p for p, _ in reference]
            g = np.asarray([w for _, w in row]); w = np.asarray([w for _, w in reference])
            assert np.allclose(g, w, atol=1e-4, rtol=1e-4)
            maximum = max(maximum, float(np.max(np.abs(g-w))))
        # A history-only fixture: insert actual preceding raw observations.
        # No forest state, hypothesis, prediction or acceptance is fabricated.
        for i, observation in zip(indices, observations, strict=True):
            raw = canonical(asdict(observation))
            tracker.db.execute('INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?)',
                (i, observation.node.node_id, observation.node.source_id, observation.node.frame_id,
                 observation.detection_index, observation.state_us, observation.node.arrival_us,
                 observation.score, raw, hashlib.sha256(raw).hexdigest()))
        tracker.n += len(observations)
        checked.append(dict(event_id=event['event_id'], rows=len(indices)))
    refusals = []
    def refuse(label, call):
        try:
            call()
        except (ValueError, TypeError):
            refusals.append(label)
        else:
            raise AssertionError(label)
    refuse('nonempty_store_factory', lambda: create_batched_cache_stream(None, tracker, original))
    refuse('future_arrival', lambda: adapter.rows(observations, min(o.node.arrival_us for o in observations)-1))
    batch.max_batch = 32
    refuse('execution_batch_mutation', lambda: adapter.rows((), events[-1]['decision_us']))
    batch.max_batch = 64
    assert adapter.rows((), events[-1]['decision_us']) == ((), ())
    tracker.db.close()
    path = output/'adapter-integration-receipt.json'
    new(path, dict(kind='rbf_real_history_SQL_batched_row_adapter_candidate_v1', seed=2027,
        events=checked, maximum_serial_factor_error=maximum, atol=1e-4, rtol=1e-4,
        exact_original_and_admitted_training_context_indices=True, refusal_controls=refusals,
        source_binding_sha256=sha(output/'source-binding.json'), SQL_fixture_sha256=sha(output/'history-only.sqlite'),
        actual_forest_step_executed=False, actual_cache_stream_step_executed=False,
        GPU_execution_or_memory_target_admitted=False, production_integration_enabled=False))
    register(path, 'rbf-real-history-SQL-batched-row-adapter-candidate')
    print(json.dumps(dict(receipt=str(path), rows=sum(e['rows'] for e in checked), max_error=maximum)))


if __name__ == '__main__':
    main()
