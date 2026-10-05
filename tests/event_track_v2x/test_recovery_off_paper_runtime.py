"""Synthetic cache-to-neural-factor replay, never real dataset acceptance."""
import ast
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from test_detection_cache_v2 import _build, _sources
from test_paper_pipeline import native_frame
from transvision.models.event_track_v2x import exclusive_paper_runtime as original
from transvision.models.event_track_v2x import recovery_off_paper_runtime as candidate
from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery, VerifiedForestCache
from transvision.models.event_track_v2x.forest_potentials import LearnedForestScorer
from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
from transvision.models.event_track_v2x.paper_native_cache import NativePaperCache, write_native_cache
from transvision.models.event_track_v2x.paper_protocol import PaperProtocol
from transvision.models.event_track_v2x.paper_runtime_selection import runtime, allocation_configuration
from transvision.models.event_track_v2x.recovery_off_tracking import RecoveryOffTracker


def test_replay_body_is_identical_and_configuration_has_only_declared_changes():
    def node(m):
        return next(n for n in ast.parse(Path(m.__file__).read_bytes()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'replay')
    assert ast.dump(node(original)) == ast.dump(node(candidate))
    before, after = original.default_configuration(), candidate.default_configuration()
    assert after.pop('backend') == candidate.BACKEND
    before.pop('backend')
    assert after['limits'].pop('recovery_off_version') == 1
    assert before == after


def test_explicit_backend_keeps_recovery_off_configuration_for_each_policy():
    c = candidate.default_configuration()
    assert runtime(c) is candidate
    for allocation in ('bound', 'learned', 'teacher'):
        config = candidate.default_configuration(allocation=allocation)
        assert runtime(config) is candidate
        bound = asdict(allocation_configuration(config))
        assert bound.pop('state') == config['state']
        assert bound == config['limits']
    with pytest.raises(ValueError, match='allocation'):
        runtime(dict(c, allocation='fixed'))
    with pytest.raises(ValueError, match='allocation'):
        candidate.default_configuration(allocation='fixed')


@pytest.mark.parametrize('dataset', ['spd', 'v2v4real'])
def test_real_cache_interfaces_and_neural_forward_keep_original_inputs(tmp_path, dataset):
    if dataset == 'spd':
        sources = _sources(tmp_path, split='train', empty_vehicle=False)
        cache = VerifiedForestCache(sources[-1], _build(sources))
    else:
        frames = [native_frame(sequence_id='0003', side=side, agent_mask=i+1, dataset_split='train')
                  for i, side in enumerate(('vehicle-side', 'infrastructure-side'))]
        root = tmp_path/'cache'
        sha = write_native_cache(root, frames, split='train', producer={'fit_split':'train'}, fixture=True)
        cache = NativePaperCache(root, sha)
    protocol = PaperProtocol(dataset, 'train')
    events = []
    for seq in sorted({key[0] for key in cache.index}):
        deliveries = []
        for (scene, side, fid), (entry, metadata) in cache.index.items():
            if scene != seq: continue
            entry, metadata = json.loads(entry), json.loads(metadata)
            arrival = max(metadata['box_reference_timestamp_us'], metadata['source_image_timestamp_us'])+100_000
            deliveries.append(asdict(CacheDelivery(seq, side, fid, arrival, entry['frame_sha256'])))
        decision = max(d['arrival_us'] for d in deliveries)
        events.append(dict(sequence_id=seq, frame_id='first', reference_us=decision,
                           decision_us=decision, event_id='first', deliveries=deliveries))
        events.append(dict(sequence_id=seq, frame_id='empty', reference_us=decision+100_000,
                           decision_us=decision+100_000, event_id='empty', deliveries=[]))
    torch.manual_seed(1337)
    model = RecoverableIdentityModel(hidden=8, heads=2, dropout=0.).eval().requires_grad_(False)
    scorer = LearnedForestScorer(model)
    binding = dict(candidate_protocol=protocol.candidates, dataset=dataset, fit_split='train')
    audits = []
    for label, module in [('on', original), ('off', candidate)]:
        out = tmp_path/label
        result = module.replay(cache, events, out, protocol=protocol,
            configuration=module.default_configuration(), scorer=scorer, model_binding=binding, fixture=True)
        assert result['completed_events'] == len(events) and not result['paper_results_verified']
        rows = [json.loads(s) for s in (out/'audit.jsonl').read_text().splitlines()]
        audits.append(rows)
        if label == 'off':
            assert all(not row['recovery_enabled'] for row in rows)
            assert all(row['new_observations']==0 for row in rows[1::2])
            preds = [json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()]
            for seq, db in result['databases'].items():
                last = next(p for p in reversed(preds) if p['sequence_id']==seq)
                t = RecoveryOffTracker.open(out/db['path'], expected_database_sha256=db['sha256'],
                                            expected_prediction_sha256=last['commit_sha256'])
                assert t.close() == db['sha256']
    assert [x['factor_rows_sha256'] for x in audits[0]] == [x['factor_rows_sha256'] for x in audits[1]]
    assert [x['observation_count'] for x in audits[0]] == [x['observation_count'] for x in audits[1]]
