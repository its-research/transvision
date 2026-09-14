"""Selection uses only train holdout and restores the selected weights, not
final weights."""
import hashlib
import json

import numpy as np

from transvision.models.event_track_v2x.allocation_policy import FEATURES, RECIPE, TARGET
from transvision.models.event_track_v2x.allocation_training import DATA_KIND, allocation_sources, fit_priority
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file


def test_best_train_holdout_restores_earlier_epoch_for_all_three_seeds(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    scenes = ['train-a', 'train-b']
    held = min(scenes, key=lambda s: hashlib.sha256(('priority-holdout-v1:' + s).encode()).hexdigest())
    records = []
    for i, scene in enumerate(scenes):
        path = data / f'{i}.jsonl'
        # Opposite targets make continued fitting worse on the train holdout.
        path.write_bytes(canonical(dict(features=[[0.] * len(FEATURES)], targets=[-.9 if scene == held else .9])) + b'\n')
        records.append(dict(sequence_id=scene, path=path.name, sha256=sha_file(path), groups=1, rows=1))
    manifest = dict(
        kind=DATA_KIND,
        split='train',
        feature_recipe=RECIPE,
        feature_names=FEATURES,
        target_recipe=TARGET,
        source_sha256=allocation_sources(),
        full_official_train_trace=False,
        shards=records,
        binding={'fixture': True})
    (data / 'manifest.json').write_bytes(canonical(manifest))
    selected = fit_priority(
        data, sha_file(data / 'manifest.json'), tmp_path / 'best', epochs=4, hidden=4, learning_rate=.05, require_full_train=False, select_best_train_holdout=True)
    first = fit_priority(data, sha_file(data / 'manifest.json'), tmp_path / 'first', epochs=1, hidden=4, learning_rate=.05, require_full_train=False)
    final = fit_priority(data, sha_file(data / 'manifest.json'), tmp_path / 'final', epochs=4, hidden=4, learning_rate=.05, require_full_train=False)
    for chosen, one, last in zip(selected['seeds'], first['seeds'], final['seeds']):
        directory = tmp_path / 'best' / str(chosen['seed'])
        checkpoint = json.loads((directory / 'checkpoint.json').read_bytes())
        epochs = [json.loads(line) for line in (directory / 'epochs.jsonl').read_bytes().splitlines()]
        losses = [e['train_sequence_holdout_mse'] for e in epochs]
        assert checkpoint['selected_epoch'] == int(np.argmin(losses)) + 1 == 1
        assert checkpoint['selected_train_holdout_mse'] == min(losses)
        assert not checkpoint['official_validation_or_test_used_for_selection']
        assert chosen['policy_signature'] == one['policy_signature'] != last['policy_signature']
        assert checkpoint['holdout_sequences'] == [held]
