"""Hash-bound, weights-only checkpoints for the row-context identity network."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .detection_cache_v2 import contained_file, sha_file
from .forest_potentials import LearnedForestScorer
from .forest_training_data import row_protocol
from .learned_identity import RecoverableIdentityModel
from .recoverable_identity import model_digest


CHECKPOINT_KIND = 'persistent_forest_row_identity_checkpoint_v1'
SCORING_SOURCES = ('learned_identity.py', 'forest_potentials.py', 'forest_row_context.py',
                   'forest_training.py', 'forest_tracking.py', 'prediction_features.py', 'tracking_v2.py')


def load_identity_checkpoint(root, manifest_sha256, *, config, device='cpu'):
    root = Path(root)
    path = contained_file(root, 'checkpoint.json')
    if sha_file(path) != manifest_sha256:
        raise ValueError('checkpoint manifest identity differs')
    manifest = json.loads(path.read_bytes())
    if (manifest['kind'] != CHECKPOINT_KIND or manifest['row_protocol'] != row_protocol(config)
            or manifest['data_split'] != 'train' or manifest['labels_in_model_inputs'] is not False
            or manifest['local_partial_label_objective'] is not True):
        raise ValueError('checkpoint training, features or row-context protocol differs')
    module = Path(__file__).resolve().parent
    if any(manifest['source_sha256'].get('transvision/models/event_track_v2x/'+name) != sha_file(module/name)
           for name in SCORING_SOURCES):
        raise ValueError('checkpoint scoring implementation differs from current sources')
    weights = contained_file(root, manifest['weights']['path'])
    if sha_file(weights) != manifest['weights']['sha256']:
        raise ValueError('checkpoint weight identity differs')
    model = RecoverableIdentityModel(**manifest['architecture'])
    state = torch.load(weights, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('invalid finite model state')
    model.load_state_dict(state, strict=True)
    if sha_file(weights) != manifest['weights']['sha256'] or model_digest(model) != manifest['model_sha256']:
        raise ValueError('checkpoint model binding differs')
    model.to(device=device).eval().requires_grad_(False)
    scorer = LearnedForestScorer(model, max_nodes=config.parent_limit+1, max_pairs=(config.parent_limit+1)**2,
                                geometry_weight=manifest['geometry_weight'], process_noise=config.process_noise)
    return scorer, manifest
